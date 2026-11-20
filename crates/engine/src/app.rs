//! Runtime: startup, the requestAnimationFrame loop with a fixed simulation step, and [`Ctx`].

use crate::audio::{Audio, SoundId};
use crate::gltf_loader::ModelCache;
use crate::input::{Input, MouseButton};
use crate::panel::{Panel, PanelId, Panels, WorldPanel};
use crate::physics::Physics;
use crate::renderer::{Material, MaterialId, MeshId, Renderer};
use crate::scene::{self, MeshRenderer, Name, Transform};
use crate::ui::Ui;
use crate::{debug, log, log_error, log_info};
use glam::{Vec2, Vec3};
use hecs::{Entity, World};
use serde::Serialize;
use serde_json::Value;
use std::cell::RefCell;
use std::rc::{Rc, Weak};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

/// Implemented by the game. All methods get full mutable access to the engine via [`Ctx`].
pub trait Game: 'static {
    /// Called once after the GPU is ready.
    fn init(&mut self, ctx: &mut Ctx);
    /// Fixed-rate simulation step (`ctx.time.fixed_dt`), runs *before* physics. Deterministic.
    /// Held input: `input.key`; edges: `input.key_pressed` (here: pressed since the last step, so
    /// short taps are never lost); for jump/fire-style actions `input.key_down_or_pressed`.
    fn fixed_update(&mut self, _ctx: &mut Ctx) {}
    /// Once per rendered frame, after the fixed steps (camera, UI, edge-triggered input).
    fn update(&mut self, _ctx: &mut Ctx, _dt: f32) {}
    /// Game-specific AI/debug commands: `engine.call("<name>", args)` for names the engine
    /// does not know. Return `None` for unknown names.
    fn debug_command(
        &mut self,
        _ctx: &mut Ctx,
        _name: &str,
        _args: &Value,
    ) -> Option<Result<Value, String>> {
        None
    }
    /// (name, description) of the commands handled in `debug_command` — shown by `help`.
    fn debug_commands(&self) -> Vec<(&'static str, &'static str)> {
        vec![]
    }
    /// (name, description, owner) of commands that exist but are not callable right now (a
    /// `ScreenStack` lists the commands of screens that are not on top) — shown by `help`.
    fn inactive_debug_commands(&self) -> Vec<(&'static str, &'static str, &'static str)> {
        vec![]
    }
    /// Game state snapshot for AI inspection (`engine.call("game_state")`).
    fn debug_state(&self, _ctx: &Ctx) -> Value {
        Value::Null
    }
}

/// CPU time per engine phase in ms (exponentially smoothed). `steps` = fixed steps last frame.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Phases {
    /// `Game::fixed_update` (all fixed steps of the frame).
    pub game_fixed: f32,
    /// rapier `step` (all fixed steps of the frame).
    pub physics: f32,
    /// `Game::update` + input, panels, touch.
    pub update: f32,
    /// transforms, panel redraw, audio, debug tools.
    pub engine: f32,
    /// `Renderer::render` CPU side (culling, sorting, uploads, command encoding).
    pub render: f32,
    pub steps: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Time {
    pub fixed_dt: f32,
    /// Number of fixed steps simulated so far.
    pub step: u64,
    /// Simulated seconds (step * fixed_dt).
    pub sim_time: f64,
    pub frame: u64,
    pub frame_dt: f32,
    pub fps: f32,
    pub frame_ms: f32,
    pub time_scale: f32,
    pub paused: bool,
    /// CPU milliseconds per engine phase (smoothed): where a frame's CPU time goes.
    pub phases: Phases,
    #[serde(skip)]
    pub(crate) accumulator: f32,
    #[serde(skip)]
    acc_game_fixed: f32,
    #[serde(skip)]
    acc_physics: f32,
}

/// Deterministic xorshift64* RNG (seed it via `engine.call("seed", {value})` for reproducible runs).
#[derive(Clone, Debug)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1).
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }
}

pub struct Ctx {
    pub world: World,
    pub renderer: Renderer,
    pub physics: Physics,
    pub input: Input,
    pub ui: Ui,
    /// Clickable 2D panels on 3D quads (see [`crate::panel`]).
    pub panels: Panels,
    /// Web Audio (see [`crate::audio`]).
    pub audio: Audio,
    /// Automatic z-fighting check (debug builds), see [`crate::zfight::AutoCheck`].
    pub zcheck: crate::zfight::AutoCheck,
    /// Draw dynamic bodies and characters between their last two fixed-step poses (smooth motion
    /// at any display rate, see `crate::interp`). Default on.
    pub interpolation: bool,
    pub(crate) interp_shown: bool,
    /// Motion trace (`trace` command): entity, frames left, samples [frame ms, steps, x, y, z].
    pub(crate) trace: Option<(Entity, u32, Vec<[f32; 5]>)>,
    /// Debug mode (` key): input + camera log file, badge (see [`crate::debugmode`]).
    pub debug_mode: crate::debugmode::DebugMode,
    /// Save slots (localStorage) and blobs (IndexedDB), see [`crate::save`].
    pub save: crate::save::Save,
    /// Scene membership of entities, see [`crate::screens`] (`begin_scene`, `unload_scene`).
    pub scenes: crate::screens::Scenes,
    /// Options for `load_gltf` (automatic LOD on/off, LOD distance factor).
    pub gltf_defaults: crate::gltf_loader::GltfOpts,
    /// Procedural generators (see [`crate::geom`]).
    pub generators: crate::geom::Generators,
    /// Streaming layers + floating origin settings (see [`crate::streaming`]).
    pub streaming: crate::streaming::Streaming,
    /// Heightmap terrain (see [`crate::terrain`]).
    pub terrain: Option<crate::terrain::Terrain>,
    /// Absolute world position of the local origin (floating origin, see `Ctx::shift_origin`).
    pub origin: glam::DVec3,
    pub time: Time,
    pub rng: Rng,
    pub canvas: HtmlCanvasElement,
    pub(crate) models: ModelCache,
    pub(crate) me: Weak<RefCell<Ctx>>,
    pub(crate) last_render_error: Option<String>,
}

impl Ctx {
    pub fn spawn_mesh(
        &mut self,
        name: &str,
        mesh: MeshId,
        material: MaterialId,
        t: Transform,
    ) -> Entity {
        self.world
            .spawn((t, Name(name.to_string()), MeshRenderer::new(mesh, material)))
    }

    /// Cube of size `size` (scale), new material with sRGB `color`.
    pub fn spawn_cube(&mut self, name: &str, pos: Vec3, size: Vec3, color: u32) -> Entity {
        let m = self.renderer.add_material(Material::color(color));
        self.spawn_mesh(
            name,
            self.renderer.builtin.cube,
            m,
            Transform::from_translation(pos).with_scale(size),
        )
    }

    pub fn spawn_sphere(&mut self, name: &str, pos: Vec3, radius: f32, color: u32) -> Entity {
        let m = self.renderer.add_material(Material::color(color));
        let e = self.spawn_mesh(
            name,
            self.renderer.builtin.sphere,
            m,
            Transform::from_translation(pos).with_scale(Vec3::splat(radius * 2.0)),
        );
        let _ = self
            .world
            .insert_one(e, self.renderer.builtin.sphere_lod.clone());
        e
    }

    /// Spawns an empty root entity now and attaches the glTF model's nodes once loaded.
    pub fn load_gltf(&mut self, url: &str, t: Transform) -> Entity {
        let opts = self.gltf_defaults;
        self.load_gltf_with(url, t, opts)
    }

    /// `load_gltf` with options (e.g. `GltfOpts { auto_lod: false, ..Default::default() }` for
    /// figures that must always be drawn at full detail).
    pub fn load_gltf_with(&mut self, url: &str, t: Transform, opts: crate::gltf_loader::GltfOpts) -> Entity {
        let name = url.rsplit('/').next().unwrap_or(url).to_string();
        let root = self.world.spawn((t, Name(name), opts));
        crate::gltf_loader::request(self, url, root);
        root
    }

    /// Loads an audio file (any format Chrome decodes). Playable once `audio` reports it ready.
    pub fn load_sound(&mut self, url: &str) -> SoundId {
        crate::audio::request(self, url)
    }

    /// Generates a procedural texture (see [`crate::texgen`]) and uploads it with mips.
    /// Falls back to the white texture (id 0 of the white is internal; returns None) on error.
    /// Colour + normal + height maps of one preset (normal/height are linear textures). Use with
    /// `Material::normal_map`, `parallax_map` or `displace`.
    pub fn generate_maps(&mut self, t: &crate::texgen::TexGen, normal_strength: f32) -> crate::texgen::TexMaps {
        let color = self.generate_texture(t);
        let label = t.label();
        let up = |r: &mut crate::renderer::Renderer, px: Vec<u8>, kind: &str| match r.add_texture_rgba_linear(t.size, t.size, &px, &format!("{label}:{kind}")) {
            Ok(id) => Some(id),
            Err(e) => {
                log_error!("texgen {label} {kind}: {e}");
                None
            }
        };
        let normal = up(&mut self.renderer, t.generate_normal(normal_strength), "normal");
        let height = up(&mut self.renderer, t.generate_height(), "height");
        crate::texgen::TexMaps { color, normal, height }
    }

    pub fn generate_texture(
        &mut self,
        t: &crate::texgen::TexGen,
    ) -> Option<crate::renderer::TextureId> {
        let t0 = log::now_ms();
        let px = t.generate();
        match self
            .renderer
            .add_texture_rgba(t.size, t.size, &px, &t.label())
        {
            Ok(id) => {
                log_info!(
                    "texture {} generated in {:.1} ms (mean {})",
                    t.label(),
                    log::now_ms() - t0,
                    crate::texgen::mean_color(&px)
                );
                Some(id)
            }
            Err(e) => {
                log_error!("texgen {}: {e}", t.label());
                None
            }
        }
    }

    /// Despawns `e` and all its descendants, including their physics bodies.
    pub fn despawn(&mut self, e: Entity) {
        for d in scene::descendants(&self.world, e) {
            self.physics.remove(&self.world, d);
            let _ = self.world.despawn(d);
        }
    }

    /// Makes `e` (an entity with `Camera`) the camera the canvas renders from: moves the
    /// [`scene::MainCamera`] marker to it. The other cameras keep existing (and keep their state).
    pub fn set_main_camera(&mut self, e: Entity) {
        let others: Vec<Entity> = self.world.query::<(Entity, &scene::MainCamera)>().iter().map(|(o, _)| o).filter(|&o| o != e).collect();
        for o in others {
            let _ = self.world.remove_one::<scene::MainCamera>(o);
        }
        let _ = self.world.insert_one(e, scene::MainCamera);
    }

    /// Puts `e` and all its descendants on render layer `layer` (also glTF nodes loaded later).
    pub fn set_layer(&mut self, e: Entity, layer: u8) {
        let _ = self.world.insert_one(e, scene::RenderLayer(layer));
        for d in scene::descendants(&self.world, e) {
            if let Ok(mut mr) = self.world.get::<&mut MeshRenderer>(d) {
                mr.layer = layer;
            }
        }
    }

    pub fn find_by_name(&self, name: &str) -> Option<Entity> {
        self.world
            .query::<(Entity, &Name)>()
            .iter()
            .find(|(_, n)| n.0 == name)
            .map(|(e, _)| e)
    }

    /// World-space ray through a canvas point (CSS pixels). Returns (origin, direction).
    pub fn screen_ray(&self, css: Vec2) -> Option<(Vec3, Vec3)> {
        let w = self.canvas.client_width().max(1) as f32;
        let h = self.canvas.client_height().max(1) as f32;
        let (view, proj, _) = self.renderer.camera_matrices(&self.world)?;
        let inv = (proj * view).inverse();
        let ndc = Vec2::new(css.x / w * 2.0 - 1.0, 1.0 - css.y / h * 2.0);
        let near = inv.project_point3(ndc.extend(0.0));
        let far = inv.project_point3(ndc.extend(1.0));
        Some((near, (far - near).normalize()))
    }

    /// Spawns a world panel: a quad (scale = world size; keep the panel's aspect ratio) showing
    /// `panel`, clickable with the pointer. Button clicks arrive in `ctx.ui.take_actions()`.
    pub fn spawn_panel(&mut self, name: &str, panel: Panel, t: Transform) -> Entity {
        let e = self.world.spawn((t, Name(name.to_string())));
        match self.panels.create(&mut self.renderer, e, panel) {
            Ok((id, mat)) => {
                let mut mr = MeshRenderer::new(self.renderer.builtin.quad, mat);
                mr.cast_shadows = false;
                let _ = self.world.insert(e, (mr, WorldPanel(id)));
            }
            Err(err) => log_error!("spawn_panel {name}: {err}"),
        }
        e
    }

    /// Panel shown by entity `e` (mutable; redrawn automatically when changed).
    pub fn panel_mut(&mut self, e: Entity) -> Option<&mut Panel> {
        let id: PanelId = self.world.get::<&WorldPanel>(e).ok()?.0;
        self.panels.get_mut(id)
    }

    /// Keyboard edge from the DOM or the AI API ("same code path"). The debug-mode toggle key is
    /// consumed here and never reaches the game.
    pub fn key_event(&mut self, code: &str, key: &str, shift: bool, down: bool, source: &str) {
        if source == "api" {
            self.input.note_api_key(code, down);
        }
        let (now, step, frame) = (log::now_ms(), self.time.step, self.time.frame);
        if crate::debugmode::is_toggle(code, key, shift) {
            if down {
                self.debug_mode.toggle(now, step, frame);
            }
            return;
        }
        let changed = self.input.key(code) != down;
        if down {
            self.input.key_down(code);
        } else {
            self.input.key_up(code);
        }
        if changed {
            self.debug_mode
                .record_input(now, step, frame, code, key, down, source);
        }
    }

/// Touch phase for [`Ctx::touch_event`].
    /// Finger down/move/up from the DOM (`pointerType == "touch"`) or the AI API.
    pub fn touch_event(&mut self, id: i32, pos: Vec2, phase: &str, source: &str) {
        let now = log::now_ms();
        let w = self.canvas.client_width() as f32;
        match phase {
            "start" => self.input.touch.start(id, pos, now, w),
            "move" => self.input.touch.move_to(id, pos),
            _ => self.input.touch.end(id, pos, now),
        }
        if phase != "move" {
            let (step, frame) = (self.time.step, self.time.frame);
            let code = format!("Touch{id}");
            self.debug_mode.record_input(now, step, frame, &code, phase, phase == "start", source);
        }
    }

    /// Mouse-button edge from the DOM or the AI API.
    pub fn button_event(&mut self, b: MouseButton, down: bool, source: &str) {
        if source == "api" {
            self.input.note_api_button(b, down);
        }
        let changed = self.input.button(b) != down;
        if down {
            self.input.button_down(b);
        } else {
            self.input.button_up(b);
        }
        if changed {
            let code = format!("Mouse{}", b as u8);
            let (now, step, frame) = (log::now_ms(), self.time.step, self.time.frame);
            self.debug_mode
                .record_input(now, step, frame, &code, "", down, source);
        }
    }

    /// Creates a render texture (see [`crate::render_texture`]). Returns (id, texture for materials).
    pub fn create_render_texture(&mut self, w: u32, h: u32, opts: crate::render_texture::RenderTextureOpts) -> (crate::render_texture::RenderTextureId, Option<crate::renderer::TextureId>) {
        let label = format!("rtt{}", self.renderer.rtts.len());
        match self.renderer.create_render_texture(w, h, opts, &label) {
            Ok((id, tex)) => (id, Some(tex)),
            Err(e) => {
                log_error!("create_render_texture {w}x{h}: {e}");
                (crate::render_texture::RenderTextureId(u32::MAX), None)
            }
        }
    }

    /// Makes `camera` (an entity with `Camera` + `Transform`) render into `rt`.
    pub fn set_render_texture_camera(&mut self, rt: crate::render_texture::RenderTextureId, camera: Entity) {
        if let Some(r) = self.renderer.rtts.get_mut(rt.0 as usize) {
            r.camera = Some(camera);
            r.pending = true;
            let _ = self.world.insert_one(camera, crate::render_texture::RenderTextureCamera(rt));
        }
    }

    /// Renders a `once` render texture again this frame (portrait changed).
    pub fn refresh_render_texture(&mut self, rt: crate::render_texture::RenderTextureId) {
        if let Some(r) = self.renderer.rtts.get_mut(rt.0 as usize) {
            r.pending = true;
        }
    }

    /// Starts a PNG export; the blob URL appears in `renderer.render_texture_url(rt)` shortly after.
    pub fn export_render_texture(&mut self, rt: crate::render_texture::RenderTextureId) {
        let Some(r) = self.renderer.rtts.get(rt.0 as usize) else { return };
        if let Err(e) = crate::render_texture::start_export(&self.renderer.device, &self.renderer.queue, r) {
            log_error!("export_render_texture: {e}");
        }
    }

    /// Identifying properties of the object drawn under a canvas point (CSS px): entity, name,
    /// scene, positions, mesh, material, parent chain, glTF root. `copy` puts the text on the
    /// clipboard (needs a recent user gesture — the Ctrl+click is one) and shows it in the
    /// debug badge; the pick is also written to the debug log. AI API: `inspect {x, y}`.
    pub fn inspect_at(&mut self, css: Vec2, copy: bool) -> serde_json::Value {
        use serde_json::json;
        scene::propagate_transforms(&mut self.world);
        let Some((o, d)) = self.screen_ray(css) else { return json!({"hit": false, "reason": "no camera"}) };
        let Some((e, dist, _)) = crate::zfight::ray_pick(&self.world, &self.renderer, o, d, 0.002) else {
            if copy {
                self.debug_mode.toast = Some("nothing under the cursor".into());
            }
            return json!({"hit": false});
        };
        let point = o + d.normalize() * dist;
        let w = &self.world;
        let name_of = |e: Entity| w.get::<&Name>(e).map(|n| n.0.clone()).unwrap_or_default();
        let name = name_of(e);
        let scene = w.get::<&crate::screens::SceneTag>(e).map(|t| self.scenes.name_of(*t).to_string()).unwrap_or_else(|_| "persistent".into());
        let local = w.get::<&scene::GlobalTransform>(e).map(|g| g.translation()).unwrap_or_default();
        let abs = self.origin + local.as_dvec3();
        let (mesh, material) = match w.get::<&MeshRenderer>(e) {
            Ok(mr) => (
                json!({"id": mr.mesh.0, "name": self.renderer.mesh(mr.mesh).map(|m| m.name.clone()).unwrap_or_default(), "layer": mr.layer}),
                json!({"id": mr.material.0, "name": self.renderer.material(mr.material).map(|m| m.name.clone()).unwrap_or_default()}),
            ),
            Err(_) => (json!(null), json!(null)),
        };
        let mut parents = vec![];
        let mut cur = e;
        while let Ok(p) = w.get::<&scene::Parent>(cur).map(|p| p.0) {
            if parents.len() > 32 {
                break;
            }
            parents.push(json!({"id": p.id(), "name": name_of(p)}));
            cur = p;
        }
        let r3 = |v: [f64; 3]| [(v[0] * 1000.0).round() / 1000.0, (v[1] * 1000.0).round() / 1000.0, (v[2] * 1000.0).round() / 1000.0];
        let info = json!({
            "hit": true, "id": e.id(), "name": name, "scene": scene,
            "pos": r3(local.as_dvec3().to_array()), "abs": r3(abs.to_array()), "hit_point": r3(point.as_dvec3().to_array()), "distance": (dist * 100.0).round() / 100.0,
            "mesh": mesh, "material": material, "parents": parents,
            "body": self.physics.body(&self.world, e).is_some(),
        });
        if copy {
            let quoted = |v: &serde_json::Value| match v.as_str() {
                Some(s) if !s.is_empty() => format!(" \"{s}\""),
                _ => String::new(),
            };
            let chain: Vec<String> = parents.iter().map(|p| format!("{}#{}", p["name"].as_str().unwrap_or(""), p["id"])).collect();
            let text = format!(
                "{name}#{id} · scene {scene}\npos {pos:?} · abs {abs:?}\nmesh #{mid}{mname} · material #{tid}{tname}\nparents: {chain}\nengine.call(\"entity\", {{\"id\": {id}}})",
                id = e.id(),
                pos = info["pos"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect::<Vec<_>>()).unwrap_or_default(),
                abs = info["abs"].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect::<Vec<_>>()).unwrap_or_default(),
                mid = info["mesh"]["id"],
                mname = quoted(&info["mesh"]["name"]),
                tid = info["material"]["id"],
                tname = quoted(&info["material"]["name"]),
                chain = if chain.is_empty() { "-".to_string() } else { chain.join(" > ") },
            );
            if let Some(clip) = web_sys::window().map(|w| w.navigator().clipboard()) {
                let _ = clip.write_text(&text);
            }
            self.debug_mode.toast = Some(format!("copied: {}#{} ({})", info["name"].as_str().unwrap_or(""), e.id(), info["scene"].as_str().unwrap_or("")));
            log_info!("inspect: {}", text.replace('\n', " | "));
            let (now, step, frame) = (log::now_ms(), self.time.step, self.time.frame);
            let mut ev = info.clone();
            ev["kind"] = json!("pick");
            self.debug_mode.record_event(now, step, frame, ev);
        }
        info
    }

    pub fn request_pointer_lock(&self) {
        self.canvas.request_pointer_lock();
    }

    pub fn exit_pointer_lock(&self) {
        if let Some(d) = web_sys::window().and_then(|w| w.document()) {
            d.exit_pointer_lock();
        }
    }
}

/// Runs one fixed simulation step (game logic, then physics).
pub(crate) fn fixed_step<G: Game>(ctx: &mut Ctx, game: &mut G) {
    let t0 = log::now_ms();
    crate::interp::before_step(ctx);
    ctx.input.begin_fixed();
    game.fixed_update(ctx);
    ctx.input.end_fixed();
    let t1 = log::now_ms();
    let dt = ctx.time.fixed_dt;
    crate::character::step_characters(ctx, dt);
    crate::particles::step_all(ctx, dt);
    crate::animation::advance_all(&mut ctx.world, dt);
    crate::batch::step_motion(&mut ctx.world, dt);
    let Ctx { physics, world, .. } = ctx;
    physics.step(world, dt);
    let t2 = log::now_ms();
    ctx.time.acc_game_fixed += (t1 - t0) as f32;
    ctx.time.acc_physics += (t2 - t1) as f32;
    ctx.time.step += 1;
    ctx.time.sim_time = ctx.time.step as f64 * dt as f64;
    log::set_step(ctx.time.step);
}

/// Entry point: call from your `#[wasm_bindgen(start)]` function.
pub fn run<G: Game>(game: G) {
    run_with(game, crate::renderer::RenderConfig::default());
}

/// Like [`run`] with startup render options (MSAA). A stored player choice
/// ([`crate::renderer::set_msaa_pref`]) overrides them, the page URL `?msaa=1|4` overrides both.
pub fn run_with<G: Game>(game: G, config: crate::renderer::RenderConfig) {
    debug::install_panic_hook();
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(e) = start(game, config).await {
            debug::fatal(&e);
        }
    });
}

async fn start<G: Game>(game: G, mut config: crate::renderer::RenderConfig) -> Result<(), String> {
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let canvas: HtmlCanvasElement = match document.get_element_by_id("canvas") {
        Some(c) => c.dyn_into().map_err(|_| "#canvas is not a <canvas>")?,
        None => {
            let c: HtmlCanvasElement = document
                .create_element("canvas")
                .map_err(|e| format!("{e:?}"))?
                .unchecked_into();
            c.set_id("canvas");
            document
                .body()
                .ok_or("no body")?
                .append_child(&c)
                .map_err(|e| format!("{e:?}"))?;
            c
        }
    };
    let _ = canvas.set_attribute("tabindex", "0");

    // Threads backend: start the Web Worker pool before anything can use the job system.
    match crate::jobs::init().await {
        Ok(n) if crate::jobs::threads_backend() => {
            log_info!("job system: {n} worker threads (shared memory + atomics)")
        }
        Ok(_) => {}
        Err(e) => crate::log_warn!("job system: thread pool failed, running serial: {e}"),
    }
    if let Some(m) = crate::renderer::msaa_pref() {
        config.msaa = m;
    }
    let query = js_sys::Reflect::get(&window, &"location".into()).and_then(|l| js_sys::Reflect::get(&l, &"search".into())).ok().and_then(|s| s.as_string()).unwrap_or_default();
    if let Some(m) = query.trim_start_matches('?').split('&').find_map(|kv| kv.strip_prefix("msaa=")).and_then(|v| v.parse::<u32>().ok()) {
        config.msaa = m;
    }
    let renderer = Renderer::new(&canvas, config).await?;
    log_info!("render config: {}x MSAA{}", renderer.samples, if renderer.samples == 1 { " (FXAA)" } else { "" });
    debug::install_gpu_error_handler(&renderer.device);
    let ui = Ui::new(&document).map_err(|e| format!("ui: {e:?}"))?;

    let ctx = Rc::new(RefCell::new(Ctx {
        world: World::new(),
        renderer,
        physics: Physics::new(),
        input: Input::default(),
        ui,
        panels: Panels::default(),
        audio: Audio::default(),
        debug_mode: Default::default(),
        zcheck: Default::default(),
        interpolation: true,
        interp_shown: false,
        trace: None,
        save: Default::default(),
        scenes: Default::default(),
        gltf_defaults: Default::default(),
        generators: Default::default(),
        streaming: Default::default(),
        terrain: None,
        origin: glam::DVec3::ZERO,
        time: Time {
            fixed_dt: 1.0 / 60.0,
            step: 0,
            sim_time: 0.0,
            frame: 0,
            frame_dt: 0.0,
            fps: 0.0,
            frame_ms: 0.0,
            time_scale: 1.0,
            paused: false,
            phases: Phases::default(),
            accumulator: 0.0,
            acc_game_fixed: 0.0,
            acc_physics: 0.0,
        },
        rng: Rng(0x9E37_79B9_7F4A_7C15),
        canvas: canvas.clone(),
        models: ModelCache::default(),
        me: Weak::new(),
        last_render_error: None,
    }));
    ctx.borrow_mut().me = Rc::downgrade(&ctx);
    install_input(&ctx, &canvas).map_err(|e| format!("input: {e:?}"))?;
    resize(&mut ctx.borrow_mut());

    let game = Rc::new(RefCell::new(game));
    game.borrow_mut().init(&mut ctx.borrow_mut());
    {
        let g = game.borrow();
        let mut cmds = g.debug_commands();
        cmds.extend(g.inactive_debug_commands().into_iter().map(|(n, d, _)| (n, d)));
        cmds.sort();
        cmds.dedup_by_key(|c| c.0);
        crate::debug::warn_shadowed(&cmds);
    }
    debug::install_api(ctx.clone(), game.clone());

    // requestAnimationFrame loop
    #[allow(clippy::type_complexity)]
    let f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    let mut last_ts: Option<f64> = None;
    *g.borrow_mut() = Some(Closure::new(move |ts: f64| {
        let dt = last_ts
            .map(|l| ((ts - l) / 1000.0) as f32)
            .unwrap_or(0.0)
            .clamp(0.0, 0.25);
        last_ts = Some(ts);
        {
            let mut c = ctx.borrow_mut();
            let mut gm = game.borrow_mut();
            frame(&mut c, &mut *gm, dt);
        }
        if let Some(cb) = f.borrow().as_ref() {
            let _ = web_sys::window()
                .unwrap()
                .request_animation_frame(cb.as_ref().unchecked_ref());
        }
    }));
    window
        .request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref())
        .map_err(|e| format!("{e:?}"))?;

    log_info!(
        "engine ready (WebGPU, fixed step {:.4}s). AI API: window.engine.call('help')",
        1.0 / 60.0
    );
    debug::mark_ready();
    Ok(())
}

fn resize(c: &mut Ctx) {
    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .unwrap_or(1.0);
    let w = ((c.canvas.client_width() as f64) * dpr).round().max(1.0) as u32;
    let h = ((c.canvas.client_height() as f64) * dpr).round().max(1.0) as u32;
    // the backing store has the scene size (render_scale applied): no full-resolution post pass
    let (bw, bh) = c.renderer.scene_size_for(w, h);
    if c.canvas.width() != bw || c.canvas.height() != bh || c.renderer.size() != (w, h) {
        c.canvas.set_width(bw);
        c.canvas.set_height(bh);
        c.renderer.resize(w, h);
    }
}

fn frame<G: Game>(c: &mut Ctx, game: &mut G, dt: f32) {
    let t0 = log::now_ms();
    let step0 = c.time.step;
    c.time.acc_game_fixed = 0.0;
    c.time.acc_physics = 0.0;
    c.time.frame += 1;
    c.time.frame_dt = dt;
    c.input.poll_gamepad();
    resize(c);

    c.physics.begin_frame();
    if !c.time.paused {
        c.time.accumulator += dt * c.time.time_scale;
        let mut n = 0;
        while c.time.accumulator >= c.time.fixed_dt && n < 8 {
            fixed_step(c, game);
            c.time.accumulator -= c.time.fixed_dt;
            n += 1;
        }
        if n == 8 {
            c.time.accumulator = 0.0; // spiral-of-death guard
        }
        crate::interp::after_steps(c, n > 0);
    } else {
        crate::interp::after_steps(c, false);
    }
    let t_fixed = log::now_ms();
    // Touch: joystick axis + hold detection; taps on world panels click them (and are consumed).
    c.input.touch.update(log::now_ms());
    if !c.input.touch.taps.is_empty() && !c.panels.is_empty() {
        let taps = std::mem::take(&mut c.input.touch.taps);
        let frame_no = c.time.frame;
        for tp in taps {
            let hit = c.screen_ray(tp).and_then(|(o, d)| c.panels.raycast(&c.world, &c.physics, o, d));
            let click = hit.and_then(|(id, p, _)| c.panels.click_at(id, p, frame_no));
            match click {
                Some(click) => c.ui.push_action(&click.action),
                None => c.input.touch.taps.push(tp),
            }
        }
    }
    {
        let Ctx { ui, input, .. } = c;
        ui.sync_stick(input.touch.stick());
    }
    // Debug mode: Ctrl+click inspects the object under the pointer (copied to the clipboard); the
    // click is consumed so the game does not also act on it. Plain clicks are untouched.
    if c.debug_mode.on() && c.input.button_pressed(MouseButton::Left) && (c.input.key("ControlLeft") || c.input.key("ControlRight")) {
        c.input.consume_press(MouseButton::Left);
        let p = if c.input.pointer_locked { Vec2::new(c.canvas.client_width() as f32, c.canvas.client_height() as f32) * 0.5 } else { c.input.pointer };
        c.inspect_at(p, true);
    }
    // World panels: hover + clicks become UI actions before the game reads them.
    if !c.panels.is_empty() {
        let p = if c.input.pointer_locked {
            Vec2::new(
                c.canvas.client_width() as f32,
                c.canvas.client_height() as f32,
            ) * 0.5
        } else {
            c.input.pointer
        };
        let ray = c.screen_ray(p);
        let clicked = c.input.button_pressed(MouseButton::Left);
        let frame_no = c.time.frame;
        let click = c
            .panels
            .pointer(&c.world, &c.physics, ray, clicked, frame_no);
        c.input.pointer_over_panel = c.panels.hovered.is_some();
        if let Some(click) = click {
            c.ui.push_action(&click.action);
            c.input.consume_press(MouseButton::Left);
        }
    }
    game.update(c, dt);
    let t_update = log::now_ms();
    {
        let Ctx { scenes, world, physics, .. } = c;
        scenes.tag_new(world, physics);
    }
    c.update_streaming();
    c.update_terrain();
    crate::character::update_cameras(c, dt);
    crate::water::follow_camera(c);
    crate::animation::apply_poses(&mut c.world);
    scene::propagate_transforms(&mut c.world);
    if let Some((e, left, samples)) = c.trace.as_mut().filter(|t| t.1 > 0) {
        if let Ok(g) = c.world.get::<&scene::GlobalTransform>(*e) {
            let p = g.translation();
            samples.push([dt * 1000.0, (c.time.step - step0) as f32, p.x, p.y, p.z]);
        }
        *left -= 1;
    }
    crate::animation::deform_all(c);
    c.panels.flush(&mut c.renderer);
    {
        let pending = c.models.pending.len();
        let Ctx { zcheck, world, renderer, .. } = c;
        let cam = world.query::<(&scene::Camera, &scene::GlobalTransform)>().without::<&crate::render_texture::RenderTextureCamera>().iter().next().map(|(_, g)| g.translation());
        zcheck.tick(world, renderer, pending, log::now_ms(), cam);
    }
    {
        let cam = scene::main_camera_pose(&c.world);
        let (now, step, frame) = (log::now_ms(), c.time.step, c.time.frame);
        c.debug_mode.tick(now, step, frame, cam);
        let snap = || {
            serde_json::json!({
                "fps": (c.time.fps * 10.0).round() / 10.0,
                "cpu_frame_ms": (c.time.frame_ms * 100.0).round() / 100.0,
                "cpu_phases": c.time.phases,
                "gpu_ms": c.renderer.stats.gpu_ms,
                "canvas": [c.renderer.stats.width, c.renderer.stats.height],
                "draw_calls": c.renderer.stats.draw_calls,
                "triangles": c.renderer.stats.triangles,
                "entities": c.world.len(),
                "bodies": c.physics.body_count(),
                // what the game was doing: a "does nothing" report can be read from the log alone
                "characters": crate::character::describe_characters(&c.world),
                "game_state": game.debug_state(c),
            })
        };
        let v = if c.debug_mode.on() { snap() } else { serde_json::Value::Null }; // only built while logging
        c.debug_mode.record_perf(now, step, frame, c.time.frame_dt * 1000.0, || v);
        let perf = if c.debug_mode.on() {
            // colour: green at the display rate, amber below 55, red below 30 fps
            let fps = c.time.fps;
            let col = if fps >= 55.0 { "#4ade80" } else if fps >= 30.0 { "#fbbf24" } else { "#f87171" };
            let gpu = c.renderer.stats.gpu_ms.filter(|g| g.samples > 0).map(|g| format!(" · GPU {:.1} ms", g.total)).unwrap_or_default();
            let scale = c.renderer.settings.render_scale;
            let res = if scale < 0.999 { format!(" · res {:.0}%", scale * 100.0) } else { String::new() };
            format!("<b style='color:{col}'>{fps:.0} fps</b> · CPU {:.1} ms{gpu}{res}", c.time.frame_ms)
        } else {
            String::new()
        };
        let badge = c.debug_mode.badge_html(&perf);
        c.ui.set("debugmode", &badge);
    }
    c.audio.step = c.time.step;
    c.audio.frame = c.time.frame;
    let cam = scene::main_camera_pose(&c.world);
    c.audio.update(&c.world, cam);

    let t_engine = log::now_ms();
    {
        let r = &mut c.renderer;
        crate::sky::update(&mut r.settings, &mut r.sky, c.time.sim_time);
    }
    match c.renderer.render(&c.world) {
        Ok(()) => c.last_render_error = None,
        Err(e) => {
            if c.last_render_error.as_deref() != Some(e.as_str()) {
                log_error!("render: {e}");
                c.last_render_error = Some(e);
            }
        }
    }
    let t_render = log::now_ms();
    c.renderer.adapt_resolution(c.time.frame_dt);
    let vp = c.renderer.last_view_proj;
    let (cw, ch) = (
        c.canvas.client_width() as f32,
        c.canvas.client_height() as f32,
    );
    let Ctx { ui, world, .. } = c;
    ui.sync_labels(world, vp, cw, ch);
    c.input.end_frame();

    let ms = (log::now_ms() - t0) as f32;
    let a = 0.1;
    c.time.frame_ms = c.time.frame_ms * (1.0 - a) + ms * a;
    {
        let fixed_other = ((t_fixed - t0) as f32 - c.time.acc_game_fixed - c.time.acc_physics).max(0.0);
        let p = &mut c.time.phases;
        let m = |old: f32, new: f32| old * (1.0 - a) + new * a;
        p.game_fixed = m(p.game_fixed, c.time.acc_game_fixed);
        p.physics = m(p.physics, c.time.acc_physics);
        p.update = m(p.update, (t_update - t_fixed) as f32 + fixed_other);
        p.engine = m(p.engine, (t_engine - t_update) as f32 + (log::now_ms() - t_render) as f32);
        p.render = m(p.render, (t_render - t_engine) as f32);
        p.steps = (c.time.step - step0) as u32;
    }
    if dt > 0.0 {
        c.time.fps = c.time.fps * (1.0 - a) + (1.0 / dt) * a;
    }
}

fn css_pos(canvas: &HtmlCanvasElement, ev: &web_sys::PointerEvent) -> Vec2 {
    let r = canvas.get_bounding_client_rect();
    Vec2::new(ev.client_x() as f32 - r.left() as f32, ev.client_y() as f32 - r.top() as f32)
}

fn install_input(ctx: &Rc<RefCell<Ctx>>, canvas: &HtmlCanvasElement) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();

    fn is_editable(ev: &web_sys::Event) -> bool {
        ev.target()
            .and_then(|t| t.dyn_into::<web_sys::Element>().ok())
            .map(|el| {
                let tag = el.tag_name().to_ascii_uppercase();
                tag == "INPUT"
                    || tag == "TEXTAREA"
                    || tag == "SELECT"
                    || el.has_attribute("contenteditable")
            })
            .unwrap_or(false)
    }

    macro_rules! listen {
        ($target:expr, $event:expr, $ty:ty, $body:expr) => {{
            let c = ctx.clone();
            let cb = Closure::<dyn FnMut($ty)>::new(move |ev: $ty| {
                if let Ok(mut c) = c.try_borrow_mut() {
                    #[allow(clippy::redundant_closure_call)]
                    ($body)(&mut *c, ev);
                }
            });
            $target.add_event_listener_with_callback($event, cb.as_ref().unchecked_ref())?;
            cb.forget();
        }};
    }

    listen!(
        window,
        "keydown",
        web_sys::KeyboardEvent,
        |c: &mut Ctx, ev: web_sys::KeyboardEvent| {
            if is_editable(&ev) {
                return;
            }
            if matches!(
                ev.code().as_str(),
                "Space" | "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight" | "Tab"
            ) {
                ev.prevent_default();
            }
            if ev.repeat() {
                return; // auto-repeat: the key is already held
            }
            c.key_event(&ev.code(), &ev.key(), ev.shift_key(), true, "dom");
            c.audio.resume();
        }
    );
    listen!(
        window,
        "keyup",
        web_sys::KeyboardEvent,
        |c: &mut Ctx, ev: web_sys::KeyboardEvent| {
            c.key_event(&ev.code(), &ev.key(), ev.shift_key(), false, "dom");
        }
    );
    listen!(
        window,
        "blur",
        web_sys::FocusEvent,
        |c: &mut Ctx, _ev: web_sys::FocusEvent| {
            c.input.release_user(); // keys held through the AI API stay held (deterministic tests)
            c.audio.set_focused(false);
        }
    );
    listen!(
        window,
        "focus",
        web_sys::FocusEvent,
        |c: &mut Ctx, _ev: web_sys::FocusEvent| {
            c.audio.set_focused(true);
        }
    );
    {
        let doc = window.document().unwrap();
        listen!(
            doc,
            "visibilitychange",
            web_sys::Event,
            |c: &mut Ctx, _ev: web_sys::Event| {
                let d = web_sys::window().and_then(|w| w.document());
                let visible = d.as_ref().map(|d| !d.hidden()).unwrap_or(true);
                let has_focus = d.and_then(|d| d.has_focus().ok()).unwrap_or(true);
                c.audio.set_focused(visible && has_focus);
            }
        );
    }
    listen!(
        canvas,
        "pointerdown",
        web_sys::PointerEvent,
        |c: &mut Ctx, ev: web_sys::PointerEvent| {
            if ev.pointer_type() == "touch" {
                // keep receiving moves for this finger even outside the canvas
                let _ = c.canvas.set_pointer_capture(ev.pointer_id());
                let p = css_pos(&c.canvas, &ev);
                c.touch_event(ev.pointer_id(), p, "start", "dom");
                c.audio.resume();
                return;
            }
            if let Some(b) = MouseButton::from_dom(ev.button()) {
                c.button_event(b, true, "dom");
            }
            let _ = c.canvas.focus();
            c.audio.resume();
        }
    );
    listen!(
        window,
        "pointerup",
        web_sys::PointerEvent,
        |c: &mut Ctx, ev: web_sys::PointerEvent| {
            if ev.pointer_type() == "touch" {
                let p = css_pos(&c.canvas, &ev);
                c.touch_event(ev.pointer_id(), p, "end", "dom");
                return;
            }
            if let Some(b) = MouseButton::from_dom(ev.button()) {
                c.button_event(b, false, "dom");
            }
        }
    );
    listen!(
        window,
        "pointercancel",
        web_sys::PointerEvent,
        |c: &mut Ctx, ev: web_sys::PointerEvent| {
            if ev.pointer_type() == "touch" {
                let p = css_pos(&c.canvas, &ev);
                c.touch_event(ev.pointer_id(), p, "end", "dom");
            }
        }
    );
    listen!(
        window,
        "pointermove",
        web_sys::PointerEvent,
        |c: &mut Ctx, ev: web_sys::PointerEvent| {
            if ev.pointer_type() == "touch" {
                let p = css_pos(&c.canvas, &ev);
                c.touch_event(ev.pointer_id(), p, "move", "dom");
                return;
            }
            let rect = c.canvas.get_bounding_client_rect();
            c.input.pointer = Vec2::new(
                ev.client_x() as f32 - rect.left() as f32,
                ev.client_y() as f32 - rect.top() as f32,
            );
            c.input.pointer_delta += Vec2::new(ev.movement_x() as f32, ev.movement_y() as f32);
        }
    );
    {
        let c = ctx.clone();
        let cb = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(move |ev: web_sys::WheelEvent| {
            ev.prevent_default();
            if let Ok(mut c) = c.try_borrow_mut() {
                c.input.wheel += ev.delta_y() as f32;
            }
        });
        let opts = web_sys::AddEventListenerOptions::new();
        opts.set_passive(false);
        canvas.add_event_listener_with_callback_and_add_event_listener_options(
            "wheel",
            cb.as_ref().unchecked_ref(),
            &opts,
        )?;
        cb.forget();
    }
    let cb = Closure::<dyn FnMut(web_sys::Event)>::new(|ev: web_sys::Event| ev.prevent_default());
    canvas.add_event_listener_with_callback("contextmenu", cb.as_ref().unchecked_ref())?;
    cb.forget();
    let doc = window.document().unwrap();
    listen!(
        doc,
        "pointerlockchange",
        web_sys::Event,
        |c: &mut Ctx, _ev: web_sys::Event| {
            let locked = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.pointer_lock_element())
                .is_some();
            c.input.pointer_locked = locked;
        }
    );
    Ok(())
}

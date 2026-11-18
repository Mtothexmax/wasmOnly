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
    /// Read held input here (`input.key`); edge events (`key_pressed`) belong in `update`.
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
    /// Game state snapshot for AI inspection (`engine.call("game_state")`).
    fn debug_state(&self, _ctx: &Ctx) -> Value {
        Value::Null
    }
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
    #[serde(skip)]
    accumulator: f32,
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
    /// Debug mode (` key): input + camera log file, badge (see [`crate::debugmode`]).
    pub debug_mode: crate::debugmode::DebugMode,
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
        let name = url.rsplit('/').next().unwrap_or(url).to_string();
        let root = self.world.spawn((t, Name(name)));
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
    game.fixed_update(ctx);
    let dt = ctx.time.fixed_dt;
    let Ctx { physics, world, .. } = ctx;
    physics.step(world, dt);
    ctx.time.step += 1;
    ctx.time.sim_time = ctx.time.step as f64 * dt as f64;
    log::set_step(ctx.time.step);
}

/// Entry point: call from your `#[wasm_bindgen(start)]` function.
pub fn run<G: Game>(game: G) {
    debug::install_panic_hook();
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(e) = start(game).await {
            debug::fatal(&e);
        }
    });
}

async fn start<G: Game>(game: G) -> Result<(), String> {
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
    let renderer = Renderer::new(&canvas).await?;
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
            accumulator: 0.0,
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
    if c.canvas.width() != w || c.canvas.height() != h || c.renderer.size() != (w, h) {
        c.canvas.set_width(w);
        c.canvas.set_height(h);
        c.renderer.resize(w, h);
    }
}

fn frame<G: Game>(c: &mut Ctx, game: &mut G, dt: f32) {
    let t0 = log::now_ms();
    c.time.frame += 1;
    c.time.frame_dt = dt;
    c.input.poll_gamepad();
    resize(c);

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
    }
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
    scene::propagate_transforms(&mut c.world);
    c.panels.flush(&mut c.renderer);
    {
        let cam = c
            .world
            .query::<(&scene::Camera, &scene::GlobalTransform)>()
            .iter()
            .next()
            .map(|(_, g)| g.0);
        let (now, step, frame) = (log::now_ms(), c.time.step, c.time.frame);
        c.debug_mode.tick(now, step, frame, cam);
        let badge = c.debug_mode.badge_html();
        c.ui.set("debugmode", &badge);
    }
    c.audio.step = c.time.step;
    c.audio.frame = c.time.frame;
    let cam = c
        .world
        .query::<(&scene::Camera, &scene::GlobalTransform)>()
        .iter()
        .next()
        .map(|(_, g)| g.0);
    c.audio.update(&c.world, cam);

    match c.renderer.render(&c.world) {
        Ok(()) => c.last_render_error = None,
        Err(e) => {
            if c.last_render_error.as_deref() != Some(e.as_str()) {
                log_error!("render: {e}");
                c.last_render_error = Some(e);
            }
        }
    }
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
            c.input.release_all();
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

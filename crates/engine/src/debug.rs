//! AI-first runtime API: `window.engine.call(name, args) -> {ok, result | error}`.
//!
//! Everything an agent needs to observe and steer the running game without reading pixels:
//! scene graph, components, physics, logs, errors, input injection, deterministic stepping.
//! `engine.call("help")` lists all commands. The CDP bridge (`tools/bridge.mjs`) drives this API
//! from the command line or as an MCP server.

use crate::app::{fixed_step, Ctx, Game};
use crate::log::{self, Level};
use crate::physics::Body;

use crate::scene::{
    Camera, GlobalTransform, MeshRenderer, Name, Parent, PointLight, Transform, WorldLabel,
};
use glam::{EulerRot, Quat, Vec2, Vec3};
use hecs::Entity;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// `name` is an engine command (a game command with this name is never called).
pub fn is_engine_command(name: &str) -> bool {
    COMMANDS.iter().any(|(n, _)| *n == name)
}

/// Logs a warning for every game command that an engine command of the same name shadows.
pub(crate) fn warn_shadowed(game_cmds: &[(&'static str, &'static str)]) {
    for (n, _) in game_cmds {
        if is_engine_command(n) {
            crate::log_warn!("game command '{n}' is shadowed by the engine command of the same name and will never be called; rename it");
        }
    }
}

pub const COMMANDS: &[(&str, &str)] = &[
    ("help", "List commands."),
    ("trace", "Record the drawn position of an entity every rendered frame (smoothness check). args: {id, frames?: 120} starts; {} returns {done, samples: [[frame_ms, fixed_steps, x, y, z]], speed_jitter_pct} (mean frame-to-frame change of the drawn speed relative to the mean speed, in %; a few % = smooth, a hitch or a 0/2-step frame shows as a jump)."),
    ("interpolation", "Render interpolation of dynamic bodies + characters between fixed steps (smooth motion at any display rate). args: {on?: bool}."),
    ("sky", "Sky + day/night: mode, time of day, sun/moon elevation, daylight, night, light, ambient. args: {time?: hours (turns the cycle on), cycle?: bool, day_length?: minutes per day (0 = still), mode?: gradient|atmosphere, clouds?, stars?, moon?, skybox?: url|null, skybox_blend?, skybox_night_blend?}."),
    ("inspect", "What is drawn at a canvas point (CSS px): entity, name, scene, local + absolute position, mesh, material, parent chain. args: {x, y, copy?: false}. In debug mode Ctrl+click does this and copies it to the clipboard."),
    ("characters", "Character controllers: pos, grounded, velocity, jumps, held jump, wish, blocked time, touching entities. args: {id?, pos?: [x,y,z] (place)}"),
    ("world", "Open world: floating origin + streaming layers (loaded cells, loads/unloads). args: {shift?: [x,y,z], origin_shift_distance?: m, layer?: name, radius?: m, enabled?: bool}"),
    ("grass", "GPU grass settings + blade count. args: any GrassSettings field (enabled, radius, spacing, height, width, layer, full_density, color_base, color_tip, wind)"),
    ("terrain", "Heightmap terrain: info, streaming state. args: {height_at?: [x,z] (absolute), flatten?: {center:[x,z], radius, height?, falloff?}, raise?: {center, radius, amount, falloff?}, paint?: {layer 0-3, center, radius, strength?, falloff?}, chunks?: true}"),
    ("lights", "Clustered lights: counts + last frame stats. args: {add?: {pos, color?, intensity?, range?, spot?: true, dir?, inner_deg?, outer_deg?}, grid?: {n, spacing?, center?} (stress test), clear_test?: true}"),
    ("generate", "Procedural generators (geom kit). No args: list. args: {name, params?: {...}, seed?: 1, pos?: [x,y,z], yaw_deg?, regenerate?: true (drop cache)}"),
    ("batch", "Static batches + moving (Motion) instances. args: {id?: root entity to batch now, cell?: 64}"),
    ("collisions", "Collision events (start/stop of solid contacts, sensor/trigger overlaps, character touches), newest last. args: {since?: step, entity?: id|name, limit?: 64} (started/stopped = counts) | {touching: id|name} | {inside: trigger id|name}"),
    ("overlap", "Entities overlapping a sphere or box. args: {pos, radius?: 1} | {pos, half: [x,y,z]}, sensors?: false"),
    ("trigger", "Spawn a box trigger zone (sensor, reports enter/exit). args: {pos, half?: [1,1,1], name?: 'trigger'}"),
    ("gpu_scene", "GPU-driven scene of the main view: object slots, draw groups, id-buffer layout, last GPU counters (frustum, occluded, visible, shadow instances, LOD levels)"),
    ("render_texture", "Render textures (portraits, mirrors). args: {id?, render?: true (re-render once), export?: true (PNG blob URL -> url)}"),
    ("animation", "Animators (glTF clips, skinning, morph targets). args: {id?, play?: clip, fade?: 0.2, speed?: 1, looped?: true, stop?: true, morph?: index, weight?: w}"),
    ("particles", "Particle emitters + live counts. args: {preset?: 'fire'|'smoke'|'sparks'|'fireflies'|'dust'|'magic'|'rain'|'splash', pos?: [x,y,z], burst?: n, layer?: 0..31, seed?: n, name?} spawns one (name defaults to the preset); without preset, {name?} filters the list."),
    ("scenes", "Scenes (entity groups) with counts. args: {unload?: name, begin?: name}"),
    ("save", "Save slots (localStorage) + blobs (IndexedDB). args: {op?: 'list'|'get'|'set'|'delete'|'blobs'|'delete_blob', slot?, data?, key?}"),
    ("stats", "Frame/sim timing, renderer stats, entity and body counts."),
    ("entities", "List entities. args: {name?: substring, limit?: 200, with?: 'mesh'|'body'|'light'|'label'|'camera'}"),
    ("entity", "Full detail of one entity. args: {id}"),
    ("set", "Modify an entity. args: {id, pos?: [x,y,z], rot?: [deg x,y,z], scale?: [x,y,z], vel?: [x,y,z], color?: '#rrggbb', opacity?: 0..1 (this entity and its children; keeps its shadow), layer?: 0..31 (MeshRenderer::layer, children too)}"),
    ("spawn", "Spawn a primitive or a glTF model. args: {shape: 'cube'|'sphere', pos, size?: 1, color?: '#rrggbb', body?: 'dynamic'|'fixed'|'kinematic'|'none', name?} or {gltf: 'assets/x.glb', pos, size?: scale, auto_lod?: bool} or {camera: true, pos?, look_at?, fov?, name?, main?: bool} or {water: [x, z], pos, spacing?: 0.25, name?} (bounded water surface)"),
    ("despawn", "Remove entity and children. args: {id}"),
    ("pause", "Pause simulation (rendering continues)."),
    ("resume", "Resume simulation."),
    ("step", "Run N fixed steps immediately (works while paused). args: {n?: 1}"),
    ("time_scale", "Set simulation speed. args: {value}"),
    ("seed", "Reseed the engine RNG (ctx.rng). args: {value}"),
    ("input", "Inject input. args: {key?: 'KeyW', down?: bool} | {button?: 0|1|2, down?} | {x?, y?: pointer position, CSS px} | {dx?, dy?} | {action?: 'ui-action'} | {touch: {id, x, y, phase: 'start'|'move'|'end'}} | {release_all: true} | {stick: bool} (virtual joystick on/off). Returns keys, buttons and touch gestures"),
    ("logs", "Structured log entries (ring buffer of the last 10000). args: {since?: seq, level?: 'debug'|'info'|'warn'|'error', limit?: 100 (newest N; 'omitted' counts the rest)}. Returns next, oldest, omitted, entries."),
    ("render", "Get/set any render setting by name (call without args to list them): shadows, debug_view 0|1|2, sun_dir, sun_intensity, sky, sky_zenith, sky_horizon, fog_density, fog_height_falloff, fog_start, fog_max, fog_color ([r,g,b] or null), exposure, bloom, bloom_intensity, bloom_threshold, depth_darkening, vignette, saturation, particles, gpu_culling, occlusion_culling, camera_jitter, depth_jitter, reverse_draw_order, freeze_time, ... Result includes gpu_culling_supported."),
    ("cameras", "Every Camera entity: id, name, main (renders the canvas), marked_main, render_texture, layers."),
    ("camera", "Main camera pose + projection (the one with MainCamera, else the first Camera); args {pos?, look_at?} moves it, {layers: mask} sets its layer mask, {main: id|name} makes another Camera entity the main one; `cameras` lists all."),
    ("raycast", "Physics ray cast. args: {origin, dir, max?: 1000, exclude?: id|name|[...] (ignore these bodies, e.g. the caster)}"),
    ("pick", "Entity under a canvas point (CSS px). args: {x, y, mesh?: true = test rendered meshes instead of physics colliders; also returns depth_ties (z-fighting partners)}"),
    ("ui", "HUD slot contents."),
    ("material", "Materials. Without id: every material (id, name, roughness, metallic, subsurface, clearcoat, bend). args: {id, roughness?, metallic?, subsurface?: 0..1, subsurface_color?: '#rrggbb' | null (= albedo), clearcoat?: 0..1, clearcoat_roughness?} sets them and returns the material."),
    ("bend", "Pushers that bend Material::bend geometry and GPU grass. args: {pushers?: [[x,y,z,radius], ...] (replaces; [] = none), material?: id, bend?: value (sets Material::bend), wind?: [dir_x, dir_z, strength] (shared with GPU grass; strength 0 = still)}. Returns pushers, wind and every material with bend > 0."),
    ("zfight", "args: {radius?: m around the camera or `center` (use in big worlds), center?: [x,y,z] (default radius 10), entity?: id|name|[...] (only pairs involving these and their children; a name matches every entity with it), exclude?: id|name|[...]}. Find z-fighting: coplanar, same-facing, overlapping triangles of different entities (opaque/masked). args: {eps?: 0.002 plane distance, tolerance?: 0.002 overlap, limit?: 50}. Returns pair count and the worst pairs with entities, world point and screen position."),
    ("debug_mode", "Debug mode (also the ` key): logs every key/mouse press and the camera pose once per second to logs/debug-*.log via the dev server, shows a badge. args: {on?: bool}. Returns file, line counts, last camera, last error."),
    ("audio", "Audio state: context state, sounds (ready/loading/failed, plays), active voices with 3D position, last 32 plays with sim step."),
    ("audio_play", "Play a sound by name. args: {sound, volume?: 1, pitch?: 1, pos?: [x,y,z], loop?, lowpass?: hz, highpass?: hz, bandpass?: [hz,q], echo?: [delay_s, feedback, mix?], reverb?: secs | [secs, mix], distort?: drive, pan?: -1..1, detune?: cents, bus?: sfx|voice|ambience|ui}. Returns the voice id."),
    ("music", "Streamed music. args: {} = state | {play: url, fade?: 1.5, volume?: 1, loop?: true} | {stop: true, fade?} | {prefetch: url}. State per track: buffering/playing/paused/blocked/error, time, duration, buffered_s/pct."),
    ("audio_stop", "Stop a voice (or all). args: {voice?, fade?: seconds (fade out, then stop)}"),
    ("audio_voice", "Change a playing voice. args: {voice, volume, fade?: seconds (linear ramp)}. Returns the voice's volume."),
    ("sfx", "Synthesize a sound from a preset and register it. args: {preset, name?: preset, seed?: 0 (0 = exact preset, else a mutation), overrides?: {wave, freq, freq_end, duty, attack, decay, sustain, sustain_time, release, vibrato_depth, vibrato_rate, arp_mult, arp_time, partials, noise_mix, lowpass, highpass, bitcrush, volume}}. Returns duration, peak, rms and the full recipe. args {} lists presets."),
    ("texgen", "Generate a seamless procedural texture. args: {preset, size?: 256, seed?: 1, scale?: 1, colors?: ['#dark','#light'], apply_to?: entity id (sets that entity's material texture), normal?: strength (also generates + applies a normal map), parallax?: uv depth (applies the height map as parallax)}. args {} lists presets."),
    ("audio_set", "Master audio. args: {volume?, music_volume?, sfx_volume?, voice_volume?, ambience_volume?, ui_volume?, muted?, pause_on_blur?: bool (suspend while the tab is unfocused, default true)}"),
    ("panels", "World panels (clickable 3D UI): widgets with rects, on-screen centres (CSS px), hover, click history."),
    ("panel_click", "Click a world panel through the engine path. args: {id, widget?: name} | {id, x, y: panel px}. Returns the action or null."),
    ("assets", "Loaded meshes/materials/textures and model load status."),
    ("game_state", "Game-defined state snapshot."),
];

thread_local! {
    static READY: RefCell<bool> = const { RefCell::new(false) };
}

pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = info.to_string();
        log::write(Level::Error, format!("PANIC: {msg}"));
        show_overlay("Rust panic", &msg);
        console_error_panic_hook::hook(info);
    }));
}

/// Fatal startup error: logged, shown in the page, available via `window.engine.fatal`.
pub fn fatal(msg: &str) {
    log::write(Level::Error, format!("FATAL: {msg}"));
    show_overlay("Engine failed to start", msg);
    if let Some(w) = web_sys::window() {
        let engine = ensure_engine_object(&w);
        let _ = js_sys::Reflect::set(&engine, &"fatal".into(), &msg.into());
    }
}

fn show_overlay(title: &str, msg: &str) {
    let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let el = match doc.get_element_by_id("engine-error") {
        Some(e) => e,
        None => {
            let Ok(e) = doc.create_element("div") else {
                return;
            };
            e.set_id("engine-error");
            if let Some(b) = doc.body() {
                let _ = b.append_child(&e);
            }
            e
        }
    };
    let esc = |s: &str| s.replace('&', "&amp;").replace('<', "&lt;");
    el.set_inner_html(&format!("<b>{}</b><pre>{}</pre>", esc(title), esc(msg)));
}

/// Creates a render pipeline inside error scopes. A pipeline the driver cannot build (e.g.
/// `VK_ERROR_UNKNOWN` from a mobile Vulkan driver) is invalid, and one `set_pipeline` with it
/// invalidates the whole command buffer, so nothing at all would be drawn. The returned flag turns
/// `true` once the failure is known (asynchronously, a frame or two later): callers skip draws with
/// that pipeline, the rest of the frame still renders. The error is logged and shown in the overlay.
pub fn create_pipeline_guarded(
    device: &web_sys::GpuDevice,
    desc: &web_sys::GpuRenderPipelineDescriptor,
    label: &str,
) -> Result<(web_sys::GpuRenderPipeline, Rc<std::cell::Cell<bool>>), String> {
    device.push_error_scope(web_sys::GpuErrorFilter::Internal);
    device.push_error_scope(web_sys::GpuErrorFilter::Validation);
    let created = device.create_render_pipeline(desc);
    let validation = device.pop_error_scope();
    let internal = device.pop_error_scope();
    let pipeline = created.map_err(|e| format!("{label}: {e:?}"))?;
    let broken = Rc::new(std::cell::Cell::new(false));
    let flag = broken.clone();
    let label = label.to_string();
    wasm_bindgen_futures::spawn_local(async move {
        for p in [validation, internal] {
            let Ok(v) = wasm_bindgen_futures::JsFuture::from(p).await else { continue };
            let Ok(err) = v.dyn_into::<web_sys::GpuError>() else { continue };
            flag.set(true);
            let msg = err.message();
            log::write(Level::Error, format!("WebGPU: render pipeline \"{label}\" failed, its draws are skipped: {msg}"));
            show_overlay(&format!("Render pipeline \"{label}\" failed on this GPU"), &msg);
            break;
        }
    });
    Ok((pipeline, broken))
}

pub fn install_gpu_error_handler(device: &web_sys::GpuDevice) {
    let cb = Closure::<dyn FnMut(web_sys::GpuUncapturedErrorEvent)>::new(
        |ev: web_sys::GpuUncapturedErrorEvent| {
            log::write(Level::Error, format!("WebGPU: {}", ev.error().message()));
        },
    );
    device.set_onuncapturederror(Some(cb.as_ref().unchecked_ref()));
    cb.forget();
}

/// Logs WGSL compile errors with line numbers (async; arrives a moment after startup).
pub fn report_shader_errors(module: &web_sys::GpuShaderModule, name: &str) {
    let promise = module.get_compilation_info();
    let name = name.to_string();
    wasm_bindgen_futures::spawn_local(async move {
        let Ok(info) = promise.await else { return };
        for m in info.messages().iter() {
            let Ok(m) = m.dyn_into::<web_sys::GpuCompilationMessage>() else {
                continue;
            };
            let level = match m.type_() {
                web_sys::GpuCompilationMessageType::Error => Level::Error,
                web_sys::GpuCompilationMessageType::Warning => Level::Warn,
                _ => Level::Info,
            };
            log::write(
                level,
                format!(
                    "WGSL {name}:{}:{}: {}",
                    m.line_num(),
                    m.line_pos(),
                    m.message()
                ),
            );
            if level == Level::Error {
                show_overlay(
                    &format!("WGSL error in {name}"),
                    &format!("line {}: {}", m.line_num(), m.message()),
                );
            }
        }
    });
}

fn ensure_engine_object(w: &web_sys::Window) -> js_sys::Object {
    let existing = js_sys::Reflect::get(w, &"engine".into())
        .ok()
        .filter(|v| v.is_object());
    match existing {
        Some(v) => v.unchecked_into(),
        None => {
            let o = js_sys::Object::new();
            let _ = js_sys::Reflect::set(w, &"engine".into(), &o);
            o
        }
    }
}

pub(crate) fn mark_ready() {
    READY.with(|r| *r.borrow_mut() = true);
    if let Some(w) = web_sys::window() {
        let engine = ensure_engine_object(&w);
        let _ = js_sys::Reflect::set(&engine, &"ready".into(), &true.into());
        if let Ok(ev) = web_sys::Event::new("engine-ready") {
            let _ = w.dispatch_event(&ev);
        }
    }
}

pub fn install_api<G: Game>(ctx: Rc<RefCell<Ctx>>, game: Rc<RefCell<G>>) {
    let w = web_sys::window().unwrap();
    let engine = ensure_engine_object(&w);
    let call = Closure::<dyn FnMut(String, JsValue) -> JsValue>::new(
        move |name: String, args: JsValue| -> JsValue {
            let args: Value = if args.is_undefined() || args.is_null() {
                Value::Null
            } else {
                js_sys::JSON::stringify(&args)
                    .ok()
                    .and_then(|s| s.as_string())
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or(Value::Null)
            };
            let res = match (ctx.try_borrow_mut(), game.try_borrow_mut()) {
                (Ok(mut c), Ok(mut g)) => handle(&name, &args, &mut c, &mut *g),
                _ => Err("engine is busy (re-entrant call)".into()),
            };
            let out = match res {
                Ok(v) => json!({"ok": true, "result": v}),
                Err(e) => json!({"ok": false, "error": e}),
            };
            js_sys::JSON::parse(&out.to_string()).unwrap_or(JsValue::NULL)
        },
    );
    let _ = js_sys::Reflect::set(&engine, &"call".into(), call.as_ref());
    let _ = js_sys::Reflect::set(
        &engine,
        &"version".into(),
        &env!("CARGO_PKG_VERSION").into(),
    );
    call.forget();
}

// ---------------------------------------------------------------------------------------------

fn v3(v: &Value) -> Option<Vec3> {
    let a = v.as_array()?;
    Some(Vec3::new(
        a.first()?.as_f64()? as f32,
        a.get(1)?.as_f64()? as f32,
        a.get(2)?.as_f64()? as f32,
    ))
}

fn arr(v: Vec3) -> Value {
    let r = |x: f32| (x as f64 * 1000.0).round() / 1000.0;
    json!([r(v.x), r(v.y), r(v.z)])
}

fn parse_color(s: &str) -> Option<u32> {
    u32::from_str_radix(s.trim_start_matches('#'), 16).ok()
}

/// `{id}` or `{name}` (the first entity with that name, lowest id).
fn entity_arg(ctx: &Ctx, args: &Value) -> Result<Entity, String> {
    if let (None, Some(name)) = (args.get("id"), args.get("name").and_then(|v| v.as_str())) {
        return ctx
            .world
            .query::<(Entity, &crate::scene::Name)>()
            .iter()
            .filter(|(_, n)| n.0 == name)
            .map(|(e, _)| e)
            .min_by_key(|e| e.id())
            .ok_or(format!("no entity named '{name}'"));
    }
    let id = args
        .get("id")
        .and_then(|v| v.as_u64())
        .ok_or("missing numeric 'id' (or 'name')")?;
    ctx.world
        .iter()
        .map(|r| r.entity())
        .find(|e| e.id() == id as u32)
        .ok_or(format!("no entity with id {id}"))
}

/// A bare id or name (`{main: 12}` / `{main: "battle-cam"}`).
fn resolve_entity(ctx: &Ctx, v: &Value) -> Result<Entity, String> {
    match v {
        Value::String(n) => entity_arg(ctx, &json!({ "name": n })),
        _ => entity_arg(ctx, &json!({ "id": v })),
    }
}

/// Entities named by `v` (id, name or an array of them; a name matches every entity with that
/// name) plus all their descendants.
fn entity_set(ctx: &Ctx, v: &Value) -> Result<Vec<Entity>, String> {
    let items: Vec<&Value> = match v {
        Value::Array(a) => a.iter().collect(),
        _ => vec![v],
    };
    let mut roots = vec![];
    for it in items {
        match it {
            Value::String(n) => {
                let before = roots.len();
                roots.extend(ctx.world.query::<(Entity, &crate::scene::Name)>().iter().filter(|(_, nm)| nm.0 == *n).map(|(e, _)| e));
                if roots.len() == before {
                    return Err(format!("no entity named '{n}'"));
                }
            }
            _ => roots.push(resolve_entity(ctx, it)?),
        }
    }
    let mut out = roots.clone();
    for (e, _) in ctx.world.query::<(Entity, &crate::scene::Parent)>().iter() {
        let mut cur = e;
        for _ in 0..64 {
            let Ok(p) = ctx.world.get::<&crate::scene::Parent>(cur).map(|p| p.0) else { break };
            if roots.contains(&p) {
                out.push(e);
                break;
            }
            cur = p;
        }
    }
    Ok(out)
}

fn entity_summary(ctx: &Ctx, e: Entity) -> Value {
    let w = &ctx.world;
    let mut tags = vec![];
    if w.get::<&MeshRenderer>(e).is_ok() {
        tags.push("mesh");
    }
    if w.get::<&Body>(e).is_ok() {
        tags.push("body");
    }
    if w.get::<&PointLight>(e).is_ok() {
        tags.push("light");
    }
    if w.get::<&Camera>(e).is_ok() {
        tags.push("camera");
    }
    if w.get::<&WorldLabel>(e).is_ok() {
        tags.push("label");
    }
    json!({
        "id": e.id(),
        "name": w.get::<&Name>(e).map(|n| n.0.clone()).unwrap_or_default(),
        "pos": w.get::<&GlobalTransform>(e).map(|g| arr(g.translation())).unwrap_or(Value::Null),
        "parent": w.get::<&Parent>(e).map(|p| json!(p.0.id())).unwrap_or(Value::Null),
        "tags": tags,
    })
}

fn entity_detail(ctx: &Ctx, e: Entity) -> Value {
    let w = &ctx.world;
    let mut v = entity_summary(ctx, e);
    if let Ok(t) = w.get::<&Transform>(e) {
        let (x, y, z) = t.rotation.to_euler(EulerRot::XYZ);
        v["local"] = json!({"pos": arr(t.translation), "rot_deg": arr(Vec3::new(x, y, z) * 57.29578), "scale": arr(t.scale)});
    }
    if let Ok(m) = w.get::<&MeshRenderer>(e) {
        let mesh = ctx.renderer.mesh(m.mesh);
        v["mesh"] = json!({
            "id": m.mesh.0, "name": mesh.map(|m| m.name.clone()), "triangles": mesh.map(|m| m.index_count / 3),
            "material": m.material.0, "material_data": ctx.renderer.material(m.material), "cast_shadows": m.cast_shadows,
        });
    }
    if let Some(rb) = ctx.physics.body(w, e) {
        v["body"] = json!({
            "type": format!("{:?}", rb.body_type()), "linvel": arr(rb.linvel()), "angvel": arr(rb.angvel()),
            "mass": rb.mass(), "sleeping": rb.is_sleeping(),
            "grounded": ctx.physics.grounded(w, e, 60.0),
            "contacts": ctx.physics.contacts(w, e).iter().map(|c| json!({"other": c.other.map(|o| o.id()), "normal": arr(c.normal), "depth": c.depth})).collect::<Vec<_>>(),
        });
    }
    if let Ok(l) = w.get::<&crate::renderer::Lod>(e) {
        v["lod"] = json!(*l);
    }
    if let Ok(l) = w.get::<&PointLight>(e) {
        v["light"] = json!({"color": arr(l.color), "intensity": l.intensity, "range": l.range});
    }
    if let Ok(c) = w.get::<&Camera>(e) {
        v["camera"] = json!({"fov_y_deg": c.fov_y_deg, "near": c.near, "far": c.far});
    }
    if let Ok(l) = w.get::<&WorldLabel>(e) {
        v["label"] = json!({"html": l.html, "offset": arr(l.offset)});
    }
    let children: Vec<u32> = w
        .query::<(Entity, &Parent)>()
        .iter()
        .filter(|(_, p)| p.0 == e)
        .map(|(c, _)| c.id())
        .collect();
    v["children"] = json!(children);
    v
}

pub(crate) fn handle<G: Game>(
    name: &str,
    args: &Value,
    ctx: &mut Ctx,
    game: &mut G,
) -> Result<Value, String> {
    let a = args;
    match name {
        "help" => {
            let mut cmds: Vec<Value> = COMMANDS
                .iter()
                .map(|(n, d)| json!({"name": n, "desc": d}))
                .collect();
            let active: Vec<&str> = game.debug_commands().iter().map(|c| c.0).collect();
            for (n, d, screen) in game.inactive_debug_commands() {
                if !active.contains(&n) && !cmds.iter().any(|c| c["name"] == n && c["screen"] == screen) {
                    cmds.push(json!({"name": n, "desc": d, "game": true, "screen": screen, "active": false}));
                }
            }
            for (n, d) in game.debug_commands() {
                if is_engine_command(n) {
                    // the engine command wins: the game's handler is never reached
                    cmds.push(json!({"name": n, "desc": d, "game": true, "shadowed_by_engine": true}));
                } else {
                    cmds.push(json!({"name": n, "desc": d, "game": true}));
                }
            }
            Ok(json!({"usage": "engine.call(name, args)", "commands": cmds}))
        }
        "stats" => {
            let (meshes, materials, textures) = ctx.renderer.counts();
            Ok(json!({
                "time": ctx.time,
                "render": ctx.renderer.stats,
                "entities": ctx.world.len(),
                "bodies": ctx.physics.body_count(),
                "labels": ctx.ui.label_count(),
                "panels": ctx.panels.len(),
                "jobs": {"threads": crate::jobs::threads(), "backend": if crate::jobs::threads_backend() { "threads" } else { "serial" },
                         "cross_origin_isolated": js_sys::Reflect::get(&js_sys::global(), &"crossOriginIsolated".into()).ok().and_then(|v| v.as_bool())},
                "assets": {"meshes": meshes, "materials": materials, "textures": textures},
                "log_seq": log::next_seq(),
            }))
        }
        "entities" => {
            let filter = a
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_lowercase());
            let with = a.get("with").and_then(|v| v.as_str());
            let limit = a.get("limit").and_then(|v| v.as_u64()).unwrap_or(200) as usize;
            let mut ents: Vec<Entity> = ctx.world.iter().map(|e| e.entity()).collect();
            ents.sort_by_key(|e| e.id());
            let list: Vec<Value> = ents
                .into_iter()
                .map(|e| entity_summary(ctx, e))
                .filter(|v| {
                    filter
                        .as_ref()
                        .map(|f| v["name"].as_str().unwrap_or("").to_lowercase().contains(f))
                        .unwrap_or(true)
                })
                .filter(|v| {
                    with.map(|w| v["tags"].as_array().unwrap().iter().any(|t| t == w))
                        .unwrap_or(true)
                })
                .collect();
            let total = list.len();
            Ok(
                json!({"total": total, "entities": list.into_iter().take(limit).collect::<Vec<_>>()}),
            )
        }
        "entity" => {
            let e = entity_arg(ctx, a)?;
            Ok(entity_detail(ctx, e))
        }
        "set" => {
            let e = entity_arg(ctx, a)?;
            if let Ok(mut t) = ctx.world.get::<&mut Transform>(e) {
                if let Some(p) = a.get("pos").and_then(v3) {
                    t.translation = p;
                }
                if let Some(r) = a.get("rot").and_then(v3) {
                    let r = r * (std::f32::consts::PI / 180.0);
                    t.rotation = Quat::from_euler(EulerRot::XYZ, r.x, r.y, r.z);
                }
                if let Some(s) = a.get("scale").and_then(v3) {
                    t.scale = s;
                }
            }
            let t = ctx.world.get::<&Transform>(e).map(|t| *t).ok();
            if let (Some(t), Some(rb)) = (t, ctx.physics.body_mut(&ctx.world, e)) {
                if a.get("pos").is_some() || a.get("rot").is_some() {
                    rb.set_position(
                        rapier3d::math::Pose::from_parts(t.translation, t.rotation),
                        true,
                    );
                }
                if let Some(v) = a.get("vel").and_then(v3) {
                    rb.set_linvel(v, true);
                }
            }
            if let Some(c) = a
                .get("color")
                .and_then(|v| v.as_str())
                .and_then(parse_color)
            {
                let mid = ctx
                    .world
                    .get::<&MeshRenderer>(e)
                    .map(|m| m.material)
                    .map_err(|_| "entity has no mesh")?;
                if let Some(m) = ctx.renderer.material_mut(mid) {
                    m.base_color = crate::renderer::srgb_hex(c);
                }
            }
            let opacity = a.get("opacity").and_then(|v| v.as_f64());
            let layer = a.get("layer").and_then(|v| v.as_u64());
            if opacity.is_some() || layer.is_some() {
                // glTF models: the root has no mesh, its descendants do
                let mut targets = vec![e];
                targets.extend(crate::scene::descendants_of(&ctx.world, &[e]));
                for t in targets {
                    if let Ok(mut mr) = ctx.world.get::<&mut MeshRenderer>(t) {
                        if let Some(o) = opacity {
                            mr.opacity = o.clamp(0.0, 1.0) as f32;
                        }
                        if let Some(l) = layer {
                            mr.layer = l.min(31) as u8;
                        }
                    }
                }
            }
            crate::scene::propagate_transforms(&mut ctx.world);
            Ok(entity_detail(ctx, e))
        }
        "spawn" => {
            if a.get("camera").and_then(|v| v.as_bool()) == Some(true) {
                let pos = a.get("pos").and_then(v3).unwrap_or(Vec3::new(0.0, 2.0, 6.0));
                let mut t = Transform::from_translation(pos);
                if let Some(target) = a.get("look_at").and_then(v3) {
                    t = t.looking_at(target, Vec3::Y);
                }
                let mut cam = Camera::default();
                if let Some(f) = a.get("fov").and_then(|v| v.as_f64()) {
                    cam.fov_y_deg = f as f32;
                }
                let name = a.get("name").and_then(|v| v.as_str()).unwrap_or("camera").to_string();
                let e = ctx.world.spawn((t, crate::scene::GlobalTransform::default(), cam, crate::scene::Name(name)));
                if a.get("main").and_then(|v| v.as_bool()) == Some(true) {
                    ctx.set_main_camera(e);
                }
                return Ok(json!({"id": e.id(), "main": crate::scene::main_camera(&ctx.world).map(|m| m.id())}));
            }
            if let Some(size) = a.get("water").and_then(|v| v.as_array()) {
                let f = |i: usize| size.get(i).and_then(|v| v.as_f64()).unwrap_or(4.0) as f32;
                let pos = a.get("pos").and_then(v3).unwrap_or(Vec3::ZERO);
                let mut m = crate::renderer::Material::water(0x0a3d5c, 0x2a9d8f);
                if let Some(sp) = a.get("spacing").and_then(|v| v.as_f64()) {
                    m.water_vertex_spacing = sp as f32;
                }
                let mat = ctx.renderer.add_material(m);
                let name = a.get("name").and_then(|v| v.as_str()).unwrap_or("water").to_string();
                let e = ctx.spawn_water(&name, mat, glam::Vec2::new(f(0), f(1)), Transform::from_translation(pos));
                crate::scene::propagate_transforms(&mut ctx.world);
                return Ok(entity_detail(ctx, e));
            }
            if let Some(url) = a.get("gltf").and_then(|v| v.as_str()) {
                let pos = a.get("pos").and_then(v3).unwrap_or(Vec3::ZERO);
                let scale = a.get("size").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                let mut opts = ctx.gltf_defaults;
                if let Some(v) = a.get("auto_lod").and_then(|v| v.as_bool()) {
                    opts.auto_lod = v;
                }
                let e = ctx.load_gltf_with(url, Transform::from_translation(pos).with_scale(Vec3::splat(scale)), opts);
                return Ok(json!({"id": e.id(), "loading": ctx.assets_pending() > 0}));
            }
            let shape = a.get("shape").and_then(|v| v.as_str()).unwrap_or("cube");
            let pos = a
                .get("pos")
                .and_then(v3)
                .unwrap_or(Vec3::new(0.0, 5.0, 0.0));
            let size = a.get("size").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let color = a
                .get("color")
                .and_then(|v| v.as_str())
                .and_then(parse_color)
                .unwrap_or(0xcccccc);
            let name = a
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(shape)
                .to_string();
            let (e, collider) = match shape {
                "sphere" => (
                    ctx.spawn_sphere(&name, pos, size * 0.5, color),
                    ColliderBuilder::ball(size * 0.5),
                ),
                "cube" => (
                    ctx.spawn_cube(&name, pos, Vec3::splat(size), color),
                    ColliderBuilder::cuboid(size * 0.5, size * 0.5, size * 0.5),
                ),
                other => return Err(format!("unknown shape '{other}' (cube|sphere)")),
            };
            match a.get("body").and_then(|v| v.as_str()).unwrap_or("dynamic") {
                "dynamic" => {
                    ctx.physics
                        .add(&mut ctx.world, e, RigidBodyBuilder::dynamic(), collider);
                }
                "fixed" => {
                    ctx.physics
                        .add(&mut ctx.world, e, RigidBodyBuilder::fixed(), collider);
                }
                "kinematic" => {
                    // moved by its Transform (`set {id, pos}`): obstacles like NPCs
                    ctx.physics
                        .add(&mut ctx.world, e, RigidBodyBuilder::kinematic_position_based(), collider);
                }
                _ => {}
            }
            crate::scene::propagate_transforms(&mut ctx.world);
            Ok(json!({"id": e.id()}))
        }
        "despawn" => {
            let e = entity_arg(ctx, a)?;
            ctx.despawn(e);
            Ok(json!({"despawned": e.id()}))
        }
        "pause" => {
            ctx.time.paused = true;
            ctx.time.held = true;
            Ok(json!({"paused": true, "step": ctx.time.step}))
        }
        "resume" => {
            ctx.time.paused = false;
            ctx.time.held = false;
            Ok(json!({"paused": false, "step": ctx.time.step}))
        }
        "step" => {
            let n = a.get("n").and_then(|v| v.as_u64()).unwrap_or(1).min(36_000);
            for _ in 0..n {
                fixed_step(ctx, game);
            }
            crate::scene::propagate_transforms(&mut ctx.world);
            Ok(json!({"step": ctx.time.step, "sim_time": ctx.time.sim_time}))
        }
        "time_scale" => {
            ctx.time.time_scale = a
                .get("value")
                .and_then(|v| v.as_f64())
                .ok_or("missing 'value'")? as f32;
            Ok(json!({"time_scale": ctx.time.time_scale}))
        }
        "seed" => {
            let v = a
                .get("value")
                .and_then(|v| v.as_u64())
                .ok_or("missing 'value'")?;
            ctx.rng = crate::app::Rng(v.max(1));
            ctx.particle_rng = crate::app::Rng((v ^ 0xD1B5_4A32_D192_ED03).max(1));
            Ok(json!({"seed": v}))
        }
        "input" => {
            if let Some(on) = a.get("stick").and_then(|v| v.as_bool()) {
                ctx.input.touch.set_stick_enabled(on);
            }
            if a.get("release_all")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                ctx.input.release_all();
            }
            let down = a.get("down").and_then(|v| v.as_bool()).unwrap_or(true);
            if let Some(k) = a.get("key").and_then(|v| v.as_str()) {
                let key = a
                    .get("key_char")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let shift = a.get("shift").and_then(|v| v.as_bool()).unwrap_or(false);
                ctx.key_event(k, &key, shift, down, "api");
            }
            if let Some(b) = a
                .get("button")
                .and_then(|v| v.as_i64())
                .and_then(|b| crate::input::MouseButton::from_dom(b as i16))
            {
                ctx.button_event(b, down, "api");
            }
            if let Some(t) = a.get("touch") {
                let id = t.get("id").and_then(|v| v.as_i64()).unwrap_or(1) as i32;
                let x = t.get("x").and_then(|v| v.as_f64()).ok_or("touch needs x")? as f32;
                let y = t.get("y").and_then(|v| v.as_f64()).ok_or("touch needs y")? as f32;
                let phase = t.get("phase").and_then(|v| v.as_str()).unwrap_or("start").to_string();
                ctx.touch_event(id, Vec2::new(x, y), &phase, "api");
            }
            if let (Some(x), Some(y)) = (
                a.get("x").and_then(|v| v.as_f64()),
                a.get("y").and_then(|v| v.as_f64()),
            ) {
                ctx.input.pointer = Vec2::new(x as f32, y as f32);
            }
            let dx = a.get("dx").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let dy = a.get("dy").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            ctx.input.pointer_delta += Vec2::new(dx, dy);
            if let Some(act) = a.get("action").and_then(|v| v.as_str()) {
                ctx.ui.push_action(act);
            }
            Ok(json!({"keys": ctx.input.keys, "buttons": ctx.input.buttons, "touch": ctx.input.touch}))
        }
        "logs" => {
            let since = a.get("since").and_then(|v| v.as_u64()).unwrap_or(0);
            let level = match a.get("level").and_then(|v| v.as_str()).unwrap_or("debug") {
                "error" => Level::Error,
                "warn" => Level::Warn,
                "info" => Level::Info,
                _ => Level::Debug,
            };
            let limit = a.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
            let all = log::entries_since(since, level, usize::MAX);
            let omitted = all.len().saturating_sub(limit);
            let entries: Vec<_> = all.into_iter().skip(omitted).collect();
            // `omitted`: matching entries left out by `limit` (ask again with a higher limit);
            // `oldest`: first seq still kept (the buffer holds the last 10000 entries)
            Ok(json!({"next": log::next_seq(), "oldest": log::oldest_seq(), "omitted": omitted, "entries": entries}))
        }
        "render" => {
            // any RenderSettings field by name (unknown names are an error, types are checked)
            if let Some(obj) = a.as_object().filter(|o| !o.is_empty()) {
                let mut cur = json!(ctx.renderer.settings);
                let hex = |v: &Value| -> Option<Value> {
                    let c = v.as_str().and_then(parse_color)?;
                    Some(json!([((c >> 16) & 255) as f32 / 255.0, ((c >> 8) & 255) as f32 / 255.0, (c & 255) as f32 / 255.0]))
                };
                for (k, v) in obj {
                    if cur.get(k).is_none() {
                        return Err(format!("unknown render setting '{k}'; known: {:?}", cur.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>())));
                    }
                    if k == "grade" && v.is_object() {
                        // merged field by field; `preset` first resets to that look; colours may be '#rrggbb'
                        if let Some(name) = v.get("preset").and_then(|p| p.as_str()) {
                            let g = crate::post::Grade::preset(name).ok_or_else(|| format!("unknown grade preset '{name}'; known: {:?}", crate::post::Grade::PRESETS))?;
                            cur["grade"] = json!(g);
                        }
                        for (gk, gv) in v.as_object().unwrap() {
                            if gk == "preset" {
                                continue;
                            }
                            if cur["grade"].get(gk).is_none() {
                                return Err(format!("unknown grade field '{gk}'; known: {:?}", cur["grade"].as_object().map(|o| o.keys().cloned().collect::<Vec<_>>())));
                            }
                            cur["grade"][gk] = hex(gv).unwrap_or_else(|| gv.clone());
                        }
                        continue;
                    }
                    cur[k] = if k == "vignette_color" { hex(v).unwrap_or_else(|| v.clone()) } else { v.clone() };
                }
                ctx.renderer.settings = serde_json::from_value(cur).map_err(|e| format!("render: {e}"))?;
            }
            let mut out = json!(ctx.renderer.settings);
            out["gpu_culling_supported"] = json!(ctx.renderer.gpu_culling_supported);
            out["grade_presets"] = json!(crate::post::Grade::PRESETS);
            Ok(out)
        }
        "camera" => {
            if let Some(id) = a.get("main") {
                let e = resolve_entity(ctx, id)?;
                if ctx.world.get::<&Camera>(e).is_err() || ctx.world.get::<&crate::render_texture::RenderTextureCamera>(e).is_ok() {
                    return Err(format!("entity {} is not a canvas camera (no Camera, or a render-texture camera)", e.id()));
                }
                ctx.set_main_camera(e);
            }
            let cam = crate::scene::main_camera(&ctx.world).ok_or("no camera")?;
            if let Some(m) = a.get("layers").and_then(|v| v.as_u64()) {
                if let Ok(mut c) = ctx.world.get::<&mut Camera>(cam) {
                    c.layers = m as u32;
                }
            }
            if let Ok(mut t) = ctx.world.get::<&mut Transform>(cam) {
                if let Some(p) = a.get("pos").and_then(v3) {
                    t.translation = p;
                }
                if let Some(target) = a.get("look_at").and_then(v3) {
                    *t = t.looking_at(target, Vec3::Y);
                }
            }
            crate::scene::propagate_transforms(&mut ctx.world);
            Ok(entity_detail(ctx, cam))
        }
        "cameras" => {
            let main = crate::scene::main_camera(&ctx.world);
            let list: Vec<Value> = ctx
                .world
                .query::<(Entity, &Camera, Option<&crate::scene::Name>)>()
                .iter()
                .map(|(e, c, n)| {
                    json!({"id": e.id(), "name": n.map(|n| n.0.clone()), "main": Some(e) == main,
                        "marked_main": ctx.world.get::<&crate::scene::MainCamera>(e).is_ok(),
                        "render_texture": ctx.world.get::<&crate::render_texture::RenderTextureCamera>(e).is_ok(),
                        "fov_y_deg": c.fov_y_deg, "layers": c.layers})
                })
                .collect();
            Ok(json!({"main": main.map(|e| e.id()), "cameras": list}))
        }
        "raycast" => {
            let o = a.get("origin").and_then(v3).ok_or("missing 'origin'")?;
            let d = a.get("dir").and_then(v3).ok_or("missing 'dir'")?;
            let max = a.get("max").and_then(|v| v.as_f64()).unwrap_or(1000.0) as f32;
            let exclude = match a.get("exclude") {
                Some(v) => entity_set(ctx, v)?,
                None => vec![],
            };
            Ok(match ctx.physics.raycast_excluding(o, d, max, &exclude) {
                Some(h) => {
                    json!({"hit": true, "id": h.entity.map(|e| e.id()), "distance": h.distance, "point": arr(h.point), "normal": arr(h.normal)})
                }
                None => json!({"hit": false}),
            })
        }
        "pick" => {
            let x = a.get("x").and_then(|v| v.as_f64()).ok_or("missing 'x'")? as f32;
            let y = a.get("y").and_then(|v| v.as_f64()).ok_or("missing 'y'")? as f32;
            let (o, d) = ctx.screen_ray(Vec2::new(x, y)).ok_or("no camera")?;
            if a.get("mesh").and_then(|v| v.as_bool()).unwrap_or(false) {
                // what is drawn (meshes), plus entities whose surface ties in depth (z-fighting)
                return Ok(match crate::zfight::ray_pick(&ctx.world, &ctx.renderer, o, d, 0.002) {
                    Some((e, t, ties)) => json!({"hit": true, "entity": entity_summary(ctx, e), "point": arr(o + d.normalize() * t),
                        "depth_ties": ties.iter().map(|t| entity_summary(ctx, *t)).collect::<Vec<_>>()}),
                    None => json!({"hit": false}),
                });
            }
            Ok(match ctx.physics.raycast(o, d, 1000.0) {
                Some(h) => {
                    json!({"hit": true, "entity": h.entity.map(|e| entity_summary(ctx, e)), "point": arr(h.point)})
                }
                None => json!({"hit": false}),
            })
        }
        "ui" => Ok(json!({"slots": ctx.ui.slots()})),
        "save" => ctx.save.command(a),
        "particles" => ctx.particles_command(a),
        "sky" => ctx.sky_command(a),
        "trace" => {
            if a.get("id").is_some() {
                let e = entity_arg(ctx, a)?;
                let n = a.get("frames").and_then(|v| v.as_u64()).unwrap_or(120) as u32;
                ctx.trace = Some((e, n, Vec::with_capacity(n as usize)));
                return Ok(serde_json::json!({"started": true, "frames": n}));
            }
            let Some((_, left, s)) = &ctx.trace else { return Err("no trace; start with {id, frames}".into()) };
            // speed per frame = distance / frame interval; jitter = mean |speed - mean| / mean
            let speeds: Vec<f32> = s.windows(2).map(|w| {
                let d = ((w[1][2] - w[0][2]).powi(2) + (w[1][3] - w[0][3]).powi(2) + (w[1][4] - w[0][4]).powi(2)).sqrt();
                d / (w[1][0] / 1000.0).max(1e-4)
            }).collect();
            let mean = speeds.iter().sum::<f32>() / speeds.len().max(1) as f32;
            // frame-to-frame speed change (acceleration over seconds does not count as jitter)
            let jitter = if mean > 1e-3 && speeds.len() > 1 { speeds.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / (speeds.len() - 1) as f32 / mean * 100.0 } else { 0.0 };
            let steps: Vec<u32> = s.iter().map(|x| x[1] as u32).collect();
            Ok(serde_json::json!({
                "done": *left == 0, "frames": s.len(), "mean_speed": mean, "speed_jitter_pct": jitter,
                "frames_with_0_steps": steps.iter().filter(|&&n| n == 0).count(),
                "frames_with_2plus_steps": steps.iter().filter(|&&n| n >= 2).count(),
                "samples": if a.get("samples").and_then(|v| v.as_bool()).unwrap_or(false) { serde_json::json!(s) } else { serde_json::json!(null) },
            }))
        }
        "interpolation" => {
            if let Some(v) = a.get("on").and_then(|v| v.as_bool()) {
                ctx.interpolation = v;
            }
            Ok(serde_json::json!({"on": ctx.interpolation}))
        }
        "inspect" => {
            let x = a.get("x").and_then(|v| v.as_f64()).ok_or("missing 'x'")? as f32;
            let y = a.get("y").and_then(|v| v.as_f64()).ok_or("missing 'y'")? as f32;
            Ok(ctx.inspect_at(Vec2::new(x, y), a["copy"].as_bool().unwrap_or(false)))
        }
        "characters" => ctx.characters_command(a),
        "world" => ctx.world_command(a),
        "grass" => {
            if let Some(obj) = a.as_object().filter(|o| !o.is_empty()) {
                let mut cur = json!(ctx.renderer.grass.settings);
                for (k, v) in obj {
                    if cur.get(k).is_none() {
                        return Err(format!("unknown grass setting '{k}'"));
                    }
                    cur[k] = v.clone();
                }
                ctx.renderer.grass.settings = serde_json::from_value(cur).map_err(|e| e.to_string())?;
            }
            Ok(json!({"settings": ctx.renderer.grass.settings, "blades_last_frame": ctx.renderer.grass.last_blades, "has_terrain": ctx.terrain.is_some()}))
        }
        "terrain" => ctx.terrain_command(a),
        "lights" => ctx.lights_command(a),
        "generate" => ctx.generate_command(a),
        "batch" => ctx.batch_command(a),
        "gpu_scene" => Ok(ctx.renderer.gpu_scene_info()),
        "collisions" => ctx.collisions_command(a),
        "overlap" => ctx.overlap_command(a),
        "trigger" => {
            let pos = v3(&a["pos"]).ok_or("pos: [x,y,z] required")?;
            let half = v3(&a["half"]).unwrap_or(Vec3::ONE);
            let e = ctx.spawn_trigger(a["name"].as_str().unwrap_or("trigger"), half, crate::scene::Transform::from_translation(pos));
            Ok(json!({"id": e.id()}))
        }
        "animation" => ctx.animation_command(a),
        "render_texture" => {
            if let Some(id) = a["id"].as_u64() {
                let rt = crate::render_texture::RenderTextureId(id as u32);
                if ctx.renderer.rtts.get(id as usize).is_none() {
                    return Err(format!("no render texture {id}"));
                }
                if a["render"].as_bool() == Some(true) {
                    ctx.refresh_render_texture(rt);
                }
                if a["export"].as_bool() == Some(true) {
                    ctx.export_render_texture(rt);
                }
                return Ok(ctx.renderer.rtts[id as usize].describe(id as usize));
            }
            Ok(json!({"render_textures": ctx.renderer.rtts.iter().enumerate().map(|(i, r)| r.describe(i)).collect::<Vec<_>>()}))
        }
        "scenes" => {
            if let Some(n) = a["unload"].as_str() {
                let removed = ctx.unload_scene(n);
                return Ok(json!({"unloaded": n, "entities": removed, "now": ctx.scenes.describe(&ctx.world)}));
            }
            if let Some(n) = a["begin"].as_str() {
                ctx.begin_scene(n);
            }
            Ok(ctx.scenes.describe(&ctx.world))
        }
        "material" => {
            use crate::renderer::MaterialId;
            let Some(id) = a.get("id").and_then(|v| v.as_u64()) else {
                let list: Vec<Value> = (0..ctx.renderer.material_count() as u32)
                    .filter_map(|i| ctx.renderer.material(MaterialId(i)).map(|m| json!({"id": i, "name": m.name, "roughness": m.roughness, "metallic": m.metallic,
                        "subsurface": m.subsurface, "clearcoat": m.clearcoat, "bend": m.bend})))
                    .collect();
                return Ok(json!({"materials": list}));
            };
            let m = ctx.renderer.material_mut(MaterialId(id as u32)).ok_or("unknown material")?;
            let f = |k: &str| a.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
            if let Some(v) = f("roughness") { m.roughness = v; }
            if let Some(v) = f("metallic") { m.metallic = v; }
            if let Some(v) = f("subsurface") { m.subsurface = v; }
            if let Some(v) = f("clearcoat") { m.clearcoat = v; }
            if let Some(v) = f("clearcoat_roughness") { m.clearcoat_roughness = v; }
            match a.get("subsurface_color") {
                Some(Value::Null) => m.subsurface_color = None,
                Some(v) => {
                    let c = v.as_str().and_then(parse_color).ok_or("subsurface_color: '#rrggbb' or null")?;
                    m.subsurface_color = Some(crate::renderer::srgb_hex(c).truncate());
                }
                None => {}
            }
            Ok(json!(m))
        }
        "bend" => {
            if let Some(list) = a.get("pushers").and_then(|v| v.as_array()) {
                let p: Vec<(Vec3, f32)> = list
                    .iter()
                    .filter_map(|q| {
                        let q = q.as_array()?;
                        let f = |i: usize| q.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                        Some((Vec3::new(f(0), f(1), f(2)), if q.len() > 3 { f(3) } else { 1.0 }))
                    })
                    .collect();
                ctx.renderer.set_pushers(&p);
            }
            if let Some(w) = a.get("wind").and_then(|v| v.as_array()) {
                let f = |i: usize| w.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                ctx.renderer.grass_settings().wind = [f(0), f(1), f(2)];
            }
            if let (Some(m), Some(b)) = (a.get("material").and_then(|v| v.as_u64()), a.get("bend").and_then(|v| v.as_f64())) {
                ctx.renderer.material_mut(crate::renderer::MaterialId(m as u32)).ok_or("unknown material")?.bend = b as f32;
            }
            let mats: Vec<Value> = (0..ctx.renderer.material_count() as u32)
                .filter_map(|i| ctx.renderer.material(crate::renderer::MaterialId(i)).filter(|m| m.bend > 0.0).map(|m| json!({"id": i, "name": m.name, "bend": m.bend})))
                .collect();
            Ok(json!({"pushers": ctx.renderer.pushers().iter().map(|(p, r)| json!([p.x, p.y, p.z, r])).collect::<Vec<_>>(), "max": crate::renderer::MAX_PUSHERS, "wind": ctx.renderer.grass_settings().wind, "materials": mats}))
        }
        "zfight" => {
            let eps = a.get("eps").and_then(|v| v.as_f64()).unwrap_or(0.002) as f32;
            let tol = a.get("tolerance").and_then(|v| v.as_f64()).unwrap_or(0.002) as f32;
            let limit = a.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
            crate::scene::propagate_transforms(&mut ctx.world);
            let css = Vec2::new(ctx.canvas.client_width() as f32, ctx.canvas.client_height() as f32);
            // radius: only instances near the camera (big worlds: a full check takes seconds)
            let center = a.get("center").and_then(v3);
            let radius = a.get("radius").and_then(|v| v.as_f64()).map(|r| r as f32);
            let near = match (center, radius) {
                (Some(c), r) => Some((c, r.unwrap_or(10.0))),
                (None, Some(r)) => crate::scene::main_camera_pose(&ctx.world).map(|m| (m.w_axis.truncate(), r)),
                (None, None) => None,
            };
            let involving = match a.get("entity") {
                Some(v) => Some(entity_set(ctx, v)?),
                None => None,
            };
            let exclude = match a.get("exclude") {
                Some(v) => entity_set(ctx, v)?,
                None => vec![],
            };
            Ok(crate::zfight::find_filtered(&ctx.world, &ctx.renderer, eps, tol, limit, ctx.renderer.last_view_proj, css, near, involving.as_deref(), &exclude))
        }
        "debug_mode" => {
            if let Some(on) = a.get("on").and_then(|v| v.as_bool()) {
                let (now, step, frame) = (log::now_ms(), ctx.time.step, ctx.time.frame);
                ctx.debug_mode.set(on, now, step, frame);
            }
            Ok(ctx.debug_mode.describe())
        }
        "audio" => Ok(ctx.audio.describe()),
        "music" => {
            if let Some(url) = a.get("play").and_then(|v| v.as_str()) {
                let mut o = crate::audio::MusicOpts::default();
                if let Some(v) = a.get("fade").and_then(|v| v.as_f64()) { o.fade = v as f32; }
                if let Some(v) = a.get("volume").and_then(|v| v.as_f64()) { o.volume = v as f32; }
                if let Some(v) = a.get("loop").and_then(|v| v.as_bool()) { o.looped = v; }
                ctx.audio.play_music(url, o).ok_or("music unavailable (no AudioContext?)")?;
            }
            if a.get("stop").and_then(|v| v.as_bool()).unwrap_or(false) {
                let fade = a.get("fade").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                ctx.audio.stop_music(fade);
            }
            if let Some(url) = a.get("prefetch").and_then(|v| v.as_str()) {
                ctx.audio.prefetch_music(url);
            }
            Ok(ctx.audio.describe_music())
        }
        "audio_voice" => {
            let v = crate::audio::VoiceId(a.get("voice").and_then(|v| v.as_u64()).ok_or("missing 'voice'")? as u32);
            let fade = a.get("fade").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            if let Some(vol) = a.get("volume").and_then(|v| v.as_f64()) {
                if !ctx.audio.set_voice_volume(v, vol as f32, fade) {
                    return Err(format!("voice {} is not playing", v.0));
                }
            }
            Ok(json!({"voice": v.0, "volume": ctx.audio.voice_volume(v)}))
        }
        "audio_stop" => {
            match a.get("voice").and_then(|v| v.as_u64()) {
                Some(v) => ctx.audio.fade_out(crate::audio::VoiceId(v as u32), a.get("fade").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32),
                None => ctx.audio.stop_all(),
            }
            Ok(json!({"stopped": a.get("voice").cloned().unwrap_or(json!("all"))}))
        }
        "sfx" => {
            use crate::audio::synth::{levels, Sfx, SfxPreset};
            let Some(name) = a.get("preset").and_then(|v| v.as_str()) else {
                return Ok(
                    json!({"presets": SfxPreset::ALL.iter().map(|p| json!({"name": p.name(), "loop": p.is_loop()})).collect::<Vec<_>>()}),
                );
            };
            let preset = SfxPreset::from_name(name).ok_or(format!(
                "unknown preset '{name}' — call sfx {{}} for the list"
            ))?;
            let seed = a.get("seed").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let mut sfx = Sfx::preset(preset);
            if seed != 0 {
                sfx = sfx.mutate(seed, 1.0);
            }
            if let Some(o) = a.get("overrides").filter(|o| o.is_object()) {
                let mut v = serde_json::to_value(&sfx).map_err(|e| e.to_string())?;
                for (k, val) in o.as_object().unwrap() {
                    v[k] = val.clone();
                }
                sfx = serde_json::from_value(v).map_err(|e| format!("overrides: {e}"))?;
            }
            let reg = a
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(preset.name())
                .to_string();
            let samples = sfx.render();
            let (peak, rms) = levels(&samples);
            ctx.audio
                .add_samples(&reg, &samples, crate::audio::synth::RATE);
            Ok(
                json!({"name": reg, "duration": sfx.duration(), "peak": peak, "rms": rms, "loop": preset.is_loop(), "recipe": sfx}),
            )
        }
        "texgen" => {
            use crate::texgen::{TexGen, TexPreset};
            let Some(name) = a.get("preset").and_then(|v| v.as_str()) else {
                return Ok(
                    json!({"presets": TexPreset::ALL.iter().map(|p| p.name()).collect::<Vec<_>>()}),
                );
            };
            let preset = TexPreset::from_name(name).ok_or(format!(
                "unknown preset '{name}' — call texgen {{}} for the list"
            ))?;
            let mut t = TexGen::new(preset);
            if let Some(v) = a.get("size").and_then(|v| v.as_u64()) {
                t = t.size(v as u32);
            }
            if let Some(v) = a.get("seed").and_then(|v| v.as_u64()) {
                t = t.seed(v as u32);
            }
            if let Some(v) = a.get("scale").and_then(|v| v.as_f64()) {
                t = t.scale(v as f32);
            }
            if let Some(c) = a.get("colors").and_then(|v| v.as_array()) {
                let cols: Vec<u32> = c
                    .iter()
                    .filter_map(|x| x.as_str())
                    .filter_map(parse_color)
                    .collect();
                if cols.len() == 2 {
                    t = t.colors(cols[0], cols[1]);
                }
            }
            let t0 = log::now_ms();
            let px = t.generate();
            let ms = log::now_ms() - t0;
            let id = ctx
                .renderer
                .add_texture_rgba(t.size, t.size, &px, &t.label())?;
            let mut applied = Value::Null;
            if a.get("apply_to").is_some() {
                let e = entity_arg(ctx, &json!({"id": a["apply_to"]}))?;
                let mid = ctx
                    .world
                    .get::<&MeshRenderer>(e)
                    .map(|m| m.material)
                    .map_err(|_| "apply_to entity has no mesh")?;
                // copy the material so other entities sharing it keep theirs
                let mut m = ctx.renderer.material(mid).cloned().unwrap_or_default();
                m.texture = Some(id);
                m.base_color = glam::Vec4::ONE;
                m.checker = 0.0;
                if let Some(strength) = a.get("normal").and_then(|v| v.as_f64()) {
                    let px = t.generate_normal(strength as f32);
                    m.normal_texture = Some(ctx.renderer.add_texture_rgba_linear(t.size, t.size, &px, &format!("{}:normal", t.label()))?);
                    m.normal_scale = 1.0;
                }
                if let Some(p) = a.get("parallax").and_then(|v| v.as_f64()) {
                    let px = t.generate_height();
                    m.height_texture = Some(ctx.renderer.add_texture_rgba_linear(t.size, t.size, &px, &format!("{}:height", t.label()))?);
                    m.parallax = p as f32;
                }
                let new_mid = ctx.renderer.add_material(m);
                if let Ok(mut mr) = ctx.world.get::<&mut MeshRenderer>(e) {
                    mr.material = new_mid;
                }
                applied = json!({"entity": e.id(), "material": new_mid.0});
            }
            Ok(
                json!({"texture": id.0, "label": t.label(), "size": t.size, "generate_ms": ms, "mean_color": crate::texgen::mean_color(&px), "applied": applied, "recipe": t}),
            )
        }
        "audio_play" => {
            let name = a
                .get("sound")
                .and_then(|v| v.as_str())
                .ok_or("missing 'sound'")?;
            let mut p = crate::audio::Play::new();
            if let Some(v) = a.get("volume").and_then(|v| v.as_f64()) {
                p = p.volume(v as f32);
            }
            if let Some(v) = a.get("pitch").and_then(|v| v.as_f64()) {
                p = p.pitch(v as f32);
            }
            if let Some(v) = a.get("pos").and_then(v3) {
                p = p.at(v);
            }
            if a.get("loop").and_then(|v| v.as_bool()).unwrap_or(false) {
                p = p.looped();
            }
            let f = |k: &str| a.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
            let fa = |k: &str, i: usize| {
                a.get(k)
                    .and_then(|v| v.as_array())
                    .and_then(|x| x.get(i))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
            };
            if let Some(v) = f("lowpass") {
                p = p.lowpass(v);
            }
            if let Some(v) = f("highpass") {
                p = p.highpass(v);
            }
            if let Some(hz) = fa("bandpass", 0) {
                p = p.bandpass(hz, fa("bandpass", 1).unwrap_or(1.0));
            }
            if let Some(d) = fa("echo", 0) {
                p.fx.echo = Some((
                    d,
                    fa("echo", 1).unwrap_or(0.4),
                    fa("echo", 2).unwrap_or(0.5),
                ));
            }
            if let Some(s) = f("reverb") {
                p = p.reverb(s);
            }
            if let Some(s) = fa("reverb", 0) {
                p.fx.reverb = Some((s, fa("reverb", 1).unwrap_or(0.4)));
            }
            if let Some(v) = f("distort") {
                p = p.distort(v);
            }
            if let Some(v) = f("pan") {
                p = p.pan(v);
            }
            if let Some(v) = f("detune") {
                p = p.detune(v);
            }
            if let Some(b) = a.get("bus").and_then(|v| v.as_str()) {
                p = p.bus(crate::audio::Bus::from_name(b).ok_or(format!("unknown bus '{b}' (sfx, voice, ambience, ui)"))?);
            }
            let id = ctx
                .audio
                .find(name)
                .ok_or(format!("unknown sound '{name}'"))?;
            Ok(json!({"voice": ctx.audio.play(id, p).map(|v| v.0), "state": ctx.audio.state()}))
        }
        "audio_set" => {
            if let Some(v) = a.get("volume").and_then(|v| v.as_f64()) {
                ctx.audio.set_volume(v as f32);
            }
            if let Some(v) = a.get("music_volume").and_then(|v| v.as_f64()) {
                ctx.audio.set_music_volume(v as f32);
            }
            for b in crate::audio::Bus::ALL {
                if let Some(v) = a.get(format!("{}_volume", b.name())).and_then(|v| v.as_f64()) {
                    ctx.audio.set_bus_volume(b, v as f32);
                }
            }
            if let Some(v) = a.get("pause_on_blur").and_then(|v| v.as_bool()) {
                ctx.audio.set_pause_on_blur(v);
            }
            if let Some(m) = a.get("muted").and_then(|v| v.as_bool()) {
                ctx.audio.set_muted(m);
            }
            Ok(
                json!({"volume": ctx.audio.volume(), "muted": ctx.audio.muted(), "state": ctx.audio.state(), "focused": ctx.audio.focused(), "pause_on_blur": ctx.audio.pause_on_blur(), "music_volume": ctx.audio.music_volume(), "buses": crate::audio::Bus::ALL.iter().map(|b| (b.name().to_string(), json!(ctx.audio.bus_volume(*b)))).collect::<serde_json::Map<_, _>>()}),
            )
        }
        "panels" => {
            let css = Vec2::new(
                ctx.canvas.client_width() as f32,
                ctx.canvas.client_height() as f32,
            );
            Ok(ctx
                .panels
                .describe(&ctx.world, ctx.renderer.last_view_proj, css))
        }
        "panel_click" => {
            let id = crate::panel::PanelId(
                a.get("id")
                    .and_then(|v| v.as_u64())
                    .ok_or("missing 'id' (panel id from 'panels')")? as u32,
            );
            let panel = ctx.panels.get(id).ok_or("no such panel")?;
            let p = if let Some(w) = a.get("widget").and_then(|v| v.as_str()) {
                let w = panel
                    .widget(w)
                    .ok_or(format!("panel has no widget '{w}'"))?;
                Vec2::new(w.rect[0] + w.rect[2] * 0.5, w.rect[1] + w.rect[3] * 0.5)
            } else {
                let x = a
                    .get("x")
                    .and_then(|v| v.as_f64())
                    .ok_or("missing 'widget' or 'x'/'y'")? as f32;
                let y = a.get("y").and_then(|v| v.as_f64()).ok_or("missing 'y'")? as f32;
                Vec2::new(x, y)
            };
            let frame = ctx.time.frame;
            match ctx.panels.click_at(id, p, frame) {
                Some(c) => {
                    ctx.ui.push_action(&c.action);
                    Ok(json!(c))
                }
                None => Ok(Value::Null),
            }
        }
        "assets" => {
            let (meshes, materials, textures) = ctx.renderer.counts();
            Ok(json!({
                "meshes": meshes, "materials": materials, "textures": textures,
                "texture_list": ctx.renderer.texture_infos(),
                "models_ready": ctx.models.ready.values().map(|m| json!({"url": m.url, "nodes": m.nodes.len(), "triangles": m.triangles})).collect::<Vec<_>>(),
                "models_pending": ctx.models.pending.keys().collect::<Vec<_>>(),
                "models_failed": ctx.models.failed,
            }))
        }
        "game_state" => Ok(game.debug_state(ctx)),
        other => match game.debug_command(ctx, other, a) {
            Some(r) => r,
            None => match game.inactive_debug_commands().iter().find(|c| c.0 == other) {
                Some((_, _, screen)) => Err(format!("command '{other}' belongs to screen '{screen}', which is not on top (screens {{switch: \"{screen}\"}})")),
                None => Err(format!("unknown command '{other}' — try engine.call('help')")),
            },
        },
    }
}

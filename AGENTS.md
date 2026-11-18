# AGENTS.md — how an AI agent works on this repo

This engine is built to be developed *by* agents. You never need to guess what the game is
doing: every piece of runtime state is queryable as JSON through `window.engine.call(...)`,
and the Chrome debug bridge exposes that to the shell and to MCP.

**Detailed, task-specific instructions live in [`.claude/`](.claude/README.md)** (skills under
`.claude/skills/*/SKILL.md`, a verifier subagent under `.claude/agents/`). Read its README first.

## The loop

```bash
node tools/dev.mjs watch            # build + serve http://localhost:8080 + rebuild/reload on change (keep running)
node tools/bridge.mjs launch        # Chrome with CDP on :9222, opens the game (separate profile)
node tools/bridge.mjs reload        # after a rebuild: reload, wait for engine-ready, prints warnings/errors
node tools/bridge.mjs call help     # list every engine + game command
node tools/bridge.mjs screenshot out.png
node tools/bridge.mjs check          # build check: reload, assets, z-fighting, engine errors (exit 1 = fail)
node tools/dev.mjs build --check     # build + check in one step (watch --check: after every rebuild)
node tools/bridge.mjs run games/demo/tests/smoke.txt   # scripted test run (input, step, expect, screenshot)
node tools/new-game.mjs ../my-game                     # new standalone game repo (see HANDOFF.md)
node tools/dev.mjs watch --game-dir ../my-game         # build/serve a game from another folder
```

Or use the MCP server (`.mcp.json` registers `wasmonly`): tools `launch_chrome`, `status`,
`engine_call`, `screenshot`, `reload`, `eval_js`.

`dev.mjs build` ends with one JSON line: `{"build":"ok"|"error", ...}`. Compiler errors are
normal `cargo` output above it.

## Verify behaviour, don't eyeball it

- **Deterministic stepping:** `call pause`, then `call step '{"n":120}'` runs exactly 120 fixed
  steps (1/60 s) synchronously. Assert with `call game_state` / `call entity '{"id":N}'`.
- **Input injection:** `call input '{"key":"KeyW","down":true}'` takes the same code path as a
  real keypress. Buttons: `{"button":0}`; mouse motion: `{"dx":10,"dy":0}`; UI buttons: `{"action":"reset"}`.
- **Logs:** `call logs '{"since":<seq>,"level":"warn"}'`. Every entry has `seq`, `step`, `t`, `level`.
  Use `stats.log_seq` before an action and `since` after it to see only what the action caused.
- **Debug mode / session logs:** the user presses ` (German: ^) to record keys, mouse buttons and the
  camera pose (1/s) into `logs/debug-*.log` (JSON lines: `t`, `step`, `frame`, `kind`, ...). Read the newest
  file to see exactly what the user did. Toggle from the API with `call debug_mode '{"on":true}'`.
  In debug mode **Ctrl+click** on an object copies its identity (name#id, scene, local/absolute position, mesh,
  material, parent chain, `engine.call("entity", {id})`) to the clipboard and logs a `"kind":"pick"` line, so the
  user can paste exactly which object they mean. Same data from the API: `call inspect '{"x":640,"y":360}'`.
- **Errors:** Rust panics, WebGPU validation errors and WGSL compile errors (with line numbers)
  all land in the log at level `error` and in a red overlay (`#engine-error`). Startup failures
  set `window.engine.fatal`.
- **Z-fighting / flicker:** a screenshot cannot show flicker. Three layers:
  1. debug builds run the z-fighting test automatically after scene changes and log a warning
     (shows up in `bridge reload` / `check`);
  2. `call zfight` — geometric test, names both entities, world point, screen position; must be `"pairs": 0`;
  3. `node tools/bridge.mjs flicker --out heat.png` — screenshots with flipped draw order and shifted depth
     mapping (pixels stay put), returns flickering regions + entity + `depth_ties` partner; catches what the
     geometric test skips (water, displaced meshes). `check --flicker` includes it.
  Fix by offsetting one surface (height, inset) instead of relying on depth bias.
- **Scripted test runs:** `bridge run <file>` or `run --commands "key Space; step 30; expect game_state.state.jumps == 1"`
  executes a list (reload, pause, step N, frames N, wait ms, key/down/up/hold, mouse, click, action, screen,
  call, wait_for, expect, screenshot) and returns a pass/fail report per step plus new engine warnings; exit 1
  on failure. MCP: `run_script`. Syntax in the header of `tools/script.mjs`; example `games/demo/tests/smoke.txt`.
  Test scripts live in `games/<g>/tests/` (the watcher ignores that folder, so screenshots don't trigger reloads).
- **Screens / scenes:** `call screens` = stack, history, scenes; `screens {switch|push|reset: name}`, `{pop:true}`.
  `call scenes` = entity count per scene, `{unload: name}`. `game_state` of a `ScreenStack` game is
  `{screen, stack, state}` (the top screen's `debug_state` is in `state`).
- **Save games:** `call save` lists slots (localStorage) + IndexedDB blob count; `{slot}` reads one (with
  version + time), `{op:"set", slot, data}`, `{op:"delete", slot}`, `{op:"blobs"}`.
- **Animation:** `call animation` lists animators; `{id}` = clips, playing clip + time, crossfade, deformers
  (skinned, joints, morph weights); `{id, play, fade?, speed?, looped?}`, `{id, morph, weight}`.
- **Particles:** `call particles` = emitters + live counts; `{preset, pos, burst?}` spawns one.
- **Render textures:** `call render_texture` = size, camera, renders, export url; `{id, render:true}`, `{id, export:true}`.
- **Open world:** `call world` = floating origin (`origin`, `shifts`), streaming layers (loaded cells, loads/unloads,
  build ms), entity + mesh counts; `{shift:[x,y,z]}`, `{origin_shift_distance}`, `{layer, radius, enabled}`.
  `call terrain` = size, chunks loaded/built/unloaded, height range; `{height_at:[x,z]}` (absolute), `{flatten|raise|paint}`.
  `call grass` (GPU grass settings, blade count), `call lights` (clustered lights; `{grid:{n}}` stress test),
  `call generate` (registered generators; `{name, params, seed, pos}` spawns one live), `call batch` (static batches,
  moving instances). Debug views: `render {debug_view:3}` = shadow cascades, `4` = lights per cluster.
  In big worlds use `zfight {radius: 150}` — a full check over ~1M triangles takes seconds.
- **Characters:** `call characters` = every character controller (pos, grounded, velocity, jumps, held jump,
  wish, blocked time, touching entities); `{id, pos}` places one. "stuck"/"jump blocked" warnings in the log
  name the blocking entity. Low edges and seams between boxes (where rapier's autostep does not trigger) are
  handled by a step-up retry (lift by `step_height`, move, drop); `step_ups` counts it. Step-up lab in the demo:
  `step_lab {height, radius, half_height, step_height, speed, lanes, start_spread}` + `step` + `characters`.
- **LOD:** glTF primitives with ≥ 256 triangles get vertex-clustered LODs (first switch at 40x their radius).
  Per load: `ctx.load_gltf_with(url, t, GltfOpts { auto_lod: false, .. })` or `lod_distance`; for all loads:
  `ctx.gltf_defaults`; globally: `render {lod_bias: 2}` (far cameras). `stats.render.lod_levels` shows the mix.
- **Input in tests:** keys/buttons held through the API (`input {key, down:true}`) survive window blur; only
  `down:false` or `release_all` releases them. Two agents at once: give each its own Chrome
  (`bridge launch --port 9223 --url http://localhost:8091/`, then `--port/--url` or `CDP_PORT`/`GAME_URL` on
  every command); `run`/`perf`/`check`/`flicker` bring their tab to the front (hidden tabs get no frames).
- **Smooth motion:** dynamic bodies and character controllers are drawn interpolated between their last two
  fixed-step poses (`crate::interp`, `call interpolation {on}`, opt out per entity with `NoInterpolation`). Check
  smoothness with `call trace '{"id":N,"frames":300}'` then `call trace` -> `speed_jitter_pct` (a few % = smooth),
  frames with 0 / 2+ steps. A high fps number can hide hitches: look at `perf` `frame_ms.max` / `missed_frames_pct`.
- **Resolution:** `render {render_scale}` (0.25..1), `auto_resolution` (default on) adapts it to
  `frame_budget_ms` of whole-frame GPU time (`gpu_ms.frame`); every change reallocates the targets (a 20-35 ms
  hitch at 4K), so it jumps once to a scale that fits and changes at most every 4 s; `stats.render` has `width/height` (scene) and `canvas`.
- **Render settings:** `call render` lists every setting; set any by name (sky, fog_*, bloom*, exposure,
  depth_darkening, vignette, saturation, particles, shadows, ...). Unknown names are an error.
- **Visual checks:** `render '{"debug_view":1}'` = normals, `2` = shadow term, `0` = lit.
- **3D UI:** `call panels` gives every world-panel widget with its on-screen centre (`screen`, CSS px);
  click it for real with `input {x,y}` + `input {button:0}`, or directly with `panel_click {id, widget}`.
- **Sky / day-night:** `call sky` = mode, `time_of_day`, sun/moon elevation, `daylight`, `night`, which body lights
  the scene, ambient; `sky {time: 18.5}` (turns the cycle on), `{cycle, day_length (min per day, 0 = still), mode:
  "atmosphere"|"gradient", clouds, stars, moon, skybox: url|null, skybox_blend, skybox_night_blend}`. The clock follows
  the simulation (`pause` + `step` are deterministic). Everything else via `render` (`sun_azimuth_deg`, ...).
  Moving sun: shadows follow the light in `shadow_step_deg` steps (default 0.5) with a `shadow_fade_ms` cross-fade
  (default 400; a second shadow-map set is allocated on first use), so the texel grid does not re-snap every frame
  (shimmering edges). `shadow_step_deg: 0` = follow every frame.
  Proof per frame: `stats.render.shadow_hash` (cascade matrices) stays constant between `shadow_steps`;
  `shadow_fade` > 0 while cross-fading.
- **Opacity:** `MeshRenderer.opacity` (0..1) fades one entity without material copies; it moves to the blend pass
  and keeps its full shadow (0 = invisible, still casts). `set {id|name, opacity}` (applies to children, e.g. glTF).
  `Material.blend_casts_shadow` lets blended materials cast. Test: `games/demo/tests/opacity.txt`. Entity commands
  accept `{name}` instead of `{id}`.
  Lens flare: `render {lens_flare: 0..}` (0 = off); test `games/demo/tests/flare.txt` (visible / hidden behind the tower).
- **Water:** `Material::water` + `ctx.spawn_water(name, mat, size, t)` / `ctx.spawn_ocean(mat, y)`; tune `wave_height`,
  `wave_scale`, `water_chop`, `water_clarity`, `water_foam`, `water_vertex_spacing`; `render {water_ssr, water_refraction,
  water_caustics}`. `render {debug_view:1}` shows the water normals. Test: `games/demo/tests/sky.txt`.
- **Music:** `call music` -> tracks with `state` (buffering/playing/paused/blocked/error), `time`, `duration`,
  `buffered_pct`; transitions are logged. Simulate a slow network for one path only:
  `curl -X POST "http://localhost:8080/__throttle?kbps=48&match=assets/music"` (`kbps=0` = off).
- **Audio:** `call audio` -> context state, sounds, live voices (with fx), last 32 plays with sim `step`.
- **Procedural assets:** `call texgen {preset, apply_to}` and `call sfx {preset, overrides}` iterate on
  textures/sounds without a rebuild (see `.claude/skills/procedural-assets`).
- **Touch:** `call input '{"touch":{"id":1,"x":200,"y":600,"phase":"start"}}'` (then `move`/`end`) uses the same
  path as real fingers; the result's `touch` shows fingers, joystick `axis`, `look_delta`, `pinch`, `taps`,
  `swipes`. Real DOM touch can be simulated with `new PointerEvent('pointerdown', {pointerType:'touch', ...})`.
- **Debug Chrome is muted** (`--mute-audio`): audio state/API checks work, nothing is audible. Use
  `launch --sound` only when the user wants to listen.
- **Culling/LOD:** `stats.render` has `gpu_culling`, `gpu_frustum_culled`, `gpu_occluded`, `gpu_visible`
  (async readback, 1-3 frames old) and `lod_levels`. `render {gpu_culling:false}` switches to the CPU path.
- **Frame rate / stutter:** `node tools/bridge.mjs perf [--ms 3000] [--profile]` measures real rAF intervals and
  returns `frame_ms` p50/p95/p99, `missed_frames_pct`, `cpu_phases` (game_fixed, physics, update, engine, render),
  `gpu_ms` per pass (cull, shadow, opaque, blend, hiz, post; WebGPU timestamp queries) and a `verdict`
  (smooth / gpu-bound / cpu-bound / stutter). `--profile [--top 40]` adds a CDP CPU profile (`top_self` functions).
  Live values: `stats.time.phases`, `stats.render.gpu_ms`. When the *user* sees low fps, ask them to press ` while
  it happens: debug mode writes a `"kind":"perf"` line per second (p50/p95/max, slow frames, phases, GPU) into the log.
- **GPU time, the honest number:** `gpu_ms.frame` = first pass begin to last pass end. It is larger than `total`
  (the sum of the passes) because clears, MSAA resolves and copies run *between* the timed passes;
  `gpu_ms.timeline` has begin/end per pass (cull, shadow, opaque, blend, hiz, post), so the gaps show them.
  Integrated GPUs lower their clock when a frame is light: at 60 fps the milliseconds barely move when you remove
  work. For A/B comparisons make the GPU the bottleneck first (fixed `render_scale`, then enlarge the canvas:
  `bridge eval "document.querySelector('canvas').style.width='150vw'"` + the same for `height`/`150vh`; a reload
  undoes it) and compare `gpu_ms.frame`.
- **Hi-Z occlusion culling** only runs when the opaque candidates add up to `render {occlusion_min_triangles}`
  (default 100000): below that the depth pyramid costs more than it saves (`gpu_occluded` stays 0, `gpu_ms.hiz` 0).
  It also switches itself off for 5 s when it culls less than `occlusion_min_ratio` (default 0.08) of the candidates
  (top-down views hide little), then probes again; `stats.render.hiz_active` shows whether it ran.
- **Shadow cache:** a cascade is re-rendered only when its matrix or its casters (mesh + model matrix) changed.
  Cascades 1.. are fitted `shadow_margin` (0.08) larger and reused while the camera stays inside that margin; when
  only casters moved they refresh every `shadow_refresh` frames (`[2,3,4]`; cascade 0 always at once). A still
  scene renders no shadow pass at all: `stats.render.shadow_passes_rendered` / `shadow_passes_cached`.
  `render {shadow_cache:false}` = old behaviour (A/B, debugging); render textures invalidate it.
  Test: `games/demo/tests/shadowcache.txt`.
- **Blend pass skip:** with no transparent draws, particles or water the opaque pass resolves straight into the HDR
  target (saves a load + store of the 4x MSAA target).
- **Instance upload:** every drawn or shadow-casting entity owns a stable slot in the instance buffer (`InstancePool`
  in `renderer.rs`; a faded entity has a second slot for its shadow copy). A record is uploaded only when it changed;
  draws go through 4-byte id lists (`visible_ids`: opaque written by `cull.wgsl` on the GPU path, blended and
  shadow casters by the CPU), so a changing visible set no longer shifts the records. Unused slots are freed after a
  few frames. `stats.render.instance_upload_bytes` (records + ids), `instance_slots`. Brickhollow walking: ~500 KB
  -> ~10 KB per frame, uncapped p95 17 -> 11 ms. The audio listener is only updated for positional voices and a
  moved camera. Pacing with `onSubmittedWorkDone` was tried and is much worse on the main thread (the promise
  resolves late).
- **Benchmarking against other engines:** count frames engine-neutrally (hook on `getCurrentTexture`), run Chrome
  uncapped (`--disable-gpu-vsync --disable-frame-rate-limit`), and make sure the GPU is quiet first: the Windows
  counter `\GPU Engine(*engtype_3D)\Utilization Percentage` per process shows other browsers or an idle debug
  Chrome still rendering the demo (on an integrated GPU they halve the numbers).

## Code map

| Path | What |
|---|---|
| `crates/engine/src/app.rs` | `Game` trait, `Ctx`, frame loop (fixed step + rAF), input listeners |
| `crates/engine/src/renderer.rs` | WebGPU forward PBR into an HDR target, shadows, MSAA, instancing, mips, LOD pick, CPU/GPU culling switch, views (canvas + render textures), layers |
| `crates/engine/src/post.rs` | HDR targets, bloom (dual-filter mips), depth darkening, ACES composite; sky + particle pipelines |
| `crates/engine/src/render_texture.rs` | render to texture (portraits, mirrors), PNG export to a blob URL |
| `crates/engine/src/particles.rs` | CPU-simulated particle systems (fixed step, presets), instanced billboards |
| `crates/engine/src/animation.rs` | glTF clips (step/linear/cubic), `Animator` crossfades, CPU skinning + morph targets (`Deformer`) |
| `crates/engine/src/character.rs` | `CharacterController` (rapier KCC: slopes, steps, snap, coyote/jump buffer) + `ThirdPersonCamera` |
| `crates/engine/src/screens.rs` | `ScreenStack`/`Screen`/`Trans` (title/menu/play/pause), scenes = entity groups loaded/unloaded together |
| `crates/engine/src/save.rs` | save slots (localStorage, versioned JSON) + blobs (IndexedDB) |
| `crates/engine/src/geom.rs` | geometry kit for procedural content (`MeshBuilder`: parts, walls with openings, extrude, sweep, lathe) + generator registry |
| `crates/engine/src/batch.rs` | static batching (`batch_static`) + `Motion` (spin/sway/bob instance animation) |
| `crates/engine/src/sky.rs` | atmosphere model (CPU twin of `shaders/sky_lut.wgsl`), day/night cycle, `SkyState`, skybox loading, `sky` command |
| `crates/engine/src/water.rs` | `spawn_water`, `spawn_ocean` (camera-following radial mesh), `Ocean` |
| `crates/engine/src/lights.rs` | clustered lighting (16x9x24 froxels, point + `SpotLight`, up to 2048) |
| `crates/engine/src/terrain.rs` | heightmap terrain: chunk LOD + skirts, splat material, heightfield colliders, streaming, edits, scatter rules |
| `crates/engine/src/vegetation.rs` | GPU grass (vertex-shader placed blades) |
| `crates/engine/src/streaming.rs` | streaming layers (cells = scenes) + floating origin (`ctx.origin`, `shift_origin`) |
| `crates/engine/src/gpu_cull.rs` | compute culling (frustum + Hi-Z occlusion), indirect args, counter readback |
| `crates/engine/src/panel.rs` | world panels: Canvas2D widgets -> texture on a quad, ray/UV hit test |
| `crates/engine/src/audio.rs` | Web Audio, 3D panner voices, per-voice effect chain (filters, echo, reverb, distortion, pan, detune) |
| `crates/engine/src/audio/music.rs` | streamed music (`<audio>` + range requests), crossfade, prefetch, music bus |
| `crates/engine/src/audio/synth.rs` | deterministic sfx synthesizer + 16 presets |
| `crates/engine/src/texgen.rs` | procedural seamless textures (16 presets, `noise` crate) |
| `crates/engine/src/jobs.rs` | job system: serial on stable, rayon Web Workers with the `threads` feature |
| `crates/engine/src/touch.rs` | multi-touch gestures: joystick, look, pinch, tap, hold, swipe |
| `crates/engine/src/mesh.rs` | primitives, `simplify_cluster` (auto LOD) |
| `crates/engine/src/shaders/*.wgsl` | `common.wgsl` (layouts, sky, fog) is prepended to pbr/shadow/sky/particles; `cull`, `hiz`, `mipgen`, `post` are standalone |
| `crates/engine/src/debug.rs` | the AI API (`COMMANDS` table = `help` output) |
| `crates/engine/src/debugmode.rs` | ` debug mode: input + camera log -> dev server `/__debuglog` -> `logs/` |
| `crates/engine/src/physics.rs` | rapier3d wrapper, `Body` component, raycasts |
| `crates/engine/src/gltf_loader.rs` | `.glb` loading via fetch + createImageBitmap, per-URL cache |
| `crates/engine/src/ui.rs` | DOM HUD slots, `data-action` clicks, `WorldLabel` positioning |
| `games/demo/` | example game; `assets/` is copied to `dist/assets/` |
| `tools/asset-cook/` | native russimp/Assimp cooker: OBJ/FBX/DAE/... -> `.glb` |
| `tools/bridge.mjs` | CDP bridge CLI + MCP server |
| `tools/script.mjs` | scripted test runs (`bridge run`, MCP `run_script`) |
| `tools/qa.mjs` | build check, flicker scan, perf capture |
| `tools/new-game.mjs` | scaffold for a standalone game repository |
| `HANDOFF.md` | guide for agents building a game on the engine elsewhere |

## Rules

- Target is `wasm32-unknown-unknown` only. Do not add native code paths, WebGL, or other browsers.
- Toolchains are pinned: stable `rust-toolchain.toml` (1.99.0) for everything; the threads backend
  alone uses the nightly in `Cargo.toml` `[workspace.metadata.wasmonly]`. The engine core must keep
  building on stable: put parallel code behind `engine::jobs`, never `#![feature]` or rayon directly.
- Bump a pinned toolchain only on purpose, together with a full build of both variants.
- One way per task: WGSL only, `.glb` only at runtime (convert others with `asset-cook`), hecs, rapier.
- Adding an engine command: add it to `COMMANDS` and the `match` in `debug.rs`.
  Game-specific commands go in `Game::debug_command` + `debug_commands`.
- `wasm-bindgen` in `Cargo.toml` is pinned to the installed CLI version (`wasm-bindgen --version`).
- Input: `input.key` = held; `key_pressed` = edge — in `update` per frame, in `fixed_update` latched since the last
  fixed step (short taps and taps while paused are never lost). Jump/fire-style actions: `key_down_or_pressed`.
  Every new action gets a scripted test with a short tap (`key Space` + `step N` while paused).

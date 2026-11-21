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
node tools/dev.mjs build --release    # optimised build; wasm-opt -O3 if found (PATH or WASM_OPT), else skipped
node tools/dev.mjs build --release --install-wasm-opt   # first cargo-installs the pinned wasm-opt (ask the user first)
node tools/bridge.mjs run games/demo/tests/smoke.txt   # scripted test run (input, step, expect, screenshot)
node tools/new-game.mjs ../my-game                     # new standalone game repo (see HANDOFF.md)
node tools/dev.mjs watch --game-dir ../my-game         # build/serve a game from another folder
```

Or use the MCP server (`.mcp.json` registers `wasmonly`): tools `launch_chrome`, `status`,
`engine_call`, `screenshot`, `reload`, `eval_js`.

`dev.mjs build` ends with one JSON line: `{"build":"ok"|"error", ...}`. Compiler errors are
normal `cargo` output above it.

wasm-opt: the version is pinned in `Cargo.toml` `[workspace.metadata.wasmonly] wasm-opt` (the `wasm-opt` crate
builds binaryen from C++ source: MSVC/clang + CMake, a few minutes, no 111 MB binaryen archive); `cargo install`
puts the CLI in `~/.cargo/bin`. A failing wasm-opt fails the build (`"stage":"wasm-opt"`).

## Verify behaviour, don't eyeball it

- **Deterministic stepping:** `call pause`, then `call step '{"n":120}'` runs exactly 120 fixed
  steps (1/60 s) synchronously. Assert with `call game_state` / `call entity '{"id":N}'`.
- **Input injection:** `call input '{"key":"KeyW","down":true}'` takes the same code path as a
  real keypress. Buttons: `{"button":0}`; mouse motion: `{"dx":10,"dy":0}`; UI buttons: `{"action":"reset"}`.
- **Logs:** `call logs '{"since":<seq>,"level":"warn"}'`. Every entry has `seq`, `step`, `t`, `level`. The ring
  buffer keeps the last 10000 entries; `limit` (default 100) returns the newest N matches, `omitted` counts the rest
  (ask again with a higher `limit`), `oldest` is the first seq still kept.
  Use `stats.log_seq` before an action and `since` after it to see only what the action caused.
- **Debug mode / session logs:** the user presses ` (German: ^) to record keys, mouse buttons and the
  camera pose (1/s) into `logs/debug-*.log` (JSON lines: `t`, `step`, `frame`, `kind`, ...). Read the newest
  file to see exactly what the user did. Toggle from the API with `call debug_mode '{"on":true}'`.
  In debug mode **Ctrl+click** on an object copies its identity (name#id, scene, local/absolute position, mesh,
  material, parent chain, `engine.call("entity", {id})`) to the clipboard and logs a `"kind":"pick"` line, so the
  user can paste exactly which object they mean. Same data from the API: `call inspect '{"x":640,"y":360}'`.
- **Errors:** Rust panics, WebGPU validation errors and WGSL compile errors (with line numbers)
  all land in the log at level `error` and in a red overlay (`#engine-error`). Startup failures
  set `window.engine.fatal`. A render pipeline the *driver* cannot build (mobile Vulkan: `VK_ERROR_UNKNOWN`)
  is caught by error scopes (`debug::create_pipeline_guarded`): logged, shown in the overlay, and its draws are
  skipped so the rest of the frame still renders. To find the construct a device chokes on:
  `node tools/gpu-probe.mjs out.html`, open the page on the device: it compiles PBR/grass variants with parts
  stubbed out and reports each (page + POST `/__log`).
- **Z-fighting / flicker:** a screenshot cannot show flicker. Three layers:
  1. debug builds run the z-fighting test automatically after scene changes and log a warning
     (shows up in `bridge reload` / `check`);
  2. `call zfight` — geometric test, names both entities, world point, screen position; must be `"pairs": 0`;
     `{center:[x,y,z], radius}` checks around a point (top-down cameras), `{entity: id|name|[..]}` only pairs
     involving those (a name matches every entity with it, children included), `{exclude}` leaves some out;
     render layers count: a pair is reported only when one camera (canvas or render texture) draws both layers,
     entities on layers no camera draws are skipped (`skipped["layer not drawn"]`); `set {id|name, layer}`;
  3. `node tools/bridge.mjs flicker --out heat.png` — screenshots with flipped draw order and shifted depth
     mapping (pixels stay put), returns flickering regions + entity + `depth_ties` partner; catches what the
     geometric test skips (water, displaced meshes). `check --flicker` includes it.
  Fix by offsetting one surface (height, inset) instead of relying on depth bias.
- **Scripted test runs:** `bridge run <file>` or `run --commands "key Space; step 30; expect game_state.state.jumps == 1"`
  executes a list (reload, pause, step N, frames N, wait ms, key/down/up/hold, mouse, click, action, screen,
  call, wait_for, expect, screenshot) and returns a pass/fail report per step plus new engine warnings; exit 1
  on failure. MCP: `run_script`. Syntax in the header of `tools/script.mjs`; example `games/demo/tests/smoke.txt`.
  Test scripts live in `games/<g>/tests/` (the watcher ignores that folder, so screenshots don't trigger reloads).
  `expect`/`wait_for` paths may carry command args with spaces: `expect zfight{"center": [0,0,0], "radius": 5}.pairs == 0`.
  Phone layouts: `viewport 390 844 3 touch` in a script emulates the page size, DPR and touch (CDP device metrics;
  `viewport off`, ends with the run). `#` comments: at the line start or ` # ` after a command (`#id` stays text).
- **Screens / scenes:** `call screens` = stack, history, scenes; `screens {switch|push|reset: name}`, `{pop:true}`.
  `call scenes` = entity count per scene, `{unload: name}`. `game_state` of a `ScreenStack` game is
  `{screen, stack, state}` (the top screen's `debug_state` is in `state`).
- **Save games:** `call save` lists slots (localStorage) + IndexedDB blob count; `{slot}` reads one (with
  version + time), `{op:"set", slot, data}`, `{op:"delete", slot}`, `{op:"blobs"}`.
- **Animation:** `call animation` lists animators; `{id}` = clips, playing clip + time, crossfade, deformers
  (skinned, joints, morph weights); `{id, play, fade?, speed?, looped?}`, `{id, morph, weight}`.
- **Particles:** `call particles` = emitters + live counts; `{preset, pos, burst?, layer?, seed?}` spawns one.
  Seeds come from `ctx.particle_rng`, never `ctx.rng`, so adding an emitter does not shift game rolls;
  `ctx.spawn_particles_seeded(.., seed)` fixes one. `ParticleEmitter::layer` = render layer (like
  `MeshRenderer::layer`: a battle camera without that bit does not draw it). `Splash` is a one-shot burst:
  `..preset(Splash)` with a `rate` also needs `one_shot: false`, or it despawns once the first drops die.
  Emitters on a layer no camera draws (canvas or render texture) are not simulated: they wait where they are. An
  overlay screen that pauses the screens below (`pauses_below`) still simulates its own scene's emitters; the AI
  API `pause` freezes those too (`step` steps everything). API `particles {preset, .., name}`, list filter `{name}`.
- **Bending vegetation:** `Material::bend` (0 = off; ~0.5..1.5) makes instances lean away from *pushers* in the vertex
  shader, growing with height above the instance origin (the mesh origin must be at its foot); GPU grass flattens
  around them too. Per frame: `ctx.renderer.set_pushers(&[(player_pos, 0.8), ..])` (max 16, world position + radius).
  Bend materials also sway in the wind (gusts phased by position, like the GPU grass) with the same wind as the grass:
  `renderer.grass_settings().wind = [dir_x, dir_z, strength]` (strength 0 = still; also when grass is off).
  No per-frame upload; shadows and culling bounds stay unbent; don't `batch_static` bendable props (the batch has one
  origin). API: `bend {pushers: [[x,y,z,r],..], material, bend, wind}`. Test: `games/demo/tests/vegetation.txt`.
- **Render textures:** `call render_texture` = size, camera, renders, export url; `{id, render:true}`, `{id, export:true}`.
- **Several cameras:** the canvas renders from the camera with `MainCamera` (`ctx.set_main_camera(e)`), else the first
  `Camera`; audio listener, streaming, terrain, ocean and debug tools follow it (`scene::main_camera(world)`). `call
  cameras` lists them; `camera {main: id|name}` switches, `{layers: mask}` sets the mask; `spawn {camera: true, pos,
  look_at, main}`. `stats.render.triangles` (drawn) and `candidate_triangles` count only the main camera's layers.
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
  handled by a step-up retry (lift by `step_height`, move, drop straight down with a shape cast, so on stairs shallower
  than the capsule is wide it may rest on the next edge instead of sliding back); `step_ups` counts it. A landing more
  than `step_height` above the feet is refused. Step-up lab in the demo: `step_lab {height, steps, depth, radius,
  half_height, step_height, speed, lanes, start_spread}` + `step` + `characters` (stairs: `steps: 4, depth: 0.38`).
- **Collision detection:** after every fixed step the touching collider pairs (solid contacts, sensor overlaps,
  character-controller touches from a probe a few cm around the capsule) are diffed against the last step:
  `ctx.physics.collision_events()` (this frame; `CollisionEvent {a, b, started, sensor, step}`), `entered(e)` /
  `exited(e)` / `inside(trigger)` / `touching(e)` / `is_touching(a, b)`. No collider flags needed. Trigger zones:
  `ctx.spawn_trigger(name, half, t)` or `physics.add_trigger(world, e, collider)` (sensor on a kinematic body that
  follows the Transform; reports dynamic and kinematic bodies incl. characters, not static ground). Queries:
  `overlap_sphere`, `overlap_box`, `sphere_cast`, `raycast`. Kinematic obstacles (NPCs: `physics.add(world, npc,
  RigidBodyBuilder::kinematic_position_based(), capsule)`, root entity) block characters and push a standing
  character out sideways when they walk into it (`characters[].pushed`; one fixed step behind the obstacle).
  `cc.colliders` / log warnings also name static colliders without an entity ("static collider Cuboid at (x,y,z)").
  `raycast_excluding(origin, dir, max, &[hero])` / `sphere_cast_excluding` skip the caster's own collider (a ray from
  inside the capsule otherwise hits it at distance 0); API `raycast {.., exclude: id|name|[..]}`.
  AI API: `collisions {since?, entity?: id|name}` (events + `started`/`stopped` counts), `{touching}`, `{inside}`,
  `overlap {pos, radius | half}`, `trigger {pos, half, name}`, `spawn {.., body: "kinematic"}`.
  Test: `games/demo/tests/collisions.txt`.
- **LOD:** glTF primitives with ≥ 256 triangles get vertex-clustered LODs (first switch at 40x their radius, i.e.
  when the object is ~30 px in radius at 60° FOV and 1350 px height: close cameras never switch). For a close
  camera (top-down at ~16 m) set it once per game: `ctx.gltf_defaults.lod_distance = 0.2` before loading.
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
- **Anti-aliasing:** 4x MSAA by default; `engine::run_with(game, RenderConfig { msaa: 1 })` (or the page URL
  `?msaa=1|4`, for A/B without a rebuild) renders without MSAA and runs FXAA in the composite: far less bandwidth on
  integrated/mobile GPUs (Brickhollow at 2400x1350 on a Radeon 780M: 74 -> 97 fps uncapped). The log says
  `render config: 1x MSAA (FXAA)`. Graphics menu: `engine::set_msaa_pref(Some(1), reload)` stores the player's choice
  (localStorage, used at the next start over the game's `RenderConfig`; the URL still wins); the sample count is baked
  into pipelines and targets, so it applies on reload.
- **Resolution:** `render {render_scale}` (0.25..1; the canvas backing store has the scene size too, the browser scales it), `auto_resolution` (default on) adapts it to
  `frame_budget_ms` of whole-frame GPU time (`gpu_ms.frame`); every change reallocates the targets (a 20-35 ms
  hitch at 4K), so it jumps once to a scale that fits and changes at most every 4 s; `stats.render` has `width/height` (scene) and `canvas`.
- **Render settings:** `call render` lists every setting; set any by name (sky, fog_*, bloom*, exposure,
  depth_darkening, vignette, saturation, particles, shadows, ...). Unknown names are an error.
- **Colour grading / looks** (composite pass, no extra pass; defaults neutral = no change): `render {grade: {preset,
  temperature, tint, lift:[r,g,b], gamma, gain, contrast, pivot, split_shadows:'#rrggbb', split_highlights,
  split_balance, saturation}}` in display space after tonemapping (white balance -> lift/gamma/gain -> contrast ->
  split toning -> saturation). Fields merge over the current grade; `preset` (`render` lists `grade_presets`: neutral,
  warm, cool, fade, vivid, noir, night, golden) resets to that look first. Game code: `engine::Grade` (also in the prelude):
  `settings.grade = Grade::preset("night").unwrap()`, tweak fields, `Grade::lerp(&day, &night, sky.night)` per frame. Masking vignette: `vignette_radius` > 0 (fraction of the half
  height) + `vignette_softness`, `vignette_color`, `vignette_roundness` (1 = circle on any aspect); radius 0 = the classic
  darkening. Film grain: `grain` (~0.3 subtle), `grain_size` px, animated per frame, stronger in the darks (frozen
  with `freeze_time`). No 3D LUT yet. Test: `games/demo/tests/grade.txt`.
- **Visual checks:** `render '{"debug_view":1}'` = normals, `2` = shadow term, `0` = lit.
- **3D UI:** `call panels` gives every world-panel widget with its on-screen centre (`screen`, CSS px);
  click it for real with `input {x,y}` + `input {button:0}`, or directly with `panel_click {id, widget}`.
- **Sky / day-night:** `call sky` = mode, `time_of_day`, sun/moon elevation, `daylight`, `night`, which body lights
  the scene, ambient; `sky {time: 18.5}` (turns the cycle on), `{cycle, day_length (min per day, 0 = still), mode:
  "atmosphere"|"gradient", clouds, stars, moon, skybox: url|null, skybox_blend, skybox_night_blend}`. The clock follows
  the simulation (`pause` + `step` are deterministic). Everything else via `render` (`sun_azimuth_deg`, ...).
  The cycle overwrites `sun_intensity` and the ambient colours every frame; to dim the lighting indoors set
  `render {sun_scale, ambient_scale}` (0..1, multiplied in after the cycle; the sky keeps its colours).
  Moving sun: shadows follow the light in `shadow_step_deg` steps (default 0.5) with a `shadow_fade_ms` cross-fade
  (default 400; a second shadow-map set is allocated on first use), so the texel grid does not re-snap every frame
  (shimmering edges). `shadow_step_deg: 0` = follow every frame.
  Proof per frame: `stats.render.shadow_hash` (cascade matrices) stays constant between `shadow_steps`;
  `shadow_fade` > 0 while cross-fading.
- **Opacity:** `MeshRenderer.opacity` (0..1) fades one entity without material copies; it moves to the blend pass
  and keeps its full shadow (0 = invisible, still casts). `set {id|name, opacity}` (applies to children, e.g. glTF).
  `Material.blend_casts_shadow` lets blended materials cast. Test: `games/demo/tests/opacity.txt`.
  **Solid fade** (`render {solid_fade: true}`, `settings.solid_fade`, default off): faded entities look like a 2D
  image with its alpha turned down: only their nearest surface per pixel (culling as the material says; inner bricks
  and far walls disappear), blended once. A depth-only pass copies the scene depth (frag_depth, per sample) into a
  second depth buffer and adds the faded entities' depth; the blend pass tests against it (equal = nearest). Main
  view only; blended materials, alpha-tested ones, water and render textures keep the classic path; blended things
  behind a faded entity (particles, water) are hidden by it. Hi-Z and post keep the real scene depth; shadows cast
  as before. `stats.render.solid_fade_draws`. Test: `games/demo/tests/solidfade.txt`. Entity commands
  accept `{name}` instead of `{id}`.
  Lens flare: `render {lens_flare: 0..}` (0 = off); test `games/demo/tests/flare.txt` (visible / hidden behind the tower).
- **Water:** `Material::water` + `ctx.spawn_water(name, mat, size, t)` / `ctx.spawn_ocean(mat, y)`; tune `wave_height`,
  `wave_scale`, `water_chop`, `water_clarity`, `water_foam`, `water_vertex_spacing`; `render {water_ssr, water_refraction,
  water_caustics}`. `render {debug_view:1}` shows the water normals.
  `spawn_water` tessellates each axis to the spacing (a 38.8x1.5 m strip at 0.1 m = 388x15 cells; at most 65536 cells,
  beyond that the spacing grows). API: `spawn {water: [x, z], pos, spacing}`. Test: `games/demo/tests/sky.txt`.
- **Music:** `call music` -> tracks with `state` (buffering/playing/paused/blocked/error), `time`, `duration`,
  `buffered_pct`; transitions are logged. Simulate a slow network for one path only:
  `curl -X POST "http://localhost:8080/__throttle?kbps=48&match=assets/music"` (`kbps=0` = off).
- **Audio:** `call audio` -> context state, sounds, live voices (with fx, bus), bus volumes, last 32 plays with sim `step`.
  Game code: `audio.is_ready(id)` / `sound_state(id)` (a sound from `ctx.load_sound` is `Loading` first; `play` before
  that returns `None` and warns once per sound), `is_playing(voice)`, `playing_count(id)`, `set_voice_volume(voice, v, fade_s)` (linear ramp on a playing voice, e.g. a looped rain that
  swells), `voice_volume(voice)`, `fade_out(voice, fade_s)` (ramp to 0, then stop); API `audio_voice {voice, volume, fade}`,
  `audio_stop {voice, fade}`. Lip sync: `voice_level(voice)` = loudness at the voice's current position, 0..1 (RMS
  envelope, 40 ms window, made once per sound on decode, divided by the sound's loudest point; not scaled by volume),
  `voice_time(voice)` = seconds into the sound (pitch/detune included, wraps when looped); both `None` once it ended.
  `sound_envelope(id)` = raw 100 Hz RMS (`ENVELOPE_RATE`) + peak. `call audio` shows `time`/`level` per voice and
  `envelope_peak` per sound. They follow the AudioContext clock (real playback), not `pause`/`step`. Test:
  `games/demo/tests/lipsync.txt`. Mixer buses `Bus::{Sfx,
  Voice, Ambience, Ui}`: `Play::new().bus(Bus::Voice)`, `set_bus_volume(Bus::Voice, v)`; API `audio_set {voice_volume}`,
  `audio_play {bus}`.
- **Procedural assets:** `call texgen {preset, apply_to}` and `call sfx {preset, overrides}` iterate on
  textures/sounds without a rebuild (see `.claude/skills/procedural-assets`).
- **Touch:** `call input '{"touch":{"id":1,"x":200,"y":600,"phase":"start"}}'` (then `move`/`end`) uses the same
  path as real fingers; the result's `touch` shows fingers, joystick `axis`, `look_delta`, `pinch`, `taps`,
  `swipes`. Real DOM touch can be simulated with `new PointerEvent('pointerdown', {pointerType:'touch', ...})`.
  Screens without walking (card table, menu): `ctx.input.touch.set_stick_enabled(false)` (API `input {stick:false}`)
  makes every finger a look/tap finger and hides the joystick ring.
- **Debug Chrome is muted** (`--mute-audio`): audio state/API checks work, nothing is audible. Use
  `launch --sound` only when the user wants to listen.
- **GPU-driven scene (main view):** every drawn or shadow-casting entity keeps a persistent object record (bounding
  sphere, LOD table, draw group per LOD level) next to its instance record (`crates/engine/src/gpu_scene.rs`). Per frame
  the CPU only compares each entity with its slot (transform, mesh, material content, opacity, `Lod`) and rebuilds the
  changed ones; one compute pass (`shaders/cull.wgsl`) picks the LOD level, culls frustum + Hi-Z and fills the id lists
  of the main view *and* of every shadow cascade rendered this frame, then each draw group (pipeline, texture, mesh) is
  one `drawIndexedIndirect`. Group slices only change when membership changes (spawn, despawn, material/mesh swap).
  Groups whose conservative bounds (AABB + LOD distance band) cannot be visible are not drawn; draw order per pipeline
  comes from the GPU's nearest-instance readback. The shadow cache marks a layer stale when a changed caster touches
  its light box. `call gpu_scene` = slots, groups, id layout, GPU counters (frustum, occluded, visible, shadow
  instances, LOD levels); test `games/demo/tests/gpuscene.txt`. `material_mut` with unchanged values costs nothing
  (content compared with a snapshot). Render textures and `render {gpu_culling:false}` use the CPU path (same
  records via the shared `classify`); `bridge flicker` uses it too.
- **Depth prepass** (`render {depth_prepass}`, default on, GPU path): opaque groups first write depth only, then shade
  with depth-equal (positions are `@invariant`), so each visible pixel runs the PBR shader once. Overdraw in foliage
  made the opaque pass 2.5x more expensive (Brickhollow 2400x1350: 4.9 -> 1.9 ms). Alpha-tested groups are not in the prepass.
- **Where the cost is:** `call gpu_scene` -> `top_groups` (mesh, members, triangles per draw group, LOD-0 candidates) and
  `gpu.triangles` / `gpu.shadow_triangles` (what the GPU really drew after culling + LOD; `stats.render.gpu_triangles`).
  To split the opaque cost into geometry and shading, compare against a trivial fragment shader: in Brickhollow it was
  1.0 ms of 4.9, the rest was shading x overdraw.
- **Culling/LOD:** `stats.render` has `gpu_culling`, `gpu_frustum_culled`, `gpu_occluded`, `gpu_visible`
  (async readback, 1-3 frames old) and `lod_levels`. On the GPU path `visible` = main-view candidates; `triangles` = drawn in
  the main view (after culling + LOD, readback, plus blended draws), `candidate_triangles` = LOD-0 upper bound of
  everything on the camera's layers before culling. `render {gpu_culling:false}` switches to the CPU path.
- **Frame rate / stutter:** `node tools/bridge.mjs perf [--ms 3000] [--profile]` measures real rAF intervals and
  returns `frame_ms` p50/p95/p99, `missed_frames_pct`, `cpu_phases` (game_fixed, physics, update, engine, render),
  `gpu_ms` per pass (cull, shadow, opaque, blend, hiz, post; WebGPU timestamp queries) and a `verdict`
  (smooth / gpu-bound / cpu-bound / stutter). `--profile [--top 40]` adds a CDP CPU profile (`top_self` functions).
  Live values: `stats.time.phases`, `stats.render.gpu_ms`. When the *user* sees low fps, ask them to press ` while
  it happens: debug mode writes a `"kind":"perf"` line per second (p50/p95/max, slow frames, phases, GPU) into the log.
- **Frame cap / even cadence:** `render {max_fps}` (0 = off) draws on every Nth display refresh, N the smallest whole
  divisor keeping the rate <= max_fps (45 on 90 Hz = every 2nd, 30 on 60 Hz = every 2nd, 45 on 120 Hz = 40). It is
  phase-locked to the refresh measured from rAF, so a device that *almost* makes the full rate (frames alternating 1/2
  vsyncs = judder, high `speed_jitter_pct`) gets an even cadence. Skipped refreshes run nothing; the simulation catches
  up in fixed steps and is drawn interpolated. `stats.time.refresh_hz`, `frame_interval`; `bridge perf` counts only drawn
  frames. While capped the sim clock advances whole refreshes x the measured vsync (`time.frame_dt`; the rAF delta is
  `frame_dt_raw`), so jitter cannot slip a step. Fixed steps still beat against the drawn rate (60 Hz steps at 45.9 fps
  = 1,1,2 per frame: walk cycles and game-moved Transforms judder); `ctx.time.lock_fixed_to_frames = true` (API `time
  {lock_fixed_to_frames, nominal_fixed_dt}`) sets `fixed_dt = frame_time / k` every frame, `k` = fewest steps no longer
  than `nominal_fixed_dt` (1/60): 45.9 fps -> 2 x 10.9 ms steps, the same count every frame (`time.steps_per_frame`;
  a late frame runs the steps for its extra refreshes). The step length then follows the display (uncapped 120 Hz =
  120 Hz simulation), so read `ctx.time.fixed_dt` instead of assuming 1/60. Test: `games/demo/tests/pacing.txt`.
- **Plastic / glossy look:** `render {ambient_specular: 0..1}` (default 0) adds the ambient sky/ground colours reflected
  with a split-sum fresnel (F0 0.04, albedo for metals) on lit opaque materials, fading with roughness and occluded
  below the surface's horizon: rims and sky sheen outside the sun's highlight (roughness ~0.2 reads as plastic).
  Per material (lit, ALU only, no extra pass): `Material::clearcoat` (0..1) + `clearcoat_roughness` (0.06) = a second
  sharp GGX lobe (F0 0.04) over the base, also in the ambient specular; `Material::subsurface` (0..1) +
  `subsurface_color` (None = albedo) = wrap diffuse for the sun (softer terminator, light into the shade) plus
  unshadowed thin back-light translucency (clustered lights stay plain). LEGO-like: roughness 0.3, clearcoat 1,
  subsurface 0.3. They ride in the unused part of the instance's normal matrix (no layout change). API: `material`
  lists every material; `material {id, roughness, metallic, subsurface, subsurface_color, clearcoat,
  clearcoat_roughness}` sets them. Test: `games/demo/tests/plastic.txt`.
- **Input lag on a GPU-bound device:** without a limit Chrome queues frames and the screen shows one that is
  up to a second old (desktop repro: 320vw canvas, 7 frames in flight, 1 s latency). `render {max_frames_in_flight}`
  (default 2) skips rendering a frame (simulation and input still run) while `gpu_latency_ms` > 6 frame intervals (min
  150 ms) or `frames_in_flight` >= 64. Desktop Chrome resolves `onSubmittedWorkDone` late (up to 5 in flight,
  30-110 ms without any backlog), so tighter rules drop frames on a smooth desktop. It skips until
  the GPU catches up. `stats.render`: `frames_in_flight`, `gpu_latency_ms` (submit -> GPU done, about in-flight x GPU
  frame time), `frames_skipped`.
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
  Cascades 1.. are fitted `shadow_margin` (0.08) larger and reused while the camera stays inside that margin. A
  changed caster whose shadow lands in a cascade's own depth slice (reach = its size / tan(sun elevation), plus the
  blend band) re-renders that cascade the same frame, so a moving hero never gets a stepped, lagging shadow; only
  changes whose shadow falls elsewhere wait up to `shadow_refresh` frames (`[2,3,4]`). A still scene renders no
  shadow pass at all: `stats.render.shadow_passes_rendered` / `shadow_passes_cached` / `shadow_cascades_rendered`
  (per cascade). `call gpu_scene` lists what changed this frame (`changed`: entity + moved / rebuilt / mesh deformed
  / removed). `render {shadow_cache:false}` = old behaviour (A/B, debugging); render textures invalidate it.
  Test: `games/demo/tests/shadowcache.txt`.
  **Static shadow cache** (`render {shadow_static_cache}`, default on, GPU path): casters that moved or deformed in the
  last 10 s are *dynamic* (cull.wgsl writes them into a second id list per pass). Each cascade keeps a copy with only
  the static casters; when only dynamic casters changed, the cascade keeps its matrix (cascade 0 gets the
  `shadow_margin` too), restores the texel rect where dynamic casters were and are (scissored frag_depth draw from the
  copy, not a full copy: WebGPU copies depth only whole) and draws just the dynamic casters. `stats.render`:
  `shadow_static_rendered`, `shadow_restored_texels`, `shadow_static_why` (totals: [no copy, sun direction, static caster
  changed, cascade re-fitted, other]); `call gpu_scene` -> `dynamic_casters`, `static_changes` (slot, entity, why). A sun
  step re-renders the new set's cascades one per frame (cascade k after k frames). Brickhollow walking at 2400x1350: shadow 3.5-5 ms -> 0.8-1.5 ms.
- **Blend pass skip:** with no transparent draws, particles or water the opaque pass resolves straight into the HDR
  target (saves a load + store of the 4x MSAA target).
- **Instance upload:** every drawn or shadow-casting entity owns a stable slot in the instance buffer (`InstancePool`
  in `renderer.rs`; a faded entity has a second slot for its shadow copy). A record is uploaded only when it changed;
  draws go through 4-byte id lists (`visible_ids`: written by `cull.wgsl` on the GPU path, by the CPU for blended
  draws and on the CPU path), so a changing visible set no longer shifts the records. Unused slots are freed after a
  few frames. `stats.render.instance_upload_bytes` (records + objects + args + ids), `instance_slots`. Brickhollow
  walking: ~500 KB -> ~5 KB per frame. The audio listener is only updated for positional voices and a
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
| `crates/engine/src/gpu_scene.rs` | GPU-driven scene: persistent objects + draw groups, change detection, `cull.wgsl` dispatch, readback |
| `crates/engine/src/gpu_cull.rs` | Hi-Z occlusion pyramid |
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
| `crates/engine/src/physics.rs` | rapier3d wrapper, `Body` component, collision events, triggers, overlap/sphere/ray queries |
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
  Game-specific commands go in `Game::debug_command` + `debug_commands`. Engine commands win on a name clash: the
  game's is never called (startup warning, `help` marks it `shadowed_by_engine`). `help` also lists the commands of
  `ScreenStack` screens that are not on top (`"screen"`, `"active": false`); calling one names the screen to switch to.
- `wasm-bindgen` in `Cargo.toml` is pinned to the installed CLI version (`wasm-bindgen --version`).
- Input: `input.key` = held; `key_pressed` = edge — in `update` per frame, in `fixed_update` latched since the last
  fixed step (short taps and taps while paused are never lost). Jump/fire-style actions: `key_down_or_pressed`.
  Every new action gets a scripted test with a short tap (`key Space` + `step N` while paused).

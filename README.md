# wasmOnly — Chrome-only 3D engine in Rust/WASM, AI-first

An engine that builds exclusively for `wasm32-unknown-unknown` and runs in Chrome (WebGPU via
`web-sys`, WGSL, hecs, rapier). It is "AI-first": every piece of runtime state can be queried and
controlled as JSON through `window.engine.call(...)`, and a Chrome debug bridge (CDP) exposes that as a
CLI and an MCP server for agents. Agent instructions live in [`.claude/`](.claude/README.md); starting a
new game in another repository: [HANDOFF.md](HANDOFF.md).

## Quick start

Requirements: `rustup` (the toolchain is pinned to **1.99.0** in `rust-toolchain.toml` and installed
automatically), `wasm-bindgen-cli` **in exactly the version from `Cargo.toml`** (currently 0.2.129),
Node ≥ 22 and Chrome. The optional threads backend additionally needs a pinned nightly (see
[Threads](#threads-optional-nightly)).

```bash
node tools/make-sample-assets.mjs   # once: generates tree.glb, crate.glb, critter.glb, rock.obj
node tools/dev.mjs watch            # build, serve http://localhost:8080, rebuild + reload on change
node tools/bridge.mjs launch        # Chrome with remote debugging (port 9222), muted (--sound for audio)
node tools/bridge.mjs call stats    # query the engine
node tools/dev.mjs build --check    # build + check (z-fighting, engine errors) in the debug Chrome
node tools/bridge.mjs run games/demo/tests/smoke.txt   # scripted test run
```

New game in its own folder/repository: `node tools/new-game.mjs ../my-game`, then
`node tools/dev.mjs watch --game-dir ../my-game` (see [HANDOFF.md](HANDOFF.md)).

Slow network test: `curl -X POST "http://localhost:8080/__throttle?kbps=48&match=assets/music"` throttles
only matching paths (`kbps=0` turns it off). The dev server answers range requests, so music streams as it
would from a real web server.

Debug mode: press <kbd>`</kbd> (on German keyboards <kbd>^</kbd>). Key presses, mouse clicks, the camera
pose, and once per second frame timings, character states and the game state are written to
`logs/debug-*.log` in the project folder; press again to close the file. While it is on, the badge shows fps, CPU/GPU time and resolution, and
**Ctrl+click** on an object copies its identifying properties (name#id, scene, position, mesh, material, parent
chain) to the clipboard (`engine.call("inspect", {x, y})` returns the same data).

Debugging in DevTools: the dev build contains DWARF for engine and game code. With the Chrome extension
"C/C++ DevTools Support (DWARF)" you can set breakpoints directly in `.rs` files.

## Decisions and status

| Area | Decision | Status |
|---|---|---|
| Target | `wasm32-unknown-unknown`, `wasm-bindgen`, `web-sys` | ✅ |
| SIMD | `+simd128` for the whole build (glam and rapier use it) | ✅ |
| Graphics | WebGPU directly via `web-sys` (unstable cfg), WGSL incl. compute shaders | ✅ |
| Rendering | Forward PBR (GGX) into an HDR target, 4× MSAA or FXAA (`engine::run_with(game, RenderConfig { msaa: 1 })`, URL `?msaa=1`), ACES tonemapping, adaptive render resolution (whole-frame GPU time, predictive, rare changes), render interpolation of physics bodies and characters between fixed steps | ✅ |
| Material maps | Normal mapping (tangent frame from screen-space derivatives, no vertex tangents needed), height maps for **vertex displacement** (on `MeshData::grid/disk`) and **parallax occlusion mapping**; normal/height maps from any texture preset (`ctx.generate_maps`) | ✅ · shadows use the undisplaced mesh |
| Water | `Material::water(deep, shallow)` (Unreal-style): 8 **Gerstner waves** with deep-water dispersion (vertex displacement + per-pixel analytic normal) and drifting ripple octaves; **refraction** of the opaque scene with per-channel absorption (`water_clarity`) and in-scattering towards the deep colour; **caustics** on the ground below; **screen-space reflections** with fallback to the full sky (atmosphere, clouds, moon, skybox); GGX sun specular + glint; subsurface glow through crests; shore and crest foam (`water_foam`, `water_chop`); soft shore edge. Meshes: `ctx.spawn_water` (bounded, tessellated to `water_vertex_spacing`), `ctx.spawn_ocean` (endless, camera-following radial mesh, 5 km). Render settings `water_ssr`, `water_refraction`, `water_caustics` | ✅ · planar reflections missing (off-screen objects reflect as sky) |
| Transparency | `AlphaMode::Opaque / Mask(cutoff) / Blend` and `double_sided` (also from glTF). Mask = alpha test with cut-out shadows, batched and GPU-culled like opaque geometry. Blend = own pass after all opaque geometry, sorted back to front per instance, no depth write | ✅ |
| Lighting | Sun with **cascaded shadow maps** (1–4 cascades up to `shadow_distance`, stable bounding spheres, texel snapping, casters culled per cascade, blend band, **shadow cache**: a cascade is re-rendered only when its matrix or casters changed, far cascades are reused within a margin while the camera moves), **clustered lighting** (16×9×24 froxels, up to 2048 point and **spot lights**, CPU assignment ~0.2 ms for 400 lights), hemispheric ambient. Debug views for cascades and lights per cluster | ✅ · IBL and spot-light shadows missing |
| Culling / draws | GPU-driven scene: persistent object records (bounds, LOD table, draw groups) updated only for changed entities; one compute pass picks the LOD, culls frustum + Hi-Z and fills the id lists of the main view and every shadow cascade, each draw group is one `drawIndexedIndirect`. Fallback (render textures, no compute): CPU frustum culling with direct draws | ✅ · no multi-draw indirect (one draw per group) |
| Instancing | One storage buffer for all instances, one draw per (texture, mesh) | ✅ (5000 cubes: 60 fps, 7 draw calls) |
| LOD | `Lod` component, distance relative to object size. Automatic levels by vertex clustering for glTF meshes from 256 triangles; the built-in sphere has three levels | ✅ |
| Scene | hecs ECS, `Transform` + `Parent` → `GlobalTransform` | ✅ |
| Assets | Runtime only glTF 2.0 (`.glb`), images via `createImageBitmap`, cache per URL, all asynchronous. Mipmaps generated on the GPU, 8× anisotropic filtering | ✅ · KTX2 and texture streaming missing |
| Foreign formats | `tools/asset-cook` (russimp/Assimp, native): OBJ/FBX/DAE/3DS/… → `.glb` | ✅ |
| Physics | rapier3d 0.36 as the fixed module, `Body` component, raycasts, CCD; **collision detection** for game logic: start/stop events for every touching pair (contacts, sensors, character touches) without collider flags, trigger zones with enter/exit/inside, overlap (sphere, box) and sphere-cast queries; kinematic obstacles push characters out | ✅ |
| GUI / HUD | DOM overlay: HUD slots, `data-action` clicks, `WorldLabel` (anchored in the world) | ✅ |
| 3D GUI | **World panels**: 2D widgets (label, button, bar, rect) drawn with Canvas2D into a texture on a 3D quad. Clicks by ray and UV, occlusion by physics, same action queue as the HUD | ✅ · SVG→texture (resvg) missing |
| Input | Keyboard, pointer, wheel, pointer lock, gamepad; DOM UI and world panels take precedence | ✅ |
| Touch (Android) | Multiple fingers via pointer events. Gestures: virtual joystick (left side of the screen, feeds `move_axis()`), look, pinch zoom, tap (clicks world panels), hold, swipe. The joystick is drawn. iOS is deliberately not a target | ✅ |
| Audio | Web Audio: master gain, 3D sound via `PannerNode` (HRTF), listener follows the camera, files via `decodeAudioData`. When the tab loses focus the AudioContext is suspended and resumes at the same position (opt-out: `set_pause_on_blur(false)`) | ✅ |
| Music | **Streaming**: long tracks play through an `<audio>` element with HTTP range requests, start as soon as enough is buffered and never block the game. Crossfade, prefetch, separate music and SFX buses, autoplay retry, focus pause. `ctx.audio.play_music(url, MusicOpts)` | ✅ |
| Procedural audio | Deterministic synth (`engine::audio::synth`): sine/square/saw/triangle, white/pink/brown noise, ADSR, pitch slide, vibrato, arpeggio, harmonics (timbre), bitcrush. 16 presets (coin, laser, explosion, powerup, …, loops wind/rain/engine). Playback effects as Web Audio nodes: low/high/band pass, echo with feedback, reverb (convolver), distortion, pitch/detune, stereo pan | ✅ |
| Procedural textures | `engine::texgen` with the `noise` crate: 18 seamlessly tiling presets (noise, clouds, marble, wood, stone, grass, sand, dirt, lava, water, bricks, tiles, metal, rust, camo, checker and with alpha: blades, leaves), parallel via the job system | ✅ |
| Animation | glTF clips (step, linear with slerp, cubic spline) for translation/rotation/scale and morph weights, `Animator` with crossfade, speed and looping; skinning and morph targets on the CPU into a per-instance vertex buffer (shadows, culling and z-fighting tools see the deformed mesh). Time advances in the fixed step (deterministic) | ✅ · GPU skinning missing |
| Screens & scenes | `ScreenStack` of `Screen`s (title, game, pause, …) with `Push/Pop/Switch/Reset`; every screen has its scene, everything created in it (including glTF nodes loaded later, static colliders, particles) is unloaded when it exits; overlays pause the simulation below | ✅ |
| Save games | Slots in `localStorage` (versioned JSON, migration by version), large data in IndexedDB | ✅ |
| Character & camera | `CharacterController` (rapier KCC: slopes, steps, ground snapping, coyote time, jump buffer, pushes bodies, jumps work against walls, logs a warning when input has no effect), `ThirdPersonCamera` (orbit, zoom, sphere-cast collision, smoothing) | ✅ |
| Atmosphere & sky | `sky_mode` 0 = gradient, 1 = **physically based atmosphere** (Rayleigh + Mie + ozone single scattering with a multiple-scattering term, rendered per frame into a 192×108 sky-view LUT used by sky, fog, clouds and reflections); procedural **clouds** (coverage, drift, self-shadowing, silver lining), **stars**, **moon** with phases; **skybox** images (`ctx.load_skybox`, equirectangular, blended by day/night); exponential height fog (colour = sky at the horizon), HDR target, bloom, depth darkening, **lens flare** (glow, starburst, coloured hexagon/ring ghosts; hidden by objects and clouds in front of the sun, `render.lens_flare`), exposure, ACES, vignette | ✅ |
| Day/night | `day_cycle` + `time_of_day` move sun and moon along their arcs (`sun_azimuth_deg`, `sun_max_elevation_deg`) and drive light direction, colour (atmospheric transmittance: orange sunsets) and intensity, moonlight at night, ambient sky/ground colours and a night exposure boost; advances with the simulation clock (`day_length_minutes`, pause stops it). `ctx.renderer.sky` = sun, moon, `daylight`, `night` for game logic (lamps). AI API `sky {time, cycle, day_length, ...}` | ✅ |
| Particles | CPU simulation in the fixed step, 8 presets (fire, smoke, sparks, fireflies, dust, magic, rain, splash), instanced billboards, additive or sorted, soft against the scene depth, sparks stretched along their velocity | ✅ |
| Render to texture | A camera renders the full pipeline into a texture (material, or PNG export as a blob URL for the DOM), render layers, transparent background, once or every frame | ✅ |
| Terrain | Heightmap (noise with island falloff or your own data), chunks with 4 LOD levels and skirts, splat material (4 layers from an atlas, splat map), heightfield collider per chunk, chunk streaming around the camera, editing (flatten for roads and building plots, raise, paint), height/normal/splat queries, scatter rules for trees, bushes, rocks | ✅ |
| Vegetation | **GPU grass**: blades are created entirely in the vertex shader (camera-centred grid, height from the heightmap texture, density from a splat channel, distance thinning, wind), ~120,000 blades in one draw call, with cascaded shadows, clustered lights and fog; bushes/trees via scatter rules (instanced, LOD cull) | ✅ |
| Streaming & precision | Streaming layers: world cells whose generator runs when the camera approaches and whose scene is unloaded when it leaves (budget per frame, hysteresis). **Floating origin**: `ctx.origin` (f64) shifts automatically when the camera is farther than `origin_shift_distance`; transforms, bodies, colliders, particles and the camera move along | ✅ |
| Procedural content | Geometry kit `MeshBuilder` (parts per material, transform stack, world UVs, boxes, polygons, extrusion, walls with door/window openings, sweep, lathe), generator registry (results cached per parameter set → instancing), parallel generation in the job system, callable live through the AI API. The engine deliberately contains no building generators | ✅ |
| Static batching & motion | `batch_static` merges static props per material and cell into few draws; `Motion` (spin, sway, bob) moves instances without game code (Ferris wheel, cranes, carousel) | ✅ |
| Test scripts | `bridge run` / MCP `run_script`: list of input, `step`, waits, `expect`/`wait_for` on JSON paths and screenshots, report per step, exit code 1 on failure | ✅ |
| Threads | Optional backend: Web Worker pool (rayon via `wasm-bindgen-rayon`), shared memory (SharedArrayBuffer), atomics. The job API `engine::jobs` is the same in both builds | ✅ · asset processing in workers missing |
| Build | `tools/dev.mjs` (cargo + wasm-bindgen, opt-in `--wasm-opt`; `--install-wasm-opt` cargo-installs the pinned version, built from source), `--game-dir` for games in other repositories, `tools/new-game.mjs` scaffold. Toolchains pinned: stable 1.99.0, threads backend nightly-2026-11-01 | ✅ |
| AI tooling | `window.engine.call`, structured logs, deterministic stepping, CDP bridge, MCP, `.claude/` skills | ✅ |
| Z-fighting / flicker | `engine.call("zfight")` finds coplanar, overlapping surfaces of different objects (runs automatically in debug builds within 80 m of the camera after scene changes and warns in the log; `{radius}` for big worlds). `bridge flicker` compares screenshots with reversed draw order and shifted depth mapping (pixels stay put) and names the flickering objects. `bridge check` / `dev.mjs build --check` is the build check (exit code 1 on problems) | ✅ |
| Performance | `bridge perf [--profile]`: real frame intervals (p50/p95/p99, missed frames), CPU time per engine phase, GPU time per pass (WebGPU timestamp queries), optional CPU profile via CDP, with a verdict (smooth/gpu-bound/cpu-bound/stutter). Opaque pipelines without `discard` (early-Z), alpha test through separate mask pipelines. Adaptive resolution keeps the GPU inside the frame budget | ✅ |
| Debug mode | Key <kbd>`</kbd> (left of 1; on German keyboards <kbd>^</kbd> or Shift+´): logs every key and mouse press, once per second the camera, frame timings, characters and the game state as JSON lines in `logs/debug-YYYYMMDD-hhmmss.log` (via the dev server), shows a DEBUG badge with fps, CPU/GPU time and resolution; **Ctrl+click** copies an object's identifying properties to the clipboard (`inspect {x, y}` in the API) | ✅ |

## Architecture

```
games/<game>  (Rust, implements engine::Game, usually as a ScreenStack of Screens)
   │
crates/engine ── app (loop, Ctx) · scene · screens · save · input · physics · character · animation · terrain
   │             streaming · geom · batch · particles · lights · gltf_loader · ui · panel · audio · jobs · debug (AI API) · log
   │
renderer ─────── WebGPU: [compute: culling] → shadow cascades → opaque (+ grass, sky) → blend (+ particles)
   │             → [compute: Hi-Z] → bloom → composite (depth darkening, tonemap) — per view (canvas, render textures)
   │
Browser ──────── web-sys: WebGPU · DOM · OffscreenCanvas · Web Audio · fetch · createImageBitmap · Gamepad · Worker · IndexedDB
```

Frame: `requestAnimationFrame` → fixed steps of 1/60 s (`Game::fixed_update`, characters, particles,
animation clocks, motion, rapier) → world panels (hover, clicks → actions) → `Game::update(dt)` →
streaming + terrain → third-person cameras → animation poses → transform propagation → skinning →
panel redraw → audio listener → render → position `WorldLabel`s.

## Writing a game

```rust
use engine::prelude::*;
use engine::physics::rapier::prelude::*;

#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() { engine::run(MyGame::default()); }

#[derive(Default)] struct MyGame;

impl Game for MyGame {
    fn init(&mut self, ctx: &mut Ctx) {
        ctx.world.spawn((Transform::from_xyz(0., 3., 8.).looking_at(Vec3::ZERO, Vec3::Y), Camera::default()));
        let b = ctx.spawn_cube("box", Vec3::new(0., 4., 0.), Vec3::ONE, 0xff8800);
        ctx.physics.add(&mut ctx.world, b, RigidBodyBuilder::dynamic(), ColliderBuilder::cuboid(0.5, 0.5, 0.5));
        ctx.load_gltf("assets/tree.glb", Transform::from_xyz(2., 0., 0.));
        ctx.ui.set("title", "<div class='panel'>Hello</div>");

        // clickable 3D GUI: a click on "Start" arrives as action "start" in ctx.ui.take_actions()
        let panel = Panel::new(512, 256).label("t", [24., 20., 460., 40.], "Hello 3D", 32.).button("go", [24., 140., 220., 70.], "Start", "start");
        ctx.spawn_panel("menu", panel, Transform::from_xyz(0., 2., 0.).with_scale(Vec3::new(2., 1., 1.)));

        // sound without asset files
        ctx.audio.add_samples("zap", &synth::zap(), synth::RATE);
        ctx.audio.play_named("zap", Play::new().at(Vec3::ZERO));
    }
}
```

A fuller skeleton (screens, save game, character controller, third-person camera, debug state, test
script) is what `node tools/new-game.mjs <dir>` creates.

The example `games/demo` ("Tumble") uses every system: roll a ball, push the crate tower off the island,
shoot, HUD buttons, a terminal as a world panel (restart, shadows, slow motion, sound), a pond (stone rim
with normal and parallax mapping, gravel bed with displacement, water shader), a sea with the same water
shader, 3D sound with effects (laser with echo, impacts with reverb, wind ambience), procedural textures
for ground, cliff and metal, glTF trees, cooked rocks and dynamic glTF crates. On top: screens (title
with a card portrait via render to texture, game, pause overlay), a save game (best score, wins, rounds),
an animated creature (`critter.glb`: skinning, morph targets, three clips) as a second playable figure
with a character controller (key <kbd>C</kbd>), third-person camera, glowing mushrooms with fireflies
(bloom), a campfire with smoke, and splashes when crates fall into the sea.
Smoke test: `node tools/bridge.mjs run games/demo/tests/smoke.txt`.
The **open world** (title → "Open world" or key <kbd>O</kbd>): 1 km island terrain with a town plaza and a
ring road, streamed cells with street lamps (hundreds of lights, <kbd>N</kbd> = night), fences via static
batching, pines and rocks via scatter rules, GPU grass, a Ferris wheel and a car with headlights (spot
lights) via `Motion`, teleport to the island edge (<kbd>T</kbd>) with an origin shift. The generators (pine,
lamp, fence, pavilion, Ferris wheel) are game code in `games/demo/src/world.rs`.
Test: `node tools/bridge.mjs run games/demo/tests/world.txt`.

## Threads (optional, nightly)

The engine core builds on stable. Only the threads backend needs nightly, because the standard library
has to be rebuilt with `+atomics,+bulk-memory` (`-Z build-std`). The version is pinned in `Cargo.toml`
under `[workspace.metadata.wasmonly] threads-toolchain`:

```bash
rustup toolchain install nightly-2026-11-01 --profile minimal --component rust-src --target wasm32-unknown-unknown
node tools/dev.mjs build --threads     # or: watch --threads; output in target/threads
```

The result uses shared memory (SharedArrayBuffer, needs COOP/COEP, which the dev server sends) and starts
a worker pool with `navigator.hardwareConcurrency` threads before `Game::init`. Game code uses
`engine::jobs::{par_map, par_for_each_mut}`: serial without the backend, parallel with it, with identical
results. `stats.jobs` shows what is active. The feature `threads-physics` additionally lets rapier run in
parallel; that is faster, but the order of contacts is not guaranteed between runs.

## Asset cooker (russimp)

Assimp is C++ and cannot be built for `wasm32-unknown-unknown`. So russimp runs as a native build tool,
and the engine only sees `.glb` at runtime:

```bash
cargo run --release --manifest-path tools/asset-cook/Cargo.toml -- games/demo/assets-src/rock.obj games/demo/assets/rock.glb
cargo run --release --manifest-path tools/asset-cook/Cargo.toml -- --batch games/demo/assets-src games/demo/assets
```

`russimp-sys` needs **libclang** at build time (for bindgen): install LLVM (`winget install LLVM.LLVM`)
or point `LIBCLANG_PATH` at a folder containing `libclang.dll`. The `prebuilt` feature downloads the
prebuilt Assimp library, so CMake is not needed.

`russimp-sys` 2.0.2 ships bindgen 0.63, which generates empty `aiNode`/`aiScene` structs with libclang 16
and newer. `tools/asset-cook/vendor/russimp-sys` therefore contains a copy that only bumps bindgen to
0.72 (wired in via `[patch.crates-io]`).

## AI API (excerpt)

`engine.call("help")` lists every command. The most important: `stats`, `entities`, `entity`, `set`,
`spawn`, `despawn`, `pause`, `resume`, `step`, `time_scale`, `seed`, `input`, `logs`, `render`, `camera`,
`raycast`, `pick`, `ui`, `debug_mode`, `music`, `zfight`, `panels`, `panel_click`, `audio`, `audio_play`,
`audio_stop`, `audio_set`, `sfx`, `sky`, `inspect`, `texgen`, `assets`, `screens`, `scenes`, `save`, `characters`,
`animation`, `particles`, `lights`, `render_texture`, `terrain`, `grass`, `world`, `generate`, `batch`,
`game_state` and the game's own commands. See [AGENTS.md](AGENTS.md).

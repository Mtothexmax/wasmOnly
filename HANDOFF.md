# HANDOFF — building a game on wasmOnly in another repository

You are an agent starting (or porting) a game on this engine. Read this file, then
[AGENTS.md](AGENTS.md) and [.claude/README.md](.claude/README.md). Everything below is verified by the
engine's own test scripts.

## 1. What the engine gives you

Rust → `wasm32-unknown-unknown` → Chrome (WebGPU). AI-first: every runtime state is JSON through
`window.engine.call(name, args)`, driven from the shell by `tools/bridge.mjs` (CLI + MCP).

| Need | Engine feature (module) |
|---|---|
| Game structure | `ScreenStack` of `Screen`s (title, menus, play, pause overlays), scenes unload with their screen (`screens`) |
| Saves | `ctx.save` slots (localStorage, versioned JSON) + blobs (IndexedDB) (`save`) |
| Languages | string tables per locale (JSON, plurals, `{name}` placeholders), the player's choice stored, `changed()` to rebuild text, `{locale}` asset paths with fallback (`ctx.i18n`, `i18n`; AI API `locale {missing}` lists untranslated keys) |
| Player | `CharacterController` (slopes, steps, jumps incl. against walls, pushes bodies) + `ThirdPersonCamera` (`character`) |
| Models | glTF `.glb` (static, skinned, morph targets, clips via `Animator`) (`gltf_loader`, `animation`); other formats via `tools/asset-cook` |
| Procedural content | `MeshBuilder` geometry kit + generator registry, cached + instanced (`geom`); static batching, `Motion` (`batch`) |
| World | heightmap terrain (LOD chunks, splat, colliders, scatter), GPU grass, streaming cells, floating origin (`terrain`, `vegetation`, `streaming`) |
| Look | PBR, cascaded shadows, clustered point/spot lights, physically based sky (clouds, stars, moon, skybox) with day/night cycle, fog, bloom, depth darkening, particles, Unreal-style water (Gerstner waves, refraction, SSR, caustics, foam), render to texture, colour grading (`render {grade}`, `engine::Grade`), plastic materials (`Material` clearcoat / subsurface), wind-bent vegetation (`Material::bend`), per-entity opacity with solid fade |
| Frame rate | automatic frame pacing (`pacing`): full rate when the GPU keeps up, every 2nd/3rd refresh when not, back when there is headroom; no GPU queue (back-pressure). `max_fps` is only an optional limit; `ctx.time.lock_fixed_to_frames` for the same step count every frame |
| UI | DOM HUD slots with `data-action` buttons (`ui`), clickable world panels (`panel`), world-anchored labels |
| Audio | 3D sounds, procedural sfx presets, streamed music (`audio`) |
| Physics | rapier3d (`physics`): bodies, collision events (`collision_events`, `entered`, `exited`, `inside`), trigger zones (`ctx.spawn_trigger`), overlap / sphere-cast / ray queries |

## 2. Start a project

```bash
node <engine>/tools/new-game.mjs <your-repo-or-folder> --name my-game      # engine as a path dependency
node <engine>/tools/new-game.mjs <dir> --name my-game --engine-git        # or as a git dependency (github main)
node <engine>/tools/dev.mjs watch --game-dir <dir>                        # build + serve :8080 + live reload
node <engine>/tools/bridge.mjs launch                                      # debug Chrome (CDP 9222), muted
node <engine>/tools/bridge.mjs run <dir>/tests/smoke.txt                   # the scaffold's smoke test passes
```

The scaffold contains: `Cargo.toml` (pinned wasm-bindgen), `rust-toolchain.toml`, `.cargo/config.toml`
(WebGPU cfg + simd128), `src/lib.rs` (ScreenStack with title + play screen, save game, character,
camera, a debug command, `debug_state`), `tests/smoke.txt`, `.mcp.json`, `CLAUDE.md`, `AGENTS.md`.
Optional: `web/index.html` in your folder replaces the engine's page (keep the `#canvas`, `#ui .hud`
structure and the `engine-ready` script).

Only one dev server can use :8080 at a time. A second project runs its own server and its **own debug
Chrome** (shared Chrome tabs steal focus and frames from each other):
`node <engine>/tools/dev.mjs watch --game-dir <dir> --port 8091`, then
`node <engine>/tools/bridge.mjs launch --port 9223 --url http://localhost:8091/` and the same `--port 9223
--url http://localhost:8091/` on every bridge command (or `CDP_PORT=9223 GAME_URL=http://localhost:8091/`).

Detailed small models (figures, faces) seen from far cameras: `ctx.load_gltf_with(url, t, GltfOpts {
auto_lod: false, ..Default::default() })`, `ctx.gltf_defaults.auto_lod = false`, or
`render.lod_distance_scale` (bigger = more detail; formerly `lod_bias`). Far forests: `GltfOpts { impostor: true, .. }`
(one camera-facing card per model beyond ~60x its radius); far static towns: `ctx.batch_static_with(root, cell,
Some(Hlod { distance, cluster }))`; a far clip that fades into the fog: `render.view_distance`.
LEGO baseplates without stud geometry: `Material::studs = Some(StudRelief::lego(pitch_uv))` (analytic per-pixel
studs, quality `render.stud_relief` 0/1/2).
Custom material looks (ice crystals, holograms, animated patterns): a game surface shader, `ctx.renderer.add_surface_shader(name,
include_str!("x.wgsl"))` + `Material::surface` / `surface_params`; the WGSL `fn surface(s: SurfaceIn) -> SurfaceOut` replaces
only the material-map block, the engine keeps lighting, shadows, fog and every pipeline variant (AGENTS.md "Surface shaders").

## 3. Working rules (they prevented real bugs)

1. **Verify with JSON and test scripts**, not screenshots: `call game_state`, `call entity {id}`,
   `run tests/*.txt`. Put every check you will want again into a test script.
2. **`debug_state`** returns everything a test needs (positions, flags, counters); game commands
   (`debug_command`) let tests set up situations (teleport, give item, open dialog).
   A long `json!` in `debug_state` hits the macro recursion limit ("recursion limit reached while expanding"):
   add `#![recursion_limit = "256"]` at the top of the game's `lib.rs`, or build it from smaller `json!` parts.
3. **Input**: `input.key` = held; `input.key_pressed` = edge (inside `fixed_update` latched since the last
   step, so taps are never lost); actions like jump/fire: `input.key_down_or_pressed`. Every action gets a
   scripted short-tap test (`pause` → `key Space` → `step 12` → `expect ...`).
4. **Bug reports**: ask the user to press <kbd>`</kbd> (German keyboards <kbd>^</kbd>) before and after
   reproducing. `logs/debug-*.log` then holds keys, camera, and once per second frame timings, all
   character states and your `game_state`. The character controller logs a warning when movement or a
   jump has no effect, with the entity that blocks it.
5. **Performance**: `bridge perf` gives a verdict (smooth / gpu-bound / cpu-bound / stutter) with the
   biggest pass or phase. Frame pacing picks the cadence (`stats.render.pacing`), adaptive resolution keeps the GPU
   inside `render.frame_budget_ms`; don't set a fixed `max_fps` per device. Call `ctx.renderer.prewarm()` at the end of
   a loading screen so first-use pipeline compiles (mobile drivers) don't hitch mid-game.
6. **Z-fighting**: the debug build checks automatically near the camera (then only new entities); `zfight {radius}`
   on demand, `{same_root: true}` ignores parts of one model, `NoZCheck` opts an entity out.
   Fix by offsetting surfaces, never with depth bias.
7. Engine gaps or bugs: fix them in the engine repository (with a test in `games/demo/tests`) so every
   game benefits; keep game content (levels, generators, art) in the game repository.

## 4. Porting from three.js / Svelte (e.g. a Lego-style town game)

| three.js / web app | wasmOnly |
|---|---|
| `Scene`, `Group`, `Object3D` hierarchy | hecs entities with `Transform` + `Parent`; `ctx.spawn_mesh`, `ctx.spawn_parts` |
| `Mesh` built in code (boxes, cylinders, minifig parts) | `MeshBuilder` parts (`block`, `cuboid`, `cylinder`, `lathe`, `extrude`, `sweep`, `wall` with openings) registered as generators; parts = materials, cached → instanced |
| `MeshStandardMaterial` | `Material` (PBR: `base_color`, `roughness`, `metallic`, `emissive`, textures, alpha modes) |
| Procedural limb animation (sin-based swing) | `Motion` (spin/sway/bob) for simple loops, or set `Transform`s in `update`; glTF clips for authored animation |
| `requestAnimationFrame` loop + `clock.getDelta()` | `Screen::fixed_update` (1/60 s, deterministic) + `Screen::update(dt)` |
| Svelte overlays (menus, dialogs, cards, battle UI) | DOM HUD: `ctx.ui.set(slot, html)` with `data-action` buttons → `ctx.ui.take_actions()`; screens per mode (`Push` for overlays) |
| `localStorage` JSON save | `ctx.save.write/read` (versioned) |
| Area switching (interiors) | one `Screen` or scene per area (`ctx.begin_scene` / `unload_scene`) or streaming layers |
| Canvas2D card art / `toDataURL` | Canvas in the DOM as before, or render to texture (`create_render_texture`, `export_render_texture` → blob URL for `<img>`) |
| Photo mode (`snapPhoto`) | render texture + `export_render_texture`, or `bridge screenshot` for tests |
| `THREE.Fog`, background colour, sky | `renderer.settings` fog_*, `sky_mode: 1` (atmosphere) + `clouds`, or sky_zenith/sky_horizon (gradient), `ctx.load_skybox(url)` |
| Water plane | `Material::water(deep, shallow)` on `ctx.spawn_water(...)` (dense mesh for vertex waves) or `ctx.spawn_ocean(mat, y)` (~145k triangles; phones: `spawn_ocean_with(mat, y, OceanOpts {..})`) |
| Many colours, one material | `MeshBuilder::color(0xRRGGBB)` / `Vertex.color` (multiplied with the base colour) |
| Deferred work under a frame budget | `JobQueue<T>` + `FrameBudget::new(ms)`, `while let Some(j) = q.pop(&b) {..}` |
| Point/spot lights, day/night | `PointLight`, `SpotLight` (clustered, hundreds); `day_cycle` + `time_of_day` (+ `day_length_minutes`) drive sun, moon, sky and ambient; lamps follow `ctx.renderer.sky.night` |
| Sprite particles | `ctx.spawn_particles(name, pos, ParticleEmitter::preset(..))` |
| Rain `InstancedMesh` | `ParticlePreset::Rain`, or instanced entities (same mesh = one draw) |
| Quality presets / auto-downgrade | frame pacing + adaptive resolution (`render.auto_resolution`, `min_render_scale`), `shadow_cascades`, grass radius |
| Voice lines / mp3 | `ctx.audio.play_music` (streamed), `ctx.load_sound` + `play_named` |
| npm test for content | scripted tests (`tests/*.txt`), game commands for content checks |

Port in vertical slices: first the town walkable with the character controller and camera, then one
screen at a time (menus, cards, battles), each with a test script before moving on.

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
- **Visual checks:** `render '{"debug_view":1}'` = normals, `2` = shadow term, `0` = lit.
- **3D UI:** `call panels` gives every world-panel widget with its on-screen centre (`screen`, CSS px);
  click it for real with `input {x,y}` + `input {button:0}`, or directly with `panel_click {id, widget}`.
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
  `gpu_ms` per pass (cull, shadow, opaque, blend, hiz; WebGPU timestamp queries) and a `verdict`
  (smooth / gpu-bound / cpu-bound / stutter). `--profile` adds a CDP CPU profile (`top_self` functions).
  Live values: `stats.time.phases`, `stats.render.gpu_ms`. When the *user* sees low fps, ask them to press ` while
  it happens: debug mode writes a `"kind":"perf"` line per second (p50/p95/max, slow frames, phases, GPU) into the log.

## Code map

| Path | What |
|---|---|
| `crates/engine/src/app.rs` | `Game` trait, `Ctx`, frame loop (fixed step + rAF), input listeners |
| `crates/engine/src/renderer.rs` | WebGPU forward PBR, shadows, MSAA, instancing, mips, LOD pick, CPU/GPU culling switch |
| `crates/engine/src/gpu_cull.rs` | compute culling (frustum + Hi-Z occlusion), indirect args, counter readback |
| `crates/engine/src/panel.rs` | world panels: Canvas2D widgets -> texture on a quad, ray/UV hit test |
| `crates/engine/src/audio.rs` | Web Audio, 3D panner voices, per-voice effect chain (filters, echo, reverb, distortion, pan, detune) |
| `crates/engine/src/audio/music.rs` | streamed music (`<audio>` + range requests), crossfade, prefetch, music bus |
| `crates/engine/src/audio/synth.rs` | deterministic sfx synthesizer + 16 presets |
| `crates/engine/src/texgen.rs` | procedural seamless textures (16 presets, `noise` crate) |
| `crates/engine/src/jobs.rs` | job system: serial on stable, rayon Web Workers with the `threads` feature |
| `crates/engine/src/touch.rs` | multi-touch gestures: joystick, look, pinch, tap, hold, swipe |
| `crates/engine/src/mesh.rs` | primitives, `simplify_cluster` (auto LOD) |
| `crates/engine/src/shaders/*.wgsl` | `common.wgsl` is prepended to pbr/shadow; `cull`, `hiz`, `mipgen` are standalone |
| `crates/engine/src/debug.rs` | the AI API (`COMMANDS` table = `help` output) |
| `crates/engine/src/debugmode.rs` | ` debug mode: input + camera log -> dev server `/__debuglog` -> `logs/` |
| `crates/engine/src/physics.rs` | rapier3d wrapper, `Body` component, raycasts |
| `crates/engine/src/gltf_loader.rs` | `.glb` loading via fetch + createImageBitmap, per-URL cache |
| `crates/engine/src/ui.rs` | DOM HUD slots, `data-action` clicks, `WorldLabel` positioning |
| `games/demo/` | example game; `assets/` is copied to `dist/assets/` |
| `tools/asset-cook/` | native russimp/Assimp cooker: OBJ/FBX/DAE/... -> `.glb` |
| `tools/bridge.mjs` | CDP bridge CLI + MCP server |

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
- In `fixed_update` read held input (`input.key`), edge input (`key_pressed`) in `update`.

# .claude/ — start here (for agents)

This repo is an **AI-first** engine: you never guess what the game is doing, you ask it.
Everything an agent needs to work here lives in this folder.

| File | Use it when |
|---|---|
| `skills/engine-loop/SKILL.md` | **Always first.** Start server + Chrome, build, reload, read errors. |
| `skills/verify-gameplay/SKILL.md` | Proving a change works: pause, step, inject input, assert JSON; scripted test runs (`bridge run`). |
| `skills/new-game/SKILL.md` | Writing game code: `Game` trait, `Ctx` API, screens + scenes, save games, character controller + 3rd-person camera, animation, particles, atmosphere, render textures, world panels (3D UI), audio, LOD, job system, new `games/<name>` crate. |
| `skills/add-debug-command/SKILL.md` | Exposing new state/controls to agents (engine or game command). |
| `skills/cook-assets/SKILL.md` | Importing OBJ/FBX/DAE/... models (russimp cooker -> `.glb`). |
| `skills/procedural-assets/SKILL.md` | Textures and sounds from presets (no files), audio effects (echo, reverb, filters, pitch), live iteration. |
| `skills/debug-rendering/SKILL.md` | Black screen, WGSL errors, wrong lighting/shadows, missing models, GPU culling / LOD / mips. |
| `../HANDOFF.md` | Starting or porting a game in another repository (`tools/new-game.mjs`, `dev.mjs --game-dir`), three.js → wasmOnly mapping. |
| `agents/engine-verifier.md` | Subagent that independently checks a change in the running game. |
| `launch.json` | Preview config (`node tools/dev.mjs watch` on :8080). |

## 30-second orientation

- Runtime: Rust -> `wasm32-unknown-unknown` -> Chrome (WebGPU). No native, no WebGL, no other browsers.
- Engine: `crates/engine` (WebGPU renderer with GPU culling + indirect draws, physics=rapier, ECS=hecs,
  glTF, DOM-UI + clickable 3D panels, Web Audio, job system, AI debug API).
- Toolchains: stable pinned in `rust-toolchain.toml`; optional threads backend on a pinned nightly.
- Games: `games/<name>` (cdylib implementing `engine::Game`). Example: `games/demo` ("Tumble").
- Tools (Node ≥ 22, zero npm deps): `tools/dev.mjs` (build/serve/watch), `tools/bridge.mjs` (CDP CLI + MCP).
- The AI API: `window.engine.call(name, args)` -> JSON. From the shell: `node tools/bridge.mjs call <name> '<json>'`.
  MCP: server `wasmonly`, tools `engine_call`, `status`, `reload`, `screenshot`, `eval_js`, `launch_chrome`,
  `check`, `flicker`, `perf`, `run_script`.

## Golden rules

1. After every code change: wait for the watch rebuild JSON line, then `bridge reload` and read `warnings_and_errors`.
2. Verify with JSON (`game_state`, `entity`, `logs`), not by looking. Screenshots are the last check, not the first.
   A check you will want again belongs in a test script (`games/<g>/tests/*.txt`, run with `bridge run`).
3. Anything you had to work out by hand that an agent will need again → make it queryable
   (a debug command or a `debug_state` field) and note it here.
4. Keep this folder current when you add tools, commands or conventions.

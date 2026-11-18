---
name: engine-loop
description: Start, build, reload and inspect the wasmOnly engine in Chrome. Use before any change to engine or game code, and whenever you need to run, see or check the game.
---

# Engine dev loop

## 1. Is it already running?

```bash
node tools/bridge.mjs status
```

- `engine.ready: true` → Chrome and the game are up; skip to step 3.
- Connection error on :9222 → Chrome is not running (step 2b).
- Page not reachable → the dev server is not running (step 2a).

## 2. Start what is missing

a) Dev server (keep it running in the background; it rebuilds and live-reloads on change in
`crates/`, `games/`, `web/`):

```bash
node tools/dev.mjs watch            # optional: --game <name> --port 8080
```

Each build ends with one JSON line: `{"build":"ok",...}` or `{"build":"error",...}`.
Compiler errors are normal cargo output above that line.

b) Chrome with remote debugging (own profile in `.chrome-debug-profile`, background throttling off,
**muted** so you never play sound at the user; `--sound` only if they ask to listen):

```bash
node tools/bridge.mjs launch        # --headless for CI
```

First time on a fresh clone: `node tools/make-sample-assets.mjs` (creates demo assets).

## 3. After every code change

1. Wait for the new `{"build":...}` line in the watch output. On `error`: fix, repeat.
2. `node tools/bridge.mjs reload` → returns `{ready, ms, warnings_and_errors: [...]}`.
   A non-empty `warnings_and_errors` means the change is not done.
3. Check `node tools/bridge.mjs call stats` (fps, entity/body counts, draw calls).
4. Prove the behaviour with the verify-gameplay skill.
5. Before you call it done: `node tools/bridge.mjs check` (or run the watcher as `watch --check`) — exit 0,
   `zfight.pairs` 0, no errors. Use `check --flicker` after visual/geometry work.
6. After render/shader or heavy gameplay changes: `node tools/bridge.mjs perf` — verdict must say `smooth`.

## Command cheat sheet

| Need | Command |
|---|---|
| All commands incl. game ones | `call help` |
| Game snapshot | `call game_state` |
| Find entities | `call entities '{"name":"rock","with":"body"}'` |
| One entity | `call entity '{"id":16}'` |
| New log lines only | `call logs '{"since":<stats.log_seq>,"level":"warn"}'` |
| Run JS in page | `eval 'document.title'` |
| Screenshot | `screenshot out.png` (write to scratch, not the repo) |
| Scripted test | `run games/demo/tests/smoke.txt` or `run --commands "key Space; step 30; expect ..."` |
| Screens / scenes | `call screens`, `call scenes` |

The same is available over MCP (server `wasmonly`): `engine_call {name,args}`, `status`, `reload`,
`screenshot`, `eval_js`, `launch_chrome`, `check`, `flicker`, `perf`, `run_script`.

## Toolchains and the threads build

- Stable is pinned in `rust-toolchain.toml` (1.99.0); rustup picks it automatically.
- The threads backend (Web Workers + SharedArrayBuffer + Atomics) needs the nightly pinned in `Cargo.toml`
  (`[workspace.metadata.wasmonly] threads-toolchain`):
  `node tools/dev.mjs build --threads` (or `watch --threads`). The first build takes ~3-4 min (build-std).
  The build JSON shows `"threads":true,"toolchain":"nightly-..."`; `call stats` -> `jobs.backend: "threads"`.
- A plain `watch` rebuild replaces a threads build in `dist/` with the stable one.

## Slow networks / streaming

```bash
curl -X POST "http://localhost:8080/__throttle?kbps=48&match=assets/music"   # only matching paths
node tools/bridge.mjs reload            # engine must still be ready in ~1 s
node tools/bridge.mjs call music        # state "buffering" -> "playing" as data arrives
curl -X POST "http://localhost:8080/__throttle?kbps=0"                       # off (running downloads speed up)
```

The dev server answers HTTP range requests (206), like a real static host.

## Gotchas

- `wasm-bindgen` crate version must equal `wasm-bindgen --version` (pinned `=0.2.129`). `dev.mjs` checks this.
- Dev wasm is ~18 MB (DWARF for engine + game). Use `build --release` for size/perf numbers.
- The debug Chrome runs with `--autoplay-policy=no-user-gesture-required`, so `audio.state` is `running`
  without clicks. In a normal browser it stays `suspended` until the first key/pointer event.
- Audio pauses while the tab is unfocused (`audio.focused`, `audio.state: "suspended"`); one-shots triggered
  then are dropped and appear in `audio.history` with `suppressed: true`. If the debug Chrome window is not
  focused, either `audio_set {"pause_on_blur":false}` for audio tests or simulate focus with
  `eval "window.dispatchEvent(new FocusEvent('focus'))"`.
- The game camera may be driven by game code each frame; `call camera '{"pos":..,"look_at":..}'`
  can be overridden by the game. Pause first or use the game's own commands.

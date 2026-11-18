---
name: verify-gameplay
description: Prove a gameplay, physics or engine change works by deterministic stepping, input injection and JSON assertions in the running game. Use after implementing anything that changes behaviour.
---

# Verify behaviour, don't eyeball it

The simulation is a fixed 1/60 s step. While paused, `step` runs exactly N steps synchronously,
so the same inputs give the same result. Build every check as: **set up → act → step → assert**.

## Recipe

```bash
node tools/bridge.mjs call pause
node tools/bridge.mjs call seed '{"value":1}'              # deterministic ctx.rng
node tools/bridge.mjs call reset                           # game command (demo); use the game's own reset
node tools/bridge.mjs call stats                           # note result.log_seq -> SEQ
node tools/bridge.mjs call game_state                      # before
node tools/bridge.mjs call input '{"key":"KeyW","down":true}'
node tools/bridge.mjs call step '{"n":60}'                 # exactly 1 s of sim
node tools/bridge.mjs call input '{"release_all":true}'
node tools/bridge.mjs call game_state                      # after -> compare
node tools/bridge.mjs call logs '{"since":SEQ,"level":"warn"}'   # must be empty
node tools/bridge.mjs call resume
```

## Input injection (same code path as real input)

| Input | args |
|---|---|
| Key held | `{"key":"KeyW","down":true}` … later `{"key":"KeyW","down":false}` |
| Mouse button | `{"button":0,"down":true}` (0 left, 1 middle, 2 right) |
| Mouse motion | `{"dx":10,"dy":0}` |
| HUD button (`data-action`) | `{"action":"reset"}` |
| Touch finger | `{"touch":{"id":1,"x":200,"y":600,"phase":"start"}}` then `"move"` / `"end"` |
| Everything up | `{"release_all":true}` |

Edge inputs (`key_pressed`) are consumed in `update`, held inputs in `fixed_update`. After a
key-down, step at least 1 so `update` sees the edge.

## Useful assertions

- Position/velocity: `call entity '{"id":N}'` → `local.pos`, `local.rot_deg`, `body.linvel`, `body.sleeping`, `children`.
- Contacts: `entity` -> `body.grounded`, `body.contacts[]` (`other`, `normal` pointing towards the body, `depth`).
- Physics: `call raycast '{"origin":[0,10,0],"dir":[0,-1,0]}'` → `{hit, id, distance, point, normal}`.
- What's on screen: `call pick '{"x":640,"y":360}'` (CSS px).
- HUD: `call ui` → slot HTML.
- Assets: `call assets` → `models_failed` must be `{}`, `models_pending` empty after load.

## World panels and audio

```bash
node tools/bridge.mjs call panels                         # widgets + on-screen centres ("screen": [x,y])
node tools/bridge.mjs call input '{"x":880,"y":291}'      # pointer onto the widget (real path)
# let ~2 animation frames pass: eval "new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))"
node tools/bridge.mjs call input '{"button":0,"down":true}'   # click; then down:false
node tools/bridge.mjs call panel_click '{"id":0,"widget":"slowmo"}'   # shortcut, same click handler
node tools/bridge.mjs call audio                          # history: [{sound, step, pos, volume}]
```

Panel hover/clicks are evaluated per *rendered frame*, not per `step`: let real frames pass (as above)
when testing the pointer path. `audio.history[].step` tells you in which sim step a sound started.

## Demo game ("Tumble") commands

`reset`, `teleport {pos:[x,y,z]}`, `shoot_at {target:[x,y,z]}`; `game_state` has `score`, `total`,
`player_pos`, `slowmo`. Terminal panel buttons: `reset`, `shadows`, `slowmo`, `sound`. Other games list theirs under `call help`.

## Reproducing what the user did

When the user reports a bug ("I pressed X and Y happened"), ask them to reproduce it with debug mode on
(press ` / German ^ before, again after). Then read the newest `logs/debug-*.log`:

```bash
ls -t logs/ | head -1
```

Each line is JSON: `{"kind":"key","code":"KeyW","down":true,"source":"dom","step":615,"t":10894.6}`,
`{"kind":"camera","pos":[x,y,z],"forward":[..],"yaw_deg":..,"pitch_deg":..}` (once per second),
`{"kind":"debug_mode","on":true|false}`. Replay the key sequence with `input` + `step` (use the `step`
deltas between lines) to reproduce deterministically. AI-injected input is logged too (`"source":"api"`).

## Done means

- The asserted JSON changed as expected,
- `call zfight` reports `"pairs": 0` if you added or moved geometry,
- `logs` since the action has no `warn`/`error`,
- the game is resumed (`call resume`) so the user sees it running.

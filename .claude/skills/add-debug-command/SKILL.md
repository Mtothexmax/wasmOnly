---
name: add-debug-command
description: Expose new runtime state or controls to agents through window.engine.call - either an engine-wide command in debug.rs or a game-specific command. Use when you had to inspect something by hand, or a test needs a control that does not exist yet.
---

# Adding a debug command

Rule of thumb: if you needed a screenshot, a guess or `eval` to learn something, make it a command.

## Game-specific (preferred for gameplay)

In the game's `impl Game`:

```rust
fn debug_commands(&self) -> Vec<(&'static str, &'static str)> {
    vec![("spawn_enemy", "Spawn an enemy. args: {pos: [x,y,z]}")]   // shows up in `call help`
}

fn debug_command(&mut self, ctx: &mut Ctx, name: &str, args: &Value) -> Option<Result<Value, String>> {
    Some(match name {
        "spawn_enemy" => { /* parse args, act */ Ok(json!({"id": e.id()})) }
        _ => return None,            // None = not mine -> "unknown command"
    })
}
```

For read-only state prefer adding a field to `debug_state` (returned by `game_state`).

## Engine-wide

`crates/engine/src/debug.rs`:

1. Add `("name", "description incl. args: {...}")` to `COMMANDS` (this *is* the `help` output).
2. Add the arm to the `match` that dispatches commands.
3. Return `Result<Value, String>`; errors come back as `{ok:false, error}`.

## Conventions

- Args are one JSON object; vectors are `[x,y,z]`, colors `'#rrggbb'`, rotations in degrees.
- Describe every arg and its default in the description string.
- Return ids and the resulting state so the caller can assert without a second call.
- Commands must work while paused; anything that should advance time is `step`'s job.

## Verify

```bash
node tools/bridge.mjs reload
node tools/bridge.mjs call help                 # new command listed?
node tools/bridge.mjs call <name> '{...}'       # returns ok:true with the expected payload
```

Then add it to the command list in the verify-gameplay skill if it is generally useful.

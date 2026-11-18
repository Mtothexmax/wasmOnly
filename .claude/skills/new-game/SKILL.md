---
name: new-game
description: Write game code against the wasmOnly engine - the Game trait, Ctx helpers, spawning meshes, physics bodies, glTF models, HUD and labels - or create a new games/<name> crate. Use when adding gameplay or starting a new game.
---

# Writing game code

Reference implementation: `games/demo/src/lib.rs`. Engine API: `crates/engine/src/app.rs`.

## New game crate

1. Copy `games/demo/Cargo.toml` to `games/<name>/Cargo.toml`, change `name`. It must stay
   `crate-type = ["cdylib"]`.
2. Add `"games/<name>"` to `members` in the root `Cargo.toml`.
3. Static files go in `games/<name>/assets/` (copied to `dist/assets/`); source models in `assets-src/`.
4. Run: `node tools/dev.mjs watch --game <name>`.

## Skeleton

```rust
use engine::prelude::*;
use engine::physics::rapier::prelude::*;

#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() { engine::run(MyGame::default()); }

#[derive(Default)]
struct MyGame { player: Option<Entity>, score: u32 }

impl Game for MyGame {
    fn init(&mut self, ctx: &mut Ctx) { /* spawn camera, world, UI */ }
    fn fixed_update(&mut self, ctx: &mut Ctx) { /* 1/60 s: forces, held input (ctx.input.key) */ }
    fn update(&mut self, ctx: &mut Ctx, dt: f32) { /* per frame: camera, edges (key_pressed), UI actions */ }
    fn debug_commands(&self) -> Vec<(&'static str, &'static str)> { vec![] }
    fn debug_command(&mut self, ctx: &mut Ctx, name: &str, args: &Value) -> Option<Result<Value, String>> { None }
    fn debug_state(&self, ctx: &Ctx) -> Value { json!({"score": self.score}) }
}
```

**Always implement `debug_state`** with everything an agent needs to assert progress (score,
player position, phase, counts). This is what `call game_state` returns.

## Ctx cheat sheet

| Field / method | What |
|---|---|
| `ctx.world` | hecs `World`. Query `(Entity, &T)` to get ids. |
| `spawn_cube(name, pos, size, 0xRRGGBB)`, `spawn_sphere(name, pos, r, color)` | primitives |
| `spawn_mesh(name, mesh, material, Transform)` | with `ctx.renderer.builtin.{cube,sphere,plane}` + `renderer.add_material(Material{..})` |
| `load_gltf("assets/x.glb", Transform)` | async load; returns root entity immediately, children appear when ready |
| `physics.add(&mut world, e, RigidBodyBuilder::dynamic(), ColliderBuilder::ball(r))` | attach a body; `fixed()` for static |
| `physics.body_mut(&world, e)` | rapier `RigidBody` (impulses, velocity) |
| `physics.raycast(origin, dir, max)` | `Option<RayHit>` with entity |
| `physics.grounded(&world, e, 60.0)` / `physics.contacts(&world, e)` | standing check from real contact normals (works on edges and in gaps; do **not** use a single ray down) |
| `despawn(e)`, `find_by_name("x")`, `screen_ray(css_px)` | |
| `ctx.input` | `key("KeyW")`, `key_pressed`, `key_released`, `button(MouseButton::Left)`, `button_pressed`, `pointer_delta`, `wheel`, `move_axis()` (WASD/arrows/gamepad) |
| `ctx.ui.set(slot, html)` / `ctx.ui.take_actions()` | DOM HUD; `<button data-action='x'>` → action `"x"` |
| `ctx.rng` | seeded RNG (`range(lo,hi)`); use it, not `Math.random`, so `seed` makes runs repeatable |
| `ctx.time` | `fixed_dt`, `step`, `sim_time`, `frame`, `fps`, `time_scale` |

Components (in `engine::scene`): `Transform`, `Parent`, `Name`, `MeshRenderer`, `Camera`,
`PointLight {color, intensity, range}` (max 16 active), `WorldLabel` (DOM label anchored in 3D).

## Transparency

```rust
// glass / water / glow: blended, sorted back to front, no shadow
let water = ctx.renderer.add_material(Material { texture: tex, ..Material::color(0xbfe6ff).transparent(0.6) });
// foliage / fences: alpha test + double-sided + cut-out shadows (texture needs alpha, e.g. TexPreset::Blades)
let grass = ctx.renderer.add_material(Material { texture: blades, ..Material::color(0xffffff).cutout(0.5) });
```

Prefer `cutout` over `transparent` whenever edges may be hard: it batches, GPU-culls and casts shadows;
blended instances are drawn one sorted run at a time. glTF `alphaMode`/`doubleSided` are imported.

## Touch (Android Chrome)

Gestures are in `ctx.input.touch`: the joystick (left 40 % of the screen) is already part of
`input.move_axis()`; use `touch.look_delta` for the camera, `touch.pinch` (log zoom per frame) for zoom,
`touch.taps` (CSS px, taps on world panels are already consumed), `touch.swipes`, `touch.holds_started`.
Show touch buttons only when `touch.touch_device` is true (DOM HUD buttons with `data-action` work on
touch). See the demo's `update` for a complete mapping.

## World panels (clickable 2D UI on a 3D surface)

```rust
let panel = Panel::new(512, 320)                       // texture pixels; keep the quad's aspect ratio
    .label("title", [24.0, 20.0, 460.0, 40.0], "Shop", 32.0)
    .bar("hp", [24.0, 80.0, 460.0, 20.0], 0.7, "#34d399")
    .button("buy", [24.0, 220.0, 220.0, 70.0], "Kaufen", "buy");   // action "buy"
let e = ctx.spawn_panel("shop", panel, Transform::from_xyz(0.0, 2.0, 0.0).with_scale(Vec3::new(2.4, 1.5, 1.0)));
// later (redrawn automatically when changed):
if let Some(p) = ctx.panel_mut(e) { p.set_text("title", "Shop (3)"); p.set_value("hp", 0.4); }
```

Quad = XY plane facing +Z (rotate the entity to face the player); invisible from behind. Clicks go to
`ctx.ui.take_actions()` like HUD buttons and are consumed (`input.button_pressed(Left)` stays false);
`input.pointer_over_panel` tells you the pointer is on a panel. Add a thin fixed collider if the panel
should block shots; colliders in front of a panel block its clicks.

## Audio

```rust
ctx.audio.add_samples("zap", &synth::zap(), synth::RATE);      // procedural: zap, thud, jump, click, chime, sweep(..)
let music = ctx.load_sound("assets/music.ogg");               // async; any format Chrome decodes
ctx.audio.play_named("zap", Play::new().volume(0.5).at(pos)); // 3D at a point
ctx.audio.play(music, Play::new().looped().volume(0.3));        // 2D; returns None until loaded
ctx.audio.play_named("jump", Play::new().follow(player));      // 3D, follows an entity
```

Music is streamed, never decoded up front, never awaited — the game runs while it downloads:

```rust
ctx.audio.play_music("assets/music/theme.mp3", MusicOpts { fade: 3.0, volume: 0.5, looped: true });
ctx.audio.prefetch_music("assets/music/cave.mp3");   // buffer the next area's track early
ctx.audio.play_music("assets/music/cave.mp3", MusicOpts::default()); // crossfade (default 1.5 s)
ctx.audio.set_music_volume(0.6); ctx.audio.set_sfx_volume(1.0);
```

Use `load_sound`/`add_samples` (decoded into memory) only for short effects; anything longer than a few
seconds is music and goes through `play_music`.

The listener follows the camera automatically. Audio suspends while the tab is unfocused and resumes where it
stopped (one-shots fired meanwhile are dropped); a music-player style game can opt out with
`ctx.audio.set_pause_on_blur(false)`. Use `ctx.rng` for pitch variation so runs stay reproducible.
Sound presets, the synth recipe fields and effects (`.echo() .reverb() .lowpass() .distort() .detune() .pan()`)
and procedural textures (`ctx.generate_texture(&TexGen::new(TexPreset::Grass))`) are in the
procedural-assets skill.

## LOD and many objects

- `spawn_sphere` already attaches `renderer.builtin.sphere_lod`; glTF meshes with >= 256 triangles get
  automatic LODs. For your own meshes: `let (mesh, lod) = ctx.renderer.add_mesh_with_lods(name, &data, &[0.5, 0.25]);`
  and insert `lod` next to the `MeshRenderer`. Distances are in mesh-local units (camera distance / scale).
- Thousands of identical objects are fine: same mesh + same texture = one indirect draw. Different
  materials (colors) do not split batches; different textures do.

## Heavy per-item work

Use `engine::jobs::par_map(&items, |x| ...)` / `par_for_each_mut`. Serial on the stable build, parallel
on Web Workers with `--threads`, identical results. Never use rayon or std threads directly.

## Rules

- Only `.glb` at runtime. Other formats: cook them (cook-assets skill).
- Deterministic gameplay goes in `fixed_update`; never read wall-clock time there.
- One code path per action: if a debug command does what the player can do (shoot, click), call the
  same function (see `Tumble::shoot_from`), so testing the command tests the real behaviour.
- Log with `log_info!/log_warn!/log_error!` — agents read them via `call logs`.
- New controls/state an agent will need → add a game debug command (add-debug-command skill).

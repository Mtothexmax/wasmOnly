---
name: procedural-assets
description: Generate textures (noise, marble, wood, stone, grass, lava, bricks, ...) and sound effects (coin, laser, explosion, wind loop, ...) from presets, tweak them live through the AI API, and apply playback effects (filters, echo, reverb, distortion, pitch/detune, pan). Use when a game needs placeholder or final art/audio without asset files.
---

# Procedural textures and sounds

No asset files needed. Everything is deterministic: same preset + seed + parameters = same pixels / samples.

## Textures (`engine::texgen`, `noise` crate)

Presets: `noise clouds marble wood stone grass sand dirt lava water bricks tiles metal rust camo checker`.
All tile seamlessly (noise sampled on a 4D torus) and get mips + anisotropic filtering.

```rust
let tex = ctx.generate_texture(&TexGen::new(TexPreset::Stone).size(256).seed(3).scale(1.5).colors(0x3a3a40, 0xa0a0a8));
let mat = ctx.renderer.add_material(Material { texture: tex, ..Material::color(0xffffff) });
```

To repeat a texture across a big surface, scale the mesh UVs (see `tiled_cube` in `games/demo/src/lib.rs`).

Iterate live (no rebuild):

```bash
node tools/bridge.mjs call texgen '{}'                                   # list presets
node tools/bridge.mjs call spawn '{"shape":"cube","pos":[0,2,4],"size":2,"body":"none","name":"swatch"}'
node tools/bridge.mjs call texgen '{"preset":"marble","size":256,"seed":5,"scale":2,"colors":["#202830","#e8e4dc"],"apply_to":ID}'
node tools/bridge.mjs screenshot <scratch>/swatch.png
```

The result has `mean_color` (a quick check that the colors are what you meant), `generate_ms`, the texture id and
the `recipe`. When happy, copy the recipe into Rust (`TexGen::new(..).size(..).seed(..)...`).

## Sounds (`engine::audio::synth`)

Presets: `coin laser explosion powerup hit jump blip click select alarm thud chime bell` and the seamless loops
`wind rain engine`. A preset is an `Sfx` recipe:

| Field | Shapes |
|---|---|
| `wave` | `sine square saw triangle noise pink brown` (pink = wind/surf, brown = rumble) |
| `freq`, `freq_end` | pitch and slide (Hz) |
| `vibrato_depth` (semitones), `vibrato_rate` (Hz), `arp_mult`, `arp_time` | pitch movement |
| `partials: [[mult, amp], ...]` | **timbre**: harmonic (2, 3, 4) = organ, inharmonic (2.76, 5.4, 8.93) = bell |
| `duty`, `noise_mix`, `bitcrush` | square width, grit, retro quantisation |
| `attack decay sustain sustain_time release` | ADSR envelope |
| `lowpass`, `highpass` | baked one-pole filters (Hz) |

```rust
ctx.audio.add_preset(SfxPreset::Laser, 0);                                  // name "laser"
ctx.audio.add_sfx("boom2", &Sfx::preset(SfxPreset::Explosion).mutate(7, 0.6)); // variation
ctx.audio.add_sfx("organ", &Sfx { wave: Wave::Sine, freq: 220.0, freq_end: 220.0, partials: vec![(2.0, 0.5), (3.0, 0.3)], ..Sfx::default() });
```

### Playback effects (Web Audio nodes, per voice)

```rust
ctx.audio.play_named("laser", Play::new().at(pos).echo(0.11, 0.3));        // delay s, feedback
ctx.audio.play_named("thud", Play::new().reverb(1.2).lowpass(2200.0));    // hall, muffled
ctx.audio.play_named("radio", Play::new().bandpass(1800.0, 4.0).distort(8.0));
ctx.audio.play_named("jump", Play::new().pitch(1.2).detune(-50.0).pan(-0.5));
ctx.audio.play_named("wind", Play::new().looped().volume(0.2).lowpass(900.0));
```

`pitch` = playback rate (also changes length), `detune` = cents. Voices with echo/reverb stay alive until
their tail has rung out, then the graph is disconnected.

Iterate live:

```bash
node tools/bridge.mjs call sfx '{}'                                                    # list presets
node tools/bridge.mjs call sfx '{"preset":"bell","name":"door","seed":4,"overrides":{"freq":330,"release":2.5}}'
node tools/bridge.mjs call audio_play '{"sound":"door","reverb":[2.5,0.5],"echo":[0.3,0.4],"lowpass":3000}'
node tools/bridge.mjs call audio      # voices with their fx, history with step/pitch/fx
```

`sfx` returns `duration`, `peak`, `rms` (0 rms = silent; peak 1.0 = clipping: lower `volume`) and the full
`recipe` to copy into Rust. You cannot hear the result: judge by recipe + levels, and say so to the user.

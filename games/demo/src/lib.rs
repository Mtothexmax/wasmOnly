//! Demo: "Tumble" — roll a ball, knock the crate tower off the island.
//!
//! Exercises every engine system: screens + scenes (title / play / pause), save games, PBR +
//! shadows, sky, fog, bloom, depth darkening, particles, point lights, glTF models (static,
//! dynamic, skinned + morph-animated), render-to-texture (title card portrait), rapier physics,
//! character controller + third-person camera, input (keyboard, mouse, wheel, gamepad, touch),
//! DOM HUD with actions, world panels, world labels, audio, and game-specific AI debug commands.

use engine::physics::rapier::prelude::{ColliderBuilder, RigidBodyBuilder};
use engine::prelude::*;
use engine::renderer::Material;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use wasm_bindgen::prelude::*;

mod world;

#[wasm_bindgen(start)]
pub fn main() {
    engine::run(
        ScreenStack::new(Data::default(), Box::new(Title::default()))
            .register("title", || Box::new(Title::default()))
            .register("play", || Box::new(PlayScreen::default()))
            .register("pause", || Box::new(Pause))
            .register("world", || Box::new(world::WorldScreen::default())),
    );
}

const PLAYER_SPAWN: Vec3 = Vec3::new(0.0, 1.0, 9.0);
/// Pond centre and inner radius.
const POND: Vec3 = Vec3::new(-6.5, 0.0, 7.5);
const POND_R: f32 = 2.1;
const ISLAND_HALF: f32 = 14.0;
const CAMPFIRE: Vec3 = Vec3::new(-8.0, 0.0, -1.5);
/// A jump press stays valid this many fixed steps (pressed slightly before landing still jumps).
const JUMP_BUFFER_STEPS: u32 = 8;
/// After leaving the ground a jump is still allowed for this many steps ("coyote time").
const COYOTE_STEPS: u32 = 6;
/// After a jump, ground contact is ignored this long (no double impulse while still touching).
const JUMP_COOLDOWN_STEPS: u32 = 12;
/// Portrait subjects live on this render layer; the main cameras skip it.
const PORTRAIT_LAYER: u8 = 1;

// ---------------------------------------------------------------------------------------------
// Shared data (all screens) + save game

/// Saved in localStorage slot "progress" (namespace "tumble").
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Progress {
    best: u32,
    wins: u32,
    plays: u32,
}

#[derive(Default)]
struct Data {
    progress: Progress,
    started: bool,
    /// Title card portrait (created once, its camera is re-attached on every title visit).
    portrait: Option<(RenderTextureId, Option<TextureId>)>,
}

impl Data {
    /// Once per page: save namespace, saved progress, sounds, music.
    fn ensure(&mut self, ctx: &mut Ctx) {
        if self.started {
            return;
        }
        self.started = true;
        ctx.save.set_namespace("tumble", 1);
        self.progress = ctx.save.read("progress").unwrap_or_default();
        // Sounds from synth presets (engine::audio::synth) — no asset files. Effects are applied
        // at playback (echo, reverb, filters) so one recipe serves several moods.
        for p in [SfxPreset::Laser, SfxPreset::Jump, SfxPreset::Click, SfxPreset::Powerup, SfxPreset::Chime, SfxPreset::Wind] {
            ctx.audio.add_preset(p, 0);
        }
        ctx.audio.add_sfx("thud", &Sfx::preset(SfxPreset::Thud).mutate(3, 0.5));
        ctx.audio.play_named("wind", Play::new().looped().volume(0.18).lowpass(900.0)); // ambience
        // Music is streamed (<audio> + range requests): the game runs at once, the track starts
        // when enough is buffered. Nothing waits for the download.
        ctx.audio.play_music("assets/music/where_the_banners_fall.mp3", MusicOpts { fade: 3.0, volume: 0.45, looped: true });
    }

    fn save(&self, ctx: &Ctx) {
        if let Err(e) = ctx.save.write("progress", &self.progress) {
            engine::log_warn!("save failed: {e}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Title screen: animated critter, particles, a card with a render-to-texture portrait.

#[derive(Default)]
struct Title {
    critter: Option<Entity>,
    portrait_model: Option<Entity>,
    /// 0 waiting for the model, 1 rendered (export next), 2 exporting, 3 shown
    portrait_state: u8,
    frames: u32,
}

impl Title {
    fn hud(&self, ctx: &mut Ctx, data: &Data) {
        let url = data.portrait.and_then(|(rt, _)| ctx.renderer.render_texture_url(rt));
        let img = url.map(|u| format!("<img src='{u}' alt='critter'>")).unwrap_or_else(|| "<img alt=''>".into());
        let p = &data.progress;
        ctx.ui.set(
            "menu",
            &format!(
                "<div class='panel menu'><h1>Tumble</h1><div>Push every crate off the island</div>\
                 <div style='margin:14px 0 4px'><div class='card'>{img}<div class='name'>Critter</div>\
                 <div class='stats'>Skinning · Morph · Clips</div></div></div>\
                 <div class='stats'>Best: {} · Wins: {} · Rounds: {}</div>\
                 <button data-action='play'>Play (Enter)</button>                 <button data-action='world'>Open world (O)</button></div>",
                p.best, p.wins, p.plays
            ),
        );
    }
}

impl Screen<Data> for Title {
    fn name(&self) -> &'static str {
        "title"
    }

    fn enter(&mut self, ctx: &mut Ctx, data: &mut Data) {
        data.ensure(ctx);
        ctx.world.spawn((
            Name("title-camera".into()),
            Transform::from_xyz(0.0, 1.5, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
            Camera { layers: !(1u32 << PORTRAIT_LAYER), ..Default::default() },
        ));
        let stone = ctx.renderer.add_material(Material::color(0x59606b).rough(0.8));
        ctx.spawn_mesh("pedestal", ctx.renderer.builtin.cube, stone, Transform::from_xyz(0.0, -0.2, 0.0).with_scale(Vec3::new(3.0, 0.4, 3.0)));
        let ground = ctx.renderer.add_material(Material::color(0x3f6b3a).rough(1.0));
        ctx.spawn_mesh("title-ground", ctx.renderer.builtin.cube, ground, Transform::from_xyz(0.0, -0.6, 0.0).with_scale(Vec3::new(60.0, 0.4, 60.0)));
        // left of the centred menu
        self.critter = Some(ctx.load_gltf("assets/critter.glb", Transform::from_xyz(-2.1, 0.0, 0.6).with_rotation(Quat::from_rotation_y(0.7))));
        let lamp = ctx.world.spawn((Transform::from_xyz(1.5, 2.2, 1.5), PointLight { color: Vec3::new(1.0, 0.7, 0.4), intensity: 18.0, range: 8.0 }));
        let _ = lamp;
        ctx.spawn_particles("title-fireflies", Vec3::new(0.0, 1.2, 0.0), ParticleEmitter::preset(ParticlePreset::Fireflies));
        ctx.spawn_particles("title-magic", Vec3::new(0.0, 0.1, 0.0), ParticleEmitter { rate: 12.0, area: Vec3::new(1.2, 0.0, 1.2), ..ParticleEmitter::preset(ParticlePreset::Magic) });

        // Card portrait: a second critter far away on its own render layer, filmed by a
        // render-texture camera (transparent background), exported as PNG for the DOM card.
        let pm = ctx.load_gltf("assets/critter.glb", Transform::from_xyz(50.0, 0.0, 0.0).with_rotation(Quat::from_rotation_y(-0.5)));
        ctx.set_layer(pm, PORTRAIT_LAYER);
        self.portrait_model = Some(pm);
        let (rt, tex) = *data.portrait.get_or_insert_with(|| ctx.create_render_texture(304, 380, RenderTextureOpts::portrait()));
        let _ = tex;
        let cam = ctx.world.spawn((
            Name("portrait-camera".into()),
            Transform::from_xyz(50.0, 1.25, 3.4).looking_at(Vec3::new(50.0, 1.05, 0.0), Vec3::Y),
            Camera { fov_y_deg: 42.0, layers: 1 << PORTRAIT_LAYER, ..Default::default() },
        ));
        let lamp2 = ctx.world.spawn((Transform::from_xyz(51.5, 2.0, 2.0), PointLight { color: Vec3::new(0.6, 0.7, 1.0), intensity: 14.0, range: 6.0 }));
        let _ = lamp2;
        ctx.set_render_texture_camera(rt, cam);
        self.portrait_state = 0;
        self.hud(ctx, data);
    }

    fn exit(&mut self, ctx: &mut Ctx, _data: &mut Data) {
        ctx.ui.set("menu", "");
    }

    fn update(&mut self, ctx: &mut Ctx, data: &mut Data, _dt: f32) -> Trans<Data> {
        self.frames += 1;
        if let Some(c) = self.critter {
            ctx.play_animation(c, "idle", PlayOpts::looped());
        }
        // portrait: wait for the model, pose it, render once, export to PNG, show in the card
        if let (Some(pm), Some((rt, _))) = (self.portrait_model, data.portrait) {
            match self.portrait_state {
                0 if ctx.world.get::<&Animator>(pm).is_ok() => {
                    ctx.play_animation(pm, "wave", PlayOpts::looped().fade(0.0));
                    ctx.refresh_render_texture(rt);
                    self.portrait_state = 1;
                    self.frames = 0;
                }
                1 if self.frames > 20 => {
                    ctx.refresh_render_texture(rt); // pose from the running clip
                    ctx.export_render_texture(rt);
                    self.portrait_state = 2;
                }
                2 if ctx.renderer.render_texture_url(rt).is_some() => {
                    self.portrait_state = 3;
                    self.hud(ctx, data);
                }
                _ => {}
            }
        }
        let mut go = ctx.input.key_pressed("Enter") || ctx.input.key_pressed("NumpadEnter");
        let mut world = ctx.input.key_pressed("KeyO");
        for a in ctx.ui.take_actions() {
            ctx.audio.play_named("click", Play::new().volume(0.5));
            go |= a == "play";
            world |= a == "world";
        }
        if world {
            return Trans::Switch(Box::new(world::WorldScreen::default()));
        }
        ctx.ui.set("fps", &format!("<div class='panel fps'>{:.0} fps · {:.1} ms</div>", ctx.time.fps, ctx.time.frame_ms));
        if go {
            return Trans::Switch(Box::new(PlayScreen::default()));
        }
        Trans::None
    }

    fn debug_state(&self, _ctx: &Ctx, data: &Data) -> Value {
        json!({"progress": data.progress, "portrait_state": self.portrait_state})
    }
}

// ---------------------------------------------------------------------------------------------
// Pause overlay

struct Pause;

impl Screen<Data> for Pause {
    fn name(&self) -> &'static str {
        "pause"
    }
    fn enter(&mut self, ctx: &mut Ctx, _data: &mut Data) {
        ctx.exit_pointer_lock();
        ctx.ui.set(
            "menu",
            "<div class='panel menu'><h1>Pause</h1><button data-action='resume'>Resume (P)</button>\
             <button data-action='menu'>Main menu</button></div>",
        );
    }
    fn exit(&mut self, ctx: &mut Ctx, _data: &mut Data) {
        ctx.ui.set("menu", "");
    }
    fn update(&mut self, ctx: &mut Ctx, _data: &mut Data, _dt: f32) -> Trans<Data> {
        let mut t = Trans::None;
        if ctx.input.key_pressed("KeyP") || ctx.input.key_pressed("Escape") {
            t = Trans::Pop;
        }
        for a in ctx.ui.take_actions() {
            ctx.audio.play_named("click", Play::new().volume(0.5));
            match a.as_str() {
                "resume" => t = Trans::Pop,
                "menu" => t = Trans::Reset(Box::new(Title::default())),
                _ => {}
            }
        }
        t
    }
}

// ---------------------------------------------------------------------------------------------
// Play screen

#[derive(Default)]
struct PlayScreen {
    player: Option<Entity>,
    camera: Option<Entity>,
    /// Character-controller figure (critter model on a capsule), toggled with C.
    walker: Option<Entity>,
    walker_model: Option<Entity>,
    walker_mode: bool,
    targets: Vec<(Entity, bool)>,
    shots: VecDeque<Entity>,
    score: u32,
    total: u32,
    falls: u32,
    jump_buffer: u32,
    coyote: u32,
    jump_cooldown: u32,
    grounded: bool,
    jumps: u32,
    /// World panel (clickable 3D UI) next to the spawn.
    terminal: Option<Entity>,
    fire_light: Option<Entity>,
    slowmo: bool,
    won: bool,
}

impl PlayScreen {
    fn player_pos(&self, ctx: &Ctx) -> Vec3 {
        self.player
            .and_then(|p| ctx.world.get::<&Transform>(p).ok().map(|t| t.translation))
            .unwrap_or(PLAYER_SPAWN)
    }

    fn camera_yaw(&self, ctx: &Ctx) -> f32 {
        self.camera.and_then(|c| ctx.world.get::<&ThirdPersonCamera>(c).ok().map(|c| c.yaw)).unwrap_or(0.0)
    }

    fn build_tower(&mut self, ctx: &mut Ctx) {
        for (e, _) in self.targets.drain(..) {
            ctx.despawn(e);
        }
        let palette = [0xe76f51, 0xf4a261, 0xe9c46a, 0x2a9d8f, 0x6d597a];
        let mut n = 0;
        // pyramid of plain cubes
        for level in 0..5 {
            let count = 5 - level;
            for i in 0..count {
                for j in 0..count {
                    let off = (count as f32 - 1.0) * 0.5;
                    let pos = Vec3::new(i as f32 - off, 0.5 + level as f32, j as f32 - off - 2.0);
                    let e = ctx.spawn_cube("block", pos, Vec3::splat(0.98), palette[(level + i + j) % palette.len()]);
                    ctx.physics.add(&mut ctx.world, e, RigidBodyBuilder::dynamic(), ColliderBuilder::cuboid(0.49, 0.49, 0.49));
                    self.targets.push((e, false));
                    n += 1;
                }
            }
        }
        // textured glTF crates (dynamic bodies on the model root)
        for k in 0..4 {
            let pos = Vec3::new(-6.0 + k as f32 * 1.3, 0.6, 3.0);
            let e = ctx.load_gltf("assets/crate.glb", Transform::from_translation(pos).with_scale(Vec3::splat(1.2)));
            ctx.physics.add(&mut ctx.world, e, RigidBodyBuilder::dynamic(), ColliderBuilder::cuboid(0.6, 0.6, 0.6));
            self.targets.push((e, false));
            n += 1;
        }
        self.total = n;
        self.score = 0;
        self.won = false;
    }

    fn reset(&mut self, ctx: &mut Ctx) {
        self.build_tower(ctx);
        self.respawn_player(ctx);
        for s in self.shots.drain(..) {
            ctx.despawn(s);
        }
        engine::log_info!("game reset: {} targets", self.total);
    }

    fn respawn_player(&mut self, ctx: &mut Ctx) {
        let Some(p) = self.player else { return };
        if let Ok(mut t) = ctx.world.get::<&mut Transform>(p) {
            t.translation = PLAYER_SPAWN;
        }
        if let Some(rb) = ctx.physics.body_mut(&ctx.world, p) {
            rb.set_position(engine::physics::rapier::math::Pose::from_translation(PLAYER_SPAWN), true);
            rb.set_linvel(Vec3::ZERO, true);
            rb.set_angvel(Vec3::ZERO, true);
        }
    }

    fn shoot(&mut self, ctx: &mut Ctx) {
        let Some((origin, dir)) = ctx.screen_ray(ctx.input.pointer) else { return };
        self.shoot_from(ctx, origin, dir);
    }

    /// One code path for mouse shots and the `shoot_at` debug command.
    fn shoot_from(&mut self, ctx: &mut Ctx, origin: Vec3, dir: Vec3) -> Entity {
        let e = ctx.spawn_sphere("shot", origin + dir * 1.5, 0.25, 0x88ccff);
        if let Ok(m) = ctx.world.get::<&MeshRenderer>(e).map(|m| m.material) {
            if let Some(mat) = ctx.renderer.material_mut(m) {
                mat.emissive = Vec3::new(0.6, 1.5, 3.0); // HDR: glows through bloom
                mat.metallic = 0.8;
                mat.roughness = 0.2;
            }
        }
        ctx.physics.add(
            &mut ctx.world,
            e,
            RigidBodyBuilder::dynamic().linvel(dir * 32.0).ccd_enabled(true),
            ColliderBuilder::ball(0.25).density(40.0),
        );
        ctx.audio.play_named("laser", Play::new().volume(0.45).pitch(0.9 + ctx.rng.f32() * 0.2).at(origin + dir * 1.5).echo(0.11, 0.3));
        self.shots.push_back(e);
        if self.shots.len() > 25 {
            let old = self.shots.pop_front().unwrap();
            ctx.despawn(old);
        }
        e
    }

    fn set_walker_mode(&mut self, ctx: &mut Ctx, on: bool) {
        self.walker_mode = on;
        let target = if on { self.walker } else { self.player };
        if let Some(cam) = self.camera {
            if let Ok(mut c) = ctx.world.get::<&mut ThirdPersonCamera>(cam) {
                c.target = target;
                c.pivot_offset = if on { Vec3::new(0.0, 0.6, 0.0) } else { Vec3::new(0.0, 0.8, 0.0) };
            }
        }
        engine::log_info!("control: {}", if on { "walker (character controller)" } else { "ball (physics)" });
    }

    fn hud_help(&self, ctx: &mut Ctx) {
        ctx.ui.set(
            "help",
            "<div class='panel'><b>Tumble</b> — push every crate off the island<br>\
             <kbd>WASD</kbd> move · <kbd>Space</kbd> jump · <kbd>Left click</kbd> shoot<br>\
             <kbd>Right drag</kbd> camera · <kbd>Wheel</kbd> zoom · <kbd>C</kbd> switch figure · <kbd>R</kbd> reset · <kbd>P</kbd> pause<br>             <kbd>N</kbd> day/night · <kbd>L</kbd> run the day cycle · <kbd>[</kbd> <kbd>]</kbd> time of day<br>\
             <button data-action='reset'>Restart</button> <button data-action='toggle-shadows'>Shadows</button> \
             <button data-action='walker'>Switch figure</button></div>",
        );
    }

    fn build_world(&mut self, ctx: &mut Ctx) {
        // Procedural, seamless textures (engine::texgen)
        let grass = ctx.generate_texture(&TexGen::new(TexPreset::Grass).size(256).seed(4));
        let dirt = ctx.generate_texture(&TexGen::new(TexPreset::Dirt).size(256).seed(9));
        let metal = ctx.generate_texture(&TexGen::new(TexPreset::Metal).size(128).seed(2));
        let tiled_cube = |ctx: &mut Ctx, name: &str, repeat: f32| {
            let mut md = engine::mesh::MeshData::cube(1.0);
            for v in &mut md.vertices {
                v.uv = [v.uv[0] * repeat, v.uv[1] * repeat];
            }
            ctx.renderer.add_mesh(name, &md)
        };
        let island_mesh = tiled_cube(ctx, "island", 7.0);

        // Sea around the island: endless camera-following ocean (Gerstner waves, refraction,
        // screen-space reflections, foam on the crests and at the island).
        let water_mat = ctx.renderer.add_material(Material {
            wave_height: 0.16,
            wave_scale: 0.3,
            water_chop: 0.9,
            water_clarity: 3.0,
            water_vertex_spacing: 0.3,
            ..Material::water(0x06304a, 0x1f8a9a)
        });
        let w = ctx.spawn_ocean(water_mat, -2.2);
        if let Ok(mut mr) = ctx.world.get::<&mut MeshRenderer>(w) {
            mr.cast_shadows = false;
        }

        // Pond: raised stone rim (normal + parallax mapped), pebble bed (vertex displacement),
        // water surface (water shader: depth colour over the bed, foam where it meets the stones).
        {
            let stone = ctx.generate_maps(&TexGen::new(TexPreset::Stone).size(256).seed(11).scale(0.6), 4.0);
            let rim_mat = ctx.renderer.add_material(Material {
                texture: stone.color,
                roughness: 0.85,
                ..Material::color(0xffffff).normal_map(stone.normal, 1.0).parallax_map(stone.height, 0.04)
            });
            let segments = 16;
            for k in 0..segments {
                let a = k as f32 / segments as f32 * std::f32::consts::TAU;
                let r = POND_R + 0.3;
                // Neighbouring stones overlap at their ends, so their faces must not share a plane
                // (z-fighting, found by engine.call("zfight")): every other stone is taller and sits deeper.
                let (bottom, top) = if k % 2 == 0 { (0.0, 0.56) } else { (-0.05, 0.64) };
                let p = POND + Vec3::new(a.cos() * r, (bottom + top) * 0.5, a.sin() * r);
                let rot = Quat::from_rotation_y(-a + std::f32::consts::FRAC_PI_2);
                let t = Transform::from_translation(p).with_rotation(rot).with_scale(Vec3::new(1.02, top - bottom, 0.6 + (k % 2) as f32 * 0.04));
                let e = ctx.spawn_mesh("pond-rim", ctx.renderer.builtin.cube, rim_mat, t);
                ctx.physics.add(&mut ctx.world, e, RigidBodyBuilder::fixed(), ColliderBuilder::cuboid(0.51, (top - bottom) * 0.5, 0.3));
            }
            let pebbles = ctx.generate_maps(&TexGen::new(TexPreset::Dirt).size(256).seed(21).scale(1.4), 3.0);
            let bed_mesh = ctx.renderer.add_mesh("pond-bed", &engine::mesh::MeshData::disk(POND_R + 0.1, 32, 64, 3.0));
            let bed_mat = ctx.renderer.add_material(Material {
                texture: pebbles.color,
                roughness: 0.9,
                ..Material::color(0xc8c0b0).normal_map(pebbles.normal, 1.0).displace(pebbles.height, 0.12)
            });
            let bed = ctx.spawn_mesh("pond-bed", bed_mesh, bed_mat, Transform::from_translation(POND + Vec3::Y * 0.04));
            ctx.physics.add(&mut ctx.world, bed, RigidBodyBuilder::fixed(), ColliderBuilder::cylinder(0.02, POND_R));
            let surface = ctx.renderer.add_mesh("pond-water", &engine::mesh::MeshData::disk(POND_R + 0.15, 40, 160, 1.0));
            let pond_mat = ctx.renderer.add_material(Material {
                wave_height: 0.012,
                wave_scale: 2.2,
                water_chop: 0.3,
                water_clarity: 0.8,
                water_foam: 0.6,
                water_vertex_spacing: 0.06,
                ..Material::water(0x0e3b3a, 0x5fa89a)
            });
            let w = ctx.spawn_mesh("pond-water", surface, pond_mat, Transform::from_translation(POND + Vec3::Y * 0.45));
            if let Ok(mut mr) = ctx.world.get::<&mut MeshRenderer>(w) {
                mr.cast_shadows = false;
            }
        }

        // Grass tufts: two crossed quads, alpha-tested (AlphaMode::Mask), double-sided, cut-out shadows.
        let blades = ctx.generate_texture(&TexGen::new(TexPreset::Blades).size(128).seed(3));
        let tuft_mesh = {
            let q = engine::mesh::MeshData::quad();
            let mut md = engine::mesh::MeshData::default();
            for rot in [0.0f32, std::f32::consts::FRAC_PI_2] {
                let r = Quat::from_rotation_y(rot);
                let base = md.vertices.len() as u32;
                for v in &q.vertices {
                    let p = r * Vec3::new(v.pos[0], v.pos[1] + 0.5, v.pos[2]);
                    let n = r * Vec3::from(v.normal);
                    md.vertices.push(engine::mesh::Vertex { pos: p.into(), normal: [n.x * 0.3, 1.0, n.z * 0.3], uv: v.uv });
                }
                md.indices.extend(q.indices.iter().map(|i| i + base));
            }
            ctx.renderer.add_mesh("grass-tuft", &md)
        };
        let tuft_mat = ctx.renderer.add_material(Material { texture: blades, roughness: 0.9, ..Material::color(0xffffff).cutout(0.5) });
        let mut placed = 0;
        while placed < 300 {
            let (x, z) = (ctx.rng.range(-13.0, 13.0), ctx.rng.range(-13.0, 13.0));
            if (x.abs() < 4.0 && z > -6.5 && z < 2.0) || (x - POND.x).hypot(z - POND.z) < POND_R + 0.9 || (x - CAMPFIRE.x).hypot(z - CAMPFIRE.z) < 1.2 {
                continue; // keep the tower area, pond and campfire clear
            }
            let s = ctx.rng.range(0.5, 0.9);
            let t = Transform::from_xyz(x, 0.0, z).with_rotation(Quat::from_rotation_y(ctx.rng.range(0.0, std::f32::consts::PI))).with_scale(Vec3::new(s, s * 0.8, s));
            ctx.spawn_mesh("grass", tuft_mesh, tuft_mat, t);
            placed += 1;
        }
        let cliff_mesh = tiled_cube(ctx, "cliff", 5.0);

        // Island
        let ground_mat = ctx.renderer.add_material(Material { texture: grass, roughness: 0.95, ..Material::color(0xffffff) });
        let ground = ctx.spawn_mesh(
            "island",
            island_mesh,
            ground_mat,
            Transform::from_xyz(0.0, -0.5, 0.0).with_scale(Vec3::new(ISLAND_HALF * 2.0, 1.0, ISLAND_HALF * 2.0)),
        );
        ctx.physics.add(&mut ctx.world, ground, RigidBodyBuilder::fixed(), ColliderBuilder::cuboid(ISLAND_HALF, 0.5, ISLAND_HALF));
        let cliff = ctx.renderer.add_material(Material { texture: dirt, ..Material::color(0xffffff).rough(1.0) });
        ctx.spawn_mesh(
            "cliff",
            cliff_mesh,
            cliff,
            Transform::from_xyz(0.0, -3.0, 0.0).with_scale(Vec3::new(ISLAND_HALF * 2.0 - 0.6, 4.0, ISLAND_HALF * 2.0 - 0.6)),
        );

        // Lamps: emissive orb + point light on a pole
        let pole = ctx.renderer.add_material(Material::color(0x333333).metal(0.9).rough(0.35));
        for (i, (x, z, color)) in [(-9.0f32, -9.0f32, Vec3::new(1.0, 0.55, 0.2)), (9.0, -9.0, Vec3::new(0.3, 0.6, 1.0)), (9.0, 9.0, Vec3::new(1.0, 0.3, 0.6))].into_iter().enumerate() {
            let p = ctx.spawn_mesh("lamp-pole", ctx.renderer.builtin.cube, pole, Transform::from_xyz(x, 1.5, z).with_scale(Vec3::new(0.15, 3.0, 0.15)));
            ctx.physics.add(&mut ctx.world, p, RigidBodyBuilder::fixed(), ColliderBuilder::cuboid(0.075, 1.5, 0.075));
            let orb = ctx.renderer.add_material(Material { unlit: true, emissive: color * 2.0, base_color: color.extend(1.0), ..Default::default() });
            let o = ctx.spawn_mesh(&format!("lamp-{i}"), ctx.renderer.builtin.sphere, orb, Transform::from_xyz(x, 3.3, z).with_scale(Vec3::splat(0.5)));
            let _ = ctx.world.insert_one(o, PointLight { color, intensity: 25.0, range: 12.0 });
            if let Ok(mut m) = ctx.world.get::<&mut MeshRenderer>(o) {
                m.cast_shadows = false;
            }
        }

        // Trees (glTF, static colliders)
        for (x, z, s) in [(-10.0, 6.0, 1.0), (-11.5, 0.0, 1.3), (11.0, -2.0, 1.1), (6.0, 11.0, 0.9), (-4.0, -11.0, 1.2), (11.5, 4.5, 0.8)] {
            let t = ctx.load_gltf("assets/tree.glb", Transform::from_xyz(x, 0.0, z).with_scale(Vec3::splat(s)));
            ctx.physics.add(&mut ctx.world, t, RigidBodyBuilder::fixed(), ColliderBuilder::cylinder(1.5 * s, 0.25 * s));
        }

        // Rocks: OBJ source cooked by tools/asset-cook (russimp/Assimp) -> assets/rock.glb
        for (x, z, s, yaw) in [(-7.0, -7.0, 1.4, 0.3), (8.0, 8.5, 1.0, 1.9), (-12.0, -4.0, 0.8, 4.0)] {
            let t = Transform::from_xyz(x, 0.0, z).with_scale(Vec3::splat(s));
            let t = Transform { rotation: Quat::from_rotation_y(yaw), ..t };
            let r = ctx.load_gltf("assets/rock.glb", t);
            ctx.physics.add(&mut ctx.world, r, RigidBodyBuilder::fixed(), ColliderBuilder::ball(0.8 * s));
        }

        // Glowing mushrooms (emissive > 1 -> bloom) with fireflies above them
        let stem = ctx.renderer.add_material(Material::color(0xe6dccb).rough(0.8));
        let cap = ctx.renderer.add_material(Material { emissive: Vec3::new(0.25, 2.4, 1.7), ..Material::color(0x2ad1a0).rough(0.5) });
        for (k, (x, z)) in [(-9.0f32, 4.2f32), (-8.3, 4.8), (-9.6, 5.1), (10.0, -4.3), (9.3, -3.6), (10.6, -3.4), (7.2, 10.2), (-3.0, -9.6), (-2.2, -10.2)].into_iter().enumerate() {
            let s = 0.7 + (k % 3) as f32 * 0.25;
            ctx.spawn_mesh("mushroom-stem", ctx.renderer.builtin.cube, stem, Transform::from_xyz(x, 0.18 * s, z).with_scale(Vec3::new(0.09, 0.36, 0.09) * s));
            ctx.spawn_mesh("mushroom-cap", ctx.renderer.builtin.sphere, cap, Transform::from_xyz(x, 0.38 * s, z).with_scale(Vec3::new(0.42, 0.2, 0.42) * s));
        }
        for p in [Vec3::new(-9.0, 1.0, 4.7), Vec3::new(10.0, 1.0, -3.8)] {
            ctx.spawn_particles("fireflies", p, ParticleEmitter { area: Vec3::new(1.6, 0.6, 1.6), ..ParticleEmitter::preset(ParticlePreset::Fireflies) });
        }

        // Campfire: logs, fire + smoke particles, flickering point light
        let wood = ctx.renderer.add_material(Material::color(0x4a2f1c).rough(0.95));
        // the second log lies on the first (different heights: crossed logs must not share a plane)
        for (a, y) in [(0.3f32, 0.08f32), (1.9, 0.2)] {
            let l = ctx.spawn_mesh("log", ctx.renderer.builtin.cube, wood, Transform::from_translation(CAMPFIRE + Vec3::Y * y).with_rotation(Quat::from_rotation_y(a)).with_scale(Vec3::new(0.9, 0.16, 0.16)));
            ctx.physics.add(&mut ctx.world, l, RigidBodyBuilder::fixed(), ColliderBuilder::cuboid(0.45, 0.08, 0.08));
        }
        ctx.spawn_particles("fire", CAMPFIRE + Vec3::Y * 0.15, ParticleEmitter::preset(ParticlePreset::Fire));
        ctx.spawn_particles("smoke", CAMPFIRE + Vec3::Y * 1.1, ParticleEmitter::preset(ParticlePreset::Smoke));
        self.fire_light = Some(ctx.world.spawn((Name("fire-light".into()), Transform::from_translation(CAMPFIRE + Vec3::Y * 0.8), PointLight { color: Vec3::new(1.0, 0.5, 0.15), intensity: 10.0, range: 7.0 })));

        // Terminal: a clickable 2D panel on a 3D quad (engine::panel)
        let panel = Panel::new(512, 320)
            .label("title", [28.0, 22.0, 456.0, 40.0], "TUMBLE TERMINAL", 34.0)
            .label("score", [28.0, 76.0, 456.0, 30.0], "", 26.0)
            .bar("progress", [28.0, 116.0, 456.0, 22.0], 0.0, "#34d399")
            .button("reset", [28.0, 160.0, 220.0, 62.0], "Restart", "reset")
            .button("shadows", [264.0, 160.0, 220.0, 62.0], "Shadows", "toggle-shadows")
            .button("slowmo", [28.0, 236.0, 220.0, 62.0], "Slow motion", "slowmo")
            .button("sound", [264.0, 236.0, 220.0, 62.0], "Sound", "toggle-sound");
        let face = Vec3::new(PLAYER_SPAWN.x - 5.5, 0.0, PLAYER_SPAWN.z + 10.0 - 6.0); // towards the default camera
        let rot = Quat::from_rotation_y(face.x.atan2(face.z));
        let t = Transform::from_xyz(5.5, 1.9, 6.0).with_rotation(rot).with_scale(Vec3::new(2.4, 1.5, 1.0));
        let term = ctx.spawn_panel("terminal", panel, t);
        ctx.physics.add(&mut ctx.world, term, RigidBodyBuilder::fixed(), ColliderBuilder::cuboid(1.2, 0.75, 0.02));
        let post_mat = ctx.renderer.add_material(Material { texture: metal, ..Material::color(0xaab4c0).rough(0.35).metal(0.8) });
        let post = ctx.spawn_mesh("terminal-post", ctx.renderer.builtin.cube, post_mat, Transform::from_xyz(5.5, 0.6, 6.0).with_rotation(rot).with_scale(Vec3::new(0.25, 1.2, 0.25)));
        ctx.physics.add(&mut ctx.world, post, RigidBodyBuilder::fixed(), ColliderBuilder::cuboid(0.125, 0.6, 0.125));
        self.terminal = Some(term);
    }
}

impl Screen<Data> for PlayScreen {
    fn name(&self) -> &'static str {
        "play"
    }

    fn enter(&mut self, ctx: &mut Ctx, data: &mut Data) {
        data.ensure(ctx);
        data.progress.plays += 1;
        data.save(ctx);
        sky_setup(ctx);
        self.build_world(ctx);

        // Player: physics ball
        let pmat = ctx.renderer.add_material(Material::color(0xffc93c).rough(0.25).metal(0.2));
        let player = ctx.spawn_mesh("player", ctx.renderer.builtin.sphere, pmat, Transform::from_translation(PLAYER_SPAWN));
        ctx.physics.add(
            &mut ctx.world,
            player,
            RigidBodyBuilder::dynamic().angular_damping(0.8).linear_damping(0.15).ccd_enabled(true).can_sleep(false),
            ColliderBuilder::ball(0.5).density(2.0).friction(1.2),
        );
        let _ = ctx.world.insert_one(player, WorldLabel::new("You"));
        self.player = Some(player);

        // Walker: character controller (capsule) with the animated critter as its body
        let cc = CharacterController { speed: 4.5, ..Default::default() };
        let foot = cc.foot_offset();
        let walker = ctx.spawn_character("walker", Vec3::new(2.5, 0.0, 10.5), cc);
        let model = ctx.load_gltf("assets/critter.glb", Transform::from_xyz(0.0, -foot, 0.0).with_scale(Vec3::splat(0.7)));
        let _ = ctx.world.insert_one(model, Parent(walker));
        self.walker = Some(walker);
        self.walker_model = Some(model);

        // Third-person camera (orbit: right drag / pointer lock, wheel + pinch zoom, collision)
        self.camera = Some(ctx.world.spawn((
            Name("camera".into()),
            Transform::from_xyz(0.0, 6.0, 20.0),
            Camera::default(),
            ThirdPersonCamera { distance: 11.0, min_distance: 4.0, max_distance: 40.0, pitch: 0.45, min_pitch: -0.2, max_pitch: 1.4, ..ThirdPersonCamera::following(player) },
        )));

        self.build_tower(ctx);
        self.hud_help(ctx);
        engine::log_info!("Tumble initialised: {} targets", self.total);
    }

    fn exit(&mut self, ctx: &mut Ctx, _data: &mut Data) {
        for slot in ["help", "score", "touch"] {
            ctx.ui.set(slot, "");
        }
        ctx.time.time_scale = 1.0;
    }

    fn fixed_update(&mut self, ctx: &mut Ctx, data: &mut Data) {
        let dt = ctx.time.fixed_dt;
        let axis = ctx.input.move_axis();
        let yaw = self.camera_yaw(ctx);
        let fwd = Vec3::new(-yaw.sin(), 0.0, -yaw.cos());
        let right = Vec3::new(-fwd.z, 0.0, fwd.x);
        let dir = if self.walker_mode { Vec3::ZERO } else { fwd * axis.y + right * axis.x };
        let pos = self.player_pos(ctx);

        // Walker: the character controller does the movement; we only set what it wants.
        if let Some(w) = self.walker {
            if let Ok(mut cc) = ctx.world.get::<&mut CharacterController>(w) {
                if self.walker_mode {
                    cc.drive(axis, yaw);
                    cc.run = ctx.input.key("ShiftLeft") || ctx.input.key("ShiftRight");
                    cc.jump = ctx.input.key_down_or_pressed("Space") || self.jump_buffer > 0;
                } else {
                    cc.wish = Vec3::ZERO;
                    cc.jump = false;
                }
            }
        }

        // Grounded = any touching contact with an upward normal (<= 60° slope). The old check was a
        // single 0.15 ray straight down from the bottom of the ball: it missed when the ball rested
        // on an edge or in a gap between blocks (e.g. on the crate tower), so jumps were dropped.
        let grounded = self.player.is_some_and(|p| ctx.physics.grounded(&ctx.world, p, 60.0))
            || ctx.physics.raycast(pos - Vec3::Y * 0.51, Vec3::NEG_Y, 0.15).is_some();
        self.grounded = grounded;
        self.jump_cooldown = self.jump_cooldown.saturating_sub(1);
        if grounded && self.jump_cooldown == 0 {
            self.coyote = COYOTE_STEPS;
        } else {
            self.coyote = self.coyote.saturating_sub(1);
        }
        let jump = !self.walker_mode && self.jump_buffer > 0 && self.coyote > 0;
        self.jump_buffer = self.jump_buffer.saturating_sub(1);
        if jump {
            self.jump_buffer = 0;
            self.coyote = 0;
            self.jump_cooldown = JUMP_COOLDOWN_STEPS;
            self.jumps += 1;
        }
        let world = &ctx.world;
        if let Some(rb) = self.player.and_then(|p| ctx.physics.body_mut(world, p)) {
            if dir.length_squared() > 0.0 {
                let speed = rb.linvel().with_y(0.0).length();
                let accel = if grounded { 14.0 } else { 5.0 };
                if speed < 9.0 {
                    rb.apply_impulse(dir * accel * rb.mass() * dt, true);
                }
                rb.apply_torque_impulse(Vec3::Y.cross(dir) * 0.6 * dt, true);
            }
            if jump {
                // Set (not add) the vertical speed: a buffered jump on landing would otherwise
                // cancel against the falling speed and become a tiny hop.
                let v = rb.linvel();
                rb.set_linvel(v.with_y(v.y.max(0.0) + 6.0), true);
            }
        }
        if jump {
            if let Some(p) = self.player {
                ctx.audio.play_named("jump", Play::new().volume(0.6).follow(p));
            }
        }

        // scoring: targets that left the island
        let mut scored = false;
        for (e, counted) in self.targets.iter_mut() {
            let y = ctx.world.get::<&Transform>(*e).map(|t| t.translation.y).unwrap_or(0.0);
            if !*counted && y < -2.0 {
                *counted = true;
                self.score += 1;
                scored = true;
                let at = ctx.world.get::<&Transform>(*e).map(|t| t.translation).unwrap_or_default();
                ctx.audio.play_named("thud", Play::new().volume(0.8).at(at).ref_distance(6.0).reverb(1.2).lowpass(2200.0));
                ctx.spawn_particles("splash", Vec3::new(at.x, -2.15, at.z), ParticleEmitter::preset(ParticlePreset::Splash));
                engine::log_info!("target {} fell off (score {}/{})", e.id(), self.score, self.total);
            }
        }
        if scored {
            data.progress.best = data.progress.best.max(self.score);
            if self.score == self.total && !self.won {
                self.won = true;
                data.progress.wins += 1;
                ctx.audio.play_named("powerup", Play::new().reverb(2.0));
                ctx.audio.play_named("chime", Play::new().echo(0.25, 0.45));
            }
            data.save(ctx);
        }
        if pos.y < -12.0 {
            self.falls += 1;
            engine::log_info!("player fell ({} falls), respawning", self.falls);
            self.respawn_player(ctx);
        }
        if let Some(w) = self.walker {
            let wy = ctx.world.get::<&Transform>(w).map(|t| t.translation.y).unwrap_or(0.0);
            if wy < -12.0 {
                if let Ok(mut t) = ctx.world.get::<&mut Transform>(w) {
                    t.translation = Vec3::new(2.5, 1.0, 10.5);
                }
            }
        }
        // far-away bodies are removed, but stay counted
        let below = |e: &Entity| ctx.world.get::<&Transform>(*e).map(|t| t.translation.y < -40.0).unwrap_or(false);
        let gone: Vec<Entity> = self.targets.iter().map(|(e, _)| *e).chain(self.shots.iter().copied()).filter(below).collect();
        for e in gone {
            ctx.despawn(e);
            self.shots.retain(|s| *s != e);
        }
    }

    fn update(&mut self, ctx: &mut Ctx, _data: &mut Data, _dt: f32) -> Trans<Data> {
        let input = &ctx.input;
        let taps = input.touch.taps.clone();
        let swipe_up = input.touch.swipes.iter().any(|s| s.dir.y < -0.7);
        let touch_device = input.touch.touch_device;
        let shoot = input.button_pressed(MouseButton::Left) || input.gamepad_buttons.get(7).copied().unwrap_or(false) && ctx.time.frame.is_multiple_of(10);
        if input.key_pressed("Space") || input.gamepad_buttons.first().copied().unwrap_or(false) || swipe_up {
            self.jump_buffer = JUMP_BUFFER_STEPS;
        }
        let reset = input.key_pressed("KeyR");
        let toggle_walker = input.key_pressed("KeyC");
        sky_keys(ctx);
        let input = &ctx.input;
        let pause = input.key_pressed("KeyP") || input.key_pressed("Escape");
        if shoot {
            self.shoot(ctx);
        }
        // touch: tap = shoot at that point (taps on world panels were already consumed by the engine)
        for tp in taps {
            if let Some((o, d)) = ctx.screen_ray(tp) {
                self.shoot_from(ctx, o, d);
            }
        }
        if touch_device {
            ctx.ui.set("touch", "<div class='touch-buttons'><button data-action='jump'>Jump</button> <button data-action='pause'>Pause</button></div>");
        }
        let mut trans = if pause { Trans::Push(Box::new(Pause)) } else { Trans::None };
        for action in ctx.ui.take_actions() {
            ctx.audio.play_named("click", Play::new().volume(0.5));
            match action.as_str() {
                "reset" => self.reset(ctx),
                "toggle-shadows" => ctx.renderer.settings.shadows = !ctx.renderer.settings.shadows,
                "jump" => self.jump_buffer = JUMP_BUFFER_STEPS,
                "walker" => self.set_walker_mode(ctx, !self.walker_mode),
                "pause" => trans = Trans::Push(Box::new(Pause)),
                "toggle-sound" => {
                    let m = !ctx.audio.muted();
                    ctx.audio.set_muted(m);
                }
                "slowmo" => {
                    self.slowmo = !self.slowmo;
                    ctx.time.time_scale = if self.slowmo { 0.3 } else { 1.0 };
                }
                other => engine::log_warn!("unknown ui action {other}"),
            }
        }
        if reset {
            self.reset(ctx);
        }
        if toggle_walker {
            self.set_walker_mode(ctx, !self.walker_mode);
        }

        // walker animation follows the character state
        if let (Some(w), Some(m)) = (self.walker, self.walker_model) {
            let state = ctx.world.get::<&CharacterController>(w).map(|c| (c.grounded, c.velocity.with_y(0.0).length())).ok();
            if let Some((grounded, speed)) = state {
                let (clip, speed) = if !grounded {
                    ("bounce", 1.0)
                } else if speed > 0.3 {
                    ("wave", 1.0 + speed * 0.25)
                } else {
                    ("idle", 1.0)
                };
                ctx.play_animation(m, clip, PlayOpts::looped().fade(0.2).speed(speed));
            }
        }
        // campfire flicker
        if let Some(l) = self.fire_light {
            let t = ctx.time.sim_time as f32;
            if let Ok(mut pl) = ctx.world.get::<&mut PointLight>(l) {
                pl.intensity = 9.0 + 2.5 * (t * 13.0).sin() * (t * 7.3).cos() + 1.5 * (t * 23.0).sin();
            }
        }

        let done = self.score == self.total && self.total > 0;
        ctx.ui.set(
            "score",
            &format!(
                "<div class='panel score'>{}{} / {} <small>crates</small></div>",
                if done { "🏆 " } else { "" },
                self.score,
                self.total
            ),
        );
        let (score, total, slowmo, shadows, muted) = (self.score, self.total, self.slowmo, ctx.renderer.settings.shadows, ctx.audio.muted());
        if let Some(p) = self.terminal.and_then(|t| ctx.panel_mut(t)) {
            p.set_text("score", &format!("{score} / {total} crates off the island"));
            p.set_value("progress", if total > 0 { score as f32 / total as f32 } else { 0.0 });
            p.set_text("slowmo", if slowmo { "Slow motion: on" } else { "Slow motion: off" });
            p.set_text("shadows", if shadows { "Shadows: on" } else { "Shadows: off" });
            p.set_text("sound", if muted { "Sound: off" } else { "Sound: on" });
        }
        ctx.ui.set("fps", &format!("<div class='panel fps'>{:.0} fps · {:.1} ms</div>", ctx.time.fps, ctx.time.frame_ms));
        trans
    }

    fn debug_commands(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("reset", "Rebuild the tower and respawn the player."),
            ("teleport", "Move the player. args: {pos: [x,y,z]}"),
            ("shoot_at", "Fire a shot from the camera towards a world point. args: {target: [x,y,z]}"),
            ("step_lab", "Character step-up lab: a low plate (edge at z -7), or a flight of `steps` stairs `depth` m deep, and `lanes` characters walking onto it from start distances spread over `start_spread` m. args: {height?: 0.152, steps?: 1, depth?: 4, radius?: 0.32, half_height?: 0.62, step_height?: 0.3, speed?: 3.2, lanes?: 16, start_spread?: 0.08}. Then step and call characters."),
            ("control", "Switch control between the physics ball and the character-controller walker. args: {walker: bool, pos?: [x,y,z] (place the walker, capsule centre)}"),
        ]
    }

    fn debug_command(&mut self, ctx: &mut Ctx, _data: &mut Data, name: &str, args: &Value) -> Option<Result<Value, String>> {
        Some(match name {
            "reset" => {
                self.reset(ctx);
                Ok(json!({"targets": self.total}))
            }
            "teleport" => {
                let Some(p) = args.get("pos").and_then(|v| v.as_array()).and_then(|a| {
                    Some(Vec3::new(a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32, a.get(2)?.as_f64()? as f32))
                }) else {
                    return Some(Err("missing pos".into()));
                };
                let player = self.player?;
                if let Some(rb) = ctx.physics.body_mut(&ctx.world, player) {
                    rb.set_position(engine::physics::rapier::math::Pose::from_translation(p), true);
                    rb.set_linvel(Vec3::ZERO, true);
                }
                if let Ok(mut t) = ctx.world.get::<&mut Transform>(player) {
                    t.translation = p;
                }
                Ok(json!({"pos": [p.x, p.y, p.z]}))
            }
            "shoot_at" => {
                let target = args.get("target").and_then(|v| v.as_array()).and_then(|a| {
                    Some(Vec3::new(a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32, a.get(2)?.as_f64()? as f32))
                });
                let (Some(target), Some(cam)) = (target, self.camera) else { return Some(Err("missing target".into())) };
                let origin = ctx.world.get::<&Transform>(cam).map(|t| t.translation).unwrap_or_default();
                let dir = (target - origin).normalize();
                let e = self.shoot_from(ctx, origin, dir);
                Ok(json!({"id": e.id()}))
            }
            "step_lab" => {
                let f = |k: &str, d: f32| args.get(k).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(d);
                let (h, r, hh, sh, speed) = (f("height", 0.152), f("radius", 0.32), f("half_height", 0.62), f("step_height", 0.3), f("speed", 3.2));
                let lanes = args.get("lanes").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
                let dz = f("start_spread", 0.08); // start distances spread over this many metres
                let spacing = (r * 2.0 + 0.6).max(1.0);
                let width = lanes as f32 * spacing;
                // plate along x (west of the terminal), top at y = h, near edge at z = -7
                let x0 = -12.5;
                // stairs: `steps` plates, each `h` higher and `depth` further on (the last one 4 m deep)
                let steps = args.get("steps").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;
                let depth = f("depth", 4.0);
                for s in 0..steps {
                    let top = h * (s + 1) as f32;
                    let d = if s + 1 == steps { 4.0 } else { depth };
                    let z_near = -7.0 - depth * s as f32;
                    let plate = ctx.spawn_cube("lab-plate", Vec3::new(x0 + width * 0.5, top * 0.5, z_near - d * 0.5), Vec3::new(width, top, d), 0x8d99ae);
                    ctx.physics.add(&mut ctx.world, plate, RigidBodyBuilder::fixed(), ColliderBuilder::cuboid(width * 0.5, top * 0.5, d * 0.5));
                }
                let mut ids = vec![];
                for i in 0..lanes {
                    let z = -5.5 + i as f32 / lanes.max(2) as f32 * dz;
                    let cc = CharacterController { radius: r, half_height: hh, step_height: sh, speed, ..Default::default() };
                    let e = ctx.spawn_character("lab-hero", Vec3::new(x0 + spacing * (i as f32 + 0.5), 0.0, z), cc);
                    if let Ok(mut c) = ctx.world.get::<&mut CharacterController>(e) {
                        c.wish = Vec3::new(0.0, 0.0, -1.0);
                    }
                    ids.push(json!({"id": e.id(), "start_z": z}));
                }
                Ok(json!({"plate_top": h, "plate_edge_z": -7.0, "characters": ids}))
            }
            "control" => {
                let on = args.get("walker").and_then(|v| v.as_bool()).unwrap_or(!self.walker_mode);
                self.set_walker_mode(ctx, on);
                if let (Some(w), Some(p)) = (self.walker, args.get("pos").and_then(|v| v.as_array())) {
                    let f = |i: usize| p.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                    if let Ok(mut t) = ctx.world.get::<&mut Transform>(w) {
                        t.translation = Vec3::new(f(0), f(1), f(2));
                    }
                }
                Ok(json!({"walker": self.walker_mode}))
            }
            _ => return None,
        })
    }

    fn debug_state(&self, ctx: &Ctx, data: &Data) -> Value {
        let p = self.player_pos(ctx);
        let cam = self.camera.and_then(|c| ctx.world.get::<&ThirdPersonCamera>(c).ok().map(|c| json!({"yaw": c.yaw, "pitch": c.pitch, "distance": c.distance, "current_distance": c.current_distance})));
        let walker = self.walker.and_then(|w| {
            let t = ctx.world.get::<&Transform>(w).ok()?.translation;
            let cc = ctx.world.get::<&CharacterController>(w).ok()?;
            Some(json!({"pos": [t.x, t.y, t.z], "grounded": cc.grounded, "velocity": cc.velocity.to_array(), "jumps": cc.jumps, "sliding": cc.sliding}))
        });
        let anim = self.walker_model.and_then(|m| ctx.world.get::<&Animator>(m).ok().and_then(|a| a.current_name().map(|s| s.to_string())));
        json!({
            "score": self.score,
            "total": self.total,
            "won": self.score == self.total && self.total > 0,
            "falls": self.falls,
            "grounded": self.grounded,
            "jumps": self.jumps,
            "jump_buffer": self.jump_buffer,
            "coyote": self.coyote,
            "slowmo": self.slowmo,
            "player_pos": [p.x, p.y, p.z],
            "camera": cam,
            "control": if self.walker_mode { "walker" } else { "ball" },
            "walker": walker,
            "walker_animation": anim,
            "shots_alive": self.shots.len(),
            "targets_remaining": self.targets.iter().filter(|(_, c)| !c).map(|(e, _)| e.id()).collect::<Vec<_>>(),
            "progress": data.progress,
        })
    }
}

/// Atmosphere sky with clouds, driven by the day/night cycle (clock stopped until L is pressed).
pub fn sky_setup(ctx: &mut Ctx) {
    let s = &mut ctx.renderer.settings;
    s.sky_mode = 1;
    s.clouds = 0.45;
    s.day_cycle = true;
    s.day_length_minutes = 0.0;
    s.time_of_day = 15.5;
    s.sun_azimuth_deg = 200.0;
    s.sun_max_elevation_deg = 58.0;
    // night sky image (Milky Way), faded in as the sun sets; the moon and clouds stay on top
    s.skybox_blend = 0.0;
    s.skybox_night_blend = 1.0;
    if ctx.renderer.skybox().is_none() {
        ctx.load_skybox("assets/skybox-night.png");
    }
}

/// N = jump between day and night, L = run/stop the cycle (2 min per day), [ / ] = -/+ 1 hour.
pub fn sky_keys(ctx: &mut Ctx) {
    let (n, l, back, fwd) = (ctx.input.key_pressed("KeyN"), ctx.input.key_pressed("KeyL"), ctx.input.key_pressed("BracketLeft"), ctx.input.key_pressed("BracketRight"));
    let s = &mut ctx.renderer.settings;
    if n {
        s.day_cycle = true;
        s.time_of_day = if (6.5..19.0).contains(&s.time_of_day) { 22.5 } else { 15.5 };
    }
    if l {
        s.day_cycle = true;
        s.day_length_minutes = if s.day_length_minutes > 0.0 { 0.0 } else { 2.0 };
    }
    if back || fwd {
        s.day_cycle = true;
        s.time_of_day = (s.time_of_day + if fwd { 1.0 } else { -1.0 }).rem_euclid(24.0);
    }
}

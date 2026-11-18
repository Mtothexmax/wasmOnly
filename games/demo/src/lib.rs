//! Demo: "Tumble" — roll a ball, knock the crate tower off the island.
//!
//! Exercises every engine system: PBR + shadows, point lights, glTF models (static + dynamic),
//! rapier physics, input (keyboard, mouse, wheel, gamepad), DOM HUD with actions, world labels,
//! and game-specific AI debug commands.

use engine::physics::rapier::prelude::{ColliderBuilder, RigidBodyBuilder};
use engine::prelude::*;
use engine::renderer::Material;
use std::collections::VecDeque;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn main() {
    engine::run(Tumble::default());
}

const PLAYER_SPAWN: Vec3 = Vec3::new(0.0, 1.0, 9.0);
/// Pond centre and inner radius.
const POND: Vec3 = Vec3::new(-6.5, 0.0, 7.5);
const POND_R: f32 = 2.1;
const ISLAND_HALF: f32 = 14.0;
/// A jump press stays valid this many fixed steps (pressed slightly before landing still jumps).
const JUMP_BUFFER_STEPS: u32 = 8;
/// After leaving the ground a jump is still allowed for this many steps ("coyote time").
const COYOTE_STEPS: u32 = 6;
/// After a jump, ground contact is ignored this long (no double impulse while still touching).
const JUMP_COOLDOWN_STEPS: u32 = 12;

#[derive(Default)]
struct Tumble {
    player: Option<Entity>,
    camera: Option<Entity>,
    yaw: f32,
    pitch: f32,
    distance: f32,
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
    slowmo: bool,
}

impl Tumble {
    fn player_pos(&self, ctx: &Ctx) -> Vec3 {
        self.player
            .and_then(|p| ctx.world.get::<&Transform>(p).ok().map(|t| t.translation))
            .unwrap_or(PLAYER_SPAWN)
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
                mat.emissive = Vec3::new(0.2, 0.5, 1.0);
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
}

impl Game for Tumble {
    fn init(&mut self, ctx: &mut Ctx) {
        self.distance = 11.0;
        self.pitch = 0.45;

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

        // Sea around the island: water shader (waves, Fresnel sky reflection, sun glint, depth colour).
        let water_mesh = ctx.renderer.add_mesh("water", &engine::mesh::MeshData::plane(1.0, 60.0));
        let water_mat = ctx.renderer.add_material(Material { wave_height: 0.09, wave_scale: 0.35, ..Material::water(0x0b3a5a, 0x2f8fa8) });
        let w = ctx.spawn_mesh("water", water_mesh, water_mat, Transform::from_xyz(0.0, -2.2, 0.0).with_scale(Vec3::new(400.0, 1.0, 400.0)));
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
                let p = POND + Vec3::new(a.cos() * r, 0.28, a.sin() * r);
                let rot = Quat::from_rotation_y(-a + std::f32::consts::FRAC_PI_2);
                let t = Transform::from_translation(p).with_rotation(rot).with_scale(Vec3::new(1.02, 0.56, 0.6));
                let e = ctx.spawn_mesh("pond-rim", ctx.renderer.builtin.cube, rim_mat, t);
                ctx.physics.add(&mut ctx.world, e, RigidBodyBuilder::fixed(), ColliderBuilder::cuboid(0.51, 0.28, 0.3));
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
            let surface = ctx.renderer.add_mesh("pond-water", &engine::mesh::MeshData::disk(POND_R + 0.15, 24, 64, 1.0));
            let pond_mat = ctx.renderer.add_material(Material { wave_height: 0.012, wave_scale: 2.2, ..Material::water(0x0e3b3a, 0x5fa89a) });
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
            if (x.abs() < 4.0 && z > -6.5 && z < 2.0) || (x - POND.x).hypot(z - POND.z) < POND_R + 0.9 {
                continue; // keep the tower area clear
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

        // Player
        let pmat = ctx.renderer.add_material(Material::color(0xffc93c).rough(0.25).metal(0.2));
        let player = ctx.spawn_mesh("player", ctx.renderer.builtin.sphere, pmat, Transform::from_translation(PLAYER_SPAWN));
        ctx.physics.add(
            &mut ctx.world,
            player,
            RigidBodyBuilder::dynamic().angular_damping(0.8).linear_damping(0.15).ccd_enabled(true).can_sleep(false),
            ColliderBuilder::ball(0.5).density(2.0).friction(1.2),
        );
        let _ = ctx.world.insert_one(player, WorldLabel::new("Du"));
        self.player = Some(player);

        // Camera
        self.camera = Some(ctx.world.spawn((Name("camera".into()), Transform::from_xyz(0.0, 6.0, 20.0), Camera::default())));

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

        // Terminal: a clickable 2D panel on a 3D quad (engine::panel)
        let panel = Panel::new(512, 320)
            .label("title", [28.0, 22.0, 456.0, 40.0], "TUMBLE TERMINAL", 34.0)
            .label("score", [28.0, 76.0, 456.0, 30.0], "", 26.0)
            .bar("progress", [28.0, 116.0, 456.0, 22.0], 0.0, "#34d399")
            .button("reset", [28.0, 160.0, 220.0, 62.0], "Neu starten", "reset")
            .button("shadows", [264.0, 160.0, 220.0, 62.0], "Schatten", "toggle-shadows")
            .button("slowmo", [28.0, 236.0, 220.0, 62.0], "Zeitlupe", "slowmo")
            .button("sound", [264.0, 236.0, 220.0, 62.0], "Ton", "toggle-sound");
        let face = Vec3::new(PLAYER_SPAWN.x - 5.5, 0.0, PLAYER_SPAWN.z + 10.0 - 6.0); // towards the default camera
        let rot = Quat::from_rotation_y(face.x.atan2(face.z));
        let t = Transform::from_xyz(5.5, 1.9, 6.0).with_rotation(rot).with_scale(Vec3::new(2.4, 1.5, 1.0));
        let term = ctx.spawn_panel("terminal", panel, t);
        ctx.physics.add(&mut ctx.world, term, RigidBodyBuilder::fixed(), ColliderBuilder::cuboid(1.2, 0.75, 0.02));
        let post_mat = ctx.renderer.add_material(Material { texture: metal, ..Material::color(0xaab4c0).rough(0.35).metal(0.8) });
        let post = ctx.spawn_mesh("terminal-post", ctx.renderer.builtin.cube, post_mat, Transform::from_xyz(5.5, 0.6, 6.0).with_rotation(rot).with_scale(Vec3::new(0.25, 1.2, 0.25)));
        ctx.physics.add(&mut ctx.world, post, RigidBodyBuilder::fixed(), ColliderBuilder::cuboid(0.125, 0.6, 0.125));
        self.terminal = Some(term);

        self.build_tower(ctx);

        ctx.ui.set(
            "help",
            "<div class='panel'><b>Tumble</b> — stoß alle Kisten von der Insel<br>\
             <kbd>WASD</kbd> rollen · <kbd>Space</kbd> springen · <kbd>Linksklick</kbd> schießen<br>\
             <kbd>Rechts ziehen</kbd> Kamera · <kbd>Mausrad</kbd> Zoom · <kbd>R</kbd> Reset<br>\
             <button data-action='reset'>Neu starten</button> <button data-action='toggle-shadows'>Schatten</button></div>",
        );
        engine::log_info!("Tumble initialised: {} targets", self.total);
    }

    fn fixed_update(&mut self, ctx: &mut Ctx) {
        let dt = ctx.time.fixed_dt;
        let axis = ctx.input.move_axis();
        let fwd = Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos());
        let right = Vec3::new(-fwd.z, 0.0, fwd.x);
        let dir = fwd * axis.y + right * axis.x;
        let pos = self.player_pos(ctx);

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
        let jump = self.jump_buffer > 0 && self.coyote > 0;
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
        for (e, counted) in self.targets.iter_mut() {
            let y = ctx.world.get::<&Transform>(*e).map(|t| t.translation.y).unwrap_or(0.0);
            if !*counted && y < -2.0 {
                *counted = true;
                self.score += 1;
                let at = ctx.world.get::<&Transform>(*e).map(|t| t.translation).unwrap_or_default();
                ctx.audio.play_named("thud", Play::new().volume(0.8).at(at).ref_distance(6.0).reverb(1.2).lowpass(2200.0));
                if self.score == self.total {
                    ctx.audio.play_named("powerup", Play::new().reverb(2.0));
                    ctx.audio.play_named("chime", Play::new().echo(0.25, 0.45));
                }
                engine::log_info!("target {} fell off (score {}/{})", e.id(), self.score, self.total);
            }
        }
        if pos.y < -12.0 {
            self.falls += 1;
            engine::log_info!("player fell ({} falls), respawning", self.falls);
            self.respawn_player(ctx);
        }
        // far-away bodies are removed, but stay counted
        let below = |e: &Entity| ctx.world.get::<&Transform>(*e).map(|t| t.translation.y < -40.0).unwrap_or(false);
        let gone: Vec<Entity> = self.targets.iter().map(|(e, _)| *e).chain(self.shots.iter().copied()).filter(below).collect();
        for e in gone {
            ctx.despawn(e);
            self.shots.retain(|s| *s != e);
        }
    }

    fn update(&mut self, ctx: &mut Ctx, _dt: f32) {
        let input = &ctx.input;
        if input.button(MouseButton::Right) || input.pointer_locked {
            self.yaw -= input.pointer_delta.x * 0.006;
            self.pitch = (self.pitch + input.pointer_delta.y * 0.004).clamp(-0.2, 1.4);
        }
        self.yaw -= input.gamepad_right.x * 0.04;
        self.pitch = (self.pitch + input.gamepad_right.y * 0.03).clamp(-0.2, 1.4);
        // touch: drag on the right side looks around, pinch zooms (left side = joystick, in move_axis)
        self.yaw -= input.touch.look_delta.x * 0.008;
        self.pitch = (self.pitch + input.touch.look_delta.y * 0.006).clamp(-0.2, 1.4);
        self.distance = (self.distance * (1.0 + input.wheel * 0.001) * (-input.touch.pinch).exp()).clamp(4.0, 40.0);
        let taps = input.touch.taps.clone();
        let swipe_up = input.touch.swipes.iter().any(|s| s.dir.y < -0.7);
        let touch_device = input.touch.touch_device;
        let shoot = input.button_pressed(MouseButton::Left) || input.gamepad_buttons.get(7).copied().unwrap_or(false) && ctx.time.frame.is_multiple_of(10);
        if input.key_pressed("Space") || input.gamepad_buttons.first().copied().unwrap_or(false) || swipe_up {
            self.jump_buffer = JUMP_BUFFER_STEPS;
        }
        let reset = input.key_pressed("KeyR");
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
            ctx.ui.set("touch", "<div class='touch-buttons'><button data-action='jump'>Springen</button></div>");
        }
        for action in ctx.ui.take_actions() {
            ctx.audio.play_named("click", Play::new().volume(0.5));
            match action.as_str() {
                "reset" => self.reset(ctx),
                "toggle-shadows" => ctx.renderer.settings.shadows = !ctx.renderer.settings.shadows,
                "jump" => self.jump_buffer = JUMP_BUFFER_STEPS,
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

        // orbit camera
        let target = self.player_pos(ctx) + Vec3::Y * 0.8;
        let offset = Vec3::new(self.yaw.sin() * self.pitch.cos(), self.pitch.sin(), self.yaw.cos() * self.pitch.cos()) * self.distance;
        if let Some(cam) = self.camera {
            if let Ok(mut t) = ctx.world.get::<&mut Transform>(cam) {
                *t = Transform::from_translation(target + offset).looking_at(target, Vec3::Y);
            }
        }

        let done = self.score == self.total && self.total > 0;
        ctx.ui.set(
            "score",
            &format!(
                "<div class='panel score'>{}{} / {} <small>Kisten</small></div>",
                if done { "🏆 " } else { "" },
                self.score,
                self.total
            ),
        );
        let (score, total, slowmo, shadows, muted) = (self.score, self.total, self.slowmo, ctx.renderer.settings.shadows, ctx.audio.muted());
        if let Some(p) = self.terminal.and_then(|t| ctx.panel_mut(t)) {
            p.set_text("score", &format!("{score} / {total} Kisten von der Insel"));
            p.set_value("progress", if total > 0 { score as f32 / total as f32 } else { 0.0 });
            p.set_text("slowmo", if slowmo { "Zeitlupe: an" } else { "Zeitlupe: aus" });
            p.set_text("shadows", if shadows { "Schatten: an" } else { "Schatten: aus" });
            p.set_text("sound", if muted { "Ton: aus" } else { "Ton: an" });
        }
        ctx.ui.set("fps", &format!("<div class='panel fps'>{:.0} fps · {:.1} ms</div>", ctx.time.fps, ctx.time.frame_ms));
    }

    fn debug_commands(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("reset", "Rebuild the tower and respawn the player."),
            ("teleport", "Move the player. args: {pos: [x,y,z]}"),
            ("shoot_at", "Fire a shot from the camera towards a world point. args: {target: [x,y,z]}"),
        ]
    }

    fn debug_command(&mut self, ctx: &mut Ctx, name: &str, args: &Value) -> Option<Result<Value, String>> {
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
            _ => return None,
        })
    }

    fn debug_state(&self, ctx: &Ctx) -> Value {
        let p = self.player_pos(ctx);
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
            "camera": {"yaw": self.yaw, "pitch": self.pitch, "distance": self.distance},
            "shots_alive": self.shots.len(),
            "targets_remaining": self.targets.iter().filter(|(_, c)| !c).map(|(e, _)| e.id()).collect::<Vec<_>>(),
        })
    }
}

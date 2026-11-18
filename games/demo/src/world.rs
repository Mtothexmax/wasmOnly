//! Open-world screen: 1 km terrain (chunked LOD, splat, colliders, streaming), GPU grass,
//! scattered pines and rocks, streamed prop cells (street lamps = hundreds of clustered lights,
//! fences merged by static batching), a Ferris wheel and a car on a ring road (instance motion,
//! spot-light headlights), cascaded shadows, day/night, floating origin.
//!
//! The generators below (pine, lamp, fence, pavilion, Ferris wheel) are *game* code built with
//! the engine's geometry kit — the engine itself ships no content generators.

use crate::Data;
use engine::geom::{circle, rect, MeshBuilder, Opening};
use engine::mesh::MeshData;
use engine::physics::rapier::prelude::ColliderBuilder;
use engine::prelude::*;
use engine::renderer::Material;

const TOWN: Vec2 = Vec2::new(0.0, 0.0);
const TOWN_R: f32 = 110.0;
const RING_R: f32 = 150.0;

/// Registers the demo's generators (idempotent: re-registering replaces them).
fn register_generators(ctx: &mut Ctx) {
    let g = &mut ctx.generators;
    g.register("pine", "{height: m}", |p, seed| {
        let h = p["height"].as_f64().unwrap_or(9.0) as f32;
        let mut b = MeshBuilder::new();
        b.part("trunk").cylinder(0.22, h * 0.35, 6);
        let mut r = 0.0;
        for k in 0..3 {
            let y0 = h * (0.25 + k as f32 * 0.22);
            let rad = h * (0.24 - k as f32 * 0.05) * (1.0 + ((seed % 7) as f32 - 3.0) * 0.02);
            b.part("leaves").lathe(&[Vec2::new(0.0, y0), Vec2::new(rad, y0), Vec2::new(0.0, y0 + h * 0.38)], 8);
            r += rad;
        }
        let _ = r;
        b.finish()
    });
    g.register("lamp", "{height: m}", |p, _| {
        let h = p["height"].as_f64().unwrap_or(5.0) as f32;
        let mut b = MeshBuilder::new();
        b.part("metal").lathe(&[Vec2::new(0.0, 0.0), Vec2::new(0.16, 0.0), Vec2::new(0.12, 0.3), Vec2::new(0.06, 0.35), Vec2::new(0.05, h), Vec2::new(0.0, h)], 8);
        b.part("metal").sweep(&[Vec3::new(0.0, h - 0.1, 0.0), Vec3::new(0.3, h + 0.15, 0.0), Vec3::new(0.8, h + 0.1, 0.0)], &circle(0.04, 5), true);
        b.part("glow").lathe(&[Vec2::new(0.0, h - 0.1), Vec2::new(0.22, h - 0.05), Vec2::new(0.12, h + 0.12), Vec2::new(0.0, h + 0.14)], 10);
        // the glow head sits at the end of the arm
        let mut parts = b.finish();
        for (name, md) in parts.iter_mut() {
            if name == "glow" {
                engine::geom::transform(md, Mat4::from_translation(Vec3::new(0.8, -0.05, 0.0)));
            }
        }
        parts
    });
    g.register("fence", "{length: m}", |p, _| {
        let len = p["length"].as_f64().unwrap_or(8.0) as f32;
        let mut b = MeshBuilder::new().uv_world(1.0);
        let posts = (len / 2.0).ceil() as i32;
        for k in 0..=posts {
            let x = k as f32 * len / posts as f32;
            b.part("wood").block(Vec3::new(x, 0.0, 0.0), Vec3::new(0.12, 1.1, 0.12));
        }
        for y in [0.45f32, 0.9] {
            b.part("wood").cuboid(Vec3::new(0.0, y - 0.05, -0.03), Vec3::new(len, y + 0.05, 0.03));
        }
        b.finish()
    });
    g.register("pavilion", "{size: m, height: m} — stone floor, pillars, roof, a wall with door + window", |p, _| {
        let s = p["size"].as_f64().unwrap_or(8.0) as f32;
        let h = p["height"].as_f64().unwrap_or(3.2) as f32;
        let mut b = MeshBuilder::new().uv_world(0.6);
        b.part("stone").extrude(&rect(s, s), 0.0, 0.3, true);
        for (x, z) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            b.push(Mat4::from_translation(Vec3::new(x * (s * 0.5 - 0.4), 0.3, z * (s * 0.5 - 0.4))));
            // pillars and wall reach into the roof slab at different depths (no coplanar tops)
            b.part("stone").cylinder(0.25, h + 0.12, 10);
            b.pop();
        }
        b.push(Mat4::from_translation(Vec3::new(-s * 0.5 + 0.4, 0.3, -s * 0.5 + 0.4)));
        b.part("wall").wall(s - 0.8, h + 0.05, 0.25, &[Opening::door(1.0, 1.1, 2.2), Opening::window(s - 3.2, 1.0, 1.4, 1.1)]);
        b.pop();
        b.push(Mat4::from_translation(Vec3::Y * (h + 0.3)));
        b.part("roof").extrude(&rect(s + 0.8, s + 0.8), 0.0, 0.25, true);
        b.part("roof").lathe(&[Vec2::new(s * 0.68, 0.25), Vec2::new(0.0, 0.25 + s * 0.35)], 4);
        b.pop();
        b.finish()
    });
    g.register("ferris_wheel", "{radius: m}", |p, _| {
        let r = p["radius"].as_f64().unwrap_or(9.0) as f32;
        let mut b = MeshBuilder::new();
        for z in [-0.6f32, 0.6] {
            let ring: Vec<Vec3> = (0..=40).map(|k| {
                let a = k as f32 / 40.0 * std::f32::consts::TAU;
                Vec3::new(a.cos() * r, a.sin() * r, z)
            }).collect();
            b.part("steel").sweep(&ring, &circle(0.09, 6), true);
            for k in 0..12 {
                let a = k as f32 / 12.0 * std::f32::consts::TAU;
                b.part("steel").sweep(&[Vec3::new(0.0, 0.0, z), Vec3::new(a.cos() * r, a.sin() * r, z)], &circle(0.05, 5), true);
            }
        }
        b.part("steel").sweep(&[Vec3::new(0.0, 0.0, -0.9), Vec3::new(0.0, 0.0, 0.9)], &circle(0.3, 10), true);
        b.finish()
    });
    g.register("cabin", "{}", |_, _| {
        let mut b = MeshBuilder::new();
        b.part("cabin").block(Vec3::new(0.0, -1.7, 0.0), Vec3::new(1.2, 1.2, 1.0));
        b.part("steel").sweep(&[Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, 0.0, 0.0)], &circle(0.04, 4), true);
        b.finish()
    });
    g.register("car", "{}", |_, _| {
        let mut b = MeshBuilder::new();
        b.part("paint").block(Vec3::new(0.0, 0.35, 0.0), Vec3::new(1.8, 0.7, 4.2));
        b.part("paint").block(Vec3::new(0.0, 1.05, 0.3), Vec3::new(1.6, 0.6, 2.2));
        b.part("glow").block(Vec3::new(0.6, 0.5, -2.11), Vec3::new(0.3, 0.15, 0.05));
        b.part("glow").block(Vec3::new(-0.6, 0.5, -2.11), Vec3::new(0.3, 0.15, 0.05));
        b.finish()
    });
}

fn materials(ctx: &mut Ctx) {
    let add = |ctx: &mut Ctx, part: &str, m: Material| {
        let id = ctx.renderer.add_material(m);
        ctx.generators.materials.insert(part.to_string(), id);
    };
    add(ctx, "trunk", Material::color(0x4a3222).rough(0.95));
    add(ctx, "leaves", Material::color(0x1f4a24).rough(0.85));
    add(ctx, "metal", Material::color(0x2b2f36).metal(0.8).rough(0.4));
    add(ctx, "glow", Material { emissive: Vec3::new(3.0, 2.2, 1.2), ..Material::color(0xffe2b0) });
    add(ctx, "wood", Material::color(0x8a6a48).rough(0.9));
    let stone = ctx.generate_texture(&TexGen::new(TexPreset::Stone).size(256).seed(3));
    add(ctx, "stone", Material { texture: stone, ..Material::color(0xd8d2c8).rough(0.8) });
    let bricks = ctx.generate_texture(&TexGen::new(TexPreset::Bricks).size(256).seed(5));
    add(ctx, "wall", Material { texture: bricks, ..Material::color(0xffffff).rough(0.85) });
    add(ctx, "roof", Material::color(0x7a2f22).rough(0.7));
    add(ctx, "steel", Material::color(0xd8dde4).metal(0.7).rough(0.35));
    add(ctx, "cabin", Material::color(0xe9c46a).rough(0.5));
    add(ctx, "paint", Material::color(0x2a6fdb).metal(0.6).rough(0.3));
    let rock = ctx.renderer.add_material(Material::color(0x8b8780).rough(0.9));
    ctx.generators.materials.insert("rock".into(), rock);
}

fn rock_mesh(ctx: &mut Ctx) -> MeshId {
    let mut md = MeshData::sphere(1.0, 10, 6);
    for (i, v) in md.vertices.iter_mut().enumerate() {
        let k = 0.75 + 0.35 * (((i * 7919) % 13) as f32 / 13.0);
        v.pos = [v.pos[0] * k * 1.3, v.pos[1] * k * 0.7, v.pos[2] * k];
    }
    engine::geom::smooth_normals(&mut md, 60.0);
    ctx.renderer.add_mesh("rock", &md)
}

#[derive(Default)]
pub struct WorldScreen {
    hero: Option<Entity>,
    hero_model: Option<Entity>,
    camera: Option<Entity>,
    night: bool,
    car_lights: Vec<Entity>,
    blink: f32,
}

impl WorldScreen {
    fn spawn_spectacles(&mut self, ctx: &mut Ctx) {
        let ground = ctx.terrain_height(TOWN.x + 30.0, TOWN.y - 30.0);
        // Ferris wheel: the wheel spins (Motion), cabins are children spinning backwards (stay upright)
        let base = ctx.local(DVec3::new(30.0, ground as f64 + 11.0, -30.0));
        let wheel = ctx.spawn_generated("ferris_wheel", &json!({"radius": 9.0}), 1, Transform::from_translation(base)).unwrap();
        let _ = ctx.world.insert_one(wheel, Motion::spin(Vec3::Z, 0.12));
        for k in 0..12 {
            let a = k as f32 / 12.0 * std::f32::consts::TAU;
            let c = ctx.spawn_generated("cabin", &json!({}), 1, Transform::from_xyz(a.cos() * 9.0, a.sin() * 9.0, 0.0)).unwrap();
            let _ = ctx.world.insert_one(c, Parent(wheel));
            let _ = ctx.world.insert_one(c, Motion::spin(Vec3::Z, -0.12));
        }
        // stand: two legs
        let leg = ctx.part_material("steel");
        // A-frame legs on both sides of the wheel; the two legs of a side sit 6 cm apart in z so
        // their faces never share a plane where they cross
        for (x, z) in [(-4.0f32, -1.0f32), (4.0, -1.06), (-4.0, 1.0), (4.0, 1.06)] {
            let e = ctx.spawn_mesh("wheel-leg", ctx.renderer.builtin.cube, leg, Transform::from_translation(base + Vec3::new(x * 0.5, -5.5, z)).with_rotation(Quat::from_rotation_z(x.signum() * 0.35)).with_scale(Vec3::new(0.35, 12.0, 0.35)));
            let _ = e;
        }
        // car on the ring road: a pivot spins, the car is its child at the ring radius
        let pivot = ctx.world.spawn((Transform::from_translation(ctx.local(DVec3::new(0.0, 0.0, 0.0))), Name("car-pivot".into()), Motion::spin(Vec3::Y, 0.09)));
        let car = ctx.spawn_generated("car", &json!({}), 1, Transform::from_xyz(RING_R, ctx.terrain_height(RING_R, 0.0) + 0.05, 0.0)).unwrap();
        let _ = ctx.world.insert_one(car, Parent(pivot));
        // headlights: two spot lights pointing forward (-Z of the car)
        for x in [-0.6f32, 0.6] {
            let l = ctx.world.spawn((
                Transform::from_xyz(x, 0.6, -2.2).looking_at(Vec3::new(x, 0.1, -12.0), Vec3::Y),
                Parent(car),
                Name("headlight".into()),
                SpotLight { color: Vec3::new(1.0, 0.95, 0.85), intensity: 60.0, range: 35.0, inner_deg: 14.0, outer_deg: 24.0 },
            ));
            self.car_lights.push(l);
        }
        // police-style blinkers on the roof (driven in update)
        for (x, c) in [(-0.4f32, Vec3::new(1.0, 0.1, 0.1)), (0.4, Vec3::new(0.1, 0.25, 1.0))] {
            let l = ctx.world.spawn((Transform::from_xyz(x, 1.5, 0.3), Parent(car), Name("blinker".into()), PointLight { color: c, intensity: 0.0, range: 12.0 }));
            self.car_lights.push(l);
        }
        // pavilion on the plaza
        let pz = ctx.terrain_height(-25.0, 20.0);
        ctx.spawn_generated("pavilion", &json!({"size": 9.0, "height": 3.4}), 1, Transform::from_translation(ctx.local(DVec3::new(-25.0, pz as f64, 20.0)))).unwrap();
        ctx.physics.add_static(ColliderBuilder::cuboid(4.5, 0.15, 4.5), ctx.local(DVec3::new(-25.0, pz as f64 + 0.15, 20.0)));
    }

    /// Night = 22:30, day = 15:30 on the day/night cycle (sun, moon, sky, ambient follow).
    fn set_night(&mut self, ctx: &mut Ctx, night: bool) {
        self.night = night;
        let s = &mut ctx.renderer.settings;
        s.day_cycle = true;
        s.time_of_day = if night { 22.5 } else { 15.5 };
    }

    fn teleport(&mut self, ctx: &mut Ctx, abs: DVec3) {
        let Some(h) = self.hero else { return };
        let y = ctx.terrain_height(abs.x as f32, abs.z as f32) + 2.0;
        let p = ctx.local(DVec3::new(abs.x, y as f64, abs.z));
        if let Ok(mut t) = ctx.world.get::<&mut Transform>(h) {
            t.translation = p;
        }
        if let Ok(mut cc) = ctx.world.get::<&mut CharacterController>(h) {
            cc.velocity = Vec3::ZERO;
        }
    }
}

impl Screen<Data> for WorldScreen {
    fn name(&self) -> &'static str {
        "world"
    }

    fn enter(&mut self, ctx: &mut Ctx, data: &mut Data) {
        data.ensure(ctx);
        crate::sky_setup(ctx);
        register_generators(ctx);
        materials(ctx);
        // --- terrain: island with a flattened town and two roads
        let t0 = engine::log::now_ms();
        let desc = TerrainDesc { origin: Vec2::new(-512.0, -512.0), size: 1024.0, res: 513, chunk_quads: 32, stream_radius: 650.0, ..Default::default() };
        let mut terrain = Terrain::noise(desc, NoiseParams { seed: 9, amplitude: 75.0, frequency: 2.2, ridged: 0.45, island: 0.25, base: 2.0, ..Default::default() });
        let town_h = terrain.height_at(TOWN.x, TOWN.y).max(6.0);
        terrain.flatten(TOWN, TOWN_R, town_h, 40.0);
        let ring: Vec<Vec2> = (0..=48).map(|k| {
            let a = k as f32 / 48.0 * std::f32::consts::TAU;
            Vec2::new(a.cos(), a.sin()) * RING_R
        }).collect();
        terrain.flatten_path(&ring, 9.0, 6.0);
        terrain.flatten_path(&[Vec2::new(RING_R, 0.0), Vec2::new(330.0, 40.0), Vec2::new(430.0, 120.0)], 8.0, 6.0);
        terrain.auto_splat(&SplatRules { low: 4.0, high: 62.0, ..Default::default() });
        for p in ring.iter() {
            terrain.paint(3, *p, 4.5, 1.0, 1.5); // dirt road
        }
        terrain.paint(2, TOWN, TOWN_R * 0.55, 0.85, 12.0); // sandy plaza
        // scatter: pines (trunk + leaves, same seed = same spots), rocks on steep ground
        let pine = ctx.generated_meshes("pine", &json!({"height": 10.0}), 1).unwrap_or_default();
        for (part, mesh) in &pine {
            let collider = if part == "trunk" { 0.35 } else { 0.0 };
            terrain.scatter.push(ScatterRule { mesh: *mesh, material: ctx.part_material(part), density: 0.55, layer: Some(0), min_height: 5.0, max_slope_deg: 28.0, scale: (0.8, 1.3), jitter: 0.5, collider_radius: collider, collider_height: 6.0, max_distance: 450.0, seed: 77, ..Default::default() });
        }
        let rock = rock_mesh(ctx);
        terrain.scatter.push(ScatterRule { mesh: rock, material: ctx.part_material("rock"), density: 0.3, layer: Some(1), max_slope_deg: 60.0, scale: (0.6, 1.8), align: true, jitter: 0.5, max_distance: 300.0, seed: 5, ..Default::default() });
        // keep the town free of trees: paint is done, scatter reads splat; also mask by height plot
        ctx.set_terrain(terrain, [TexPreset::Grass, TexPreset::Stone, TexPreset::Sand, TexPreset::Dirt]);
        engine::log_info!("world terrain ready in {:.0} ms", engine::log::now_ms() - t0);
        ctx.renderer.grass_settings().layer = 0;

        // sea
        let water_mat = ctx.renderer.add_material(Material {
            wave_height: 0.2,
            wave_scale: 0.25,
            water_chop: 0.9,
            water_clarity: 3.0,
            water_vertex_spacing: 0.4,
            ..Material::water(0x06304a, 0x1f8a9a)
        });
        let sea_y = ctx.local(DVec3::new(0.0, -0.5, 0.0)).y;
        let w = ctx.spawn_ocean(water_mat, sea_y);
        if let Ok(mut mr) = ctx.world.get::<&mut MeshRenderer>(w) {
            mr.cast_shadows = false;
        }

        // streamed prop cells: lamps along the ring road + fences on the plaza, batched per cell
        ctx.add_stream_layer("props", 96.0, 420.0, |ctx, cell| {
            let root = ctx.world.spawn((Transform::from_translation(ctx.local(cell.min)), Name(format!("cell {},{}", cell.x, cell.z))));
            let lamp = ctx.generated_meshes("lamp", &json!({"height": 5.0}), 1).unwrap_or_default();
            let mut placed = 0;
            // ring road lamps every ~12 m that fall inside this cell
            for k in 0..80 {
                let a = k as f32 / 80.0 * std::f32::consts::TAU;
                let p = Vec2::new(a.cos(), a.sin()) * (RING_R + 6.0);
                let (lx, lz) = (p.x as f64 - cell.min.x, p.y as f64 - cell.min.z);
                if lx < 0.0 || lz < 0.0 || lx >= cell.size || lz >= cell.size {
                    continue;
                }
                let y = ctx.terrain_height(p.x, p.y);
                let t = Transform::from_xyz(lx as f32, y, lz as f32).with_rotation(Quat::from_rotation_y(-a + std::f32::consts::PI));
                let post = ctx.world.spawn((t, Parent(root), Name("lamp".into())));
                for (part, mesh) in &lamp {
                    ctx.world.spawn((Transform::default(), Parent(post), MeshRenderer::new(*mesh, ctx.part_material(part))));
                }
                ctx.world.spawn((Transform::from_xyz(0.8, 4.8, 0.0), Parent(post), PointLight { color: Vec3::new(1.0, 0.72, 0.4), intensity: 14.0, range: 14.0 }));
                placed += 1;
            }
            // plaza lamps on an 18 m grid
            for gx in -4..=4 {
                for gz in -4..=4 {
                    let p = Vec2::new(gx as f32 * 18.0 + 9.0, gz as f32 * 18.0 + 9.0);
                    let (lx, lz) = (p.x as f64 - cell.min.x, p.y as f64 - cell.min.z);
                    if p.length() > TOWN_R * 0.75 || lx < 0.0 || lz < 0.0 || lx >= cell.size || lz >= cell.size || (p - Vec2::new(-25.0, 20.0)).length() < 9.0 || (p - Vec2::new(30.0, -30.0)).length() < 12.0 {
                        continue; // not on the pavilion / under the wheel
                    }
                    let y = ctx.terrain_height(p.x, p.y);
                    let post = ctx.world.spawn((Transform::from_xyz(lx as f32, y, lz as f32).with_rotation(Quat::from_rotation_y((gx * 3 + gz) as f32)), Parent(root), Name("plaza-lamp".into())));
                    for (part, mesh) in &lamp {
                        ctx.world.spawn((Transform::default(), Parent(post), MeshRenderer::new(*mesh, ctx.part_material(part))));
                    }
                    let warm = if (gx + gz) % 3 == 0 { Vec3::new(0.5, 0.8, 1.0) } else { Vec3::new(1.0, 0.7, 0.4) };
                    ctx.world.spawn((Transform::from_xyz(0.8, 4.8, 0.0), Parent(post), PointLight { color: warm, intensity: 16.0, range: 15.0 }));
                    placed += 1;
                }
            }
            // fences around the plaza edge in this cell
            let fence = ctx.generated_meshes("fence", &json!({"length": 8.0}), 1).unwrap_or_default();
            for k in 0..60 {
                let a = k as f32 / 60.0 * std::f32::consts::TAU;
                let p = Vec2::new(a.cos(), a.sin()) * (TOWN_R - 8.0);
                let (lx, lz) = (p.x as f64 - cell.min.x, p.y as f64 - cell.min.z);
                if lx < 0.0 || lz < 0.0 || lx >= cell.size || lz >= cell.size || k % 6 == 0 {
                    continue; // gaps for paths
                }
                let y = ctx.terrain_height(p.x, p.y);
                let t = Transform::from_xyz(lx as f32, y, lz as f32).with_rotation(Quat::from_rotation_y(-a - std::f32::consts::FRAC_PI_2));
                for (part, mesh) in &fence {
                    ctx.world.spawn((t, Parent(root), MeshRenderer::new(*mesh, ctx.part_material(part))));
                }
            }
            let _ = placed;
            ctx.batch_static(root, 48.0); // static parts of the cell merge per material
        });
        if let Some(l) = ctx.streaming.layers.iter_mut().find(|l| l.name == "props") {
            l.budget = 3;
        }

        // hero: character controller + animated critter, third-person camera
        let cc = CharacterController { speed: 6.0, run_multiplier: 3.5, jump_speed: 7.5, ..Default::default() };
        let foot = cc.foot_offset();
        let start = DVec3::new(-10.0, 0.0, 70.0); // on the plaza, pavilion + Ferris wheel ahead
        let hero = ctx.spawn_character("hero", ctx.local(start) + Vec3::Y * (town_h + 1.0), cc);
        let model = ctx.load_gltf("assets/critter.glb", Transform::from_xyz(0.0, -foot, 0.0).with_scale(Vec3::splat(0.7)));
        let _ = ctx.world.insert_one(model, Parent(hero));
        let _ = ctx.world.insert_one(hero, WorldLabel::new("You"));
        self.hero = Some(hero);
        self.hero_model = Some(model);
        self.camera = Some(ctx.world.spawn((
            Name("camera".into()),
            Transform::from_xyz(0.0, town_h + 8.0, 60.0),
            Camera { far: 2500.0, ..Default::default() },
            ThirdPersonCamera { distance: 9.0, max_distance: 60.0, pitch: 0.3, ..ThirdPersonCamera::following(hero) },
        )));
        self.spawn_spectacles(ctx);
        ctx.ui.set(
            "help",
            "<div class='panel'><b>Open world</b> — 1 km terrain, streaming, grass, lights<br>\
             <kbd>WASD</kbd> walk · <kbd>Shift</kbd> run · <kbd>Space</kbd> jump · <kbd>Right drag</kbd> camera<br>\
             <kbd>N</kbd> night · <kbd>L</kbd> day cycle · <kbd>[</kbd> <kbd>]</kbd> time · <kbd>T</kbd> teleport (island edge) · <kbd>H</kbd> back to town · <kbd>P</kbd> pause</div>",
        );
    }

    fn exit(&mut self, ctx: &mut Ctx, _data: &mut Data) {
        ctx.remove_stream_layer("props");
        ctx.remove_terrain();
        let d = engine::renderer::RenderSettings::default();
        let s = &mut ctx.renderer.settings;
        s.sun_intensity = d.sun_intensity;
        s.sky_zenith = d.sky_zenith;
        s.sky_horizon = d.sky_horizon;
        s.sky_color = d.sky_color;
        s.ground_color = d.ground_color;
        s.exposure = d.exposure;
        let o = ctx.origin;
        ctx.shift_origin(-o);
        for slot in ["help", "world"] {
            ctx.ui.set(slot, "");
        }
    }

    fn fixed_update(&mut self, ctx: &mut Ctx, _data: &mut Data) {
        let yaw = self.camera.and_then(|c| ctx.world.get::<&ThirdPersonCamera>(c).ok().map(|c| c.yaw)).unwrap_or(0.0);
        let axis = ctx.input.move_axis();
        let (run, jump) = (ctx.input.key("ShiftLeft") || ctx.input.key("ShiftRight"), ctx.input.key_down_or_pressed("Space"));
        if let Some(h) = self.hero {
            if let Ok(mut cc) = ctx.world.get::<&mut CharacterController>(h) {
                cc.drive(axis, yaw);
                cc.run = run;
                cc.jump = jump;
            }
        }
    }

    fn update(&mut self, ctx: &mut Ctx, _data: &mut Data, dt: f32) -> Trans<Data> {
        crate::sky_keys(ctx); // N day/night, L run the cycle, [ ] time of day
        self.night = ctx.renderer.sky.night > 0.5;
        if ctx.input.key_pressed("KeyT") {
            self.teleport(ctx, DVec3::new(380.0, 0.0, -380.0));
        }
        if ctx.input.key_pressed("KeyH") {
            self.teleport(ctx, DVec3::new(-10.0, 0.0, 70.0));
        }
        // animation follows the controller
        if let (Some(h), Some(m)) = (self.hero, self.hero_model) {
            if let Ok((g, s)) = ctx.world.get::<&CharacterController>(h).map(|c| (c.grounded, c.velocity.with_y(0.0).length())) {
                let (clip, speed) = if !g { ("bounce", 1.0) } else if s > 0.3 { ("wave", 1.0 + s * 0.15) } else { ("idle", 1.0) };
                ctx.play_animation(m, clip, PlayOpts::looped().fade(0.2).speed(speed));
            }
        }
        // police-style blinkers
        self.blink += dt;
        let on = (self.blink * 4.0) as i32 % 2 == 0;
        for (k, l) in self.car_lights.iter().enumerate().skip(2) {
            if let Ok(mut pl) = ctx.world.get::<&mut PointLight>(*l) {
                pl.intensity = if (k % 2 == 0) == on { 30.0 } else { 0.0 };
            }
        }
        let r = &ctx.renderer.stats;
        let cam = ctx.camera_absolute().unwrap_or_default();
        let chunks = ctx.terrain.as_ref().map(|t| t.loaded_chunks()).unwrap_or(0);
        let cells = ctx.streaming.layers.iter().map(|l| l.loads - l.unloads).sum::<u64>();
        ctx.ui.set(
            "world",
            &format!(
                "<div class='panel fps'>{:.0} fps · {} draws · {} lights · {} grass<br>chunks {} · cells {} · pos {:.0},{:.0} · origin {:.0},{:.0}</div>",
                ctx.time.fps, r.draw_calls, r.lights.visible, r.grass_blades, chunks, cells, cam.x, cam.z, ctx.origin.x, ctx.origin.z
            ),
        );
        if ctx.input.key_pressed("KeyP") || ctx.input.key_pressed("Escape") {
            return Trans::Push(Box::new(crate::Pause));
        }
        Trans::None
    }

    fn debug_commands(&self) -> Vec<(&'static str, &'static str)> {
        vec![("night", "Toggle night. args: {on?: bool}"), ("teleport", "Move the hero to an absolute position. args: {pos: [x, z]}")]
    }

    fn debug_command(&mut self, ctx: &mut Ctx, _data: &mut Data, name: &str, a: &Value) -> Option<Result<Value, String>> {
        match name {
            "night" => {
                let on = a["on"].as_bool().unwrap_or(!self.night);
                self.set_night(ctx, on);
                Some(Ok(json!({"night": on})))
            }
            "teleport" => {
                let p = a["pos"].as_array()?;
                let (x, z) = (p.first()?.as_f64()?, p.get(1)?.as_f64()?);
                self.teleport(ctx, DVec3::new(x, 0.0, z));
                Some(Ok(json!({"pos": [x, z]})))
            }
            _ => None,
        }
    }

    fn debug_state(&self, ctx: &Ctx, _data: &Data) -> Value {
        let hero = self.hero.and_then(|h| {
            let t = ctx.world.get::<&Transform>(h).ok()?.translation;
            let cc = ctx.world.get::<&CharacterController>(h).ok()?;
            let abs = ctx.to_absolute(t);
            Some(json!({"local": t.to_array(), "absolute": [abs.x, abs.y, abs.z], "grounded": cc.grounded, "ground_height": ctx.terrain_height(abs.x as f32, abs.z as f32)}))
        });
        json!({"night": self.night, "hero": hero, "origin": [ctx.origin.x, ctx.origin.z]})
    }
}

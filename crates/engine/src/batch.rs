//! Static batching and instance motion.
//!
//! **Static batching** ([`Ctx::batch_static`]): thousands of small static props (fences, lamp
//! posts, curbs, generated buildings) become a few merged meshes — one per (material, layer,
//! shadow flag, grid cell). Cells keep frustum culling effective; the originals lose their
//! `MeshRenderer` (their colliders stay). Use it after a chunk / scene was built.
//!
//! **HLOD** ([`Ctx::batch_static_with`] + [`Hlod`]): additionally every grid cell gets one
//! simplified mesh of *all* its batches (vertex clustering at `Hlod::cluster` metres, material base
//! colours baked into vertex colours, one material): beyond `Hlod::distance` the cell is drawn as
//! that single mesh instead of its per-material batches. Textures are not baked (base colours only).
//!
//! **Motion** ([`Motion`]): transform animation for moving parts without game code — Ferris
//! wheels (spin), cabins that stay upright (child with the opposite spin), carousels (spin + bob),
//! cranes (sway), signs (bob). Evaluated in the fixed step relative to the transform the entity
//! had when the motion started, so it is deterministic and works with `pause`/`step`.

use crate::app::Ctx;
use crate::geom;
use crate::mesh::MeshData;
use crate::renderer::{MaterialId, MeshId};
use crate::scene::{GlobalTransform, MeshRenderer, Name, Transform};
use glam::{Quat, Vec3};
use hecs::Entity;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Procedural transform animation (all optional parts are summed).
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Motion {
    /// Spin around a local axis: (axis, radians per second).
    pub spin: Option<(Vec3, f32)>,
    /// Back-and-forth rotation: (axis, amplitude rad, frequency Hz, phase rad).
    pub sway: Option<(Vec3, f32, f32, f32)>,
    /// Translation wobble: (amplitude per axis, frequency Hz, phase rad).
    pub bob: Option<(Vec3, f32, f32)>,
    /// Elapsed time (s); set a different start per instance to desynchronise copies.
    pub time: f32,
    #[serde(skip)]
    base: Option<Transform>,
}

impl Motion {
    pub fn spin(axis: Vec3, rad_per_s: f32) -> Self {
        Self { spin: Some((axis.normalize_or(Vec3::Y), rad_per_s)), ..Default::default() }
    }
    pub fn sway(axis: Vec3, amplitude: f32, hz: f32) -> Self {
        Self { sway: Some((axis.normalize_or(Vec3::Y), amplitude, hz, 0.0)), ..Default::default() }
    }
    pub fn bob(amplitude: Vec3, hz: f32) -> Self {
        Self { bob: Some((amplitude, hz, 0.0)), ..Default::default() }
    }
    pub fn with_bob(mut self, amplitude: Vec3, hz: f32, phase: f32) -> Self {
        self.bob = Some((amplitude, hz, phase));
        self
    }
    pub fn starting_at(mut self, t: f32) -> Self {
        self.time = t;
        self
    }
    /// Origin shift: the captured base moves with the world.
    pub(crate) fn shift(&mut self, d: Vec3) {
        if let Some(b) = self.base.as_mut() {
            b.translation -= d;
        }
    }

    /// Forget the captured base transform (after moving the entity by hand).
    pub fn rebase(&mut self) {
        self.base = None;
    }
}

/// Fixed step: advance every Motion and write its transform.
pub(crate) fn step_motion(world: &mut hecs::World, dt: f32) {
    for (t, m) in world.query_mut::<(&mut Transform, &mut Motion)>() {
        let base = *m.base.get_or_insert(*t);
        m.time += dt;
        let tt = m.time;
        let mut rot = base.rotation;
        let mut pos = base.translation;
        if let Some((axis, w)) = m.spin {
            rot *= Quat::from_axis_angle(axis, w * tt);
        }
        if let Some((axis, amp, hz, ph)) = m.sway {
            rot *= Quat::from_axis_angle(axis, amp * (tt * hz * std::f32::consts::TAU + ph).sin());
        }
        if let Some((amp, hz, ph)) = m.bob {
            pos += amp * (tt * hz * std::f32::consts::TAU + ph).sin();
        }
        t.rotation = rot;
        t.translation = pos;
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct BatchReport {
    pub sources: usize,
    pub skipped: usize,
    pub batches: usize,
    pub triangles: usize,
    pub draws_before: usize,
    /// HLOD meshes (one per cell) and their triangles.
    pub hlods: usize,
    pub hlod_triangles: usize,
}

/// Far representation of static batches (see the module docs).
#[derive(Clone, Copy, Debug, Serialize, serde::Deserialize)]
pub struct Hlod {
    /// Camera distance (m, from the cell's centre, times `render.lod_distance_scale`) beyond which
    /// a cell is drawn as its HLOD mesh.
    pub distance: f32,
    /// Vertex-clustering cell size (m): detail smaller than this merges (0.4 m bricks: ~0.5-1).
    pub cluster: f32,
}

impl Default for Hlod {
    fn default() -> Self {
        Self { distance: 150.0, cluster: 0.75 }
    }
}

/// Marks the HLOD mesh of a static-batch cell.
#[derive(Clone, Copy, Debug)]
pub struct HlodMesh {
    pub batches: u32,
}

impl Ctx {
    /// Merges the static meshes below `root` (inclusive) into batches per (material, layer,
    /// shadow flag, `cell` x `cell` m grid cell). Skips anything that moves or is special:
    /// dynamic/kinematic bodies, `Motion`, skinned meshes, LOD chains, world panels, meshes
    /// without a CPU copy. Merged entities are children of `root`, so they unload with it.
    pub fn batch_static(&mut self, root: Entity, cell: f32) -> BatchReport {
        self.batch_static_with(root, cell, None)
    }

    /// [`Self::batch_static`] plus, with `hlod`, one simplified mesh per cell drawn beyond
    /// `hlod.distance` instead of the cell's batches (see the module docs).
    pub fn batch_static_with(&mut self, root: Entity, cell: f32, hlod: Option<Hlod>) -> BatchReport {
        crate::scene::propagate_transforms(&mut self.world);
        let cell = cell.max(1.0);
        let mut groups: BTreeMap<(u32, u8, bool, i32, i32), Vec<(Entity, MeshId, glam::Mat4)>> = BTreeMap::new();
        let mut rep = BatchReport::default();
        let mut keys_before = std::collections::HashSet::new();
        for e in crate::scene::descendants(&self.world, root) {
            let Ok(mr) = self.world.get::<&MeshRenderer>(e).map(|m| *m) else { continue };
            let moving = self.world.get::<&Motion>(e).is_ok()
                || self.world.get::<&crate::animation::Deformer>(e).is_ok()
                || self.world.get::<&crate::renderer::Lod>(e).is_ok()
                || self.world.get::<&crate::panel::WorldPanel>(e).is_ok()
                || self.physics.body(&self.world, e).is_some_and(|b| !b.is_fixed())
                || ancestors(&self.world, e).iter().any(|a| self.world.get::<&Motion>(*a).is_ok() || self.physics.body(&self.world, *a).is_some_and(|b| !b.is_fixed()));
            let has_cpu = self.renderer.mesh(mr.mesh).is_some_and(|m| m.data.is_some());
            if moving || !has_cpu {
                rep.skipped += 1;
                continue;
            }
            let g = self.world.get::<&GlobalTransform>(e).map(|g| g.0).unwrap_or_default();
            let c = g.w_axis;
            keys_before.insert((mr.material.0, mr.mesh.0));
            let key = (mr.material.0, mr.layer, mr.cast_shadows, (c.x / cell).floor() as i32, (c.z / cell).floor() as i32);
            groups.entry(key).or_default().push((e, mr.mesh, g));
            rep.sources += 1;
        }
        rep.draws_before = keys_before.len();
        let root_g = self.world.get::<&GlobalTransform>(root).map(|g| g.0).unwrap_or_default();
        let root_inv = root_g.inverse();
        let root_scale = root_g.x_axis.truncate().length().max(root_g.y_axis.truncate().length()).max(1e-4);
        // merge first: with HLOD every batch of a cell and the cell's HLOD share one origin (the
        // cell's foot centre), so they switch at exactly the same camera distance
        let mut merged_all = vec![];
        for ((mat, layer, shadows, cx, cz), items) in groups {
            if items.len() < 2 {
                continue; // nothing to gain
            }
            let mut merged = MeshData::default();
            for (_, mesh, m) in &items {
                if let Some(src) = self.renderer.mesh(*mesh).and_then(|g| g.data.clone()) {
                    let mut md = (*src).clone();
                    geom::transform(&mut md, root_inv * *m);
                    let base = merged.vertices.len() as u32;
                    merged.vertices.extend(md.vertices);
                    merged.indices.extend(md.indices.iter().map(|i| i + base));
                }
            }
            if merged.indices.is_empty() {
                continue;
            }
            merged_all.push(((mat, layer, shadows, cx, cz), items, merged));
        }
        let mut centres: BTreeMap<(u8, bool, i32, i32), (Vec3, Vec3)> = BTreeMap::new();
        if hlod.is_some() {
            for ((_, layer, shadows, cx, cz), _, merged) in &merged_all {
                let b = centres.entry((*layer, *shadows, *cx, *cz)).or_insert((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)));
                for v in &merged.vertices {
                    b.0 = b.0.min(Vec3::from(v.pos));
                    b.1 = b.1.max(Vec3::from(v.pos));
                }
            }
        }
        // per cell (layer, shadows, cx, cz): (batch entity, material, mesh in root space)
        let mut cells: BTreeMap<(u8, bool, i32, i32), Vec<(Entity, u32, MeshData)>> = BTreeMap::new();
        for ((mat, layer, shadows, cx, cz), items, merged) in merged_all {
            rep.triangles += merged.indices.len() / 3;
            let centre = centres.get(&(layer, shadows, cx, cz)).map_or(Vec3::ZERO, |(mn, mx)| Vec3::new((mn.x + mx.x) * 0.5, mn.y, (mn.z + mx.z) * 0.5));
            let mut local = merged.clone();
            if centre != Vec3::ZERO {
                for v in local.vertices.iter_mut() {
                    v.pos = (Vec3::from(v.pos) - centre).into();
                }
            }
            let id = self.renderer.add_mesh(&format!("batch:{mat}:{cx}:{cz}"), &local);
            let mut mr = MeshRenderer::new(id, MaterialId(mat));
            mr.layer = layer;
            mr.cast_shadows = shadows;
            let e = self.world.spawn((Transform::from_translation(centre), crate::scene::Parent(root), mr, Name(format!("batch {cx},{cz}")), StaticBatch { sources: items.len() as u32 }));
            if hlod.is_some() {
                cells.entry((layer, shadows, cx, cz)).or_default().push((e, mat, merged));
            }
            if let Ok(tag) = self.world.get::<&crate::screens::SceneTag>(root).map(|t| *t) {
                let _ = self.world.insert_one(e, tag);
            }
            for (src, _, _) in &items {
                let _ = self.world.remove_one::<MeshRenderer>(*src);
            }
            rep.batches += 1;
        }
        if let Some(h) = hlod {
            self.build_hlods(root, &cells, h, root_scale, &mut rep);
        }
        if rep.batches > 0 {
            crate::log_info!(
                "batch_static: {} meshes -> {} batches ({} skipped, {} triangles){}",
                rep.sources,
                rep.batches,
                rep.skipped,
                rep.triangles,
                if rep.hlods > 0 { format!(", {} HLOD cells ({} triangles)", rep.hlods, rep.hlod_triangles) } else { String::new() }
            );
        }
        rep
    }

    #[allow(clippy::type_complexity)]
    fn build_hlods(&mut self, root: Entity, cells: &BTreeMap<(u8, bool, i32, i32), Vec<(Entity, u32, MeshData)>>, h: Hlod, root_scale: f32, rep: &mut BatchReport) {
        let mut hmat: Option<MaterialId> = None;
        let tag = self.world.get::<&crate::screens::SceneTag>(root).map(|t| *t).ok();
        for ((layer, shadows, cx, cz), parts) in cells {
            // all materials of the cell in one mesh, base colours as vertex colours
            let mut all = MeshData::default();
            let mut rough = (0.0f32, 0.0f32);
            for (_, mat, md) in parts {
                let m = self.renderer.material(MaterialId(*mat)).cloned().unwrap_or_default();
                let enc = |x: f32| (if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.max(0.0).powf(1.0 / 2.4) - 0.055 }).clamp(0.0, 1.0);
                let b = m.base_color;
                let col = [(enc(b.x) * 255.0).round() as u8, (enc(b.y) * 255.0).round() as u8, (enc(b.z) * 255.0).round() as u8, 255];
                let base = all.vertices.len() as u32;
                all.vertices.extend(md.vertices.iter().map(|v| crate::mesh::Vertex { color: crate::mesh::Vertex::mul_color(v.color, col), ..*v }));
                all.indices.extend(md.indices.iter().map(|i| i + base));
                let w = (md.indices.len() / 3) as f32;
                rough = (rough.0 + m.roughness * w, rough.1 + w);
            }
            let centre = foot_centre(&all);
            let mut simple = all.simplify_cell_size(h.cluster.max(0.01) / root_scale);
            if simple.indices.is_empty() {
                continue;
            }
            for v in simple.vertices.iter_mut() {
                v.pos = (Vec3::from(v.pos) - centre).into();
            }
            let mat = *hmat.get_or_insert_with(|| {
                let r = if rough.1 > 0.0 { rough.0 / rough.1 } else { 0.6 };
                self.renderer.add_material(crate::renderer::Material { name: "hlod".into(), roughness: r, ..Default::default() })
            });
            let d = h.distance.max(0.0) / root_scale;
            let id = self.renderer.add_mesh(&format!("hlod:{cx}:{cz}"), &simple);
            let mut mr = MeshRenderer::new(id, mat);
            mr.layer = *layer;
            mr.cast_shadows = *shadows;
            let empty = self.renderer.empty_mesh();
            let e = self.world.spawn((
                Transform::from_translation(centre),
                crate::scene::Parent(root),
                mr,
                Name(format!("hlod {cx},{cz}")),
                crate::renderer::Lod { meshes: vec![empty, id], distances: vec![d], cull_beyond: false },
                HlodMesh { batches: parts.len() as u32 },
                crate::zfight::NoZCheck,
            ));
            if let Some(t) = tag {
                let _ = self.world.insert_one(e, t);
            }
            // each batch: its own mesh up to d, then nothing (the HLOD takes over)
            for (b, _, _) in parts {
                let Ok(mesh) = self.world.get::<&MeshRenderer>(*b).map(|m| m.mesh) else { continue };
                let _ = self.world.insert_one(*b, crate::renderer::Lod { meshes: vec![mesh], distances: vec![d], cull_beyond: true });
            }
            rep.hlods += 1;
            rep.hlod_triangles += simple.indices.len() / 3;
        }
    }

    /// AI API `batch`.
    pub(crate) fn batch_command(&mut self, a: &Value) -> Result<Value, String> {
        if let Some(id) = a["id"].as_u64() {
            let e = self.world.iter().map(|r| r.entity()).find(|e| e.id() == id as u32).ok_or("no such entity")?;
            let hlod = if a["hlod"].is_object() {
                let d = Hlod::default();
                Some(Hlod { distance: a["hlod"]["distance"].as_f64().map_or(d.distance, |v| v as f32), cluster: a["hlod"]["cluster"].as_f64().map_or(d.cluster, |v| v as f32) })
            } else {
                None
            };
            let r = self.batch_static_with(e, a["cell"].as_f64().unwrap_or(64.0) as f32, hlod);
            return Ok(json!(r));
        }
        let batches: Vec<Value> = self.world.query::<(Entity, &StaticBatch, &MeshRenderer)>().iter().map(|(e, b, mr)| {
            let tris = self.renderer.mesh(mr.mesh).map(|m| m.index_count / 3).unwrap_or(0);
            json!({"id": e.id(), "sources": b.sources, "triangles": tris, "material": mr.material.0})
        }).collect();
        let motions = self.world.query::<&Motion>().iter().count();
        let hlods: Vec<Value> = self.world.query::<(Entity, &HlodMesh, &crate::renderer::Lod)>().iter().map(|(e, h, l)| {
            let tris = l.meshes.get(1).and_then(|m| self.renderer.mesh(*m)).map(|m| m.index_count / 3).unwrap_or(0);
            json!({"id": e.id(), "batches": h.batches, "triangles": tris, "distance": l.distances.first()})
        }).collect();
        Ok(json!({"batches": batches, "hlods": hlods, "moving_instances": motions}))
    }
}

/// Centre of the bounding box at its lowest point (the origin of a batch / HLOD with LOD).
fn foot_centre(md: &MeshData) -> Vec3 {
    let (mn, mx) = md.vertices.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), v| (a.min(Vec3::from(v.pos)), b.max(Vec3::from(v.pos))));
    if mn.x > mx.x {
        return Vec3::ZERO;
    }
    Vec3::new((mn.x + mx.x) * 0.5, mn.y, (mn.z + mx.z) * 0.5)
}

fn ancestors(world: &hecs::World, e: Entity) -> Vec<Entity> {
    let mut out = vec![];
    let mut cur = e;
    while let Ok(p) = world.get::<&crate::scene::Parent>(cur).map(|p| p.0) {
        if out.len() > 64 {
            break;
        }
        out.push(p);
        cur = p;
    }
    out
}

/// Marks a merged static batch.
#[derive(Clone, Copy, Debug)]
pub struct StaticBatch {
    pub sources: u32,
}

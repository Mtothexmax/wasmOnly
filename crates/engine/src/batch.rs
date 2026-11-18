//! Static batching and instance motion.
//!
//! **Static batching** ([`Ctx::batch_static`]): thousands of small static props (fences, lamp
//! posts, curbs, generated buildings) become a few merged meshes — one per (material, layer,
//! shadow flag, grid cell). Cells keep frustum culling effective; the originals lose their
//! `MeshRenderer` (their colliders stay). Use it after a chunk / scene was built.
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
}

impl Ctx {
    /// Merges the static meshes below `root` (inclusive) into batches per (material, layer,
    /// shadow flag, `cell` x `cell` m grid cell). Skips anything that moves or is special:
    /// dynamic/kinematic bodies, `Motion`, skinned meshes, LOD chains, world panels, meshes
    /// without a CPU copy. Merged entities are children of `root`, so they unload with it.
    pub fn batch_static(&mut self, root: Entity, cell: f32) -> BatchReport {
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
        let root_inv = self.world.get::<&GlobalTransform>(root).map(|g| g.0.inverse()).unwrap_or_default();
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
            rep.triangles += merged.indices.len() / 3;
            let id = self.renderer.add_mesh(&format!("batch:{mat}:{cx}:{cz}"), &merged);
            let mut mr = MeshRenderer::new(id, MaterialId(mat));
            mr.layer = layer;
            mr.cast_shadows = shadows;
            let e = self.world.spawn((Transform::default(), crate::scene::Parent(root), mr, Name(format!("batch {cx},{cz}")), StaticBatch { sources: items.len() as u32 }));
            if let Ok(tag) = self.world.get::<&crate::screens::SceneTag>(root).map(|t| *t) {
                let _ = self.world.insert_one(e, tag);
            }
            for (src, _, _) in &items {
                let _ = self.world.remove_one::<MeshRenderer>(*src);
            }
            rep.batches += 1;
        }
        if rep.batches > 0 {
            crate::log_info!("batch_static: {} meshes -> {} batches ({} skipped, {} triangles)", rep.sources, rep.batches, rep.skipped, rep.triangles);
        }
        rep
    }

    /// AI API `batch`.
    pub(crate) fn batch_command(&mut self, a: &Value) -> Result<Value, String> {
        if let Some(id) = a["id"].as_u64() {
            let e = self.world.iter().map(|r| r.entity()).find(|e| e.id() == id as u32).ok_or("no such entity")?;
            let r = self.batch_static(e, a["cell"].as_f64().unwrap_or(64.0) as f32);
            return Ok(json!(r));
        }
        let batches: Vec<Value> = self.world.query::<(Entity, &StaticBatch, &MeshRenderer)>().iter().map(|(e, b, mr)| {
            let tris = self.renderer.mesh(mr.mesh).map(|m| m.index_count / 3).unwrap_or(0);
            json!({"id": e.id(), "sources": b.sources, "triangles": tris, "material": mr.material.0})
        }).collect();
        let motions = self.world.query::<&Motion>().iter().count();
        Ok(json!({"batches": batches, "moving_instances": motions}))
    }
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

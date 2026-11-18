//! Scene components. The scene *is* the hecs `World`; these are the components the engine knows.

use crate::renderer::{MaterialId, MeshId};
use glam::{Mat4, Quat, Vec3};
use hecs::{Entity, World};
use std::collections::HashMap;

/// Local transform (relative to [`Parent`] if present).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self {
            translation: Vec3::new(x, y, z),
            ..Default::default()
        }
    }
    pub fn from_translation(t: Vec3) -> Self {
        Self {
            translation: t,
            ..Default::default()
        }
    }
    pub fn with_scale(mut self, s: Vec3) -> Self {
        self.scale = s;
        self
    }
    pub fn with_rotation(mut self, r: Quat) -> Self {
        self.rotation = r;
        self
    }
    pub fn looking_at(mut self, target: Vec3, up: Vec3) -> Self {
        let fwd = (target - self.translation).normalize_or(Vec3::NEG_Z);
        let right = fwd.cross(up).normalize_or(Vec3::X);
        let up = right.cross(fwd);
        self.rotation = Quat::from_mat3(&glam::Mat3::from_cols(right, up, -fwd));
        self
    }
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }
}

/// World-space matrix, computed by the engine every frame from `Transform` + `Parent`.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlobalTransform(pub Mat4);

impl GlobalTransform {
    pub fn translation(&self) -> Vec3 {
        self.0.w_axis.truncate()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Parent(pub Entity);

#[derive(Clone, Debug)]
pub struct Name(pub String);

#[derive(Clone, Copy, Debug)]
pub struct MeshRenderer {
    pub mesh: MeshId,
    pub material: MaterialId,
    pub cast_shadows: bool,
    /// Render layer 0..31; a camera draws it if bit `layer` is set in `Camera::layers`.
    pub layer: u8,
}

impl MeshRenderer {
    pub fn new(mesh: MeshId, material: MaterialId) -> Self {
        Self {
            mesh,
            material,
            cast_shadows: true,
            layer: 0,
        }
    }
}

/// Render layer for an entity and everything below it, including glTF nodes that arrive later
/// (see `Ctx::set_layer`, `MeshRenderer::layer`, `Camera::layers`).
#[derive(Clone, Copy, Debug)]
pub struct RenderLayer(pub u8);

/// Perspective camera. The entity's `GlobalTransform` is the camera pose (looks down -Z).
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
    /// Bit mask of `MeshRenderer::layer`s this camera draws (default: all).
    pub layers: u32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov_y_deg: 60.0,
            near: 0.1,
            far: 500.0,
            layers: u32::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
}

/// A DOM element pinned to a 3D position (name plates, markers). The engine positions it via
/// CSS transform each frame. `html` is set when it changes; style via the `.world-label` class.
#[derive(Clone, Debug)]
pub struct WorldLabel {
    pub html: String,
    pub offset: Vec3,
    /// Extra CSS class(es) for this label.
    pub class: String,
}

impl WorldLabel {
    pub fn new(html: impl Into<String>) -> Self {
        Self {
            html: html.into(),
            offset: Vec3::new(0.0, 1.0, 0.0),
            class: String::new(),
        }
    }
}

/// Cheap hasher for entity keys (FxHash): SipHash over thousands of entities per frame showed
/// up as the top cost of an open-world frame.
#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);
impl std::hash::Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write_u64(*b as u64);
        }
    }
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}
pub type FxMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<FxHasher>>;

/// Recomputes `GlobalTransform` for every entity that has a `Transform`: roots directly, children
/// through their parent chain (memoised, cheap hashing).
pub fn propagate_transforms(world: &mut World) {
    let mut missing: Vec<(Entity, Mat4)> = vec![];
    for (e, t, g) in world.query_mut::<hecs::Without<(Entity, &Transform, Option<&mut GlobalTransform>), &Parent>>() {
        let m = t.matrix();
        match g {
            Some(g) => g.0 = m,
            None => missing.push((e, m)),
        }
    }
    for (e, m) in missing.drain(..) {
        let _ = world.insert_one(e, GlobalTransform(m));
    }
    let kids: Vec<(Entity, Entity, Mat4)> = world.query::<(Entity, &Parent, &Transform)>().iter().map(|(e, p, t)| (e, p.0, t.matrix())).collect();
    if kids.is_empty() {
        return;
    }
    // Entity id -> index into `kids` (a dense table instead of a hash map: this runs for thousands
    // of children every frame). The generation is checked on lookup (`kids[i].0 == parent`).
    const NONE: u32 = u32::MAX;
    let table_len = kids.iter().map(|k| k.0.id()).max().unwrap_or(0) as usize + 1;
    let mut slot: Vec<u32> = vec![NONE; table_len];
    for (i, k) in kids.iter().enumerate() {
        slot[k.0.id() as usize] = i as u32;
    }
    let kid_index = |e: Entity| -> Option<usize> {
        let i = *slot.get(e.id() as usize)?;
        (i != NONE && kids[i as usize].0 == e).then_some(i as usize)
    };
    let mut result: Vec<Mat4> = vec![Mat4::IDENTITY; kids.len()];
    let mut done: Vec<bool> = vec![false; kids.len()];
    // Siblings are stored next to each other, so remembering the last root parent saves most
    // world lookups.
    let mut last_root: Option<(Entity, Mat4)> = None;
    let mut chain: Vec<usize> = vec![];
    for start in 0..kids.len() {
        if done[start] {
            continue;
        }
        // walk up until a resolved ancestor or a root, then multiply back down
        chain.clear();
        let mut i = start;
        let mut pm = loop {
            chain.push(i);
            let parent = kids[i].1;
            match kid_index(parent) {
                Some(pi) if done[pi] => break result[pi],
                Some(pi) if chain.len() < 64 && !chain.contains(&pi) => i = pi,
                Some(_) => break Mat4::IDENTITY, // cycle or absurd depth
                None => {
                    break match last_root {
                        Some((e, m)) if e == parent => m,
                        _ => {
                            let m = world.get::<&GlobalTransform>(parent).map(|g| g.0).unwrap_or(Mat4::IDENTITY);
                            last_root = Some((parent, m));
                            m
                        }
                    };
                }
            }
        };
        for &c in chain.iter().rev() {
            pm *= kids[c].2;
            result[c] = pm;
            done[c] = true;
        }
    }
    let mut written = 0usize;
    for (e, g) in world.query_mut::<hecs::With<(Entity, &mut GlobalTransform), (&Parent, &Transform)>>() {
        if let Some(i) = kid_index(e) {
            g.0 = result[i];
            written += 1;
        }
    }
    if written < kids.len() {
        // children that have no GlobalTransform yet (just spawned)
        for k in kids.iter() {
            if world.get::<&GlobalTransform>(k.0).is_err() {
                if let Some(i) = kid_index(k.0) {
                    missing.push((k.0, result[i]));
                }
            }
        }
    }
    for (e, m) in missing {
        let _ = world.insert_one(e, GlobalTransform(m));
    }
}

/// Entities whose `Parent` chain leads to `root` (including root), depth-first.
pub fn descendants(world: &World, root: Entity) -> Vec<Entity> {
    descendants_of(world, &[root])
}

/// Descendants of several roots (including the roots) with one pass over the hierarchy.
pub fn descendants_of(world: &World, roots: &[Entity]) -> Vec<Entity> {
    let mut children: FxMap<Entity, Vec<Entity>> = FxMap::default();
    for (e, p) in world.query::<(Entity, &Parent)>().iter() {
        children.entry(p.0).or_default().push(e);
    }
    let mut out = vec![];
    let mut seen: std::collections::HashSet<Entity, std::hash::BuildHasherDefault<FxHasher>> = Default::default();
    let mut stack: Vec<Entity> = roots.to_vec();
    while let Some(e) = stack.pop() {
        if !seen.insert(e) {
            continue;
        }
        out.push(e);
        if let Some(c) = children.get(&e) {
            stack.extend(c.iter().copied());
        }
    }
    out
}

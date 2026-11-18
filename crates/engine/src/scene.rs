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
    let mut index: FxMap<Entity, usize> = FxMap::default();
    index.reserve(kids.len());
    for (i, k) in kids.iter().enumerate() {
        index.insert(k.0, i);
    }
    let mut result: Vec<Option<Mat4>> = vec![None; kids.len()];
    fn resolve(i: usize, kids: &[(Entity, Entity, Mat4)], index: &FxMap<Entity, usize>, result: &mut [Option<Mat4>], world: &World, depth: u32) -> Mat4 {
        if let Some(m) = result[i] {
            return m;
        }
        let (_, parent, local) = kids[i];
        let pm = match index.get(&parent) {
            Some(&pi) if depth < 64 => resolve(pi, kids, index, result, world, depth + 1),
            Some(_) => Mat4::IDENTITY,
            None => world.get::<&GlobalTransform>(parent).map(|g| g.0).unwrap_or(Mat4::IDENTITY),
        };
        let m = pm * local;
        result[i] = Some(m);
        m
    }
    for i in 0..kids.len() {
        resolve(i, &kids, &index, &mut result, world, 0);
    }
    for (i, k) in kids.iter().enumerate() {
        let m = result[i].unwrap_or(Mat4::IDENTITY);
        match world.get::<&mut GlobalTransform>(k.0) {
            Ok(mut g) => g.0 = m,
            Err(_) => missing.push((k.0, m)),
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

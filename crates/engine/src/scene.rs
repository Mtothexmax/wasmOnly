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
}

impl MeshRenderer {
    pub fn new(mesh: MeshId, material: MaterialId) -> Self {
        Self {
            mesh,
            material,
            cast_shadows: true,
        }
    }
}

/// Perspective camera. The entity's `GlobalTransform` is the camera pose (looks down -Z).
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov_y_deg: 60.0,
            near: 0.1,
            far: 500.0,
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

/// Recomputes `GlobalTransform` for every entity that has a `Transform`.
pub fn propagate_transforms(world: &mut World) {
    let mut locals: HashMap<Entity, (Mat4, Option<Entity>)> = HashMap::new();
    for (e, t, p) in world
        .query::<(Entity, &Transform, Option<&Parent>)>()
        .iter()
    {
        locals.insert(e, (t.matrix(), p.map(|p| p.0)));
    }
    let mut globals: HashMap<Entity, Mat4> = HashMap::with_capacity(locals.len());
    fn resolve(
        e: Entity,
        locals: &HashMap<Entity, (Mat4, Option<Entity>)>,
        globals: &mut HashMap<Entity, Mat4>,
        depth: u32,
    ) -> Mat4 {
        if let Some(m) = globals.get(&e) {
            return *m;
        }
        let Some((local, parent)) = locals.get(&e) else {
            return Mat4::IDENTITY;
        };
        let m = match parent {
            Some(p) if depth < 64 && locals.contains_key(p) => {
                resolve(*p, locals, globals, depth + 1) * *local
            }
            _ => *local,
        };
        globals.insert(e, m);
        m
    }
    let keys: Vec<Entity> = locals.keys().copied().collect();
    for e in keys {
        resolve(e, &locals, &mut globals, 0);
    }
    let mut missing = vec![];
    for (e, m) in &globals {
        match world.get::<&mut GlobalTransform>(*e) {
            Ok(mut g) => g.0 = *m,
            Err(_) => missing.push((*e, *m)),
        }
    }
    for (e, m) in missing {
        let _ = world.insert_one(e, GlobalTransform(m));
    }
}

/// Entities whose `Parent` chain leads to `root` (including root), depth-first.
pub fn descendants(world: &World, root: Entity) -> Vec<Entity> {
    let mut children: HashMap<Entity, Vec<Entity>> = HashMap::new();
    for (e, p) in world.query::<(Entity, &Parent)>().iter() {
        children.entry(p.0).or_default().push(e);
    }
    let mut out = vec![];
    let mut stack = vec![root];
    while let Some(e) = stack.pop() {
        out.push(e);
        if let Some(c) = children.get(&e) {
            stack.extend(c.iter().copied());
        }
    }
    out
}

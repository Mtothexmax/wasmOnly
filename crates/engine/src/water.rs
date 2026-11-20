//! Water surfaces.
//!
//! The look comes from the water shading mode in `pbr.wgsl` ([`Material::water`]): Gerstner waves
//! (vertex displacement + per-pixel normal), ripples, refraction with absorption, caustics,
//! screen-space reflections over the sky, sun specular, subsurface glow and foam. This module
//! provides meshes dense enough for the waves:
//!
//! * [`Ctx::spawn_water`] — a bounded surface (pond, lake, pool): a grid whose vertex spacing
//!   matches the material's `water_vertex_spacing` along each axis (at most 65536 cells, so a
//!   long strip gets a long grid, not a square one).
//! * [`Ctx::spawn_ocean`] — an endless sea: a radial mesh (dense in the centre, cells growing with
//!   distance, ~5 km radius) that follows the camera ([`Ocean`]), so detail is always where you look.
//!
//! Waves shorter than about two vertex spacings are left out of the vertex displacement (they
//! would alias) but stay in the per-pixel normal, so even a two-triangle plane shows waves.
//! Render settings: `water_ssr`, `water_refraction`, `water_caustics`.

use crate::app::Ctx;
use crate::mesh::{MeshData, Vertex};
use crate::renderer::{MaterialId, MeshId};
use crate::scene::Transform;
use glam::{Vec2, Vec3};
use hecs::Entity;

/// Follows the main camera in XZ (snapped to `snap` metres, so vertices do not swim).
#[derive(Clone, Copy, Debug)]
pub struct Ocean {
    pub snap: f32,
}

/// Radial water mesh: rings of `segments` vertices, linear spacing `inner` up to the radius where
/// the ring spacing equals it, then geometric growth (square-ish cells) out to `radius`.
pub fn ocean_mesh(radius: f32, inner: f32, segments: u32) -> MeshData {
    let segs = segments.max(16);
    let tau = std::f32::consts::TAU;
    let r0 = segs as f32 * inner / tau; // ring spacing == inner here
    let mut radii = vec![0.0f32];
    let mut r = inner;
    while r < r0 {
        radii.push(r);
        r += inner;
    }
    let growth = 1.0 + tau / segs as f32;
    r = r.max(r0);
    while r < radius {
        radii.push(r);
        r *= growth;
    }
    radii.push(radius);
    let mut vertices = vec![Vertex { pos: [0.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0], uv: [0.5, 0.5] }];
    for &rr in &radii[1..] {
        for s in 0..segs {
            let a = s as f32 / segs as f32 * tau;
            let (x, z) = (a.cos() * rr, a.sin() * rr);
            vertices.push(Vertex { pos: [x, 0.0, z], normal: [0.0, 1.0, 0.0], uv: [0.5 + x / (2.0 * radius), 0.5 + z / (2.0 * radius)] });
        }
    }
    let mut indices = vec![];
    for s in 0..segs {
        indices.extend([0, 1 + (s + 1) % segs, 1 + s]);
    }
    for ring in 0..radii.len() as u32 - 2 {
        let a0 = 1 + ring * segs;
        let b0 = a0 + segs;
        for s in 0..segs {
            let s1 = (s + 1) % segs;
            indices.extend([a0 + s, a0 + s1, b0 + s, a0 + s1, b0 + s1, b0 + s]);
        }
    }
    MeshData { vertices, indices }
}

/// Cells (x, z) of a bounded water grid: `spacing` per axis, at most 65536 cells in total.
pub(crate) fn water_cells(size: Vec2, spacing: f32) -> (u32, u32) {
    const MAX_CELLS: f32 = 65536.0;
    let mut s = spacing.max(0.05);
    let cells = |s: f32| ((size.x / s).ceil().max(1.0), (size.y / s).ceil().max(1.0));
    let (x, z) = cells(s);
    if x * z > MAX_CELLS {
        s *= (x * z / MAX_CELLS).sqrt();
        let (mut x, mut z) = cells(s);
        while x * z > MAX_CELLS {
            s *= 1.01;
            (x, z) = cells(s);
        }
        return (x as u32, z as u32);
    }
    (x as u32, z as u32)
}

impl Ctx {
    /// Bounded water surface of `size` (x, z) metres centred on `t`, tessellated to the material's
    /// `water_vertex_spacing` along each axis (at most 65536 cells; beyond that the spacing grows evenly).
    /// Uses the material as given (see `Material::water`).
    pub fn spawn_water(&mut self, name: &str, material: MaterialId, size: Vec2, t: Transform) -> Entity {
        let spacing = self.renderer.material(material).map(|m| m.water_vertex_spacing).unwrap_or(0.25).max(0.05);
        let (nx, nz) = water_cells(size, spacing);
        let mesh = self.renderer.add_mesh(name, &MeshData::grid_rect(1.0, 1.0, nx, nz, 1.0));
        let t = t.with_scale(t.scale * Vec3::new(size.x, 1.0, size.y));
        self.spawn_mesh(name, mesh, material, t)
    }

    /// Endless sea at height `y` (local coordinates) that follows the camera: dense near the
    /// viewer (the material's `water_vertex_spacing`), coarser with distance, 5 km radius.
    pub fn spawn_ocean(&mut self, material: MaterialId, y: f32) -> Entity {
        let spacing = self.renderer.material(material).map(|m| m.water_vertex_spacing).unwrap_or(0.25).max(0.05);
        let mesh: MeshId = self.renderer.add_mesh("ocean", &ocean_mesh(5000.0, spacing, 256));
        let e = self.spawn_mesh("ocean", mesh, material, Transform::from_xyz(0.0, y, 0.0));
        let _ = self.world.insert_one(e, Ocean { snap: spacing * 4.0 });
        follow_camera(self);
        e
    }
}

/// Moves every [`Ocean`] under the main camera (called each frame before transforms propagate).
pub fn follow_camera(ctx: &mut Ctx) {
    let cam = crate::scene::main_camera(&ctx.world).and_then(|e| ctx.world.get::<&Transform>(e).ok().map(|t| t.translation));
    let Some(c) = cam else { return };
    for (o, t) in ctx.world.query_mut::<(&Ocean, &mut Transform)>() {
        let s = o.snap.max(0.01);
        t.translation.x = (c.x / s).round() * s;
        t.translation.z = (c.z / s).round() * s;
    }
}

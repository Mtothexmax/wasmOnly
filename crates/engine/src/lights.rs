//! Clustered lighting: point and spot lights are assigned to a 16 x 9 x 24 froxel grid (screen
//! tiles x exponential depth slices) on the CPU each frame; the fragment shader only loops over
//! the lights of its cluster. Hundreds of lights (street lamps, neon, headlights, sirens) cost
//! roughly what the few that touch a pixel cost.
//!
//! Lights are components on entities: [`crate::scene::PointLight`] and [`SpotLight`] (direction =
//! the entity's -Z, like the camera). Up to `MAX_LIGHTS` nearest visible lights are used.
//! Stats: `stats.render.lights` (total, visible, max per cluster, index count).

use crate::scene::{GlobalTransform, PointLight};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use hecs::World;
use serde::Serialize;

pub const CLUSTER_X: u32 = 16;
pub const CLUSTER_Y: u32 = 9;
pub const CLUSTER_Z: u32 = 24;
pub const CLUSTERS: u32 = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
pub const MAX_LIGHTS: usize = 2048;

/// Cone light. Points along the entity's -Z axis (use `Transform::looking_at`).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct SpotLight {
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    /// Full intensity inside this half-angle (degrees).
    pub inner_deg: f32,
    /// Zero outside this half-angle (degrees).
    pub outer_deg: f32,
}

impl Default for SpotLight {
    fn default() -> Self {
        Self { color: Vec3::ONE, intensity: 30.0, range: 20.0, inner_deg: 18.0, outer_deg: 28.0 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuLight {
    pub pos_range: [f32; 4],
    pub color_intensity: [f32; 4],
    /// xyz = direction the light points, w = cos(outer) (-2 = point light)
    pub dir_cos: [f32; 4],
    /// x = cos(inner)
    pub extra: [f32; 4],
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LightStats {
    pub total: u32,
    pub visible: u32,
    pub max_per_cluster: u32,
    pub indices: u32,
    pub cpu_ms: f32,
}

impl crate::app::Ctx {
    /// AI API `lights`: stats, list nearest, spawn one or a test grid.
    pub(crate) fn lights_command(&mut self, a: &serde_json::Value) -> Result<serde_json::Value, String> {
        use serde_json::json;
        let v3 = |v: &serde_json::Value, d: Vec3| v.as_array().map(|a| Vec3::new(a[0].as_f64().unwrap_or(0.0) as f32, a[1].as_f64().unwrap_or(0.0) as f32, a[2].as_f64().unwrap_or(0.0) as f32)).unwrap_or(d);
        let mut spawned = 0;
        if let Some(g) = a.get("grid") {
            // test grid: n x n lights, alternating colours, every 3rd a downward spot
            let n = g["n"].as_u64().unwrap_or(10) as i32;
            let sp = g["spacing"].as_f64().unwrap_or(4.0) as f32;
            let c = v3(&g["center"], Vec3::new(0.0, 2.5, 0.0));
            let palette = [Vec3::new(1.0, 0.3, 0.6), Vec3::new(0.2, 0.7, 1.0), Vec3::new(1.0, 0.7, 0.2), Vec3::new(0.4, 1.0, 0.4)];
            for x in 0..n {
                for z in 0..n {
                    let p = c + Vec3::new((x as f32 - (n - 1) as f32 * 0.5) * sp, 0.0, (z as f32 - (n - 1) as f32 * 0.5) * sp);
                    let col = palette[((x + z) % 4) as usize];
                    let t = crate::scene::Transform::from_translation(p).looking_at(p - Vec3::Y + Vec3::X * 0.01, Vec3::Y);
                    if (x * n + z) % 3 == 0 {
                        self.world.spawn((t, crate::scene::Name("test-spot".into()), SpotLight { color: col, intensity: 40.0, range: sp * 2.5, ..Default::default() }));
                    } else {
                        self.world.spawn((t, crate::scene::Name("test-light".into()), PointLight { color: col, intensity: 12.0, range: sp * 1.6 }));
                    }
                    spawned += 1;
                }
            }
        }
        if let Some(l) = a.get("add") {
            let p = v3(&l["pos"], Vec3::Y * 3.0);
            let col = v3(&l["color"], Vec3::ONE);
            let t = crate::scene::Transform::from_translation(p).looking_at(p + v3(&l["dir"], Vec3::NEG_Y) + Vec3::X * 1e-3, Vec3::Y);
            let intensity = l["intensity"].as_f64().unwrap_or(20.0) as f32;
            let range = l["range"].as_f64().unwrap_or(10.0) as f32;
            let e = if l["spot"].as_bool() == Some(true) {
                self.world.spawn((t, crate::scene::Name("light".into()), SpotLight { color: col, intensity, range, inner_deg: l["inner_deg"].as_f64().unwrap_or(18.0) as f32, outer_deg: l["outer_deg"].as_f64().unwrap_or(28.0) as f32 }))
            } else {
                self.world.spawn((t, crate::scene::Name("light".into()), PointLight { color: col, intensity, range }))
            };
            return Ok(json!({"id": e.id()}));
        }
        if a["clear_test"].as_bool() == Some(true) {
            let doomed: Vec<hecs::Entity> = self.world.query::<(hecs::Entity, &crate::scene::Name)>().iter().filter(|(_, n)| n.0.starts_with("test-")).map(|(e, _)| e).collect();
            for e in doomed {
                self.despawn(e);
            }
        }
        let points = self.world.query::<&PointLight>().iter().count();
        let spots = self.world.query::<&SpotLight>().iter().count();
        Ok(json!({"point": points, "spot": spots, "spawned": spawned, "frame": self.renderer.stats.lights,
                  "clusters": [CLUSTER_X, CLUSTER_Y, CLUSTER_Z], "max_lights": MAX_LIGHTS,
                  "hint": "render {debug_view: 4} shows lights per cluster, 3 shows shadow cascades"}))
    }
}

pub(crate) struct Clusters {
    pub lights: Vec<GpuLight>,
    /// Per cluster: (offset into indices, count).
    pub grid: Vec<[u32; 2]>,
    pub indices: Vec<u32>,
    pub stats: LightStats,
}

/// Gathers lights, culls them against the view, and assigns them to clusters.
/// `planes`: frustum planes (xyz normal, w d); depth slices span `near..far`.
pub(crate) fn build(world: &World, view: Mat4, proj: Mat4, planes: &[Vec4; 6], cam: Vec3, near: f32, far: f32) -> Clusters {
    let t0 = crate::log::now_ms();
    // (distance, light, radius)
    let mut cand: Vec<(f32, GpuLight, Vec3, f32)> = vec![];
    let mut total = 0u32;
    let visible = |p: Vec3, r: f32| planes.iter().all(|pl| pl.truncate().dot(p) + pl.w >= -r);
    for (pl, g) in world.query::<(&PointLight, &GlobalTransform)>().iter() {
        total += 1;
        let p = g.translation();
        if pl.intensity <= 0.0 || !visible(p, pl.range) {
            continue;
        }
        let gl = GpuLight { pos_range: [p.x, p.y, p.z, pl.range], color_intensity: [pl.color.x, pl.color.y, pl.color.z, pl.intensity], dir_cos: [0.0, -1.0, 0.0, -2.0], extra: [1.0, 0.0, 0.0, 0.0] };
        cand.push((p.distance(cam) - pl.range, gl, p, pl.range));
    }
    for (sl, g) in world.query::<(&SpotLight, &GlobalTransform)>().iter() {
        total += 1;
        let p = g.translation();
        let d = (-g.0.z_axis.truncate()).normalize_or(Vec3::NEG_Y);
        // bounding sphere of the cone (centre halfway along it, conservative radius)
        let (center, radius) = if sl.outer_deg < 45.0 {
            let half = sl.range * 0.5;
            let side = sl.range * sl.outer_deg.to_radians().tan();
            (p + d * half, (half * half + side * side).sqrt())
        } else {
            (p, sl.range)
        };
        if sl.intensity <= 0.0 || !visible(center, radius) {
            continue;
        }
        let gl = GpuLight {
            pos_range: [p.x, p.y, p.z, sl.range],
            color_intensity: [sl.color.x, sl.color.y, sl.color.z, sl.intensity],
            dir_cos: [d.x, d.y, d.z, sl.outer_deg.to_radians().cos()],
            extra: [sl.inner_deg.min(sl.outer_deg - 0.1).to_radians().cos(), 0.0, 0.0, 0.0],
        };
        cand.push((center.distance(cam) - radius, gl, center, radius));
    }
    if cand.len() > MAX_LIGHTS {
        cand.sort_by(|a, b| a.0.total_cmp(&b.0));
        cand.truncate(MAX_LIGHTS);
    }

    // froxel ranges per light
    let log_ratio = (far / near).ln();
    let slice = |z: f32| (((z.max(near) / near).ln() / log_ratio) * CLUSTER_Z as f32).floor().clamp(0.0, CLUSTER_Z as f32 - 1.0) as u32;
    let mut ranges: Vec<(u32, [u32; 6])> = Vec::with_capacity(cand.len());
    for (i, (_, _, center, r)) in cand.iter().enumerate() {
        let c = view.transform_point3(*center);
        let (zmin, zmax) = (-c.z - r, -c.z + r);
        if zmax < near || zmin > far {
            continue;
        }
        let (z0, z1) = (slice(zmin), slice(zmax));
        let (mut x0, mut x1, mut y0, mut y1) = (0u32, CLUSTER_X - 1, 0u32, CLUSTER_Y - 1);
        if zmin > near {
            // screen rect of the sphere's view-space AABB
            let (mut lo, mut hi) = (glam::Vec2::splat(f32::MAX), glam::Vec2::splat(f32::MIN));
            for k in 0..8 {
                let o = Vec3::new(if k & 1 == 0 { -r } else { *r }, if k & 2 == 0 { -r } else { *r }, if k & 4 == 0 { -r } else { *r });
                let q = proj * (c + o).extend(1.0);
                let ndc = q.truncate().truncate() / q.w.max(1e-4);
                lo = lo.min(ndc);
                hi = hi.max(ndc);
            }
            if hi.x < -1.0 || lo.x > 1.0 || hi.y < -1.0 || lo.y > 1.0 {
                continue;
            }
            let tx = |x: f32| (((x * 0.5 + 0.5) * CLUSTER_X as f32).floor()).clamp(0.0, CLUSTER_X as f32 - 1.0) as u32;
            let ty = |y: f32| (((0.5 - y * 0.5) * CLUSTER_Y as f32).floor()).clamp(0.0, CLUSTER_Y as f32 - 1.0) as u32;
            x0 = tx(lo.x);
            x1 = tx(hi.x);
            y0 = ty(hi.y);
            y1 = ty(lo.y);
        }
        ranges.push((i as u32, [x0, x1, y0, y1, z0, z1]));
    }
    // two passes: count, prefix sum, fill
    let mut grid = vec![[0u32; 2]; CLUSTERS as usize];
    let idx = |x: u32, y: u32, z: u32| (x + y * CLUSTER_X + z * CLUSTER_X * CLUSTER_Y) as usize;
    for (_, r) in &ranges {
        for z in r[4]..=r[5] {
            for y in r[2]..=r[3] {
                for x in r[0]..=r[1] {
                    grid[idx(x, y, z)][1] += 1;
                }
            }
        }
    }
    let mut off = 0u32;
    let mut maxc = 0u32;
    for g in grid.iter_mut() {
        g[0] = off;
        off += g[1];
        maxc = maxc.max(g[1]);
        g[1] = 0;
    }
    let mut indices = vec![0u32; off as usize];
    for (li, r) in &ranges {
        for z in r[4]..=r[5] {
            for y in r[2]..=r[3] {
                for x in r[0]..=r[1] {
                    let g = &mut grid[idx(x, y, z)];
                    indices[(g[0] + g[1]) as usize] = *li;
                    g[1] += 1;
                }
            }
        }
    }
    let lights: Vec<GpuLight> = cand.into_iter().map(|c| c.1).collect();
    let stats = LightStats { total, visible: lights.len() as u32, max_per_cluster: maxc, indices: off, cpu_ms: (crate::log::now_ms() - t0) as f32 };
    Clusters { lights, grid, indices, stats }
}

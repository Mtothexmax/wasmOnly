//! Z-fighting detector (AI-first: no screenshot needed). Z-fighting happens where two surfaces of
//! different objects lie in (almost) the same plane, face the same way and overlap — the depth test
//! then picks a winner per pixel by rounding noise, which flickers as the camera moves. A still
//! screenshot rarely shows it; this test finds the cause geometrically and names the entities.
//!
//! Algorithm: every opaque / alpha-tested triangle in world space -> plane (normal, distance);
//! triangles are grouped by quantized normal and swept by plane distance; pairs from different
//! entities with |Δd| < `eps` and the same facing are projected into the plane and tested for a real
//! area overlap (separating-axis test with a small tolerance, so neighbours that only touch at an
//! edge do not count). Skipped: blended / water materials and vertex-displaced meshes (their CPU
//! shape is not what the GPU draws).
//!
//! `engine.call("zfight", {eps?: 0.002, limit?: 50})`.

use crate::renderer::{AlphaMode, Renderer};
use crate::scene::{GlobalTransform, MeshRenderer, Name};
use glam::{Mat4, Vec2, Vec3};
use hecs::{Entity, World};
use serde_json::{json, Value};
use std::collections::HashMap;

struct Tri {
    owner: u32, // index into `owners`
    n: Vec3,
    d: f32,
    p: [Vec3; 3],
}

/// Overlap of two coplanar triangles projected into the plane (SAT). Returns the smallest
/// penetration depth over all axes (<= 0: no overlap).
fn overlap_depth(a: &[Vec2; 3], b: &[Vec2; 3]) -> f32 {
    let mut min_pen = f32::MAX;
    for tri in [a, b] {
        for k in 0..3 {
            let e = tri[(k + 1) % 3] - tri[k];
            let axis = Vec2::new(-e.y, e.x).normalize_or_zero();
            if axis == Vec2::ZERO {
                return 0.0;
            }
            let proj = |t: &[Vec2; 3]| {
                let v = [axis.dot(t[0]), axis.dot(t[1]), axis.dot(t[2])];
                (v[0].min(v[1]).min(v[2]), v[0].max(v[1]).max(v[2]))
            };
            let (a0, a1) = proj(a);
            let (b0, b1) = proj(b);
            let pen = a1.min(b1) - a0.max(b0);
            if pen <= 0.0 {
                return pen;
            }
            min_pen = min_pen.min(pen);
        }
    }
    min_pen
}

/// Automatic check (debug builds): re-runs a moment after the scene changed (entity count or
/// pending models) and warns in the log, so every `bridge reload` / `bridge check` surfaces new
/// z-fighting without anyone asking for it.
#[derive(Default)]
pub struct AutoCheck {
    last_entities: u32,
    last_pending: usize,
    due_ms: Option<f64>,
    last_report: String,
    pub runs: u32,
    pub last_pairs: u32,
}

impl AutoCheck {
    /// Automatic check radius around the camera (m): big streamed worlds would take seconds.
    pub const RADIUS: f32 = 80.0;

    pub(crate) fn tick(&mut self, world: &World, renderer: &Renderer, pending_models: usize, now: f64, camera: Option<Vec3>) {
        if !cfg!(debug_assertions) {
            return;
        }
        let n = world.len();
        if n != self.last_entities || pending_models != self.last_pending {
            self.last_entities = n;
            self.last_pending = pending_models;
            self.due_ms = Some(now + 1500.0);
        }
        let Some(due) = self.due_ms else { return };
        if now < due || pending_models > 0 {
            return;
        }
        self.due_ms = None;
        self.runs += 1;
        let r = find_near(world, renderer, 0.002, 0.002, 3, Mat4::IDENTITY, Vec2::ZERO, camera.map(|c| (c, Self::RADIUS)));
        let pairs = r["pairs"].as_u64().unwrap_or(0) as u32;
        self.last_pairs = pairs;
        let report = if pairs == 0 {
            String::new()
        } else {
            let worst: Vec<String> = r["worst"]
                .as_array()
                .map(|w| w.iter().map(|p| format!("{}#{} x {}#{} at {}", p["a"]["name"].as_str().unwrap_or(""), p["a"]["id"], p["b"]["name"].as_str().unwrap_or(""), p["b"]["id"], p["point"])).collect())
                .unwrap_or_default();
            format!("z-fighting: {pairs} overlapping coplanar surface pair(s), e.g. {} — engine.call('zfight') for all", worst.join("; "))
        };
        if report != self.last_report {
            if pairs > 0 {
                crate::log_warn!("{report}");
            } else {
                crate::log_info!("z-fighting resolved (0 pairs)");
            }
            self.last_report = report;
        }
    }
}

pub fn find(world: &World, renderer: &Renderer, eps: f32, tolerance: f32, limit: usize, view_proj: Mat4, css: Vec2) -> Value {
    find_near(world, renderer, eps, tolerance, limit, view_proj, css, None)
}

/// Like [`find`], limited to instances whose bounds touch the sphere `near` (centre, radius).
#[allow(clippy::too_many_arguments)]
pub fn find_near(world: &World, renderer: &Renderer, eps: f32, tolerance: f32, limit: usize, view_proj: Mat4, css: Vec2, near: Option<(Vec3, f32)>) -> Value {
    let mut owners: Vec<Entity> = vec![];
    let mut tris: Vec<Tri> = vec![];
    let mut skipped: HashMap<&'static str, u32> = HashMap::new();
    for (e, g, mr) in world.query::<(Entity, &GlobalTransform, &MeshRenderer)>().iter() {
        let Some(mat) = renderer.material(mr.material) else { continue };
        if mat.alpha_mode == AlphaMode::Blend || mat.water {
            *skipped.entry("blended/water").or_default() += 1;
            continue;
        }
        if mat.displacement != 0.0 && mat.height_texture.is_some() {
            *skipped.entry("vertex-displaced").or_default() += 1;
            continue;
        }
        let Some(mesh) = renderer.mesh(mr.mesh) else { continue };
        if let Some((c, r)) = near {
            let scale = g.0.x_axis.truncate().length().max(g.0.y_axis.truncate().length()).max(g.0.z_axis.truncate().length());
            if g.0.transform_point3(mesh.bounds_center).distance(c) > r + mesh.bounds_radius * scale {
                *skipped.entry("outside radius").or_default() += 1;
                continue;
            }
        }
        let Some(cpu) = &mesh.cpu else {
            *skipped.entry("mesh too large for CPU copy").or_default() += 1;
            continue;
        };
        let owner = owners.len() as u32;
        owners.push(e);
        let m = g.0;
        for t in cpu.1.as_chunks::<3>().0 {
            let p = [0, 1, 2].map(|k| m.transform_point3(Vec3::from(cpu.0[t[k] as usize])));
            let c = (p[1] - p[0]).cross(p[2] - p[0]);
            let area2 = c.length();
            if area2 < 1e-8 {
                continue;
            }
            let n = c / area2;
            tris.push(Tri { owner, n, d: n.dot(p[0]), p });
        }
    }

    // group by quantized normal, sweep by plane distance
    let key = |n: Vec3| ((n.x * 64.0).round() as i32, (n.y * 64.0).round() as i32, (n.z * 64.0).round() as i32);
    let mut groups: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, t) in tris.iter().enumerate() {
        groups.entry(key(t.n)).or_default().push(i);
    }
    // (owner a, owner b) -> (triangle pairs, max |Δd|, point sum)
    let mut pairs: HashMap<(u32, u32), (u32, f32, Vec3)> = HashMap::new();
    for idx in groups.values_mut() {
        idx.sort_by(|a, b| tris[*a].d.total_cmp(&tris[*b].d));
        for i in 0..idx.len() {
            let ta = &tris[idx[i]];
            for &jb in &idx[i + 1..] {
                let tb = &tris[jb];
                if tb.d - ta.d > eps {
                    break;
                }
                if ta.owner == tb.owner || ta.n.dot(tb.n) < 0.9995 {
                    continue;
                }
                // 2D basis in the plane
                let u = ta.n.any_orthonormal_vector();
                let v = ta.n.cross(u);
                let to2 = |p: &[Vec3; 3]| p.map(|q| Vec2::new(q.dot(u), q.dot(v)));
                if overlap_depth(&to2(&ta.p), &to2(&tb.p)) > tolerance {
                    let k = if ta.owner < tb.owner { (ta.owner, tb.owner) } else { (tb.owner, ta.owner) };
                    let centre = (ta.p[0] + ta.p[1] + ta.p[2] + tb.p[0] + tb.p[1] + tb.p[2]) / 6.0;
                    let entry = pairs.entry(k).or_insert((0, 0.0, Vec3::ZERO));
                    entry.0 += 1;
                    entry.1 = entry.1.max(tb.d - ta.d);
                    entry.2 += centre;
                }
            }
        }
    }

    let describe = |o: u32| {
        let e = owners[o as usize];
        json!({"id": e.id(), "name": world.get::<&Name>(e).map(|n| n.0.clone()).unwrap_or_default()})
    };
    let r3 = |v: Vec3| json!([(v.x * 1000.0).round() / 1000.0, (v.y * 1000.0).round() / 1000.0, (v.z * 1000.0).round() / 1000.0]);
    let mut list: Vec<(u32, Value)> = pairs
        .iter()
        .map(|((a, b), (count, dist, sum))| {
            let p = *sum / *count as f32;
            let clip = view_proj * p.extend(1.0);
            let screen = (clip.w > 0.01).then(|| {
                let ndc = clip.truncate() / clip.w;
                [((ndc.x * 0.5 + 0.5) * css.x).round(), ((0.5 - ndc.y * 0.5) * css.y).round()]
            });
            (*count, json!({"a": describe(*a), "b": describe(*b), "triangle_pairs": count, "max_plane_distance": dist, "point": r3(p), "screen": screen}))
        })
        .collect();
    list.sort_by_key(|x| std::cmp::Reverse(x.0));
    json!({
        "pairs": list.len(),
        "worst": list.into_iter().take(limit).map(|x| x.1).collect::<Vec<_>>(),
        "checked": {"entities": owners.len(), "triangles": tris.len()},
        "skipped": skipped,
        "eps": eps,
    })
}

/// Ray against the rendered meshes (CPU copies, bounding-sphere pre-test) — finds what is *drawn*,
/// unlike the physics raycast (objects without colliders, visual-only parts). Returns the nearest
/// hit and every entity whose surface lies within `tie` of it (z-fighting candidates).
pub fn ray_pick(world: &World, renderer: &Renderer, origin: Vec3, dir: Vec3, tie: f32) -> Option<(Entity, f32, Vec<Entity>)> {
    let dir = dir.normalize_or_zero();
    let mut hits: Vec<(f32, Entity)> = vec![];
    for (e, g, mr) in world.query::<(Entity, &GlobalTransform, &MeshRenderer)>().iter() {
        let Some(mesh) = renderer.mesh(mr.mesh) else { continue };
        let Some(cpu) = &mesh.cpu else { continue };
        let m = g.0;
        let scale = m.x_axis.truncate().length().max(m.y_axis.truncate().length()).max(m.z_axis.truncate().length());
        let c = m.transform_point3(mesh.bounds_center);
        let r = mesh.bounds_radius * scale;
        let oc = c - origin;
        let t_c = oc.dot(dir);
        if oc.length_squared() - t_c * t_c > r * r || t_c < -r {
            continue;
        }
        let mut best = f32::MAX;
        for t in cpu.1.as_chunks::<3>().0 {
            let p = [0, 1, 2].map(|k| m.transform_point3(Vec3::from(cpu.0[t[k] as usize])));
            // Möller–Trumbore, both faces
            let e1 = p[1] - p[0];
            let e2 = p[2] - p[0];
            let h = dir.cross(e2);
            let det = e1.dot(h);
            if det.abs() < 1e-9 {
                continue;
            }
            let inv = 1.0 / det;
            let s = origin - p[0];
            let u = s.dot(h) * inv;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let q = s.cross(e1);
            let v = dir.dot(q) * inv;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let t_hit = e2.dot(q) * inv;
            if t_hit > 1e-4 && t_hit < best {
                best = t_hit;
            }
        }
        if best < f32::MAX {
            hits.push((best, e));
        }
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (t0, e0) = *hits.first()?;
    let ties = hits.iter().skip(1).filter(|h| h.0 - t0 <= tie).map(|h| h.1).collect();
    Some((e0, t0, ties))
}

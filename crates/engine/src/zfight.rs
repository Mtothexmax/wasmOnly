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
//! `engine.call("zfight", {eps?: 0.002, limit?: 50, same_root?: false})`. Opt-out per entity: [`NoZCheck`].

use crate::renderer::{AlphaMode, Renderer};
use crate::scene::{GlobalTransform, MeshRenderer, Name};
use glam::{Mat4, Vec2, Vec3};
use hecs::{Entity, World};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

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

/// Leaves an entity (and its children) out of every z-fighting check, automatic and `zfight`
/// (e.g. decals or overlays that are coplanar on purpose and drawn in a fixed order).
#[derive(Clone, Copy, Debug, Default)]
pub struct NoZCheck;

/// Filters of [`find_with`].
#[derive(Default)]
pub struct Filter<'a> {
    /// Only instances whose bounds touch this sphere (centre, radius).
    pub near: Option<(Vec3, f32)>,
    /// Report only pairs with at least one of these; the others are checked only where their
    /// bounds touch one of these (cheap for a few new entities in a big world).
    pub involving: Option<&'a [Entity]>,
    pub exclude: &'a [Entity],
    /// Skip pairs whose entities share a root (top of the `Parent` chain): parts of one model.
    pub skip_same_root: bool,
}

/// Automatic check (debug builds): re-runs a moment after the scene changed (entity count or
/// pending models) and warns in the log, so every `bridge reload` / `bridge check` surfaces new
/// z-fighting without anyone asking for it. The first run checks everything within [`Self::RADIUS`]
/// of the camera; later runs only the drawn entities that are new since the last run, against
/// whatever their bounds touch (streamed chunks cost their own size, not the world's).
#[derive(Default)]
pub struct AutoCheck {
    last_entities: u32,
    last_pending: usize,
    due_ms: Option<f64>,
    last_report: String,
    checked: HashSet<Entity>,
    baseline: bool,
    pub runs: u32,
    /// Runs that only checked new entities.
    pub incremental_runs: u32,
    pub last_pairs: u32,
    /// Entities the last run was about (all near the camera, or the new ones).
    pub last_entities_checked: u32,
    /// Triangles to check at the last skip (0 = not skipped).
    pub skipped_triangles: u64,
    pub last_ms: f64,
}

impl AutoCheck {
    /// Automatic check radius around the camera (m): big streamed worlds would take seconds.
    pub const RADIUS: f32 = 80.0;
    /// Above this many triangles to check the automatic check is skipped (it is O(n log n)
    /// on the CPU and blocks the frame: ~1 s per 100k triangles in a debug build). Run
    /// `zfight {radius}` by hand instead.
    pub const MAX_TRIANGLES: u64 = 150_000;

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
        let drawn: HashSet<Entity> = world.query::<(Entity, &MeshRenderer)>().iter().map(|(e, _)| e).collect();
        // after the first evaluation (run or skipped as too big) only new entities are checked
        let incremental = self.baseline;
        self.baseline = true;
        let fresh: Vec<Entity> = if incremental { drawn.iter().filter(|e| !self.checked.contains(e)).copied().collect() } else { vec![] };
        self.checked = drawn;
        if incremental && fresh.is_empty() {
            return; // only despawns: nothing new can fight
        }
        // cheap estimate first: triangles of the instances to check (near the camera / the new ones)
        let near = camera.map(|c| (c, Self::RADIUS));
        let tri_count = |mr: &MeshRenderer| renderer.mesh(mr.mesh).map(|m| m.index_count as u64 / 3).unwrap_or(0);
        let mut tris = 0u64;
        if incremental {
            for &e in &fresh {
                let (Ok(mr), Ok(g)) = (world.get::<&MeshRenderer>(e), world.get::<&GlobalTransform>(e)) else { continue };
                if camera.is_none_or(|c| g.translation().distance_squared(c) < (Self::RADIUS * 1.2).powi(2)) {
                    tris += tri_count(&mr);
                }
            }
        } else if let Some(c) = camera {
            for (g, mr) in world.query::<(&GlobalTransform, &MeshRenderer)>().iter() {
                if g.translation().distance_squared(c) < (Self::RADIUS * 1.2).powi(2) {
                    tris += tri_count(mr);
                }
            }
        }
        if tris > Self::MAX_TRIANGLES {
            if self.skipped_triangles == 0 {
                crate::log_info!("automatic z-fighting check skipped: {tris} triangles to check (limit {}); run engine.call('zfight', {{radius: 20}}) by hand", Self::MAX_TRIANGLES);
            }
            self.skipped_triangles = tris;
            return;
        }
        self.skipped_triangles = 0;
        self.runs += 1;
        if incremental {
            self.incremental_runs += 1;
        }
        let t0 = crate::log::now_ms();
        let filter = Filter { near, involving: incremental.then_some(fresh.as_slice()), ..Default::default() };
        let r = find_with(world, renderer, 0.002, 0.002, 3, Mat4::IDENTITY, Vec2::ZERO, &filter);
        let pairs = r["pairs"].as_u64().unwrap_or(0) as u32;
        self.last_pairs = pairs;
        self.last_entities_checked = if incremental { fresh.len() as u32 } else { r["checked"]["entities"].as_u64().unwrap_or(0) as u32 };
        self.last_ms = crate::log::now_ms() - t0;
        if self.last_ms > 100.0 {
            crate::log_warn!("automatic z-fighting check blocked the frame for {:.0} ms ({} entities, {} triangles)", self.last_ms, r["checked"]["entities"], r["checked"]["triangles"]);
        }
        let report = if pairs == 0 {
            String::new()
        } else {
            let worst: Vec<String> = r["worst"]
                .as_array()
                .map(|w| w.iter().map(|p| format!("{}#{} x {}#{} at {}", p["a"]["name"].as_str().unwrap_or(""), p["a"]["id"], p["b"]["name"].as_str().unwrap_or(""), p["b"]["id"], p["point"])).collect())
                .unwrap_or_default();
            let what = if incremental { "new " } else { "" };
            format!("z-fighting: {pairs} {what}overlapping coplanar surface pair(s), e.g. {} — engine.call('zfight') for all", worst.join("; "))
        };
        if report != self.last_report {
            if pairs > 0 {
                crate::log_warn!("{report}");
                self.last_report = report;
            } else if !incremental {
                crate::log_info!("z-fighting resolved (0 pairs)");
                self.last_report = report;
            }
        }
    }

    pub fn describe(&self) -> Value {
        json!({"runs": self.runs, "incremental_runs": self.incremental_runs, "last_pairs": self.last_pairs,
            "last_entities_checked": self.last_entities_checked, "last_ms": self.last_ms, "skipped_triangles": self.skipped_triangles})
    }
}

pub fn find(world: &World, renderer: &Renderer, eps: f32, tolerance: f32, limit: usize, view_proj: Mat4, css: Vec2) -> Value {
    find_near(world, renderer, eps, tolerance, limit, view_proj, css, None)
}

/// Like [`find`], limited to instances whose bounds touch the sphere `near` (centre, radius).
#[allow(clippy::too_many_arguments)]
pub fn find_near(world: &World, renderer: &Renderer, eps: f32, tolerance: f32, limit: usize, view_proj: Mat4, css: Vec2, near: Option<(Vec3, f32)>) -> Value {
    find_filtered(world, renderer, eps, tolerance, limit, view_proj, css, near, None, &[])
}

/// Like [`find_near`]; `involving`: report only pairs with at least one of these entities;
/// `exclude`: leave these entities out of the check.
#[allow(clippy::too_many_arguments)]
pub fn find_filtered(world: &World, renderer: &Renderer, eps: f32, tolerance: f32, limit: usize, view_proj: Mat4, css: Vec2, near: Option<(Vec3, f32)>, involving: Option<&[Entity]>, exclude: &[Entity]) -> Value {
    find_with(world, renderer, eps, tolerance, limit, view_proj, css, &Filter { near, involving, exclude, skip_same_root: false })
}

fn root_of(world: &World, mut e: Entity) -> Entity {
    for _ in 0..64 {
        match world.get::<&crate::scene::Parent>(e) {
            Ok(p) => e = p.0,
            Err(_) => break,
        }
    }
    e
}

fn opted_out(world: &World, mut e: Entity) -> bool {
    for _ in 0..64 {
        if world.get::<&NoZCheck>(e).is_ok() {
            return true;
        }
        match world.get::<&crate::scene::Parent>(e) {
            Ok(p) => e = p.0,
            Err(_) => return false,
        }
    }
    false
}

/// The z-fighting test with every filter (see [`Filter`]).
#[allow(clippy::too_many_arguments)]
pub fn find_with(world: &World, renderer: &Renderer, eps: f32, tolerance: f32, limit: usize, view_proj: Mat4, css: Vec2, f: &Filter) -> Value {
    let mut owners: Vec<Entity> = vec![];
    let mut owner_layer: Vec<u32> = vec![];
    let mut owner_root: Vec<Entity> = vec![];
    let mut tris: Vec<Tri> = vec![];
    let mut skipped: HashMap<&'static str, u32> = HashMap::new();
    // layer masks of every camera (canvas and render textures): surfaces fight only where one
    // camera draws both
    let masks: Vec<u32> = world.query::<&crate::scene::Camera>().iter().map(|c| c.layers).collect();
    let drawn = masks.iter().fold(0u32, |m, l| m | l);
    let involving: Option<HashSet<Entity>> = f.involving.map(|v| v.iter().copied().collect());
    // candidates: (entity, model matrix, mesh, layer bit, world bounding sphere)
    let mut cands = vec![];
    for (e, g, mr) in world.query::<(Entity, &GlobalTransform, &MeshRenderer)>().iter() {
        let Some(mat) = renderer.material(mr.material) else { continue };
        if f.exclude.contains(&e) {
            *skipped.entry("excluded").or_default() += 1;
            continue;
        }
        let bit = 1u32 << (mr.layer & 31);
        if drawn & bit == 0 {
            *skipped.entry("layer not drawn").or_default() += 1;
            continue;
        }
        if mat.alpha_mode == AlphaMode::Blend || mat.water {
            *skipped.entry("blended/water").or_default() += 1;
            continue;
        }
        if mat.displacement != 0.0 && mat.height_texture.is_some() {
            *skipped.entry("vertex-displaced").or_default() += 1;
            continue;
        }
        let Some(mesh) = renderer.mesh(mr.mesh) else { continue };
        let scale = g.0.x_axis.truncate().length().max(g.0.y_axis.truncate().length()).max(g.0.z_axis.truncate().length());
        let (c, r) = (g.0.transform_point3(mesh.bounds_center), mesh.bounds_radius * scale);
        if let Some((nc, nr)) = f.near {
            if c.distance(nc) > nr + r {
                *skipped.entry("outside radius").or_default() += 1;
                continue;
            }
        }
        if opted_out(world, e) {
            *skipped.entry("NoZCheck").or_default() += 1;
            continue;
        }
        cands.push((e, g.0, mr.mesh, bit, c, r));
    }
    // with `involving`: only entities whose bounds touch an involving one can be in a reported pair
    if let Some(inv) = &involving {
        let spheres: Vec<(Vec3, f32)> = cands.iter().filter(|c| inv.contains(&c.0)).map(|c| (c.4, c.5 + eps + tolerance)).collect();
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for (c, r) in &spheres {
            lo = lo.min(*c - *r);
            hi = hi.max(*c + *r);
        }
        let before = cands.len();
        cands.retain(|c| {
            if inv.contains(&c.0) {
                return true;
            }
            let (p, r) = (c.4, c.5);
            if (p + r).cmplt(lo).any() || (p - r).cmpgt(hi).any() {
                return false;
            }
            spheres.iter().any(|(q, qr)| p.distance_squared(*q) <= (r + qr) * (r + qr))
        });
        let n = (before - cands.len()) as u32;
        if n > 0 {
            *skipped.entry("not near an involved entity").or_default() += n;
        }
    }
    for (e, m, mesh_id, bit, _, _) in cands {
        let Some(mesh) = renderer.mesh(mesh_id) else { continue };
        let Some(cpu) = &mesh.cpu else {
            *skipped.entry("mesh too large for CPU copy").or_default() += 1;
            continue;
        };
        let owner = owners.len() as u32;
        owners.push(e);
        owner_layer.push(bit);
        owner_root.push(if f.skip_same_root { root_of(world, e) } else { e });
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
    let mut same_root_pairs = 0u32;

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
                if f.skip_same_root && owner_root[ta.owner as usize] == owner_root[tb.owner as usize] {
                    same_root_pairs += 1;
                    continue;
                }
                let both = owner_layer[ta.owner as usize] | owner_layer[tb.owner as usize];
                if !masks.iter().any(|m| m & both == both) {
                    continue;
                }
                // 2D basis in the plane
                let u = ta.n.any_orthonormal_vector();
                let v = ta.n.cross(u);
                let to2 = |p: &[Vec3; 3]| p.map(|q| Vec2::new(q.dot(u), q.dot(v)));
                if overlap_depth(&to2(&ta.p), &to2(&tb.p)) > tolerance {
                    if let Some(inv) = &involving {
                        if !inv.contains(&owners[ta.owner as usize]) && !inv.contains(&owners[tb.owner as usize]) {
                            continue;
                        }
                    }
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
        "same_root_triangle_pairs_skipped": same_root_pairs,
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

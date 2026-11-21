//! Geometry kit for procedural content: the building blocks generators are written with
//! (cities, props, fences, roads, …). The engine ships no specific generators — games register
//! their own (see [`Generators`]) and the engine caches, instances and exposes them to agents.
//!
//! [`MeshBuilder`] collects triangles into named **parts** (one part per material, so a whole
//! building is one draw per material), with a transform stack and world-projected UVs:
//!
//! ```ignore
//! let mut b = MeshBuilder::new().uv_world(1.0);          // 1 texture tile per metre
//! b.part("wall").wall(6.0, 3.0, 0.25, &[Opening::door(1.0, 1.0, 2.1), Opening::window(3.5, 1.0, 1.2, 1.2)]);
//! b.push(Mat4::from_translation(Vec3::Y * 3.0));
//! b.part("roof").extrude(&footprint, 0.0, 0.3, true);
//! b.pop();
//! b.part("rail").sweep(&path, &circle(0.04, 6), false);
//! let parts = b.finish();                                // Vec<(part name, MeshData)>
//! let e = ctx.spawn_parts("house", &parts, |part| materials[part], Transform::from_xyz(..));
//! ```
//!
//! Heavy generation runs in parallel with `engine::jobs::par_map` (pure CPU, returns MeshData).
//! Static scenery can additionally be merged per material and cell (`ctx.batch_static`).

use crate::mesh::{MeshData, Vertex};
use glam::{Mat3, Mat4, Vec2, Vec3};
use serde_json::Value;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UvMode {
    /// UVs as the primitive defines them (0..1 per face / along the shape).
    Local,
    /// Box projection in builder space: `scale` texture tiles per metre (continuous across parts).
    World(f32),
}

/// Rectangular opening in a wall: `x` = left edge along the wall, `y` = bottom, `w` x `h`.
#[derive(Clone, Copy, Debug)]
pub struct Opening {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Opening {
    pub fn door(x: f32, w: f32, h: f32) -> Self {
        Self { x, y: 0.0, w, h }
    }
    pub fn window(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }
}

pub struct MeshBuilder {
    parts: BTreeMap<String, MeshData>,
    current: String,
    stack: Vec<Mat4>,
    uv: UvMode,
    color: [u8; 4],
}

impl Default for MeshBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Circle profile (for `sweep` / `lathe`), counter-clockwise.
pub fn circle(r: f32, segments: u32) -> Vec<Vec2> {
    (0..segments).map(|i| {
        let a = i as f32 / segments as f32 * std::f32::consts::TAU;
        Vec2::new(a.cos() * r, a.sin() * r)
    }).collect()
}

/// Rectangle outline centred at the origin (XZ footprint), counter-clockwise seen from above.
pub fn rect(w: f32, d: f32) -> Vec<Vec2> {
    let (x, z) = (w * 0.5, d * 0.5);
    vec![Vec2::new(-x, z), Vec2::new(x, z), Vec2::new(x, -z), Vec2::new(-x, -z)]
}

/// Ear-clipping triangulation of a simple polygon (any winding). Returns index triples.
pub fn triangulate(poly: &[Vec2]) -> Vec<[usize; 3]> {
    let n = poly.len();
    if n < 3 {
        return vec![];
    }
    let area: f32 = (0..n).map(|i| poly[i].perp_dot(poly[(i + 1) % n])).sum::<f32>() * 0.5;
    let ccw = area > 0.0;
    let mut idx: Vec<usize> = (0..n).collect();
    let mut out = vec![];
    let mut guard = 0;
    while idx.len() > 3 && guard < n * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for k in 0..m {
            let (ia, ib, ic) = (idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]);
            let (a, b, c) = (poly[ia], poly[ib], poly[ic]);
            let cross = (b - a).perp_dot(c - b);
            if (cross > 0.0) != ccw || cross.abs() < 1e-9 {
                continue; // reflex or degenerate
            }
            let inside = idx.iter().any(|&j| {
                if j == ia || j == ib || j == ic {
                    return false;
                }
                let p = poly[j];
                let s = |u: Vec2, v: Vec2| (v - u).perp_dot(p - u);
                let (d1, d2, d3) = (s(a, b), s(b, c), s(c, a));
                if ccw { d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0 } else { d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0 }
            });
            if inside {
                continue;
            }
            out.push(if ccw { [ia, ib, ic] } else { [ia, ic, ib] });
            idx.remove(k);
            clipped = true;
            break;
        }
        if !clipped {
            break; // self-intersecting: give up on the rest
        }
    }
    if idx.len() == 3 {
        out.push(if ccw { [idx[0], idx[1], idx[2]] } else { [idx[0], idx[2], idx[1]] });
    }
    out
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self { parts: BTreeMap::new(), current: "default".into(), stack: vec![Mat4::IDENTITY], uv: UvMode::Local, color: Vertex::WHITE }
    }

    /// Vertex colour (sRGB `0xRRGGBB`) for everything added from now on (multiplied with the
    /// material's base colour; `add` multiplies it with the added mesh's own colours).
    /// `0xffffff` = none. One material, many colours: per-brick shading without a palette texture.
    pub fn color(&mut self, rgb: u32) -> &mut Self {
        self.color = Vertex::rgb(rgb);
        self
    }

    /// Like [`MeshBuilder::color`] with alpha (0..255, multiplied with the material alpha).
    pub fn color_rgba(&mut self, rgba: [u8; 4]) -> &mut Self {
        self.color = rgba;
        self
    }

    /// World-projected UVs: `tiles_per_m` texture repeats per metre.
    pub fn uv_world(mut self, tiles_per_m: f32) -> Self {
        self.uv = UvMode::World(tiles_per_m);
        self
    }

    pub fn set_uv(&mut self, uv: UvMode) -> &mut Self {
        self.uv = uv;
        self
    }

    /// Select the part (material bucket) the next primitives go to.
    pub fn part(&mut self, name: &str) -> &mut Self {
        self.current = name.to_string();
        self
    }

    fn m(&self) -> Mat4 {
        *self.stack.last().unwrap()
    }

    /// Push a transform (relative to the current one).
    pub fn push(&mut self, t: Mat4) -> &mut Self {
        let m = self.m() * t;
        self.stack.push(m);
        self
    }

    pub fn pop(&mut self) -> &mut Self {
        if self.stack.len() > 1 {
            self.stack.pop();
        }
        self
    }

    /// Run `f` with an extra transform.
    pub fn with(&mut self, t: Mat4, f: impl FnOnce(&mut Self)) -> &mut Self {
        self.push(t);
        f(self);
        self.pop()
    }

    fn uv_for(&self, p: Vec3, n: Vec3, local: Vec2) -> [f32; 2] {
        match self.uv {
            UvMode::Local => local.into(),
            UvMode::World(s) => {
                let a = n.abs();
                let uv = if a.y >= a.x && a.y >= a.z { Vec2::new(p.x, -p.z) } else if a.x >= a.z { Vec2::new(-p.z * n.x.signum(), -p.y) } else { Vec2::new(p.x * n.z.signum(), -p.y) };
                (uv * s).into()
            }
        }
    }

    /// Adds one triangle (builder space, counter-clockwise = front) with per-vertex local UVs.
    pub fn tri_uv(&mut self, p: [Vec3; 3], uv: [Vec2; 3]) -> &mut Self {
        let m = self.m();
        let w = p.map(|q| m.transform_point3(q));
        let n = (w[1] - w[0]).cross(w[2] - w[0]).normalize_or(Vec3::Y);
        let verts: Vec<Vertex> = (0..3).map(|i| Vertex { pos: w[i].into(), normal: n.into(), uv: self.uv_for(w[i], n, uv[i]), color: self.color }).collect();
        let part = self.parts.entry(self.current.clone()).or_default();
        let base = part.vertices.len() as u32;
        part.vertices.extend(verts);
        part.indices.extend_from_slice(&[base, base + 1, base + 2]);
        self
    }

    /// Quad a-b-c-d (counter-clockwise), UVs 0..1 or world-projected.
    pub fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> &mut Self {
        self.quad_uv(a, b, c, d, [Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)])
    }

    pub fn quad_uv(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, uv: [Vec2; 4]) -> &mut Self {
        let m = self.m();
        let w = [a, b, c, d].map(|q| m.transform_point3(q));
        let n = (w[1] - w[0]).cross(w[2] - w[0]).normalize_or((w[2] - w[0]).cross(w[3] - w[0]).normalize_or(Vec3::Y));
        let verts: Vec<Vertex> = (0..4).map(|i| Vertex { pos: w[i].into(), normal: n.into(), uv: self.uv_for(w[i], n, uv[i]), color: self.color }).collect();
        let part = self.parts.entry(self.current.clone()).or_default();
        let base = part.vertices.len() as u32;
        part.vertices.extend(verts);
        part.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        self
    }

    /// Adds existing mesh data (transformed; normals with the inverse-transpose).
    pub fn add(&mut self, md: &MeshData) -> &mut Self {
        let m = self.m();
        let nm = Mat3::from_mat4(m).inverse().transpose();
        let uvm = self.uv;
        let col = self.color;
        let part = self.parts.entry(self.current.clone()).or_default();
        let base = part.vertices.len() as u32;
        for v in &md.vertices {
            let p = m.transform_point3(Vec3::from(v.pos));
            let n = (nm * Vec3::from(v.normal)).normalize_or(Vec3::Y);
            let uv = match uvm {
                UvMode::Local => v.uv,
                UvMode::World(_) => {
                    let tmp = MeshBuilder { parts: BTreeMap::new(), current: String::new(), stack: vec![], uv: uvm, color: Vertex::WHITE };
                    tmp.uv_for(p, n, Vec2::from(v.uv))
                }
            };
            part.vertices.push(Vertex { pos: p.into(), normal: n.into(), uv, color: Vertex::mul_color(v.color, col) });
        }
        part.indices.extend(md.indices.iter().map(|i| i + base));
        self
    }

    /// Axis-aligned box from `min` to `max` (builder space), all 6 faces.
    pub fn cuboid(&mut self, min: Vec3, max: Vec3) -> &mut Self {
        let (a, b) = (min, max);
        let p = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        self.quad(p(a.x, a.y, b.z), p(b.x, a.y, b.z), p(b.x, b.y, b.z), p(a.x, b.y, b.z)); // +z
        self.quad(p(b.x, a.y, a.z), p(a.x, a.y, a.z), p(a.x, b.y, a.z), p(b.x, b.y, a.z)); // -z
        self.quad(p(b.x, a.y, b.z), p(b.x, a.y, a.z), p(b.x, b.y, a.z), p(b.x, b.y, b.z)); // +x
        self.quad(p(a.x, a.y, a.z), p(a.x, a.y, b.z), p(a.x, b.y, b.z), p(a.x, b.y, a.z)); // -x
        self.quad(p(a.x, b.y, b.z), p(b.x, b.y, b.z), p(b.x, b.y, a.z), p(a.x, b.y, a.z)); // +y
        self.quad(p(a.x, a.y, a.z), p(b.x, a.y, a.z), p(b.x, a.y, b.z), p(a.x, a.y, b.z)); // -y
        self
    }

    /// Box with its base centred at `base` (y up), size `size`.
    pub fn block(&mut self, base: Vec3, size: Vec3) -> &mut Self {
        let h = Vec3::new(size.x * 0.5, 0.0, size.z * 0.5);
        self.cuboid(base - h, base + h + Vec3::Y * size.y)
    }

    /// Flat polygon in the XZ plane at height `y` (facing up if `up`), any simple outline.
    pub fn polygon(&mut self, outline: &[Vec2], y: f32, up: bool) -> &mut Self {
        for t in triangulate(outline) {
            let v = t.map(|i| Vec3::new(outline[i].x, y, outline[i].y));
            let uv = t.map(|i| outline[i]);
            // outline (x, z) seen from above with +z towards the viewer = clockwise in XZ
            if up {
                self.tri_uv([v[0], v[2], v[1]], [uv[0], uv[2], uv[1]]);
            } else {
                self.tri_uv(v, uv);
            }
        }
        self
    }

    /// Prism: XZ outline extruded from `y0` to `y1` (sides + optional caps). Footprints of
    /// buildings, curbs, platforms, pillars.
    pub fn extrude(&mut self, outline: &[Vec2], y0: f32, y1: f32, caps: bool) -> &mut Self {
        let n = outline.len();
        let area: f32 = (0..n).map(|i| outline[i].perp_dot(outline[(i + 1) % n])).sum();
        let mut perim = 0.0;
        for i in 0..n {
            let (a, b) = if area > 0.0 { (outline[i], outline[(i + 1) % n]) } else { (outline[(i + 1) % n], outline[i]) };
            let len = a.distance(b);
            let (pa, pb) = (Vec3::new(a.x, y0, a.y), Vec3::new(b.x, y0, b.y));
            let up = Vec3::Y * (y1 - y0);
            self.quad_uv(pb, pa, pa + up, pb + up, [Vec2::new(perim, 1.0), Vec2::new(perim + len, 1.0), Vec2::new(perim + len, 0.0), Vec2::new(perim, 0.0)]);
            perim += len;
        }
        if caps {
            self.polygon(outline, y1, true);
            self.polygon(outline, y0, false);
        }
        self
    }

    /// Wall panel along +X from x=0 to `len`, from y=0 to `height`, centred on z=0 with
    /// `thickness`, with rectangular openings (doors, windows) including their reveals.
    pub fn wall(&mut self, len: f32, height: f32, thickness: f32, openings: &[Opening]) -> &mut Self {
        let t = thickness * 0.5;
        // split the face into a grid along the opening edges and skip cells inside an opening
        let mut xs = vec![0.0, len];
        let mut ys = vec![0.0, height];
        for o in openings {
            xs.extend([o.x.clamp(0.0, len), (o.x + o.w).clamp(0.0, len)]);
            ys.extend([o.y.clamp(0.0, height), (o.y + o.h).clamp(0.0, height)]);
        }
        let srt = |v: &mut Vec<f32>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v.dedup_by(|a, b| (*a - *b).abs() < 1e-5);
        };
        srt(&mut xs);
        srt(&mut ys);
        let hole = |cx: f32, cy: f32| openings.iter().any(|o| cx > o.x && cx < o.x + o.w && cy > o.y && cy < o.y + o.h);
        for i in 0..xs.len() - 1 {
            for j in 0..ys.len() - 1 {
                let (x0, x1, y0, y1) = (xs[i], xs[i + 1], ys[j], ys[j + 1]);
                if hole((x0 + x1) * 0.5, (y0 + y1) * 0.5) {
                    continue;
                }
                let uv = |x: f32, y: f32| Vec2::new(x, height - y);
                // front (+z) and back (-z)
                self.quad_uv(Vec3::new(x0, y0, t), Vec3::new(x1, y0, t), Vec3::new(x1, y1, t), Vec3::new(x0, y1, t), [uv(x0, y0), uv(x1, y0), uv(x1, y1), uv(x0, y1)]);
                self.quad_uv(Vec3::new(x1, y0, -t), Vec3::new(x0, y0, -t), Vec3::new(x0, y1, -t), Vec3::new(x1, y1, -t), [uv(x1, y0), uv(x0, y0), uv(x0, y1), uv(x1, y1)]);
            }
        }
        // reveals of each opening (inner faces), end caps, top
        for o in openings {
            let (x0, x1, y0, y1) = (o.x, o.x + o.w, o.y, o.y + o.h);
            self.quad(Vec3::new(x0, y0, t), Vec3::new(x0, y0, -t), Vec3::new(x0, y1, -t), Vec3::new(x0, y1, t)); // left reveal faces +x
            self.quad(Vec3::new(x1, y0, -t), Vec3::new(x1, y0, t), Vec3::new(x1, y1, t), Vec3::new(x1, y1, -t)); // right reveal faces -x
            self.quad(Vec3::new(x0, y1, -t), Vec3::new(x1, y1, -t), Vec3::new(x1, y1, t), Vec3::new(x0, y1, t)); // lintel underside faces -y
            if y0 > 0.0 {
                self.quad(Vec3::new(x0, y0, t), Vec3::new(x1, y0, t), Vec3::new(x1, y0, -t), Vec3::new(x0, y0, -t)); // sill faces +y
            }
        }
        self.quad(Vec3::new(0.0, 0.0, -t), Vec3::new(0.0, 0.0, t), Vec3::new(0.0, height, t), Vec3::new(0.0, height, -t));
        self.quad(Vec3::new(len, 0.0, t), Vec3::new(len, 0.0, -t), Vec3::new(len, height, -t), Vec3::new(len, height, t));
        self.quad(Vec3::new(0.0, height, t), Vec3::new(len, height, t), Vec3::new(len, height, -t), Vec3::new(0.0, height, -t));
        self
    }

    /// Sweeps `profile` (2D, in the plane perpendicular to the path) along `path`: pipes, rails,
    /// cables, curbs, road strips. `closed_profile` connects the last profile point to the first.
    pub fn sweep(&mut self, path: &[Vec3], profile: &[Vec2], closed_profile: bool) -> &mut Self {
        if path.len() < 2 || profile.len() < 2 {
            return self;
        }
        let mut frames: Vec<(Vec3, Vec3, Vec3)> = vec![]; // (pos, side, up)
        let mut prev_up = Vec3::Y;
        for i in 0..path.len() {
            let t = if i == 0 { path[1] - path[0] } else if i == path.len() - 1 { path[i] - path[i - 1] } else { (path[i + 1] - path[i - 1]) * 0.5 }.normalize_or(Vec3::X);
            let mut side = prev_up.cross(t);
            if side.length_squared() < 1e-6 {
                side = Vec3::X.cross(t);
            }
            let side = side.normalize();
            let up = t.cross(side).normalize();
            prev_up = up;
            frames.push((path[i], side, up));
        }
        let ring = |f: &(Vec3, Vec3, Vec3), p: Vec2| f.0 + f.1 * p.x + f.2 * p.y;
        let mut along = 0.0;
        let segs = if closed_profile { profile.len() } else { profile.len() - 1 };
        for i in 0..path.len() - 1 {
            let len = path[i].distance(path[i + 1]);
            for k in 0..segs {
                let (p0, p1) = (profile[k], profile[(k + 1) % profile.len()]);
                let (a, b) = (ring(&frames[i], p0), ring(&frames[i], p1));
                let (c, d) = (ring(&frames[i + 1], p1), ring(&frames[i + 1], p0));
                let (u0, u1) = (k as f32 / segs as f32, (k + 1) as f32 / segs as f32);
                self.quad_uv(a, b, c, d, [Vec2::new(u0, along), Vec2::new(u1, along), Vec2::new(u1, along + len), Vec2::new(u0, along + len)]);
            }
            along += len;
        }
        self
    }

    /// Surface of revolution around +Y: `profile` points are (radius, height). Posts, lamps,
    /// domes, wheels, bottles.
    pub fn lathe(&mut self, profile: &[Vec2], segments: u32) -> &mut Self {
        let segments = segments.max(3);
        for i in 0..profile.len().saturating_sub(1) {
            let (p0, p1) = (profile[i], profile[i + 1]);
            for s in 0..segments {
                let (a0, a1) = (s as f32 / segments as f32 * std::f32::consts::TAU, (s + 1) as f32 / segments as f32 * std::f32::consts::TAU);
                let pt = |p: Vec2, a: f32| Vec3::new(a.cos() * p.x, p.y, -a.sin() * p.x);
                let (u0, u1) = (s as f32 / segments as f32, (s + 1) as f32 / segments as f32);
                let (v0, v1) = (i as f32 / (profile.len() - 1) as f32, (i + 1) as f32 / (profile.len() - 1) as f32);
                self.quad_uv(pt(p0, a0), pt(p0, a1), pt(p1, a1), pt(p1, a0), [Vec2::new(u0, v0), Vec2::new(u1, v0), Vec2::new(u1, v1), Vec2::new(u0, v1)]);
            }
        }
        self
    }

    /// Cylinder (y from 0 to `h`) with caps.
    pub fn cylinder(&mut self, r: f32, h: f32, segments: u32) -> &mut Self {
        self.lathe(&[Vec2::new(0.0, 0.0), Vec2::new(r, 0.0), Vec2::new(r, h), Vec2::new(0.0, h)], segments)
    }

    pub fn triangle_count(&self) -> usize {
        self.parts.values().map(|m| m.indices.len() / 3).sum()
    }

    /// Finished parts: (name, mesh), sorted by name. Flat normals; call
    /// [`smooth_normals`] on a part for smooth shading.
    pub fn finish(self) -> Vec<(String, MeshData)> {
        self.parts.into_iter().filter(|(_, m)| !m.indices.is_empty()).collect()
    }
}

/// Welds vertices at the same position (within `eps`) whose normals differ by less than
/// `max_angle_deg` and averages their normals.
pub fn smooth_normals(md: &mut MeshData, max_angle_deg: f32) {
    let cos = max_angle_deg.to_radians().cos();
    let key = |p: [f32; 3]| [(p[0] * 1000.0).round() as i64, (p[1] * 1000.0).round() as i64, (p[2] * 1000.0).round() as i64];
    let mut groups: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (i, v) in md.vertices.iter().enumerate() {
        groups.entry(key(v.pos)).or_default().push(i);
    }
    let orig: Vec<Vec3> = md.vertices.iter().map(|v| Vec3::from(v.normal)).collect();
    for idx in groups.values() {
        for &i in idx {
            let n: Vec3 = idx.iter().filter(|&&j| orig[i].dot(orig[j]) >= cos).map(|&j| orig[j]).sum();
            md.vertices[i].normal = n.normalize_or(orig[i]).into();
        }
    }
}

/// Merges meshes (already in a common space) into one.
pub fn merge(meshes: &[&MeshData]) -> MeshData {
    let mut out = MeshData::default();
    for m in meshes {
        let base = out.vertices.len() as u32;
        out.vertices.extend_from_slice(&m.vertices);
        out.indices.extend(m.indices.iter().map(|i| i + base));
    }
    out
}

/// Transforms mesh data in place (normals with the inverse-transpose).
pub fn transform(md: &mut MeshData, m: Mat4) {
    let nm = Mat3::from_mat4(m).inverse().transpose();
    for v in &mut md.vertices {
        v.pos = m.transform_point3(Vec3::from(v.pos)).into();
        v.normal = (nm * Vec3::from(v.normal)).normalize_or(Vec3::Y).into();
    }
}

// ---------------------------------------------------------------------------------------------
// Generator registry

/// A generator: params (JSON) + seed -> parts. Pure CPU, so it can run in the job system.
pub type GeneratorFn = Arc<dyn Fn(&Value, u64) -> Vec<(String, MeshData)> + Send + Sync>;

/// Registered generators + a cache of generated GPU meshes per (generator, params, seed), so the
/// same building/prop spawned many times is one mesh (instanced), not many.
#[derive(Default)]
pub struct Generators {
    pub(crate) list: BTreeMap<String, (GeneratorFn, String)>,
    pub(crate) cache: HashMap<String, Vec<(String, crate::renderer::MeshId, usize)>>,
    /// part name -> material, used by `spawn_generated` (unknown parts get the default material).
    pub materials: HashMap<String, crate::renderer::MaterialId>,
}

impl Generators {
    /// Registers a generator under `name` (`doc` describes its params for agents).
    pub fn register(&mut self, name: &str, doc: &str, f: impl Fn(&Value, u64) -> Vec<(String, MeshData)> + Send + Sync + 'static) {
        self.list.insert(name.to_string(), (Arc::new(f), doc.to_string()));
        self.cache.retain(|k, _| !k.starts_with(&format!("{name}|")));
    }

    pub fn names(&self) -> Vec<&str> {
        self.list.keys().map(|s| s.as_str()).collect()
    }
}

impl crate::app::Ctx {
    /// Spawns `parts` as one entity (one child per part / material) at `t`.
    pub fn spawn_parts(&mut self, name: &str, parts: &[(String, MeshData)], material: impl Fn(&str) -> crate::renderer::MaterialId, t: crate::scene::Transform) -> hecs::Entity {
        let root = self.world.spawn((t, crate::scene::Name(name.to_string())));
        for (pname, md) in parts {
            let mesh = self.renderer.add_mesh(&format!("{name}:{pname}"), md);
            self.world.spawn((crate::scene::Transform::default(), crate::scene::Parent(root), crate::scene::MeshRenderer::new(mesh, material(pname)), crate::scene::Name(pname.clone())));
        }
        root
    }

    /// The cached GPU meshes of a generator run: (part, mesh). Use them in scatter rules,
    /// streaming cells or your own spawns; materials via `generators.materials[part]`.
    pub fn generated_meshes(&mut self, name: &str, params: &Value, seed: u64) -> Result<Vec<(String, crate::renderer::MeshId)>, String> {
        let key = format!("{name}|{params}|{seed}");
        if !self.generators.cache.contains_key(&key) {
            let e = self.spawn_generated(name, params, seed, crate::scene::Transform::default())?;
            self.despawn(e);
        }
        Ok(self.generators.cache[&key].iter().map(|(p, m, _)| (p.clone(), *m)).collect())
    }

    /// Material registered for a generator part (default material otherwise).
    pub fn part_material(&self, part: &str) -> crate::renderer::MaterialId {
        self.generators.materials.get(part).copied().unwrap_or(self.renderer.builtin.default_material)
    }

    /// Runs generator `name` (cached per params + seed: repeated calls share GPU meshes and are
    /// drawn instanced) and spawns the result at `t`. Materials come from `generators.materials`.
    pub fn spawn_generated(&mut self, name: &str, params: &Value, seed: u64, t: crate::scene::Transform) -> Result<hecs::Entity, String> {
        let key = format!("{name}|{params}|{seed}");
        if !self.generators.cache.contains_key(&key) {
            let (f, _) = self.generators.list.get(name).ok_or_else(|| format!("no generator '{name}' (have {:?})", self.generators.names()))?.clone();
            let t0 = crate::log::now_ms();
            let parts = f(params, seed);
            let meshes: Vec<_> = parts
                .iter()
                .map(|(p, md)| (p.clone(), self.renderer.add_mesh(&format!("gen:{name}:{p}"), md), md.indices.len() / 3))
                .collect();
            crate::log_info!("generator {name}: {} parts, {} triangles, {:.1} ms", meshes.len(), meshes.iter().map(|m| m.2).sum::<usize>(), crate::log::now_ms() - t0);
            self.generators.cache.insert(key.clone(), meshes);
        }
        let parts = self.generators.cache[&key].clone();
        let root = self.world.spawn((t, crate::scene::Name(name.to_string())));
        let default = self.renderer.builtin.default_material;
        for (p, mesh, _) in parts {
            let mat = self.generators.materials.get(&p).copied().unwrap_or(default);
            self.world.spawn((crate::scene::Transform::default(), crate::scene::Parent(root), crate::scene::MeshRenderer::new(mesh, mat), crate::scene::Name(p)));
        }
        Ok(root)
    }

    /// Generates many variants in parallel (job system) and returns their parts (not spawned).
    pub fn generate_many(&self, name: &str, jobs: &[(Value, u64)]) -> Result<Vec<Vec<(String, MeshData)>>, String> {
        let (f, _) = self.generators.list.get(name).ok_or_else(|| format!("no generator '{name}'"))?.clone();
        Ok(crate::jobs::par_map(jobs, |(p, s)| f(p, *s)))
    }

    /// AI API `generate`.
    pub(crate) fn generate_command(&mut self, a: &Value) -> Result<Value, String> {
        use serde_json::json;
        let Some(name) = a["name"].as_str() else {
            let list: Vec<Value> = self.generators.list.iter().map(|(n, (_, d))| json!({"name": n, "params": d})).collect();
            return Ok(json!({"generators": list, "cached": self.generators.cache.len(), "materials": self.generators.materials.keys().collect::<Vec<_>>()}));
        };
        let p = a.get("params").cloned().unwrap_or(json!({}));
        let seed = a["seed"].as_u64().unwrap_or(1);
        let pos = a["pos"].as_array().map(|v| Vec3::new(v[0].as_f64().unwrap_or(0.0) as f32, v[1].as_f64().unwrap_or(0.0) as f32, v[2].as_f64().unwrap_or(0.0) as f32)).unwrap_or(Vec3::ZERO);
        let yaw = a["yaw_deg"].as_f64().unwrap_or(0.0) as f32;
        if a["regenerate"].as_bool() == Some(true) {
            self.generators.cache.retain(|k, _| !k.starts_with(&format!("{name}|")));
        }
        let t = crate::scene::Transform::from_translation(pos).with_rotation(glam::Quat::from_rotation_y(yaw.to_radians()));
        let e = self.spawn_generated(name, &p, seed, t)?;
        let key = format!("{name}|{p}|{seed}");
        let tris: usize = self.generators.cache[&key].iter().map(|m| m.2).sum();
        Ok(json!({"id": e.id(), "triangles": tris, "parts": self.generators.cache[&key].iter().map(|m| m.0.clone()).collect::<Vec<_>>()}))
    }
}

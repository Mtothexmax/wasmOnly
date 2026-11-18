//! Heightmap terrain for open worlds: chunked meshes with distance LOD (and skirts against
//! cracks), splat material (4 layers blended by a splat map), per-chunk heightfield colliders,
//! chunk streaming around the camera, editing (flatten for roads / building plots, raise, paint),
//! height/normal/splat queries, and scatter rules for foliage and props.
//!
//! ```ignore
//! let mut t = Terrain::noise(TerrainDesc { size: 1024.0, res: 513, ..Default::default() }, NoiseParams { seed: 7, amplitude: 60.0, ..Default::default() });
//! t.flatten(Vec2::new(0., 0.), 80.0, 4.0, 30.0);               // a town plot
//! t.auto_splat(&SplatRules::default());                        // grass / rock by slope, sand low, snow high
//! t.scatter.push(ScatterRule { mesh: tree, material: bark, density: 0.8, layer: 0, ..Default::default() });
//! ctx.set_terrain(t, [TexPreset::Grass, TexPreset::Stone, TexPreset::Sand, TexPreset::Dirt]);
//! let h = ctx.terrain_height(x, z);                             // absolute world coordinates
//! ```
//!
//! Chunks within `stream_radius` of the camera exist as entities (mesh + LOD + collider +
//! scattered instances); others are unloaded. Edits rebuild the touched chunks next frame.
//! AI API: `terrain` (info, `{height_at:[x,z]}`, `{flatten|raise|paint: ...}`, `{chunks:true}`).

use crate::app::Ctx;
use crate::mesh::{MeshData, Vertex};
use crate::renderer::{Lod, Material, MaterialId, MeshId, TextureId};
use crate::scene::{MeshRenderer, Name, Parent, Transform};
use glam::{DVec3, Quat, Vec2, Vec3};
use hecs::Entity;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Serialize)]
pub struct TerrainDesc {
    /// Absolute world XZ of the terrain's minimum corner.
    pub origin: Vec2,
    /// Edge length in metres (square).
    pub size: f32,
    /// Height samples per side (2^n + 1 recommended, e.g. 513 for 2 m spacing at 1 km).
    pub res: u32,
    /// Quads per chunk side (power of two, 16..64).
    pub chunk_quads: u32,
    /// Chunks whose centre is within this distance of the camera are loaded.
    pub stream_radius: f32,
    /// Max chunk builds per frame (amortises streaming hitches).
    pub builds_per_frame: u32,
    /// Skirt depth (m) hiding LOD cracks.
    pub skirt: f32,
    pub collider: bool,
    pub cast_shadows: bool,
    /// Texture tiles: metres per tile for each splat layer.
    pub tiling: [f32; 4],
}

impl Default for TerrainDesc {
    fn default() -> Self {
        Self { origin: Vec2::new(-512.0, -512.0), size: 1024.0, res: 513, chunk_quads: 32, stream_radius: 700.0, builds_per_frame: 4, skirt: 3.0, collider: true, cast_shadows: true, tiling: [6.0, 8.0, 5.0, 7.0] }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct NoiseParams {
    pub seed: u32,
    /// Height range (m) of the noise.
    pub amplitude: f32,
    /// Features per km.
    pub frequency: f32,
    pub octaves: usize,
    /// 0 = rolling hills, 1 = ridged mountains.
    pub ridged: f32,
    /// > 0: edges fall off into the sea (island), value = falloff width as fraction of the size.
    pub island: f32,
    /// Added to every height.
    pub base: f32,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self { seed: 1, amplitude: 50.0, frequency: 3.0, octaves: 6, ridged: 0.35, island: 0.0, base: 0.0 }
    }
}

/// Auto-splat: layer 0 = flat ground (grass), 1 = steep (rock), 2 = low (sand/beach), 3 = high (snow/dirt).
#[derive(Clone, Debug, Serialize)]
pub struct SplatRules {
    /// Slope (degrees) where layer 1 takes over.
    pub rock_slope: f32,
    /// Below this height layer 2.
    pub low: f32,
    /// Above this height layer 3.
    pub high: f32,
    /// Blend width (m / degrees).
    pub blend: f32,
}

impl Default for SplatRules {
    fn default() -> Self {
        Self { rock_slope: 32.0, low: 1.5, high: 70.0, blend: 4.0 }
    }
}

/// Places instances of a mesh on the terrain per chunk (deterministic, re-created when the chunk
/// streams in). Trees, bushes, rocks, lamp posts along roads (with your own rule data).
#[derive(Clone, Debug, Serialize)]
pub struct ScatterRule {
    #[serde(skip)]
    pub mesh: MeshId,
    #[serde(skip)]
    pub material: MaterialId,
    /// Instances per 100 m² at full splat weight.
    pub density: f32,
    /// Splat layer whose weight scales the density (None = everywhere).
    pub layer: Option<usize>,
    pub min_height: f32,
    pub max_height: f32,
    pub max_slope_deg: f32,
    pub scale: (f32, f32),
    /// Tilt with the ground normal (rocks) instead of standing upright (trees).
    pub align: bool,
    /// Collider radius (0 = none): a fixed cylinder of that radius.
    pub collider_radius: f32,
    pub collider_height: f32,
    pub cast_shadows: bool,
    // (rules with the same `seed` + density + filters place instances at the same positions)
    /// Random offset inside each placement cell (0..1 of the cell). < 1 keeps a minimum spacing
    /// of (1 - jitter) cells between instances (no interpenetrating rocks).
    pub jitter: f32,
    /// Drawn up to this distance (LOD cull).
    pub max_distance: f32,
    pub seed: u32,
}

impl Default for ScatterRule {
    fn default() -> Self {
        Self { mesh: MeshId(0), material: MaterialId(0), density: 1.0, layer: Some(0), min_height: f32::MIN, max_height: f32::MAX, max_slope_deg: 30.0, scale: (0.8, 1.2), align: false, collider_radius: 0.0, collider_height: 4.0, cast_shadows: true, jitter: 1.0, max_distance: 400.0, seed: 1 }
    }
}

pub struct Terrain {
    pub desc: TerrainDesc,
    /// Row-major heights (z rows, x columns), metres.
    pub heights: Vec<f32>,
    /// Row-major splat weights (layers 0..3) per height sample.
    pub splat: Vec<[u8; 4]>,
    pub scatter: Vec<ScatterRule>,
    pub(crate) chunks: HashMap<(i32, i32), Entity>,
    pub(crate) chunk_meshes: HashMap<(i32, i32), Vec<MeshId>>,
    pub(crate) dirty: HashSet<(i32, i32)>,
    pub(crate) splat_dirty: bool,
    pub(crate) material: Option<MaterialId>,
    pub(crate) splat_tex: Option<TextureId>,
    pub root: Option<Entity>,
    pub built: u64,
    pub unloaded: u64,
}

fn hash(a: i32, b: i32, s: u32) -> u32 {
    let mut h = (a as u32).wrapping_mul(0x8da6b343) ^ (b as u32).wrapping_mul(0xd8163841) ^ s.wrapping_mul(0xcb1ab31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1e995);
    h ^ (h >> 15)
}
fn hashf(a: i32, b: i32, s: u32) -> f32 {
    (hash(a, b, s) & 0xffffff) as f32 / 16777216.0
}

impl Terrain {
    pub fn flat(desc: TerrainDesc, height: f32) -> Self {
        let n = (desc.res * desc.res) as usize;
        Self { heights: vec![height; n], splat: vec![[255, 0, 0, 0]; n], desc, scatter: vec![], chunks: HashMap::new(), chunk_meshes: HashMap::new(), dirty: HashSet::new(), splat_dirty: true, material: None, splat_tex: None, root: None, built: 0, unloaded: 0 }
    }

    /// Fractal noise heights (fbm blended with ridged multifractal), optional island falloff.
    pub fn noise(desc: TerrainDesc, p: NoiseParams) -> Self {
        use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};
        let mut t = Self::flat(desc, 0.0);
        let res = t.desc.res as usize;
        let fbm = Fbm::<Perlin>::new(p.seed).set_octaves(p.octaves);
        let ridged = RidgedMulti::<Perlin>::new(p.seed.wrapping_add(11)).set_octaves(p.octaves.min(6));
        let f = p.frequency as f64 / 1000.0;
        let rows: Vec<usize> = (0..res).collect();
        let size = t.desc.size;
        let origin = t.desc.origin;
        let out = crate::jobs::par_map(&rows, |&z| {
            (0..res)
                .map(|x| {
                    let wx = origin.x as f64 + x as f64 / (res - 1) as f64 * size as f64;
                    let wz = origin.y as f64 + z as f64 / (res - 1) as f64 * size as f64;
                    let a = fbm.get([wx * f, wz * f]) * 0.5 + 0.5;
                    let r = ridged.get([wx * f * 0.7, wz * f * 0.7]) * 0.5 + 0.5;
                    let mut h = (a * (1.0 - p.ridged as f64) + r * r * p.ridged as f64) as f32 * p.amplitude + p.base;
                    if p.island > 0.0 {
                        let u = Vec2::new(x as f32, z as f32) / (res - 1) as f32 * 2.0 - 1.0;
                        let d = u.abs().max_element().max(u.length() * 0.8);
                        let k = ((1.0 - d) / p.island).clamp(0.0, 1.0);
                        let k = k * k * (3.0 - 2.0 * k);
                        h = h * k - (1.0 - k) * 6.0;
                    }
                    h
                })
                .collect::<Vec<f32>>()
        });
        t.heights = out.into_iter().flatten().collect();
        t
    }

    pub fn spacing(&self) -> f32 {
        self.desc.size / (self.desc.res - 1) as f32
    }

    fn sample(&self, x: usize, z: usize) -> f32 {
        let r = self.desc.res as usize;
        self.heights[z.min(r - 1) * r + x.min(r - 1)]
    }

    /// Grid coordinates (fractional) of an absolute world XZ.
    fn grid(&self, x: f32, z: f32) -> Vec2 {
        (Vec2::new(x, z) - self.desc.origin) / self.spacing()
    }

    /// Bilinear height at absolute world XZ (clamped to the terrain).
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let g = self.grid(x, z).clamp(Vec2::ZERO, Vec2::splat((self.desc.res - 1) as f32));
        let (x0, z0) = (g.x.floor() as usize, g.y.floor() as usize);
        let (fx, fz) = (g.x - x0 as f32, g.y - z0 as f32);
        let h00 = self.sample(x0, z0);
        let h10 = self.sample(x0 + 1, z0);
        let h01 = self.sample(x0, z0 + 1);
        let h11 = self.sample(x0 + 1, z0 + 1);
        (h00 * (1.0 - fx) + h10 * fx) * (1.0 - fz) + (h01 * (1.0 - fx) + h11 * fx) * fz
    }

    pub fn normal_at(&self, x: f32, z: f32) -> Vec3 {
        let e = self.spacing();
        let dx = self.height_at(x + e, z) - self.height_at(x - e, z);
        let dz = self.height_at(x, z + e) - self.height_at(x, z - e);
        Vec3::new(-dx, 2.0 * e, -dz).normalize()
    }

    /// Splat weights (0..1) at absolute world XZ (nearest sample).
    pub fn splat_at(&self, x: f32, z: f32) -> [f32; 4] {
        let g = self.grid(x, z).round().clamp(Vec2::ZERO, Vec2::splat((self.desc.res - 1) as f32));
        let s = self.splat[g.y as usize * self.desc.res as usize + g.x as usize];
        s.map(|v| v as f32 / 255.0)
    }

    pub fn contains(&self, x: f32, z: f32) -> bool {
        let g = self.grid(x, z);
        g.x >= 0.0 && g.y >= 0.0 && g.x <= (self.desc.res - 1) as f32 && g.y <= (self.desc.res - 1) as f32
    }

    fn mark_dirty_around(&mut self, c: Vec2, r: f32) {
        let cs = self.chunk_size();
        let o = self.desc.origin;
        let (x0, x1) = (((c.x - r - o.x) / cs).floor() as i32 - 1, ((c.x + r - o.x) / cs).floor() as i32 + 1);
        let (z0, z1) = (((c.y - r - o.y) / cs).floor() as i32 - 1, ((c.y + r - o.y) / cs).floor() as i32 + 1);
        for cx in x0..=x1 {
            for cz in z0..=z1 {
                if self.chunks.contains_key(&(cx, cz)) {
                    self.dirty.insert((cx, cz));
                }
            }
        }
    }

    /// Visits samples within `radius` of `center` (absolute XZ) with a 0..1 falloff weight.
    fn brush(&mut self, center: Vec2, radius: f32, falloff: f32, mut f: impl FnMut(&mut f32, &mut [u8; 4], f32)) {
        let g = self.grid(center.x, center.y);
        let rr = (radius + falloff) / self.spacing();
        let res = self.desc.res as i32;
        for z in (g.y - rr).floor().max(0.0) as i32..=((g.y + rr).ceil() as i32).min(res - 1) {
            for x in (g.x - rr).floor().max(0.0) as i32..=((g.x + rr).ceil() as i32).min(res - 1) {
                let d = Vec2::new(x as f32, z as f32).distance(g) * self.spacing();
                if d > radius + falloff {
                    continue;
                }
                let w = if d <= radius { 1.0 } else { 1.0 - ((d - radius) / falloff.max(1e-3)) };
                let w = w * w * (3.0 - 2.0 * w);
                let i = (z * res + x) as usize;
                let (h, s) = (&mut self.heights[i], &mut self.splat[i]);
                f(h, s, w);
            }
        }
        self.mark_dirty_around(center, radius + falloff);
    }

    /// Levels a disc to `height` (with a smooth `falloff` ring): building plots, plazas.
    pub fn flatten(&mut self, center: Vec2, radius: f32, height: f32, falloff: f32) {
        self.brush(center, radius, falloff, |h, _, w| *h += (height - *h) * w);
    }

    /// Levels a corridor along a polyline (roads, rails): height follows the terrain smoothed
    /// along the path unless `height` is given.
    pub fn flatten_path(&mut self, path: &[Vec2], width: f32, falloff: f32) {
        let pts: Vec<(Vec2, f32)> = path.iter().map(|p| (*p, self.height_at(p.x, p.y))).collect();
        for w in pts.windows(2) {
            let ((a, ha), (b, hb)) = (w[0], w[1]);
            let steps = (a.distance(b) / (self.spacing() * 0.5)).ceil().max(1.0) as usize;
            for s in 0..=steps {
                let t = s as f32 / steps as f32;
                self.flatten(a.lerp(b, t), width * 0.5, ha + (hb - ha) * t, falloff);
            }
        }
    }

    pub fn raise(&mut self, center: Vec2, radius: f32, amount: f32, falloff: f32) {
        self.brush(center, radius, falloff, |h, _, w| *h += amount * w);
    }

    /// Paints splat layer `layer` with `strength` (0..1) in a disc.
    pub fn paint(&mut self, layer: usize, center: Vec2, radius: f32, strength: f32, falloff: f32) {
        let layer = layer.min(3);
        self.brush(center, radius, falloff, |_, s, w| {
            let a = (strength * w).clamp(0.0, 1.0);
            for (k, v) in s.iter_mut().enumerate() {
                let target = if k == layer { 255.0 } else { 0.0 };
                *v = (*v as f32 + (target - *v as f32) * a).round() as u8;
            }
        });
        self.splat_dirty = true;
    }

    /// Splat from slope and height (see [`SplatRules`]).
    pub fn auto_splat(&mut self, r: &SplatRules) {
        let res = self.desc.res as usize;
        let e = self.spacing();
        for z in 0..res {
            for x in 0..res {
                let dx = self.sample(x + 1, z) - self.sample(x.saturating_sub(1), z);
                let dz = self.sample(x, z + 1) - self.sample(x, z.saturating_sub(1));
                let slope = (Vec2::new(dx, dz).length() / (2.0 * e)).atan().to_degrees();
                let h = self.sample(x, z);
                let s = |v: f32| v.clamp(0.0, 1.0);
                let rock = s((slope - r.rock_slope) / r.blend.max(0.1) + 0.5);
                let low = s((r.low - h) / r.blend.max(0.1) + 0.5) * (1.0 - rock);
                let high = s((h - r.high) / (r.blend * 3.0).max(0.1) + 0.5) * (1.0 - rock);
                let grass = (1.0 - rock - low - high).max(0.0);
                let sum = (grass + rock + low + high).max(1e-3);
                self.splat[z * res + x] = [grass, rock, low, high].map(|w| (w / sum * 255.0).round() as u8);
            }
        }
        self.splat_dirty = true;
        for k in self.chunks.keys().copied().collect::<Vec<_>>() {
            self.dirty.insert(k);
        }
    }

    /// Chunks currently loaded (streamed in).
    pub fn loaded_chunks(&self) -> usize {
        self.chunks.len()
    }

    pub fn chunk_size(&self) -> f32 {
        self.desc.chunk_quads as f32 * self.spacing()
    }

    pub fn chunk_count(&self) -> i32 {
        ((self.desc.res - 1) / self.desc.chunk_quads) as i32
    }

    /// LOD meshes of chunk (cx, cz), vertices relative to the chunk centre.
    fn build_chunk_meshes(&self, cx: i32, cz: i32) -> (Vec<MeshData>, Vec3) {
        let q = self.desc.chunk_quads as usize;
        let (x0, z0) = (cx as usize * q, cz as usize * q);
        let e = self.spacing();
        let cs = self.chunk_size();
        let center = Vec3::new(self.desc.origin.x + (cx as f32 + 0.5) * cs, 0.0, self.desc.origin.y + (cz as f32 + 0.5) * cs);
        let normal = |x: usize, z: usize| {
            let dx = self.sample(x + 1, z) - self.sample(x.saturating_sub(1), z);
            let dz = self.sample(x, z + 1) - self.sample(x, z.saturating_sub(1));
            Vec3::new(-dx, 2.0 * e, -dz).normalize()
        };
        let mut lods = vec![];
        for l in 0..4 {
            let step = 1usize << l;
            if step > q / 2 {
                break;
            }
            let n = q / step + 1;
            let mut md = MeshData::default();
            for j in 0..n {
                for i in 0..n {
                    let (x, z) = (x0 + i * step, z0 + j * step);
                    let p = Vec3::new(self.desc.origin.x + x as f32 * e, self.sample(x, z), self.desc.origin.y + z as f32 * e) - center;
                    md.vertices.push(Vertex { pos: p.into(), normal: normal(x, z).into(), uv: [i as f32 / (n - 1) as f32, j as f32 / (n - 1) as f32] });
                }
            }
            for j in 0..n - 1 {
                for i in 0..n - 1 {
                    let a = (j * n + i) as u32;
                    let (b, c, d) = (a + 1, a + n as u32, a + n as u32 + 1);
                    md.indices.extend_from_slice(&[a, c, b, b, c, d]);
                }
            }
            // skirts: the border ring duplicated downwards (hides cracks between LOD levels)
            let depth = self.desc.skirt * (1.0 + l as f32);
            let border: Vec<u32> = (0..n).map(|i| i as u32) // z = 0 edge, x increasing
                .chain((1..n).map(|j| (j * n + n - 1) as u32))
                .chain((0..n - 1).rev().map(|i| ((n - 1) * n + i) as u32))
                .chain((1..n - 1).rev().map(|j| (j * n) as u32))
                .collect();
            let base = md.vertices.len() as u32;
            for &b in &border {
                let mut v = md.vertices[b as usize];
                v.pos[1] -= depth;
                md.vertices.push(v);
            }
            let m = border.len();
            for k in 0..m {
                let (a, b) = (border[k], border[(k + 1) % m]);
                let (c, d) = (base + k as u32, base + ((k + 1) % m) as u32);
                md.indices.extend_from_slice(&[a, b, c, b, d, c]);
            }
            lods.push(md);
        }
        (lods, center)
    }

    /// Height samples of chunk (cx, cz) for its heightfield collider (parry: rows = z, cols = x, column-major).
    fn chunk_heightfield(&self, cx: i32, cz: i32) -> (Vec<f32>, usize) {
        let q = self.desc.chunk_quads as usize;
        let (x0, z0) = (cx as usize * q, cz as usize * q);
        let n = q + 1;
        let mut data = vec![0.0; n * n];
        for x in 0..n {
            for z in 0..n {
                data[x * n + z] = self.sample(x0 + x, z0 + z);
            }
        }
        (data, n)
    }

    pub fn describe(&self) -> Value {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for h in &self.heights {
            lo = lo.min(*h);
            hi = hi.max(*h);
        }
        json!({"desc": self.desc, "spacing": self.spacing(), "chunk_size": self.chunk_size(), "chunks_total": self.chunk_count() * self.chunk_count(),
               "chunks_loaded": self.chunks.len(), "chunks_built": self.built, "chunks_unloaded": self.unloaded, "dirty": self.dirty.len(),
               "height_range": [lo, hi], "scatter_rules": self.scatter})
    }
}

/// Builds a 2x2 atlas (512²) from four seamless 256² layer textures with wrapped padding, so a
/// layer can tile inside its quadrant without bleeding into its neighbours.
fn atlas(layers: &[Vec<u8>; 4], size: usize) -> Vec<u8> {
    let cell = size;
    let pad = cell / 16;
    let mut out = vec![0u8; cell * 2 * cell * 2 * 4];
    for (k, src) in layers.iter().enumerate() {
        let (ox, oy) = ((k % 2) * cell, (k / 2) * cell);
        for y in 0..cell {
            for x in 0..cell {
                let u = (x as f32 - pad as f32) / (cell - 2 * pad) as f32;
                let v = (y as f32 - pad as f32) / (cell - 2 * pad) as f32;
                let sx = ((u.rem_euclid(1.0)) * cell as f32) as usize % cell;
                let sy = ((v.rem_euclid(1.0)) * cell as f32) as usize % cell;
                let si = (sy * cell + sx) * 4;
                let di = ((oy + y) * cell * 2 + ox + x) * 4;
                out[di..di + 4].copy_from_slice(&src[si..si + 4]);
            }
        }
    }
    out
}

impl Ctx {
    /// Installs `t` as the world terrain with the four splat layer textures (texgen presets).
    pub fn set_terrain(&mut self, mut t: Terrain, layers: [crate::texgen::TexPreset; 4]) {
        if let Some(old) = self.terrain.take() {
            for (_, e) in old.chunks {
                self.despawn(e);
            }
            if let Some(r) = old.root {
                self.despawn(r);
            }
        }
        let t0 = crate::log::now_ms();
        let px: Vec<Vec<u8>> = crate::jobs::par_map(&layers.to_vec(), |p| crate::texgen::TexGen::new(*p).size(256).seed(7).generate());
        let px: [Vec<u8>; 4] = [px[0].clone(), px[1].clone(), px[2].clone(), px[3].clone()];
        let atlas_tex = self.renderer.add_texture_rgba(512, 512, &atlas(&px, 256), "terrain-atlas").ok();
        let splat_bytes: Vec<u8> = t.splat.iter().flatten().copied().collect();
        let res = t.desc.res;
        t.splat_tex = self.renderer.add_texture_rgba_linear(res, res, &splat_bytes, "terrain-splat").ok();
        t.splat_dirty = false;
        let mat = Material { name: "terrain".into(), texture: atlas_tex, normal_texture: t.splat_tex, roughness: 0.92, terrain: true, ..Material::default() };
        t.material = Some(self.renderer.add_material(mat));
        // GPU grass inputs: heightmap (r32float) + splat map
        let (dev, q) = (self.renderer.device.clone(), self.renderer.queue.clone());
        if let Err(e) = self.renderer.grass.upload_heights(&dev, &q, &t.heights, res) {
            crate::log_warn!("grass heightmap: {e}");
        }
        if let Some(v) = t.splat_tex.and_then(|id| self.renderer.texture_view(id)) {
            self.renderer.grass.set_splat(v);
        }
        // the root lives in the persistent scene: terrain chunks stream themselves
        let prev = self.scenes.current().to_string();
        self.end_scene();
        t.root = Some(self.world.spawn((Transform::default(), Name("terrain".into()))));
        self.begin_scene(&prev);
        crate::log_info!("terrain {}x{} m, {}² samples, {} chunks ({:.0} ms setup)", t.desc.size, t.desc.size, res, t.chunk_count() * t.chunk_count(), crate::log::now_ms() - t0);
        self.terrain = Some(t);
    }

    /// Removes the terrain: unloads every chunk (meshes, colliders, scattered instances).
    pub fn remove_terrain(&mut self) {
        let keys: Vec<(i32, i32)> = self.terrain.as_ref().map(|t| t.chunks.keys().copied().collect()).unwrap_or_default();
        for k in keys {
            self.unload_chunk(k);
        }
        if let Some(t) = self.terrain.take() {
            if let Some(r) = t.root {
                self.despawn(r);
            }
        }
        self.renderer.grass.terrain = None;
    }

    /// Height at absolute world XZ (0 without terrain).
    pub fn terrain_height(&self, x: f32, z: f32) -> f32 {
        self.terrain.as_ref().map(|t| t.height_at(x, z)).unwrap_or(0.0)
    }

    /// Absolute XZ of a local position (origin shift aware).
    pub fn to_absolute(&self, local: Vec3) -> DVec3 {
        self.origin + local.as_dvec3()
    }

    fn build_chunk(&mut self, key: (i32, i32)) {
        let Some(t) = self.terrain.as_ref() else { return };
        let (lods, center) = t.build_chunk_meshes(key.0, key.1);
        let hf = (t.desc.collider).then(|| t.chunk_heightfield(key.0, key.1));
        let (cs, mat, cast, root, origin) = (t.chunk_size(), t.material.unwrap_or(self.renderer.builtin.default_material), t.desc.cast_shadows, t.root, self.origin);
        let rules = t.scatter.clone();
        // old meshes of this chunk (rebuild after edits) are freed
        if let Some(old) = self.terrain.as_mut().and_then(|t| t.chunk_meshes.remove(&key)) {
            for m in old {
                self.renderer.remove_mesh(m);
            }
        }
        if let Some(e) = self.terrain.as_mut().and_then(|t| t.chunks.remove(&key)) {
            self.despawn(e);
        }
        let ids: Vec<MeshId> = lods.iter().enumerate().map(|(l, md)| self.renderer.add_mesh(&format!("terrain:{}:{}#{l}", key.0, key.1), md)).collect();
        let local = (center.as_dvec3() - origin).as_vec3();
        let mut mr = MeshRenderer::new(ids[0], mat);
        mr.cast_shadows = cast;
        let lod = Lod { meshes: ids.clone(), distances: (1..ids.len()).map(|l| cs * (0.9 + 1.2 * l as f32 * l as f32)).collect(), cull_beyond: false };
        let prev = self.scenes.current().to_string();
        self.end_scene();
        let mut builder = hecs::EntityBuilder::new();
        builder.add(Transform::from_translation(local)).add(Name(format!("terrain {},{}", key.0, key.1))).add(mr).add(lod);
        if let Some(r) = root {
            builder.add(Parent(r));
        }
        let e = self.world.spawn(builder.build());
        // parented chunks are not physics roots: keep chunks as roots for colliders
        if let Some((data, n)) = hf {
            let _ = self.world.remove_one::<Parent>(e);
            let scale = Vec3::new(cs, 1.0, cs);
            let col = crate::physics::rapier::prelude::ColliderBuilder::heightfield(crate::physics::rapier::parry::utils::Array2::new(n, n, data), scale).friction(0.9);
            self.physics.add(&mut self.world, e, crate::physics::rapier::prelude::RigidBodyBuilder::fixed(), col);
        }
        // scatter
        for (ri, r) in rules.iter().enumerate() {
            self.scatter_chunk(e, key, center, cs, ri as u32, r);
        }
        self.begin_scene(&prev);
        if let Some(t) = self.terrain.as_mut() {
            t.chunks.insert(key, e);
            t.chunk_meshes.insert(key, ids);
            t.built += 1;
        }
    }

    fn scatter_chunk(&mut self, chunk: Entity, key: (i32, i32), center: Vec3, cs: f32, ri: u32, r: &ScatterRule) {
        let Some(t) = self.terrain.as_ref() else { return };
        // One candidate per cell of a *global* grid (anchored at the terrain corner): every cell
        // belongs to exactly one chunk, so neighbouring chunks never place duplicates and the
        // layout does not depend on the chunk size.
        let cell = (10.0 / r.density.max(1e-3).sqrt()).max(0.5);
        let o = t.desc.origin;
        let (min_x, min_z) = (center.x - cs * 0.5 - o.x, center.z - cs * 0.5 - o.y);
        let (i0, i1) = ((min_x / cell).ceil() as i32, ((min_x + cs) / cell).ceil() as i32);
        let (j0, j1) = ((min_z / cell).ceil() as i32, ((min_z + cs) / cell).ceil() as i32);
        let _ = key;
        let mut placed: Vec<(Vec3, Quat, f32)> = vec![];
        for gz in j0..j1 {
            for gx in i0..i1 {
                // positions depend on the rule's seed only: rules with the same seed place the
                // parts of one multi-material prop (trunk + leaves) at the same spots
                let _ = ri;
                let s = r.seed;
                let jit = r.jitter.clamp(0.0, 1.0);
                let px = o.x + (gx as f32 + 0.5 + (hashf(gx, gz, s) - 0.5) * jit) * cell;
                let pz = o.y + (gz as f32 + 0.5 + (hashf(gx, gz, s ^ 0x55) - 0.5) * jit) * cell;
                let w = r.layer.map(|l| t.splat_at(px, pz)[l.min(3)]).unwrap_or(1.0);
                if hashf(gx, gz, s ^ 0xaa) > w {
                    continue;
                }
                let h = t.height_at(px, pz);
                let nrm = t.normal_at(px, pz);
                if h < r.min_height || h > r.max_height || nrm.y.acos().to_degrees() > r.max_slope_deg {
                    continue;
                }
                let yaw = Quat::from_rotation_y(hashf(gx, gz, s ^ 0x33) * std::f32::consts::TAU);
                let rot = if r.align { Quat::from_rotation_arc(Vec3::Y, nrm) * yaw } else { yaw };
                let sc = r.scale.0 + (r.scale.1 - r.scale.0) * hashf(gx, gz, s ^ 0x77);
                placed.push((Vec3::new(px, h, pz), rot, sc));
            }
        }
        let origin = self.origin;
        let lod = Lod { meshes: vec![r.mesh], distances: vec![r.max_distance], cull_beyond: true };
        for (p, rot, sc) in placed {
            let local = (p.as_dvec3() - origin).as_vec3();
            let chunk_pos = self.world.get::<&Transform>(chunk).map(|t| t.translation).unwrap_or_default();
            let mut mr = MeshRenderer::new(r.mesh, r.material);
            mr.cast_shadows = r.cast_shadows;
            let e = self.world.spawn((Transform::from_translation(local - chunk_pos).with_rotation(rot).with_scale(Vec3::splat(sc)), Parent(chunk), mr, lod.clone(), Name("scatter".into())));
            if r.collider_radius > 0.0 {
                // colliders on roots only: a separate fixed collider at the world position
                let h = self.physics.add_static(crate::physics::rapier::prelude::ColliderBuilder::cylinder(r.collider_height * 0.5 * sc, r.collider_radius * sc), local + Vec3::Y * r.collider_height * 0.5 * sc);
                let _ = self.world.insert_one(e, ScatterCollider(h));
            }
        }
    }

    /// Per frame: stream chunks around the camera, rebuild edited ones, re-upload the splat map.
    pub(crate) fn update_terrain(&mut self) {
        let Some(t) = self.terrain.as_ref() else {
            self.renderer.grass.terrain = None;
            return;
        };
        let cam = self
            .world
            .query::<(&crate::scene::Camera, &crate::scene::GlobalTransform)>()
            .without::<&crate::render_texture::RenderTextureCamera>()
            .iter()
            .next()
            .map(|(_, g)| g.translation());
        let Some(cam) = cam else { return };
        let cam_abs = self.to_absolute(cam);
        let cs = t.chunk_size();
        let n = t.chunk_count();
        let r = t.desc.stream_radius;
        let o = t.desc.origin;
        let center = |cx: i32, cz: i32| Vec2::new(o.x + (cx as f32 + 0.5) * cs, o.y + (cz as f32 + 0.5) * cs);
        let cam2 = Vec2::new(cam_abs.x as f32, cam_abs.z as f32);
        let mut want: Vec<(f32, (i32, i32))> = vec![];
        let x0 = (((cam2.x - r - o.x) / cs).floor() as i32).max(0);
        let x1 = (((cam2.x + r - o.x) / cs).floor() as i32).min(n - 1);
        let z0 = (((cam2.y - r - o.y) / cs).floor() as i32).max(0);
        let z1 = (((cam2.y + r - o.y) / cs).floor() as i32).min(n - 1);
        for cz in z0..=z1 {
            for cx in x0..=x1 {
                let d = center(cx, cz).distance(cam2);
                if d <= r && !t.chunks.contains_key(&(cx, cz)) {
                    want.push((d, (cx, cz)));
                }
            }
        }
        want.sort_by(|a, b| a.0.total_cmp(&b.0));
        let drop: Vec<(i32, i32)> = t.chunks.keys().copied().filter(|&(cx, cz)| center(cx, cz).distance(cam2) > r * 1.15 + cs).collect();
        let budget = t.desc.builds_per_frame.max(1) as usize;
        let mut dirty: Vec<(i32, i32)> = t.dirty.iter().copied().collect();
        dirty.sort();
        let splat_dirty = t.splat_dirty;
        let heights_dirty = !t.dirty.is_empty();
        for key in drop {
            self.unload_chunk(key);
        }
        for (_, key) in want.into_iter().take(budget) {
            self.build_chunk(key);
        }
        for key in dirty.into_iter().take(budget * 2) {
            if let Some(t) = self.terrain.as_mut() {
                t.dirty.remove(&key);
            }
            if self.terrain.as_ref().is_some_and(|t| t.chunks.contains_key(&key)) {
                self.build_chunk(key);
            }
        }
        if heights_dirty {
            if let Some(t) = self.terrain.as_ref() {
                let (dev, q) = (self.renderer.device.clone(), self.renderer.queue.clone());
                let _ = self.renderer.grass.upload_heights(&dev, &q, &t.heights, t.desc.res);
            }
        }
        if splat_dirty {
            if let Some(t) = self.terrain.as_mut() {
                t.splat_dirty = false;
                let bytes: Vec<u8> = t.splat.iter().flatten().copied().collect();
                if let Some(id) = t.splat_tex {
                    let res = t.desc.res;
                    if let Err(e) = self.renderer.write_texture_rgba(id, res, res, &bytes) {
                        crate::log_warn!("terrain splat upload: {e}");
                    }
                }
            }
        }
        // terrain shading needs the local (origin-shifted) corner
        if let Some(t) = self.terrain.as_ref() {
            let local = (DVec3::new(t.desc.origin.x as f64, 0.0, t.desc.origin.y as f64) - self.origin).as_vec3();
            let (tiling, size) = (t.desc.tiling, t.desc.size);
            self.renderer.grass.terrain = Some(([local.x, local.z], size));
            if let Some(m) = t.material.and_then(|m| self.renderer.material_mut(m)) {
                m.terrain_params = [local.x, local.z, 1.0 / size, 0.0];
                m.terrain_tiling = tiling;
            }
        }
    }

    fn unload_chunk(&mut self, key: (i32, i32)) {
        let Some(t) = self.terrain.as_mut() else { return };
        let e = t.chunks.remove(&key);
        let meshes = t.chunk_meshes.remove(&key).unwrap_or_default();
        t.unloaded += 1;
        if let Some(e) = e {
            // scatter colliders are separate statics
            let statics: Vec<crate::physics::rapier::prelude::ColliderHandle> = crate::scene::descendants(&self.world, e).iter().filter_map(|d| self.world.get::<&ScatterCollider>(*d).ok().map(|c| c.0)).collect();
            for h in statics {
                self.physics.world.remove_collider(h);
            }
            self.despawn(e);
        }
        for m in meshes {
            self.renderer.remove_mesh(m);
        }
    }

    /// AI API `terrain`.
    pub(crate) fn terrain_command(&mut self, a: &Value) -> Result<Value, String> {
        let v2 = |v: &Value| v.as_array().map(|a| Vec2::new(a[0].as_f64().unwrap_or(0.0) as f32, a[1].as_f64().unwrap_or(0.0) as f32));
        let t = self.terrain.as_mut().ok_or("no terrain (ctx.set_terrain)")?;
        if let Some(p) = v2(&a["height_at"]) {
            return Ok(json!({"pos": [p.x, p.y], "height": t.height_at(p.x, p.y), "normal": t.normal_at(p.x, p.y).to_array(), "splat": t.splat_at(p.x, p.y), "inside": t.contains(p.x, p.y)}));
        }
        if let Some(f) = a.get("flatten") {
            let c = v2(&f["center"]).ok_or("flatten needs center [x,z]")?;
            let h = f["height"].as_f64().map(|h| h as f32).unwrap_or_else(|| t.height_at(c.x, c.y));
            t.flatten(c, f["radius"].as_f64().unwrap_or(10.0) as f32, h, f["falloff"].as_f64().unwrap_or(8.0) as f32);
        }
        if let Some(f) = a.get("raise") {
            let c = v2(&f["center"]).ok_or("raise needs center [x,z]")?;
            t.raise(c, f["radius"].as_f64().unwrap_or(10.0) as f32, f["amount"].as_f64().unwrap_or(2.0) as f32, f["falloff"].as_f64().unwrap_or(8.0) as f32);
        }
        if let Some(f) = a.get("paint") {
            let c = v2(&f["center"]).ok_or("paint needs center [x,z]")?;
            t.paint(f["layer"].as_u64().unwrap_or(0) as usize, c, f["radius"].as_f64().unwrap_or(6.0) as f32, f["strength"].as_f64().unwrap_or(1.0) as f32, f["falloff"].as_f64().unwrap_or(3.0) as f32);
        }
        let mut out = t.describe();
        if a["chunks"].as_bool() == Some(true) {
            let mut keys: Vec<_> = t.chunks.keys().map(|k| [k.0, k.1]).collect();
            keys.sort();
            out["loaded"] = json!(keys);
        }
        Ok(out)
    }
}

/// Static collider created for a scattered instance (removed with its chunk).
#[derive(Clone, Copy, Debug)]
pub struct ScatterCollider(pub crate::physics::rapier::prelude::ColliderHandle);

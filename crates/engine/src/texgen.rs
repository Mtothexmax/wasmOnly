//! Procedural textures (presets). CPU generation with the `noise` crate (pure Rust, wasm-safe),
//! rows computed with the job system (parallel on the threads backend), uploaded with mips.
//!
//! Every preset tiles seamlessly: noise is sampled on a 4D torus `(cos u, sin u, cos v, sin v)`,
//! so `Material.texture` + a mesh with UVs > 1 repeats without seams. Same `TexGen` = same pixels.
//!
//! ```ignore
//! let tex = ctx.generate_texture(&TexGen::new(TexPreset::Grass).size(256).seed(7));
//! let mat = ctx.renderer.add_material(Material { texture: Some(tex), ..Default::default() });
//! ```
//! AI: `engine.call("texgen", {preset, size?, seed?, scale?, colors?: ['#rrggbb','#rrggbb'], apply_to?: id})`.

use noise::{Billow, Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti, Worley};
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TexPreset {
    /// Grayscale fractal noise between the two colors.
    Noise,
    Clouds,
    Marble,
    Wood,
    Stone,
    Grass,
    Sand,
    Dirt,
    Lava,
    Water,
    Bricks,
    Tiles,
    Metal,
    Rust,
    Camo,
    Checker,
    /// Grass blades with alpha (use with `Material::cutout`).
    Blades,
    /// Leaf clusters with alpha (use with `Material::cutout`).
    Leaves,
}

impl TexPreset {
    pub const ALL: [TexPreset; 18] = [
        TexPreset::Noise,
        TexPreset::Clouds,
        TexPreset::Marble,
        TexPreset::Wood,
        TexPreset::Stone,
        TexPreset::Grass,
        TexPreset::Sand,
        TexPreset::Dirt,
        TexPreset::Lava,
        TexPreset::Water,
        TexPreset::Bricks,
        TexPreset::Tiles,
        TexPreset::Metal,
        TexPreset::Rust,
        TexPreset::Camo,
        TexPreset::Checker,
        TexPreset::Blades,
        TexPreset::Leaves,
    ];

    pub fn name(self) -> &'static str {
        match self {
            TexPreset::Noise => "noise",
            TexPreset::Clouds => "clouds",
            TexPreset::Marble => "marble",
            TexPreset::Wood => "wood",
            TexPreset::Stone => "stone",
            TexPreset::Grass => "grass",
            TexPreset::Sand => "sand",
            TexPreset::Dirt => "dirt",
            TexPreset::Lava => "lava",
            TexPreset::Water => "water",
            TexPreset::Bricks => "bricks",
            TexPreset::Tiles => "tiles",
            TexPreset::Metal => "metal",
            TexPreset::Rust => "rust",
            TexPreset::Camo => "camo",
            TexPreset::Checker => "checker",
            TexPreset::Blades => "blades",
            TexPreset::Leaves => "leaves",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == s)
    }

    /// Default (dark, light) sRGB colors.
    pub fn default_colors(self) -> [u32; 2] {
        match self {
            TexPreset::Noise => [0x202020, 0xe0e0e0],
            TexPreset::Clouds => [0x5b8fd6, 0xffffff],
            TexPreset::Marble => [0x6b6f78, 0xf2f0ea],
            TexPreset::Wood => [0x5a3416, 0xb9814a],
            TexPreset::Stone => [0x4a4a4f, 0x9c9a96],
            TexPreset::Grass => [0x2f5a1e, 0x7fb04a],
            TexPreset::Sand => [0xc2a36b, 0xead39b],
            TexPreset::Dirt => [0x3b2a1c, 0x7a5a3a],
            TexPreset::Lava => [0x1a0a05, 0xffb02e],
            TexPreset::Water => [0x0d3b66, 0x5fb4d9],
            TexPreset::Bricks => [0xb7b1a5, 0x9c4a32],
            TexPreset::Tiles => [0x8f8f8f, 0xe8e4dc],
            TexPreset::Metal => [0x6c7177, 0xc9ced3],
            TexPreset::Rust => [0x5a2d14, 0xb8642c],
            TexPreset::Camo => [0x2e3b22, 0x8a8a5c],
            TexPreset::Checker => [0x404040, 0xd0d0d0],
            TexPreset::Blades => [0x2c5a1a, 0x9ccf5a],
            TexPreset::Leaves => [0x24461a, 0x6fa83c],
        }
    }
}

/// Texture recipe (builder). `scale` = feature frequency (number of repeats of the pattern).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TexGen {
    pub preset: TexPreset,
    pub size: u32,
    pub seed: u32,
    pub scale: f32,
    pub colors: Option<[u32; 2]>,
}

impl TexGen {
    pub fn new(preset: TexPreset) -> Self {
        Self {
            preset,
            size: 256,
            seed: 1,
            scale: 1.0,
            colors: None,
        }
    }
    pub fn size(mut self, s: u32) -> Self {
        self.size = s.clamp(4, 2048);
        self
    }
    pub fn seed(mut self, s: u32) -> Self {
        self.seed = s;
        self
    }
    pub fn scale(mut self, s: f32) -> Self {
        self.scale = s.max(0.05);
        self
    }
    pub fn colors(mut self, dark: u32, light: u32) -> Self {
        self.colors = Some([dark, light]);
        self
    }
    /// Height map (linear RGBA, grey = height) derived from the preset: brightness = height, so
    /// mortar, cracks and grout are low. Tiles seamlessly like the colour texture.
    pub fn generate_height(&self) -> Vec<u8> {
        let px = self.generate();
        let mut out = Vec::with_capacity(px.len());
        for p in px.as_chunks::<4>().0 {
            let l = (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) as u8;
            out.extend_from_slice(&[l, l, l, 255]);
        }
        out
    }

    /// Tangent-space normal map (linear RGBA, +Y = up in the texture / -v) from the height map,
    /// Sobel filter with wrap-around (seamless). `strength` ≈ 1–8.
    pub fn generate_normal(&self, strength: f32) -> Vec<u8> {
        let n = self.size as usize;
        let hp = self.generate_height();
        let h = |x: isize, y: isize| -> f32 {
            let (x, y) = (x.rem_euclid(n as isize) as usize, y.rem_euclid(n as isize) as usize);
            hp[(y * n + x) * 4] as f32 / 255.0
        };
        let mut out = Vec::with_capacity(n * n * 4);
        let k = strength * n as f32 / 256.0;
        for y in 0..n as isize {
            for x in 0..n as isize {
                let dx = (h(x + 1, y - 1) + 2.0 * h(x + 1, y) + h(x + 1, y + 1)) - (h(x - 1, y - 1) + 2.0 * h(x - 1, y) + h(x - 1, y + 1));
                let dy = (h(x - 1, y + 1) + 2.0 * h(x, y + 1) + h(x + 1, y + 1)) - (h(x - 1, y - 1) + 2.0 * h(x, y - 1) + h(x + 1, y - 1));
                // texture rows grow downwards (+v); tangent-space +Y points along +v (bitangent)
                let v = glam::Vec3::new(-dx * k, -dy * k, 1.0).normalize();
                let e = |c: f32| ((c * 0.5 + 0.5) * 255.0).round() as u8;
                out.extend_from_slice(&[e(v.x), e(v.y), e(v.z), 255]);
            }
        }
        out
    }

    pub fn label(&self) -> String {
        format!(
            "texgen:{}:{}:s{}:x{}",
            self.preset.name(),
            self.size,
            self.seed,
            self.scale
        )
    }

    /// RGBA8 (sRGB) pixels, `size * size * 4` bytes, row-major from the top-left.
    pub fn generate(&self) -> Vec<u8> {
        let n = self.size as usize;
        // Bands of rows per job; generators are built per band (noise::Worley is not Sync).
        let bands: Vec<usize> = (0..n.div_ceil(16)).collect();
        let out = crate::jobs::par_map_min(2, &bands, |&band| {
            let g = Gen::new(self);
            let mut px = Vec::with_capacity(16 * n * 4);
            for y in band * 16..((band + 1) * 16).min(n) {
                let v = y as f64 / n as f64;
                for x in 0..n {
                    let u = x as f64 / n as f64;
                    let c = g.sample(u, v);
                    px.extend_from_slice(&[to8(c[0]), to8(c[1]), to8(c[2]), to8(g.alpha(u, v))]);
                }
            }
            px
        });
        out.concat()
    }
}

fn to8(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn rgb(hex: u32) -> [f32; 3] {
    [
        ((hex >> 16) & 0xff) as f32 / 255.0,
        ((hex >> 8) & 0xff) as f32 / 255.0,
        (hex & 0xff) as f32 / 255.0,
    ]
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn shade(c: [f32; 3], k: f32) -> [f32; 3] {
    [c[0] * k, c[1] * k, c[2] * k]
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Point on a 4D torus: seamless in u and v. `f` = number of feature repeats per tile.
fn torus(u: f64, v: f64, f: f64) -> [f64; 4] {
    let r = f / TAU;
    [
        (u * TAU).cos() * r,
        (u * TAU).sin() * r,
        (v * TAU).cos() * r,
        (v * TAU).sin() * r,
    ]
}

struct Gen {
    p: TexPreset,
    scale: f64,
    a: [f32; 3],
    b: [f32; 3],
    fbm: Fbm<Perlin>,
    fbm2: Fbm<Perlin>,
    ridged: RidgedMulti<Perlin>,
    billow: Billow<Perlin>,
    worley: Worley,
}

impl Gen {
    fn new(t: &TexGen) -> Self {
        let [a, b] = t.colors.unwrap_or(t.preset.default_colors());
        Self {
            p: t.preset,
            scale: t.scale as f64,
            a: rgb(a),
            b: rgb(b),
            fbm: Fbm::<Perlin>::new(t.seed).set_octaves(5),
            fbm2: Fbm::<Perlin>::new(t.seed.wrapping_add(17)).set_octaves(3),
            ridged: RidgedMulti::<Perlin>::new(t.seed.wrapping_add(31)).set_octaves(4),
            billow: Billow::<Perlin>::new(t.seed.wrapping_add(53)).set_octaves(4),
            worley: Worley::new(t.seed.wrapping_add(71)),
        }
    }

    /// fBm in 0..1 at feature frequency `f` (seamless).
    fn n(&self, u: f64, v: f64, f: f64) -> f32 {
        (self.fbm.get(torus(u, v, f * self.scale)) * 0.5 + 0.5) as f32
    }
    fn n2(&self, u: f64, v: f64, f: f64) -> f32 {
        (self.fbm2.get(torus(u, v, f * self.scale)) * 0.5 + 0.5) as f32
    }

    /// Coverage 0..1 (only the cut-out presets are not fully opaque).
    fn alpha(&self, u: f64, v: f64) -> f32 {
        match self.p {
            TexPreset::Blades => self.blade(u, v).map(|_| 1.0).unwrap_or(0.0),
            TexPreset::Leaves => {
                let n = self.n(u, v, 5.0) * 0.7 + self.n2(u, v, 14.0) * 0.3;
                if n > 0.5 { 1.0 } else { 0.0 }
            }
            _ => 1.0,
        }
    }

    /// Grass blade hit: returns height fraction (0 = root, 1 = tip) when (u, v) is inside a blade.
    /// Blades wrap horizontally so the card tiles; v = 0 is the top of the texture.
    fn blade(&self, u: f64, v: f64) -> Option<f32> {
        let n = (14.0 * self.scale).round().max(3.0) as usize;
        let t = (1.0 - v) as f32; // height from the bottom
        for b in 0..n {
            let h = |k: u32| ((b as u32).wrapping_mul(2_654_435_761).wrapping_add(k).wrapping_mul(1_597_334_677) >> 8) as f32 / (1u32 << 24) as f32;
            let height = 0.55 + 0.42 * h(1);
            if t > height {
                continue;
            }
            let x0 = (b as f32 + 0.5 + (h(2) - 0.5) * 0.6) / n as f32;
            let lean = (h(3) - 0.5) * 0.25;
            let k = t / height;
            let cx = x0 + lean * k * k;
            let half = 0.55 / n as f32 * (1.0 - k).powf(0.8);
            let mut dx = (u as f32 - cx).abs();
            dx = dx.min(1.0 - dx); // wrap
            if dx < half {
                return Some(k);
            }
        }
        None
    }

    fn sample(&self, u: f64, v: f64) -> [f32; 3] {
        let (a, b) = (self.a, self.b);
        let s = self.scale;
        match self.p {
            TexPreset::Noise => mix(a, b, self.n(u, v, 4.0)),
            TexPreset::Clouds => {
                let c = self.billow.get(torus(u, v, 3.0 * s)) as f32 * 0.5 + 0.5;
                mix(a, b, smooth(0.35, 0.85, c))
            }
            TexPreset::Marble => {
                let t = self.n(u, v, 3.0);
                let veins = ((u * TAU * 2.0 * s) as f32 + t * 9.0)
                    .sin()
                    .abs()
                    .powf(0.35);
                mix(shade(a, 0.9), b, veins)
            }
            TexPreset::Wood => {
                // growth rings = bands of a seamless noise value (so the pattern still tiles)
                let w = self.n(u, v, 2.0);
                let rings = ((w * 18.0 * s as f32) % 1.0 + 1.0) % 1.0;
                let grain = self.n2(u, v * 0.15, 24.0) * 0.25;
                mix(a, b, smooth(0.15, 0.85, rings) * 0.8 + grain)
            }
            TexPreset::Stone => {
                let cells = self.worley.get(torus(u, v, 6.0 * s)) as f32 * 0.5 + 0.5;
                let base = mix(a, b, self.n(u, v, 6.0));
                let cracks = smooth(0.0, 0.12, cells);
                shade(base, 0.55 + 0.45 * cracks)
            }
            TexPreset::Grass => {
                let blades = self.n2(u, v, 40.0);
                let patches = self.n(u, v, 3.0);
                let c = mix(a, b, patches * 0.6 + blades * 0.5);
                shade(c, 0.85 + blades * 0.25)
            }
            TexPreset::Sand => {
                let ripples =
                    ((v * TAU * 10.0 * s) as f32 + self.n(u, v, 3.0) * 6.0).sin() * 0.5 + 0.5;
                let grain = self.n2(u, v, 64.0);
                shade(mix(a, b, ripples * 0.5 + 0.3), 0.85 + grain * 0.3)
            }
            TexPreset::Dirt => {
                let base = self.n(u, v, 5.0);
                let pebbles = 1.0
                    - smooth(
                        0.0,
                        0.25,
                        self.worley.get(torus(u, v, 14.0 * s)) as f32 * 0.5 + 0.5,
                    );
                mix(mix(a, b, base), shade(b, 1.2), pebbles * 0.5)
            }
            TexPreset::Lava => {
                let r = self.ridged.get(torus(u, v, 3.0 * s)) as f32 * 0.5 + 0.5;
                let hot = smooth(0.55, 0.95, r);
                mix(a, mix([0.9, 0.15, 0.02], b, hot), hot.powf(0.7))
            }
            TexPreset::Water => {
                let c = self.worley.get(torus(u, v, 5.0 * s)) as f32 * 0.5 + 0.5;
                let caustic = (1.0 - smooth(0.0, 0.18, c)) * 0.6;
                mix(a, b, self.n(u, v, 3.0) * 0.6 + caustic)
            }
            TexPreset::Bricks => {
                let rows = (8.0 * s) as f32;
                let cols = (4.0 * s) as f32;
                let y = v as f32 * rows;
                let row = y.floor();
                let x = u as f32 * cols + if (row as i32) % 2 == 0 { 0.0 } else { 0.5 };
                let (fx, fy) = (x.fract(), y.fract());
                let mortar = fx < 0.04 || fy < 0.08;
                if mortar {
                    shade(a, 0.8 + self.n2(u, v, 32.0) * 0.3)
                } else {
                    let id = (x.floor() * 7.3 + row * 3.1).sin() * 0.5 + 0.5;
                    shade(b, 0.75 + id * 0.35 + (self.n(u, v, 16.0) - 0.5) * 0.3)
                }
            }
            TexPreset::Tiles => {
                let k = (6.0 * s) as f32;
                let (fx, fy) = ((u as f32 * k).fract(), (v as f32 * k).fract());
                if fx < 0.05 || fy < 0.05 {
                    shade(a, 0.6)
                } else {
                    shade(b, 0.9 + (self.n(u, v, 12.0) - 0.5) * 0.15)
                }
            }
            TexPreset::Metal => {
                // brushed: noise stretched along u, plus faint scratches
                let streak = (self.fbm.get(torus(u, v, 2.0 * s)) * 0.5 + 0.5) as f32 * 0.3
                    + (self.fbm2.get([
                        ((u * TAU).cos() * 0.3),
                        ((u * TAU).sin() * 0.3),
                        (v * TAU).cos() * 40.0 * s / TAU,
                        (v * TAU).sin() * 40.0 * s / TAU,
                    ]) * 0.5
                        + 0.5) as f32
                        * 0.7;
                mix(a, b, streak)
            }
            TexPreset::Rust => {
                let r = self.n(u, v, 5.0);
                let spots = smooth(0.45, 0.65, self.n2(u, v, 9.0));
                mix(
                    mix([0.45, 0.47, 0.5], [0.6, 0.62, 0.65], r),
                    mix(a, b, r),
                    spots,
                )
            }
            TexPreset::Camo => {
                let n1 = self.n(u, v, 3.0);
                let n2 = self.n2(u, v, 4.0);
                let mid = mix(a, b, 0.5);
                if n1 > 0.58 {
                    shade(a, 0.8)
                } else if n2 > 0.55 {
                    b
                } else {
                    mid
                }
            }
            TexPreset::Blades => {
                // colour also outside blades (no dark fringes when mips blend alpha)
                let k = self.blade(u, v).unwrap_or(0.5);
                shade(mix(a, b, k * 0.9 + 0.1), 0.85 + self.n2(u, v, 20.0) * 0.3)
            }
            TexPreset::Leaves => {
                let veins = self.n2(u, v, 30.0);
                mix(a, b, self.n(u, v, 3.0) * 0.7 + veins * 0.3)
            }
            TexPreset::Checker => {
                let k = (8.0 * s) as f32;
                let c = ((u as f32 * k).floor() + (v as f32 * k).floor()) as i32 % 2 == 0;
                if c {
                    a
                } else {
                    b
                }
            }
        }
    }
}

/// Texture ids produced by `Ctx::generate_maps`.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TexMaps {
    pub color: Option<crate::renderer::TextureId>,
    pub normal: Option<crate::renderer::TextureId>,
    pub height: Option<crate::renderer::TextureId>,
}

/// Mean color (sRGB hex) of generated pixels — cheap AI-side sanity check.
pub fn mean_color(px: &[u8]) -> String {
    let n = (px.len() / 4).max(1) as u64;
    let mut s = [0u64; 3];
    for p in px.as_chunks::<4>().0 {
        s[0] += p[0] as u64;
        s[1] += p[1] as u64;
        s[2] += p[2] as u64;
    }
    format!("#{:02x}{:02x}{:02x}", s[0] / n, s[1] / n, s[2] / n)
}

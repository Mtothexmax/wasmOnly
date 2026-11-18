// Sky-view LUT: single scattering (Rayleigh + Mie) of sun and moon light through an Earth-like
// atmosphere, for every view direction (u = azimuth, v = elevation, see common.wgsl sky_lut_uv).
// Rendered once per view into a small rgba16float texture; the sky pass, fog, clouds and water
// reflections sample it (the costly integral never runs per screen pixel).

const R_GROUND: f32 = 6360e3;
const R_TOP: f32 = 6420e3;
const BETA_R: vec3f = vec3f(5.8e-6, 13.5e-6, 33.1e-6);
const BETA_M: f32 = 8e-6;
const H_R: f32 = 8000.0;
const H_M: f32 = 1200.0;
const MIE_G: f32 = 0.76;
const BETA_OZONE: vec3f = vec3f(1.713e-6, 4.149e-6, 0.178e-6); // follows the Rayleigh density (crude)

struct LutOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_lut(@builtin(vertex_index) i: u32) -> LutOut {
    let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var o: LutOut;
    o.pos = vec4f(xy, 0.0, 1.0);
    o.uv = vec2f(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
    return o;
}

// (near, far) distances of a ray from `o` along `d` through a sphere of radius r at the origin.
fn sphere(o: vec3f, d: vec3f, r: f32) -> vec2f {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if (disc < 0.0) { return vec2f(-1.0, -1.0); }
    let s = sqrt(disc);
    return vec2f(-b - s, -b + s);
}

fn scatter(dir: vec3f, light: vec3f) -> vec3f {
    let o = vec3f(0.0, R_GROUND + 2.0, 0.0);
    var t_max = sphere(o, dir, R_TOP).y;
    let ground = sphere(o, dir, R_GROUND);
    if (ground.x > 0.0) { t_max = ground.x; }
    let n_view = 16;
    let n_light = 6;
    let ds = t_max / f32(n_view);
    var od_r = 0.0;
    var od_m = 0.0;
    var sum_r = vec3f(0.0);
    var sum_m = vec3f(0.0);
    var sum_ms = vec3f(0.0);
    for (var i = 0; i < n_view; i++) {
        let p = o + dir * (ds * (f32(i) + 0.5));
        let h = length(p) - R_GROUND;
        let hr = exp(-h / H_R) * ds;
        let hm = exp(-h / H_M) * ds;
        od_r += hr;
        od_m += hm;
        // light path to the top of the atmosphere (blocked by the planet = in its shadow)
        if (sphere(p, light, R_GROUND).x > 0.0) { continue; }
        let tl = sphere(p, light, R_TOP).y;
        let dl = tl / f32(n_light);
        var lr = 0.0;
        var lm = 0.0;
        for (var j = 0; j < n_light; j++) {
            let q = p + light * (dl * (f32(j) + 0.5));
            let hq = max(length(q) - R_GROUND, 0.0);
            lr += exp(-hq / H_R) * dl;
            lm += exp(-hq / H_M) * dl;
        }
        let tau = (BETA_R + BETA_OZONE) * (od_r + lr) + vec3f(BETA_M * 1.1 * (od_m + lm));
        let att = exp(-tau);
        sum_r += att * hr;
        sum_m += att * hm;
        // cheap multiple scattering: second-order Rayleigh (beta^2) -> blue zenith, pale-blue horizon
        sum_ms += att * hr * BETA_R * (BETA_R * 8000.0) * 0.64;
    }
    let mu = dot(dir, light);
    let phase_r = 3.0 / (16.0 * 3.14159265) * (1.0 + mu * mu);
    let g2 = MIE_G * MIE_G;
    let phase_m = 3.0 / (8.0 * 3.14159265) * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * MIE_G * mu, 1.5));
    return sum_r * BETA_R * phase_r + sum_m * BETA_M * phase_m + sum_ms;
}

@fragment
fn fs_lut(i: LutOut) -> @location(0) vec4f {
    let az = (i.uv.x - 0.5) * 2.0 * 3.14159265;
    let t = 1.0 - 2.0 * i.uv.y;
    let el = sign(t) * t * t * 3.14159265 * 0.5;
    let dir = vec3f(cos(el) * cos(az), sin(el), cos(el) * sin(az));
    let k = g.misc.w; // atmosphere intensity (sun irradiance in scene units)
    var c = scatter(dir, g.sky_sun.xyz) * k;
    if (g.sky_moon.w > 0.0) {
        c += scatter(dir, g.sky_moon.xyz) * k * 0.004 * g.sky_moon.w;
    }
    // faint airglow so a moonless night is not pure black
    c += vec3f(0.0012, 0.0016, 0.0032) * g.sky_sun_color.w * (1.0 - 0.5 * max(dir.y, 0.0));
    return vec4f(c, 1.0);
}

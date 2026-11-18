// Shared declarations. Prepended to every engine shader (one source of truth for layouts).

const MAX_CASCADES: u32 = 4u;
const CLUSTER_X: u32 = 16u;
const CLUSTER_Y: u32 = 9u;
const CLUSTER_Z: u32 = 24u;

struct Light {
    pos_range: vec4f,       // xyz = world position, w = range
    color_intensity: vec4f, // rgb = color, a = intensity
    dir_cos: vec4f,         // xyz = direction (spot), w = cos(outer angle); -2 = point light
    extra: vec4f,           // x = cos(inner angle)
}

struct Globals {
    view_proj: mat4x4f,
    cascades: array<mat4x4f, 4>, // sun light view-projection per shadow cascade
    camera_pos: vec4f,
    sun_dir: vec4f,      // xyz = direction the light travels, w = intensity
    sun_color: vec4f,
    sky_color: vec4f,
    ground_color: vec4f,
    counts: vec4u,       // x = visible lights, y = shadows on, z = debug view, w = cascade count
    time: vec4f,         // x = seconds, y = camera near, z = camera far
    inv_view_proj: mat4x4f,
    fog: vec4f,          // x = density (0 = off), y = height falloff, z = start distance, w = max opacity
    fog_color: vec4f,    // rgb = fixed fog colour if w == 1, else the sky colour towards the horizon
    sky_zenith: vec4f,   // rgb
    sky_horizon: vec4f,  // rgb, w = sun disk angular radius (radians)
    cam_right: vec4f,    // billboards (particles)
    cam_up: vec4f,
    view: mat4x4f,
    cascade_splits: vec4f, // far view distance of each cascade
    cascade_texel: vec4f,  // world size of one shadow-map texel per cascade
    cluster: vec4f,        // x = near, y = far, z = slices / ln(far / near)
    screen: vec4f,         // w, h, 1/w, 1/h (pixels)
    sky_sun: vec4f,        // xyz = direction to the sun (sky, disk, clouds), w = sun disk brightness
    sky_sun_color: vec4f,  // rgb = sunlight after the atmosphere (clouds, disk), w = night factor 0..1
    sky_moon: vec4f,       // xyz = direction to the moon, w = moon brightness (0 = none)
    sky_params: vec4f,     // x = mode (0 gradient, 1 atmosphere), y = cloud coverage, z = cloud offset, w = stars
    skybox: vec4f,         // x = blend (0 = procedural sky, 1 = skybox), y = rotation (rad), z = intensity, w = 1 if bound
    misc: vec4f,           // x = water SSR on, y = water refraction, z = water caustics, w = atmosphere intensity
}

struct Instance {
    model: mat4x4f,
    normal: mat4x4f,
    base_color: vec4f,
    emissive_metallic: vec4f, // rgb = emissive, a = metallic
    params: vec4f,            // x = roughness, y = checker scale (0 = off), z = 0 lit / 1 unlit / 2 ui,
                              // w = alpha: 0 opaque, (0,1] mask cutoff, -1 blend
    maps: vec4f,              // x = normal-map strength (0 = none), y = displacement (water: wave height),
                              // z = parallax scale (0 = off), w = water wave scale
}

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
// Sky-view LUT (atmosphere, rendered each frame by sky_lut.wgsl), its sampler (u repeat, v clamp)
// and the optional equirectangular skybox image (1x1 dummy when none is set).
@group(0) @binding(10) var sky_lut: texture_2d<f32>;
@group(0) @binding(11) var sky_sampler: sampler;
@group(0) @binding(12) var skybox_tex: texture_2d<f32>;

const SKY_PI: f32 = 3.14159265;

fn sky_hash3(p: vec3f) -> vec3f {
    var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.xxy + q.yxx) * q.zyx);
}

fn sky_h2(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn sky_noise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(sky_h2(i), sky_h2(i + vec2f(1.0, 0.0)), u.x), mix(sky_h2(i + vec2f(0.0, 1.0)), sky_h2(i + vec2f(1.0, 1.0)), u.x), u.y);
}

// Direction -> sky-view LUT coordinates: u = azimuth, v = elevation with more texels near the horizon.
fn sky_lut_uv(dir: vec3f) -> vec2f {
    let az = atan2(dir.z, dir.x);
    let el = asin(clamp(dir.y, -1.0, 1.0));
    let t = sign(el) * sqrt(abs(el) / (SKY_PI * 0.5));
    return vec2f(az / (2.0 * SKY_PI) + 0.5, 0.5 - 0.5 * t);
}

// --- Atmosphere (sky + fog), shared by the sky pass, PBR, water and particles -----------------

// Sky colour in direction `dir` (linear HDR): zenith/horizon gradient, ground tint below the horizon,
// forward scattering around the sun. No sun disk (see sun_disk) so fog can use it as its colour.
// Mode 0: zenith/horizon gradient + sun glow. Mode 1: physically based atmosphere (Rayleigh + Mie
// single scattering of sun and moon light) from the sky-view LUT.
fn sky_radiance(dir: vec3f) -> vec3f {
    let h = dir.y;
    var c: vec3f;
    if (g.sky_params.x > 0.5) {
        c = textureSampleLevel(sky_lut, sky_sampler, sky_lut_uv(normalize(vec3f(dir.x, max(h, 0.0), dir.z))), 0.0).rgb;
    } else {
        c = mix(g.sky_horizon.rgb, g.sky_zenith.rgb, pow(clamp(h, 0.0, 1.0), 0.5));
        let mu = max(dot(dir, g.sky_sun.xyz), 0.0);
        c += g.sun_color.rgb * (pow(mu, 6.0) * 0.18 + pow(mu, 48.0) * 0.5) * clamp(g.sun_dir.w * 0.3, 0.0, 2.0);
    }
    let below = select(c * 0.55 + g.ground_color.rgb * 0.8, c * 0.85 + g.ground_color.rgb * 0.15, g.sky_params.x > 0.5);
    return mix(c, below, smoothstep(0.0, -0.2, h));
}

fn sun_disk(dir: vec3f) -> vec3f {
    let r = g.sky_horizon.w;
    let a = acos(clamp(dot(dir, g.sky_sun.xyz), -1.0, 1.0));
    let limb = 1.0 - 0.4 * smoothstep(0.0, r, a); // limb darkening
    return g.sky_sun_color.rgb * g.sky_sun.w * 12.0 * limb * (1.0 - smoothstep(r * 0.8, r, a)) * smoothstep(-0.02, 0.01, dir.y);
}

// Moon: lit by the sun (phase), faint maria, small halo.
fn moon_disk(dir: vec3f) -> vec3f {
    if (g.sky_moon.w <= 0.0) { return vec3f(0.0); }
    let m = g.sky_moon.xyz;
    let r = 0.018;
    let ca = dot(dir, m);
    let halo = vec3f(0.55, 0.65, 0.85) * g.sky_moon.w * 0.06 * pow(max(ca, 0.0), 900.0);
    if (ca < cos(r * 1.3)) { return halo; }
    let right = normalize(cross(m, select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(m.y) > 0.99)));
    let up = cross(right, m);
    let q = vec2f(dot(dir, right), dot(dir, up)) / r;
    let rr = dot(q, q);
    let n = right * q.x + up * q.y - m * sqrt(max(1.0 - rr, 0.0));
    let lit = max(dot(n, g.sky_sun.xyz), 0.0) + 0.03;
    let albedo = 0.8 - 0.35 * smoothstep(0.45, 0.75, sky_noise(q * 2.3 + 7.0)) - 0.1 * sky_noise(q * 7.0);
    return vec3f(1.0, 0.97, 0.92) * lit * albedo * g.sky_moon.w * 3.0 * (1.0 - smoothstep(0.92, 1.0, rr)) + halo;
}

fn stars(dir: vec3f) -> vec3f {
    let p = dir * 140.0;
    let cell = floor(p);
    let h = sky_hash3(cell);
    if (h.x < 0.93) { return vec3f(0.0); }
    let center = vec3f(0.5) + (sky_hash3(cell + 17.0) - 0.5) * 0.6;
    let d = length(fract(p) - center);
    let b = pow((h.x - 0.93) / 0.07, 3.0) * smoothstep(0.16, 0.02, d);
    let twinkle = 0.7 + 0.3 * sin(g.time.x * (1.5 + h.y * 3.0) + h.z * 40.0);
    return mix(vec3f(0.7, 0.8, 1.0), vec3f(1.0, 0.85, 0.65), h.z) * b * twinkle * 2.5;
}

fn cloud_fbm(p: vec2f) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < 5; i++) {
        s += a * sky_noise(q);
        q = q * 2.03 + vec2f(1.7, 9.2);
        a *= 0.5;
    }
    return s;
}

// One procedural cloud layer 1.5 km above the camera: coverage, self-shadowing towards the sun,
// silver lining, lit by sun + moon + sky; fades out towards the horizon.
fn clouds(dir: vec3f) -> vec4f {
    if (dir.y < 0.01 || g.sky_params.y <= 0.0) { return vec4f(0.0); }
    let t = 1500.0 / dir.y;
    let p = (g.camera_pos.xz + dir.xz * t) / 2200.0 + vec2f(1.0, 0.35) * g.sky_params.z;
    let cov = g.sky_params.y;
    let lo = 0.85 - cov * 0.6;
    let n = cloud_fbm(p);
    let dens = smoothstep(lo, lo + 0.3, n);
    if (dens <= 0.0) { return vec4f(0.0); }
    let ps = p + g.sky_sun.xz * 0.06;
    let shade = 1.0 - 0.55 * smoothstep(lo, lo + 0.3, cloud_fbm(ps));
    let mu = max(dot(dir, g.sky_sun.xyz), 0.0);
    let sun = g.sky_sun_color.rgb * (0.55 * shade + pow(mu, 6.0) * (1.0 - dens) * 1.5);
    let moon = vec3f(0.35, 0.42, 0.6) * g.sky_moon.w * 0.25 * shade;
    let amb = sky_radiance(vec3f(0.0, 1.0, 0.0)) * 0.9 + sky_radiance(normalize(vec3f(dir.x, 0.05, dir.z))) * 0.3;
    return vec4f(sun + moon + amb, dens * smoothstep(0.01, 0.12, dir.y) * 0.95);
}

// Everything visible in the sky in direction `dir`: atmosphere/gradient, skybox, stars, moon,
// clouds; `with_sun` adds the sun disk and stars (reflections leave the sun to the specular term
// and skip the sub-pixel stars, which would sparkle on waves).
fn sky_full(dir: vec3f, with_sun: bool) -> vec3f {
    var c = sky_radiance(dir);
    let night = g.sky_sun_color.w;
    var box_k = 0.0;
    if (g.skybox.w > 0.5 && g.skybox.x > 0.0) {
        let a = atan2(dir.z, dir.x) + g.skybox.y;
        let uv = vec2f(fract(a / (2.0 * SKY_PI) + 0.5), acos(clamp(dir.y, -1.0, 1.0)) / SKY_PI);
        let s = textureSampleLevel(skybox_tex, sky_sampler, uv, 0.0).rgb * g.skybox.z;
        box_k = clamp(g.skybox.x, 0.0, 1.0);
        c = mix(c, s, box_k);
    }
    if (dir.y > -0.02) {
        if (with_sun && g.sky_params.w > 0.0 && night > 0.0) { // no stars in reflections (sparkle noise)
            c += stars(dir) * g.sky_params.w * night * smoothstep(-0.02, 0.1, dir.y); // also over a skybox (render.stars = 0 hides them)
        }
        c += moon_disk(dir) * smoothstep(-0.02, 0.02, dir.y);
        if (with_sun) { c += sun_disk(dir); }
        let cl = clouds(dir);
        c = mix(c, cl.rgb, cl.a);
    }
    return c;
}

// Exponential height fog along the view ray from the camera to `world`.
fn fog_amount(world: vec3f) -> f32 {
    let density = g.fog.x;
    if (density <= 0.0) { return 0.0; }
    let d = world - g.camera_pos.xyz;
    let dist = length(d);
    let travel = max(dist - g.fog.z, 0.0);
    let k = g.fog.y;
    var optical = density * travel;
    if (k > 0.0) {
        // integral of density * exp(-k * y) over the fogged part of the ray
        let y0 = g.camera_pos.y + d.y * (g.fog.z / max(dist, 1e-4));
        let fy = k * d.y * (travel / max(dist, 1e-4));
        let shape = select((1.0 - exp(-fy)) / fy, 1.0, abs(fy) < 1e-3);
        optical = density * exp(-k * y0) * travel * shape;
    }
    return min(1.0 - exp(-max(optical, 0.0)), g.fog.w);
}

fn fog_color(world: vec3f) -> vec3f {
    if (g.fog_color.w > 0.5) { return g.fog_color.rgb; }
    let d = normalize(world - g.camera_pos.xyz);
    return sky_radiance(normalize(vec3f(d.x, max(d.y, 0.0) * 0.5 + 0.02, d.z)));
}

fn apply_fog(color: vec3f, world: vec3f) -> vec3f {
    let f = fog_amount(world);
    if (f <= 0.0) { return color; }
    return mix(color, fog_color(world), f);
}

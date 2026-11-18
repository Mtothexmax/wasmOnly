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

// --- Atmosphere (sky + fog), shared by the sky pass, PBR, water and particles -----------------

// Sky colour in direction `dir` (linear HDR): zenith/horizon gradient, ground tint below the horizon,
// forward scattering around the sun. No sun disk (see sun_disk) so fog can use it as its colour.
fn sky_radiance(dir: vec3f) -> vec3f {
    let h = dir.y;
    var c = mix(g.sky_horizon.rgb, g.sky_zenith.rgb, pow(clamp(h, 0.0, 1.0), 0.5));
    let below = g.sky_horizon.rgb * 0.55 + g.ground_color.rgb * 0.8;
    c = mix(c, below, smoothstep(0.0, -0.2, h));
    let l = normalize(-g.sun_dir.xyz);
    let mu = max(dot(dir, l), 0.0);
    c += g.sun_color.rgb * (pow(mu, 6.0) * 0.18 + pow(mu, 48.0) * 0.5) * clamp(g.sun_dir.w * 0.3, 0.0, 2.0);
    return c;
}

fn sun_disk(dir: vec3f) -> vec3f {
    let l = normalize(-g.sun_dir.xyz);
    let r = g.sky_horizon.w;
    let a = acos(clamp(dot(dir, l), -1.0, 1.0));
    return g.sun_color.rgb * g.sun_dir.w * 12.0 * (1.0 - smoothstep(r * 0.8, r, a));
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

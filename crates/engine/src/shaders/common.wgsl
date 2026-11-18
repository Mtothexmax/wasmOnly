// Shared declarations. Prepended to every engine shader (one source of truth for layouts).

const MAX_POINT_LIGHTS: u32 = 16u;

struct PointLight {
    pos_range: vec4f,       // xyz = world position, w = range
    color_intensity: vec4f, // rgb = color, a = intensity
}

struct Globals {
    view_proj: mat4x4f,
    light_view_proj: mat4x4f,
    camera_pos: vec4f,
    sun_dir: vec4f,      // xyz = direction the light travels, w = intensity
    sun_color: vec4f,
    sky_color: vec4f,
    ground_color: vec4f,
    counts: vec4u,       // x = point light count, y = shadows on, z = debug view
    point_lights: array<PointLight, MAX_POINT_LIGHTS>,
    time: vec4f,         // x = seconds, y = camera near, z = camera far
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

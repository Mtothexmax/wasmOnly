// Particles: camera-facing quads (6 vertices per instance, no vertex buffer), drawn in the blend
// pass. Soft against the opaque scene (fade where the quad cuts into geometry), fogged.
// Additive pipeline: blend One/One (glow, feeds bloom); alpha pipeline: premultiplied over.

struct Particle {
    pos_size: vec4f,  // xyz = world position, w = size (m)
    color: vec4f,     // linear rgb (HDR allowed), a = opacity
    params: vec4f,    // x = rotation (rad), y = shape (0 soft dot, 1 square, 2 spark), z = stretch, w = unused
    vel: vec4f,       // xyz = velocity (spark stretch direction)
}

@group(0) @binding(5) var scene_depth: texture_depth_multisampled_2d;
@group(1) @binding(0) var<storage, read> particles: array<Particle>;

struct POut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
    @location(2) world: vec3f,
    @location(3) @interpolate(flat) shape: f32,
}

@vertex
fn vs_particle(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> POut {
    // corners (-1,-1) (1,-1) (1,1) (-1,-1) (1,1) (-1,1) without an array (PowerVR drops draws that index one)
    let c = vec2f(select(-1.0, 1.0, vi == 1u || vi == 2u || vi == 4u), select(-1.0, 1.0, vi == 2u || vi == 4u || vi == 5u));
    let p = particles[ii];
    let r = p.params.x;
    let rc = vec2f(c.x * cos(r) - c.y * sin(r), c.x * sin(r) + c.y * cos(r));
    var right = g.cam_right.xyz;
    var up = g.cam_up.xyz;
    var half = vec2f(p.pos_size.w * 0.5);
    if (p.params.y > 1.5) {
        // spark: long axis along the screen-projected velocity
        let v = p.vel.xyz;
        let vs = vec2f(dot(v, right), dot(v, up));
        if (length(vs) > 1e-4) {
            let d = normalize(vs);
            let r2 = right * d.x + up * d.y;
            up = r2;
            right = normalize(cross(r2, normalize(g.camera_pos.xyz - p.pos_size.xyz)));
            half = vec2f(p.pos_size.w * 0.25, p.pos_size.w * 0.5 * (1.0 + p.params.z * length(vs)));
        }
    }
    let world = p.pos_size.xyz + right * rc.x * half.x + up * rc.y * half.y;
    var o: POut;
    o.clip = g.view_proj * vec4f(world, 1.0);
    o.uv = c;
    o.color = p.color;
    o.world = world;
    o.shape = p.params.y;
    return o;
}

fn lin_depth(z: f32) -> f32 {
    let n = g.time.y;
    let f = g.time.z;
    return n * f / (f - z * (f - n));
}

fn particle_alpha(i: POut) -> f32 {
    var a: f32;
    let r2 = dot(i.uv, i.uv);
    if (i.shape > 0.5 && i.shape < 1.5) {
        a = 1.0 - smoothstep(0.8, 1.0, max(abs(i.uv.x), abs(i.uv.y)));
    } else {
        a = pow(max(1.0 - r2, 0.0), 1.5);
    }
    // soft particles: fade out where the quad gets close to the scene behind it
    let scene = lin_depth(textureLoad(scene_depth, vec2i(i.clip.xy), 0));
    let me = lin_depth(i.clip.z);
    a *= clamp((scene - me) / 0.4, 0.0, 1.0);
    return a * i.color.a;
}

@fragment
fn fs_additive(i: POut) -> @location(0) vec4f {
    let a = particle_alpha(i);
    let f = fog_amount(i.world);
    return ldr_additive(vec4f(i.color.rgb * a * (1.0 - f), 0.0));
}

@fragment
fn fs_alpha(i: POut) -> @location(0) vec4f {
    let a = particle_alpha(i);
    return ldr_blended(vec4f(apply_fog(i.color.rgb, i.world), a), i.clip.xy);
}

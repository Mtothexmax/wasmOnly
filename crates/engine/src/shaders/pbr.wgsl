// Forward PBR (GGX / Smith / Schlick), one shadowed sun + up to 16 point lights, hemispheric ambient.

@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;
@group(1) @binding(0) var base_tex: texture_2d<f32>;
@group(1) @binding(1) var base_sampler: sampler;

struct VsIn {
    @location(0) pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @builtin(instance_index) inst: u32,
}

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) @interpolate(flat) inst: u32,
}

// Instance indirection: the main pass draws visible_ids[first_instance + k] (identity on the CPU
// path, compacted by cull.wgsl on the GPU path).
@group(0) @binding(4) var<storage, read> visible_ids: array<u32>;

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let idx = visible_ids[v.inst];
    let inst = instances[idx];
    let world = inst.model * vec4f(v.pos, 1.0);
    var o: VsOut;
    o.clip = g.view_proj * world;
    o.world = world.xyz;
    o.normal = (inst.normal * vec4f(v.normal, 0.0)).xyz;
    o.uv = v.uv;
    o.inst = idx;
    return o;
}

const PI: f32 = 3.14159265;

fn d_ggx(n_h: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = n_h * n_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn v_smith(n_v: f32, n_l: f32, a: f32) -> f32 {
    let k = a * 0.5;
    let gv = n_v / (n_v * (1.0 - k) + k);
    let gl = n_l / (n_l * (1.0 - k) + k);
    return gv * gl / max(4.0 * n_v * n_l, 1e-4);
}

fn f_schlick(f0: vec3f, v_h: f32) -> vec3f {
    return f0 + (vec3f(1.0) - f0) * pow(1.0 - v_h, 5.0);
}

fn brdf(n: vec3f, v: vec3f, l: vec3f, albedo: vec3f, metallic: f32, rough: f32) -> vec3f {
    let h = normalize(v + l);
    let n_l = max(dot(n, l), 0.0);
    let n_v = max(dot(n, v), 1e-4);
    let n_h = max(dot(n, h), 0.0);
    let v_h = max(dot(v, h), 0.0);
    let a = rough * rough;
    let f0 = mix(vec3f(0.04), albedo, metallic);
    let f = f_schlick(f0, v_h);
    let spec = d_ggx(n_h, a) * v_smith(n_v, n_l, a) * f;
    let kd = (vec3f(1.0) - f) * (1.0 - metallic);
    return (kd * albedo / PI + spec) * n_l;
}

fn shadow_factor(world: vec3f, n: vec3f) -> f32 {
    let biased = world + n * 0.04;
    let lp = g.light_view_proj * vec4f(biased, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = ndc.xy * vec2f(0.5, -0.5) + vec2f(0.5);
    let dims = vec2f(textureDimensions(shadow_map));
    let texel = 1.0 / dims;
    var sum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2f(f32(x), f32(y)) * texel, ndc.z - 0.0015);
        }
    }
    let inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)) && ndc.z <= 1.0;
    return select(1.0, sum / 9.0, inside);
}

fn aces(x: vec3f) -> vec3f {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs_main(i: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let inst = instances[i.inst];
    // Geometric face normal from screen-space derivatives (computed first, in uniform control flow).
    let face_n = normalize(cross(dpdx(i.world), dpdy(i.world)));
    let tex = textureSample(base_tex, base_sampler, i.uv);
    let alpha_mode = inst.params.w;
    var alpha = inst.base_color.a * tex.a;
    if (alpha_mode > 0.0) {            // mask (alpha test)
        if (alpha < alpha_mode) { discard; }
        alpha = 1.0;
    } else if (alpha_mode == 0.0) {    // opaque
        alpha = 1.0;
    }
    var albedo = tex.rgb * inst.base_color.rgb;

    // Procedural checker (useful for ground planes without textures)
    let checker = inst.params.y;
    if (checker > 0.0) {
        let c = floor(i.world.x * checker) + floor(i.world.z * checker);
        albedo *= select(1.0, 0.78, (i32(c) & 1) == 0);
    }

    // Double-sided back faces: mirror the shading normal at the triangle plane. For a flat pane this
    // equals flipping it; for foliage cards with up-tilted normals the up component is kept, so the
    // card stays lit from both sides and at every distance (flipping turned far grass black).
    let nn = normalize(i.normal);
    let n = select(nn, normalize(nn - 2.0 * dot(nn, face_n) * face_n), !front);
    let shadow = select(1.0, shadow_factor(i.world, n), g.counts.y == 1u);

    if (inst.params.z > 1.5) { // ui surface (world panels): texture as-is, alpha-tested
        if (tex.a < 0.5) { discard; }
        return vec4f(pow(albedo, vec3f(1.0 / 2.2)), 1.0);
    }
    if (inst.params.z > 0.5) { // unlit
        return vec4f(pow(aces(albedo + inst.emissive_metallic.rgb), vec3f(1.0 / 2.2)), alpha);
    }

    let v = normalize(g.camera_pos.xyz - i.world);
    let metallic = inst.emissive_metallic.a;
    let rough = clamp(inst.params.x, 0.04, 1.0);

    var color = vec3f(0.0);
    let l_sun = normalize(-g.sun_dir.xyz);
    color += brdf(n, v, l_sun, albedo, metallic, rough) * g.sun_color.rgb * g.sun_dir.w * shadow;

    for (var k = 0u; k < min(g.counts.x, MAX_POINT_LIGHTS); k++) {
        let pl = g.point_lights[k];
        let to_l = pl.pos_range.xyz - i.world;
        let dist = length(to_l);
        let range = pl.pos_range.w;
        let falloff = pow(clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0), 2.0) / (dist * dist + 1.0);
        color += brdf(n, v, to_l / max(dist, 1e-4), albedo, metallic, rough) * pl.color_intensity.rgb * pl.color_intensity.a * falloff;
    }

    let hemi = mix(g.ground_color.rgb, g.sky_color.rgb, n.y * 0.5 + 0.5);
    color += hemi * albedo * (1.0 - metallic * 0.6);
    color += inst.emissive_metallic.rgb;

    if (g.counts.z == 1u) { return vec4f(n * 0.5 + 0.5, 1.0); }       // debug: normals
    if (g.counts.z == 2u) { return vec4f(vec3f(shadow), 1.0); }        // debug: shadow term

    return vec4f(pow(aces(color), vec3f(1.0 / 2.2)), alpha);
}

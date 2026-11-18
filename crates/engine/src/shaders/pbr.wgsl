// Forward PBR (GGX / Smith / Schlick), one shadowed sun + up to 16 point lights, hemispheric ambient.
// Material maps: base colour, tangent-space normal map, height map (vertex displacement and/or
// parallax occlusion mapping). Tangent frames come from screen-space derivatives (no vertex
// tangents needed, works for every mesh). Water is its own shading mode (params.z == 3), drawn in
// the blend pass where it reads the opaque scene depth for depth colour and shore foam.

@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;
@group(1) @binding(0) var base_tex: texture_2d<f32>;
@group(1) @binding(1) var base_sampler: sampler;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var height_tex: texture_2d<f32>;

// Pipeline constant: false for the opaque pipelines, so they contain no `discard` and the GPU can
// reject hidden fragments before shading (early-Z). Mask (alpha-test) and UI materials use the
// pipelines with ALPHA_TEST = true.
override ALPHA_TEST: bool = true;

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

const PI: f32 = 3.14159265;

// --- Water waves (shared by vertex displacement and the fragment normal) ----------------------
// Sum of directional sine waves in world XZ. Returns (height, d/dx, d/dz). `fp` = world size of one
// pixel: waves shorter than ~2 pixels fade out (no moiré on distant water; 0 in the vertex shader).
fn waves(p: vec2f, t: f32, amp: f32, scale: f32, fp: f32) -> vec3f {
    var h = 0.0;
    var d = vec2f(0.0);
    let dirs = array<vec2f, 4>(vec2f(0.8, 0.6), vec2f(-0.5, 0.86), vec2f(0.2, -0.98), vec2f(-0.94, -0.33));
    let freq = array<f32, 4>(1.0, 1.7, 2.9, 4.3);
    let speed = array<f32, 4>(0.9, 1.3, 1.8, 2.4);
    let weight = array<f32, 4>(1.0, 0.55, 0.3, 0.18);
    for (var k = 0; k < 4; k++) {
        let f = freq[k] * scale;
        let ph = dot(dirs[k], p) * f + t * speed[k];
        let a = amp * weight[k] * (1.0 - smoothstep(0.6, 2.2, f * fp));
        h += a * sin(ph);
        d += dirs[k] * (a * f * cos(ph));
    }
    return vec3f(h, d.x, d.y);
}

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let idx = visible_ids[v.inst];
    let inst = instances[idx];
    var local = v.pos;
    // Vertex displacement along the normal from the height map (needs a tessellated mesh).
    if (inst.maps.y != 0.0 && inst.params.z < 2.5) {
        let h = textureSampleLevel(height_tex, base_sampler, v.uv, 0.0).r;
        local += normalize(v.normal) * (h - 0.5) * inst.maps.y;
    }
    var world = inst.model * vec4f(local, 1.0);
    var normal = (inst.normal * vec4f(v.normal, 0.0)).xyz;
    if (inst.params.z > 2.5) { // water: animated waves in world space
        let w = waves(world.xz, g.time.x, inst.maps.y, inst.maps.w, 0.0);
        world = vec4f(world.x, world.y + w.x, world.z, 1.0);
        normal = vec3f(-w.y, 1.0, -w.z);
    }
    var o: VsOut;
    o.clip = g.view_proj * world;
    o.world = world.xyz;
    o.normal = normal;
    o.uv = v.uv;
    o.inst = idx;
    return o;
}

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

fn tonemap(c: vec3f) -> vec3f {
    return pow(aces(c), vec3f(1.0 / 2.2));
}

// Derivatives of everything that needs them, taken once in uniform control flow.
struct Deriv {
    dpx: vec3f,
    dpy: vec3f,
    duvx: vec2f,
    duvy: vec2f,
}

fn derivs(i: VsOut) -> Deriv {
    return Deriv(dpdx(i.world), dpdy(i.world), dpdx(i.uv), dpdy(i.uv));
}

// Cotangent frame (Schüler): tangent/bitangent along +u/+v from screen-space derivatives.
fn cotangent_frame(n: vec3f, d: Deriv) -> mat3x3f {
    let dp2perp = cross(d.dpy, n);
    let dp1perp = cross(n, d.dpx);
    let t = dp2perp * d.duvx.x + dp1perp * d.duvy.x;
    let b = dp2perp * d.duvx.y + dp1perp * d.duvy.y;
    let invmax = inverseSqrt(max(max(dot(t, t), dot(b, b)), 1e-12));
    return mat3x3f(t * invmax, b * invmax, n);
}

// Parallax occlusion mapping: march the height field along the view ray in tangent space.
fn parallax_uv(uv: vec2f, view_ts: vec3f, scale: f32, d: Deriv) -> vec2f {
    let layers = mix(24.0, 8.0, clamp(abs(view_ts.z), 0.0, 1.0));
    let step = 1.0 / layers;
    let delta = (view_ts.xy / max(abs(view_ts.z), 0.15)) * scale * step;
    var cur_uv = uv;
    var cur_depth = 0.0;
    var map_depth = 1.0 - textureSampleGrad(height_tex, base_sampler, cur_uv, d.duvx, d.duvy).r;
    var prev_uv = uv;
    var prev_diff = 0.0;
    for (var k = 0; k < 32; k++) {
        if (cur_depth >= map_depth || f32(k) >= layers) { break; }
        prev_uv = cur_uv;
        prev_diff = map_depth - cur_depth;
        cur_uv -= delta;
        cur_depth += step;
        map_depth = 1.0 - textureSampleGrad(height_tex, base_sampler, cur_uv, d.duvx, d.duvy).r;
    }
    let after = map_depth - cur_depth;
    let w = after / min(after - prev_diff, -1e-4);
    return mix(cur_uv, prev_uv, clamp(w, 0.0, 1.0));
}

struct Surface {
    color: vec4f,  // final tonemapped colour + alpha
    n: vec3f,      // shading normal (for debug views)
    shadow: f32,
}

fn shade(i: VsOut, front: bool, d: Deriv, face_n: vec3f) -> Surface {
    let inst = instances[i.inst];
    // Double-sided back faces: mirror the shading normal at the triangle plane (equals flipping for
    // a flat pane; keeps the up tilt of foliage normals, which flipping turned black).
    let nn = normalize(i.normal);
    var n = select(nn, normalize(nn - 2.0 * dot(nn, face_n) * face_n), !front);
    let v = normalize(g.camera_pos.xyz - i.world);
    let tbn = cotangent_frame(n, d);

    var uv = i.uv;
    if (inst.maps.z > 0.0) { // parallax occlusion mapping
        let view_ts = vec3f(dot(v, tbn[0]), dot(v, tbn[1]), dot(v, tbn[2]));
        uv = parallax_uv(uv, view_ts, inst.maps.z, d);
    }
    let tex = textureSampleGrad(base_tex, base_sampler, uv, d.duvx, d.duvy);
    if (inst.maps.x != 0.0) { // tangent-space normal map
        var tn = textureSampleGrad(normal_tex, base_sampler, uv, d.duvx, d.duvy).xyz * 2.0 - 1.0;
        tn = vec3f(tn.xy * inst.maps.x, tn.z);
        n = normalize(tbn * tn);
    }

    let alpha_mode = inst.params.w;
    var alpha = inst.base_color.a * tex.a;
    if (alpha_mode > 0.0) {            // mask (alpha test)
        if (ALPHA_TEST && alpha < alpha_mode) { discard; }
        alpha = 1.0;
    } else if (alpha_mode == 0.0) {    // opaque
        alpha = 1.0;
    }
    var albedo = tex.rgb * inst.base_color.rgb;
    let checker = inst.params.y;
    if (checker > 0.0) {
        let c = floor(i.world.x * checker) + floor(i.world.z * checker);
        albedo *= select(1.0, 0.78, (i32(c) & 1) == 0);
    }
    var shadow = 1.0;
    if (g.counts.y == 1u) { shadow = shadow_factor(i.world, n); }

    if (inst.params.z > 1.5) { // ui surface (world panels): texture as-is, alpha-tested
        if (ALPHA_TEST && tex.a < 0.5) { discard; }
        return Surface(vec4f(pow(albedo, vec3f(1.0 / 2.2)), 1.0), n, shadow);
    }
    if (inst.params.z > 0.5) { // unlit
        return Surface(vec4f(tonemap(albedo + inst.emissive_metallic.rgb), alpha), n, shadow);
    }

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
    return Surface(vec4f(tonemap(color), alpha), n, shadow);
}

fn debug_out(s: Surface) -> vec4f {
    if (g.counts.z == 1u) { return vec4f(s.n * 0.5 + 0.5, 1.0); }  // normals (incl. normal maps)
    if (g.counts.z == 2u) { return vec4f(vec3f(s.shadow), 1.0); }   // shadow term
    return s.color;
}

// Opaque + alpha-tested geometry.
@fragment
fn fs_main(i: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let d = derivs(i);
    let face_n = normalize(cross(d.dpx, d.dpy));
    return debug_out(shade(i, front, d, face_n));
}

// --- Blend pass: the opaque depth buffer is readable here -------------------------------------
@group(0) @binding(5) var scene_depth: texture_depth_multisampled_2d;

fn linear_depth(z: f32) -> f32 {
    let n = g.time.y;
    let f = g.time.z;
    return n * f / (f - z * (f - n));
}

fn hash2(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn vnoise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(i), hash2(i + vec2f(1.0, 0.0)), u.x), mix(hash2(i + vec2f(0.0, 1.0)), hash2(i + vec2f(1.0, 1.0)), u.x), u.y);
}

// Water: wave normal + fine ripples, Fresnel sky reflection, sun glint, depth-tinted body colour
// (shallow -> deep by the distance to the ground below), soft edges and foam at the shore.
fn water(i: VsOut, d: Deriv) -> vec4f {
    let inst = instances[i.inst];
    let t = g.time.x;
    let fp = max(length(d.dpx), length(d.dpy));
    let wv = waves(i.world.xz, t, inst.maps.y, inst.maps.w, fp);
    // fine ripples: gradient of two scrolling value-noise layers (organic, no regular stripes),
    // faded out where a ripple would be smaller than ~2 pixels
    var r = vec2f(0.0);
    let scales = array<f32, 2>(3.5, 9.0);
    let drift = array<vec2f, 2>(vec2f(0.35, 0.2), vec2f(-0.25, 0.4));
    for (var k = 0; k < 2; k++) {
        let f = scales[k] * inst.maps.w;
        let q = i.world.xz * f + drift[k] * t * f * 0.6;
        let e = 0.05;
        let h0 = vnoise(q);
        let gx = (vnoise(q + vec2f(e, 0.0)) - h0) / e;
        let gz = (vnoise(q + vec2f(0.0, e)) - h0) / e;
        let fade = 1.0 - smoothstep(0.6, 2.2, f * fp * 6.28);
        r += vec2f(gx, gz) * f * 0.035 * fade;
    }
    let n = normalize(vec3f(-wv.y - r.x, 1.0, -wv.z - r.y));
    let v = normalize(g.camera_pos.xyz - i.world);
    let n_v = max(dot(n, v), 0.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - n_v, 5.0);

    // scene depth below this pixel -> water thickness along the view ray
    let px = vec2i(i.clip.xy);
    let scene_z = textureLoad(scene_depth, px, 0);
    let thickness = max(linear_depth(scene_z) - linear_depth(i.clip.z), 0.0);
    let depth_t = 1.0 - exp(-thickness * 2.2);

    let deep = inst.base_color.rgb;
    let shallow = inst.emissive_metallic.rgb;
    let l_sun = normalize(-g.sun_dir.xyz);
    var shadow = 1.0;
    if (g.counts.y == 1u) { shadow = shadow_factor(i.world, vec3f(0.0, 1.0, 0.0)); }
    let lit = g.sun_color.rgb * g.sun_dir.w * (0.25 + 0.35 * max(dot(n, l_sun), 0.0)) * shadow + g.sky_color.rgb * 0.6;
    let body = mix(shallow, deep, depth_t) * lit;

    let refl = reflect(-v, n);
    let sky = mix(g.sky_color.rgb * 2.2, g.sky_color.rgb * 1.2 + vec3f(0.05, 0.08, 0.12), clamp(refl.y, 0.0, 1.0));
    let h = normalize(l_sun + v);
    let glint = pow(max(dot(n, h), 0.0), 600.0) * 6.0 * g.sun_dir.w * shadow;

    let foam_noise = vnoise(i.world.xz * 6.0 + vec2f(t * 0.4, -t * 0.3));
    let foam = smoothstep(0.28, 0.0, thickness) * smoothstep(0.35, 0.75, foam_noise + 0.25);

    var color = mix(body, sky, fresnel) + g.sun_color.rgb * glint;
    color = mix(color, vec3f(0.95, 0.97, 1.0) * (0.6 + 0.4 * shadow), foam);
    let edge = smoothstep(0.0, 0.12, thickness); // soft intersection with the shore
    let alpha = clamp((mix(0.45, 0.94, depth_t) + fresnel * 0.4 + foam) * edge, 0.0, 1.0) * inst.base_color.a;
    if (g.counts.z == 1u) { return vec4f(n * 0.5 + 0.5, 1.0); }
    return vec4f(tonemap(color), alpha);
}

@fragment
fn fs_blend(i: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let d = derivs(i);
    let face_n = normalize(cross(d.dpx, d.dpy));
    if (instances[i.inst].params.z > 2.5) {
        return water(i, d);
    }
    return debug_out(shade(i, front, d, face_n));
}

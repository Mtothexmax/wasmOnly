// Forward PBR (GGX / Smith / Schlick): sun with cascaded shadow maps, clustered point + spot lights,
// hemispheric ambient, terrain splat shading, GPU grass entry points (vs_grass / fs_grass).
// Material maps: base colour, tangent-space normal map, height map (vertex displacement and/or
// parallax occlusion mapping). Tangent frames come from screen-space derivatives (no vertex
// tangents needed, works for every mesh). Water is its own shading mode (params.z == 3), drawn in
// the blend pass where it reads the opaque scene depth for depth colour and shore foam.

@group(0) @binding(2) var shadow_map: texture_depth_2d_array;
@group(0) @binding(6) var<storage, read> lights: array<Light>;
@group(0) @binding(7) var<storage, read> clusters: array<vec2u>;
@group(0) @binding(8) var<storage, read> light_indices: array<u32>;
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
    if (inst.maps.y != 0.0 && inst.params.z < 2.5 && inst.params.z >= 0.0) {
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

fn cascade_shadow(world: vec3f, n: vec3f, k: u32) -> f32 {
    // normal offset scaled by the cascade's texel size (no acne on near cascades, no peter-panning far)
    let biased = world + n * g.cascade_texel[k] * 1.5;
    let lp = g.cascades[k] * vec4f(biased, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = ndc.xy * vec2f(0.5, -0.5) + vec2f(0.5);
    let texel = 1.0 / vec2f(textureDimensions(shadow_map));
    var sum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2f(f32(x), f32(y)) * texel, i32(k), ndc.z - 0.0008);
        }
    }
    let inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)) && ndc.z <= 1.0;
    return select(1.0, sum / 9.0, inside);
}

// Cascaded shadow maps: cascade by view depth, blended over the last 15% of each cascade, faded
// out at the shadow distance.
fn shadow_factor(world: vec3f, n: vec3f) -> f32 {
    let vz = -(g.view * vec4f(world, 1.0)).z;
    let count = g.counts.w;
    var k = 0u;
    loop {
        if (k >= count || vz <= g.cascade_splits[k]) { break; }
        k++;
    }
    if (k >= count) { return 1.0; }
    var s = cascade_shadow(world, n, k);
    let start = select(0.0, g.cascade_splits[max(k, 1u) - 1u], k > 0u);
    let end = g.cascade_splits[k];
    let band = mix(start, end, 0.85);
    let t = clamp((vz - band) / max(end - band, 1e-3), 0.0, 1.0);
    if (t > 0.0) {
        var next = 1.0;
        if (k + 1u < count) { next = cascade_shadow(world, n, k + 1u); }
        s = mix(s, next, t);
    }
    return s;
}

// Clustered point + spot lights: only the lights assigned to this pixel's froxel.
fn cluster_lights(i_clip: vec4f, world: vec3f, n: vec3f, v: vec3f, albedo: vec3f, metallic: f32, rough: f32) -> vec3f {
    let vz = -(g.view * vec4f(world, 1.0)).z;
    let cx = u32(clamp(i_clip.x * g.screen.z * f32(CLUSTER_X), 0.0, f32(CLUSTER_X - 1u)));
    let cy = u32(clamp(i_clip.y * g.screen.w * f32(CLUSTER_Y), 0.0, f32(CLUSTER_Y - 1u)));
    let cz = u32(clamp(floor(log(max(vz, g.cluster.x) / g.cluster.x) * g.cluster.z), 0.0, f32(CLUSTER_Z - 1u)));
    let cl = clusters[cx + cy * CLUSTER_X + cz * CLUSTER_X * CLUSTER_Y];
    var color = vec3f(0.0);
    for (var k = 0u; k < cl.y; k++) {
        let L = lights[light_indices[cl.x + k]];
        let to_l = L.pos_range.xyz - world;
        let dist = length(to_l);
        let range = L.pos_range.w;
        if (dist >= range) { continue; }
        let l = to_l / max(dist, 1e-4);
        var falloff = pow(clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0), 2.0) / (dist * dist + 1.0);
        if (L.dir_cos.w > -1.5) { // spot
            falloff *= smoothstep(L.dir_cos.w, L.extra.x, dot(-l, L.dir_cos.xyz));
        }
        color += brdf(n, v, l, albedo, metallic, rough) * L.color_intensity.rgb * L.color_intensity.a * falloff;
    }
    return color;
}

// Output is linear HDR into the scene target; the post pass tonemaps (alpha = "tonemap me" mask:
// 1 for scene colour, 0 for UI panels and debug views, which are shown as they are).
fn finish(c: vec3f, world: vec3f) -> vec3f {
    return apply_fog(c, world);
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

// Terrain: 4 layers from a 2x2 atlas (base_tex), weights from the splat map (normal_tex slot),
// UVs from world XZ (instance: emissive_metallic = local corner xz + 1/size, maps = metres per tile).
fn terrain_albedo(i: VsOut, d: Deriv, inst: Instance) -> vec3f {
    let inv = inst.emissive_metallic.z;
    let suv = (i.world.xz - inst.emissive_metallic.xy) * inv;
    var w = textureSampleGrad(normal_tex, base_sampler, clamp(suv, vec2f(0.0), vec2f(1.0)), d.dpx.xz * inv, d.dpy.xz * inv);
    w = w / max(w.x + w.y + w.z + w.w, 1e-3);
    let pad = 1.0 / 16.0;
    let scale = 0.5 * (1.0 - 2.0 * pad);
    var c = vec3f(0.0);
    for (var k = 0; k < 4; k++) {
        let tile = max(inst.maps[k], 0.01);
        let q = vec2f(f32(k % 2), f32(k / 2)) * 0.5 + vec2f(0.5 * pad);
        let uv = q + fract(i.world.xz / tile) * scale;
        c += textureSampleGrad(base_tex, base_sampler, uv, d.dpx.xz / tile * scale, d.dpy.xz / tile * scale).rgb * w[k];
    }
    // large-scale variation against visible tiling
    return c * (0.82 + 0.36 * vnoise(i.world.xz * 0.035));
}

struct Surface {
    color: vec4f,  // linear HDR colour (fogged) + alpha
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

    let is_terrain = inst.params.z < -0.5;
    var uv = i.uv;
    if (inst.maps.z > 0.0 && !is_terrain) { // parallax occlusion mapping
        let view_ts = vec3f(dot(v, tbn[0]), dot(v, tbn[1]), dot(v, tbn[2]));
        uv = parallax_uv(uv, view_ts, inst.maps.z, d);
    }
    var tex = textureSampleGrad(base_tex, base_sampler, uv, d.duvx, d.duvy);
    if (is_terrain) {
        tex = vec4f(terrain_albedo(i, d, inst), 1.0);
    }
    if (inst.maps.x != 0.0 && !is_terrain) { // tangent-space normal map
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
        return Surface(vec4f(albedo, 0.0), n, shadow); // alpha 0: not tonemapped
    }
    if (inst.params.z > 0.5) { // unlit
        return Surface(vec4f(finish(albedo + inst.emissive_metallic.rgb, i.world), alpha), n, shadow);
    }

    let metallic = select(inst.emissive_metallic.a, 0.0, is_terrain);
    let rough = clamp(inst.params.x, 0.04, 1.0);
    var color = vec3f(0.0);
    let l_sun = normalize(-g.sun_dir.xyz);
    color += brdf(n, v, l_sun, albedo, metallic, rough) * g.sun_color.rgb * g.sun_dir.w * shadow;
    color += cluster_lights(i.clip, i.world, n, v, albedo, metallic, rough);
    let hemi = mix(g.ground_color.rgb, g.sky_color.rgb, n.y * 0.5 + 0.5);
    color += hemi * albedo * (1.0 - metallic * 0.6);
    if (!is_terrain) { color += inst.emissive_metallic.rgb; }
    return Surface(vec4f(finish(color, i.world), alpha), n, shadow);
}

fn debug_out(s: Surface) -> vec4f {
    if (g.counts.z == 1u) { return vec4f(pow(s.n * 0.5 + 0.5, vec3f(2.2)), 0.0); }  // normals (incl. normal maps)
    if (g.counts.z == 2u) { return vec4f(vec3f(s.shadow), 0.0); }   // shadow term
    return s.color;
}

// debug_view 3: shadow cascade (red, green, blue, yellow; grey = beyond), 4: lights per cluster
fn debug_extra(i: VsOut) -> vec4f {
    if (g.counts.z == 3u) {
        let vz = -(g.view * vec4f(i.world, 1.0)).z;
        var k = 0u;
        loop {
            if (k >= g.counts.w || vz <= g.cascade_splits[k]) { break; }
            k++;
        }
        let cols = array<vec3f, 5>(vec3f(1.0, 0.25, 0.2), vec3f(0.25, 1.0, 0.3), vec3f(0.3, 0.45, 1.0), vec3f(1.0, 0.9, 0.2), vec3f(0.35));
        return vec4f(cols[min(k, 4u)] * 0.6, 0.0);
    }
    let vz = -(g.view * vec4f(i.world, 1.0)).z;
    let cx = u32(clamp(i.clip.x * g.screen.z * f32(CLUSTER_X), 0.0, f32(CLUSTER_X - 1u)));
    let cy = u32(clamp(i.clip.y * g.screen.w * f32(CLUSTER_Y), 0.0, f32(CLUSTER_Y - 1u)));
    let cz = u32(clamp(floor(log(max(vz, g.cluster.x) / g.cluster.x) * g.cluster.z), 0.0, f32(CLUSTER_Z - 1u)));
    let n = f32(clusters[cx + cy * CLUSTER_X + cz * CLUSTER_X * CLUSTER_Y].y);
    let t = clamp(n / 32.0, 0.0, 1.0);
    return vec4f(mix(vec3f(0.02, 0.02, 0.1), vec3f(1.0, 0.85, 0.1), sqrt(t)) * select(1.0, 0.0, n == 0.0) + vec3f(0.03), 0.0);
}

// Opaque + alpha-tested geometry.
@fragment
fn fs_main(i: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let d = derivs(i);
    let face_n = normalize(cross(d.dpx, d.dpy));
    let s = shade(i, front, d, face_n);
    if (g.counts.z >= 3u) { return debug_extra(i); }
    return debug_out(s);
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
    let sky = sky_radiance(normalize(vec3f(refl.x, max(refl.y, 0.02), refl.z)));
    let h = normalize(l_sun + v);
    let glint = pow(max(dot(n, h), 0.0), 600.0) * 6.0 * g.sun_dir.w * shadow;

    let foam_noise = vnoise(i.world.xz * 6.0 + vec2f(t * 0.4, -t * 0.3));
    let foam = smoothstep(0.28, 0.0, thickness) * smoothstep(0.35, 0.75, foam_noise + 0.25);

    var color = mix(body, sky, fresnel) + g.sun_color.rgb * glint;
    color = mix(color, vec3f(0.95, 0.97, 1.0) * (0.6 + 0.4 * shadow), foam);
    let edge = smoothstep(0.0, 0.12, thickness); // soft intersection with the shore
    let alpha = clamp((mix(0.45, 0.94, depth_t) + fresnel * 0.4 + foam) * edge, 0.0, 1.0) * inst.base_color.a;
    if (g.counts.z == 1u) { return vec4f(pow(n * 0.5 + 0.5, vec3f(2.2)), 0.0); }
    return vec4f(finish(color, i.world), alpha);
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

// --- GPU grass (see vegetation.rs) -------------------------------------------------------------
struct GrassParams {
    terrain: vec4f,     // local corner x, z, size, res
    center: vec4f,      // local camera x, z, radius, spacing
    blade: vec4f,       // height, width, full-density fraction, time
    wind: vec4f,        // dir x, z, strength, grid side
    color_base: vec4f,
    color_tip: vec4f,
    layer: vec4u,       // x = splat channel, y = grid side
}
@group(2) @binding(0) var<uniform> grass: GrassParams;
@group(2) @binding(1) var height_map: texture_2d<f32>;
@group(2) @binding(2) var splat_map: texture_2d<f32>;

fn hash1(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn terrain_h(local_xz: vec2f) -> f32 {
    let res = i32(grass.terrain.w);
    let g = clamp((local_xz - grass.terrain.xy) / grass.terrain.z * f32(res - 1), vec2f(0.0), vec2f(f32(res - 1)));
    let i0 = vec2i(floor(g));
    let f = g - vec2f(i0);
    let i1 = min(i0 + 1, vec2i(res - 1));
    let h00 = textureLoad(height_map, i0, 0).r;
    let h10 = textureLoad(height_map, vec2i(i1.x, i0.y), 0).r;
    let h01 = textureLoad(height_map, vec2i(i0.x, i1.y), 0).r;
    let h11 = textureLoad(height_map, i1, 0).r;
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

@vertex
fn vs_grass(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var o: VsOut;
    o.inst = 0u;
    o.clip = vec4f(2.0, 2.0, 2.0, 1.0); // culled unless placed below
    let n = grass.layer.y;
    let sp = grass.center.w;
    let ix = i32(ii % n) - i32(n / 2u);
    let iz = i32(ii / n) - i32(n / 2u);
    let cell = floor(grass.center.xy / sp) + vec2f(f32(ix), f32(iz));
    let r1 = hash1(cell);
    let r2 = hash1(cell + 17.31);
    let r3 = hash1(cell * 1.7 + 3.1);
    let p = (cell + vec2f(r1, r2)) * sp;
    let d = distance(p, grass.center.xy);
    let radius = grass.center.z;
    if (d > radius) { return o; }
    // terrain bounds + density (splat weight) + distance thinning
    let suv = (p - grass.terrain.xy) / grass.terrain.z;
    if (any(suv < vec2f(0.0)) || any(suv > vec2f(1.0))) { return o; }
    let res = i32(grass.terrain.w);
    let st = textureLoad(splat_map, vec2i(suv * f32(res - 1) + 0.5), 0);
    let total = max(st.x + st.y + st.z + st.w, 1e-3);
    let w = st[grass.layer.x] / total;
    let full = radius * grass.blade.z;
    let keep = select(1.0, mix(1.0, 0.12, clamp((d - full) / max(radius - full, 1e-3), 0.0, 1.0)), d > full);
    if (r3 > w * keep) { return o; }
    // blade: 5 triangles, 3 segments tapering to a tip
    // 15 vertices: (side, level) per vertex — two quads (4 tris) and a tip triangle
    let sides = array<f32, 15>(-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 0.0);
    let lvls = array<u32, 15>(0u, 0u, 1u, 0u, 1u, 1u, 1u, 1u, 2u, 1u, 2u, 2u, 2u, 2u, 3u);
    let side = sides[vi];
    let lvl = lvls[vi];
    let levels = array<f32, 4>(0.0, 0.4, 0.75, 1.0);
    let widths = array<f32, 4>(1.0, 0.8, 0.45, 0.0);
    let t = levels[lvl];
    let thin = 1.0 / sqrt(max(keep, 0.12));
    let h = grass.blade.x * (0.6 + 0.8 * r1) * mix(1.0, 0.85, 1.0 - keep);
    let width = grass.blade.y * widths[lvl] * min(thin, 2.5);
    let yaw = r2 * 6.2831;
    let across = vec3f(cos(yaw), 0.0, sin(yaw));
    let facing = vec3f(-sin(yaw), 0.0, cos(yaw));
    // wind + natural lean, growing with height²
    let time = grass.blade.w;
    let gust = sin(time * 1.7 + dot(p, vec2f(0.21, 0.17)) + r1 * 2.0) * 0.6 + sin(time * 3.1 + p.x * 0.5) * 0.25;
    let bend = (grass.wind.xy * grass.wind.z * (0.5 + gust) + (vec2f(r1, r2) - 0.5) * 0.35) * t * t * h;
    let base = vec3f(p.x, terrain_h(p), p.y);
    let world = base + across * side * width * 0.5 + vec3f(bend.x, h * t * (1.0 - 0.15 * length(bend) / max(h, 1e-3)), bend.y);
    o.clip = g.view_proj * vec4f(world, 1.0);
    o.world = world;
    o.normal = normalize(mix(facing + vec3f(bend.x, 0.0, bend.y), vec3f(0.0, 1.0, 0.0), 0.55));
    o.uv = vec2f(r3, t);
    return o;
}

@fragment
fn fs_grass(i: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let n = normalize(select(-i.normal, i.normal, front) * vec3f(1.0, 0.0, 1.0) * 0.4 + vec3f(0.0, 1.0, 0.0));
    let tint = 0.8 + 0.4 * vnoise(i.world.xz * 0.15);
    let albedo = mix(grass.color_base.rgb, grass.color_tip.rgb, i.uv.y) * tint;
    let l = normalize(-g.sun_dir.xyz);
    var shadow = 1.0;
    if (g.counts.y == 1u) { shadow = shadow_factor(i.world, vec3f(0.0, 1.0, 0.0)); }
    // diffuse + translucency (sun through the blade) + sky + clustered lights
    let ndl = max(dot(n, l), 0.0);
    var color = albedo * g.sun_color.rgb * g.sun_dir.w * (ndl * 0.75 + 0.25) * shadow / 3.14159;
    color += albedo * mix(g.ground_color.rgb, g.sky_color.rgb, 0.7) * (0.35 + 0.65 * i.uv.y);
    let v = normalize(g.camera_pos.xyz - i.world);
    color += cluster_lights(i.clip, i.world, n, v, albedo, 0.0, 0.8);
    if (g.counts.z == 1u) { return vec4f(pow(n * 0.5 + 0.5, vec3f(2.2)), 0.0); }
    if (g.counts.z == 2u) { return vec4f(vec3f(shadow), 0.0); }
    return vec4f(finish(color, i.world), 1.0);
}

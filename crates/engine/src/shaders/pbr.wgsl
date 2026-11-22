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
    @location(3) color: vec4f, // sRGB bytes (Unorm8x4)
    @builtin(instance_index) inst: u32,
}

struct VsOut {
    // invariant: the depth prepass and the shading pass (depth equal) must compute bit-identical
    // positions although they are different pipelines
    @invariant @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) @interpolate(flat) inst: u32,
    @location(4) color: vec4f, // linear vertex colour (white = none)
}

// Instance indirection: draw instance k reads the record in slot visible_ids[first_instance + k].
// Records keep a stable slot per entity (only changed ones are uploaded); the id lists are written
// per frame (opaque: by cull.wgsl on the GPU path).
@group(0) @binding(4) var<storage, read> visible_ids: array<u32>;

const PI: f32 = 3.14159265;

// --- Water waves (shared by vertex displacement and the fragment normal) ----------------------
// Eight Gerstner waves around a wind direction with deep-water dispersion (speed = sqrt(g k)):
// sharp crests and wide troughs (`chop` = horizontal displacement), analytic normal, crest height
// for foam and subsurface scattering. `fp` = world size of one pixel (fragment) or one vertex
// spacing (vertex): waves shorter than ~2 samples fade out (no aliasing on far or coarse water).
struct Wave {
    disp: vec3f,
    n: vec3f,
    crest: f32,
}

// Per-wave constants, evaluated when the shader is compiled: direction (wind angle 0.65 + spread),
// wave number, weight, sqrt(g * wave number) for the dispersion.
const WAVE_DIR = array<vec2f, 8>(
    vec2f(cos(0.65), sin(0.65)), vec2f(cos(1.07), sin(1.07)), vec2f(cos(0.10), sin(0.10)), vec2f(cos(1.70), sin(1.70)),
    vec2f(cos(-0.55), sin(-0.55)), vec2f(cos(0.87), sin(0.87)), vec2f(cos(0.35), sin(0.35)), vec2f(cos(2.75), sin(2.75)));
const WAVE_LEN = array<f32, 8>(1.0, 1.37, 1.83, 2.6, 3.4, 4.7, 6.3, 8.9);
const WAVE_WGT = array<f32, 8>(1.0, 0.72, 0.52, 0.36, 0.25, 0.17, 0.11, 0.07);
const WAVE_SPEED = array<f32, 8>(
    sqrt(9.81 * 1.0), sqrt(9.81 * 1.37), sqrt(9.81 * 1.83), sqrt(9.81 * 2.6),
    sqrt(9.81 * 3.4), sqrt(9.81 * 4.7), sqrt(9.81 * 6.3), sqrt(9.81 * 8.9));

fn gerstner(p: vec2f, t: f32, amp: f32, scale: f32, chop: f32, fp: f32) -> Wave {
    var disp = vec3f(0.0);
    var nx = 0.0;
    var nz = 0.0;
    var ny = 0.0;
    let ts = sqrt(scale) * t; // sqrt(g * len * scale) * t = WAVE_SPEED * sqrt(scale) * t
    for (var k = 0; k < 8; k++) {
        let d = WAVE_DIR[k];
        let f = WAVE_LEN[k] * scale;
        let a = amp * WAVE_WGT[k] * (1.0 - smoothstep(0.6, 2.2, f * fp));
        if (a <= 0.0) { continue; }
        let q = min(chop, 1.0 / (8.0 * f * a + 1e-5)); // no loops at the crests
        let ph = f * dot(d, p) - WAVE_SPEED[k] * ts + f32(k) * 1.7;
        let s = sin(ph);
        let c = cos(ph);
        disp += vec3f(d.x * q * a * c, a * s, d.y * q * a * c);
        nx -= d.x * f * a * c;
        nz -= d.y * f * a * c;
        ny -= q * f * a * s;
    }
    var w: Wave;
    w.disp = disp;
    w.n = vec3f(nx, 1.0 + ny, nz);
    w.crest = disp.y / max(amp * 3.6, 1e-5);
    return w;
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
    // Material::bend: wind sway + lean away from the pushers, growing with the height above the origin
    if (inst.maps.w > 0.0 && inst.params.z >= 0.0 && inst.params.z < 2.5 && (g.push.x > 0.0 || dot(g.push.yz, g.push.yz) > 0.0)) {
        let base = inst.model[3].xyz;
        world = vec4f(world.xyz + bend_offset(base, max(world.y - base.y, 0.0), inst.maps.w), 1.0);
    }
    var uv = v.uv;
    if (inst.normal[2].w > 0.5) { // impostor card: faces the camera
        let iv = impostor_vertex(inst.model, v.pos, v.uv, inst.normal[2].w, g.camera_pos.xyz - inst.model[3].xyz);
        world = iv.world;
        normal = iv.normal;
        uv = iv.uv;
    }
    if (inst.params.z > 2.5) { // water: Gerstner waves in world space; uv = undisplaced world xz
        uv = world.xz;
        let spacing = max(inst.maps.x, length(world.xyz - g.camera_pos.xyz) * 0.025);
        let w = gerstner(world.xz, g.time.x, inst.maps.y, inst.maps.w, inst.maps.z, spacing);
        world = vec4f(world.xyz + w.disp, 1.0);
        normal = w.n;
    }
    var o: VsOut;
    o.clip = g.view_proj * world;
    o.world = world.xyz;
    o.normal = normal;
    o.uv = uv;
    o.inst = idx;
    let c = v.color.rgb;
    o.color = vec4f(c * (c * (c * 0.305306011 + 0.682171111) + 0.012522878), v.color.a); // sRGB -> linear
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

// brdf with Material::subsurface (wrap diffuse: wrap = sss / 2) and Material::clearcoat (second GGX
// lobe, F0 0.04, over the base, which loses what the coat reflects).
fn brdf_plastic(n: vec3f, v: vec3f, l: vec3f, albedo: vec3f, metallic: f32, rough: f32, sss: f32, coat: f32, coat_rough: f32) -> vec3f {
    let h = normalize(v + l);
    let nl_raw = dot(n, l);
    let n_l = max(nl_raw, 0.0);
    let wrap = sss * 0.5;
    let n_l_w = clamp((nl_raw + wrap) / (1.0 + wrap), 0.0, 1.0);
    let n_v = max(dot(n, v), 1e-4);
    let n_h = max(dot(n, h), 0.0);
    let v_h = max(dot(v, h), 0.0);
    let a = rough * rough;
    let f0 = mix(vec3f(0.04), albedo, metallic);
    let f = f_schlick(f0, v_h);
    let spec = d_ggx(n_h, a) * v_smith(n_v, n_l, a) * f;
    let kd = (vec3f(1.0) - f) * (1.0 - metallic);
    var c = kd * albedo / PI * n_l_w + spec * n_l;
    if (coat > 0.0) {
        let ac = coat_rough * coat_rough;
        let fc = f_schlick(vec3f(0.04), v_h).x * coat;
        c = c * (1.0 - fc) + d_ggx(n_h, ac) * v_smith(n_v, n_l, ac) * fc * n_l;
    }
    return c;
}

// Shadow-map coordinate of `world` in cascade k: (u, v, depth, 1) or w = 0 outside the map.
// No texture access here: it may be called from several places (see shadow_factor).
// p0 / n0: position and normal in cascade 0's light space (current set).
fn cascade_coord(world: vec3f, n: vec3f, p0: vec3f, n0: vec3f, k: u32, prev: bool) -> vec4f {
    // normal offset scaled by the cascade's texel size (no acne on near cascades, no peter-panning far)
    let bias = g.cascade_texel[k] * 1.5;
    var ndc: vec3f;
    if (prev) {
        // previous direction while cross-fading: its own matrices (rare, a few hundred ms)
        let lp = g.cascades_prev[k] * vec4f(world + n * bias, 1.0);
        ndc = lp.xyz / lp.w;
    } else {
        // cascade k = cascade 0 scaled + offset (ortho, same rotation): no dynamically indexed
        // mat4 per pixel (that cost ~2 ms of the opaque pass at 2400x1350 on an RDNA3 iGPU)
        ndc = (p0 + n0 * bias) * g.cascade_scale[k].xyz + g.cascade_offset[k].xyz;
    }
    let uv = ndc.xy * vec2f(0.5, -0.5) + vec2f(0.5);
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || ndc.z > 1.0) { return vec4f(0.0); }
    return vec4f(uv, ndc.z - 0.0008, 1.0);
}

// 3x3 PCF of bilinear comparison taps = a 4x4 texel footprint with the separable weights
// (1-f, 1, 1, f). Two bilinear taps per axis reproduce exactly that (weights 2-f and 1+f, each
// placed between its two texels), so 4 taps give the same result as the 9 they replace.
// Returns (ua, ub) tap positions; weights via pcf_weights.
fn pcf_taps(uv: vec2f, size: vec2f) -> vec4f {
    let st = uv * size - 0.5;
    let base = floor(st);
    let f = st - base;
    let ua = (base - 0.5 + 1.0 / (2.0 - f)) / size;
    let ub = (base + 1.5 + f / (1.0 + f)) / size;
    return vec4f(ua, ub);
}

fn pcf_weights(uv: vec2f, size: vec2f) -> vec4f {
    let f = fract(uv * size - 0.5);
    return vec4f(2.0 - f, 1.0 + f) / 3.0;
}

// Cascaded shadow maps: cascade by view depth, blended into the next cascade over the last 15%
// and faded out at the shadow distance. The texture taps are written out here, inside ONE
// function with ONE call site per entry point: PowerVR's Vulkan driver fails to build the
// pipeline (VK_ERROR_UNKNOWN) when a helper that samples is called from several places (the
// coordinate helpers above do not sample). No loop around the taps, no dynamically indexed mat4
// copied into a variable: both cost 2.5-3.5 ms of the opaque pass at 4K on an RDNA3 iGPU.
// While a stepped sun cross-fades, the previous direction's set comes in dithered per pixel
// (weight shadow_sets.z, a few hundred milliseconds).
fn shadow_factor(world: vec3f, n: vec3f, frag: vec2f) -> f32 {
    let vz = -(g.view * vec4f(world, 1.0)).z;
    let count = g.counts.w;
    var k = 0u;
    loop {
        if (k >= count || vz <= g.cascade_splits[k]) { break; }
        k++;
    }
    if (k >= count) { return 1.0; }
    let start = select(0.0, g.cascade_splits[max(k, 1u) - 1u], k > 0u);
    let end = g.cascade_splits[k];
    let band = mix(start, end, 0.85);
    let t = clamp((vz - band) / max(end - band, 1e-3), 0.0, 1.0);
    // right after a sun step the current set's cascades k > 0 are re-rendered one per frame; until
    // then they hold an older direction (cascade_scale[k].w = 1): those pixels take the previous set
    let stale = g.shadow_sets.z > 0.0 && (g.cascade_scale[k].w > 0.5 || (t > 0.0 && k + 1u < count && g.cascade_scale[k + 1u].w > 0.5));
    let prev = stale || fract(52.9829189 * fract(dot(floor(frag), vec2f(0.06711056, 0.00583715)))) < g.shadow_sets.z;
    let layer_base = i32(select(g.shadow_sets.x, g.shadow_sets.y, prev));

    let size = vec2f(textureDimensions(shadow_map));
    let p0 = (g.cascades[0] * vec4f(world, 1.0)).xyz;
    let n0 = (g.cascades[0] * vec4f(n, 0.0)).xyz;
    var s = 1.0;
    let ca = cascade_coord(world, n, p0, n0, k, prev);
    if (ca.w > 0.0) {
        let tp = pcf_taps(ca.xy, size);
        let w = pcf_weights(ca.xy, size);
        let l = i32(k) + layer_base;
        s = w.x * w.y * textureSampleCompareLevel(shadow_map, shadow_sampler, tp.xy, l, ca.z)
          + w.z * w.y * textureSampleCompareLevel(shadow_map, shadow_sampler, vec2f(tp.z, tp.y), l, ca.z)
          + w.x * w.w * textureSampleCompareLevel(shadow_map, shadow_sampler, vec2f(tp.x, tp.w), l, ca.z)
          + w.z * w.w * textureSampleCompareLevel(shadow_map, shadow_sampler, tp.zw, l, ca.z);
    }
    if (t > 0.0) {
        // blend band: the next cascade (beyond the last one: lit)
        var s2 = 1.0;
        if (k + 1u < count) {
            let cb = cascade_coord(world, n, p0, n0, k + 1u, prev);
            if (cb.w > 0.0) {
                let tp = pcf_taps(cb.xy, size);
                let w = pcf_weights(cb.xy, size);
                let l = i32(k + 1u) + layer_base;
                s2 = w.x * w.y * textureSampleCompareLevel(shadow_map, shadow_sampler, tp.xy, l, cb.z)
                   + w.z * w.y * textureSampleCompareLevel(shadow_map, shadow_sampler, vec2f(tp.z, tp.y), l, cb.z)
                   + w.x * w.w * textureSampleCompareLevel(shadow_map, shadow_sampler, vec2f(tp.x, tp.w), l, cb.z)
                   + w.z * w.w * textureSampleCompareLevel(shadow_map, shadow_sampler, tp.zw, l, cb.z);
            }
        }
        s = mix(s, s2, t);
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
    return sanitize(apply_fog(c, world));
}

// normalize() of a zero vector is NaN: edge-on triangles have zero screen-space derivatives, and a
// single NaN pixel showed up as a black dot with a bloom halo. Fall back instead.
fn safe_normal(v: vec3f, fallback: vec3f) -> vec3f {
    let l2 = dot(v, v);
    if (l2 > 1e-20) { return v * inverseSqrt(l2); }
    let f2 = dot(fallback, fallback);
    return select(vec3f(0.0, 1.0, 0.0), fallback * inverseSqrt(max(f2, 1e-20)), f2 > 1e-20);
}

// Last line of defence for the HDR target: NaN / inf / absurd values (grazing-angle specular spikes)
// become finite. Comparisons with NaN are false, so NaN lanes take the 0 branch.
fn sanitize(c: vec3f) -> vec3f {
    let ok = (c >= vec3f(0.0)) & (c <= vec3f(6.0e4));
    return select(vec3f(0.0), min(c, vec3f(2.0e3)), ok);
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
        if (w[k] < 0.002) { continue; } // layer not present here: skip its (anisotropic) tap
        let tile = max(inst.maps[k], 0.01);
        let q = vec2f(f32(k % 2), f32(k / 2)) * 0.5 + vec2f(0.5 * pad);
        let uv = q + fract(i.world.xz / tile) * scale;
        c += textureSampleGrad(base_tex, base_sampler, uv, d.dpx.xz / tile * scale, d.dpy.xz / tile * scale).rgb * w[k];
    }
    // large-scale variation against visible tiling
    return c * (0.82 + 0.36 * vnoise(i.world.xz * 0.035));
}

// Material::studs: analytic stud relief. A flat face shows a grid of standing LEGO studs (cylinder
// r_base, top bevel = cone to r_top over `bevel`, flat cap at `height`; stud units, centres at
// (i + 0.5) in uv * studs-per-uv). The view ray enters the stud slab at the top (shifted by
// view.xy / max(view.z, 0.16) in tangent space) and descends to the face; it is intersected with the
// 3x3 or 5x5 neighbouring studs analytically, the first hit wins. Result: tangent-space normal of the
// hit (cap = up, bevel = cone, wall = radial; plate = up), the uv of the hit point (the colour map
// follows the stud), a soft shade (0.92..1 by height on a stud, 0.95..1 within 0.16 of a stud's
// foot on the plate) and the weight (fades out where a stud is smaller than ~2 pixels).
struct StudHit {
    n: vec3f,
    uv: vec2f,
    shade: f32,
    w: f32,
}

fn stud_relief(uv: vec2f, view_ts: vec3f, st: vec4f, st2: vec4f, wide: bool, d: Deriv) -> StudHit {
    var r: StudHit;
    r.n = vec3f(0.0, 0.0, 1.0);
    r.uv = uv;
    r.shade = 1.0;
    let inv = st.x;
    let foot = max(length(d.duvx), length(d.duvy)) * inv; // studs per pixel
    r.w = 1.0 - smoothstep(0.3, 0.6, foot);
    if (r.w <= 0.0 || view_ts.z <= 0.0) { r.w = 0.0; return r; }
    let h = st.y;
    let rb = st.z;
    let rt = st.w;
    let bev = st2.x;
    let k = select(0.0, (rb - rt) / bev, bev > 1e-5);
    let s = uv * inv;
    let e = view_ts.xy / max(view_ts.z, 0.16);
    let o = s + e * h;  // ray entry at the top of the slab; p(t) = o - e t, height h - t
    let dq = -e;
    // only the cells whose stud can touch the ray's footprint (o -> s), at most 3x3 / 5x5 around its middle
    let mid = floor(s + e * (h * 0.5));
    let rad = select(1.0, 2.0, wide);
    let lo = max(ceil(min(o, s) - rb - 0.5), mid - rad);
    let hi = min(floor(max(o, s) + rb - 0.5), mid + rad);
    var best = 1e9;
    var bn = vec3f(0.0, 0.0, 1.0);
    for (var cy = lo.y; cy <= hi.y; cy += 1.0) {
        for (var cx = lo.x; cx <= hi.x; cx += 1.0) {
            let q0 = o - (vec2f(cx, cy) + 0.5);
            let c0 = dot(q0, q0);
            // cap (the ray starts at cap height)
            if (c0 <= rt * rt) {
                if (0.0 < best) { best = 0.0; bn = vec3f(0.0, 0.0, 1.0); }
                continue;
            }
            // bevel cone, t in [0, bev]: |q0 + t dq| = rt + k t
            if (bev > 1e-5) {
                let a = dot(dq, dq) - k * k;
                let b = 2.0 * (dot(q0, dq) - rt * k);
                let c = c0 - rt * rt;
                var t = -1.0;
                if (abs(a) < 1e-6) {
                    if (abs(b) > 1e-6) { t = -c / b; }
                } else {
                    let disc = b * b - 4.0 * a * c;
                    if (disc >= 0.0) {
                        let sq = sqrt(disc);
                        let t1 = (-b - sq) / (2.0 * a);
                        let t2 = (-b + sq) / (2.0 * a);
                        let lo = min(t1, t2);
                        let hi = max(t1, t2);
                        t = select(hi, lo, lo >= 0.0);
                    }
                }
                if (t >= 0.0 && t <= bev && t < best && rt + k * t >= 0.0) {
                    let q = q0 + dq * t;
                    best = t;
                    bn = normalize(vec3f(q / max(length(q), 1e-5), k));
                    continue;
                }
            }
            // wall, t in [bev, h]: |q0 + t dq| = rb
            let a = dot(dq, dq);
            if (a > 1e-8) {
                let b = 2.0 * dot(q0, dq);
                let c = c0 - rb * rb;
                let disc = b * b - 4.0 * a * c;
                if (disc >= 0.0) {
                    let t = (-b - sqrt(disc)) / (2.0 * a);
                    if (t >= bev && t <= h && t < best) {
                        let q = q0 + dq * t;
                        best = t;
                        bn = vec3f(q / max(length(q), 1e-5), 0.0);
                    }
                }
            }
        }
    }
    if (best <= h) {
        r.n = bn;
        r.uv = (o + dq * best) / inv;
        r.shade = mix(0.92, 1.0, (h - best) / max(h, 1e-5));
    } else {
        // plate: soft contact shade around the nearest stud's foot
        let dist = length(fract(s) - 0.5);
        r.shade = mix(0.95, 1.0, smoothstep(rb, rb + 0.16, dist));
    }
    r.shade = mix(1.0, r.shade, r.w);
    return r;
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
    let nn = safe_normal(i.normal, vec3f(0.0, 1.0, 0.0));
    var n = select(nn, normalize(nn - 2.0 * dot(nn, face_n) * face_n), !front);
    let n_geo = n; // before normal mapping (horizon occlusion of the ambient specular)
    let v = normalize(g.camera_pos.xyz - i.world);
    let tbn = cotangent_frame(n, d);

    let is_terrain = inst.params.z < -0.5;
    var uv = i.uv;
    if (inst.maps.z > 0.0 && !is_terrain) { // parallax occlusion mapping
        let view_ts = vec3f(dot(v, tbn[0]), dot(v, tbn[1]), dot(v, tbn[2]));
        uv = parallax_uv(uv, view_ts, inst.maps.z, d);
    }
    // Material::studs (render.stud_relief = quality, 0 = off)
    var stud = StudHit(vec3f(0.0, 0.0, 1.0), uv, 1.0, 0.0);
    var stud_tbn = tbn;
    if (inst.studs.x > 0.0 && g.shadow_sets.w > 0.5 && !is_terrain && (inst.studs2.z < 0.5 || n_geo.y > 0.5)) {
        // cotangent_frame drops the determinant's sign (WebGPU's y-down screen: T, B = -du, -dv on
        // front faces); the ray needs the true +u / +v directions, so orient the frame by the uv gradient
        let ori = dot(d.dpx, tbn[0]) * d.duvx.x + dot(d.dpy, tbn[0]) * d.duvy.x;
        let fs = select(-1.0, 1.0, ori >= 0.0);
        stud_tbn = mat3x3f(tbn[0] * fs, tbn[1] * fs, tbn[2]);
        let view_ts = vec3f(dot(v, stud_tbn[0]), dot(v, stud_tbn[1]), dot(v, stud_tbn[2]));
        stud = stud_relief(uv, view_ts, inst.studs, inst.studs2, g.shadow_sets.w > 1.5, d);
        uv = mix(uv, stud.uv, step(0.5, stud.w)); // the colour map follows the stud (no half shift)
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
    if (stud.w > 0.0) { // the stud normal replaces the surface (and normal-map) normal
        n = normalize(mix(n, normalize(stud_tbn * stud.n), stud.w * inst.studs2.y));
    }

    let alpha_mode = inst.params.w;
    var alpha = inst.base_color.a * tex.a * i.color.a;
    if (alpha_mode > 0.0) {            // mask (alpha test)
        if (ALPHA_TEST && alpha < alpha_mode) { discard; }
        alpha = 1.0;
    } else if (alpha_mode == 0.0) {    // opaque
        alpha = 1.0;
    }
    var albedo = tex.rgb * inst.base_color.rgb * i.color.rgb * stud.shade;
    let checker = inst.params.y;
    if (checker > 0.0) {
        let c = floor(i.world.x * checker) + floor(i.world.z * checker);
        albedo *= select(1.0, 0.78, (i32(c) & 1) == 0);
    }
    // The sun term is brdf * n_l * shadow: surfaces facing away from the sun (n_l = 0) and unlit / UI
    // surfaces never use the shadow term, so they skip the shadow-map taps (debug view 2 shows it).
    let l_sun = normalize(-g.sun_dir.xyz);
    let sss = select(0.0, inst.normal[3].w, !is_terrain); // Material::subsurface
    var shadow = 1.0;
    if (g.counts.y == 1u && ((dot(n, l_sun) > -0.5 * sss && inst.params.z < 0.5) || g.counts.z == 2u)) {
        shadow = shadow_factor(i.world, n, i.clip.xy);
    }

    if (inst.params.z > 1.5) { // ui surface (world panels): texture as-is, alpha-tested
        if (ALPHA_TEST && tex.a < 0.5) { discard; }
        return Surface(vec4f(albedo, 0.0), n, shadow); // alpha 0: not tonemapped
    }
    // one call of finish (fog) for lit and unlit (see shadow_factor: one call site per sampling helper)
    var color = albedo + inst.emissive_metallic.rgb; // unlit
    if (inst.params.z <= 0.5) {
        let metallic = select(inst.emissive_metallic.a, 0.0, is_terrain);
        let rough = clamp(inst.params.x, 0.04, 1.0);
        let coat = select(0.0, inst.normal[0].w, !is_terrain);   // Material::clearcoat
        let coat_rough = inst.normal[1].w;
        let sun = g.sun_color.rgb * g.sun_dir.w;
        if (sss > 0.0 || coat > 0.0) {
            color = brdf_plastic(n, v, l_sun, albedo, metallic, rough, sss, coat, coat_rough) * sun * shadow;
            if (sss > 0.0) { // thin back-light translucency (unshadowed: back faces would self-shadow)
                let sc = select(inst.normal[3].rgb, albedo, inst.normal[3].r < 0.0);
                let back = pow(clamp(dot(v, -normalize(l_sun + n * 0.3)), 0.0, 1.0), 4.0) * sss * 0.25;
                color += back * sc * sun;
            }
        } else {
            color = brdf(n, v, l_sun, albedo, metallic, rough) * sun * shadow;
        }
        color += cluster_lights(i.clip, i.world, n, v, albedo, metallic, rough);
        let hemi = mix(g.ground_color.rgb, g.sky_color.rgb, n.y * 0.5 + 0.5);
        color += hemi * albedo * (1.0 - metallic * 0.6);
        if (g.push.w > 0.0) {
            color += ambient_specular(n, n_geo, v, albedo, metallic, rough) * g.push.w;
            if (coat > 0.0) { color += ambient_specular(n, n_geo, v, vec3f(0.04), 0.0, coat_rough) * g.push.w * coat; }
        }
        if (!is_terrain) { color += inst.emissive_metallic.rgb; }
    }
    return Surface(vec4f(finish(color, i.world), alpha), n, shadow);
}

// render.ambient_specular: the hemisphere (ground -> sky colour) seen in the reflection direction,
// weighted by Karis' analytic split-sum environment BRDF (F0 0.04 for dielectrics, albedo for
// metals): a fresnel rim and sky sheen outside the sun's highlight that fades with roughness.
// Reflections that point below the geometric surface are occluded.
fn ambient_specular(n: vec3f, n_geo: vec3f, v: vec3f, albedo: vec3f, metallic: f32, rough: f32) -> vec3f {
    let r = reflect(-v, n);
    let env = mix(g.ground_color.rgb, g.sky_color.rgb, smoothstep(-0.3, 0.3, r.y));
    let nv = max(dot(n, v), 1e-4);
    let c = rough * vec4f(-1.0, -0.0275, -0.572, 0.022) + vec4f(1.0, 0.0425, 1.04, -0.04);
    let a004 = min(c.x * c.x, exp2(-9.28 * nv)) * c.x + c.y;
    let ab = vec2f(-1.04, 1.04) * a004 + c.zw;
    let f0 = mix(vec3f(0.04), albedo, metallic);
    let horizon = clamp(1.0 + 1.5 * dot(r, n_geo), 0.0, 1.0);
    return env * (f0 * ab.x + ab.y) * horizon * horizon;
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
// Depth prepass (render.depth_prepass): same vertex shader, no colour written.
@fragment
fn fs_depth() -> @location(0) vec4f {
    return vec4f(0.0);
}

@fragment
fn fs_main(i: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let d = derivs(i);
    let face_n = safe_normal(cross(d.dpx, d.dpy), i.normal);
    let s = shade(i, front, d, face_n);
    if (g.counts.z >= 3u) { return ldr_opaque(debug_extra(i), i.clip.xy); }
    return ldr_opaque(debug_out(s), i.clip.xy);
}

// --- Blend pass: the opaque depth buffer is readable here -------------------------------------
@group(0) @binding(5) var scene_depth: texture_depth_multisampled_2d;
// Resolved opaque scene colour (water refraction + screen-space reflections).
@group(0) @binding(9) var scene_color: texture_2d<f32>;

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

// Forward differences of vnoise over a step `e` in x and in y, i.e. (vnoise(p + (e, 0)) - vnoise(p)) / e
// and the same for y, from the four corner hashes of the cell of p (one noise lookup instead of
// three; the smoothstep weights have zero slope at the cell borders, so clamping the step at a
// border changes nothing visible).
fn vnoise_grad(p: vec2f, e: f32) -> vec2f {
    let i = floor(p);
    let f = fract(p);
    let a = hash2(i);
    let b = hash2(i + vec2f(1.0, 0.0));
    let c = hash2(i + vec2f(0.0, 1.0));
    let d = hash2(i + vec2f(1.0, 1.0));
    let u = f * f * (3.0 - 2.0 * f);
    let fe = min(f + e, vec2f(1.0));
    let ue = fe * fe * (3.0 - 2.0 * fe);
    let h0 = mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    let hx = mix(mix(a, b, ue.x), mix(c, d, ue.x), u.y);
    let hy = mix(mix(a, b, u.x), mix(c, d, u.x), ue.y);
    return vec2f(hx - h0, hy - h0) / e;
}

// Screen-space reflection: march the reflected ray through the opaque depth buffer (geometric
// steps, then bisection). Returns (colour, confidence); confidence fades at the screen border and
// with distance, the caller falls back to the sky.
fn ssr(origin: vec3f, r: vec3f) -> vec4f {
    // Only the part of the ray inside the bounds of the opaque scene can hit anything: rays that
    // miss them (open sea reflecting the sky) cost nothing, the others skip the steps outside.
    let rs = select(r, vec3f(1e-6), abs(r) < vec3f(1e-6));
    let ta = (g.scene_min.xyz - origin) / rs;
    let tb = (g.scene_max.xyz - origin) / rs;
    let t_lo = min(ta, tb);
    let t_hi = max(ta, tb);
    let t_in = max(max(t_lo.x, t_lo.y), t_lo.z);
    let t_out = min(min(t_hi.x, t_hi.y), t_hi.z);
    if (t_out < max(t_in, 0.0)) { return vec4f(0.0); }
    // the clip position is linear in t: two matrix products for the whole march
    let c0 = g.view_proj * vec4f(origin, 1.0);
    let cd = g.view_proj * vec4f(r, 0.0);
    var t = 0.35;
    var prev = 0.0;
    let max_t = 400.0;
    for (var s = 0; s < 22; s++) {
        if (prev > t_out) { break; } // left the scene bounds
        if (t < t_in) { // not there yet: same step sequence, no depth lookups
            prev = t;
            t = t * 1.3 + 0.25;
            if (t > max_t) { break; }
            continue;
        }
        let c = c0 + cd * t;
        if (c.w <= g.time.y) { break; }
        let uv = vec2f(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
        if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { break; }
        let sd = linear_depth(textureLoad(scene_depth, vec2i(uv * g.screen.xy), 0));
        if (c.w > sd) {
            if (c.w - sd > max(t * 0.06, 0.25) + (t - prev)) { break; } // passed behind: no hit
            var lo = prev;
            var hi = t;
            var huv = uv;
            for (var b = 0; b < 5; b++) {
                let mid = 0.5 * (lo + hi);
                let cm = c0 + cd * mid;
                let um = vec2f(cm.x / cm.w * 0.5 + 0.5, 0.5 - cm.y / cm.w * 0.5);
                let dm = linear_depth(textureLoad(scene_depth, vec2i(clamp(um, vec2f(0.0), vec2f(0.9999)) * g.screen.xy), 0));
                if (cm.w > dm) { hi = mid; huv = um; } else { lo = mid; }
            }
            huv = clamp(huv, vec2f(0.0), vec2f(0.9999));
            let col = textureLoad(scene_color, vec2i(huv * g.screen.xy), 0).rgb;
            let edge = smoothstep(0.0, 0.1, min(min(huv.x, 1.0 - huv.x), min(huv.y, 1.0 - huv.y)));
            let far = 1.0 - smoothstep(max_t * 0.5, max_t, t);
            return vec4f(col, edge * far);
        }
        prev = t;
        t = t * 1.3 + 0.25;
        if (t > max_t) { break; }
    }
    return vec4f(0.0);
}

// Light patterns on the ground below water: two layers of warped noise ridges.
fn caustics(p: vec2f, t: f32) -> f32 {
    var c = 0.0;
    for (var k = 0; k < 2; k++) {
        let s = 1.1 + f32(k) * 0.8;
        let q = p * s + vec2f(t * 0.21, -t * 0.17) * (1.0 + f32(k) * 0.6);
        let w = vnoise(q * 0.6 + vec2f(t * 0.05, 0.0)) * 1.4;
        let n = vnoise(q + vec2f(w, -w));
        c += pow(1.0 - abs(n * 2.0 - 1.0), 7.0);
    }
    return c;
}

// Water (params.z == 3): Gerstner waves + three octaves of drifting ripples; refraction of the
// opaque scene with per-channel absorption (red fades first) and in-scattering towards the deep
// colour; caustics on the ground below; reflections = screen-space reflections over the full sky
// (atmosphere, clouds, moon, skybox); GGX sun specular + glint; subsurface glow through crests;
// shore and crest foam; soft shore edge.
// Material: base_color = deep colour, emissive = shallow colour, emissive.a = foam,
// params.y = clarity (m), maps = (vertex spacing, wave height, chop, wave scale).
fn water(i: VsOut, d: Deriv) -> vec4f {
    let inst = instances[i.inst];
    let t = g.time.x;
    let fp = max(length(d.dpx), length(d.dpy));
    let base = i.uv; // undisplaced world xz (set by vs_main)
    let wv = gerstner(base, t, inst.maps.y, inst.maps.w, inst.maps.z, fp);
    var r = vec2f(0.0);
    let scales = array<f32, 3>(3.1, 7.3, 15.7);
    let drift = array<vec2f, 3>(vec2f(0.35, 0.2), vec2f(-0.25, 0.4), vec2f(0.15, -0.45));
    for (var k = 0; k < 3; k++) {
        let f = scales[k] * inst.maps.w;
        let q = base * f + drift[k] * t * sqrt(f) * 0.9;
        let gr = vnoise_grad(q, 0.05);
        let gx = gr.x;
        let gz = gr.y;
        let fade = 1.0 - smoothstep(0.5, 2.0, f * fp * 6.28);
        r += vec2f(gx, gz) * f * 0.03 * fade * (inst.maps.y / max(inst.maps.y, 0.02) * 0.6 + 0.4);
    }
    let n = normalize(normalize(wv.n) + vec3f(-r.x, 0.0, -r.y));
    let v = normalize(g.camera_pos.xyz - i.world);
    let n_v = clamp(dot(n, v), 0.0, 1.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - n_v, 5.0);
    let l_sun = normalize(-g.sun_dir.xyz);
    var shadow = 1.0;
    if (g.counts.y == 1u) { shadow = shadow_factor(i.world, vec3f(0.0, 1.0, 0.0), i.clip.xy); }

    // --- refraction: offset the scene lookup along the wave normal, unless that lands in front
    let px = vec2i(i.clip.xy);
    let water_lin = linear_depth(i.clip.z);
    let max_px = vec2i(g.screen.xy) - vec2i(1);
    let off = n.xz * g.misc.y * g.screen.y * 0.035 / (1.0 + water_lin * 0.06);
    var q = clamp(vec2i(i.clip.xy + off), vec2i(0), max_px);
    var raw = textureLoad(scene_depth, q, 0);
    if (linear_depth(raw) < water_lin) {
        q = px;
        raw = textureLoad(scene_depth, px, 0);
    }
    let scene_lin = linear_depth(raw);
    let thickness = max(scene_lin - water_lin, 0.0); // view-depth difference below the surface
    var refracted = textureLoad(scene_color, q, 0).rgb;

    // caustics on the ground below (reconstructed from the depth buffer)
    if (g.misc.z > 0.0 && raw < 1.0) {
        let uvq = (vec2f(q) + 0.5) * g.screen.zw;
        let wp = g.inv_view_proj * vec4f(uvq.x * 2.0 - 1.0, 1.0 - uvq.y * 2.0, raw, 1.0);
        let bottom = wp.xyz / wp.w;
        let below = i.world.y - bottom.y;
        let k = caustics(bottom.xz * inst.maps.w * 1.5, t) * g.misc.z * exp(-below * 0.5) * smoothstep(0.02, 0.3, below);
        refracted *= 1.0 + k * clamp(g.sun_dir.w * 0.25, 0.0, 1.5) * shadow;
    }

    // --- absorption + in-scattering (Beer-Lambert, red absorbed first)
    let clarity = max(inst.params.y, 0.05);
    let absorb = exp(-thickness / clarity * vec3f(2.2, 0.9, 0.65));
    let depth_t = 1.0 - exp(-thickness / clarity);
    let deep = inst.base_color.rgb;
    let shallow = inst.emissive_metallic.rgb;
    let lit = g.sun_color.rgb * g.sun_dir.w * (0.2 + 0.3 * max(dot(n, l_sun), 0.0)) * shadow + g.sky_color.rgb * 0.7;
    let scatter = mix(shallow, deep, depth_t) * lit;
    var body = refracted * absorb + scatter * (1.0 - absorb.g);
    // subsurface glow: sunlight through the thin crests towards the viewer
    let hs = normalize(l_sun + n * 0.6);
    let sss = pow(clamp(dot(v, -hs), 0.0, 1.0), 4.0) * clamp(wv.crest * 0.6 + 0.4, 0.0, 1.0);
    body += shallow * g.sun_color.rgb * g.sun_dir.w * sss * 0.8 * shadow;

    // --- reflection: SSR over the full sky
    var rd = reflect(-v, n);
    rd = normalize(vec3f(rd.x, max(rd.y, 0.015), rd.z));
    var reflection = sky_full(rd, false);
    if (g.misc.x > 0.5 && fresnel > 0.04) { // below that a reflection is barely visible
        let hit = ssr(i.world + n * 0.02, rd);
        reflection = mix(reflection, hit.rgb, hit.a);
    }

    // --- sun specular (GGX, smooth) + glint
    let h = normalize(l_sun + v);
    let n_l = max(dot(n, l_sun), 0.0);
    let rough = clamp(inst.params.x, 0.02, 0.5) + fp * 0.02;
    let spec = d_ggx(max(dot(n, h), 0.0), rough * rough) * v_smith(max(n_v, 1e-3), n_l, rough * rough) * f_schlick(vec3f(0.02), max(dot(v, h), 0.0)) * n_l;
    let glint = pow(max(dot(n, h), 0.0), 900.0) * 4.0;
    let sun_spec = g.sun_color.rgb * g.sun_dir.w * shadow * (spec + glint);

    // --- foam: shore (thin water) and crests
    let foam_amt = inst.emissive_metallic.a;
    let fn1 = vnoise(base * 5.5 + vec2f(t * 0.4, -t * 0.3));
    let fn2 = vnoise(base * 13.0 - vec2f(t * 0.2, t * 0.5));
    let shore = smoothstep(0.35, 0.0, thickness) * smoothstep(0.35, 0.75, fn1 * 0.7 + fn2 * 0.3 + 0.25);
    let crest = smoothstep(0.55, 0.95, wv.crest + (fn2 - 0.5) * 0.35) * smoothstep(0.3, 0.7, fn1) * clamp(inst.maps.z, 0.0, 1.0);
    let foam = clamp((shore + crest) * foam_amt, 0.0, 1.0);

    var color = mix(body, reflection, fresnel) + sun_spec;
    color = mix(color, vec3f(0.92, 0.95, 0.97) * (lit * 0.9 + 0.05), foam);
    let edge = smoothstep(0.0, 0.08, thickness); // soft intersection with the shore
    let alpha = clamp(edge, 0.0, 1.0) * inst.base_color.a;
    if (g.counts.z == 1u) { return vec4f(pow(n * 0.5 + 0.5, vec3f(2.2)), 0.0); }
    return vec4f(finish(color, i.world), alpha);
}

@fragment
fn fs_blend(i: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    let d = derivs(i);
    let face_n = safe_normal(cross(d.dpx, d.dpy), i.normal);
    if (instances[i.inst].params.z > 2.5) {
        return ldr_blended(water(i, d), i.clip.xy);
    }
    return ldr_blended(debug_out(shade(i, front, d, face_n)), i.clip.xy);
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
    o.color = vec4f(1.0);
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
    var bend = (grass.wind.xy * grass.wind.z * (0.5 + gust) + (vec2f(r1, r2) - 0.5) * 0.35) * t * t * h;
    let base = vec3f(p.x, terrain_h(p), p.y);
    if (g.push.x > 0.0) { // pushers flatten blades around them
        let pb = push_offset(base, h, 1.6).xz;
        let lim = 0.85 * h;
        bend = (bend + pb * t * t);
        let bl = length(bend);
        if (bl > lim) { bend = bend / bl * lim; }
    }
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
    if (g.counts.y == 1u) { shadow = shadow_factor(i.world, vec3f(0.0, 1.0, 0.0), i.clip.xy); }
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

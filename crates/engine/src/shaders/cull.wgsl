// GPU-driven culling (compute) for the main view and the shadow cascades (see gpu_scene.rs).
// One thread per object slot:
//   1. LOD level from the distance to the camera (same rule as `Lod::pick` on the CPU)
//   2. main view: frustum test, then occlusion against the Hi-Z pyramid of the *previous* frame
//      (conservative; objects revealed by fast camera moves can pop in one frame late); survivors
//      append their record slot to their draw group's slice of `ids` and bump its instance count
//   3. every shadow cascade rendered this frame: light-space box test, append to the cascade's
//      slice of the caster's shadow group
// `reset` runs first and clears the instance counts, the per-group nearest distances and the counters.
// Standalone shader (not prefixed with common.wgsl).

const NONE: u32 = 0xffffffffu;

struct Params {
    planes: array<vec4f, 6>,
    prev_view_proj: mat4x4f,
    cam: vec4f,        // xyz = camera position, w = 1 / lod_bias
    hiz_size: vec2f,   // texel size of Hi-Z mip 0
    count: u32,        // object slots
    flags: u32,        // bit 0 = frustum, bit 1 = occlusion
    hiz_mips: u32,
    layers: u32,       // camera layer mask
    n_groups: u32,     // main draw groups (args 0..n_groups)
    n_sgroups: u32,    // shadow groups; pass q uses args n_groups + q * n_sgroups + sg
    pass_mask: u32,    // shadow passes rendered this frame
    n_args: u32,
    _p0: u32,
    _p1: u32,
    pass_lv: array<mat4x4f, 8>, // light view per shadow pass
    pass_box: array<vec4f, 8>,  // x = half size, y = depth range
}

struct Object {
    sphere: vec4f,     // world-space bounding sphere (all LOD levels)
    origin: vec4f,     // xyz = model origin, w = 1 / scale
    lod_a: vec4f,      // LOD switch distances 0..3
    lod_b: vec4f,      // 4..7
    groups_a: vec4u,   // main draw group per LOD level (NONE = not drawn)
    groups_b: vec4u,
    sgroups_a: vec4u,  // shadow group per LOD level (NONE = no shadow)
    sgroups_b: vec4u,
    info: vec4u,       // x = meshes | distances << 8 | cull_beyond << 16 | has_lod << 17 | layer << 24
                       // y = 1 main view, 2 casts (0 = free slot)
}

struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> objects: array<Object>;
@group(0) @binding(2) var<storage, read_write> args: array<DrawArgs>;
@group(0) @binding(3) var<storage, read_write> ids: array<u32>;
// 0 frustum-culled, 1 occluded, 2 visible (main), 3 shadow instances, 4..7 LOD level 0..3+, 8 LOD-culled
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>, 12>;
@group(0) @binding(5) var hiz: texture_2d<f32>;
// nearest distance of a visible instance per main group (f32 bits; draw order for the next frame)
@group(0) @binding(6) var<storage, read_write> near: array<atomic<u32>>;

fn pick_u(a: vec4u, b: vec4u, k: u32) -> u32 {
    if (k < 4u) { return a[k]; }
    return b[k - 4u];
}

fn pick_f(a: vec4f, b: vec4f, k: u32) -> f32 {
    if (k < 4u) { return a[k]; }
    return b[k - 4u];
}

fn occluded(c: vec3f, r: f32) -> bool {
    var mn = vec3f(1e9);
    var mx = vec3f(-1e9);
    for (var k = 0u; k < 8u; k++) {
        let o = vec3f(select(-r, r, (k & 1u) != 0u), select(-r, r, (k & 2u) != 0u), select(-r, r, (k & 4u) != 0u));
        let cp = p.prev_view_proj * vec4f(c + o, 1.0);
        if (cp.w <= 1e-4) { return false; } // crosses the camera plane: keep
        let n = cp.xyz / cp.w;
        mn = min(mn, n);
        mx = max(mx, n);
    }
    if (mn.z <= 0.0) { return false; }
    let a = clamp(vec2f(mn.x, -mx.y) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
    let b = clamp(vec2f(mx.x, -mn.y) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
    let size = (b - a) * p.hiz_size;
    let level = min(u32(ceil(log2(max(max(size.x, size.y), 1.0)))), p.hiz_mips - 1u);
    let dims = vec2i(textureDimensions(hiz, level));
    let t0 = clamp(vec2i(a * vec2f(dims)), vec2i(0), dims - 1);
    let t1 = clamp(vec2i(b * vec2f(dims)), vec2i(0), dims - 1);
    let d = max(
        max(textureLoad(hiz, t0, level).r, textureLoad(hiz, vec2i(t1.x, t0.y), level).r),
        max(textureLoad(hiz, vec2i(t0.x, t1.y), level).r, textureLoad(hiz, t1, level).r),
    );
    return mn.z > d + 1e-5;
}

@compute @workgroup_size(64)
fn reset(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i < p.n_args) { atomicStore(&args[i].instance_count, 0u); }
    if (i < p.n_groups) { atomicStore(&near[i], NONE); }
    if (i < 12u) { atomicStore(&counters[i], 0u); }
}

fn append(a: u32, slot: u32) {
    let k = atomicAdd(&args[a].instance_count, 1u);
    ids[args[a].first_instance + k] = slot;
}

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= p.count) { return; }
    let o = objects[i];
    let flags = o.info.y;
    if (flags == 0u) { return; }
    let layer = (o.info.x >> 24u) & 31u;
    if (((p.layers >> layer) & 1u) == 0u) { return; }

    // LOD level (Lod::pick): first switch distance >= d, beyond the last one the last mesh (or culled)
    let n = o.info.x & 0xffu;
    let nd = (o.info.x >> 8u) & 0xffu;
    let has_lod = ((o.info.x >> 17u) & 1u) == 1u;
    var level = NONE;
    var found = false;
    let d = length(o.origin.xyz - p.cam.xyz) * o.origin.w * p.cam.w;
    for (var k = 0u; k < nd; k++) {
        if (d <= pick_f(o.lod_a, o.lod_b, k)) {
            found = true;
            if (k < n) { level = k; }
            break;
        }
    }
    if (!found && ((o.info.x >> 16u) & 1u) == 0u && n > 0u) {
        level = n - 1u;
    }
    if (level == NONE) {
        if (has_lod) { atomicAdd(&counters[8], 1u); }
        return;
    }
    if (has_lod) { atomicAdd(&counters[4u + min(level, 3u)], 1u); }

    let c = o.sphere.xyz;
    let r = o.sphere.w;
    if ((flags & 1u) != 0u) {
        let g = pick_u(o.groups_a, o.groups_b, level);
        if (g != NONE) {
            var keep = true;
            if ((p.flags & 1u) != 0u) {
                for (var k = 0u; k < 6u; k++) {
                    if (dot(p.planes[k].xyz, c) + p.planes[k].w < -r) {
                        keep = false;
                        break;
                    }
                }
                if (!keep) { atomicAdd(&counters[0], 1u); }
            }
            if (keep && (p.flags & 2u) != 0u && occluded(c, r)) {
                keep = false;
                atomicAdd(&counters[1], 1u);
            }
            if (keep) {
                append(g, i);
                atomicAdd(&counters[2], 1u);
                atomicMin(&near[g], bitcast<u32>(max(length(c - p.cam.xyz) - r, 0.0)));
            }
        }
    }
    if ((flags & 2u) != 0u && p.pass_mask != 0u) {
        let sg = pick_u(o.sgroups_a, o.sgroups_b, level);
        if (sg != NONE) {
            for (var q = 0u; q < 8u; q++) {
                if (((p.pass_mask >> q) & 1u) == 0u) { continue; }
                let b = p.pass_box[q];
                let lp = p.pass_lv[q] * vec4f(c, 1.0);
                if (abs(lp.x) <= b.x + r && abs(lp.y) <= b.x + r && -lp.z >= -r && -lp.z <= b.y + r) {
                    append(p.n_groups + q * p.n_sgroups + sg, i);
                    atomicAdd(&counters[3], 1u);
                }
            }
        }
    }
}

// GPU culling (compute). One thread per instance candidate:
//   1. frustum test (bounding sphere vs 6 planes of the current camera)
//   2. occlusion test against the Hi-Z pyramid of the *previous* frame (conservative; objects
//      revealed by fast camera moves can pop in one frame late)
//   3. survivors append their instance record slot to their batch's slice of `visible` and bump the
//      batch's indirect draw `instance_count` -> the render pass issues drawIndexedIndirect.
// Standalone shader (not prefixed with common.wgsl).

struct CullParams {
    planes: array<vec4f, 6>,
    prev_view_proj: mat4x4f,
    hiz_size: vec2f,   // texel size of Hi-Z mip 0
    count: u32,        // number of candidates
    flags: u32,        // bit 0 = frustum, bit 1 = occlusion
    hiz_mips: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Item {
    center_radius: vec4f, // world-space bounding sphere
    batch: u32,
    inst: u32, // instance record slot
    _b: u32,
    _c: u32,
}

struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> p: CullParams;
@group(0) @binding(1) var<storage, read> items: array<Item>;
@group(0) @binding(2) var<storage, read_write> args: array<DrawArgs>;
@group(0) @binding(3) var<storage, read_write> visible: array<u32>;
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>, 4>; // frustum, occluded, visible
@group(0) @binding(5) var hiz: texture_2d<f32>;

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
fn cull(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= p.count) { return; }
    let it = items[i];
    let c = it.center_radius.xyz;
    let r = it.center_radius.w;
    if ((p.flags & 1u) != 0u) {
        for (var k = 0u; k < 6u; k++) {
            if (dot(p.planes[k].xyz, c) + p.planes[k].w < -r) {
                atomicAdd(&counters[0], 1u);
                return;
            }
        }
    }
    if ((p.flags & 2u) != 0u && occluded(c, r)) {
        atomicAdd(&counters[1], 1u);
        return;
    }
    let slot = atomicAdd(&args[it.batch].instance_count, 1u);
    visible[args[it.batch].first_instance + slot] = it.inst;
    atomicAdd(&counters[2], 1u);
}

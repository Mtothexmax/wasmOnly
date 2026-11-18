// Hi-Z pyramid (max depth) for occlusion culling. Standalone shader.
// `init`: 4x MSAA depth (full res) -> mip 0 at half resolution.
// `down`: mip n-1 -> mip n.
// Conservative for every size: texel i of a level of size n stands for the uv range [i/n, (i+1)/n]
// (that is how cull.wgsl looks it up), so it takes the max over every source texel that touches
// this range. WebGPU mip sizes are floor(size / 2): an odd source gives up to 3x3 texels, and
// without this the edge rows/columns were dropped and the uv mapping drifted towards the
// bottom-right (visible objects near the right/bottom screen edge got culled).

// Source texels [lo, hi] covering the uv range of destination texel `i` (dst size n, src size m).
fn src_range(i: vec2u, n: vec2u, m: vec2u) -> array<vec2i, 2> {
    let lo = (i * m) / n;
    let hi = ((i + 1u) * m + n - 1u) / n - 1u;
    return array<vec2i, 2>(vec2i(lo), min(vec2i(hi), vec2i(m) - 1));
}

@group(0) @binding(0) var depth_ms: texture_depth_multisampled_2d;
@group(0) @binding(1) var out0: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) id: vec3u) {
    let out_dims = textureDimensions(out0);
    if (id.x >= out_dims.x || id.y >= out_dims.y) { return; }
    let r = src_range(id.xy, out_dims, textureDimensions(depth_ms));
    var m = 0.0;
    for (var y = r[0].y; y <= r[1].y; y++) {
        for (var x = r[0].x; x <= r[1].x; x++) {
            for (var s = 0; s < 4; s++) {
                m = max(m, textureLoad(depth_ms, vec2i(x, y), s));
            }
        }
    }
    textureStore(out0, vec2i(id.xy), vec4f(m, 0.0, 0.0, 0.0));
}

@group(0) @binding(2) var src_level: texture_2d<f32>;
@group(0) @binding(3) var dst_level: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) id: vec3u) {
    let out_dims = textureDimensions(dst_level);
    if (id.x >= out_dims.x || id.y >= out_dims.y) { return; }
    let r = src_range(id.xy, out_dims, textureDimensions(src_level));
    var m = 0.0;
    for (var y = r[0].y; y <= r[1].y; y++) {
        for (var x = r[0].x; x <= r[1].x; x++) {
            m = max(m, textureLoad(src_level, vec2i(x, y), 0).r);
        }
    }
    textureStore(dst_level, vec2i(id.xy), vec4f(m, 0.0, 0.0, 0.0));
}

// Hi-Z pyramid (max depth) for occlusion culling. Standalone shader.
// `init`: 4x MSAA depth (full res) -> mip 0 at half resolution (max over 2x2 pixels x 4 samples).
// `down`: mip n-1 -> mip n (max over 2x2, edge texels clamped).

@group(0) @binding(0) var depth_ms: texture_depth_multisampled_2d;
@group(0) @binding(1) var out0: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) id: vec3u) {
    let out_dims = textureDimensions(out0);
    if (id.x >= out_dims.x || id.y >= out_dims.y) { return; }
    let src = vec2i(textureDimensions(depth_ms)) - 1;
    var m = 0.0;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let p = min(vec2i(id.xy) * 2 + vec2i(x, y), src);
            for (var s = 0; s < 4; s++) {
                m = max(m, textureLoad(depth_ms, p, s));
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
    let src = vec2i(textureDimensions(src_level)) - 1;
    let b = vec2i(id.xy) * 2;
    let m = max(
        max(textureLoad(src_level, min(b, src), 0).r, textureLoad(src_level, min(b + vec2i(1, 0), src), 0).r),
        max(textureLoad(src_level, min(b + vec2i(0, 1), src), 0).r, textureLoad(src_level, min(b + vec2i(1, 1), src), 0).r),
    );
    textureStore(dst_level, vec2i(id.xy), vec4f(m, 0.0, 0.0, 0.0));
}

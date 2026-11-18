// Mip generation: one fullscreen triangle per level, linear-filtered sample of the level above.
// (Not prefixed with common.wgsl; group 0 here is the texture+sampler layout.)

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2f(p.x, 1.0 - p.y);
    return o;
}

@fragment
fn fs(i: VsOut) -> @location(0) vec4f {
    return textureSampleLevel(src, samp, i.uv, 0.0);
}

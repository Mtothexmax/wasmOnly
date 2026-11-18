// Depth-only passes from the sun's point of view.

@vertex
fn vs_shadow(@location(0) pos: vec3f, @builtin(instance_index) inst: u32) -> @builtin(position) vec4f {
    return g.light_view_proj * instances[inst].model * vec4f(pos, 1.0);
}

// Alpha-tested casters (AlphaMode::Mask): cut-out shadows for leaves, grass cards, fences.
@group(1) @binding(0) var base_tex: texture_2d<f32>;
@group(1) @binding(1) var base_sampler: sampler;

struct MaskOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
    @location(1) @interpolate(flat) alpha_cut: vec2f, // x = base alpha, y = cutoff
}

@vertex
fn vs_shadow_mask(@location(0) pos: vec3f, @location(2) uv: vec2f, @builtin(instance_index) inst: u32) -> MaskOut {
    let it = instances[inst];
    var o: MaskOut;
    o.clip = g.light_view_proj * it.model * vec4f(pos, 1.0);
    o.uv = uv;
    o.alpha_cut = vec2f(it.base_color.a, it.params.w);
    return o;
}

@fragment
fn fs_shadow_mask(i: MaskOut) {
    if (textureSample(base_tex, base_sampler, i.uv).a * i.alpha_cut.x < i.alpha_cut.y) {
        discard;
    }
}

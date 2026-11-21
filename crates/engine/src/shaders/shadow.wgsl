// Depth-only passes from the sun's point of view, one per shadow cascade.

@group(0) @binding(2) var<uniform> shadow_vp: mat4x4f;
// caster k of a draw -> its instance record slot
@group(0) @binding(4) var<storage, read> visible_ids: array<u32>;

@vertex
fn vs_shadow(@location(0) pos: vec3f, @builtin(instance_index) inst: u32) -> @builtin(position) vec4f {
    return shadow_vp * instances[visible_ids[inst]].model * vec4f(pos, 1.0);
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
    let it = instances[visible_ids[inst]];
    var o: MaskOut;
    o.clip = shadow_vp * it.model * vec4f(pos, 1.0);
    o.uv = uv;
    if (it.normal[2].w > 0.5) { // impostor card: faces the light (its silhouette from the sun's azimuth)
        let iv = impostor_vertex(it.model, pos, uv, it.normal[2].w, vec3f(shadow_vp[0][2], shadow_vp[1][2], shadow_vp[2][2]));
        o.clip = shadow_vp * iv.world;
        o.uv = iv.uv;
    }
    o.alpha_cut = vec2f(it.base_color.a, it.params.w);
    return o;
}

@fragment
fn fs_shadow_mask(i: MaskOut) {
    if (textureSample(base_tex, base_sampler, i.uv).a * i.alpha_cut.x < i.alpha_cut.y) {
        discard;
    }
}

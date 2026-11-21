// Impostor bake (impostor.rs): renders a model's primitives from N azimuths into an atlas with an
// orthographic camera per view. Target 0 = albedo (texture * base colour * vertex colour, sRGB
// target), target 1 = normal in the card's tangent space as pbr.wgsl's derivative TBN reconstructs
// it (x = -right, y = up, z toward the viewer).
// `dilate` then copies the bake into the final mip 0 and fills empty texels with the nearest
// covered colour, so mips and bilinear filtering don't bleed black into the silhouette.
// Standalone shader (not prefixed with common.wgsl).

struct Draw {
    mvp: mat4x4f,
    nrm: mat4x4f,   // node model -> view rotation, inverse-transposed
    color: vec4f,   // linear base colour
    params: vec4f,  // x = alpha cutoff (< 0 = opaque)
}

@group(0) @binding(0) var<uniform> d: Draw;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) n: vec3f,
    @location(1) uv: vec2f,
    @location(2) col: vec4f,
}

@vertex
fn vs_bake(@location(0) pos: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f, @location(3) color: vec4f) -> VOut {
    var o: VOut;
    o.clip = d.mvp * vec4f(pos, 1.0);
    o.n = (d.nrm * vec4f(normal, 0.0)).xyz;
    o.uv = uv;
    let c = color.rgb;
    o.col = vec4f(c * (c * (c * 0.305306011 + 0.682171111) + 0.012522878), color.a);
    return o;
}

struct FOut {
    @location(0) albedo: vec4f,
    @location(1) normal: vec4f,
}

@fragment
fn fs_bake(i: VOut, @builtin(front_facing) front: bool) -> FOut {
    let t = textureSample(tex, samp, i.uv);
    let a = t.a * d.color.a * i.col.a;
    if (d.params.x >= 0.0 && a < d.params.x) {
        discard;
    }
    var n = normalize(i.n);
    if (!front) {
        n = -n;
    }
    var o: FOut;
    o.albedo = vec4f(t.rgb * d.color.rgb * i.col.rgb, 1.0);
    // pbr.wgsl's cotangent_frame (WebGPU screen y points down) gives T = -right, B = +up on the card
    o.normal = vec4f(vec3f(-n.x, n.y, n.z) * 0.5 + 0.5, 1.0);
    return o;
}

// --- dilation: full-screen triangle, both targets --------------------------------------------------
@group(0) @binding(0) var src_albedo: texture_2d<f32>;
@group(0) @binding(1) var src_normal: texture_2d<f32>;

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_dilate(@builtin(position) pos: vec4f) -> FOut {
    let dims = vec2i(textureDimensions(src_albedo));
    let p = vec2i(pos.xy);
    var o: FOut;
    o.albedo = textureLoad(src_albedo, p, 0);
    o.normal = textureLoad(src_normal, p, 0);
    if (o.albedo.a > 0.5) {
        return o;
    }
    // nearest covered texel within 8 px (ring by ring); colour only, alpha stays 0
    var best = 1e9;
    var acc_a = vec3f(0.0);
    var acc_n = vec3f(0.5, 0.5, 1.0);
    for (var dy = -8; dy <= 8; dy++) {
        for (var dx = -8; dx <= 8; dx++) {
            let q = clamp(p + vec2i(dx, dy), vec2i(0), dims - 1);
            let s = textureLoad(src_albedo, q, 0);
            let dd = f32(dx * dx + dy * dy);
            if (s.a > 0.5 && dd < best) {
                best = dd;
                acc_a = s.rgb;
                acc_n = textureLoad(src_normal, q, 0).rgb;
            }
        }
    }
    o.albedo = vec4f(acc_a, 0.0);
    o.normal = vec4f(acc_n, 0.0);
    return o;
}

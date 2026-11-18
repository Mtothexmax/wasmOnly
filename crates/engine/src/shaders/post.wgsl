// Post-processing on the linear HDR scene colour (standalone, no common.wgsl):
//   prefilter  : HDR -> bloom mip 0 (quarter res), soft threshold, scene-colour pixels only (alpha mask)
//   down       : bloom mip i -> mip i+1 (dual-filter downsample, 5 taps)
//   up         : bloom mip i -> mip i-1, added on top (dual-filter upsample, 8 taps)
//   composite  : HDR + bloom, depth darkening (unsharp-masked depth), exposure, ACES, gamma
// Alpha of the scene colour is the tonemap mask: 1 = scene, 0 = UI panels / debug views (as-is).

struct Params {
    a: vec4f,
    b: vec4f,
    c: vec4f,
    d: vec4f,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> p: Params;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> VsOut {
    let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2f(xy.x, 1.0 - xy.y);
    return o;
}

fn luma(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// a = (src texel w, src texel h, threshold, knee)
@fragment
fn fs_prefilter(i: VsOut) -> @location(0) vec4f {
    let t = p.a.xy;
    var acc = vec3f(0.0);
    var wsum = 0.0;
    // 4 bilinear taps, Karis-weighted (1 / (1 + luma)) against single bright pixels flickering
    let offs = array<vec2f, 4>(vec2f(-0.5, -0.5), vec2f(0.5, -0.5), vec2f(-0.5, 0.5), vec2f(0.5, 0.5));
    for (var k = 0; k < 4; k++) {
        let s = textureSampleLevel(src, samp, i.uv + offs[k] * t * 2.0, 0.0);
        let c = s.rgb * clamp(s.a, 0.0, 1.0);
        let w = 1.0 / (1.0 + luma(c));
        acc += c * w;
        wsum += w;
    }
    let c = acc / wsum;
    let br = max(c.r, max(c.g, c.b));
    let th = p.a.z;
    let knee = max(p.a.w, 1e-4);
    var soft = clamp(br - th + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    let contrib = max(soft, br - th) / max(br, 1e-4);
    return vec4f(min(c * contrib, vec3f(64.0)), 1.0);
}

// a = (src texel w, src texel h, _, _)
@fragment
fn fs_down(i: VsOut) -> @location(0) vec4f {
    let t = p.a.xy;
    var c = textureSampleLevel(src, samp, i.uv, 0.0).rgb * 4.0;
    c += textureSampleLevel(src, samp, i.uv + vec2f(-t.x, -t.y), 0.0).rgb;
    c += textureSampleLevel(src, samp, i.uv + vec2f(t.x, -t.y), 0.0).rgb;
    c += textureSampleLevel(src, samp, i.uv + vec2f(-t.x, t.y), 0.0).rgb;
    c += textureSampleLevel(src, samp, i.uv + vec2f(t.x, t.y), 0.0).rgb;
    return vec4f(c / 8.0, 1.0);
}

// a = (src texel w, src texel h, weight, _); blended additively onto the larger mip
@fragment
fn fs_up(i: VsOut) -> @location(0) vec4f {
    let t = p.a.xy;
    var c = vec3f(0.0);
    c += textureSampleLevel(src, samp, i.uv + vec2f(-2.0 * t.x, 0.0), 0.0).rgb;
    c += textureSampleLevel(src, samp, i.uv + vec2f(2.0 * t.x, 0.0), 0.0).rgb;
    c += textureSampleLevel(src, samp, i.uv + vec2f(0.0, -2.0 * t.y), 0.0).rgb;
    c += textureSampleLevel(src, samp, i.uv + vec2f(0.0, 2.0 * t.y), 0.0).rgb;
    c += textureSampleLevel(src, samp, i.uv + vec2f(-t.x, -t.y), 0.0).rgb * 2.0;
    c += textureSampleLevel(src, samp, i.uv + vec2f(t.x, -t.y), 0.0).rgb * 2.0;
    c += textureSampleLevel(src, samp, i.uv + vec2f(-t.x, t.y), 0.0).rgb * 2.0;
    c += textureSampleLevel(src, samp, i.uv + vec2f(t.x, t.y), 0.0).rgb * 2.0;
    return vec4f(c / 12.0 * p.a.z, 1.0);
}

// --- composite ---------------------------------------------------------------------------------
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;
@group(0) @binding(4) var depth_tex: texture_depth_multisampled_2d;

// Output encoding: true = write gamma-encoded values to a non-sRGB target (canvas); false = write
// linear values into an sRGB target (render-to-texture), the hardware encodes.
override GAMMA_OUT: bool = true;

fn aces(x: vec3f) -> vec3f {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

fn lin_depth(z: f32) -> f32 {
    let n = p.b.x;
    let f = p.b.y;
    return n * f / (f - z * (f - n));
}

// a = (exposure, bloom intensity, darkening strength, darkening radius in px at 1 m)
// b = (near, far, bloom on, darkening range in m)
// c = (vignette, saturation, transparent background (alpha = scene mask), _)
@fragment
fn fs_composite(i: VsOut) -> @location(0) vec4f {
    // the scene may be rendered smaller than the canvas (render_scale): sample by UV, read depth at
    // the matching scene pixel
    let s = textureSampleLevel(src, samp, i.uv, 0.0);
    let px = vec2i(i.uv * vec2f(textureDimensions(depth_tex)));
    let mask = clamp(s.a, 0.0, 1.0);
    var c = s.rgb;

    // Depth darkening (unsharp mask of the depth buffer, Luft et al. 2006): pixels that lie behind
    // their surroundings (creases, contact areas, under overhangs) get darker. 8 taps on a ring.
    let strength = p.a.z;
    if (strength > 0.0 && mask > 0.0) {
        let z = textureLoad(depth_tex, px, 0);
        if (z < 1.0) {
            let d = lin_depth(z);
            let r = clamp(p.a.w / d, 2.0, 48.0);
            let dims = vec2i(textureDimensions(depth_tex));
            var sum = 0.0;
            var n = 0.0;
            // rotated per pixel (interleaved gradient noise) so the ring pattern does not show
            let ign = fract(52.9829189 * fract(dot(vec2f(px), vec2f(0.06711056, 0.00583715))));
            for (var k = 0; k < 8; k++) {
                let ang = (f32(k) + ign) * 0.785398;
                let rr = r * (0.5 + 0.5 * fract(f32(k) * 0.618 + ign));
                let q = clamp(px + vec2i(vec2f(cos(ang), sin(ang)) * rr), vec2i(0), dims - 1);
                let zq = textureLoad(depth_tex, q, 0);
                let dq = select(lin_depth(zq), d, zq >= 1.0);
                // ignore neighbours far in front (silhouettes against the background)
                let w = 1.0 - smoothstep(p.b.w * 0.5, p.b.w, d - dq);
                sum += dq * w + d * (1.0 - w);
                n += 1.0;
            }
            let blurred = sum / n;
            let behind = max(d - blurred, 0.0);
            let dark = clamp(behind * strength / max(d * 0.05, 0.15), 0.0, 0.6);
            c *= 1.0 - dark * mask;
        }
    }

    if (p.b.z > 0.5) {
        let bl = textureSampleLevel(bloom_tex, samp, i.uv, 0.0).rgb;
        c += bl * p.a.y * mask;
    }
    var mapped = aces(c * p.a.x);
    // saturation
    let l = luma(mapped);
    mapped = clamp(mix(vec3f(l), mapped, p.c.y), vec3f(0.0), vec3f(1.0));
    // vignette
    let v = i.uv - 0.5;
    mapped *= 1.0 - p.c.x * dot(v, v) * 1.6;
    var outc = mix(c, mapped, mask);
    if (GAMMA_OUT) {
        outc = pow(max(outc, vec3f(0.0)), vec3f(1.0 / 2.2));
    }
    return vec4f(outc, select(1.0, mask, p.c.z > 0.5));
}

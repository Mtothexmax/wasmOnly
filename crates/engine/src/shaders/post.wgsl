// Post-processing on the linear HDR scene colour (standalone, no common.wgsl):
//   prefilter  : HDR -> bloom mip 0 (quarter res), soft threshold, scene-colour pixels only (alpha mask)
//   down       : bloom mip i -> mip i+1 (dual-filter downsample, 5 taps)
//   up         : bloom mip i -> mip i-1, added on top (dual-filter upsample, 8 taps)
//   composite  : HDR + bloom, depth darkening (unsharp-masked depth), lens flare, exposure, ACES, gamma
// Alpha of the scene colour is the tonemap mask: 1 = scene, 0 = UI panels / debug views (as-is).

struct Params {
    a: vec4f,
    b: vec4f,
    c: vec4f,
    d: vec4f,
    e: vec4f,
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

// NaN / inf / absurd HDR values never reach bloom or the tonemapper (one bad pixel = a black dot with
// a white halo). Comparisons with NaN are false, so NaN lanes take the 0 branch.
fn sanitize(c: vec3f) -> vec3f {
    let ok = (c >= vec3f(0.0)) & (c <= vec3f(6.0e4));
    return select(vec3f(0.0), min(c, vec3f(2.0e3)), ok);
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
        let c = sanitize(s.rgb) * clamp(s.a, 0.0, 1.0);
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

// --- Lens flare ---------------------------------------------------------------------------------
// d = (sun uv x, sun uv y, strength (0 = off), sun disk radius in uv height units)
// e = (sun colour rgb, expected HDR luminance of the visible sun disk)

// How much of the sun disk is visible: 6 taps inside the disk in the HDR scene, compared with the
// brightness of an unobstructed disk. Geometry, clouds and the horizon all hide it.
fn sun_visibility(aspect: f32) -> f32 {
    let r = p.d.w * 0.7;
    var vis = 0.0;
    for (var k = 0; k < 6; k++) {
        let a = f32(k) * 2.39996;
        let rr = r * sqrt((f32(k) + 0.5) / 6.0);
        let uv = p.d.xy + vec2f(cos(a) * rr / aspect, sin(a) * rr);
        if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { continue; }
        let l = luma(sanitize(textureSampleLevel(src, samp, uv, 0.0).rgb));
        vis += clamp((l / max(p.e.w, 1e-3) - 0.25) * 2.0, 0.0, 1.0);
    }
    return vis / 6.0;
}

fn ghost(q: vec2f, center: vec2f, size: f32, ring: bool) -> f32 {
    let d = length(q - center) / size;
    if (d >= 1.0) { return 0.0; } // most pixels: no atan2
    if (ring) {
        return smoothstep(0.75, 0.93, d) * (1.0 - smoothstep(0.93, 1.0, d));
    }
    // hexagonal aperture ghost (6 blades), soft edge, brighter rim
    let v = q - center;
    let ang = atan2(v.y, v.x);
    let seg = 3.14159265 / 3.0;
    let x = ang + seg * 0.5;
    let hex = cos(seg * 0.5) / cos(x - floor(x / seg) * seg - seg * 0.5);
    let dh = d / hex;
    return (1.0 - smoothstep(0.85, 1.0, dh)) * (0.55 + 0.45 * smoothstep(0.4, 0.95, dh));
}

// Classic game lens flare: glow + starburst at the sun, a faint halo ring, and a chain of coloured
// hexagon/ring ghosts along the line from the sun through the screen centre.
fn lens_flare(uv: vec2f) -> vec3f {
    let aspect = p.c.w;
    let vis = sun_visibility(aspect);
    if (vis <= 0.0) { return vec3f(0.0); }
    let k = p.d.z * vis;
    let sun = vec2f(p.d.x * aspect, p.d.y);
    let q = vec2f(uv.x * aspect, uv.y);
    let ctr = vec2f(0.5 * aspect, 0.5);
    let col = p.e.rgb;
    var c = vec3f(0.0);
    // glow + 6-ray starburst around the sun
    let v = q - sun;
    let d = length(v);
    let ang = atan2(v.y, v.x);
    c += col * (0.045 / (d * 9.0 + 0.04)) * 0.5;
    let fall = exp(-d * 5.0);
    if (fall > 0.004) {
        let rays = pow(abs(cos(ang * 3.0 + 0.3)), 60.0) + 0.6 * pow(abs(cos(ang * 3.0 + 1.35)), 90.0);
        c += col * rays * fall * 1.4;
    }
    // halo ring with a faint rainbow edge
    let hr = abs(d - 0.32);
    c += vec3f(1.0, 0.75, 0.55) * col * exp(-hr * hr * 9000.0) * 0.05;
    c += vec3f(0.45, 0.6, 1.0) * col * exp(-(hr - 0.012) * (hr - 0.012) * 9000.0) * 0.04;
    // ghosts: position t along sun -> centre -> beyond, size, colour, ring?
    let axis = ctr - sun;
    c += vec3f(1.0, 0.55, 0.2) * ghost(q, sun + axis * 0.45, 0.035, false) * 0.29;
    c += vec3f(0.35, 1.0, 0.45) * ghost(q, sun + axis * 0.72, 0.022, false) * 0.25;
    c += vec3f(0.4, 0.55, 1.0) * ghost(q, sun + axis * 1.12, 0.06, false) * 0.20;
    c += vec3f(1.0, 0.85, 0.4) * ghost(q, sun + axis * 1.3, 0.016, false) * 0.36;
    c += vec3f(0.7, 0.4, 1.0) * ghost(q, sun + axis * 1.55, 0.09, true) * 0.22;
    c += vec3f(0.3, 0.9, 1.0) * ghost(q, sun + axis * 1.8, 0.03, false) * 0.25;
    c += vec3f(1.0, 0.45, 0.35) * ghost(q, sun + axis * 2.2, 0.12, false) * 0.11;
    return c * k;
}

// a = (exposure, bloom intensity, darkening strength, darkening radius in px at 1 m)
// b = (near, far, bloom on, darkening range in m)
// c = (vignette, saturation, transparent background (alpha = scene mask), aspect w/h)
@fragment
fn fs_composite(i: VsOut) -> @location(0) vec4f {
    // the scene may be rendered smaller than the canvas (render_scale): sample by UV, read depth at
    // the matching scene pixel
    let raw = textureSampleLevel(src, samp, i.uv, 0.0);
    let s = vec4f(sanitize(raw.rgb), select(0.0, raw.a, raw.a == raw.a));
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
    if (p.d.z > 0.0) {
        c += lens_flare(i.uv);
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

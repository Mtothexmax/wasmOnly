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
    // grading: g[0] = (temperature, tint, contrast, pivot), g[1] = (lift rgb, split balance),
    // g[2] = (gamma rgb, grain), g[3] = (gain rgb, grain size px), g[4] = (split shadows rgb, vignette
    // radius), g[5] = (split highlights rgb, vignette softness), g[6] = (vignette colour rgb,
    // roundness), g[7] = (frame, grading on, grade saturation, -)
    g: array<vec4f, 8>,
}

fn soft_light(a: vec3f, b: vec3f) -> vec3f { // Pegtop: b = 0.5 leaves a unchanged
    return (1.0 - 2.0 * b) * a * a + 2.0 * b * a;
}

// Colour grading in display space (0..1), see post::Grade.
fn grade(c0: vec3f) -> vec3f {
    var c = c0;
    // white balance: warm/cool and green/magenta gains, luminance kept
    let t = p.g[0].x;
    let tn = p.g[0].y;
    let wb = vec3f(1.0 + 0.2 * t, 1.0 - 0.2 * tn, 1.0 - 0.2 * t);
    c *= wb / max(luma(wb), 1e-3);
    // lift / gamma / gain
    c = p.g[3].rgb * (c + p.g[1].rgb * (1.0 - c));
    c = pow(max(c, vec3f(0.0)), 1.0 / p.g[2].rgb);
    // contrast around the pivot
    c = (c - p.g[0].w) * p.g[0].z + p.g[0].w;
    c = clamp(c, vec3f(0.0), vec3f(1.0));
    // split toning: shadows / highlights colour, soft light
    let w = smoothstep(0.0, 1.0, clamp(luma(c) + p.g[1].w * 0.5, 0.0, 1.0));
    c = soft_light(c, mix(p.g[4].rgb, p.g[5].rgb, w));
    return c;
}

// pcg3d (Jarzynski & Olano 2020): every (cell, frame) gets independent bits, no drifting pattern
fn pcg3d(q: vec3u) -> vec3u {
    var v = q * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

// film grain sample for a cell and frame: sum of two uniforms (triangular, -1..1), scaled to the RMS
// of the old uniform -0.5..0.5 grain so existing `grain` values keep their strength
fn grain_noise(cell: vec2f, frame: f32) -> f32 {
    let c = vec2i(cell);
    let h = pcg3d(vec3u(bitcast<u32>(c.x), bitcast<u32>(c.y), u32(frame)));
    let u = vec2f(h.xy >> vec2u(8u)) * (1.0 / 16777216.0);
    return (u.x + u.y - 1.0) * 0.70710678;
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
// Scene rendered without MSAA (RenderConfig::msaa = 1): FXAA on the HDR input.
override FXAA: bool = false;

// Luma for edge detection on HDR input: compressed so bright highlights do not dominate.
fn aa_luma(c: vec3f) -> f32 {
    return dot(c / (1.0 + c), vec3f(0.299, 0.587, 0.114));
}

fn tap(uv: vec2f) -> vec3f {
    return textureSampleLevel(src, samp, uv, 0.0).rgb;
}

// FXAA (Lottes, the 'console' variant): 4 diagonal neighbours find edge contrast and direction,
// then 2 or 4 taps along the edge. Pixels without contrast return the centre sample unchanged.
fn fxaa(uv: vec2f, m: vec4f) -> vec4f {
    let t = 1.0 / vec2f(textureDimensions(src));
    let l_nw = aa_luma(tap(uv + vec2f(-t.x, -t.y)));
    let l_ne = aa_luma(tap(uv + vec2f(t.x, -t.y)));
    let l_sw = aa_luma(tap(uv + vec2f(-t.x, t.y)));
    let l_se = aa_luma(tap(uv + vec2f(t.x, t.y)));
    let l_m = aa_luma(m.rgb);
    let l_min = min(l_m, min(min(l_nw, l_ne), min(l_sw, l_se)));
    let l_max = max(l_m, max(max(l_nw, l_ne), max(l_sw, l_se)));
    if (l_max - l_min < max(0.0312, l_max * 0.125)) {
        return m;
    }
    var dir = vec2f(-((l_nw + l_ne) - (l_sw + l_se)), (l_nw + l_sw) - (l_ne + l_se));
    let reduce = max((l_nw + l_ne + l_sw + l_se) * 0.03125, 1.0 / 128.0);
    let rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcp, vec2f(-8.0), vec2f(8.0)) * t;
    let a = 0.5 * (tap(uv + dir * (1.0 / 3.0 - 0.5)) + tap(uv + dir * (2.0 / 3.0 - 0.5)));
    let b = a * 0.5 + 0.25 * (tap(uv - dir * 0.5) + tap(uv + dir * 0.5));
    let l_b = aa_luma(b);
    if (l_b < l_min || l_b > l_max) {
        return vec4f(a, m.a);
    }
    return vec4f(b, m.a);
}

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
fn lens_flare(uv: vec2f, vis: f32) -> vec3f {
    let aspect = p.c.w;
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

struct CompOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    // how much of the sun disk is visible: the same for every pixel, so it is sampled here (three
    // vertices) instead of per pixel
    @location(1) @interpolate(flat) sun_vis: f32,
}

@vertex
fn vs_composite(@builtin(vertex_index) i: u32) -> CompOut {
    let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var o: CompOut;
    o.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2f(xy.x, 1.0 - xy.y);
    o.sun_vis = 0.0;
    if (p.d.z > 0.0) { o.sun_vis = sun_visibility(p.c.w); }
    return o;
}

// 8 ring directions for depth darkening (45 degree steps); rotated per pixel by one angle
const RING = array<vec2f, 8>(
    vec2f(1.0, 0.0), vec2f(0.70710678, 0.70710678), vec2f(0.0, 1.0), vec2f(-0.70710678, 0.70710678),
    vec2f(-1.0, 0.0), vec2f(-0.70710678, -0.70710678), vec2f(0.0, -1.0), vec2f(0.70710678, -0.70710678));

// a = (exposure, bloom intensity, darkening strength, darkening radius in px at 1 m)
// b = (near, far, bloom on, darkening range in m)
// c = (vignette, saturation, transparent background (alpha = scene mask), aspect w/h)
@fragment
fn fs_composite(i: CompOut) -> @location(0) vec4f {
    // the scene may be rendered smaller than the canvas (render_scale): sample by UV, read depth at
    // the matching scene pixel
    var raw = textureSampleLevel(src, samp, i.uv, 0.0);
    if (FXAA) {
        raw = fxaa(i.uv, raw);
    }
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
            // direction k = RING[k] rotated by ign * 45 degrees (one sin/cos per pixel instead of 8)
            let rot = vec2f(cos(ign * 0.785398), sin(ign * 0.785398));
            for (var k = 0; k < 8; k++) {
                let dir = vec2f(RING[k].x * rot.x - RING[k].y * rot.y, RING[k].x * rot.y + RING[k].y * rot.x);
                let rr = r * (0.5 + 0.5 * fract(f32(k) * 0.618 + ign));
                let q = clamp(px + vec2i(dir * rr), vec2i(0), dims - 1);
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
        c += lens_flare(i.uv, i.sun_vis);
    }
    var mapped = aces(c * p.a.x);
    var sat = p.c.y;
    if (p.g[7].y > 0.5) {
        mapped = grade(mapped);
        sat *= p.g[7].z;
    }
    // saturation
    let l = luma(mapped);
    mapped = clamp(mix(vec3f(l), mapped, sat), vec3f(0.0), vec3f(1.0));
    // vignette: classic radial darkening, or (radius > 0) a masking vignette in a colour
    let v = i.uv - 0.5;
    if (p.g[4].w > 0.0) {
        let q = vec2f(v.x * mix(1.0, p.c.w, p.g[6].w), v.y) * 2.0; // roundness 1: aspect-correct circle
        let m = smoothstep(p.g[4].w, p.g[4].w + p.g[5].w, length(q));
        mapped = mix(mapped, p.g[6].rgb, m * clamp(p.c.x, 0.0, 1.0));
    } else {
        mapped *= 1.0 - p.c.x * dot(v, v) * 1.6;
    }
    // film grain: per frame, per grain cell, stronger in the darks
    if (p.g[2].w > 0.0) {
        let cell = floor(i.pos.xy / p.g[3].w);
        let n = grain_noise(cell, p.g[7].x);
        let lw = mix(1.0, 0.35, luma(mapped));
        mapped = clamp(mapped + n * p.g[2].w * 0.25 * lw, vec3f(0.0), vec3f(1.0));
    }
    var outc = mix(c, mapped, mask);
    if (GAMMA_OUT) {
        outc = pow(max(outc, vec3f(0.0)), vec3f(1.0 / 2.2));
    }
    return vec4f(outc, select(1.0, mask, p.c.z > 0.5));
}

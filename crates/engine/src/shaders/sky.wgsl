// Sky: fullscreen triangle at the far plane, drawn at the end of the opaque pass with depth test
// less-equal, so it only shades pixels no geometry covered. Colour from common.wgsl's sky_full
// (atmosphere or gradient, skybox, stars, moon, clouds, sun disk).

struct SkyOut {
    @builtin(position) pos: vec4f,
    @location(0) ndc: vec2f,
}

@vertex
fn vs_sky(@builtin(vertex_index) i: u32) -> SkyOut {
    let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var o: SkyOut;
    o.pos = vec4f(xy, 1.0, 1.0);
    o.ndc = xy;
    return o;
}

@fragment
fn fs_sky(i: SkyOut) -> @location(0) vec4f {
    let far = g.inv_view_proj * vec4f(i.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - g.camera_pos.xyz);
    var c = sky_full(dir, true);
    if (g.fog.x > 0.0 && g.fog_color.w > 0.5) {
        c = mix(c, g.fog_color.rgb, g.fog.w);
    }
    return ldr_opaque(vec4f(c, 1.0), i.pos.xy);
}

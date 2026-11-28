// Example surface shader (Renderer::add_surface_shader, Material::surface): animated stripes along u with
// a grooved base normal under a smooth clear coat. Registered by the demo's `surface_demo` command.
// params0: x = stripes per uv unit, y = scroll speed (stripes / s), z = groove depth (0..1)
// params1: rgb = colour of the second stripe
fn stripes_wave(s: SurfaceIn) -> f32 {
    return fract(s.uv.x * s.params0.x - s.time * s.params0.y);
}

fn surface(s: SurfaceIn) -> SurfaceOut {
    var o = surface_defaults(s);
    let f = stripes_wave(s);
    // anti-aliased stripe edge: width of one pixel in stripe units
    let w = max(length(vec2f(s.duvx.x, s.duvy.x)) * s.params0.x, 1e-4);
    let band = smoothstep(0.5 - w, 0.5 + w, f) * (1.0 - smoothstep(1.0 - w, 1.0, f));
    o.albedo = mix(o.albedo, s.params1.rgb * s.color.rgb, band);
    // grooves: the base normal tilts along +u at the stripe edges, the coat stays smooth
    let slope = (cos(f * 6.2831853) * s.params0.z) * (1.0 - smoothstep(0.1, 0.5, w));
    o.normal = normalize(s.n - s.t * slope * 0.6);
    o.coat_normal = s.n;
    o.roughness = mix(0.55, 0.25, band);
    o.clearcoat = 1.0;
    o.clearcoat_roughness = 0.05;
    return o;
}

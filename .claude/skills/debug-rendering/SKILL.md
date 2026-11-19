---
name: debug-rendering
description: Diagnose rendering problems in the WebGPU renderer - black or frozen canvas, WGSL compile errors, validation errors, wrong lighting or shadows, invisible or missing models. Use when the picture is wrong or the engine reports errors.
---

# Debugging rendering

## 1. Read the errors (they are structured)

```bash
node tools/bridge.mjs reload                                # warnings_and_errors
node tools/bridge.mjs call logs '{"level":"error"}'
node tools/bridge.mjs eval 'window.engine.fatal'            # startup failure (no WebGPU, init panic)
```

Rust panics, WebGPU validation errors and WGSL compile errors (with shader line numbers) all land
in the log at level `error` and in the red `#engine-error` overlay.

WGSL line numbers count from the top of **`common.wgsl` + the shader**: `common.wgsl` is
prepended to every shader in `crates/engine/src/shaders/`. Shared layouts (`Globals`,
`Instance`) live there and must match the Rust structs in `renderer.rs` byte for byte.

Works on desktop, nothing 3D on a phone, log says `CreateGraphicsPipelines failed` / `VK_ERROR_UNKNOWN`:
the driver rejects the generated SPIR-V (not a WGSL error). Those pipelines are skipped (overlay names
them). Bisect on the device with `node tools/gpu-probe.mjs <page.html>` (variants with stubbed functions).
Known cause (PowerVR B-Series, Android 10): a WGSL helper that samples a texture (shadow taps, fog's sky
LUT) called from *several* places in one entry point. Keep one call site per sampling helper (loop over the
taps, compute the colour in branches and call `finish` once).

## 2. Narrow it down with numbers

```bash
node tools/bridge.mjs call stats      # render.visible, culled, draw_calls, triangles
node tools/bridge.mjs call camera     # where is the camera looking?
node tools/bridge.mjs call entity '{"id":N}'   # has mesh? scale 0? under a parent far away?
node tools/bridge.mjs call assets     # model failed / pending?
```

- `visible: 0` with entities present → camera or culling (bounding sphere / scale).
- Entity exists, model `pending` forever → fetch path wrong (`assets/…` relative to `dist/`).

## Flicker / z-fighting

Still screenshots hide z-fighting. Ask the engine instead:

```bash
node tools/bridge.mjs call zfight            # {"pairs": N, "worst": [{a, b, triangle_pairs, point, screen}]}
```

Coplanar overlapping faces of different entities (opaque/masked; blended, water and vertex-displaced
meshes are skipped). Debug builds also run it automatically after scene changes and log a warning.

For everything else that flickers (water, displaced meshes, shadow acne):

```bash
node tools/bridge.mjs flicker --out <scratch>/heat.png   # regions: [{x, y, pixels, entity, depth_ties}]
node tools/bridge.mjs call pick '{"x":708,"y":292,"mesh":true}'   # what is drawn there + depth-tie partners
```

The scan pauses the game, freezes shader time, hides the HUD and uses the CPU draw path; frames
alternate draw order and the near plane (`render {reverse_draw_order, depth_jitter}`), so pixels never
move — only depth ties change colour. A clean scene gives exactly 0 flicker pixels. Typical causes: modular pieces overlapping at their ends, decals/planes placed
exactly on a surface, duplicated entities. Fix the geometry (offset ≥ 1–2 mm, different heights, inset).

## Frame rate below the display rate

```bash
node tools/bridge.mjs perf --ms 3000            # verdict + frame_ms + cpu_phases + gpu_ms per pass
node tools/bridge.mjs perf --ms 3000 --profile  # + CDP CPU profile: top self-time functions
```

Read `verdict` first. `gpu_ms` comes from timestamp queries (Chrome quantizes to ~0.1 ms): a big `opaque` means
fill rate (canvas size x MSAA x shader cost, overdraw) — `discard` in the opaque pipelines would kill early-Z, so
alpha-tested materials go through the separate mask pipelines (`ALPHA_TEST` override constant). Compare by
switching things off (`render {shadows:false}`, despawning a system) and re-running `perf`. A low `fps` with
`focus:false` in the debug Chrome is often just a background window. Logs from the user: `"kind":"perf"` lines in
`logs/debug-*.log` (debug mode, ` key) show what their machine measured.

## 3. Visual debug views

```bash
node tools/bridge.mjs call render '{"debug_view":1}'   # normals
node tools/bridge.mjs call render '{"debug_view":2}'   # shadow term (white = lit)
node tools/bridge.mjs call render '{"debug_view":0}'   # back to lit
node tools/bridge.mjs call render '{"shadows":false}'
node tools/bridge.mjs call render '{"sun_dir":[-0.4,-1,-0.3],"sun_intensity":3,"shadow_distance":160}'
node tools/bridge.mjs screenshot <scratch>/view.png
```

Reset debug settings before finishing.

## Atmosphere and post-processing

Frame: opaque pass (+ sky, drawn last at the far plane) and blend pass (+ particles) render into a 4x MSAA
`rgba16float` target; then bloom (quarter-res mip chain) and a composite pass (depth darkening, exposure, ACES,
gamma) write the canvas. Shaders output **linear HDR**; alpha of the scene colour is the tonemap mask
(1 = scene, 0 = UI panels / debug views shown as-is). Fog is applied per fragment (`apply_fog` in
`common.wgsl`), its colour is the sky towards the horizon unless `fog_color` is set.

```bash
node tools/bridge.mjs call render                                      # all settings
node tools/bridge.mjs call render '{"fog_density":0}'                  # isolate: fog off
node tools/bridge.mjs call render '{"bloom":false,"depth_darkening":0}'  # isolate post effects
node tools/bridge.mjs call stats   # render.gpu_ms.post = bloom + composite, render.particles
```

Something glows too much: bloom picks up HDR values above `bloom_threshold` (emissive > ~1.2, additive
particles). Everything washed out / too dark: `exposure`. Far objects vanish: `fog_max` / `fog_density`.
Dark halos around objects against far background: `depth_darkening_range` (silhouette cutoff, metres).
Render textures (`call render_texture`) render the same pipeline with their camera's `layers` mask;
`RenderTextureOpts` override sky/fog/bloom/shadows. A camera with `RenderTextureCamera` is never the main camera.

## Shadows (cascades) and many lights

`render {debug_view: 3}` colours each pixel by shadow cascade (red, green, blue, yellow, grey = beyond
`shadow_distance`); `stats.render.cascades` lists their far distances. Blocky/shimmering far shadows: fewer metres per
cascade (`shadow_distance` down, `shadow_split_lambda` up). Missing shadow from a tall object just outside the view:
casters are kept up to 120 m towards the sun. `render {debug_view: 4}` = lights per cluster (heatmap, 32+ = yellow);
`stats.render.lights` = total / visible / max per cluster / CPU ms. Stress test: `lights {grid: {n: 20}}`,
`lights {clear_test: true}`.

## Terrain and grass

`terrain` shows chunks loaded/built and the height range; `terrain {height_at: [x, z]}` compares with what you see.
Cracks between chunks: `TerrainDesc.skirt`. Grass missing: `grass` (`has_terrain`, `blades_last_frame`), the splat
layer (`terrain {height_at}` → `splat`), `render {grass: true}`. Terrain looks tiled: `TerrainDesc.tiling`.

## Culling, LOD, mips

```bash
node tools/bridge.mjs call stats      # render: gpu_culling, gpu_frustum_culled, gpu_occluded, gpu_visible, lod_levels
node tools/bridge.mjs call render '{"gpu_culling":false}'       # CPU frustum path (direct draws) for comparison
node tools/bridge.mjs call render '{"occlusion_culling":false}' # GPU frustum only
node tools/bridge.mjs call gpu_scene  # GPU-driven scene: slots, draw groups, id layout, GPU counters incl. shadow instances
```

- Missing/ghost object only with GPU culling on: compare with `gpu_culling:false` at a paused, frozen frame
  (`pause` + `render {freeze_time:true, auto_resolution:false}`); both paths build records with `classify`,
  so any pixel difference beyond HUD text is a GPU-scene bug (change detection in `GpuScene::sync`, group
  bounds in `draw_order`/`shadow_groups`, LOD rule in `cull.wgsl` vs `Lod::pick`).

- Object missing only with GPU culling on -> bounding sphere too small (mesh bounds/scale) or Hi-Z
  popping (occlusion uses the *previous* frame: objects revealed by a fast camera move appear 1 frame late).
- `gpu_*` counters are an async readback (1-3 frames old); wait ~2 frames before asserting.
- `gpu_occluded` is always 0 in a small scene: Hi-Z occlusion only runs from
  `render {occlusion_min_triangles}` (default 100000) candidate triangles; set it to 0 to force it.
- Opaque batches are drawn front to back per pipeline (nearest bounding sphere first) for early-Z;
  `render {reverse_draw_order:true}` flips that too.
- `triangles` on the GPU path is an upper bound (candidates); the exact count stays on the GPU.
- `assets.texture_list` shows every texture with size, mip count and upload count.

## Renderer facts

- Forward PBR (GGX), 4× MSAA, ACES tonemap, manual sRGB output.
- 1 sun with 1–4 cascaded 2048² shadow maps (3×3 PCF, texel-snapped bounding spheres up to `shadow_distance`, cascade blend band).
- Clustered point + spot lights (16x9x24 froxels, up to 2048 lights, assigned on the CPU each frame).
- Up to 16 point lights (`PointLight` component); more are ignored.
- Instances sorted by texture then mesh; one storage buffer that grows as needed.
- Textures get full mip chains (GPU blit) and 8x anisotropic filtering. KTX2 is not supported.
- Frame = shadow pass -> **opaque pass** (stores colour + depth) -> **blend pass** (depth read-only and
  bound as `scene_depth` for water) -> Hi-Z. Water/blend shaders use `fs_blend`, opaque `fs_main`.
- Material maps: `Instance.maps` = (normal strength, displacement, parallax, water wave scale). Normal and
  height maps must be *linear* textures (`add_texture_rgba_linear` / `generate_maps`); a normal map uploaded
  as sRGB looks wrong (too bright, tilted). `render {"debug_view":1}` shows the mapped normals.
- Water flicker/moiré far away = waves smaller than a pixel; `waves()` fades them by pixel footprint.
- Main and shadow passes draw `instances[visible_ids[instance_index]]` (binding 4): record slots, written by
  the CPU on the CPU path and by `cull.wgsl` (per draw group, main view + each rendered cascade) on the GPU path. `hiz.wgsl` builds the max-depth pyramid after the main pass.
- Transparency: `stats.render.transparent` / `transparent_draw_calls` count blended instances (CPU frustum
  culled, sorted far->near, drawn after opaque, no depth write). Mask materials discard below the cutoff in
  both the main and the shadow pass (`shadow-mask` pipeline). Black foliage = normals pointing down.
- Material `ui: true` (world panels) = texture as-is, no lighting/tonemapping, alpha-tested.

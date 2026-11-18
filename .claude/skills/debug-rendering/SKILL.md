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

## 2. Narrow it down with numbers

```bash
node tools/bridge.mjs call stats      # render.visible, culled, draw_calls, triangles
node tools/bridge.mjs call camera     # where is the camera looking?
node tools/bridge.mjs call entity '{"id":N}'   # has mesh? scale 0? under a parent far away?
node tools/bridge.mjs call assets     # model failed / pending?
```

- `visible: 0` with entities present → camera or culling (bounding sphere / scale).
- Entity exists, model `pending` forever → fetch path wrong (`assets/…` relative to `dist/`).

## 3. Visual debug views

```bash
node tools/bridge.mjs call render '{"debug_view":1}'   # normals
node tools/bridge.mjs call render '{"debug_view":2}'   # shadow term (white = lit)
node tools/bridge.mjs call render '{"debug_view":0}'   # back to lit
node tools/bridge.mjs call render '{"shadows":false}'
node tools/bridge.mjs call render '{"sun_dir":[-0.4,-1,-0.3],"sun_intensity":3,"shadow_extent":30}'
node tools/bridge.mjs screenshot <scratch>/view.png
```

Reset debug settings before finishing.

## Culling, LOD, mips

```bash
node tools/bridge.mjs call stats      # render: gpu_culling, gpu_frustum_culled, gpu_occluded, gpu_visible, lod_levels
node tools/bridge.mjs call render '{"gpu_culling":false}'       # CPU frustum path (direct draws) for comparison
node tools/bridge.mjs call render '{"occlusion_culling":false}' # GPU frustum only
```

- Object missing only with GPU culling on -> bounding sphere too small (mesh bounds/scale) or Hi-Z
  popping (occlusion uses the *previous* frame: objects revealed by a fast camera move appear 1 frame late).
- `gpu_*` counters are an async readback (1-3 frames old); wait ~2 frames before asserting.
- `triangles` on the GPU path is an upper bound (candidates); the exact count stays on the GPU.
- `assets.texture_list` shows every texture with size, mip count and upload count.

## Renderer facts

- Forward PBR (GGX), 4× MSAA, ACES tonemap, manual sRGB output.
- 1 sun with 2048² shadow map (3×3 PCF, texel-snapped, fits `shadow_extent` around the camera target).
- Up to 16 point lights (`PointLight` component); more are ignored.
- Instances sorted by texture then mesh; one storage buffer that grows as needed.
- Textures get full mip chains (GPU blit) and 8x anisotropic filtering. KTX2 is not supported.
- Main pass draws `instances[visible_ids[instance_index]]` (binding 4): identity on the CPU path,
  compacted by `cull.wgsl` on the GPU path. `hiz.wgsl` builds the max-depth pyramid after the main pass.
- Transparency: `stats.render.transparent` / `transparent_draw_calls` count blended instances (CPU frustum
  culled, sorted far->near, drawn after opaque, no depth write). Mask materials discard below the cutoff in
  both the main and the shadow pass (`shadow-mask` pipeline). Black foliage = normals pointing down.
- Material `ui: true` (world panels) = texture as-is, no lighting/tonemapping, alpha-tested.

//! The one renderer: forward PBR over WebGPU (via `web-sys`) into an HDR target, 4x MSAA, sun with
//! cascaded shadow maps, clustered point + spot lights, GPU/CPU culling, instanced draws sorted by
//! texture+mesh, terrain + GPU grass, views (canvas and render textures), post-processing.

#![allow(deprecated)] // glam 0.33 deprecates Mat4::perspective_rh etc. in favour of glam::camera

use crate::mesh::{MeshData, Vertex};
use crate::scene::{Camera, GlobalTransform, MeshRenderer};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use hecs::World;
use js_sys::{JsNullable, Number};
use serde::Serialize;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::*;

const SAMPLES: u32 = 4;
const DEPTH_FORMAT: GpuTextureFormat = GpuTextureFormat::Depth24plus;
const SHADOW_FORMAT: GpuTextureFormat = GpuTextureFormat::Depth32float;
/// First automatic LOD switch at this multiple of the mesh's bounding radius (see `add_mesh_with_lods`).
pub const AUTO_LOD_FACTOR: f32 = 40.0;
/// Shadow cascades (layers of the shadow map array).
const MAX_CASCADES: usize = 4;
/// Uniform slot stride for per-cascade matrices (minUniformBufferOffsetAlignment).
const CASCADE_SLOT: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct MeshId(pub u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct MaterialId(pub u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct TextureId(pub u32);

#[derive(Clone, Debug, Serialize)]
pub struct Material {
    pub name: String,
    /// Linear RGBA.
    pub base_color: Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: Vec3,
    pub texture: Option<TextureId>,
    /// World-space checker frequency; 0 = off.
    pub checker: f32,
    pub unlit: bool,
    /// Screen-like surface (world panels): texture * base_color, no lighting, no tonemapping.
    pub ui: bool,
    /// How `base_color.a * texture.a` is used (see [`AlphaMode`]).
    pub alpha_mode: AlphaMode,
    /// Render both faces (leaves, grass cards, glass panes). Back faces get a mirrored normal.
    pub double_sided: bool,
    /// Tangent-space normal map (linear texture, e.g. [`crate::texgen::TexGen::normal_map`]).
    pub normal_texture: Option<TextureId>,
    /// Normal-map strength (1 = as authored).
    pub normal_scale: f32,
    /// Height map (linear, white = high) used by `displacement` and `parallax`.
    pub height_texture: Option<TextureId>,
    /// Vertex displacement along the normal in mesh units, centred on height 0.5. Needs a
    /// tessellated mesh (`MeshData::grid`). Shadows use the undisplaced mesh.
    pub displacement: f32,
    /// Parallax occlusion mapping depth in UV units (≈ 0.02–0.08). Per pixel, works on low-poly meshes.
    pub parallax: f32,
    /// Water surface (see [`Material::water`]): `base_color` = deep colour, `emissive` = shallow colour.
    pub water: bool,
    /// Water: wave height (world units) and wave frequency scale.
    pub wave_height: f32,
    pub wave_scale: f32,
    /// Water: Gerstner choppiness 0..1.5 (sharp crests, horizontal motion, crest foam).
    pub water_chop: f32,
    /// Water: metres of water after which ~63% of the light is absorbed (clear pool 6, sea 2, murky pond 0.6).
    pub water_clarity: f32,
    /// Water: foam amount at the shore and on crests (0 = none).
    pub water_foam: f32,
    /// Water: vertex spacing (m) of the mesh near the camera; `spawn_water`/`spawn_ocean` build
    /// meshes this dense, and vertex waves shorter than ~2 spacings are left to the pixel shader.
    pub water_vertex_spacing: f32,
    /// Terrain splat shading: `texture` = 2x2 layer atlas, `normal_texture` = splat map (RGBA
    /// weights), UVs from world XZ. Set up by `Ctx::set_terrain`.
    pub terrain: bool,
    /// Terrain: (local corner x, local corner z, 1 / size, unused) — updated by the engine.
    pub terrain_params: [f32; 4],
    /// Terrain: metres per texture tile for each layer.
    pub terrain_tiling: [f32; 4],
}

/// Transparency of a material (glTF semantics).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum AlphaMode {
    /// Alpha ignored.
    #[default]
    Opaque,
    /// Alpha test: pixels with alpha < cutoff are discarded (foliage, fences, decals). Writes
    /// depth, casts cut-out shadows, batches and GPU-culls like opaque geometry.
    Mask(f32),
    /// Alpha blending (water, glass, glows, smoke). Drawn after all opaque geometry, sorted back
    /// to front per instance, no depth write, no shadow.
    Blend,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: Vec4::ONE,
            metallic: 0.0,
            roughness: 0.6,
            emissive: Vec3::ZERO,
            texture: None,
            checker: 0.0,
            unlit: false,
            ui: false,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            normal_texture: None,
            normal_scale: 1.0,
            height_texture: None,
            displacement: 0.0,
            parallax: 0.0,
            water: false,
            wave_height: 0.0,
            wave_scale: 1.0,
            water_chop: 0.6,
            water_clarity: 2.0,
            water_foam: 1.0,
            water_vertex_spacing: 0.25,
            terrain: false,
            terrain_params: [0.0, 0.0, 1.0, 0.0],
            terrain_tiling: [6.0; 4],
        }
    }
}

impl Material {
    /// sRGB hex color (e.g. 0xff8800) -> linear material.
    pub fn color(hex: u32) -> Self {
        Self {
            base_color: srgb_hex(hex),
            ..Default::default()
        }
    }
    pub fn rough(mut self, r: f32) -> Self {
        self.roughness = r;
        self
    }
    pub fn metal(mut self, m: f32) -> Self {
        self.metallic = m;
        self
    }
    pub fn glow(mut self, e: Vec3) -> Self {
        self.emissive = e;
        self
    }
    /// Alpha blending with the given opacity (multiplies `base_color.a`).
    pub fn transparent(mut self, opacity: f32) -> Self {
        self.alpha_mode = AlphaMode::Blend;
        self.base_color.w *= opacity;
        self
    }
    /// Alpha test (cut-out) at `cutoff`, double-sided — the usual foliage setup.
    pub fn cutout(mut self, cutoff: f32) -> Self {
        self.alpha_mode = AlphaMode::Mask(cutoff);
        self.double_sided = true;
        self
    }
    pub fn two_sided(mut self) -> Self {
        self.double_sided = true;
        self
    }
    /// Normal map with strength.
    pub fn normal_map(mut self, tex: Option<TextureId>, strength: f32) -> Self {
        self.normal_texture = tex;
        self.normal_scale = strength;
        self
    }
    /// Height map driving parallax occlusion mapping (`parallax` in UV units).
    pub fn parallax_map(mut self, height: Option<TextureId>, parallax: f32) -> Self {
        self.height_texture = height;
        self.parallax = parallax;
        self
    }
    /// Height map driving vertex displacement (`amount` in mesh units; use a `MeshData::grid` mesh).
    pub fn displace(mut self, height: Option<TextureId>, amount: f32) -> Self {
        self.height_texture = height;
        self.displacement = amount;
        self
    }
    /// Water surface: Gerstner waves + ripples, refraction with depth absorption (`shallow` ->
    /// `deep` sRGB hex), caustics, screen-space reflections over the sky, sun specular, subsurface
    /// glow, shore + crest foam. Drawn in the blend pass. Spawn it with `ctx.spawn_water` (bounded)
    /// or `ctx.spawn_ocean` (endless) so the mesh is dense enough for the vertex waves; tune
    /// `wave_height`, `wave_scale`, `water_chop`, `water_clarity`, `water_foam`.
    pub fn water(deep: u32, shallow: u32) -> Self {
        Self {
            name: "water".into(),
            base_color: srgb_hex(deep),
            emissive: srgb_hex(shallow).truncate(),
            roughness: 0.05,
            alpha_mode: AlphaMode::Blend,
            water: true,
            wave_height: 0.035,
            wave_scale: 1.0,
            water_chop: 0.6,
            water_clarity: 2.0,
            water_foam: 1.0,
            ..Default::default()
        }
    }
}

pub fn srgb_hex(hex: u32) -> Vec4 {
    let c = |s: u32| ((hex >> s) & 0xff) as f32 / 255.0;
    let lin = |x: f32| {
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    };
    Vec4::new(lin(c(16)), lin(c(8)), lin(c(0)), 1.0)
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct RenderSettings {
    /// Direction the sunlight travels (normalised on use).
    pub sun_dir: Vec3,
    pub sun_color: Vec3,
    pub sun_intensity: f32,
    pub sky_color: Vec3,
    pub ground_color: Vec3,
    /// Linear clear color.
    pub clear_color: Vec3,
    pub shadows: bool,
    /// Sun shadows reach this far from the camera (m), split into `shadow_cascades` cascades.
    pub shadow_distance: f32,
    /// 1..4 cascaded shadow maps (2048² each).
    pub shadow_cascades: u32,
    /// Split distribution: 0 = linear, 1 = logarithmic.
    pub shadow_split_lambda: f32,
    /// Light clusters cover depths up to this distance (lights beyond use the last slice).
    pub light_cluster_far: f32,
    /// Cull on the GPU (compute) and draw with drawIndexedIndirect (needs indirect-first-instance).
    pub gpu_culling: bool,
    /// Hi-Z occlusion culling inside the GPU culling pass (previous frame's depth).
    pub occlusion_culling: bool,
    /// World-space offset added to the camera position (flicker detection: tiny shifts between
    /// screenshots reveal z-fighting and shimmering).
    pub camera_jitter: Vec3,
    /// Freeze the shader clock (water animation) so consecutive screenshots are comparable.
    pub freeze_time: bool,
    /// Relative change of the camera near plane. Moves depth values only, never pixel positions:
    /// the image stays identical except where two surfaces tie in depth (z-fighting flips).
    pub depth_jitter: f32,
    /// Draw opaque instances in reverse order (flicker scan: exact depth ties change their winner).
    pub reverse_draw_order: bool,
    /// 0 = lit, 1 = normals, 2 = shadow term, 3 = shadow cascade, 4 = lights per cluster (heatmap).
    pub debug_view: u32,
    // ---- atmosphere ----
    /// Draw the gradient sky (otherwise `clear_color`).
    pub sky: bool,
    pub sky_zenith: Vec3,
    pub sky_horizon: Vec3,
    /// Angular radius of the sun disk (degrees, 0 = none).
    pub sun_size_deg: f32,
    /// Exponential fog density per metre (0 = off).
    pub fog_density: f32,
    /// Height fog: density falls off with exp(-falloff * height). 0 = uniform fog.
    pub fog_height_falloff: f32,
    /// No fog closer than this (m).
    pub fog_start: f32,
    /// Maximum fog opacity (keeps far objects faintly visible).
    pub fog_max: f32,
    /// Fixed fog colour; None = the sky colour towards the horizon (fog blends into the sky).
    pub fog_color: Option<Vec3>,
    // ---- sky + day/night (see `crate::sky`) ----
    /// 0 = gradient (`sky_zenith`/`sky_horizon`), 1 = physically based atmosphere (follows the sun).
    pub sky_mode: u32,
    /// Atmosphere brightness (sun irradiance in scene units).
    pub atmosphere_intensity: f32,
    /// Cloud coverage 0..1 (0 = no clouds) and drift speed.
    pub clouds: f32,
    pub cloud_speed: f32,
    /// Star brightness at night (0 = none).
    pub stars: f32,
    /// Time of day in hours (0..24). With `day_cycle` it places sun + moon and drives the lighting.
    pub time_of_day: f32,
    /// Let `time_of_day` drive sun/moon direction, light colour and intensity, ambient and sky.
    /// Overwrites `sun_dir`, `sun_color`, `sun_intensity`, `sky_color`, `ground_color`, `sky_zenith`, `sky_horizon`.
    pub day_cycle: bool,
    /// Real-time minutes per 24 h of game time (0 = the clock stands still). Follows the simulation clock.
    pub day_length_minutes: f32,
    /// Sunrise direction (degrees around Y from +X) and the sun's height at noon (degrees).
    pub sun_azimuth_deg: f32,
    pub sun_max_elevation_deg: f32,
    /// Sun intensity at noon (the cycle scales it by the atmosphere's transmittance).
    pub daylight_intensity: f32,
    /// Moon brightness (moonlight, disk, moonlit sky).
    pub moon_intensity: f32,
    /// Brighten the image at night (exposure x up to 2.6).
    pub day_cycle_exposure: bool,
    /// Lens flare when looking towards the sun (glow, starburst, coloured ghosts); 0 = off. Hidden
    /// by anything in front of the sun disk (objects, clouds).
    pub lens_flare: f32,
    /// Skybox image (`ctx.load_skybox`): blend over the procedural sky by day / by night (0..1),
    /// rotation around Y, brightness.
    pub skybox_blend: f32,
    pub skybox_night_blend: f32,
    pub skybox_rotation_deg: f32,
    pub skybox_intensity: f32,
    // ---- water ----
    /// Screen-space reflections on water (off = sky only).
    pub water_ssr: bool,
    /// Refraction strength (screen-space distortion of what is below the surface).
    pub water_refraction: f32,
    /// Caustics brightness on the ground below water.
    pub water_caustics: f32,
    // ---- post-processing ----
    pub exposure: f32,
    pub bloom: bool,
    pub bloom_intensity: f32,
    /// HDR brightness above which pixels bloom (soft knee below).
    pub bloom_threshold: f32,
    pub bloom_knee: f32,
    /// Weight of the coarse bloom levels (wider glow).
    pub bloom_radius: f32,
    /// Depth darkening (unsharp-masked depth buffer): darkens creases and contact areas. 0 = off.
    pub depth_darkening: f32,
    /// Sample radius in pixels per 1000 px of screen height at 1 m distance.
    pub depth_darkening_radius: f32,
    /// Depth differences beyond this (m) are silhouettes, not creases.
    pub depth_darkening_range: f32,
    pub vignette: f32,
    pub saturation: f32,
    /// Draw particle systems.
    pub particles: bool,
    /// Draw GPU grass (needs a terrain; tune with the `grass` command / `renderer.grass_settings()`).
    pub grass: bool,
    // ---- resolution ----
    /// Scene resolution relative to the canvas (0.4..1). The composite pass upscales; the HUD stays sharp.
    pub render_scale: f32,
    /// Adjust `render_scale` automatically to stay inside `frame_budget_ms` of GPU time.
    pub auto_resolution: bool,
    /// GPU milliseconds per frame the adaptive resolution aims to stay under (60 Hz = 16.7 ms incl. CPU).
    pub frame_budget_ms: f32,
    /// Lowest scale the adaptive resolution may pick.
    pub min_render_scale: f32,
    /// Multiplies every LOD switch distance (2 = keep detail twice as far; top-down cameras far
    /// from small models want > 1). Also lowers itself never: set it per game.
    pub lod_bias: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            sun_dir: Vec3::new(-0.4, -1.0, -0.3),
            sun_color: Vec3::new(1.0, 0.96, 0.9),
            sun_intensity: 3.2,
            sky_color: Vec3::new(0.30, 0.38, 0.52),
            ground_color: Vec3::new(0.12, 0.10, 0.09),
            clear_color: Vec3::new(0.30, 0.42, 0.60),
            shadows: true,
            shadow_distance: 160.0,
            shadow_cascades: 4,
            shadow_split_lambda: 0.8,
            light_cluster_far: 400.0,
            gpu_culling: true,
            occlusion_culling: true,
            camera_jitter: Vec3::ZERO,
            freeze_time: false,
            depth_jitter: 0.0,
            reverse_draw_order: false,
            debug_view: 0,
            sky: true,
            sky_zenith: Vec3::new(0.16, 0.30, 0.62),
            sky_horizon: Vec3::new(0.60, 0.72, 0.86),
            sun_size_deg: 1.2,
            fog_density: 0.012,
            fog_height_falloff: 0.08,
            fog_start: 15.0,
            fog_max: 0.9,
            fog_color: None,
            sky_mode: 0,
            atmosphere_intensity: 12.0,
            clouds: 0.0,
            cloud_speed: 1.0,
            stars: 1.0,
            time_of_day: 14.0,
            day_cycle: false,
            day_length_minutes: 0.0,
            sun_azimuth_deg: 20.0,
            sun_max_elevation_deg: 60.0,
            daylight_intensity: 3.2,
            moon_intensity: 1.0,
            day_cycle_exposure: true,
            lens_flare: 1.0,
            skybox_blend: 1.0,
            skybox_night_blend: 1.0,
            skybox_rotation_deg: 0.0,
            skybox_intensity: 1.0,
            water_ssr: true,
            water_refraction: 1.0,
            water_caustics: 1.0,
            exposure: 1.0,
            bloom: true,
            bloom_intensity: 0.35,
            bloom_threshold: 1.2,
            bloom_knee: 0.6,
            bloom_radius: 0.8,
            depth_darkening: 1.0,
            depth_darkening_radius: 30.0,
            depth_darkening_range: 1.5,
            vignette: 0.15,
            saturation: 1.05,
            particles: true,
            grass: true,
            render_scale: 1.0,
            auto_resolution: true,
            frame_budget_ms: 11.0,
            min_render_scale: 0.5,
            lod_bias: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FrameStats {
    pub draw_calls: u32,
    pub shadow_draw_calls: u32,
    pub visible: u32,
    pub culled: u32,
    pub triangles: u64,
    /// Instances with a `Lod` component drawn at level 0, 1, 2, 3+.
    pub lod_levels: [u32; 4],
    /// Alpha-blended instances drawn this frame (sorted back to front) and their draw calls.
    pub transparent: u32,
    pub transparent_draw_calls: u32,
    /// GPU milliseconds per pass from timestamp queries (None: unsupported). Async, 1–3 frames old.
    pub gpu_ms: Option<crate::gpu_timing::GpuTimes>,
    /// True when this frame was culled on the GPU. Then `visible` = candidates sent to the GPU and
    /// the gpu_* counters below come from an async readback (1-3 frames old).
    pub gpu_culling: bool,
    pub gpu_frustum_culled: u32,
    pub gpu_occluded: u32,
    pub gpu_visible: u32,
    pub width: u32,
    pub height: u32,
    /// Canvas size; `width`/`height` are the (possibly scaled) scene resolution.
    pub canvas: [u32; 2],
    pub render_scale: f32,
    /// Live particles drawn this frame.
    pub particles: u32,
    /// Grass blade instances drawn (upper bound; the vertex shader drops empty cells).
    pub grass_blades: u32,
    /// Clustered lighting (total lights, visible, max per cluster, index count, CPU ms).
    pub lights: crate::lights::LightStats,
    /// Far distance of each shadow cascade (m).
    pub cascades: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Builtins {
    pub cube: MeshId,
    pub sphere: MeshId,
    pub plane: MeshId,
    /// 1x1 quad in the XY plane facing +Z, UV (0,0) top-left (world panels, billboards).
    pub quad: MeshId,
    /// LOD chain for `sphere` (32x16 -> 16x8 -> 8x6 segments).
    pub sphere_lod: Lod,
    pub default_material: MaterialId,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    cascades: [[[f32; 4]; 4]; MAX_CASCADES],
    camera_pos: [f32; 4],
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    sky_color: [f32; 4],
    ground_color: [f32; 4],
    counts: [u32; 4],
    time: [f32; 4],
    inv_view_proj: [[f32; 4]; 4],
    fog: [f32; 4],
    fog_color: [f32; 4],
    sky_zenith: [f32; 4],
    sky_horizon: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    view: [[f32; 4]; 4],
    cascade_splits: [f32; 4],
    cascade_texel: [f32; 4],
    cluster: [f32; 4],
    screen: [f32; 4],
    sky_sun: [f32; 4],
    sky_sun_color: [f32; 4],
    sky_moon: [f32; 4],
    sky_params: [f32; 4],
    skybox: [f32; 4],
    misc: [f32; 4],
}

/// Sky resources shared by every view: the atmosphere LUT (rendered per view), its sampler, the
/// skybox image (1x1 dummy until `set_skybox`) and the LUT pipeline.
struct SkyGpu {
    lut_view: GpuTextureView,
    sampler: GpuSampler,
    skybox_view: GpuTextureView,
    pipeline: GpuRenderPipeline,
    group: GpuBindGroup,
}

const SKY_LUT_SIZE: (u32, u32) = (192, 108);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 4],
    base_color: [f32; 4],
    emissive_metallic: [f32; 4],
    params: [f32; 4],
    maps: [f32; 4],
}

#[derive(Clone, Debug, Serialize)]
pub struct TexInfo {
    #[serde(skip)]
    tex: Option<GpuTexture>,
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub mips: u32,
    pub uploads: u32,
    /// Linear (data) texture: normal / height maps. Colour textures are sRGB.
    pub linear: bool,
}

/// Distance-based level of detail. `meshes[0]` is used up to `distances[0]` (camera distance
/// divided by the entity's largest scale axis, i.e. in mesh-local units), `meshes[i]` up to `distances[i]`; beyond the last distance the last mesh is used
/// (or the instance is skipped when `cull_beyond` is set). See [`Renderer::add_mesh_with_lods`].
#[derive(Clone, Debug, Serialize)]
pub struct Lod {
    pub meshes: Vec<MeshId>,
    pub distances: Vec<f32>,
    pub cull_beyond: bool,
}

impl Lod {
    pub fn pick(&self, dist: f32) -> Option<(usize, MeshId)> {
        for (i, d) in self.distances.iter().enumerate() {
            if dist <= *d {
                return self.meshes.get(i).map(|m| (i, *m));
            }
        }
        if self.cull_beyond {
            None
        } else {
            self.meshes.last().map(|m| (self.meshes.len() - 1, *m))
        }
    }
}

/// CPU copy of a mesh: positions and triangle indices.
pub type CpuMesh = (Vec<[f32; 3]>, Vec<u32>);

pub struct GpuMesh {
    pub name: String,
    vb: GpuBuffer,
    ib: GpuBuffer,
    pub index_count: u32,
    pub vertex_count: u32,
    pub bounds_center: Vec3,
    pub bounds_radius: f32,
    /// CPU copy (positions, indices) for analysis tools such as the z-fighting check; kept for
    /// meshes up to 60k indices.
    pub cpu: Option<std::rc::Rc<CpuMesh>>,
    /// Full CPU copy (positions, normals, UVs) for static batching; same size limit as `cpu`.
    pub data: Option<std::rc::Rc<MeshData>>,
}

pub struct Renderer {
    pub device: GpuDevice,
    pub queue: GpuQueue,
    context: GpuCanvasContext,
    /// [opaque, opaque 2-sided, alpha-test, alpha-test 2-sided, blended, blended 2-sided].
    /// Opaque pipelines are compiled without `discard` (early-Z).
    pipelines: [GpuRenderPipeline; 6],
    shadow_pipeline: GpuRenderPipeline,
    shadow_mask_pipeline: GpuRenderPipeline,
    bgl0: GpuBindGroupLayout,
    bgl0_shadow: GpuBindGroupLayout,
    bgl1: GpuBindGroupLayout,
    /// Material group: base colour, sampler, normal map, height map.
    bgl_mat: GpuBindGroupLayout,
    /// Group 0 for the blend pass: bgl0 + the opaque scene depth (binding 5).
    bgl0_t: GpuBindGroupLayout,
    bind0_t: Option<GpuBindGroup>,
    sky_gpu: SkyGpu,
    skybox: Option<TextureId>,
    /// Derived sky / day-night state (sun, moon, daylight, night), see `crate::sky`.
    pub sky: crate::sky::SkyState,
    globals_buf: GpuBuffer,
    instance_buf: GpuBuffer,
    /// Instance indices drawn by the main pass (identity on CPU path, written by cull.wgsl on GPU path).
    visible_buf: GpuBuffer,
    cull: Option<crate::gpu_cull::GpuCull>,
    /// GPU pass timing (timestamp queries), None when the adapter lacks `timestamp-query`.
    timing: Option<crate::gpu_timing::GpuTiming>,
    pub gpu_culling_supported: bool,
    instance_cap: u32,
    bind0: GpuBindGroup,
    /// One shadow bind group per cascade (its light matrix at a different uniform offset).
    bind0_shadow: Vec<GpuBindGroup>,
    /// Shadow map array (sampling view) + one attachment view per cascade layer.
    shadow_view: GpuTextureView,
    shadow_layers: Vec<GpuTextureView>,
    shadow_size: u32,
    cascade_buf: GpuBuffer,
    /// Clustered lighting buffers (lights, cluster grid, light index list).
    light_bufs: LightBufs,
    shadow_sampler: GpuSampler,
    color_sampler: GpuSampler,
    msaa_view: Option<GpuTextureView>,
    depth_view: Option<GpuTextureView>,
    /// Post-processing, sky and particle pipelines.
    pub(crate) post: crate::post::Post,
    /// GPU grass on the terrain (settings: `renderer.grass.settings`).
    pub(crate) grass: crate::vegetation::GrassRenderer,
    /// HDR/bloom targets of the canvas.
    targets: Option<crate::post::PostTargets>,
    /// Render textures (rendered before the main view each frame they are due).
    pub(crate) rtts: Vec<crate::render_texture::RenderTexture>,
    /// Scene render size (canvas size x render_scale).
    size: (u32, u32),
    canvas_size: (u32, u32),
    adapt: (f32, f32, u32),
    meshes: Vec<GpuMesh>,
    /// Slots of removed meshes, reused by `add_mesh` (streaming frees and re-creates chunks).
    free_meshes: Vec<u32>,
    materials: Vec<Material>,
    /// Index 0 = 1x1 white default texture.
    tex_views: Vec<GpuTextureView>,
    /// 1x1 flat normal (0.5, 0.5, 1) used when a material has no normal map.
    flat_normal: Option<GpuTextureView>,
    /// Material bind groups by (base, normal, height) internal texture index (u32::MAX = default).
    mat_groups: std::collections::HashMap<(u32, u32, u32), GpuBindGroup>,
    /// Seconds since start (water animation).
    start_ms: f64,
    frozen_time: Option<f32>,
    /// GPU texture objects parallel to `textures` (for updates + mip generation).
    tex_objects: Vec<TexInfo>,
    mip_pipeline: GpuRenderPipeline,
    mip_pipeline_linear: GpuRenderPipeline,
    mip_sampler: GpuSampler,
    pub settings: RenderSettings,
    pub stats: FrameStats,
    pub builtin: Builtins,
    /// Matrices of the last rendered frame (for picking / world labels).
    pub last_view_proj: Mat4,
}

fn js_err(e: JsValue) -> String {
    e.as_string()
        .or_else(|| js_sys::JSON::stringify(&e).ok().and_then(|s| s.as_string()))
        .unwrap_or_else(|| format!("{e:?}"))
}

fn nums(v: &[u32]) -> Vec<Number> {
    v.iter().map(|&x| Number::from(x)).collect()
}

impl Renderer {
    pub async fn new(canvas: &HtmlCanvasElement) -> Result<Renderer, String> {
        let gpu: Gpu = window().ok_or("no window")?.navigator().gpu();
        if gpu.is_undefined() {
            return Err(
                "WebGPU is not available (navigator.gpu is undefined). Use a current Chrome."
                    .into(),
            );
        }
        let opts = GpuRequestAdapterOptions::new();
        opts.set_power_preference(GpuPowerPreference::HighPerformance);
        let adapter = gpu
            .request_adapter_with_options(&opts)
            .await
            .map_err(js_err)?
            .as_option()
            .ok_or("navigator.gpu.requestAdapter() returned null (no suitable GPU)")?;
        let gpu_cull_supported = adapter.features().has(crate::gpu_cull::FEATURE);
        let timing_supported = adapter.features().has(crate::gpu_timing::FEATURE);
        let dd = GpuDeviceDescriptor::new();
        let mut features = vec![];
        if gpu_cull_supported {
            features.push(js_sys::JsString::from(crate::gpu_cull::FEATURE));
        }
        if timing_supported {
            features.push(js_sys::JsString::from(crate::gpu_timing::FEATURE));
        }
        dd.set_required_features(&features);
        let device: GpuDevice = adapter
            .request_device_with_descriptor(&dd)
            .await
            .map_err(js_err)?;
        let queue = device.queue();

        let context: GpuCanvasContext = canvas
            .get_context("webgpu")
            .map_err(js_err)?
            .ok_or("canvas.getContext('webgpu') returned null")?
            .unchecked_into();
        let format = gpu.get_preferred_canvas_format();
        let cfg = GpuCanvasConfiguration::new(&device, format);
        cfg.set_alpha_mode(GpuCanvasAlphaMode::Opaque);
        context.configure(&cfg).map_err(js_err)?;

        // --- Bind group layouts ---------------------------------------------------------
        let vf = gpu_shader_stage::VERTEX | gpu_shader_stage::FRAGMENT;
        let uniform_entry = |binding: u32, vis: u32| {
            let e = GpuBindGroupLayoutEntry::new(binding, vis);
            let b = GpuBufferBindingLayout::new();
            b.set_type(GpuBufferBindingType::Uniform);
            e.set_buffer(&b);
            e
        };
        let storage_entry = |binding: u32, vis: u32| {
            let e = GpuBindGroupLayoutEntry::new(binding, vis);
            let b = GpuBufferBindingLayout::new();
            b.set_type(GpuBufferBindingType::ReadOnlyStorage);
            e.set_buffer(&b);
            e
        };
        let shadow_tex_entry = GpuBindGroupLayoutEntry::new(2, gpu_shader_stage::FRAGMENT);
        let t = GpuTextureBindingLayout::new();
        t.set_sample_type(GpuTextureSampleType::Depth);
        t.set_view_dimension(GpuTextureViewDimension::N2dArray);
        shadow_tex_entry.set_texture(&t);
        let shadow_samp_entry = GpuBindGroupLayoutEntry::new(3, gpu_shader_stage::FRAGMENT);
        let s = GpuSamplerBindingLayout::new();
        s.set_type(GpuSamplerBindingType::Comparison);
        shadow_samp_entry.set_sampler(&s);
        // sky LUT (10), sky sampler (11), skybox (12); scene colour (9, blend pass only)
        let float_tex = |binding: u32| {
            let e = GpuBindGroupLayoutEntry::new(binding, gpu_shader_stage::FRAGMENT);
            let t = GpuTextureBindingLayout::new();
            t.set_sample_type(GpuTextureSampleType::Float);
            e.set_texture(&t);
            e
        };
        let sky_samp = || {
            let e = GpuBindGroupLayoutEntry::new(11, gpu_shader_stage::FRAGMENT);
            let s = GpuSamplerBindingLayout::new();
            s.set_type(GpuSamplerBindingType::Filtering);
            e.set_sampler(&s);
            e
        };

        let bgl0 = device
            .create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[
                uniform_entry(0, vf),
                storage_entry(1, vf),
                storage_entry(4, gpu_shader_stage::VERTEX),
                shadow_tex_entry,
                shadow_samp_entry,
                storage_entry(6, gpu_shader_stage::FRAGMENT),
                storage_entry(7, gpu_shader_stage::FRAGMENT),
                storage_entry(8, gpu_shader_stage::FRAGMENT),
                float_tex(10),
                sky_samp(),
                float_tex(12),
            ]))
            .map_err(js_err)?;
        let bgl0_shadow = device
            .create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[
                uniform_entry(0, gpu_shader_stage::VERTEX),
                storage_entry(1, gpu_shader_stage::VERTEX),
                uniform_entry(2, gpu_shader_stage::VERTEX),
            ]))
            .map_err(js_err)?;
        let tex_entry = GpuBindGroupLayoutEntry::new(0, gpu_shader_stage::FRAGMENT);
        let t = GpuTextureBindingLayout::new();
        t.set_sample_type(GpuTextureSampleType::Float);
        tex_entry.set_texture(&t);
        let samp_entry = GpuBindGroupLayoutEntry::new(1, gpu_shader_stage::FRAGMENT);
        let s = GpuSamplerBindingLayout::new();
        s.set_type(GpuSamplerBindingType::Filtering);
        samp_entry.set_sampler(&s);
        let bgl1 = device
            .create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[tex_entry, samp_entry]))
            .map_err(js_err)?;
        let mat_tex = |binding: u32, vis: u32| {
            let e = GpuBindGroupLayoutEntry::new(binding, vis);
            let t = GpuTextureBindingLayout::new();
            t.set_sample_type(GpuTextureSampleType::Float);
            e.set_texture(&t);
            e
        };
        let mat_samp = GpuBindGroupLayoutEntry::new(1, vf);
        let s = GpuSamplerBindingLayout::new();
        s.set_type(GpuSamplerBindingType::Filtering);
        mat_samp.set_sampler(&s);
        let bgl_mat = device
            .create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[
                mat_tex(0, vf),
                mat_samp,
                mat_tex(2, gpu_shader_stage::FRAGMENT),
                mat_tex(3, vf),
            ]))
            .map_err(js_err)?;
        let depth_entry = GpuBindGroupLayoutEntry::new(5, gpu_shader_stage::FRAGMENT);
        let dt = GpuTextureBindingLayout::new();
        dt.set_sample_type(GpuTextureSampleType::Depth);
        dt.set_multisampled(true);
        depth_entry.set_texture(&dt);
        let st2 = GpuBindGroupLayoutEntry::new(2, gpu_shader_stage::FRAGMENT);
        let t2 = GpuTextureBindingLayout::new();
        t2.set_sample_type(GpuTextureSampleType::Depth);
        t2.set_view_dimension(GpuTextureViewDimension::N2dArray);
        st2.set_texture(&t2);
        let ss2 = GpuBindGroupLayoutEntry::new(3, gpu_shader_stage::FRAGMENT);
        let s2 = GpuSamplerBindingLayout::new();
        s2.set_type(GpuSamplerBindingType::Comparison);
        ss2.set_sampler(&s2);
        let bgl0_t = device
            .create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[
                uniform_entry(0, vf),
                storage_entry(1, vf),
                storage_entry(4, gpu_shader_stage::VERTEX),
                st2,
                ss2,
                depth_entry,
                storage_entry(6, gpu_shader_stage::FRAGMENT),
                storage_entry(7, gpu_shader_stage::FRAGMENT),
                storage_entry(8, gpu_shader_stage::FRAGMENT),
                float_tex(9),
                float_tex(10),
                sky_samp(),
                float_tex(12),
            ]))
            .map_err(js_err)?;

        // --- Shaders & pipelines ----------------------------------------------------------
        let common = include_str!("shaders/common.wgsl");
        let pbr_src = format!("{common}\n{}", include_str!("shaders/pbr.wgsl"));
        let shadow_src = format!("{common}\n{}", include_str!("shaders/shadow.wgsl"));
        let pbr_module = device.create_shader_module(&GpuShaderModuleDescriptor::new(&pbr_src));
        pbr_module.set_label("pbr.wgsl");
        let shadow_module =
            device.create_shader_module(&GpuShaderModuleDescriptor::new(&shadow_src));
        shadow_module.set_label("shadow.wgsl");
        crate::debug::report_shader_errors(&pbr_module, "pbr.wgsl");
        crate::debug::report_shader_errors(&shadow_module, "shadow.wgsl");

        let vertex_layout = || {
            let attrs = [
                GpuVertexAttribute::new(GpuVertexFormat::Float32x3, 0, 0),
                GpuVertexAttribute::new(GpuVertexFormat::Float32x3, 12, 1),
                GpuVertexAttribute::new(GpuVertexFormat::Float32x2, 24, 2),
            ];
            GpuVertexBufferLayout::new(std::mem::size_of::<Vertex>() as u32, &attrs)
        };

        let layout = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[
            JsNullable::wrap(bgl0.clone()),
            JsNullable::wrap(bgl_mat.clone()),
        ]));
        let layout_t = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[
            JsNullable::wrap(bgl0_t.clone()),
            JsNullable::wrap(bgl_mat.clone()),
        ]));
        // Main pipelines: (kind, double_sided) variants of the same PBR shader.
        // kind 0 = opaque (no discard -> early-Z), 1 = alpha-test / UI, 2 = blend.
        let make_pipeline = |kind: u8, two_sided: bool| -> Result<GpuRenderPipeline, String> {
            let blend = kind == 2;
            let vs = GpuVertexState::new(&pbr_module);
            vs.set_entry_point("vs_main");
            vs.set_buffers(&[JsNullable::wrap(vertex_layout())]);
            let target = GpuColorTargetState::new(crate::post::HDR_FORMAT);
            if blend {
                let c = GpuBlendComponent::new();
                c.set_src_factor(GpuBlendFactor::SrcAlpha);
                c.set_dst_factor(GpuBlendFactor::OneMinusSrcAlpha);
                c.set_operation(GpuBlendOperation::Add);
                let a = GpuBlendComponent::new();
                a.set_src_factor(GpuBlendFactor::One);
                a.set_dst_factor(GpuBlendFactor::OneMinusSrcAlpha);
                a.set_operation(GpuBlendOperation::Add);
                target.set_blend(&GpuBlendState::new(&a, &c));
            }
            let fs = GpuFragmentState::new(&pbr_module, &[JsNullable::wrap(target)]);
            fs.set_entry_point(if blend { "fs_blend" } else { "fs_main" });
            let consts = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&consts, &"ALPHA_TEST".into(), &(if kind == 0 { 0.0 } else { 1.0 }).into());
            fs.set_constants(consts.unchecked_ref());
            let prim = GpuPrimitiveState::new();
            prim.set_topology(GpuPrimitiveTopology::TriangleList);
            prim.set_cull_mode(if two_sided { GpuCullMode::None } else { GpuCullMode::Back });
            prim.set_front_face(GpuFrontFace::Ccw);
            let ds = GpuDepthStencilState::new(DEPTH_FORMAT);
            ds.set_depth_write_enabled(!blend); // blended surfaces test but do not write depth
            ds.set_depth_compare(GpuCompareFunction::Less);
            let ms = GpuMultisampleState::new();
            ms.set_count(SAMPLES);
            let desc = GpuRenderPipelineDescriptor::new(if blend { &layout_t } else { &layout }, &vs);
            desc.set_label(match (kind, two_sided) {
                (0, false) => "pbr-opaque",
                (0, true) => "pbr-opaque-2s",
                (1, false) => "pbr-mask",
                (1, true) => "pbr-mask-2s",
                (_, false) => "pbr-blend",
                (_, true) => "pbr-blend-2s",
            });
            desc.set_fragment(&fs);
            desc.set_primitive(&prim);
            desc.set_depth_stencil(&ds);
            desc.set_multisample(&ms);
            device.create_render_pipeline(&desc).map_err(js_err)
        };
        let pipelines = [
            make_pipeline(0, false)?,
            make_pipeline(0, true)?,
            make_pipeline(1, false)?,
            make_pipeline(1, true)?,
            make_pipeline(2, false)?,
            make_pipeline(2, true)?,
        ];

        let shadow_layout =
            device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[JsNullable::wrap(
                bgl0_shadow.clone(),
            )]));
        let svs = GpuVertexState::new(&shadow_module);
        svs.set_entry_point("vs_shadow");
        svs.set_buffers(&[JsNullable::wrap(vertex_layout())]);
        let sprim = GpuPrimitiveState::new();
        sprim.set_topology(GpuPrimitiveTopology::TriangleList);
        sprim.set_cull_mode(GpuCullMode::None);
        let sds = GpuDepthStencilState::new(SHADOW_FORMAT);
        sds.set_depth_write_enabled(true);
        sds.set_depth_compare(GpuCompareFunction::Less);
        sds.set_depth_bias(2);
        sds.set_depth_bias_slope_scale(2.0);
        let sdesc = GpuRenderPipelineDescriptor::new(&shadow_layout, &svs);
        sdesc.set_label("shadow");
        sdesc.set_primitive(&sprim);
        sdesc.set_depth_stencil(&sds);
        let shadow_pipeline = device.create_render_pipeline(&sdesc).map_err(js_err)?;
        // Alpha-tested casters (foliage): same depth-only pass + a fragment stage that discards.
        let mask_layout = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[
            JsNullable::wrap(bgl0_shadow.clone()),
            JsNullable::wrap(bgl_mat.clone()),
        ]));
        let mvs = GpuVertexState::new(&shadow_module);
        mvs.set_entry_point("vs_shadow_mask");
        mvs.set_buffers(&[JsNullable::wrap(vertex_layout())]);
        let mfs = GpuFragmentState::new(&shadow_module, &[]);
        mfs.set_entry_point("fs_shadow_mask");
        let mdesc = GpuRenderPipelineDescriptor::new(&mask_layout, &mvs);
        mdesc.set_label("shadow-mask");
        mdesc.set_primitive(&sprim);
        mdesc.set_depth_stencil(&sds);
        mdesc.set_fragment(&mfs);
        let shadow_mask_pipeline = device.create_render_pipeline(&mdesc).map_err(js_err)?;

        // --- Resources --------------------------------------------------------------------
        let globals_buf = device
            .create_buffer(&GpuBufferDescriptor::new(
                std::mem::size_of::<Globals>() as u32,
                gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
            ))
            .map_err(js_err)?;
        let shadow_size = 2048;
        let st = GpuTextureDescriptor::new(
            SHADOW_FORMAT,
            &nums(&[shadow_size, shadow_size, MAX_CASCADES as u32]),
            gpu_texture_usage::RENDER_ATTACHMENT | gpu_texture_usage::TEXTURE_BINDING,
        );
        st.set_label("shadow-cascades");
        let shadow_tex = device.create_texture(&st).map_err(js_err)?;
        let svd = GpuTextureViewDescriptor::new();
        svd.set_dimension(GpuTextureViewDimension::N2dArray);
        let shadow_view = shadow_tex.create_view_with_descriptor(&svd).map_err(js_err)?;
        let mut shadow_layers = vec![];
        for k in 0..MAX_CASCADES as u32 {
            let d = GpuTextureViewDescriptor::new();
            d.set_dimension(GpuTextureViewDimension::N2d);
            d.set_base_array_layer(k);
            d.set_array_layer_count(1);
            shadow_layers.push(shadow_tex.create_view_with_descriptor(&d).map_err(js_err)?);
        }
        let cascade_buf = device
            .create_buffer(&GpuBufferDescriptor::new(CASCADE_SLOT * MAX_CASCADES as u32, gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST))
            .map_err(js_err)?;
        let light_bufs = LightBufs::new(&device, 256, 16384)?;
        let sd = GpuSamplerDescriptor::new();
        sd.set_compare(GpuCompareFunction::Less);
        sd.set_mag_filter(GpuFilterMode::Linear);
        sd.set_min_filter(GpuFilterMode::Linear);
        let shadow_sampler = device.create_sampler_with_descriptor(&sd);
        let cd = GpuSamplerDescriptor::new();
        cd.set_mag_filter(GpuFilterMode::Linear);
        cd.set_min_filter(GpuFilterMode::Linear);
        cd.set_mipmap_filter(GpuMipmapFilterMode::Linear);
        cd.set_max_anisotropy(8);
        cd.set_address_mode_u(GpuAddressMode::Repeat);
        cd.set_address_mode_v(GpuAddressMode::Repeat);
        let color_sampler = device.create_sampler_with_descriptor(&cd);

        // Mip generation: fullscreen triangle sampling the previous level (linear filter).
        let mip_module = device.create_shader_module(&GpuShaderModuleDescriptor::new(
            include_str!("shaders/mipgen.wgsl"),
        ));
        mip_module.set_label("mipgen.wgsl");
        crate::debug::report_shader_errors(&mip_module, "mipgen.wgsl");
        let mip_layout =
            device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[JsNullable::wrap(
                bgl1.clone(),
            )]));
        let mip_for = |fmt: GpuTextureFormat, label: &str| -> Result<GpuRenderPipeline, String> {
            let mvs = GpuVertexState::new(&mip_module);
            mvs.set_entry_point("vs");
            let mfs = GpuFragmentState::new(&mip_module, &[JsNullable::wrap(GpuColorTargetState::new(fmt))]);
            mfs.set_entry_point("fs");
            let mdesc = GpuRenderPipelineDescriptor::new(&mip_layout, &mvs);
            mdesc.set_label(label);
            mdesc.set_fragment(&mfs);
            device.create_render_pipeline(&mdesc).map_err(js_err)
        };
        let mip_pipeline = mip_for(GpuTextureFormat::Rgba8unormSrgb, "mipgen")?;
        let mip_pipeline_linear = mip_for(GpuTextureFormat::Rgba8unorm, "mipgen-linear")?;
        let md = GpuSamplerDescriptor::new();
        md.set_mag_filter(GpuFilterMode::Linear);
        md.set_min_filter(GpuFilterMode::Linear);
        let mip_sampler = device.create_sampler_with_descriptor(&md);

        let instance_cap = 1024;
        let instance_buf = Self::make_instance_buffer(&device, instance_cap)?;
        let visible_buf = Self::make_visible_buffer(&device, instance_cap)?;
        let sky_gpu = Self::make_sky(&device, common, &globals_buf)?;
        let (bind0, bind0_shadow) = Self::make_bind0(
            &device,
            &bgl0,
            &bgl0_shadow,
            &globals_buf,
            &instance_buf,
            &visible_buf,
            &shadow_view,
            &shadow_sampler,
            &light_bufs,
            &cascade_buf,
            &sky_gpu,
        );
        let cull = if gpu_cull_supported {
            match crate::gpu_cull::GpuCull::new(&device) {
                Ok(c) => Some(c),
                Err(e) => {
                    crate::log_warn!("GPU culling disabled: {e}");
                    None
                }
            }
        } else {
            crate::log_info!(
                "adapter lacks {}: CPU frustum culling + direct draws",
                crate::gpu_cull::FEATURE
            );
            None
        };

        let post = crate::post::Post::new(&device, format, common, &bgl0, &bgl0_t, DEPTH_FORMAT, SAMPLES)?;
        let grass = crate::vegetation::GrassRenderer::new(&device, &pbr_module, &bgl0, &bgl_mat, crate::post::HDR_FORMAT, DEPTH_FORMAT, SAMPLES)?;
        let timing = if timing_supported {
            crate::gpu_timing::GpuTiming::new(&device).map_err(|e| crate::log_warn!("GPU timing disabled: {e}")).ok()
        } else {
            None
        };
        let mut r = Renderer {
            device,
            queue,
            context,
            pipelines,
            shadow_pipeline,
            shadow_mask_pipeline,
            bgl0,
            bgl0_shadow,
            bgl1,
            bgl_mat,
            bgl0_t,
            bind0_t: None,
            sky_gpu,
            skybox: None,
            sky: Default::default(),
            globals_buf,
            instance_buf,
            visible_buf,
            cull,
            timing,
            gpu_culling_supported: gpu_cull_supported,
            instance_cap,
            bind0,
            bind0_shadow,
            shadow_view,
            shadow_layers,
            shadow_size,
            cascade_buf,
            light_bufs,
            shadow_sampler,
            color_sampler,
            msaa_view: None,
            depth_view: None,
            post,
            grass,
            targets: None,
            rtts: vec![],
            size: (0, 0),
            canvas_size: (0, 0),
            adapt: (0.0, 0.0, 0),
            meshes: vec![],
            free_meshes: vec![],
            materials: vec![],
            tex_views: vec![],
            flat_normal: None,
            mat_groups: std::collections::HashMap::new(),
            start_ms: crate::log::now_ms(),
            frozen_time: None,
            tex_objects: vec![],
            mip_pipeline,
            mip_pipeline_linear,
            mip_sampler,
            settings: RenderSettings::default(),
            stats: FrameStats::default(),
            builtin: Builtins {
                cube: MeshId(0),
                sphere: MeshId(0),
                plane: MeshId(0),
                quad: MeshId(0),
                sphere_lod: Lod {
                    meshes: vec![],
                    distances: vec![],
                    cull_beyond: false,
                },
                default_material: MaterialId(0),
            },
            last_view_proj: Mat4::IDENTITY,
        };
        r.create_white_texture()?;
        r.flat_normal = Some(r.create_solid(GpuTextureFormat::Rgba8unorm, (0.5, 0.5, 1.0, 1.0), "flat-normal")?);
        let cube = r.add_mesh("cube", &MeshData::cube(1.0));
        let sphere = r.add_mesh("sphere", &MeshData::sphere(0.5, 32, 16));
        let plane = r.add_mesh("plane", &MeshData::plane(1.0, 1.0));
        let quad = r.add_mesh("quad", &MeshData::quad());
        let s1 = r.add_mesh("sphere#lod1", &MeshData::sphere(0.5, 16, 8));
        let s2 = r.add_mesh("sphere#lod2", &MeshData::sphere(0.5, 8, 6));
        let sphere_lod = Lod {
            meshes: vec![sphere, s1, s2],
            distances: vec![12.0, 30.0],
            cull_beyond: false,
        };
        let default_material = r.add_material(Material {
            name: "default".into(),
            ..Default::default()
        });
        r.builtin = Builtins {
            cube,
            sphere,
            plane,
            quad,
            sphere_lod,
            default_material,
        };
        Ok(r)
    }

    fn make_instance_buffer(device: &GpuDevice, cap: u32) -> Result<GpuBuffer, String> {
        device
            .create_buffer(&GpuBufferDescriptor::new(
                cap * std::mem::size_of::<InstanceRaw>() as u32,
                gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
            ))
            .map_err(js_err)
    }

    fn make_visible_buffer(device: &GpuDevice, cap: u32) -> Result<GpuBuffer, String> {
        device
            .create_buffer(&GpuBufferDescriptor::new(
                cap * 4,
                gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
            ))
            .map_err(js_err)
    }

    #[allow(clippy::too_many_arguments)]
    fn make_bind0(
        device: &GpuDevice,
        bgl0: &GpuBindGroupLayout,
        bgl0_shadow: &GpuBindGroupLayout,
        globals: &GpuBuffer,
        instances: &GpuBuffer,
        visible: &GpuBuffer,
        shadow_view: &GpuTextureView,
        shadow_sampler: &GpuSampler,
        lb: &LightBufs,
        cascade_buf: &GpuBuffer,
        sky: &SkyGpu,
    ) -> (GpuBindGroup, Vec<GpuBindGroup>) {
        let full = device.create_bind_group(&GpuBindGroupDescriptor::new(
            &[
                GpuBindGroupEntry::new_with_gpu_buffer(0, globals),
                GpuBindGroupEntry::new_with_gpu_buffer(1, instances),
                GpuBindGroupEntry::new_with_gpu_texture_view(2, shadow_view),
                GpuBindGroupEntry::new(3, shadow_sampler),
                GpuBindGroupEntry::new_with_gpu_buffer(4, visible),
                GpuBindGroupEntry::new_with_gpu_buffer(6, &lb.lights),
                GpuBindGroupEntry::new_with_gpu_buffer(7, &lb.grid),
                GpuBindGroupEntry::new_with_gpu_buffer(8, &lb.indices),
                GpuBindGroupEntry::new_with_gpu_texture_view(10, &sky.lut_view),
                GpuBindGroupEntry::new(11, &sky.sampler),
                GpuBindGroupEntry::new_with_gpu_texture_view(12, &sky.skybox_view),
            ],
            bgl0,
        ));
        let shadow = (0..MAX_CASCADES as u32)
            .map(|k| {
                let b = GpuBufferBinding::new(cascade_buf);
                b.set_offset(k * CASCADE_SLOT);
                b.set_size(64);
                device.create_bind_group(&GpuBindGroupDescriptor::new(
                    &[
                        GpuBindGroupEntry::new_with_gpu_buffer(0, globals),
                        GpuBindGroupEntry::new_with_gpu_buffer(1, instances),
                        GpuBindGroupEntry::new_with_gpu_buffer_binding(2, &b),
                    ],
                    bgl0_shadow,
                ))
            })
            .collect();
        (full, shadow)
    }

    /// Sky LUT target + pipeline (uniform globals only), sky sampler, 1x1 skybox placeholder.
    fn make_sky(device: &GpuDevice, common: &str, globals: &GpuBuffer) -> Result<SkyGpu, String> {
        use gpu_texture_usage::{RENDER_ATTACHMENT as RA, TEXTURE_BINDING as TB};
        let td = GpuTextureDescriptor::new(crate::post::HDR_FORMAT, &nums(&[SKY_LUT_SIZE.0, SKY_LUT_SIZE.1]), RA | TB);
        td.set_label("sky-lut");
        let lut_view = device.create_texture(&td).map_err(js_err)?.create_view().map_err(js_err)?;
        let bd = GpuTextureDescriptor::new(crate::post::HDR_FORMAT, &nums(&[1, 1]), TB);
        bd.set_label("skybox-none");
        let skybox_view = device.create_texture(&bd).map_err(js_err)?.create_view().map_err(js_err)?;
        let sd = GpuSamplerDescriptor::new();
        sd.set_mag_filter(GpuFilterMode::Linear);
        sd.set_min_filter(GpuFilterMode::Linear);
        sd.set_address_mode_u(GpuAddressMode::Repeat);
        sd.set_address_mode_v(GpuAddressMode::ClampToEdge);
        let sampler = device.create_sampler_with_descriptor(&sd);
        let e = GpuBindGroupLayoutEntry::new(0, gpu_shader_stage::VERTEX | gpu_shader_stage::FRAGMENT);
        let b = GpuBufferBindingLayout::new();
        b.set_type(GpuBufferBindingType::Uniform);
        e.set_buffer(&b);
        let bgl = device.create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[e])).map_err(js_err)?;
        let src = format!("{common}\n{}", include_str!("shaders/sky_lut.wgsl"));
        let module = device.create_shader_module(&GpuShaderModuleDescriptor::new(&src));
        module.set_label("sky_lut.wgsl");
        crate::debug::report_shader_errors(&module, "sky_lut.wgsl");
        let layout = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[JsNullable::wrap(bgl.clone())]));
        let vs = GpuVertexState::new(&module);
        vs.set_entry_point("vs_lut");
        let fs = GpuFragmentState::new(&module, &[JsNullable::wrap(GpuColorTargetState::new(crate::post::HDR_FORMAT))]);
        fs.set_entry_point("fs_lut");
        let d = GpuRenderPipelineDescriptor::new(&layout, &vs);
        d.set_label("sky-lut");
        d.set_fragment(&fs);
        let pipeline = device.create_render_pipeline(&d).map_err(js_err)?;
        let group = device.create_bind_group(&GpuBindGroupDescriptor::new(&[GpuBindGroupEntry::new_with_gpu_buffer(0, globals)], &bgl));
        Ok(SkyGpu { lut_view, sampler, skybox_view, pipeline, group })
    }

    /// Lens flare parameters for the composite pass: sun position on screen, strength (fades as the
    /// sun leaves the screen, sets or is off), disk radius, colour and the brightness an unblocked
    /// sun disk has in the HDR image (the shader compares against it to find clouds/objects in front).
    fn lens_flare(&self, view_proj: Mat4, cam_pos: Vec3) -> ([f32; 4], [f32; 4]) {
        let s = &self.settings;
        let sun = self.sky.sun.normalize_or(Vec3::Y);
        if s.lens_flare <= 0.0 || !s.sky || s.sun_size_deg <= 0.0 || sun.y < -0.02 {
            return ([0.0; 4], [0.0; 4]);
        }
        let project = |d: Vec3| {
            let c = view_proj * (cam_pos + d * 1000.0).extend(1.0);
            (c.w > 0.0).then(|| glam::Vec2::new(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5))
        };
        let Some(uv) = project(sun) else { return ([0.0; 4], [0.0; 4]) };
        let side = sun.cross(Vec3::Y).normalize_or(Vec3::X);
        let edge = project((sun + side * s.sun_size_deg.to_radians().tan()).normalize()).unwrap_or(uv);
        let aspect = self.size.0 as f32 / self.size.1.max(1) as f32;
        let radius = ((edge - uv) * glam::Vec2::new(aspect, 1.0)).length().max(1e-4);
        // fade out as the sun leaves the screen (ghosts would swing wildly) and near the horizon
        let out = (uv - glam::Vec2::splat(0.5)).abs().max_element();
        let on_screen = 1.0 - ((out - 0.45) / 0.2).clamp(0.0, 1.0);
        let col = self.sky.sun_radiance;
        let lum = col.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        if on_screen <= 0.0 || lum <= 1e-4 {
            return ([0.0; 4], [0.0; 4]);
        }
        let tint = col / col.max_element().max(1e-4);
        let strength = s.lens_flare * on_screen * (lum / 3.0).clamp(0.0, 1.0) * self.sky.daylight.max(0.15);
        // sun_disk() in common.wgsl: radiance * 12, limb-darkened to ~0.8 inside
        ([uv.x, uv.y, strength, radius], [tint.x, tint.y, tint.z, lum * 12.0 * 0.8])
    }

    /// Equirectangular skybox image (a texture from `add_texture_*` / `ctx.load_skybox`), or None.
    /// Blended over the procedural sky by `skybox_blend` (day) / `skybox_night_blend` (night).
    pub fn set_skybox(&mut self, tex: Option<TextureId>) {
        // public texture ids are offset by one (index 0 = internal white texture)
        self.skybox = tex.filter(|t| (t.0 as usize + 1) < self.tex_views.len());
        if let Some(t) = self.skybox {
            self.sky_gpu.skybox_view = self.tex_views[t.0 as usize + 1].clone();
        }
        self.rebuild_bind0();
    }

    pub fn skybox(&self) -> Option<TextureId> {
        self.skybox
    }

    fn rebuild_bind0(&mut self) {
        let (a, b) = Self::make_bind0(
            &self.device,
            &self.bgl0,
            &self.bgl0_shadow,
            &self.globals_buf,
            &self.instance_buf,
            &self.visible_buf,
            &self.shadow_view,
            &self.shadow_sampler,
            &self.light_bufs,
            &self.cascade_buf,
            &self.sky_gpu,
        );
        self.bind0 = a;
        self.bind0_shadow = b;
        self.bind0_t = None; // rebuilt lazily (needs the depth view)
    }

    /// 1x1 texture of one colour (clear pass), not registered as a public texture.
    fn create_solid(&self, fmt: GpuTextureFormat, c: (f64, f64, f64, f64), label: &str) -> Result<GpuTextureView, String> {
        let desc = GpuTextureDescriptor::new(fmt, &nums(&[1, 1]), gpu_texture_usage::TEXTURE_BINDING | gpu_texture_usage::RENDER_ATTACHMENT);
        desc.set_label(label);
        let tex = self.device.create_texture(&desc).map_err(js_err)?;
        let view = tex.create_view().map_err(js_err)?;
        let enc = self.device.create_command_encoder();
        let att = GpuRenderPassColorAttachment::new_with_gpu_texture_view(GpuLoadOp::Clear, GpuStoreOp::Store, &view);
        att.set_clear_value_gpu_color_dict(&GpuColorDict::new(c.3, c.2, c.1, c.0));
        let pass = enc.begin_render_pass(&GpuRenderPassDescriptor::new(&[JsNullable::wrap(att)])).map_err(js_err)?;
        pass.end();
        self.queue.submit(&[enc.finish()]);
        Ok(view)
    }

    /// 1x1 white texture, filled by a clear pass (no texel upload API needed).
    fn create_white_texture(&mut self) -> Result<(), String> {
        let desc = GpuTextureDescriptor::new(
            GpuTextureFormat::Rgba8unorm,
            &nums(&[1, 1]),
            gpu_texture_usage::TEXTURE_BINDING | gpu_texture_usage::RENDER_ATTACHMENT,
        );
        let tex = self.device.create_texture(&desc).map_err(js_err)?;
        let view = tex.create_view().map_err(js_err)?;
        let enc = self.device.create_command_encoder();
        let att = GpuRenderPassColorAttachment::new_with_gpu_texture_view(
            GpuLoadOp::Clear,
            GpuStoreOp::Store,
            &view,
        );
        att.set_clear_value_gpu_color_dict(&GpuColorDict::new(1.0, 1.0, 1.0, 1.0));
        let pass = enc
            .begin_render_pass(&GpuRenderPassDescriptor::new(&[JsNullable::wrap(att)]))
            .map_err(js_err)?;
        pass.end();
        self.queue.submit(&[enc.finish()]);
        self.push_texture_view(
            &view,
            TexInfo {
                tex: Some(tex),
                label: "white".into(),
                width: 1,
                height: 1,
                mips: 1,
                uploads: 1,
                linear: true,
            },
        );
        Ok(())
    }

    fn push_texture_view(&mut self, view: &GpuTextureView, info: TexInfo) -> TextureId {
        self.tex_views.push(view.clone());
        self.tex_objects.push(info);
        // Public ids are 0-based and exclude the internal white texture.
        TextureId((self.tex_views.len() as u32).saturating_sub(2))
    }

    /// Bind group for a (base, normal, height) combination of internal texture indices.
    fn material_group(&mut self, key: (u32, u32, u32)) -> GpuBindGroup {
        if let Some(g) = self.mat_groups.get(&key) {
            return g.clone();
        }
        let view = |i: u32| self.tex_views.get(i as usize).unwrap_or(&self.tex_views[0]).clone();
        let base = view(key.0);
        let normal = if key.1 == u32::MAX { self.flat_normal.clone().unwrap_or_else(|| view(0)) } else { view(key.1) };
        let height = if key.2 == u32::MAX { view(0) } else { view(key.2) };
        let g = self.device.create_bind_group(&GpuBindGroupDescriptor::new(
            &[
                GpuBindGroupEntry::new_with_gpu_texture_view(0, &base),
                GpuBindGroupEntry::new(1, &self.color_sampler),
                GpuBindGroupEntry::new_with_gpu_texture_view(2, &normal),
                GpuBindGroupEntry::new_with_gpu_texture_view(3, &height),
            ],
            &self.bgl_mat,
        ));
        self.mat_groups.insert(key, g.clone());
        g
    }

    /// Creates an empty sRGB color texture with a full mip chain (fill it with [`Self::update_texture`]).
    pub fn create_texture(&mut self, w: u32, h: u32, label: &str) -> Result<TextureId, String> {
        self.create_texture_fmt(w, h, label, false)
    }

    /// Like [`Self::create_texture`]; `linear` = data texture (normal/height maps, no sRGB decode).
    pub fn create_texture_fmt(&mut self, w: u32, h: u32, label: &str, linear: bool) -> Result<TextureId, String> {
        let (w, h) = (w.max(1), h.max(1));
        let mips = 32 - w.max(h).leading_zeros();
        let desc = GpuTextureDescriptor::new(
            if linear { GpuTextureFormat::Rgba8unorm } else { GpuTextureFormat::Rgba8unormSrgb },
            &nums(&[w, h]),
            gpu_texture_usage::TEXTURE_BINDING
                | gpu_texture_usage::COPY_DST
                | gpu_texture_usage::RENDER_ATTACHMENT,
        );
        desc.set_label(label);
        desc.set_mip_level_count(mips);
        let tex = self.device.create_texture(&desc).map_err(js_err)?;
        let view = tex.create_view().map_err(js_err)?;
        Ok(self.push_texture_view(
            &view,
            TexInfo {
                tex: Some(tex),
                label: label.into(),
                width: w,
                height: h,
                mips,
                uploads: 0,
                linear,
            },
        ))
    }

    /// Replaces mip 0 of `id` with `bmp` (must match the texture size) and regenerates the mips.
    pub fn update_texture(&mut self, id: TextureId, bmp: &ImageBitmap) -> Result<(), String> {
        let info = self
            .tex_objects
            .get_mut(id.0 as usize + 1)
            .ok_or("unknown texture")?;
        let tex = info.tex.clone().ok_or("texture has no GPU object")?;
        let (w, h) = (bmp.width().min(info.width), bmp.height().min(info.height));
        let src = GpuCopyExternalImageSourceInfo::new(bmp);
        let dst = GpuCopyExternalImageDestInfo::new(&tex);
        self.queue
            .copy_external_image_to_texture_with_u32_sequence(&src, &dst, &nums(&[w, h]))
            .map_err(js_err)?;
        info.uploads += 1;
        let (mips, linear) = (info.mips, info.linear);
        self.generate_mips(&tex, mips, linear)
    }

    /// Downsamples level i-1 into level i for every mip (render passes, linear filter).
    fn generate_mips(&self, tex: &GpuTexture, mips: u32, linear: bool) -> Result<(), String> {
        if mips <= 1 {
            return Ok(());
        }
        let enc = self.device.create_command_encoder();
        let level_view = |level: u32| {
            let d = GpuTextureViewDescriptor::new();
            d.set_base_mip_level(level);
            d.set_mip_level_count(1);
            tex.create_view_with_descriptor(&d).map_err(js_err)
        };
        for level in 1..mips {
            let src = level_view(level - 1)?;
            let dst = level_view(level)?;
            let bg = self.device.create_bind_group(&GpuBindGroupDescriptor::new(
                &[
                    GpuBindGroupEntry::new_with_gpu_texture_view(0, &src),
                    GpuBindGroupEntry::new(1, &self.mip_sampler),
                ],
                &self.bgl1,
            ));
            let att = GpuRenderPassColorAttachment::new_with_gpu_texture_view(
                GpuLoadOp::Clear,
                GpuStoreOp::Store,
                &dst,
            );
            let pass = enc
                .begin_render_pass(&GpuRenderPassDescriptor::new(&[JsNullable::wrap(att)]))
                .map_err(js_err)?;
            pass.set_pipeline(if linear { &self.mip_pipeline_linear } else { &self.mip_pipeline });
            pass.set_bind_group(0, Some(&bg));
            pass.draw(3);
            pass.end();
        }
        self.queue.submit(&[enc.finish()]);
        Ok(())
    }

    /// Uploads a decoded `ImageBitmap` (from `createImageBitmap`) as an sRGB color texture with mips.
    pub fn add_texture_bitmap(
        &mut self,
        bmp: &ImageBitmap,
        label: &str,
    ) -> Result<TextureId, String> {
        let id = self.create_texture(bmp.width(), bmp.height(), label)?;
        self.update_texture(id, bmp)?;
        Ok(id)
    }

    /// Uploads raw RGBA8 sRGB pixels (`w * h * 4` bytes, top-left first) and builds mips.
    pub fn add_texture_rgba(&mut self, w: u32, h: u32, data: &[u8], label: &str) -> Result<TextureId, String> {
        self.add_texture_rgba_fmt(w, h, data, label, false)
    }

    /// Uploads raw RGBA8 *linear* pixels (normal / height maps) and builds mips.
    pub fn add_texture_rgba_linear(&mut self, w: u32, h: u32, data: &[u8], label: &str) -> Result<TextureId, String> {
        self.add_texture_rgba_fmt(w, h, data, label, true)
    }

    fn add_texture_rgba_fmt(&mut self, w: u32, h: u32, data: &[u8], label: &str, linear: bool) -> Result<TextureId, String> {
        if data.len() != (w * h * 4) as usize {
            return Err(format!(
                "add_texture_rgba: {} bytes for {w}x{h}",
                data.len()
            ));
        }
        let id = self.create_texture_fmt(w, h, label, linear)?;
        let info = &mut self.tex_objects[id.0 as usize + 1];
        let tex = info.tex.clone().ok_or("texture has no GPU object")?;
        info.uploads += 1;
        let mips = info.mips;
        let dst = GpuTexelCopyTextureInfo::new(&tex);
        let layout = GpuTexelCopyBufferLayout::new();
        layout.set_bytes_per_row(w * 4);
        layout.set_rows_per_image(h);
        self.queue
            .write_texture_with_u8_slice_and_u32_sequence(&dst, data, &layout, &nums(&[w, h]))
            .map_err(js_err)?;
        self.generate_mips(&tex, mips, linear)?;
        Ok(id)
    }

    /// Texture metadata (size, mips, upload count) for AI inspection.
    pub fn texture_infos(&self) -> &[TexInfo] {
        &self.tex_objects[1..]
    }

    /// Adds `data` plus automatically simplified versions (vertex clustering). `ratios` are
    /// target grid resolutions relative to LOD 0 (e.g. `[0.5, 0.25]`); switch distances are
    /// derived from the mesh size. Attach the returned [`Lod`] to the entity next to its `MeshRenderer`.
    pub fn add_mesh_with_lods(
        &mut self,
        name: &str,
        data: &MeshData,
        ratios: &[f32],
    ) -> (MeshId, Lod) {
        let base = self.add_mesh(name, data);
        let (_, radius) = data.bounds();
        let mut meshes = vec![base];
        // Vertex clustering visibly deforms small detailed models (faces, caps), so the first
        // switch happens late: at 40x the bounding radius (a 0.4 m head keeps full detail to 16 m).
        let mut distances = vec![radius.max(0.25) * AUTO_LOD_FACTOR];
        for (i, r) in ratios.iter().enumerate() {
            let cells = (32.0 * r).round().max(2.0) as u32;
            let simple = data.simplify_cluster(cells);
            if simple.indices.is_empty() || simple.indices.len() >= data.indices.len() {
                continue;
            }
            meshes.push(self.add_mesh(&format!("{name}#lod{}", i + 1), &simple));
            distances.push(radius.max(0.25) * AUTO_LOD_FACTOR * 2f32.powi(i as i32 + 1));
        }
        distances.pop(); // last level has no upper bound
        (
            base,
            Lod {
                meshes,
                distances,
                cull_beyond: false,
            },
        )
    }

    pub fn add_mesh(&mut self, name: &str, data: &MeshData) -> MeshId {
        let vbytes: &[u8] = bytemuck::cast_slice(&data.vertices);
        let ibytes: &[u8] = bytemuck::cast_slice(&data.indices);
        let mk = |bytes: &[u8], usage: u32| {
            let size = (bytes.len().max(4) as u32 + 3) & !3;
            let buf = self
                .device
                .create_buffer(&GpuBufferDescriptor::new(
                    size,
                    usage | gpu_buffer_usage::COPY_DST,
                ))
                .expect("create_buffer");
            let mut padded = bytes.to_vec();
            padded.resize(size as usize, 0);
            self.queue
                .write_buffer_with_u32_and_u8_slice(&buf, 0, &padded)
                .expect("write_buffer");
            buf
        };
        let vb = mk(vbytes, gpu_buffer_usage::VERTEX);
        let ib = mk(ibytes, gpu_buffer_usage::INDEX);
        vb.set_label(name);
        let (c, r) = data.bounds();
        let gm = GpuMesh {
            name: name.to_string(),
            vb,
            ib,
            index_count: data.indices.len() as u32,
            vertex_count: data.vertices.len() as u32,
            bounds_center: c,
            bounds_radius: r,
            cpu: (data.indices.len() <= 60_000).then(|| std::rc::Rc::new((data.vertices.iter().map(|v| v.pos).collect(), data.indices.clone()))),
            data: (data.indices.len() <= 60_000).then(|| std::rc::Rc::new(data.clone())),
        };
        if let Some(slot) = self.free_meshes.pop() {
            self.meshes[slot as usize] = gm;
            return MeshId(slot);
        }
        self.meshes.push(gm);
        MeshId(self.meshes.len() as u32 - 1)
    }

    /// Frees a mesh's GPU buffers; the id is reused by a later `add_mesh`. Nothing may still
    /// reference it (despawn its entities first).
    pub fn remove_mesh(&mut self, id: MeshId) {
        let Some(m) = self.meshes.get_mut(id.0 as usize) else { return };
        if m.index_count == 0 && m.vertex_count == 0 {
            return; // already free
        }
        m.vb.destroy();
        m.ib.destroy();
        m.index_count = 0;
        m.vertex_count = 0;
        m.cpu = None;
        m.data = None;
        m.name = "(free)".into();
        self.free_meshes.push(id.0);
    }

    /// Drops frame-to-frame history (Hi-Z occlusion) after a discontinuity such as an origin shift.
    pub fn invalidate_history(&mut self) {
        if let Some(c) = self.cull.as_mut() {
            c.hiz_valid = false;
        }
    }

    /// GPU grass settings (radius, spacing, height, colours, wind, splat layer).
    pub fn grass_settings(&mut self) -> &mut crate::vegetation::GrassSettings {
        &mut self.grass.settings
    }

    /// Texture view of a public texture id (internal: grass splat input).
    pub(crate) fn texture_view(&self, id: TextureId) -> Option<GpuTextureView> {
        self.tex_views.get(id.0 as usize + 1).cloned()
    }

    /// Live (non-freed) mesh count.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len() - self.free_meshes.len()
    }

    /// Replaces mip 0 of an RGBA texture created with `add_texture_rgba(_linear)` and regenerates mips.
    pub fn write_texture_rgba(&mut self, id: TextureId, w: u32, h: u32, data: &[u8]) -> Result<(), String> {
        let info = self.tex_objects.get_mut(id.0 as usize + 1).ok_or("no such texture")?;
        if info.width != w || info.height != h || data.len() != (w * h * 4) as usize {
            return Err(format!("write_texture_rgba: {w}x{h} ({} bytes) into {}x{}", data.len(), info.width, info.height));
        }
        let tex = info.tex.clone().ok_or("texture has no GPU object")?;
        info.uploads += 1;
        let (mips, linear) = (info.mips, info.linear);
        let dst = GpuTexelCopyTextureInfo::new(&tex);
        let layout = GpuTexelCopyBufferLayout::new();
        layout.set_bytes_per_row(w * 4);
        layout.set_rows_per_image(h);
        self.queue.write_texture_with_u8_slice_and_u32_sequence(&dst, data, &layout, &nums(&[w, h])).map_err(js_err)?;
        self.generate_mips(&tex, mips, linear)
    }

    /// Creates a render texture (sRGB colour + its own HDR/bloom targets). Returns its id and the
    /// texture id to use in materials. Assign a camera with [`Self::set_render_texture_camera`].
    pub fn create_render_texture(&mut self, w: u32, h: u32, opts: crate::render_texture::RenderTextureOpts, label: &str) -> Result<(crate::render_texture::RenderTextureId, TextureId), String> {
        let (w, h) = (w.max(1), h.max(1));
        let mips = 32 - w.max(h).leading_zeros();
        let desc = GpuTextureDescriptor::new(
            GpuTextureFormat::Rgba8unormSrgb,
            &nums(&[w, h]),
            gpu_texture_usage::TEXTURE_BINDING | gpu_texture_usage::RENDER_ATTACHMENT | gpu_texture_usage::COPY_SRC | gpu_texture_usage::COPY_DST,
        );
        desc.set_label(label);
        desc.set_mip_level_count(mips);
        let tex = self.device.create_texture(&desc).map_err(js_err)?;
        let full = tex.create_view().map_err(js_err)?;
        let vd = GpuTextureViewDescriptor::new();
        vd.set_base_mip_level(0);
        vd.set_mip_level_count(1);
        let view0 = tex.create_view_with_descriptor(&vd).map_err(js_err)?;
        let texture = self.push_texture_view(&full, TexInfo { tex: Some(tex.clone()), label: label.into(), width: w, height: h, mips, uploads: 0, linear: false });
        let targets = self.post.targets(&self.device, w, h, DEPTH_FORMAT, SAMPLES, label)?;
        self.rtts.push(crate::render_texture::RenderTexture {
            size: (w, h),
            texture,
            gpu: tex,
            view0,
            mips,
            targets,
            camera: None,
            opts,
            pending: true,
            renders: 0,
            label: label.to_string(),
            export: Default::default(),
            exporting: Default::default(),
        });
        Ok((crate::render_texture::RenderTextureId(self.rtts.len() as u32 - 1), texture))
    }

    /// Blob URL of the last PNG export of a render texture (see `Ctx::export_render_texture`).
    pub fn render_texture_url(&self, id: crate::render_texture::RenderTextureId) -> Option<String> {
        self.rtts.get(id.0 as usize).and_then(|r| r.export.borrow().clone())
    }

    /// Replaces a mesh's vertices (same count; skinning, morph targets, procedural meshes) and
    /// refreshes its bounds.
    pub fn update_mesh_vertices(&mut self, id: MeshId, vertices: &[Vertex]) {
        let Some(m) = self.meshes.get_mut(id.0 as usize) else { return };
        if vertices.len() as u32 != m.vertex_count {
            crate::log_warn!("update_mesh_vertices {}: {} vertices, mesh has {}", m.name, vertices.len(), m.vertex_count);
            return;
        }
        let _ = self.queue.write_buffer_with_u32_and_u8_slice(&m.vb, 0, bytemuck::cast_slice(vertices));
        let md = MeshData { vertices: vertices.to_vec(), indices: vec![] };
        let (c, r) = md.bounds();
        m.bounds_center = c;
        m.bounds_radius = r;
        if let Some(cpu) = &m.cpu {
            m.cpu = Some(std::rc::Rc::new((vertices.iter().map(|v| v.pos).collect(), cpu.1.clone())));
        }
        if let Some(d) = &m.data {
            m.data = Some(std::rc::Rc::new(MeshData { vertices: vertices.to_vec(), indices: d.indices.clone() }));
        }
    }

    /// Index list of a mesh (from the CPU copy; None for meshes over 60k indices).
    pub fn mesh_indices(&self, id: MeshId) -> Option<Vec<u32>> {
        self.meshes.get(id.0 as usize)?.cpu.as_ref().map(|c| c.1.clone())
    }

    pub fn add_material(&mut self, m: Material) -> MaterialId {
        self.materials.push(m);
        MaterialId(self.materials.len() as u32 - 1)
    }

    pub fn material(&self, id: MaterialId) -> Option<&Material> {
        self.materials.get(id.0 as usize)
    }
    pub fn material_mut(&mut self, id: MaterialId) -> Option<&mut Material> {
        self.materials.get_mut(id.0 as usize)
    }
    pub fn mesh(&self, id: MeshId) -> Option<&GpuMesh> {
        self.meshes.get(id.0 as usize)
    }
    pub fn counts(&self) -> (usize, usize, usize) {
        (
            self.meshes.len(),
            self.materials.len(),
            self.tex_views.len().saturating_sub(1),
        )
    }

    /// Canvas size in device pixels (the scene may render smaller, see `render_scale`).
    pub fn size(&self) -> (u32, u32) {
        self.canvas_size
    }

    /// Scene render size (canvas x `render_scale`).
    pub fn render_size(&self) -> (u32, u32) {
        self.size
    }

    /// Canvas size changed (device pixels).
    pub fn resize(&mut self, w: u32, h: u32) {
        self.canvas_size = (w.max(1), h.max(1));
        self.apply_render_scale();
    }

    /// Recreates the scene targets when the canvas size or `render_scale` changed.
    fn apply_render_scale(&mut self) {
        let s = self.settings.render_scale.clamp(0.25, 1.0);
        let (cw, ch) = self.canvas_size;
        let (w, h) = (((cw as f32 * s).round() as u32).max(1), ((ch as f32 * s).round() as u32).max(1));
        self.resize_targets(w, h);
    }

    /// Adaptive resolution: called once per frame with the frame interval. Every 30 frames the
    /// average GPU time (timestamp queries; else the frame interval) is compared with the budget:
    /// over it the scale drops by 10 %, well under it rises by 5 % (hysteresis, never above 1).
    pub fn adapt_resolution(&mut self, frame_dt: f32) {
        if !self.settings.auto_resolution {
            return;
        }
        let gpu = self.stats.gpu_ms.filter(|g| g.samples > 0).map(|g| g.total);
        self.adapt.0 += gpu.unwrap_or(0.0);
        self.adapt.1 += frame_dt * 1000.0;
        self.adapt.2 += 1;
        if self.adapt.2 < 30 {
            return;
        }
        let n = self.adapt.2 as f32;
        let (avg_gpu, avg_frame) = (self.adapt.0 / n, self.adapt.1 / n);
        self.adapt = (0.0, 0.0, 0);
        let budget = self.settings.frame_budget_ms.max(2.0);
        let (over, under) = match gpu {
            Some(_) => (avg_gpu > budget, avg_gpu < budget * 0.8),
            None => (avg_frame > 18.5, avg_frame < 17.2),
        };
        let s = self.settings.render_scale;
        let next = if over { (s * 0.9).max(self.settings.min_render_scale) } else if under { (s * 1.05).min(1.0) } else { s };
        if (next - s).abs() > 0.004 {
            self.settings.render_scale = next;
            crate::log_info!("adaptive resolution: scale {:.2} -> {:.2} (gpu {:.1} ms, frame {:.1} ms, budget {:.1} ms)", s, next, avg_gpu, avg_frame, budget);
        }
    }

    fn resize_targets(&mut self, w: u32, h: u32) {
        if self.size == (w, h) && self.msaa_view.is_some() {
            return;
        }
        self.size = (w, h);
        match self.post.targets(&self.device, w, h, DEPTH_FORMAT, SAMPLES, "canvas") {
            Ok(t) => {
                self.msaa_view = Some(t.msaa.clone());
                self.depth_view = Some(t.depth.clone());
                self.targets = Some(t);
            }
            Err(e) => crate::log_error!("render targets {w}x{h}: {e}"),
        }
        self.bind0_t = None;
        if let (Some(c), Some(d)) = (&mut self.cull, &self.depth_view) {
            if let Err(e) = c.resize(&self.device, w, h, d) {
                crate::log_warn!("hi-z resize: {e}");
            }
        }
    }

    fn ensure_instance_capacity(&mut self, n: u32) -> Result<(), String> {
        if n <= self.instance_cap {
            return Ok(());
        }
        let mut cap = self.instance_cap;
        while cap < n {
            cap *= 2;
        }
        self.instance_buf = Self::make_instance_buffer(&self.device, cap)?;
        self.instance_cap = cap;
        self.visible_buf = Self::make_visible_buffer(&self.device, cap)?;
        if let Some(c) = &mut self.cull {
            c.invalidate_bindings();
        }
        self.rebuild_bind0();
        Ok(())
    }

    /// Camera (view, proj, position) of the first entity with `Camera` + `GlobalTransform`.
    pub fn camera_matrices(&self, world: &World) -> Option<(Mat4, Mat4, Vec3)> {
        let aspect = self.size.0 as f32 / self.size.1.max(1) as f32;
        world
            .query::<(&Camera, &GlobalTransform)>()
            .without::<&crate::render_texture::RenderTextureCamera>()
            .iter()
            .next()
            .map(|(c, g)| {
                let view = (Mat4::from_translation(self.settings.camera_jitter) * g.0).inverse();
                let proj = Mat4::perspective_rh(c.fov_y_deg.to_radians(), aspect, c.near * (1.0 + self.settings.depth_jitter), c.far);
                (view, proj, g.translation())
            })
    }

    /// (view, proj, camera position, near, far, layer mask) of the main camera or a render texture's.
    fn view_setup(&self, world: &World, rtt: Option<usize>) -> Option<(Mat4, Mat4, Vec3, f32, f32, u32)> {
        let (cam, aspect) = match rtt {
            None => {
                let e = world.query::<(hecs::Entity, &Camera)>().without::<&crate::render_texture::RenderTextureCamera>().iter().next().map(|(e, _)| e)?;
                (e, self.size.0 as f32 / self.size.1.max(1) as f32)
            }
            Some(i) => {
                let rt = &self.rtts[i];
                (rt.camera?, rt.size.0 as f32 / rt.size.1.max(1) as f32)
            }
        };
        let c = *world.get::<&Camera>(cam).ok()?;
        let g = world.get::<&GlobalTransform>(cam).ok()?.0;
        let near = c.near * (1.0 + self.settings.depth_jitter);
        let view = (Mat4::from_translation(self.settings.camera_jitter) * g).inverse();
        let proj = Mat4::perspective_rh(c.fov_y_deg.to_radians(), aspect, near, c.far);
        Some((view, proj, g.w_axis.truncate(), c.near, c.far, c.layers))
    }

    /// Seconds for shader animation; stops while `settings.freeze_time` is set.
    fn shader_time(&mut self) -> f32 {
        let now = ((crate::log::now_ms() - self.start_ms) / 1000.0) as f32;
        if self.settings.freeze_time {
            *self.frozen_time.get_or_insert(now)
        } else {
            self.frozen_time = None;
            now
        }
    }

    /// Renders due render textures, then the main view to the canvas.
    pub fn render(&mut self, world: &World) -> Result<(), String> {
        self.apply_render_scale();
        for i in 0..self.rtts.len() {
            if !self.rtts[i].due() {
                continue;
            }
            let saved = self.settings.clone();
            self.rtts[i].apply(&mut self.settings);
            let r = self.render_view(world, Some(i));
            self.settings = saved;
            match r {
                Ok(()) => {
                    let rt = &mut self.rtts[i];
                    rt.renders += 1;
                    rt.pending = false;
                    if rt.mips > 1 {
                        let gpu = rt.gpu.clone();
                        let mips = rt.mips;
                        if let Err(e) = self.generate_mips(&gpu, mips, false) {
                            crate::log_warn!("render texture mips: {e}");
                        }
                    }
                }
                Err(e) => {
                    crate::log_error!("render texture {}: {e}", self.rtts[i].label);
                    self.rtts[i].pending = false;
                }
            }
        }
        self.render_view(world, None)
    }

    /// One view: the main camera into the canvas (`rtt` None) or a render texture's camera into it.
    fn render_view(&mut self, world: &World, rtt: Option<usize>) -> Result<(), String> {
        let main = rtt.is_none();
        let (msaa, depth, hdr_view, scene_color) = match rtt {
            None => match (self.msaa_view.clone(), self.depth_view.clone(), self.targets.as_ref()) {
                (Some(m), Some(d), Some(t)) => (m, d, t.hdr_view.clone(), t.scene_color.clone()),
                _ => return Ok(()),
            },
            Some(i) => {
                let t = &self.rtts[i].targets;
                (t.msaa.clone(), t.depth.clone(), t.hdr_view.clone(), t.scene_color.clone())
            }
        };
        let Some((view, proj, cam_pos, near, far, layers)) = self.view_setup(world, rtt) else {
            return if main { Err("no entity with Camera + Transform in the world".into()) } else { Ok(()) };
        };
        let view_proj = proj * view;
        if main {
            self.last_view_proj = view_proj;
        }
        let cam_fwd = -view.inverse().z_axis.truncate();

        // --- Shadow cascades: bounding sphere per view-depth slice (rotation-stable), texel-snapped
        let s = &self.settings;
        let dir = s.sun_dir.normalize_or(Vec3::NEG_Y);
        let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
        let n_casc = s.shadow_cascades.clamp(1, MAX_CASCADES as u32) as usize;
        let shadow_far = s.shadow_distance.max(near * 4.0).min(far);
        let (tan_x, tan_y) = (1.0 / proj.x_axis.x, 1.0 / proj.y_axis.y);
        let inv_view = view.inverse();
        let rot = Mat4::look_at_rh(Vec3::ZERO, dir, up);
        let mut cascades = [[[0.0f32; 4]; 4]; MAX_CASCADES];
        let mut cascade_mats = [Mat4::IDENTITY; MAX_CASCADES];
        let mut splits = [0.0f32; 4];
        let mut texels = [0.0f32; 4];
        // (light view matrix, half size, depth range) per cascade, for caster culling
        let mut cascade_boxes: Vec<(Mat4, f32, f32)> = vec![];
        for k in 0..n_casc {
            let f = (k + 1) as f32 / n_casc as f32;
            let lin = near + (shadow_far - near) * f;
            let log = near * (shadow_far / near).powf(f);
            let z1 = lin + (log - lin) * s.shadow_split_lambda;
            let z0 = if k == 0 { near } else { splits[k - 1] };
            splits[k] = z1;
            let mut center = Vec3::ZERO;
            let mut corners = [Vec3::ZERO; 8];
            for (ci, c) in corners.iter_mut().enumerate() {
                let z = if ci < 4 { z0 } else { z1 };
                let sx = if ci & 1 == 0 { -1.0 } else { 1.0 };
                let sy = if ci & 2 == 0 { -1.0 } else { 1.0 };
                *c = inv_view.transform_point3(Vec3::new(sx * tan_x * z, sy * tan_y * z, -z));
                center += *c / 8.0;
            }
            let r = corners.iter().map(|c| c.distance(center)).fold(0.0f32, f32::max);
            let r = (r * 16.0).ceil() / 16.0;
            let texel = 2.0 * r / self.shadow_size as f32;
            let mut c_ls = rot.transform_point3(center);
            c_ls.x = (c_ls.x / texel).floor() * texel;
            c_ls.y = (c_ls.y / texel).floor() * texel;
            let center = rot.inverse().transform_point3(c_ls);
            let back = r + 120.0; // casters up to 120 m beyond the slice towards the sun
            let lv = Mat4::look_at_rh(center - dir * back, center, up);
            let lp = Mat4::orthographic_rh(-r, r, -r, r, 0.0, back + r);
            cascade_mats[k] = lp * lv;
            cascades[k] = cascade_mats[k].to_cols_array_2d();
            texels[k] = texel;
            cascade_boxes.push((lv, r, back + r));
        }
        let mut cascade_data = vec![0u8; CASCADE_SLOT as usize * MAX_CASCADES];
        for (k, m) in cascade_mats.iter().enumerate() {
            cascade_data[k * CASCADE_SLOT as usize..k * CASCADE_SLOT as usize + 64].copy_from_slice(bytemuck::bytes_of(&m.to_cols_array()));
        }
        self.queue.write_buffer_with_u32_and_u8_slice(&self.cascade_buf, 0, &cascade_data).map_err(js_err)?;

        // --- Clustered lights -----------------------------------------------------------------
        let cluster_far = self.settings.light_cluster_far.min(far).max(near * 2.0);
        let cl = crate::lights::build(world, view, proj, &frustum_planes(&view_proj), cam_pos, near, cluster_far);
        if self.light_bufs.ensure(&self.device, cl.lights.len().max(1) as u32, cl.indices.len().max(1) as u32)? {
            self.rebuild_bind0();
        }
        if !cl.lights.is_empty() {
            self.queue.write_buffer_with_u32_and_u8_slice(&self.light_bufs.lights, 0, bytemuck::cast_slice(&cl.lights)).map_err(js_err)?;
        }
        self.queue.write_buffer_with_u32_and_u8_slice(&self.light_bufs.grid, 0, bytemuck::cast_slice(&cl.grid)).map_err(js_err)?;
        if !cl.indices.is_empty() {
            self.queue.write_buffer_with_u32_and_u8_slice(&self.light_bufs.indices, 0, bytemuck::cast_slice(&cl.indices)).map_err(js_err)?;
        }
        let n_lights = cl.lights.len();
        let light_stats = cl.stats;
        let (vw, vh) = match rtt {
            None => self.size,
            Some(i) => self.rtts[i].size,
        };
        let shader_t = self.shader_time();
        let s = &self.settings;
        let globals = Globals {
            view_proj: view_proj.to_cols_array_2d(),
            cascades,
            camera_pos: cam_pos.extend(1.0).into(),
            sun_dir: dir.extend(s.sun_intensity).into(),
            sun_color: s.sun_color.extend(1.0).into(),
            sky_color: s.sky_color.extend(1.0).into(),
            ground_color: s.ground_color.extend(1.0).into(),
            counts: [n_lights as u32, s.shadows as u32, s.debug_view, n_casc as u32],
            // near must match the projection (depth_jitter) or water depth would shift with it
            time: [shader_t, near * (1.0 + s.depth_jitter), far, 0.0],
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            fog: [s.fog_density.max(0.0), s.fog_height_falloff.max(0.0), s.fog_start.max(0.0), s.fog_max.clamp(0.0, 1.0)],
            fog_color: s.fog_color.map(|c| c.extend(1.0)).unwrap_or(Vec4::ZERO).into(),
            sky_zenith: s.sky_zenith.extend(0.0).into(),
            sky_horizon: s.sky_horizon.extend(s.sun_size_deg.to_radians()).into(),
            cam_right: view.inverse().x_axis.truncate().normalize_or(Vec3::X).extend(0.0).into(),
            cam_up: view.inverse().y_axis.truncate().normalize_or(Vec3::Y).extend(0.0).into(),
            view: view.to_cols_array_2d(),
            cascade_splits: splits,
            cascade_texel: texels,
            cluster: [near, cluster_far, crate::lights::CLUSTER_Z as f32 / (cluster_far / near).ln(), 0.0],
            screen: [vw as f32, vh as f32, 1.0 / vw.max(1) as f32, 1.0 / vh.max(1) as f32],
            sky_sun: self.sky.sun.normalize_or(Vec3::Y).extend(1.0).into(),
            sky_sun_color: self.sky.sun_radiance.extend(self.sky.night).into(),
            sky_moon: self.sky.moon.normalize_or(Vec3::NEG_Y).extend(self.sky.moon_brightness).into(),
            sky_params: [s.sky_mode as f32, s.clouds.clamp(0.0, 1.0), shader_t * s.cloud_speed * 0.004, s.stars.max(0.0)],
            skybox: [
                if self.skybox.is_some() { s.skybox_blend + (s.skybox_night_blend - s.skybox_blend) * self.sky.night } else { 0.0 },
                s.skybox_rotation_deg.to_radians(),
                s.skybox_intensity,
                self.skybox.is_some() as u32 as f32,
            ],
            misc: [s.water_ssr as u32 as f32, s.water_refraction.max(0.0), s.water_caustics.max(0.0), s.atmosphere_intensity.max(0.0)],
        };
        self.queue
            .write_buffer_with_u32_and_u8_slice(&self.globals_buf, 0, bytemuck::bytes_of(&globals))
            .map_err(js_err)?;

        // --- Instances -------------------------------------------------------------------------
        // Buffer layout: [opaque/mask candidates sorted by (sides, texture, mesh)]
        //             ++ [blended, frustum-culled on the CPU, sorted back to front]
        //             ++ [shadow casters sorted by (mask, texture, mesh)]
        // GPU path: opaque candidates are culled by cull.wgsl; CPU path: frustum test here.
        // GPU culling + Hi-Z belong to the main view (render textures use the CPU path)
        let gpu = main && self.settings.gpu_culling && self.cull.is_some();
        let planes = frustum_planes(&view_proj);
        let cam_fwd_n = cam_fwd.normalize_or(Vec3::NEG_Z);
        // (double_sided, texture, mesh, instance, bounds, model index)
        // (pipeline 0..3, texture key, mesh, instance, bounds, model index); opaque pipelines sort first
        let mut visible: Vec<(u8, u32, u32, InstanceRaw, [f32; 4], u32)> = vec![];
        // (view depth, double_sided, texture, mesh, instance, model index)
        let mut blended: Vec<(f32, bool, u32, u32, InstanceRaw, u32)> = vec![];
        let mut has_water = false;
        // (alpha-tested, texture, mesh, instance, model index, bounding sphere)
        let mut casters: Vec<(bool, u32, u32, InstanceRaw, u32, Vec3, f32)> = vec![];
        // Model matrices; normal matrices (inverse-transpose) are computed afterwards with the job system.
        let mut models: Vec<Mat4> = vec![];
        let fallback_mat = Material::default();
        // Material bind groups are keyed by (base, normal, height); `tex` below is an index into key_list.
        let mut key_ids: std::collections::HashMap<(u32, u32, u32), u32> = std::collections::HashMap::new();
        let mut key_list: Vec<(u32, u32, u32)> = vec![];
        let mut culled = 0u32;
        let mut lod_counts = [0u32; 4];
        for (g, mr, lod) in world.query::<(&GlobalTransform, &MeshRenderer, Option<&Lod>)>().iter() {
            if layers & (1u32 << (mr.layer & 31)) == 0 {
                continue;
            }
            let model = g.0;
            let scale = model.x_axis.truncate().length().max(model.y_axis.truncate().length()).max(model.z_axis.truncate().length());
            let mut mesh_id = mr.mesh;
            if let Some(lod) = lod {
                match lod.pick((model.w_axis.truncate() - cam_pos).length() / scale.max(1e-4) / self.settings.lod_bias.max(0.01)) {
                    Some((level, m)) => {
                        mesh_id = m;
                        lod_counts[level.min(3)] += 1;
                    }
                    None => {
                        culled += 1;
                        continue;
                    }
                }
            }
            let Some(mesh) = self.meshes.get(mesh_id.0 as usize).filter(|m| m.index_count > 0) else { continue };
            let mat = self.materials.get(mr.material.0 as usize).unwrap_or(&fallback_mat);
            let shading = if mat.terrain { -1.0 } else if mat.water { 3.0 } else if mat.ui { 2.0 } else if mat.unlit { 1.0 } else { 0.0 };
            let alpha = match mat.alpha_mode {
                AlphaMode::Opaque => 0.0,
                AlphaMode::Mask(c) => c.clamp(0.001, 1.0),
                AlphaMode::Blend => -1.0,
            };
            let raw = InstanceRaw {
                model: model.to_cols_array_2d(),
                normal: [[0.0; 4]; 4], // filled below (jobs::par_map)
                base_color: mat.base_color.into(),
                emissive_metallic: if mat.terrain {
                    mat.terrain_params
                } else if mat.water {
                    mat.emissive.extend(mat.water_foam).into()
                } else {
                    mat.emissive.extend(mat.metallic).into()
                },
                params: [mat.roughness, if mat.water { mat.water_clarity } else { mat.checker }, shading, alpha],
                maps: if mat.terrain {
                    mat.terrain_tiling
                } else if mat.water {
                    [mat.water_vertex_spacing, mat.wave_height, mat.water_chop, mat.wave_scale]
                } else {
                    [
                        if mat.normal_texture.is_some() { mat.normal_scale } else { 0.0 },
                        if mat.height_texture.is_some() { mat.displacement } else { 0.0 },
                        if mat.height_texture.is_some() { mat.parallax } else { 0.0 },
                        0.0,
                    ]
                },
            };
            let key = (
                mat.texture.map(|t| t.0 + 1).unwrap_or(0),
                mat.normal_texture.map(|t| t.0 + 1).unwrap_or(u32::MAX),
                mat.height_texture.map(|t| t.0 + 1).unwrap_or(u32::MAX),
            );
            let tex = *key_ids.entry(key).or_insert_with(|| {
                key_list.push(key);
                key_list.len() as u32 - 1
            });
            let mi = models.len() as u32;
            models.push(model);
            let is_blend = mat.alpha_mode == AlphaMode::Blend;
            let is_mask = matches!(mat.alpha_mode, AlphaMode::Mask(_));
            let center = model.transform_point3(mesh.bounds_center);
            let radius = mesh.bounds_radius * scale;
            if mr.cast_shadows && !is_blend && self.settings.shadows {
                casters.push((is_mask, if is_mask { tex } else { 0 }, mesh_id.0, raw, mi, center, radius));
            }
            let in_frustum = planes.iter().all(|p| p.truncate().dot(center) + p.w >= -radius);
            if is_blend {
                if in_frustum {
                    has_water |= mat.water;
                    blended.push(((center - cam_pos).dot(cam_fwd_n), mat.double_sided, tex, mesh_id.0, raw, mi));
                } else {
                    culled += 1;
                }
            } else if gpu || in_frustum {
                let pipe = (if is_mask || mat.ui { 2 } else { 0 }) + mat.double_sided as u8;
                visible.push((pipe, tex, mesh_id.0, raw, [center.x, center.y, center.z, radius], mi));
            } else {
                culled += 1;
            }
        }
        let normals = crate::jobs::par_map(&models, |m| m.inverse().transpose().to_cols_array_2d());
        for v in visible.iter_mut() {
            v.3.normal = normals[v.5 as usize];
        }
        for b in blended.iter_mut() {
            b.4.normal = normals[b.5 as usize];
        }
        for c in casters.iter_mut() {
            c.3.normal = normals[c.4 as usize];
        }
        visible.sort_by_key(|v| (v.0, v.1, v.2));
        if self.settings.reverse_draw_order {
            visible.reverse();
        }
        blended.sort_by(|a, b| b.0.total_cmp(&a.0)); // far -> near
        casters.sort_by_key(|c| (c.0, c.1, c.2));
        // per cascade: the casters whose bounding sphere touches the cascade's light-space box
        let cascade_casters: Vec<Vec<usize>> = cascade_boxes
            .iter()
            .map(|(lv, r, depth)| {
                (0..casters.len())
                    .filter(|&i| {
                        let (c, rad) = (casters[i].5, casters[i].6);
                        let p = lv.transform_point3(c);
                        p.x.abs() <= r + rad && p.y.abs() <= r + rad && -p.z >= -rad && -p.z <= depth + rad
                    })
                    .collect()
            })
            .collect();

        // Opaque batches: runs of equal (sides, texture, mesh) -> one (indirect) draw each.
        let mut batches: Vec<(u8, u32, u32, u32, u32)> = vec![]; // (pipeline, texture, mesh, first, count)
        for (i, v) in visible.iter().enumerate() {
            match batches.last_mut() {
                Some(b) if b.0 == v.0 && b.1 == v.1 && b.2 == v.2 => b.4 += 1,
                _ => batches.push((v.0, v.1, v.2, i as u32, 1)),
            }
        }
        // Blended runs: only *adjacent* equal keys merge, so the back-to-front order is kept.
        let blend_base = visible.len() as u32;
        let mut blend_runs: Vec<(bool, u32, u32, u32, u32)> = vec![];
        for (i, b) in blended.iter().enumerate() {
            match blend_runs.last_mut() {
                Some(r) if r.0 == b.1 && r.1 == b.2 && r.2 == b.3 => r.4 += 1,
                _ => blend_runs.push((b.1, b.2, b.3, blend_base + i as u32, 1)),
            }
        }

        let shadow_instances: usize = cascade_casters.iter().map(|c| c.len()).sum();
        let total = (visible.len() + blended.len() + shadow_instances) as u32;
        self.ensure_instance_capacity(total.max(1))?;
        let mut data: Vec<InstanceRaw> = Vec::with_capacity(total as usize);
        data.extend(visible.iter().map(|v| v.3));
        data.extend(blended.iter().map(|b| b.4));
        let mut cascade_base = vec![];
        for list in &cascade_casters {
            cascade_base.push(data.len() as u32);
            data.extend(list.iter().map(|&i| casters[i].3));
        }
        if !data.is_empty() {
            self.queue.write_buffer_with_u32_and_u8_slice(&self.instance_buf, 0, bytemuck::cast_slice(&data)).map_err(js_err)?;
        }
        // visible_ids: identity for the CPU-drawn ranges (all opaque on the CPU path, blended always).
        let id_from = if gpu { blend_base } else { 0 };
        let id_to = blend_base + blended.len() as u32;
        if id_to > id_from {
            let ids: Vec<u32> = (id_from..id_to).collect();
            self.queue.write_buffer_with_u32_and_u8_slice(&self.visible_buf, id_from * 4, bytemuck::cast_slice(&ids)).map_err(js_err)?;
        }

        let mut stats = FrameStats {
            width: self.size.0,
            height: self.size.1,
            canvas: [self.canvas_size.0, self.canvas_size.1],
            render_scale: self.settings.render_scale,
            culled,
            visible: visible.len() as u32,
            transparent: blended.len() as u32,
            lod_levels: lod_counts,
            gpu_culling: gpu,
            lights: light_stats,
            cascades: splits[..n_casc].to_vec(),
            ..Default::default()
        };
        let groups: Vec<GpuBindGroup> = key_list.iter().map(|k| self.material_group(*k)).collect();
        let mut bind0_rt: Option<GpuBindGroup> = None;
        if self.bind0_t.is_none() || !main {
            let g = Some(self.device.create_bind_group(&GpuBindGroupDescriptor::new(
                &[
                    GpuBindGroupEntry::new_with_gpu_buffer(0, &self.globals_buf),
                    GpuBindGroupEntry::new_with_gpu_buffer(1, &self.instance_buf),
                    GpuBindGroupEntry::new_with_gpu_texture_view(2, &self.shadow_view),
                    GpuBindGroupEntry::new(3, &self.shadow_sampler),
                    GpuBindGroupEntry::new_with_gpu_buffer(4, &self.visible_buf),
                    GpuBindGroupEntry::new_with_gpu_texture_view(5, &depth),
                    GpuBindGroupEntry::new_with_gpu_buffer(6, &self.light_bufs.lights),
                    GpuBindGroupEntry::new_with_gpu_buffer(7, &self.light_bufs.grid),
                    GpuBindGroupEntry::new_with_gpu_buffer(8, &self.light_bufs.indices),
                    GpuBindGroupEntry::new_with_gpu_texture_view(9, &scene_color.1),
                    GpuBindGroupEntry::new_with_gpu_texture_view(10, &self.sky_gpu.lut_view),
                    GpuBindGroupEntry::new(11, &self.sky_gpu.sampler),
                    GpuBindGroupEntry::new_with_gpu_texture_view(12, &self.sky_gpu.skybox_view),
                ],
                &self.bgl0_t,
            )));
            if main {
                self.bind0_t = g;
            } else {
                bind0_rt = g;
            }
        }
        // adaptive quality: lower resolution also shortens the grass radius (vertex cost)
        let quality = ((self.settings.render_scale - self.settings.min_render_scale) / (1.0 - self.settings.min_render_scale).max(0.01)).clamp(0.0, 1.0);
        let grass_blades = if self.settings.grass && layers & 1 != 0 { self.grass.prepare(&self.device, &self.queue, cam_pos, shader_t, quality) } else { 0 };
        stats.grass_blades = grass_blades;
        // particles: upload [additive ++ alpha-blended]
        let (n_add, n_alpha) = if self.settings.particles {
            let (add, alpha) = crate::particles::collect(world, cam_pos);
            let n = (add.len() + alpha.len()) as u32;
            if n > 0 {
                self.post.ensure_particles(&self.device, n)?;
                let mut all = add;
                let na = all.len() as u32;
                all.extend(alpha);
                self.queue.write_buffer_with_u32_and_u8_slice(&self.post.particle_buf, 0, bytemuck::cast_slice(&all)).map_err(js_err)?;
                (na, n - na)
            } else {
                (0, 0)
            }
        } else {
            (0, 0)
        };
        stats.particles = n_add + n_alpha;
        if let Some(t) = self.timing.as_mut().filter(|_| main) {
            t.begin_frame();
        }
        let encoder = self.device.create_command_encoder();

        // --- Sky-view LUT (atmosphere), only when the atmosphere sky is on ---------------------
        if self.settings.sky_mode == 1 {
            let att = GpuRenderPassColorAttachment::new_with_gpu_texture_view(GpuLoadOp::Clear, GpuStoreOp::Store, &self.sky_gpu.lut_view);
            let desc = GpuRenderPassDescriptor::new(&[JsNullable::wrap(att)]);
            desc.set_label("sky-lut");
            let pass = encoder.begin_render_pass(&desc).map_err(js_err)?;
            pass.set_pipeline(&self.sky_gpu.pipeline);
            pass.set_bind_group(0, Some(&self.sky_gpu.group));
            pass.draw(3);
            pass.end();
        }

        // --- GPU culling (compute) ----------------------------------------------------------
        if gpu {
            let occlusion = self.settings.occlusion_culling;
            if let Some(c) = self.cull.as_mut() {
                c.ensure_capacity(&self.device, visible.len() as u32, batches.len() as u32)?;
                let mut items = Vec::with_capacity(visible.len());
                for (b, batch) in batches.iter().enumerate() {
                    for v in &visible[batch.3 as usize..(batch.3 + batch.4) as usize] {
                        items.push(crate::gpu_cull::CullItem { center_radius: v.4, batch: b as u32, _pad: [0; 3] });
                    }
                }
                let args: Vec<crate::gpu_cull::DrawArgs> = batches
                    .iter()
                    .map(|b| crate::gpu_cull::DrawArgs {
                        index_count: self.meshes[b.2 as usize].index_count,
                        instance_count: 0,
                        first_index: 0,
                        base_vertex: 0,
                        first_instance: b.3,
                    })
                    .collect();
                let params = crate::gpu_cull::CullParams {
                    planes: planes.map(|p| p.into()),
                    prev_view_proj: c.prev_view_proj.to_cols_array_2d(),
                    hiz_size: [c.hiz_size.0 as f32, c.hiz_size.1 as f32],
                    count: items.len() as u32,
                    flags: 1 | if occlusion && c.hiz_valid { 2 } else { 0 },
                    hiz_mips: c.hiz_mips,
                    _pad: [0; 3],
                };
                let ts = self.timing.as_mut().filter(|_| main).and_then(|t| t.compute_writes(crate::gpu_timing::CULL));
                c.encode_cull(&self.device, &self.queue, &encoder, &self.visible_buf, &params, &items, &args, ts)?;
                let k = *c.last_counters.borrow();
                stats.gpu_frustum_culled = k[0];
                stats.gpu_occluded = k[1];
                stats.gpu_visible = k[2];
            }
        } else if main {
            if let Some(c) = self.cull.as_mut() {
                c.hiz_valid = false;
            }
        }

        // --- Shadow passes: one per cascade (opaque depth-only, alpha-tested casters sample their texture)
        {
            let n_pass = n_casc;
            let ts = self.timing.as_mut().filter(|_| main).and_then(|t| t.split_writes(crate::gpu_timing::SHADOW));
            for (k, list) in cascade_casters.iter().enumerate().take(n_pass) {
                let att = GpuRenderPassDepthStencilAttachment::new_with_gpu_texture_view(&self.shadow_layers[k]);
                att.set_depth_clear_value(1.0);
                att.set_depth_load_op(GpuLoadOp::Clear);
                att.set_depth_store_op(GpuStoreOp::Store);
                let desc = GpuRenderPassDescriptor::new(&[]);
                desc.set_depth_stencil_attachment(&att);
                desc.set_label("shadow");
                if let Some((b, e)) = &ts {
                    if k == 0 && n_pass == 1 {
                        if let Some(i) = js_sys::Reflect::get(e, &"endOfPassWriteIndex".into()).ok().and_then(|v| v.as_f64()) {
                            b.set_end_of_pass_write_index(i as u32);
                        }
                        desc.set_timestamp_writes(b);
                    } else if k == 0 {
                        desc.set_timestamp_writes(b);
                    } else if k == n_pass - 1 {
                        desc.set_timestamp_writes(e);
                    }
                }
                let pass = encoder.begin_render_pass(&desc).map_err(js_err)?;
                if self.settings.shadows && !list.is_empty() {
                    pass.set_bind_group(0, Some(&self.bind0_shadow[k]));
                    let base = cascade_base[k];
                    let mut bound: Option<(bool, u32)> = None;
                    let mut i = 0usize;
                    while i < list.len() {
                        let c0 = &casters[list[i]];
                        let (mask, tex, mesh_id) = (c0.0, c0.1, c0.2);
                        let mut j = i;
                        while j < list.len() && casters[list[j]].0 == mask && casters[list[j]].1 == tex && casters[list[j]].2 == mesh_id {
                            j += 1;
                        }
                        if bound.map(|b| b.0) != Some(mask) {
                            pass.set_pipeline(if mask { &self.shadow_mask_pipeline } else { &self.shadow_pipeline });
                        }
                        if mask && bound != Some((true, tex)) {
                            pass.set_bind_group(1, Some(&groups[tex as usize]));
                        }
                        bound = Some((mask, tex));
                        let mesh = &self.meshes[mesh_id as usize];
                        pass.set_vertex_buffer(0, Some(&mesh.vb));
                        pass.set_index_buffer(&mesh.ib, GpuIndexFormat::Uint32);
                        pass.draw_indexed_with_instance_count_and_first_index_and_base_vertex_and_first_instance(mesh.index_count, (j - i) as u32, 0, 0, base + i as u32);
                        stats.shadow_draw_calls += 1;
                        i = j;
                    }
                }
                pass.end();
            }
        }

        // --- Opaque pass (indirect on the GPU path); keeps colour + depth for the blend pass ---
        {
            let att = GpuRenderPassColorAttachment::new_with_gpu_texture_view(GpuLoadOp::Clear, GpuStoreOp::Store, &msaa);
            if has_water {
                // water refracts and reflects the resolved opaque scene (copied below)
                att.set_resolve_target_gpu_texture_view(&hdr_view);
            }
            let cc = self.settings.clear_color; // linear HDR target, alpha 1 = tonemapped
            let clear_a = if rtt.is_some_and(|i| self.rtts[i].opts.transparent) { 0.0 } else { 1.0 };
            att.set_clear_value_gpu_color_dict(&GpuColorDict::new(clear_a, cc.z as f64, cc.y as f64, cc.x as f64));
            let datt = GpuRenderPassDepthStencilAttachment::new_with_gpu_texture_view(&depth);
            datt.set_depth_clear_value(1.0);
            datt.set_depth_load_op(GpuLoadOp::Clear);
            datt.set_depth_store_op(GpuStoreOp::Store);
            let desc = GpuRenderPassDescriptor::new(&[JsNullable::wrap(att)]);
            desc.set_depth_stencil_attachment(&datt);
            desc.set_label("opaque");
            if let Some(w) = self.timing.as_mut().filter(|_| main).and_then(|t| t.render_writes(crate::gpu_timing::OPAQUE)) {
                desc.set_timestamp_writes(&w);
            }
            let pass = encoder.begin_render_pass(&desc).map_err(js_err)?;
            pass.set_bind_group(0, Some(&self.bind0));
            let mut bound_tex = u32::MAX;
            let mut bound_pipe = usize::MAX;
            let args_buf = self.cull.as_ref().filter(|_| gpu).map(|c| c.args_buf.clone());
            for (b, &(two_sided, tex, mesh_id, first, count)) in batches.iter().enumerate() {
                let pi = two_sided as usize; // pipeline index 0..3
                if pi != bound_pipe {
                    pass.set_pipeline(&self.pipelines[pi]);
                    bound_pipe = pi;
                }
                if tex != bound_tex {
                    pass.set_bind_group(1, Some(&groups[tex as usize]));
                    bound_tex = tex;
                }
                let mesh = &self.meshes[mesh_id as usize];
                pass.set_vertex_buffer(0, Some(&mesh.vb));
                pass.set_index_buffer(&mesh.ib, GpuIndexFormat::Uint32);
                match &args_buf {
                    Some(args) => pass.draw_indexed_indirect_with_u32(args, b as u32 * 20),
                    None => pass.draw_indexed_with_instance_count_and_first_index_and_base_vertex_and_first_instance(mesh.index_count, count, 0, 0, first),
                }
                stats.draw_calls += 1;
                // GPU path: upper bound (candidates); the exact count lives on the GPU.
                stats.triangles += (mesh.index_count / 3) as u64 * count as u64;
            }
            if grass_blades > 0 {
                let mg = self.material_group((0, u32::MAX, u32::MAX));
                self.grass.draw(&pass, &self.bind0, &mg, grass_blades);
                stats.draw_calls += 1;
            }
            if self.settings.sky {
                // last: only pixels no geometry covered pass the depth test (less-equal at z = 1)
                pass.set_pipeline(&self.post.sky_pipeline);
                pass.set_bind_group(0, Some(&self.bind0));
                pass.draw(3);
            }
            pass.end();
        }

        if has_water {
            let src = GpuTexelCopyTextureInfo::new(&scene_color.2);
            let dst = GpuTexelCopyTextureInfo::new(&scene_color.0);
            let size: Vec<Number> = [vw, vh].iter().map(|x| Number::from(*x)).collect();
            encoder.copy_texture_to_texture_with_u32_sequence(&src, &dst, &size).map_err(js_err)?;
        }

        // --- Blend pass: back to front, depth read-only *and* bound as a texture (water) -----
        {
            let att = GpuRenderPassColorAttachment::new_with_gpu_texture_view(GpuLoadOp::Load, GpuStoreOp::Discard, &msaa);
            att.set_resolve_target_gpu_texture_view(&hdr_view);
            let datt = GpuRenderPassDepthStencilAttachment::new_with_gpu_texture_view(&depth);
            datt.set_depth_read_only(true);
            let desc = GpuRenderPassDescriptor::new(&[JsNullable::wrap(att)]);
            desc.set_depth_stencil_attachment(&datt);
            desc.set_label("blend");
            if let Some(w) = self.timing.as_mut().filter(|_| main).and_then(|t| t.render_writes(crate::gpu_timing::BLEND)) {
                desc.set_timestamp_writes(&w);
            }
            let pass = encoder.begin_render_pass(&desc).map_err(js_err)?;
            if let Some(b0) = if main { &self.bind0_t } else { &bind0_rt } {
                pass.set_bind_group(0, Some(b0));
            }
            let mut bound_tex = u32::MAX;
            let mut bound_pipe = usize::MAX;
            for &(two_sided, tex, mesh_id, first, count) in &blend_runs {
                let pi = 4 + two_sided as usize;
                if pi != bound_pipe {
                    pass.set_pipeline(&self.pipelines[pi]);
                    bound_pipe = pi;
                }
                if tex != bound_tex {
                    pass.set_bind_group(1, Some(&groups[tex as usize]));
                    bound_tex = tex;
                }
                let mesh = &self.meshes[mesh_id as usize];
                pass.set_vertex_buffer(0, Some(&mesh.vb));
                pass.set_index_buffer(&mesh.ib, GpuIndexFormat::Uint32);
                pass.draw_indexed_with_instance_count_and_first_index_and_base_vertex_and_first_instance(mesh.index_count, count, 0, 0, first);
                stats.draw_calls += 1;
                stats.transparent_draw_calls += 1;
                stats.triangles += (mesh.index_count / 3) as u64 * count as u64;
            }
            // particles: additive first (order-independent), then alpha-blended far -> near
            if n_add + n_alpha > 0 {
                pass.set_bind_group(1, Some(&self.post.particle_group));
                if n_add > 0 {
                    pass.set_pipeline(&self.post.particle_additive);
                    pass.draw_with_instance_count_and_first_vertex_and_first_instance(6, n_add, 0, 0);
                    stats.draw_calls += 1;
                }
                if n_alpha > 0 {
                    pass.set_pipeline(&self.post.particle_alpha);
                    pass.draw_with_instance_count_and_first_vertex_and_first_instance(6, n_alpha, 0, n_add);
                    stats.draw_calls += 1;
                }
            }
            pass.end();
        }

        // --- Post: bloom + depth darkening + tonemap -> canvas ------------------------------
        self.post.exposure_mul = self.sky.exposure_mul;
        self.post.flare = if main { self.lens_flare(view_proj, cam_pos) } else { ([0.0; 4], [0.0; 4]) };
        {
            let ts = self.timing.as_mut().filter(|_| main).and_then(|t| t.split_writes(crate::gpu_timing::POST));
            let nf = (near * (1.0 + self.settings.depth_jitter), far);
            match rtt {
                None => {
                    let surface = self.context.get_current_texture().map_err(js_err)?;
                    let surface_view = surface.create_view().map_err(js_err)?;
                    if let Some(t) = &self.targets {
                        self.post.encode(&self.queue, &encoder, t, &self.settings, nf, &surface_view, false, false, ts)?;
                    }
                }
                Some(i) => {
                    let rt = &self.rtts[i];
                    self.post.encode(&self.queue, &encoder, &rt.targets, &self.settings, nf, &rt.view0, true, rt.opts.transparent, ts)?;
                }
            }
        }

        // --- Hi-Z for next frame + counter readback --------------------------------------
        let mut readback = false;
        if gpu {
            if let Some(c) = self.cull.as_mut() {
                let ts = self.timing.as_mut().filter(|_| main).and_then(|t| t.compute_writes(crate::gpu_timing::HIZ));
                c.encode_hiz(&encoder, view_proj, ts);
                readback = c.encode_readback(&encoder);
            }
        }
        if let Some(t) = self.timing.as_ref().filter(|_| main) {
            t.encode_resolve(&encoder);
        }
        self.queue.submit(&[encoder.finish()]);
        if let Some(t) = self.timing.as_ref().filter(|_| main) {
            t.start_readback();
            stats.gpu_ms = Some(*t.last.borrow());
        }
        if readback {
            if let Some(c) = &self.cull {
                c.start_readback();
            }
        }
        if main {
            self.stats = stats;
        }
        Ok(())
    }
}

/// Frustum planes (xyz = normal, w = d), for WebGPU clip space (z in 0..1).
fn frustum_planes(m: &Mat4) -> [Vec4; 6] {
    let r0 = m.row(0);
    let r1 = m.row(1);
    let r2 = m.row(2);
    let r3 = m.row(3);
    let n = |p: Vec4| p / p.truncate().length();
    [
        n(r3 + r0),
        n(r3 - r0),
        n(r3 + r1),
        n(r3 - r1),
        n(r2),
        n(r3 - r2),
    ]
}

/// Storage buffers of the clustered lighting (grown on demand; growth rebuilds bind group 0).
pub(crate) struct LightBufs {
    pub lights: GpuBuffer,
    pub grid: GpuBuffer,
    pub indices: GpuBuffer,
    light_cap: u32,
    index_cap: u32,
}

impl LightBufs {
    fn buf(device: &GpuDevice, bytes: u32) -> Result<GpuBuffer, String> {
        device.create_buffer(&GpuBufferDescriptor::new(bytes.max(16), gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST)).map_err(js_err)
    }
    fn new(device: &GpuDevice, light_cap: u32, index_cap: u32) -> Result<Self, String> {
        Ok(Self {
            lights: Self::buf(device, light_cap * std::mem::size_of::<crate::lights::GpuLight>() as u32)?,
            grid: Self::buf(device, crate::lights::CLUSTERS * 8)?,
            indices: Self::buf(device, index_cap * 4)?,
            light_cap,
            index_cap,
        })
    }
    /// Grows the buffers; true if they were replaced (bind groups must be rebuilt).
    fn ensure(&mut self, device: &GpuDevice, lights: u32, indices: u32) -> Result<bool, String> {
        let mut changed = false;
        if lights > self.light_cap {
            self.light_cap = lights.next_power_of_two();
            self.lights = Self::buf(device, self.light_cap * std::mem::size_of::<crate::lights::GpuLight>() as u32)?;
            changed = true;
        }
        if indices > self.index_cap {
            self.index_cap = indices.next_power_of_two();
            self.indices = Self::buf(device, self.index_cap * 4)?;
            changed = true;
        }
        Ok(changed)
    }
}

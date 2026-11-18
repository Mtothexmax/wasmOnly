//! The one renderer: forward PBR over WebGPU (via `web-sys`), 4x MSAA, one shadowed directional
//! light (3x3 PCF), up to 16 point lights, frustum culling, instanced draws sorted by texture+mesh.

#![allow(deprecated)] // glam 0.33 deprecates Mat4::perspective_rh etc. in favour of glam::camera

use crate::mesh::{MeshData, Vertex};
use crate::scene::{Camera, GlobalTransform, MeshRenderer, PointLight};
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
const MAX_POINT_LIGHTS: usize = 16;

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
    /// Water surface: animated waves, Fresnel sky reflection, sun glint, colour by water depth
    /// (`shallow` -> `deep` sRGB hex), foam and soft edges at the shore. Drawn in the blend pass;
    /// use a tessellated mesh (`MeshData::grid`) so the waves can move the vertices.
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

#[derive(Clone, Debug, Serialize)]
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
    /// Half-size (world units) of the area covered by the shadow map, centred in front of the camera.
    pub shadow_extent: f32,
    /// Cull on the GPU (compute) and draw with drawIndexedIndirect (needs indirect-first-instance).
    pub gpu_culling: bool,
    /// Hi-Z occlusion culling inside the GPU culling pass (previous frame's depth).
    pub occlusion_culling: bool,
    /// 0 = lit, 1 = normals, 2 = shadow term.
    pub debug_view: u32,
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
            shadow_extent: 30.0,
            gpu_culling: true,
            occlusion_culling: true,
            debug_view: 0,
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
    /// True when this frame was culled on the GPU. Then `visible` = candidates sent to the GPU and
    /// the gpu_* counters below come from an async readback (1-3 frames old).
    pub gpu_culling: bool,
    pub gpu_frustum_culled: u32,
    pub gpu_occluded: u32,
    pub gpu_visible: u32,
    pub width: u32,
    pub height: u32,
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
struct GpuPointLight {
    pos_range: [f32; 4],
    color_intensity: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    light_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    sky_color: [f32; 4],
    ground_color: [f32; 4],
    counts: [u32; 4],
    point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
    time: [f32; 4],
}

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

pub struct GpuMesh {
    pub name: String,
    vb: GpuBuffer,
    ib: GpuBuffer,
    pub index_count: u32,
    pub vertex_count: u32,
    pub bounds_center: Vec3,
    pub bounds_radius: f32,
}

pub struct Renderer {
    pub device: GpuDevice,
    pub queue: GpuQueue,
    context: GpuCanvasContext,
    format: GpuTextureFormat,
    /// [opaque, opaque double-sided, blended, blended double-sided]
    pipelines: [GpuRenderPipeline; 4],
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
    globals_buf: GpuBuffer,
    instance_buf: GpuBuffer,
    /// Instance indices drawn by the main pass (identity on CPU path, written by cull.wgsl on GPU path).
    visible_buf: GpuBuffer,
    cull: Option<crate::gpu_cull::GpuCull>,
    pub gpu_culling_supported: bool,
    instance_cap: u32,
    bind0: GpuBindGroup,
    bind0_shadow: GpuBindGroup,
    shadow_view: GpuTextureView,
    shadow_size: u32,
    shadow_sampler: GpuSampler,
    color_sampler: GpuSampler,
    msaa_view: Option<GpuTextureView>,
    depth_view: Option<GpuTextureView>,
    size: (u32, u32),
    meshes: Vec<GpuMesh>,
    materials: Vec<Material>,
    /// Index 0 = 1x1 white default texture.
    tex_views: Vec<GpuTextureView>,
    /// 1x1 flat normal (0.5, 0.5, 1) used when a material has no normal map.
    flat_normal: Option<GpuTextureView>,
    /// Material bind groups by (base, normal, height) internal texture index (u32::MAX = default).
    mat_groups: std::collections::HashMap<(u32, u32, u32), GpuBindGroup>,
    /// Seconds since start (water animation).
    start_ms: f64,
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
        let dd = GpuDeviceDescriptor::new();
        if gpu_cull_supported {
            dd.set_required_features(&[js_sys::JsString::from(crate::gpu_cull::FEATURE)]);
        }
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
        shadow_tex_entry.set_texture(&t);
        let shadow_samp_entry = GpuBindGroupLayoutEntry::new(3, gpu_shader_stage::FRAGMENT);
        let s = GpuSamplerBindingLayout::new();
        s.set_type(GpuSamplerBindingType::Comparison);
        shadow_samp_entry.set_sampler(&s);

        let bgl0 = device
            .create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[
                uniform_entry(0, vf),
                storage_entry(1, vf),
                storage_entry(4, gpu_shader_stage::VERTEX),
                shadow_tex_entry,
                shadow_samp_entry,
            ]))
            .map_err(js_err)?;
        let bgl0_shadow = device
            .create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[
                uniform_entry(0, gpu_shader_stage::VERTEX),
                storage_entry(1, gpu_shader_stage::VERTEX),
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
        // Main pipelines: (blend, double_sided) variants of the same PBR shader.
        let make_pipeline = |blend: bool, two_sided: bool| -> Result<GpuRenderPipeline, String> {
            let vs = GpuVertexState::new(&pbr_module);
            vs.set_entry_point("vs_main");
            vs.set_buffers(&[JsNullable::wrap(vertex_layout())]);
            let target = GpuColorTargetState::new(format);
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
            desc.set_label(match (blend, two_sided) {
                (false, false) => "pbr-opaque",
                (false, true) => "pbr-opaque-2s",
                (true, false) => "pbr-blend",
                (true, true) => "pbr-blend-2s",
            });
            desc.set_fragment(&fs);
            desc.set_primitive(&prim);
            desc.set_depth_stencil(&ds);
            desc.set_multisample(&ms);
            device.create_render_pipeline(&desc).map_err(js_err)
        };
        let pipelines = [make_pipeline(false, false)?, make_pipeline(false, true)?, make_pipeline(true, false)?, make_pipeline(true, true)?];

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
            &nums(&[shadow_size, shadow_size]),
            gpu_texture_usage::RENDER_ATTACHMENT | gpu_texture_usage::TEXTURE_BINDING,
        );
        st.set_label("shadow-map");
        let shadow_tex = device.create_texture(&st).map_err(js_err)?;
        let shadow_view = shadow_tex.create_view().map_err(js_err)?;
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
        let (bind0, bind0_shadow) = Self::make_bind0(
            &device,
            &bgl0,
            &bgl0_shadow,
            &globals_buf,
            &instance_buf,
            &visible_buf,
            &shadow_view,
            &shadow_sampler,
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

        let mut r = Renderer {
            device,
            queue,
            context,
            format,
            pipelines,
            shadow_pipeline,
            shadow_mask_pipeline,
            bgl0,
            bgl0_shadow,
            bgl1,
            bgl_mat,
            bgl0_t,
            bind0_t: None,
            globals_buf,
            instance_buf,
            visible_buf,
            cull,
            gpu_culling_supported: gpu_cull_supported,
            instance_cap,
            bind0,
            bind0_shadow,
            shadow_view,
            shadow_size,
            shadow_sampler,
            color_sampler,
            msaa_view: None,
            depth_view: None,
            size: (0, 0),
            meshes: vec![],
            materials: vec![],
            tex_views: vec![],
            flat_normal: None,
            mat_groups: std::collections::HashMap::new(),
            start_ms: crate::log::now_ms(),
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
    ) -> (GpuBindGroup, GpuBindGroup) {
        let full = device.create_bind_group(&GpuBindGroupDescriptor::new(
            &[
                GpuBindGroupEntry::new_with_gpu_buffer(0, globals),
                GpuBindGroupEntry::new_with_gpu_buffer(1, instances),
                GpuBindGroupEntry::new_with_gpu_texture_view(2, shadow_view),
                GpuBindGroupEntry::new(3, shadow_sampler),
                GpuBindGroupEntry::new_with_gpu_buffer(4, visible),
            ],
            bgl0,
        ));
        let shadow = device.create_bind_group(&GpuBindGroupDescriptor::new(
            &[
                GpuBindGroupEntry::new_with_gpu_buffer(0, globals),
                GpuBindGroupEntry::new_with_gpu_buffer(1, instances),
            ],
            bgl0_shadow,
        ));
        (full, shadow)
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
        let mut distances = vec![radius.max(0.25) * 12.0];
        for (i, r) in ratios.iter().enumerate() {
            let cells = (32.0 * r).round().max(2.0) as u32;
            let simple = data.simplify_cluster(cells);
            if simple.indices.is_empty() || simple.indices.len() >= data.indices.len() {
                continue;
            }
            meshes.push(self.add_mesh(&format!("{name}#lod{}", i + 1), &simple));
            distances.push(radius.max(0.25) * 12.0 * 2f32.powi(i as i32 + 1));
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
        self.meshes.push(GpuMesh {
            name: name.to_string(),
            vb,
            ib,
            index_count: data.indices.len() as u32,
            vertex_count: data.vertices.len() as u32,
            bounds_center: c,
            bounds_radius: r,
        });
        MeshId(self.meshes.len() as u32 - 1)
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

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        let (w, h) = (w.max(1), h.max(1));
        if self.size == (w, h) && self.msaa_view.is_some() {
            return;
        }
        self.size = (w, h);
        let color = GpuTextureDescriptor::new(
            self.format,
            &nums(&[w, h]),
            gpu_texture_usage::RENDER_ATTACHMENT,
        );
        color.set_sample_count(SAMPLES);
        color.set_label("msaa-color");
        // TEXTURE_BINDING: the Hi-Z pass reads this frame's depth for next frame's occlusion culling.
        let depth = GpuTextureDescriptor::new(
            DEPTH_FORMAT,
            &nums(&[w, h]),
            gpu_texture_usage::RENDER_ATTACHMENT | gpu_texture_usage::TEXTURE_BINDING,
        );
        depth.set_sample_count(SAMPLES);
        depth.set_label("msaa-depth");
        self.msaa_view = self
            .device
            .create_texture(&color)
            .ok()
            .and_then(|t| t.create_view().ok());
        self.depth_view = self
            .device
            .create_texture(&depth)
            .ok()
            .and_then(|t| t.create_view().ok());
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
        let (a, b) = Self::make_bind0(
            &self.device,
            &self.bgl0,
            &self.bgl0_shadow,
            &self.globals_buf,
            &self.instance_buf,
            &self.visible_buf,
            &self.shadow_view,
            &self.shadow_sampler,
        );
        self.bind0 = a;
        self.bind0_shadow = b;
        self.bind0_t = None; // rebuilt lazily (needs the depth view)
        Ok(())
    }

    /// Camera (view, proj, position) of the first entity with `Camera` + `GlobalTransform`.
    pub fn camera_matrices(&self, world: &World) -> Option<(Mat4, Mat4, Vec3)> {
        let aspect = self.size.0 as f32 / self.size.1.max(1) as f32;
        world
            .query::<(&Camera, &GlobalTransform)>()
            .iter()
            .next()
            .map(|(c, g)| {
                let view = g.0.inverse();
                let proj = Mat4::perspective_rh(c.fov_y_deg.to_radians(), aspect, c.near, c.far);
                (view, proj, g.translation())
            })
    }

    pub fn render(&mut self, world: &World) -> Result<(), String> {
        let (Some(msaa), Some(depth)) = (self.msaa_view.clone(), self.depth_view.clone()) else {
            return Ok(());
        };
        let Some((view, proj, cam_pos)) = self.camera_matrices(world) else {
            return Err("no entity with Camera + Transform in the world".into());
        };
        let view_proj = proj * view;
        self.last_view_proj = view_proj;
        let cam_fwd = -view.inverse().z_axis.truncate();

        // --- Shadow camera (texel-snapped to avoid shimmering) ------------------------------
        let s = &self.settings;
        let dir = s.sun_dir.normalize_or(Vec3::NEG_Y);
        let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
        let e = s.shadow_extent;
        let center = cam_pos + cam_fwd * e * 0.5;
        let rot = Mat4::look_at_rh(Vec3::ZERO, dir, up);
        let mut c_ls = rot.transform_point3(center);
        let texel = 2.0 * e / self.shadow_size as f32;
        c_ls.x = (c_ls.x / texel).floor() * texel;
        c_ls.y = (c_ls.y / texel).floor() * texel;
        let center = rot.inverse().transform_point3(c_ls);
        let light_view = Mat4::look_at_rh(center - dir * e * 2.0, center, up);
        let light_proj = Mat4::orthographic_rh(-e, e, -e, e, 0.0, e * 4.0);
        let light_view_proj = light_proj * light_view;

        // --- Lights ---------------------------------------------------------------------------
        let mut point_lights = [GpuPointLight::zeroed(); MAX_POINT_LIGHTS];
        let mut n_lights = 0usize;
        for (pl, g) in world.query::<(&PointLight, &GlobalTransform)>().iter() {
            if n_lights == MAX_POINT_LIGHTS {
                break;
            }
            let p = g.translation();
            point_lights[n_lights] = GpuPointLight {
                pos_range: [p.x, p.y, p.z, pl.range],
                color_intensity: [pl.color.x, pl.color.y, pl.color.z, pl.intensity],
            };
            n_lights += 1;
        }
        let (near, far) = world.query::<&Camera>().iter().next().map(|c| (c.near, c.far)).unwrap_or((0.1, 500.0));
        let globals = Globals {
            view_proj: view_proj.to_cols_array_2d(),
            light_view_proj: light_view_proj.to_cols_array_2d(),
            camera_pos: cam_pos.extend(1.0).into(),
            sun_dir: dir.extend(s.sun_intensity).into(),
            sun_color: s.sun_color.extend(1.0).into(),
            sky_color: s.sky_color.extend(1.0).into(),
            ground_color: s.ground_color.extend(1.0).into(),
            counts: [n_lights as u32, s.shadows as u32, s.debug_view, 0],
            point_lights,
            time: [((crate::log::now_ms() - self.start_ms) / 1000.0) as f32, near, far, 0.0],
        };
        self.queue
            .write_buffer_with_u32_and_u8_slice(&self.globals_buf, 0, bytemuck::bytes_of(&globals))
            .map_err(js_err)?;

        // --- Instances -------------------------------------------------------------------------
        // Buffer layout: [opaque/mask candidates sorted by (sides, texture, mesh)]
        //             ++ [blended, frustum-culled on the CPU, sorted back to front]
        //             ++ [shadow casters sorted by (mask, texture, mesh)]
        // GPU path: opaque candidates are culled by cull.wgsl; CPU path: frustum test here.
        let gpu = self.settings.gpu_culling && self.cull.is_some();
        let planes = frustum_planes(&view_proj);
        let cam_fwd_n = cam_fwd.normalize_or(Vec3::NEG_Z);
        // (double_sided, texture, mesh, instance, bounds, model index)
        let mut visible: Vec<(bool, u32, u32, InstanceRaw, [f32; 4], u32)> = vec![];
        // (view depth, double_sided, texture, mesh, instance, model index)
        let mut blended: Vec<(f32, bool, u32, u32, InstanceRaw, u32)> = vec![];
        // (alpha-tested, texture, mesh, instance, model index)
        let mut casters: Vec<(bool, u32, u32, InstanceRaw, u32)> = vec![];
        // Model matrices; normal matrices (inverse-transpose) are computed afterwards with the job system.
        let mut models: Vec<Mat4> = vec![];
        let fallback_mat = Material::default();
        // Material bind groups are keyed by (base, normal, height); `tex` below is an index into key_list.
        let mut key_ids: std::collections::HashMap<(u32, u32, u32), u32> = std::collections::HashMap::new();
        let mut key_list: Vec<(u32, u32, u32)> = vec![];
        let mut culled = 0u32;
        let mut lod_counts = [0u32; 4];
        for (g, mr, lod) in world.query::<(&GlobalTransform, &MeshRenderer, Option<&Lod>)>().iter() {
            let model = g.0;
            let scale = model.x_axis.truncate().length().max(model.y_axis.truncate().length()).max(model.z_axis.truncate().length());
            let mut mesh_id = mr.mesh;
            if let Some(lod) = lod {
                match lod.pick((model.w_axis.truncate() - cam_pos).length() / scale.max(1e-4)) {
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
            let Some(mesh) = self.meshes.get(mesh_id.0 as usize) else { continue };
            let mat = self.materials.get(mr.material.0 as usize).unwrap_or(&fallback_mat);
            let shading = if mat.water { 3.0 } else if mat.ui { 2.0 } else if mat.unlit { 1.0 } else { 0.0 };
            let alpha = match mat.alpha_mode {
                AlphaMode::Opaque => 0.0,
                AlphaMode::Mask(c) => c.clamp(0.001, 1.0),
                AlphaMode::Blend => -1.0,
            };
            let raw = InstanceRaw {
                model: model.to_cols_array_2d(),
                normal: [[0.0; 4]; 4], // filled below (jobs::par_map)
                base_color: mat.base_color.into(),
                emissive_metallic: mat.emissive.extend(mat.metallic).into(),
                params: [mat.roughness, mat.checker, shading, alpha],
                maps: if mat.water {
                    [0.0, mat.wave_height, 0.0, mat.wave_scale]
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
            if mr.cast_shadows && !is_blend {
                casters.push((is_mask, if is_mask { tex } else { 0 }, mesh_id.0, raw, mi));
            }
            let center = model.transform_point3(mesh.bounds_center);
            let radius = mesh.bounds_radius * scale;
            let in_frustum = planes.iter().all(|p| p.truncate().dot(center) + p.w >= -radius);
            if is_blend {
                if in_frustum {
                    blended.push(((center - cam_pos).dot(cam_fwd_n), mat.double_sided, tex, mesh_id.0, raw, mi));
                } else {
                    culled += 1;
                }
            } else if gpu || in_frustum {
                visible.push((mat.double_sided, tex, mesh_id.0, raw, [center.x, center.y, center.z, radius], mi));
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
        blended.sort_by(|a, b| b.0.total_cmp(&a.0)); // far -> near
        casters.sort_by_key(|c| (c.0, c.1, c.2));

        // Opaque batches: runs of equal (sides, texture, mesh) -> one (indirect) draw each.
        let mut batches: Vec<(bool, u32, u32, u32, u32)> = vec![]; // (double_sided, texture, mesh, first, count)
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

        let total = (visible.len() + blended.len() + casters.len()) as u32;
        self.ensure_instance_capacity(total.max(1))?;
        let mut data: Vec<InstanceRaw> = Vec::with_capacity(total as usize);
        data.extend(visible.iter().map(|v| v.3));
        data.extend(blended.iter().map(|b| b.4));
        data.extend(casters.iter().map(|c| c.3));
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
            culled,
            visible: visible.len() as u32,
            transparent: blended.len() as u32,
            lod_levels: lod_counts,
            gpu_culling: gpu,
            ..Default::default()
        };
        let groups: Vec<GpuBindGroup> = key_list.iter().map(|k| self.material_group(*k)).collect();
        if self.bind0_t.is_none() {
            self.bind0_t = Some(self.device.create_bind_group(&GpuBindGroupDescriptor::new(
                &[
                    GpuBindGroupEntry::new_with_gpu_buffer(0, &self.globals_buf),
                    GpuBindGroupEntry::new_with_gpu_buffer(1, &self.instance_buf),
                    GpuBindGroupEntry::new_with_gpu_texture_view(2, &self.shadow_view),
                    GpuBindGroupEntry::new(3, &self.shadow_sampler),
                    GpuBindGroupEntry::new_with_gpu_buffer(4, &self.visible_buf),
                    GpuBindGroupEntry::new_with_gpu_texture_view(5, &depth),
                ],
                &self.bgl0_t,
            )));
        }
        let encoder = self.device.create_command_encoder();

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
                c.encode_cull(&self.device, &self.queue, &encoder, &self.visible_buf, &params, &items, &args)?;
                let k = *c.last_counters.borrow();
                stats.gpu_frustum_culled = k[0];
                stats.gpu_occluded = k[1];
                stats.gpu_visible = k[2];
            }
        } else if let Some(c) = self.cull.as_mut() {
            c.hiz_valid = false;
        }

        // --- Shadow pass (opaque depth-only, alpha-tested casters sample their texture) ----
        {
            let att = GpuRenderPassDepthStencilAttachment::new_with_gpu_texture_view(&self.shadow_view);
            att.set_depth_clear_value(1.0);
            att.set_depth_load_op(GpuLoadOp::Clear);
            att.set_depth_store_op(GpuStoreOp::Store);
            let desc = GpuRenderPassDescriptor::new(&[]);
            desc.set_depth_stencil_attachment(&att);
            desc.set_label("shadow");
            let pass = encoder.begin_render_pass(&desc).map_err(js_err)?;
            if self.settings.shadows && !casters.is_empty() {
                pass.set_bind_group(0, Some(&self.bind0_shadow));
                let base = blend_base + blended.len() as u32;
                let mut bound: Option<(bool, u32)> = None;
                let mut i = 0usize;
                while i < casters.len() {
                    let (mask, tex, mesh_id) = (casters[i].0, casters[i].1, casters[i].2);
                    let mut j = i;
                    while j < casters.len() && casters[j].0 == mask && casters[j].1 == tex && casters[j].2 == mesh_id {
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
                    pass.draw_indexed_with_instance_count_and_first_index_and_base_vertex_and_first_instance(
                        mesh.index_count,
                        (j - i) as u32,
                        0,
                        0,
                        base + i as u32,
                    );
                    stats.shadow_draw_calls += 1;
                    i = j;
                }
            }
            pass.end();
        }

        // --- Opaque pass (indirect on the GPU path); keeps colour + depth for the blend pass ---
        {
            let att = GpuRenderPassColorAttachment::new_with_gpu_texture_view(GpuLoadOp::Clear, GpuStoreOp::Store, &msaa);
            let cc = self.settings.clear_color;
            // clear color is linear; canvas is not sRGB -> approximate gamma
            let gm = |x: f32| x.max(0.0).powf(1.0 / 2.2) as f64;
            att.set_clear_value_gpu_color_dict(&GpuColorDict::new(1.0, gm(cc.z), gm(cc.y), gm(cc.x)));
            let datt = GpuRenderPassDepthStencilAttachment::new_with_gpu_texture_view(&depth);
            datt.set_depth_clear_value(1.0);
            datt.set_depth_load_op(GpuLoadOp::Clear);
            datt.set_depth_store_op(GpuStoreOp::Store);
            let desc = GpuRenderPassDescriptor::new(&[JsNullable::wrap(att)]);
            desc.set_depth_stencil_attachment(&datt);
            desc.set_label("opaque");
            let pass = encoder.begin_render_pass(&desc).map_err(js_err)?;
            pass.set_bind_group(0, Some(&self.bind0));
            let mut bound_tex = u32::MAX;
            let mut bound_pipe = usize::MAX;
            let args_buf = self.cull.as_ref().filter(|_| gpu).map(|c| c.args_buf.clone());
            for (b, &(two_sided, tex, mesh_id, first, count)) in batches.iter().enumerate() {
                let pi = two_sided as usize;
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
            pass.end();
        }

        // --- Blend pass: back to front, depth read-only *and* bound as a texture (water) -----
        {
            let surface = self.context.get_current_texture().map_err(js_err)?;
            let surface_view = surface.create_view().map_err(js_err)?;
            let att = GpuRenderPassColorAttachment::new_with_gpu_texture_view(GpuLoadOp::Load, GpuStoreOp::Discard, &msaa);
            att.set_resolve_target_gpu_texture_view(&surface_view);
            let datt = GpuRenderPassDepthStencilAttachment::new_with_gpu_texture_view(&depth);
            datt.set_depth_read_only(true);
            let desc = GpuRenderPassDescriptor::new(&[JsNullable::wrap(att)]);
            desc.set_depth_stencil_attachment(&datt);
            desc.set_label("blend");
            let pass = encoder.begin_render_pass(&desc).map_err(js_err)?;
            if let Some(b0) = &self.bind0_t {
                pass.set_bind_group(0, Some(b0));
            }
            let mut bound_tex = u32::MAX;
            let mut bound_pipe = usize::MAX;
            for &(two_sided, tex, mesh_id, first, count) in &blend_runs {
                let pi = 2 + two_sided as usize;
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
            pass.end();
        }

        // --- Hi-Z for next frame + counter readback --------------------------------------
        let mut readback = false;
        if gpu {
            if let Some(c) = self.cull.as_mut() {
                c.encode_hiz(&encoder, view_proj);
                readback = c.encode_readback(&encoder);
            }
        }
        self.queue.submit(&[encoder.finish()]);
        if readback {
            if let Some(c) = &self.cull {
                c.start_readback();
            }
        }
        self.stats = stats;
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

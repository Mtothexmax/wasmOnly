//! HDR scene targets and post-processing: bloom (dual-filter mip chain), depth darkening,
//! exposure + ACES tonemapping. Also the sky pipeline (drawn in the opaque pass) and the particle
//! pipelines (drawn in the blend pass).
//!
//! Frame: opaque pass (+ sky) and blend pass (+ particles) render into a 4x MSAA `rgba16float`
//! target, which resolves into `hdr`. Then: prefilter -> bloom mip 0, down to mip n-1, up again
//! (additive), composite (HDR + bloom, darkening, tonemap) -> output (canvas or a render texture).

use bytemuck::{Pod, Zeroable};
use js_sys::JsNullable;
use wasm_bindgen::JsCast;
use web_sys::*;

pub(crate) const HDR_FORMAT: GpuTextureFormat = GpuTextureFormat::Rgba16float;
/// 5 mips from quarter resolution: the glow reaches far enough; every mip is two render passes.
const MAX_BLOOM_MIPS: u32 = 5;
/// Uniform slot stride (minUniformBufferOffsetAlignment).
const SLOT: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct PostParams {
    pub a: [f32; 4],
    pub b: [f32; 4],
    pub c: [f32; 4],
    pub d: [f32; 4],
    pub e: [f32; 4],
    /// Colour grading, vignette shape, grain (composite only; see `fs_composite`).
    pub g: [[f32; 4]; 8],
}

/// Colour grading in display space, after tonemapping (`RenderSettings::grade`). Default = neutral
/// (costs nothing). Order: white balance -> lift/gamma/gain -> contrast -> split toning ->
/// saturation. [`Grade::preset`] gives looks ("warm", "cool", "fade", "vivid", "noir", "night",
/// "golden"); API `render {grade: {preset: "warm", temperature: 0.2, ..}}`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Grade {
    /// White balance: -1 blue .. 1 yellow.
    pub temperature: f32,
    /// -1 green .. 1 magenta.
    pub tint: f32,
    /// Added to the shadows (0 = neutral): `c = gain * (c + lift * (1 - c))`.
    pub lift: [f32; 3],
    /// Midtone power (1 = neutral; > 1 brightens).
    pub gamma: [f32; 3],
    /// Highlights multiplier (1 = neutral).
    pub gain: [f32; 3],
    /// Contrast around `pivot` (1 = neutral).
    pub contrast: f32,
    pub pivot: f32,
    /// Split toning: colours soft-light blended into the shadows / highlights (display-space
    /// RGB 0..1; 0.5 grey = neutral; the API also takes "#rrggbb").
    pub split_shadows: [f32; 3],
    pub split_highlights: [f32; 3],
    /// -1 (more shadows) .. 1 (more highlights).
    pub split_balance: f32,
    /// Multiplies `RenderSettings::saturation`.
    pub saturation: f32,
}

impl Default for Grade {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            tint: 0.0,
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
            contrast: 1.0,
            pivot: 0.5,
            split_shadows: [0.5; 3],
            split_highlights: [0.5; 3],
            split_balance: 0.0,
            saturation: 1.0,
        }
    }
}

impl Grade {
    pub const PRESETS: [&'static str; 8] = ["neutral", "warm", "cool", "fade", "vivid", "noir", "night", "golden"];

    /// Blend of two grades (`k` 0 = `a`, 1 = `b`), e.g. day -> night per frame by `sky.night`.
    pub fn lerp(a: &Grade, b: &Grade, k: f32) -> Grade {
        let k = k.clamp(0.0, 1.0);
        let f = |x: f32, y: f32| x + (y - x) * k;
        let v = |x: [f32; 3], y: [f32; 3]| [f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2])];
        Grade {
            temperature: f(a.temperature, b.temperature),
            tint: f(a.tint, b.tint),
            lift: v(a.lift, b.lift),
            gamma: v(a.gamma, b.gamma),
            gain: v(a.gain, b.gain),
            contrast: f(a.contrast, b.contrast),
            pivot: f(a.pivot, b.pivot),
            split_shadows: v(a.split_shadows, b.split_shadows),
            split_highlights: v(a.split_highlights, b.split_highlights),
            split_balance: f(a.split_balance, b.split_balance),
            saturation: f(a.saturation, b.saturation),
        }
    }

    /// A named look (see [`Grade::PRESETS`]).
    pub fn preset(name: &str) -> Option<Self> {
        let n = Self::default();
        Some(match name {
            "neutral" => n,
            "warm" => Self { temperature: 0.35, tint: 0.05, gain: [1.03, 1.0, 0.96], contrast: 1.05, ..n },
            "cool" => Self { temperature: -0.35, lift: [0.0, 0.01, 0.03], contrast: 1.03, ..n },
            "fade" => Self { lift: [0.08, 0.08, 0.09], gain: [0.96, 0.96, 0.95], contrast: 0.85, saturation: 0.8, ..n },
            "vivid" => Self { contrast: 1.15, saturation: 1.3, split_shadows: [0.45, 0.5, 0.58], split_highlights: [0.58, 0.54, 0.46], ..n },
            "noir" => Self { contrast: 1.3, lift: [0.02; 3], saturation: 0.0, ..n },
            "night" => Self { temperature: -0.5, lift: [0.0, 0.01, 0.04], gain: [0.92, 0.95, 1.05], saturation: 0.85, ..n },
            "golden" => Self { temperature: 0.5, split_highlights: [0.65, 0.55, 0.4], split_shadows: [0.48, 0.47, 0.52], contrast: 1.05, ..n },
            _ => return None,
        })
    }
}

/// Size-dependent resources of one render target (the canvas, or one render texture).
pub(crate) struct PostTargets {
    pub size: (u32, u32),
    pub msaa: GpuTextureView,
    pub depth: GpuTextureView,
    pub hdr_view: GpuTextureView,
    /// Textures behind `hdr_view` / `scene_color` (copies without MSAA).
    pub hdr_tex: GpuTexture,
    pub scene_tex: GpuTexture,
    /// The resolved opaque scene for water refraction/reflection (resolve target of the opaque pass
    /// while water is visible; the blend pass samples it and resolves the final image into `hdr_view`).
    pub scene_color: GpuTextureView,
    bloom_views: Vec<GpuTextureView>,
    bloom_sizes: Vec<(u32, u32)>,
    params: GpuBuffer,
    pre_group: GpuBindGroup,
    /// down_groups[i] samples mip i (writes mip i+1); up_groups[i] samples mip i (adds onto mip i-1).
    down_groups: Vec<GpuBindGroup>,
    up_groups: Vec<GpuBindGroup>,
    comp_group: GpuBindGroup,
}

pub(crate) struct Post {
    pub sky_pipeline: GpuRenderPipeline,
    /// Exposure multiplier from the day/night cycle (set by the renderer each frame).
    pub exposure_mul: f32,
    /// Frame counter for the animated film grain.
    pub grain_frame: u32,
    /// Lens flare for the next composite: (sun uv x, y, strength, disk radius in uv height) and
    /// (sun colour rgb, expected disk luminance). Set by the renderer per view; strength 0 = off.
    pub flare: ([f32; 4], [f32; 4]),
    /// Target pixels per device pixel for the grain cell size (`render_scale` on the canvas, 1 for render textures).
    pub grain_px: f32,
    pub particle_additive: GpuRenderPipeline,
    pub particle_alpha: GpuRenderPipeline,
    pub bgl_particles: GpuBindGroupLayout,
    pub particle_buf: GpuBuffer,
    pub particle_cap: u32,
    pub particle_group: GpuBindGroup,
    prefilter: GpuRenderPipeline,
    down: GpuRenderPipeline,
    up: GpuRenderPipeline,
    /// [gamma-encoded output (canvas), linear output into an sRGB texture (render-to-texture)]
    composite_canvas: GpuRenderPipeline,
    composite_srgb: GpuRenderPipeline,
    bgl_post: GpuBindGroupLayout,
    bgl_comp: GpuBindGroupLayout,
    sampler: GpuSampler,
}

fn js_err(e: wasm_bindgen::JsValue) -> String {
    e.dyn_ref::<js_sys::Error>().map(|e| String::from(e.message())).unwrap_or_else(|| format!("{e:?}"))
}

fn nums(v: &[u32]) -> Vec<js_sys::Number> {
    v.iter().map(|x| js_sys::Number::from(*x)).collect()
}

fn fullscreen(device: &GpuDevice, module: &GpuShaderModule, layout: &GpuPipelineLayout, vs_entry: &str, fs_entry: &str, target: GpuColorTargetState, constants: Option<&js_sys::Object>, label: &str) -> Result<GpuRenderPipeline, String> {
    let vs = GpuVertexState::new(module);
    vs.set_entry_point(vs_entry);
    let fs = GpuFragmentState::new(module, &[JsNullable::wrap(target)]);
    fs.set_entry_point(fs_entry);
    if let Some(c) = constants {
        fs.set_constants(c.unchecked_ref());
    }
    let d = GpuRenderPipelineDescriptor::new(layout, &vs);
    d.set_label(label);
    d.set_fragment(&fs);
    device.create_render_pipeline(&d).map_err(js_err)
}

fn additive_blend() -> GpuBlendState {
    let c = GpuBlendComponent::new();
    c.set_src_factor(GpuBlendFactor::One);
    c.set_dst_factor(GpuBlendFactor::One);
    c.set_operation(GpuBlendOperation::Add);
    let a = GpuBlendComponent::new();
    a.set_src_factor(GpuBlendFactor::Zero);
    a.set_dst_factor(GpuBlendFactor::One);
    a.set_operation(GpuBlendOperation::Add);
    GpuBlendState::new(&a, &c)
}

fn alpha_blend() -> GpuBlendState {
    let c = GpuBlendComponent::new();
    c.set_src_factor(GpuBlendFactor::SrcAlpha);
    c.set_dst_factor(GpuBlendFactor::OneMinusSrcAlpha);
    c.set_operation(GpuBlendOperation::Add);
    let a = GpuBlendComponent::new();
    a.set_src_factor(GpuBlendFactor::One);
    a.set_dst_factor(GpuBlendFactor::OneMinusSrcAlpha);
    a.set_operation(GpuBlendOperation::Add);
    GpuBlendState::new(&a, &c)
}

impl Post {
    pub fn new(device: &GpuDevice, canvas_format: GpuTextureFormat, common: &str, bgl0: &GpuBindGroupLayout, bgl0_t: &GpuBindGroupLayout, depth_format: GpuTextureFormat, samples: u32) -> Result<Self, String> {
        let module = device.create_shader_module(&GpuShaderModuleDescriptor::new(&crate::renderer::ms_src(include_str!("shaders/post.wgsl"), samples)));
        module.set_label("post.wgsl");
        crate::debug::report_shader_errors(&module, "post.wgsl");

        let frag = gpu_shader_stage::FRAGMENT;
        // the composite vertex shader also reads the scene (lens-flare sun visibility, once per frame)
        let both = gpu_shader_stage::FRAGMENT | gpu_shader_stage::VERTEX;
        let tex_v = |b: u32, vis: u32| {
            let e = GpuBindGroupLayoutEntry::new(b, vis);
            let t = GpuTextureBindingLayout::new();
            t.set_sample_type(GpuTextureSampleType::Float);
            e.set_texture(&t);
            e
        };
        let tex = |b: u32| tex_v(b, frag);
        let samp_v = |vis: u32| {
            let e = GpuBindGroupLayoutEntry::new(1, vis);
            let s = GpuSamplerBindingLayout::new();
            s.set_type(GpuSamplerBindingType::Filtering);
            e.set_sampler(&s);
            e
        };
        let samp = || samp_v(frag);
        let uni_v = |vis: u32| {
            let e = GpuBindGroupLayoutEntry::new(2, vis);
            let b = GpuBufferBindingLayout::new();
            b.set_type(GpuBufferBindingType::Uniform);
            e.set_buffer(&b);
            e
        };
        let uni = || uni_v(frag);
        let bgl_post = device.create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[tex(0), samp(), uni()])).map_err(js_err)?;
        let depth_e = GpuBindGroupLayoutEntry::new(4, frag);
        let dt = GpuTextureBindingLayout::new();
        dt.set_sample_type(GpuTextureSampleType::Depth);
        dt.set_multisampled(samples > 1);
        depth_e.set_texture(&dt);
        let bgl_comp = device.create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[tex_v(0, both), samp_v(both), uni_v(both), tex(3), depth_e])).map_err(js_err)?;
        let layout_post = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[JsNullable::wrap(bgl_post.clone())]));
        let layout_comp = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[JsNullable::wrap(bgl_comp.clone())]));

        let prefilter = fullscreen(device, &module, &layout_post, "vs_full", "fs_prefilter", GpuColorTargetState::new(HDR_FORMAT), None, "bloom-prefilter")?;
        let down = fullscreen(device, &module, &layout_post, "vs_full", "fs_down", GpuColorTargetState::new(HDR_FORMAT), None, "bloom-down")?;
        let up_t = GpuColorTargetState::new(HDR_FORMAT);
        up_t.set_blend(&additive_blend());
        let up = fullscreen(device, &module, &layout_post, "vs_full", "fs_up", up_t, None, "bloom-up")?;
        let gamma = |on: bool| {
            let o = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&o, &"GAMMA_OUT".into(), &(if on { 1.0 } else { 0.0 }).into());
            // no MSAA: anti-aliasing in the composite
            let _ = js_sys::Reflect::set(&o, &"FXAA".into(), &(if samples == 1 { 1.0 } else { 0.0 }).into());
            o
        };
        let composite_canvas = fullscreen(device, &module, &layout_comp, "vs_composite", "fs_composite", GpuColorTargetState::new(canvas_format), Some(&gamma(true)), "composite")?;
        let composite_srgb = fullscreen(device, &module, &layout_comp, "vs_composite", "fs_composite", GpuColorTargetState::new(GpuTextureFormat::Rgba8unormSrgb), Some(&gamma(false)), "composite-rtt")?;

        // --- sky (opaque pass, far plane, depth less-equal) ---
        let sky_src = crate::renderer::ms_src(&format!("{common}\n{}", include_str!("shaders/sky.wgsl")), samples);
        let sky_module = device.create_shader_module(&GpuShaderModuleDescriptor::new(&sky_src));
        sky_module.set_label("sky.wgsl");
        crate::debug::report_shader_errors(&sky_module, "sky.wgsl");
        let layout_sky = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[JsNullable::wrap(bgl0.clone())]));
        let ms = GpuMultisampleState::new();
        ms.set_count(samples);
        let vs = GpuVertexState::new(&sky_module);
        vs.set_entry_point("vs_sky");
        let fs = GpuFragmentState::new(&sky_module, &[JsNullable::wrap(GpuColorTargetState::new(HDR_FORMAT))]);
        fs.set_entry_point("fs_sky");
        let ds = GpuDepthStencilState::new(depth_format);
        ds.set_depth_write_enabled(false);
        ds.set_depth_compare(GpuCompareFunction::LessEqual);
        let d = GpuRenderPipelineDescriptor::new(&layout_sky, &vs);
        d.set_label("sky");
        d.set_fragment(&fs);
        d.set_depth_stencil(&ds);
        d.set_multisample(&ms);
        let sky_pipeline = device.create_render_pipeline(&d).map_err(js_err)?;

        // --- particles (blend pass) ---
        let p_src = crate::renderer::ms_src(&format!("{common}\n{}", include_str!("shaders/particles.wgsl")), samples);
        let p_module = device.create_shader_module(&GpuShaderModuleDescriptor::new(&p_src));
        p_module.set_label("particles.wgsl");
        crate::debug::report_shader_errors(&p_module, "particles.wgsl");
        let pe = GpuBindGroupLayoutEntry::new(0, gpu_shader_stage::VERTEX);
        let pb = GpuBufferBindingLayout::new();
        pb.set_type(GpuBufferBindingType::ReadOnlyStorage);
        pe.set_buffer(&pb);
        let bgl_particles = device.create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[pe])).map_err(js_err)?;
        let layout_p = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[
            JsNullable::wrap(bgl0_t.clone()),
            JsNullable::wrap(bgl_particles.clone()),
        ]));
        let particle = |fs_entry: &str, blend: GpuBlendState, label: &str| -> Result<GpuRenderPipeline, String> {
            let vs = GpuVertexState::new(&p_module);
            vs.set_entry_point("vs_particle");
            let t = GpuColorTargetState::new(HDR_FORMAT);
            t.set_blend(&blend);
            let fs = GpuFragmentState::new(&p_module, &[JsNullable::wrap(t)]);
            fs.set_entry_point(fs_entry);
            let ds = GpuDepthStencilState::new(depth_format);
            ds.set_depth_write_enabled(false);
            ds.set_depth_compare(GpuCompareFunction::Less);
            let prim = GpuPrimitiveState::new();
            prim.set_cull_mode(GpuCullMode::None);
            let d = GpuRenderPipelineDescriptor::new(&layout_p, &vs);
            d.set_label(label);
            d.set_fragment(&fs);
            d.set_depth_stencil(&ds);
            d.set_multisample(&ms);
            d.set_primitive(&prim);
            device.create_render_pipeline(&d).map_err(js_err)
        };
        let particle_additive = particle("fs_additive", additive_blend(), "particles-add")?;
        let particle_alpha = particle("fs_alpha", alpha_blend(), "particles-alpha")?;
        let particle_cap = 1024;
        let (particle_buf, particle_group) = Self::make_particle_buf(device, &bgl_particles, particle_cap)?;

        let sd = GpuSamplerDescriptor::new();
        sd.set_mag_filter(GpuFilterMode::Linear);
        sd.set_min_filter(GpuFilterMode::Linear);
        let sampler = device.create_sampler_with_descriptor(&sd);
        Ok(Self {
            sky_pipeline,
            exposure_mul: 1.0,
            grain_frame: 0,
            flare: ([0.0; 4], [0.0; 4]),
            grain_px: 1.0,
            particle_additive,
            particle_alpha,
            bgl_particles,
            particle_buf,
            particle_cap,
            particle_group,
            prefilter,
            down,
            up,
            composite_canvas,
            composite_srgb,
            bgl_post,
            bgl_comp,
            sampler,
        })
    }

    fn make_particle_buf(device: &GpuDevice, bgl: &GpuBindGroupLayout, cap: u32) -> Result<(GpuBuffer, GpuBindGroup), String> {
        let buf = device
            .create_buffer(&GpuBufferDescriptor::new(cap * std::mem::size_of::<crate::particles::ParticleRaw>() as u32, gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST))
            .map_err(js_err)?;
        let group = device.create_bind_group(&GpuBindGroupDescriptor::new(&[GpuBindGroupEntry::new_with_gpu_buffer(0, &buf)], bgl));
        Ok((buf, group))
    }

    pub fn ensure_particles(&mut self, device: &GpuDevice, n: u32) -> Result<(), String> {
        if n <= self.particle_cap {
            return Ok(());
        }
        let cap = n.next_power_of_two();
        let (b, g) = Self::make_particle_buf(device, &self.bgl_particles, cap)?;
        self.particle_buf.destroy();
        self.particle_buf = b;
        self.particle_group = g;
        self.particle_cap = cap;
        Ok(())
    }

    /// Creates the targets for a `w` x `h` output.
    pub fn targets(&self, device: &GpuDevice, w: u32, h: u32, depth_format: GpuTextureFormat, samples: u32, label: &str) -> Result<PostTargets, String> {
        let tex = |fmt: GpuTextureFormat, size: (u32, u32), usage: u32, samples: u32, mips: u32, l: &str| -> Result<GpuTexture, String> {
            let d = GpuTextureDescriptor::new(fmt, &nums(&[size.0.max(1), size.1.max(1)]), usage);
            d.set_sample_count(samples);
            d.set_mip_level_count(mips);
            d.set_label(&format!("{label}:{l}"));
            device.create_texture(&d).map_err(js_err)
        };
        use gpu_texture_usage::{RENDER_ATTACHMENT as RA, TEXTURE_BINDING as TB};
        use gpu_texture_usage::{COPY_DST as CD, COPY_SRC as CS};
        // TEXTURE_BINDING: Hi-Z, water and particles read this frame's depth
        let depth = tex(depth_format, (w, h), RA | TB, samples, 1, "msaa-depth")?.create_view().map_err(js_err)?;
        let hdr_tex = tex(HDR_FORMAT, (w, h), RA | TB | CD, 1, 1, "hdr")?;
        let hdr_view = hdr_tex.create_view().map_err(js_err)?;
        // without MSAA the scene is drawn straight into the HDR target
        let msaa = if samples > 1 { tex(HDR_FORMAT, (w, h), RA, samples, 1, "msaa-hdr")?.create_view().map_err(js_err)? } else { hdr_view.clone() };
        // resolve target of the opaque pass when water is visible (refraction + SSR read it); without
        // MSAA the opaque pass draws into it and it is copied into the HDR target
        let scene_tex = tex(HDR_FORMAT, (w, h), RA | TB | CS, 1, 1, "scene-color")?;
        let scene_color = scene_tex.create_view().map_err(js_err)?;
        // bloom starts at quarter resolution (the glow is soft; half res costs ~4x for no visible gain)
        let (bw, bh) = ((w / 4).max(1), (h / 4).max(1));
        let mips = (32 - bw.min(bh).max(1).leading_zeros()).saturating_sub(2).clamp(1, MAX_BLOOM_MIPS);
        let bloom = tex(HDR_FORMAT, (bw, bh), RA | TB, 1, mips, "bloom")?;
        let mut bloom_views = vec![];
        let mut bloom_sizes = vec![];
        for m in 0..mips {
            let d = GpuTextureViewDescriptor::new();
            d.set_base_mip_level(m);
            d.set_mip_level_count(1);
            bloom_views.push(bloom.create_view_with_descriptor(&d).map_err(js_err)?);
            bloom_sizes.push(((bw >> m).max(1), (bh >> m).max(1)));
        }
        // slots: 0 prefilter, 1..=mips down (index i+1), then ups, last = composite
        let slots = 2 + 2 * mips;
        let params = device
            .create_buffer(&GpuBufferDescriptor::new(slots * SLOT, gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST))
            .map_err(js_err)?;
        let slot = |i: u32| {
            let b = GpuBufferBinding::new(&params);
            b.set_offset(i * SLOT);
            b.set_size(std::mem::size_of::<PostParams>() as u32);
            GpuBindGroupEntry::new_with_gpu_buffer_binding(2, &b)
        };
        let group = |src: &GpuTextureView, s: u32| {
            device.create_bind_group(&GpuBindGroupDescriptor::new(
                &[GpuBindGroupEntry::new_with_gpu_texture_view(0, src), GpuBindGroupEntry::new(1, &self.sampler), slot(s)],
                &self.bgl_post,
            ))
        };
        let pre_group = group(&hdr_view, 0);
        let down_groups = (0..mips).map(|i| group(&bloom_views[i as usize], 1 + i)).collect();
        let up_groups = (0..mips).map(|i| group(&bloom_views[i as usize], 1 + mips + i)).collect();
        let comp_group = device.create_bind_group(&GpuBindGroupDescriptor::new(
            &[
                GpuBindGroupEntry::new_with_gpu_texture_view(0, &hdr_view),
                GpuBindGroupEntry::new(1, &self.sampler),
                slot(slots - 1),
                GpuBindGroupEntry::new_with_gpu_texture_view(3, &bloom_views[0]),
                GpuBindGroupEntry::new_with_gpu_texture_view(4, &depth),
            ],
            &self.bgl_comp,
        ));
        Ok(PostTargets { size: (w, h), msaa, depth, hdr_view, hdr_tex, scene_tex, scene_color, bloom_views, bloom_sizes, params, pre_group, down_groups, up_groups, comp_group })
    }

    /// Bloom + composite into `output` (canvas texture or an sRGB render texture).
    pub fn encode(&self, queue: &GpuQueue, enc: &GpuCommandEncoder, t: &PostTargets, s: &crate::renderer::RenderSettings, near_far: (f32, f32), output: &GpuTextureView, srgb_output: bool, transparent: bool, ts: Option<(GpuRenderPassTimestampWrites, GpuRenderPassTimestampWrites)>) -> Result<(), String> {
        let mips = t.bloom_views.len() as u32;
        let slots = 2 + 2 * mips;
        let mut data = vec![0u8; (slots * SLOT) as usize];
        let mut put = |i: u32, p: PostParams| {
            let o = (i * SLOT) as usize;
            data[o..o + std::mem::size_of::<PostParams>()].copy_from_slice(bytemuck::bytes_of(&p));
        };
        let texel = |sz: (u32, u32)| [1.0 / sz.0 as f32, 1.0 / sz.1 as f32];
        let bloom = s.bloom && s.debug_view == 0;
        put(0, PostParams { a: [1.0 / t.size.0 as f32, 1.0 / t.size.1 as f32, s.bloom_threshold, s.bloom_knee], ..Default::default() });
        for i in 0..mips {
            let [tx, ty] = texel(t.bloom_sizes[i as usize]);
            put(1 + i, PostParams { a: [tx, ty, 0.0, 0.0], ..Default::default() });
            // larger radius = more weight on the coarse levels
            put(1 + mips + i, PostParams { a: [tx, ty, s.bloom_radius, 0.0], ..Default::default() });
        }
        let dark = if s.debug_view == 0 { s.depth_darkening } else { 0.0 };
        put(
            slots - 1,
            PostParams {
                a: [s.exposure * self.exposure_mul, s.bloom_intensity, dark, s.depth_darkening_radius * t.size.1 as f32 / 1000.0],
                b: [near_far.0, near_far.1, if bloom { 1.0 } else { 0.0 }, s.depth_darkening_range],
                c: [s.vignette, s.saturation, if transparent { 1.0 } else { 0.0 }, t.size.0 as f32 / t.size.1.max(1) as f32],
                d: if s.debug_view == 0 { self.flare.0 } else { [0.0; 4] },
                e: self.flare.1,
                g: {
                    let gr = &s.grade;
                    let on = gr != &Grade::default() && s.debug_view == 0;
                    let [lr, lg, lb] = gr.lift;
                    let [gmr, gmg, gmb] = gr.gamma;
                    let [gnr, gng, gnb] = gr.gain;
                    let [sr, sg, sb] = gr.split_shadows;
                    let [hr, hg, hb] = gr.split_highlights;
                    let [vr, vg, vb] = s.vignette_color;
                    [
                        [gr.temperature, gr.tint, gr.contrast, gr.pivot],
                        [lr, lg, lb, gr.split_balance],
                        [gmr.max(0.01), gmg.max(0.01), gmb.max(0.01), if s.debug_view == 0 { s.grain.max(0.0) } else { 0.0 }],
                        [gnr, gng, gnb, s.grain_size.max(0.5) * self.grain_px],
                        [sr, sg, sb, s.vignette_radius.max(0.0)],
                        [hr, hg, hb, s.vignette_softness.max(0.001)],
                        [vr, vg, vb, s.vignette_roundness.clamp(0.0, 1.0)],
                        [(self.grain_frame % 4096) as f32, if on { 1.0 } else { 0.0 }, gr.saturation.max(0.0), 0.0],
                    ]
                },
            },
        );
        queue.write_buffer_with_u32_and_u8_slice(&t.params, 0, &data).map_err(js_err)?;

        let (ts_begin, ts_end) = match ts {
            Some((b, e)) => (Some(b), Some(e)),
            None => (None, None),
        };
        let first = std::cell::Cell::new(true);
        let run = |pipeline: &GpuRenderPipeline, group: &GpuBindGroup, target: &GpuTextureView, load: bool, label: &str| -> Result<(), String> {
            let att = GpuRenderPassColorAttachment::new_with_gpu_texture_view(if load { GpuLoadOp::Load } else { GpuLoadOp::Clear }, GpuStoreOp::Store, target);
            att.set_clear_value_gpu_color_dict(&GpuColorDict::new(1.0, 0.0, 0.0, 0.0));
            let d = GpuRenderPassDescriptor::new(&[JsNullable::wrap(att)]);
            d.set_label(label);
            let last = label == "composite";
            match (first.replace(false), last, &ts_begin, &ts_end) {
                (true, true, Some(b), Some(e)) => {
                    // bloom off: composite is the only pass -> begin + end on it
                    if let Some(i) = js_sys::Reflect::get(e, &"endOfPassWriteIndex".into()).ok().and_then(|v| v.as_f64()) {
                        b.set_end_of_pass_write_index(i as u32);
                    }
                    d.set_timestamp_writes(b);
                }
                (true, _, Some(b), _) => d.set_timestamp_writes(b),
                (_, true, _, Some(e)) => d.set_timestamp_writes(e),
                _ => {}
            }
            let pass = enc.begin_render_pass(&d).map_err(js_err)?;
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(group));
            pass.draw(3);
            pass.end();
            Ok(())
        };
        if bloom {
            run(&self.prefilter, &t.pre_group, &t.bloom_views[0], false, "bloom-prefilter")?;
            for i in 0..mips.saturating_sub(1) {
                run(&self.down, &t.down_groups[i as usize], &t.bloom_views[i as usize + 1], false, "bloom-down")?;
            }
            for i in (1..mips).rev() {
                run(&self.up, &t.up_groups[i as usize], &t.bloom_views[i as usize - 1], true, "bloom-up")?;
            }
        }
        run(if srgb_output { &self.composite_srgb } else { &self.composite_canvas }, &t.comp_group, output, false, "composite")
    }
}

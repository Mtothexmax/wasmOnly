//! Hi-Z occlusion pyramid for the GPU-driven scene (`gpu_scene.rs`): `hiz.wgsl` builds a
//! conservative max-depth mip chain from this frame's depth buffer; the next frame's culling pass
//! (`cull.wgsl`) tests bounding spheres against it.
//!
//! Needs the `indirect-first-instance` device feature (indirect draws with a first instance); without
//! it the renderer uses CPU frustum culling with direct draws.

use js_sys::{JsNullable, Number};
use web_sys::*;

pub(crate) const FEATURE: &str = "indirect-first-instance";
const HIZ_FORMAT: GpuTextureFormat = GpuTextureFormat::R32float;

fn js_err(e: wasm_bindgen::JsValue) -> String {
    format!("{e:?}")
}

fn nums(v: &[u32]) -> Vec<Number> {
    v.iter().map(|&x| Number::from(x)).collect()
}

pub(crate) fn compute_pass(enc: &GpuCommandEncoder, ts: Option<GpuComputePassTimestampWrites>) -> GpuComputePassEncoder {
    match ts {
        Some(w) => {
            let d = GpuComputePassDescriptor::new();
            d.set_timestamp_writes(&w);
            enc.begin_compute_pass_with_descriptor(&d)
        }
        None => enc.begin_compute_pass(),
    }
}

pub(crate) struct GpuCull {
    init_pipeline: GpuComputePipeline,
    down_pipeline: GpuComputePipeline,
    init_bgl: GpuBindGroupLayout,
    down_bgl: GpuBindGroupLayout,
    hiz_view_all: Option<GpuTextureView>,
    hiz_init_bg: Option<GpuBindGroup>,
    hiz_down: Vec<(GpuBindGroup, u32, u32)>,
    pub hiz_size: (u32, u32),
    pub hiz_mips: u32,
    /// The pyramid holds a depth buffer that matches `prev_view_proj`.
    pub hiz_valid: bool,
    pub prev_view_proj: glam::Mat4,
    /// Bumped whenever the pyramid texture is recreated (bind groups that sample it go stale).
    pub generation: u32,
}

impl GpuCull {
    pub fn new(device: &GpuDevice) -> Result<Self, String> {
        let cs = gpu_shader_stage::COMPUTE;
        let tex_entry = |binding: u32, sample: GpuTextureSampleType, ms: bool| {
            let e = GpuBindGroupLayoutEntry::new(binding, cs);
            let t = GpuTextureBindingLayout::new();
            t.set_sample_type(sample);
            t.set_multisampled(ms);
            e.set_texture(&t);
            e
        };
        let storage_tex_entry = |binding: u32| {
            let e = GpuBindGroupLayoutEntry::new(binding, cs);
            let s = GpuStorageTextureBindingLayout::new(HIZ_FORMAT);
            s.set_access(GpuStorageTextureAccess::WriteOnly);
            e.set_storage_texture(&s);
            e
        };
        let bgl = |entries: &[GpuBindGroupLayoutEntry]| {
            device
                .create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(entries))
                .map_err(js_err)
        };
        let init_bgl = bgl(&[
            tex_entry(0, GpuTextureSampleType::Depth, true),
            storage_tex_entry(1),
        ])?;
        let down_bgl = bgl(&[
            tex_entry(2, GpuTextureSampleType::UnfilterableFloat, false),
            storage_tex_entry(3),
        ])?;

        let module = |src: &str, label: &str| {
            let m = device.create_shader_module(&GpuShaderModuleDescriptor::new(src));
            m.set_label(label);
            crate::debug::report_shader_errors(&m, label);
            m
        };
        let hiz_mod = module(include_str!("shaders/hiz.wgsl"), "hiz.wgsl");
        let pipeline = |m: &GpuShaderModule, entry: &str, layout: &GpuBindGroupLayout| {
            let pl = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[
                JsNullable::wrap(layout.clone()),
            ]));
            let stage = GpuProgrammableStage::new(m);
            stage.set_entry_point(entry);
            let d = GpuComputePipelineDescriptor::new(&pl, &stage);
            d.set_label(entry);
            device.create_compute_pipeline(&d)
        };
        let init_pipeline = pipeline(&hiz_mod, "init", &init_bgl);
        let down_pipeline = pipeline(&hiz_mod, "down", &down_bgl);

        Ok(Self {
            init_pipeline,
            down_pipeline,
            init_bgl,
            down_bgl,
            hiz_view_all: None,
            hiz_init_bg: None,
            hiz_down: vec![],
            hiz_size: (1, 1),
            hiz_mips: 1,
            hiz_valid: false,
            prev_view_proj: glam::Mat4::IDENTITY,
            generation: 0,
        })
    }

    /// The whole pyramid (all mips), for the culling pass.
    pub fn hiz_view(&self) -> Option<&GpuTextureView> {
        self.hiz_view_all.as_ref()
    }

    /// (Re)creates the Hi-Z pyramid for a new render size; `depth` is the MSAA depth view.
    pub fn resize(
        &mut self,
        device: &GpuDevice,
        w: u32,
        h: u32,
        depth: &GpuTextureView,
    ) -> Result<(), String> {
        let (hw, hh) = (w.div_ceil(2).max(1), h.div_ceil(2).max(1));
        let mips = 32 - hw.max(hh).leading_zeros();
        let d = GpuTextureDescriptor::new(
            HIZ_FORMAT,
            &nums(&[hw, hh]),
            gpu_texture_usage::STORAGE_BINDING | gpu_texture_usage::TEXTURE_BINDING,
        );
        d.set_mip_level_count(mips);
        d.set_label("hi-z");
        let tex = device.create_texture(&d).map_err(js_err)?;
        let level = |l: u32| {
            let vd = GpuTextureViewDescriptor::new();
            vd.set_base_mip_level(l);
            vd.set_mip_level_count(1);
            tex.create_view_with_descriptor(&vd).map_err(js_err)
        };
        self.hiz_init_bg = Some(device.create_bind_group(&GpuBindGroupDescriptor::new(
            &[
                GpuBindGroupEntry::new_with_gpu_texture_view(0, depth),
                GpuBindGroupEntry::new_with_gpu_texture_view(1, &level(0)?),
            ],
            &self.init_bgl,
        )));
        self.hiz_down.clear();
        for l in 1..mips {
            let bg = device.create_bind_group(&GpuBindGroupDescriptor::new(
                &[
                    GpuBindGroupEntry::new_with_gpu_texture_view(2, &level(l - 1)?),
                    GpuBindGroupEntry::new_with_gpu_texture_view(3, &level(l)?),
                ],
                &self.down_bgl,
            ));
            self.hiz_down.push((bg, (hw >> l).max(1), (hh >> l).max(1)));
        }
        self.hiz_view_all = Some(tex.create_view().map_err(js_err)?);
        self.hiz_size = (hw, hh);
        self.hiz_mips = mips;
        self.hiz_valid = false;
        self.generation += 1;
        Ok(())
    }

    /// Builds the Hi-Z pyramid from this frame's depth.
    pub fn encode_hiz(&mut self, enc: &GpuCommandEncoder, view_proj: glam::Mat4, ts: Option<GpuComputePassTimestampWrites>) {
        let Some(init) = &self.hiz_init_bg else {
            return;
        };
        let pass = compute_pass(enc, ts);
        pass.set_label("hi-z");
        pass.set_pipeline(&self.init_pipeline);
        pass.set_bind_group(0, Some(init));
        pass.dispatch_workgroups_with_workgroup_count_y(
            self.hiz_size.0.div_ceil(8),
            self.hiz_size.1.div_ceil(8),
        );
        pass.set_pipeline(&self.down_pipeline);
        for (bg, w, h) in &self.hiz_down {
            pass.set_bind_group(0, Some(bg));
            pass.dispatch_workgroups_with_workgroup_count_y(w.div_ceil(8), h.div_ceil(8));
        }
        pass.end();
        self.hiz_valid = true;
        self.prev_view_proj = view_proj;
    }
}

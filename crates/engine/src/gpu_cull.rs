//! GPU-driven culling: a compute pass does frustum + Hi-Z occlusion culling per instance and
//! writes `drawIndexedIndirect` arguments; the main pass draws every batch indirectly.
//!
//! Per frame (see `Renderer::render`):
//!   CPU: gather instances (+LOD pick), sort into (texture, mesh) batches, upload instances,
//!        bounding spheres (`items`) and indirect args with `instance_count = 0`.
//!   GPU: `cull.wgsl` appends surviving instance indices to each batch's slice of `visible`
//!        and increments `instance_count` -> main pass `drawIndexedIndirect` per batch
//!        -> `hiz.wgsl` builds the max-depth pyramid from this frame's depth for the next frame.
//! Counters (frustum-culled / occluded / visible) are read back asynchronously into
//! `FrameStats` (1-3 frames late) so agents can assert on them.
//!
//! Needs the `indirect-first-instance` device feature; without it the renderer falls back to
//! CPU frustum culling with direct draws.

use bytemuck::{Pod, Zeroable};
use js_sys::{JsNullable, Number};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::JsCast;
use web_sys::*;

pub(crate) const FEATURE: &str = "indirect-first-instance";
const HIZ_FORMAT: GpuTextureFormat = GpuTextureFormat::R32float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CullParams {
    pub planes: [[f32; 4]; 6],
    pub prev_view_proj: [[f32; 4]; 4],
    pub hiz_size: [f32; 2],
    pub count: u32,
    pub flags: u32,
    pub hiz_mips: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CullItem {
    pub center_radius: [f32; 4],
    pub batch: u32,
    /// Slot of the instance record in the instance buffer (written to the visible list).
    pub inst: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct DrawArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

fn js_err(e: wasm_bindgen::JsValue) -> String {
    format!("{e:?}")
}

fn nums(v: &[u32]) -> Vec<Number> {
    v.iter().map(|&x| Number::from(x)).collect()
}

fn compute_pass(enc: &GpuCommandEncoder, ts: Option<GpuComputePassTimestampWrites>) -> GpuComputePassEncoder {
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
    cull_pipeline: GpuComputePipeline,
    init_pipeline: GpuComputePipeline,
    down_pipeline: GpuComputePipeline,
    cull_bgl: GpuBindGroupLayout,
    init_bgl: GpuBindGroupLayout,
    down_bgl: GpuBindGroupLayout,
    params_buf: GpuBuffer,
    pub items_buf: GpuBuffer,
    items_cap: u32,
    pub args_buf: GpuBuffer,
    args_cap: u32,
    counters_buf: GpuBuffer,
    readback_buf: GpuBuffer,
    readback_pending: Rc<Cell<bool>>,
    /// frustum-culled, occluded, visible, (unused) — from the last completed readback.
    pub last_counters: Rc<RefCell<[u32; 4]>>,
    hiz_view_all: Option<GpuTextureView>,
    hiz_init_bg: Option<GpuBindGroup>,
    hiz_down: Vec<(GpuBindGroup, u32, u32)>,
    pub hiz_size: (u32, u32),
    pub hiz_mips: u32,
    /// The pyramid holds a depth buffer that matches `prev_view_proj`.
    pub hiz_valid: bool,
    pub prev_view_proj: glam::Mat4,
    cull_bg: Option<GpuBindGroup>,
}

impl GpuCull {
    pub fn new(device: &GpuDevice) -> Result<Self, String> {
        let cs = gpu_shader_stage::COMPUTE;
        let buf_entry = |binding: u32, ty: GpuBufferBindingType| {
            let e = GpuBindGroupLayoutEntry::new(binding, cs);
            let b = GpuBufferBindingLayout::new();
            b.set_type(ty);
            e.set_buffer(&b);
            e
        };
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
        let cull_bgl = bgl(&[
            buf_entry(0, GpuBufferBindingType::Uniform),
            buf_entry(1, GpuBufferBindingType::ReadOnlyStorage),
            buf_entry(2, GpuBufferBindingType::Storage),
            buf_entry(3, GpuBufferBindingType::Storage),
            buf_entry(4, GpuBufferBindingType::Storage),
            tex_entry(5, GpuTextureSampleType::UnfilterableFloat, false),
        ])?;
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
        let cull_mod = module(include_str!("shaders/cull.wgsl"), "cull.wgsl");
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
        let cull_pipeline = pipeline(&cull_mod, "cull", &cull_bgl);
        let init_pipeline = pipeline(&hiz_mod, "init", &init_bgl);
        let down_pipeline = pipeline(&hiz_mod, "down", &down_bgl);

        let buf = |size: u32, usage: u32, label: &str| -> Result<GpuBuffer, String> {
            let b = device
                .create_buffer(&GpuBufferDescriptor::new(size, usage))
                .map_err(js_err)?;
            b.set_label(label);
            Ok(b)
        };
        let params_buf = buf(
            std::mem::size_of::<CullParams>() as u32,
            gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST,
            "cull-params",
        )?;
        let items_cap = 1024;
        let items_buf = buf(
            items_cap * 32,
            gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
            "cull-items",
        )?;
        let args_cap = 256;
        let args_buf = buf(
            args_cap * 20,
            gpu_buffer_usage::INDIRECT | gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
            "indirect-args",
        )?;
        let counters_buf = buf(
            16,
            gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST | gpu_buffer_usage::COPY_SRC,
            "cull-counters",
        )?;
        let readback_buf = buf(
            16,
            gpu_buffer_usage::MAP_READ | gpu_buffer_usage::COPY_DST,
            "cull-readback",
        )?;

        Ok(Self {
            cull_pipeline,
            init_pipeline,
            down_pipeline,
            cull_bgl,
            init_bgl,
            down_bgl,
            params_buf,
            items_buf,
            items_cap,
            args_buf,
            args_cap,
            counters_buf,
            readback_buf,
            readback_pending: Rc::new(Cell::new(false)),
            last_counters: Rc::new(RefCell::new([0; 4])),
            hiz_view_all: None,
            hiz_init_bg: None,
            hiz_down: vec![],
            hiz_size: (1, 1),
            hiz_mips: 1,
            hiz_valid: false,
            prev_view_proj: glam::Mat4::IDENTITY,
            cull_bg: None,
        })
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
        self.cull_bg = None;
        Ok(())
    }

    /// Grows item/args buffers; returns true if bind groups must be rebuilt.
    pub fn ensure_capacity(
        &mut self,
        device: &GpuDevice,
        items: u32,
        batches: u32,
    ) -> Result<bool, String> {
        let mut changed = false;
        if items > self.items_cap {
            while self.items_cap < items {
                self.items_cap *= 2;
            }
            self.items_buf = device
                .create_buffer(&GpuBufferDescriptor::new(
                    self.items_cap * 32,
                    gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST,
                ))
                .map_err(js_err)?;
            changed = true;
        }
        if batches > self.args_cap {
            while self.args_cap < batches {
                self.args_cap *= 2;
            }
            self.args_buf = device
                .create_buffer(&GpuBufferDescriptor::new(
                    self.args_cap * 20,
                    gpu_buffer_usage::INDIRECT
                        | gpu_buffer_usage::STORAGE
                        | gpu_buffer_usage::COPY_DST,
                ))
                .map_err(js_err)?;
            changed = true;
        }
        if changed {
            self.cull_bg = None;
        }
        Ok(changed)
    }

    /// Encodes the culling dispatch. `visible` is the renderer's instance-index buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_cull(
        &mut self,
        device: &GpuDevice,
        queue: &GpuQueue,
        enc: &GpuCommandEncoder,
        visible: &GpuBuffer,
        params: &CullParams,
        items: &[CullItem],
        args: &[DrawArgs],
        ts: Option<GpuComputePassTimestampWrites>,
    ) -> Result<(), String> {
        queue
            .write_buffer_with_u32_and_u8_slice(&self.params_buf, 0, bytemuck::bytes_of(params))
            .map_err(js_err)?;
        if !items.is_empty() {
            queue
                .write_buffer_with_u32_and_u8_slice(&self.items_buf, 0, bytemuck::cast_slice(items))
                .map_err(js_err)?;
        }
        if !args.is_empty() {
            queue
                .write_buffer_with_u32_and_u8_slice(&self.args_buf, 0, bytemuck::cast_slice(args))
                .map_err(js_err)?;
        }
        queue
            .write_buffer_with_u32_and_u8_slice(&self.counters_buf, 0, &[0u8; 16])
            .map_err(js_err)?;
        let Some(hiz) = &self.hiz_view_all else {
            return Err("hi-z not initialised".into());
        };
        if self.cull_bg.is_none() {
            self.cull_bg = Some(device.create_bind_group(&GpuBindGroupDescriptor::new(
                &[
                    GpuBindGroupEntry::new_with_gpu_buffer(0, &self.params_buf),
                    GpuBindGroupEntry::new_with_gpu_buffer(1, &self.items_buf),
                    GpuBindGroupEntry::new_with_gpu_buffer(2, &self.args_buf),
                    GpuBindGroupEntry::new_with_gpu_buffer(3, visible),
                    GpuBindGroupEntry::new_with_gpu_buffer(4, &self.counters_buf),
                    GpuBindGroupEntry::new_with_gpu_texture_view(5, hiz),
                ],
                &self.cull_bgl,
            )));
        }
        if items.is_empty() {
            return Ok(());
        }
        let pass = compute_pass(enc, ts);
        pass.set_label("cull");
        pass.set_pipeline(&self.cull_pipeline);
        pass.set_bind_group(0, self.cull_bg.as_ref());
        pass.dispatch_workgroups((items.len() as u32).div_ceil(64));
        pass.end();
        Ok(())
    }

    /// Visible-index buffer was recreated by the renderer.
    pub fn invalidate_bindings(&mut self) {
        self.cull_bg = None;
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

    /// Copies the counters for async readback (skipped while a previous readback is in flight).
    pub fn encode_readback(&self, enc: &GpuCommandEncoder) -> bool {
        if self.readback_pending.get() {
            return false;
        }
        enc.copy_buffer_to_buffer_with_u32_and_u32(&self.counters_buf, 0, &self.readback_buf, 0)
            .is_ok()
    }

    /// Call after `queue.submit` if `encode_readback` returned true.
    pub fn start_readback(&self) {
        self.readback_pending.set(true);
        let buf = self.readback_buf.clone();
        let pending = self.readback_pending.clone();
        let out = self.last_counters.clone();
        let promise = buf.map_async(gpu_map_mode::READ);
        wasm_bindgen_futures::spawn_local(async move {
            if wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(
                promise.unchecked_into::<wasm_bindgen::JsValue>(),
            ))
            .await
            .is_ok()
            {
                if let Ok(range) = buf.get_mapped_range() {
                    let arr = js_sys::Uint32Array::new(&range);
                    let mut o = out.borrow_mut();
                    for (i, v) in o.iter_mut().enumerate() {
                        *v = arr.get_index(i as u32);
                    }
                }
                buf.unmap();
            }
            pending.set(false);
        });
    }
}

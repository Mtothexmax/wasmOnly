//! GPU grass: blades are generated entirely in the vertex shader from the instance index —
//! a camera-centred grid with hashed jitter, height from the terrain heightmap texture, density
//! from a terrain splat channel, distance thinning, wind. Nothing is uploaded per blade, so
//! 50–100k blades cost one draw call. Receives cascaded shadows, clustered lights and fog.
//!
//! Needs a terrain (`ctx.set_terrain`); tune via `ctx.renderer.grass` (`GrassSettings`) or the AI
//! command `grass`. Bushes, trees and rocks use terrain `ScatterRule`s (instanced meshes).

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use js_sys::JsNullable;
use serde::{Deserialize, Serialize};
use web_sys::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrassSettings {
    pub enabled: bool,
    /// Blades exist within this distance of the camera (m).
    pub radius: f32,
    /// Grid spacing between blades near the camera (m). Blade count ≈ (2·radius / spacing)².
    pub spacing: f32,
    pub height: f32,
    pub width: f32,
    /// Terrain splat layer (0..3) whose weight is the grass density.
    pub layer: u32,
    /// Fraction of the radius with full density; beyond, blades thin out (and widen).
    pub full_density: f32,
    pub color_base: Vec3,
    pub color_tip: Vec3,
    /// Wind direction (XZ) and strength.
    pub wind: [f32; 3],
}

impl Default for GrassSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            radius: 38.0,
            spacing: 0.22,
            height: 0.55,
            width: 0.07,
            layer: 0,
            full_density: 0.3,
            color_base: Vec3::new(0.05, 0.13, 0.03),
            color_tip: Vec3::new(0.32, 0.5, 0.12),
            wind: [0.8, 0.4, 0.25],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GrassParams {
    terrain: [f32; 4], // local corner x, z, size, res
    center: [f32; 4],  // local camera x, z, radius, spacing
    blade: [f32; 4],   // height, width, full density fraction, time
    wind: [f32; 4],    // dir x, z, strength, grid side n
    color_base: [f32; 4],
    color_tip: [f32; 4],
    layer: [u32; 4],
}

pub(crate) struct GrassRenderer {
    pub settings: GrassSettings,
    pipeline: GpuRenderPipeline,
    bgl: GpuBindGroupLayout,
    params: GpuBuffer,
    height: Option<(GpuTexture, GpuTextureView, u32)>,
    group: Option<GpuBindGroup>,
    splat: Option<GpuTextureView>,
    /// (local corner, size) of the terrain this frame.
    pub terrain: Option<([f32; 2], f32)>,
    pub last_blades: u32,
}

fn js_err(e: wasm_bindgen::JsValue) -> String {
    format!("{e:?}")
}

impl GrassRenderer {
    pub fn new(device: &GpuDevice, module: &GpuShaderModule, bgl0: &GpuBindGroupLayout, bgl_mat: &GpuBindGroupLayout, hdr: GpuTextureFormat, depth: GpuTextureFormat, samples: u32) -> Result<Self, String> {
        let vf = gpu_shader_stage::VERTEX | gpu_shader_stage::FRAGMENT;
        let u = GpuBindGroupLayoutEntry::new(0, vf);
        let b = GpuBufferBindingLayout::new();
        b.set_type(GpuBufferBindingType::Uniform);
        u.set_buffer(&b);
        let tex = |binding: u32| {
            let e = GpuBindGroupLayoutEntry::new(binding, gpu_shader_stage::VERTEX);
            let t = GpuTextureBindingLayout::new();
            t.set_sample_type(GpuTextureSampleType::UnfilterableFloat);
            e.set_texture(&t);
            e
        };
        let bgl = device.create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[u, tex(1), tex(2)])).map_err(js_err)?;
        let layout = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[JsNullable::wrap(bgl0.clone()), JsNullable::wrap(bgl_mat.clone()), JsNullable::wrap(bgl.clone())]));
        let vs = GpuVertexState::new(module);
        vs.set_entry_point("vs_grass");
        let fs = GpuFragmentState::new(module, &[JsNullable::wrap(GpuColorTargetState::new(hdr))]);
        fs.set_entry_point("fs_grass");
        let prim = GpuPrimitiveState::new();
        prim.set_cull_mode(GpuCullMode::None);
        let ds = GpuDepthStencilState::new(depth);
        ds.set_depth_write_enabled(true);
        ds.set_depth_compare(GpuCompareFunction::Less);
        let ms = GpuMultisampleState::new();
        ms.set_count(samples);
        let d = GpuRenderPipelineDescriptor::new(&layout, &vs);
        d.set_label("grass");
        d.set_fragment(&fs);
        d.set_primitive(&prim);
        d.set_depth_stencil(&ds);
        d.set_multisample(&ms);
        let pipeline = device.create_render_pipeline(&d).map_err(js_err)?;
        let params = device.create_buffer(&GpuBufferDescriptor::new(std::mem::size_of::<GrassParams>() as u32, gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST)).map_err(js_err)?;
        Ok(Self { settings: GrassSettings::default(), pipeline, bgl, params, height: None, group: None, splat: None, terrain: None, last_blades: 0 })
    }

    /// (Re)uploads the heightmap (row-major, `res`²) as an r32float texture.
    pub fn upload_heights(&mut self, device: &GpuDevice, queue: &GpuQueue, heights: &[f32], res: u32) -> Result<(), String> {
        if self.height.as_ref().map(|h| h.2) != Some(res) {
            let d = GpuTextureDescriptor::new(GpuTextureFormat::R32float, &[js_sys::Number::from(res), js_sys::Number::from(res)], gpu_texture_usage::TEXTURE_BINDING | gpu_texture_usage::COPY_DST);
            d.set_label("terrain-heights");
            let t = device.create_texture(&d).map_err(js_err)?;
            let v = t.create_view().map_err(js_err)?;
            self.height = Some((t, v, res));
            self.group = None;
        }
        let (t, _, _) = self.height.as_ref().unwrap();
        let dst = GpuTexelCopyTextureInfo::new(t);
        let layout = GpuTexelCopyBufferLayout::new();
        layout.set_bytes_per_row(res * 4);
        layout.set_rows_per_image(res);
        queue
            .write_texture_with_u8_slice_and_u32_sequence(&dst, bytemuck::cast_slice(heights), &layout, &[js_sys::Number::from(res), js_sys::Number::from(res)])
            .map_err(js_err)?;
        Ok(())
    }

    pub fn set_splat(&mut self, view: GpuTextureView) {
        self.splat = Some(view);
        self.group = None;
    }

    /// Writes this frame's parameters; returns the instance count to draw (0 = nothing).
    /// `quality` (0..1, from the adaptive resolution) shrinks the radius down to half.
    pub fn prepare(&mut self, device: &GpuDevice, queue: &GpuQueue, cam: Vec3, time: f32, quality: f32) -> u32 {
        let s = &self.settings;
        let (Some((corner, size)), Some((_, hv, res)), Some(splat)) = (self.terrain, self.height.as_ref(), self.splat.as_ref()) else { return 0 };
        if !s.enabled || s.spacing <= 0.01 {
            return 0;
        }
        if self.group.is_none() {
            self.group = Some(device.create_bind_group(&GpuBindGroupDescriptor::new(
                &[GpuBindGroupEntry::new_with_gpu_buffer(0, &self.params), GpuBindGroupEntry::new_with_gpu_texture_view(1, hv), GpuBindGroupEntry::new_with_gpu_texture_view(2, splat)],
                &self.bgl,
            )));
        }
        let radius = s.radius * (0.5 + 0.5 * quality.clamp(0.0, 1.0));
        let n = ((2.0 * radius / s.spacing).ceil() as u32).clamp(2, 1200);
        let p = GrassParams {
            terrain: [corner[0], corner[1], size, *res as f32],
            center: [cam.x, cam.z, radius, s.spacing],
            blade: [s.height, s.width, s.full_density, time],
            wind: [s.wind[0], s.wind[1], s.wind[2], n as f32],
            color_base: s.color_base.extend(1.0).into(),
            color_tip: s.color_tip.extend(1.0).into(),
            layer: [s.layer.min(3), n, 0, 0],
        };
        let _ = queue.write_buffer_with_u32_and_u8_slice(&self.params, 0, bytemuck::bytes_of(&p));
        self.last_blades = n * n;
        n * n
    }

    pub fn draw(&self, pass: &GpuRenderPassEncoder, bind0: &GpuBindGroup, any_material: &GpuBindGroup, blades: u32) {
        let Some(g) = &self.group else { return };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, Some(bind0));
        pass.set_bind_group(1, Some(any_material));
        pass.set_bind_group(2, Some(g));
        pass.draw_with_instance_count(15, blades);
    }
}

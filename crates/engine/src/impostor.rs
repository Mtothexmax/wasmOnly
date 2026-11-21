//! Impostors: the last LOD level of a model as one camera-facing card (2 triangles).
//!
//! [`Renderer::bake_impostor`] renders the model's primitives once from `views` azimuths around its
//! up axis (orthographic, `shaders/impostor.wgsl`) into an albedo atlas and a normal atlas (normals
//! in the card's tangent frame, so the card is lit like the model: sun, shadows, fog, day/night).
//! The card is an ordinary alpha-tested material with `Material::impostor_views` set: the vertex
//! shader turns it around the model's up axis toward the camera (shadow pass: toward the sun) and
//! picks the atlas cell of the nearest baked azimuth. glTF models get it with
//! `GltfOpts { impostor: true, .. }` (see `gltf_loader::instantiate`).

#![allow(deprecated)] // glam 0.33 deprecates Mat4::look_at_rh / orthographic_rh (see renderer.rs)

use crate::mesh::{MeshData, Vertex};
use crate::renderer::{js_err, nums, AlphaMode, Material, MaterialId, MeshId, Renderer, TextureId};
use glam::{Mat3, Mat4, Vec3, Vec4};
use js_sys::JsNullable;
use serde::Serialize;
use web_sys::*;

/// A baked impostor (shared by every instance of a model).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Impostor {
    /// The card (local x = across, y = up, centred on the model's vertical axis).
    pub mesh: MeshId,
    /// Mesh without triangles: the "nothing" level of the card's `Lod` below the switch distance.
    pub empty: MeshId,
    pub material: MaterialId,
    pub albedo: TextureId,
    pub normal: TextureId,
    /// Foot of the card's axis in model space (x, 0, z): the card entity's translation.
    pub pivot: Vec3,
    /// Bounding radius of the model (model space).
    pub radius: f32,
    pub views: u32,
    pub atlas: (u32, u32),
    /// Card size (width, height) in model units.
    pub size: (f32, f32),
}

pub(crate) struct ImpostorGpu {
    bake: GpuRenderPipeline,
    dilate: GpuRenderPipeline,
}

const SLOT: u32 = 256;

fn make_gpu(device: &GpuDevice) -> Result<ImpostorGpu, String> {
    let module = device.create_shader_module(&GpuShaderModuleDescriptor::new(include_str!("shaders/impostor.wgsl")));
    crate::debug::report_shader_errors(&module, "impostor.wgsl");
    let auto = wasm_bindgen::JsValue::from_str("auto");
    let auto: &GpuPipelineLayout = wasm_bindgen::JsCast::unchecked_ref(&auto);
    let targets = || {
        [
            JsNullable::wrap(GpuColorTargetState::new(GpuTextureFormat::Rgba8unormSrgb)),
            JsNullable::wrap(GpuColorTargetState::new(GpuTextureFormat::Rgba8unorm)),
        ]
    };
    // bake
    let attrs = [
        GpuVertexAttribute::new(GpuVertexFormat::Float32x3, 0, 0),
        GpuVertexAttribute::new(GpuVertexFormat::Float32x3, 12, 1),
        GpuVertexAttribute::new(GpuVertexFormat::Float32x2, 24, 2),
        GpuVertexAttribute::new(GpuVertexFormat::Unorm8x4, 32, 3),
    ];
    let vs = GpuVertexState::new(&module);
    vs.set_entry_point("vs_bake");
    vs.set_buffers(&[JsNullable::wrap(GpuVertexBufferLayout::new(std::mem::size_of::<Vertex>() as u32, &attrs))]);
    let fs = GpuFragmentState::new(&module, &targets());
    fs.set_entry_point("fs_bake");
    let prim = GpuPrimitiveState::new();
    prim.set_topology(GpuPrimitiveTopology::TriangleList);
    prim.set_cull_mode(GpuCullMode::None);
    let ds = GpuDepthStencilState::new(GpuTextureFormat::Depth24plus);
    ds.set_depth_write_enabled(true);
    ds.set_depth_compare(GpuCompareFunction::Less);
    let d = GpuRenderPipelineDescriptor::new(auto, &vs);
    d.set_label("impostor-bake");
    d.set_fragment(&fs);
    d.set_primitive(&prim);
    d.set_depth_stencil(&ds);
    let bake = device.create_render_pipeline(&d).map_err(js_err)?;
    // dilate
    let vs = GpuVertexState::new(&module);
    vs.set_entry_point("vs_full");
    let fs = GpuFragmentState::new(&module, &targets());
    fs.set_entry_point("fs_dilate");
    let d = GpuRenderPipelineDescriptor::new(auto, &vs);
    d.set_label("impostor-dilate");
    d.set_fragment(&fs);
    let dilate = device.create_render_pipeline(&d).map_err(js_err)?;
    Ok(ImpostorGpu { bake, dilate })
}

impl Renderer {
    /// Bakes an impostor of the primitives `parts` (mesh, material, part -> model matrix):
    /// `views` azimuths (1..16, 8 is a good default), `res` texels across one view (32..512).
    pub fn bake_impostor(&mut self, name: &str, parts: &[(MeshId, MaterialId, Mat4)], views: u32, res: u32) -> Result<Impostor, String> {
        let views = views.clamp(1, 16);
        let res = res.clamp(32, 512);
        // bounds in model space: AABB + horizontal radius around the vertical axis through its centre
        let mut pts: Vec<Vec3> = vec![];
        for (m, _, xf) in parts {
            let g = self.meshes.get(m.0 as usize).ok_or("unknown mesh")?;
            match &g.cpu {
                Some(c) => pts.extend(c.0.iter().map(|p| xf.transform_point3(Vec3::from(*p)))),
                None => {
                    let c = xf.transform_point3(g.bounds_center);
                    let r = g.bounds_radius * xf.x_axis.length().max(xf.y_axis.length()).max(xf.z_axis.length());
                    pts.extend((0..8).map(|k| c + Vec3::new([-r, r][k & 1], [-r, r][(k >> 1) & 1], [-r, r][(k >> 2) & 1])));
                }
            }
        }
        if pts.is_empty() {
            return Err("impostor: nothing to bake".into());
        }
        let (mn, mx) = pts.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
        let (cx, cz) = ((mn.x + mx.x) * 0.5, (mn.z + mx.z) * 0.5);
        let hw = pts.iter().map(|p| ((p.x - cx).powi(2) + (p.z - cz).powi(2)).sqrt()).fold(0.0f32, f32::max).max(1e-3);
        let pad = hw * 2.0 * 3.0 / res as f32; // 3 texels of margin per cell (mips, filtering)
        let (hw, y0, y1) = (hw + pad, mn.y - pad, mx.y + pad.max(1e-3));
        let ym = (y0 + y1) * 0.5;
        let radius = (hw * hw + (ym - y0).powi(2)).sqrt();
        let cols = views.min(4);
        let rows = views.div_ceil(cols);
        let cw = res;
        let ch = (((res as f32 * (y1 - y0) / (2.0 * hw)).clamp(res as f32 * 0.25, res as f32 * 4.0) / 4.0).round() as u32 * 4).clamp(8, 2048);
        let (aw, ah) = (cols * cw, rows * ch);

        if self.impostor_gpu.is_none() {
            self.impostor_gpu = Some(make_gpu(&self.device)?);
        }
        let mk = |fmt: GpuTextureFormat, label: &str| -> Result<GpuTexture, String> {
            let d = GpuTextureDescriptor::new(fmt, &nums(&[aw, ah]), gpu_texture_usage::RENDER_ATTACHMENT | gpu_texture_usage::TEXTURE_BINDING);
            d.set_label(label);
            self.device.create_texture(&d).map_err(js_err)
        };
        let tmp_a = mk(GpuTextureFormat::Rgba8unormSrgb, "impostor-bake-albedo")?;
        let tmp_n = mk(GpuTextureFormat::Rgba8unorm, "impostor-bake-normal")?;
        let depth = mk(GpuTextureFormat::Depth24plus, "impostor-bake-depth")?;
        let albedo = self.create_texture_fmt(aw, ah, &format!("{name}#impostor"), false)?;
        let normal = self.create_texture_fmt(aw, ah, &format!("{name}#impostor-normal"), true)?;

        // one uniform slot per (view, part)
        let n = views as usize * parts.len();
        let ubuf = self.device.create_buffer(&GpuBufferDescriptor::new(SLOT * n as u32, gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST)).map_err(js_err)?;
        let mut data = vec![0u8; SLOT as usize * n];
        let mut draws = vec![];
        for k in 0..views {
            let phi = k as f32 / views as f32 * std::f32::consts::TAU;
            let dir = Vec3::new(phi.sin(), 0.0, phi.cos());
            let target = Vec3::new(cx, ym, cz);
            let view = Mat4::look_at_rh(target + dir * (radius + 1.0), target, Vec3::Y);
            let proj = Mat4::orthographic_rh(-hw, hw, y0 - ym, y1 - ym, 0.01, 2.0 * radius + 2.0);
            for (pi, (mesh, mat, xf)) in parts.iter().enumerate() {
                let m = self.materials.get(mat.0 as usize).cloned().unwrap_or_default();
                let mv = view * *xf;
                let nrm = Mat4::from_mat3(Mat3::from_mat4(mv).inverse().transpose());
                let cut = match m.alpha_mode {
                    AlphaMode::Opaque => -1.0,
                    AlphaMode::Mask(c) => c,
                    AlphaMode::Blend => 0.5,
                };
                let mut f = [0f32; 40];
                f[..16].copy_from_slice(&(proj * mv).to_cols_array());
                f[16..32].copy_from_slice(&nrm.to_cols_array());
                f[32..36].copy_from_slice(&m.base_color.to_array());
                f[36] = cut;
                let o = (k as usize * parts.len() + pi) * SLOT as usize;
                data[o..o + 160].copy_from_slice(bytemuck::cast_slice(&f));
                let tex = m.texture.map(|t| t.0 + 1).unwrap_or(0);
                draws.push((k, *mesh, tex, o as u32));
            }
        }
        self.queue.write_buffer_with_u32_and_u8_slice(&ubuf, 0, &data).map_err(js_err)?;

        let gpu = self.impostor_gpu.as_ref().unwrap();
        let enc = self.device.create_command_encoder();
        {
            let va = tmp_a.create_view().map_err(js_err)?;
            let vn = tmp_n.create_view().map_err(js_err)?;
            let vd = depth.create_view().map_err(js_err)?;
            let att = |v: &GpuTextureView| {
                let a = GpuRenderPassColorAttachment::new_with_gpu_texture_view(GpuLoadOp::Clear, GpuStoreOp::Store, v);
                a.set_clear_value_gpu_color_dict(&GpuColorDict::new(0.0, 0.0, 0.0, 0.0));
                JsNullable::wrap(a)
            };
            let datt = GpuRenderPassDepthStencilAttachment::new_with_gpu_texture_view(&vd);
            datt.set_depth_clear_value(1.0);
            datt.set_depth_load_op(GpuLoadOp::Clear);
            datt.set_depth_store_op(GpuStoreOp::Discard);
            let desc = GpuRenderPassDescriptor::new(&[att(&va), att(&vn)]);
            desc.set_depth_stencil_attachment(&datt);
            desc.set_label("impostor-bake");
            let pass = enc.begin_render_pass(&desc).map_err(js_err)?;
            pass.set_pipeline(&gpu.bake);
            let bgl = gpu.bake.get_bind_group_layout(0);
            for (k, mesh, tex, off) in draws {
                let Some(g) = self.meshes.get(mesh.0 as usize).filter(|g| g.index_count > 0) else { continue };
                let b = GpuBufferBinding::new(&ubuf);
                b.set_offset(off);
                b.set_size(160);
                let tv = self.tex_views.get(tex as usize).unwrap_or(&self.tex_views[0]);
                let bg = self.device.create_bind_group(&GpuBindGroupDescriptor::new(
                    &[
                        GpuBindGroupEntry::new_with_gpu_buffer_binding(0, &b),
                        GpuBindGroupEntry::new_with_gpu_texture_view(1, tv),
                        GpuBindGroupEntry::new(2, &self.color_sampler),
                    ],
                    &bgl,
                ));
                pass.set_viewport(((k % cols) * cw) as f32, ((k / cols) * ch) as f32, cw as f32, ch as f32, 0.0, 1.0);
                pass.set_bind_group(0, Some(&bg));
                pass.set_vertex_buffer(0, Some(&g.vb));
                pass.set_index_buffer(&g.ib, GpuIndexFormat::Uint32);
                pass.draw_indexed(g.index_count);
            }
            pass.end();
        }
        // dilate into mip 0 of the final textures
        {
            let level0 = |id: TextureId| -> Result<GpuTextureView, String> {
                let t = self.tex_objects[id.0 as usize + 1].tex.clone().ok_or("impostor texture")?;
                let d = GpuTextureViewDescriptor::new();
                d.set_base_mip_level(0);
                d.set_mip_level_count(1);
                t.create_view_with_descriptor(&d).map_err(js_err)
            };
            let (oa, on) = (level0(albedo)?, level0(normal)?);
            let att = |v: &GpuTextureView| JsNullable::wrap(GpuRenderPassColorAttachment::new_with_gpu_texture_view(GpuLoadOp::Clear, GpuStoreOp::Store, v));
            let desc = GpuRenderPassDescriptor::new(&[att(&oa), att(&on)]);
            desc.set_label("impostor-dilate");
            let pass = enc.begin_render_pass(&desc).map_err(js_err)?;
            pass.set_pipeline(&gpu.dilate);
            let bg = self.device.create_bind_group(&GpuBindGroupDescriptor::new(
                &[
                    GpuBindGroupEntry::new_with_gpu_texture_view(0, &tmp_a.create_view().map_err(js_err)?),
                    GpuBindGroupEntry::new_with_gpu_texture_view(1, &tmp_n.create_view().map_err(js_err)?),
                ],
                &gpu.dilate.get_bind_group_layout(0),
            ));
            pass.set_bind_group(0, Some(&bg));
            pass.draw(3);
            pass.end();
        }
        self.queue.submit(&[enc.finish()]);
        for id in [albedo, normal] {
            let info = &self.tex_objects[id.0 as usize + 1];
            let (t, mips, linear) = (info.tex.clone().ok_or("impostor texture")?, info.mips, info.linear);
            self.generate_mips(&t, mips, linear)?;
            self.tex_objects[id.0 as usize + 1].uploads += 1;
        }
        tmp_a.destroy();
        tmp_n.destroy();
        depth.destroy();
        ubuf.destroy();

        // the card: x across, y up, facing +z; uv v = 0 at the top
        let q = |x: f32, y: f32, u: f32, v: f32| Vertex::new([x, y, 0.0], [0.0, 0.0, 1.0], [u, v]);
        let card = MeshData { vertices: vec![q(-hw, y0, 0.0, 1.0), q(hw, y0, 1.0, 1.0), q(hw, y1, 1.0, 0.0), q(-hw, y1, 0.0, 0.0)], indices: vec![0, 1, 2, 0, 2, 3] };
        let mesh = self.add_mesh(&format!("{name}#impostor"), &card);
        let empty = self.empty_mesh();
        // surface parameters of the part with the most triangles
        let main = parts.iter().max_by_key(|(m, _, _)| self.meshes.get(m.0 as usize).map_or(0, |g| g.index_count)).map(|p| p.1);
        let src = main.and_then(|m| self.materials.get(m.0 as usize).cloned()).unwrap_or_default();
        let mat = Material {
            name: format!("{name}#impostor"),
            base_color: Vec4::ONE,
            metallic: src.metallic,
            roughness: src.roughness,
            texture: Some(albedo),
            normal_texture: Some(normal),
            normal_scale: 1.0,
            alpha_mode: AlphaMode::Mask(0.5),
            double_sided: true,
            subsurface: src.subsurface,
            subsurface_color: src.subsurface_color,
            impostor_views: views,
            ..Default::default()
        };
        let material = self.add_material(mat);
        crate::log_info!("impostor {name}: {views} views, atlas {aw}x{ah}, card {:.2} x {:.2}", 2.0 * hw, y1 - y0);
        Ok(Impostor { mesh, empty, material, albedo, normal, pivot: Vec3::new(cx, 0.0, cz), radius, views, atlas: (aw, ah), size: (2.0 * hw, y1 - y0) })
    }

    /// A shared mesh without triangles (an "invisible" LOD level).
    pub fn empty_mesh(&mut self) -> MeshId {
        if let Some(m) = self.empty_mesh_id {
            return m;
        }
        let m = self.add_mesh("empty", &MeshData::default());
        self.empty_mesh_id = Some(m);
        m
    }
}

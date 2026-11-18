//! Render to texture: a camera renders the scene (full pipeline: shadows, sky/fog, bloom,
//! tonemapping) into a texture, e.g. card portraits, mirrors, minimaps, security cameras.
//!
//! ```ignore
//! // portrait subjects live on layer 1, the main camera skips that layer
//! let (rt, tex) = ctx.create_render_texture(256, 320, RenderTextureOpts::portrait());
//! let cam = ctx.world.spawn((Transform::from_xyz(0., 1.2, 2.5).looking_at(Vec3::Y, Vec3::Y),
//!                            Camera { layers: 1 << 1, ..Default::default() }));
//! ctx.set_render_texture_camera(rt, cam);
//! // use it: as a material texture in 3D ...
//! let card = ctx.renderer.add_material(Material { texture: Some(tex), unlit: true, ..Material::color(0xffffff) });
//! // ... or in the DOM: export once rendered, then <img src=url>
//! ctx.export_render_texture(rt);
//! if let Some(url) = ctx.renderer.render_texture_url(rt) { ctx.ui.set("card", &format!("<img src='{url}'>")); }
//! ```
//!
//! The camera of a render texture is marked with [`RenderTextureCamera`] so it never becomes the
//! main camera. `layers` on `Camera` and `layer` on `MeshRenderer` select what each camera sees.
//!
//! AI API: `render_texture` lists them (size, camera, renders, export url); `{id, render: true}`
//! re-renders a `once` texture; `{id, export: true}` creates a PNG blob URL (`url` on the next call).

use crate::post::PostTargets;
use crate::renderer::{RenderSettings, TextureId};
use glam::Vec3;
use hecs::Entity;
use serde::Serialize;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use web_sys::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct RenderTextureId(pub u32);

/// Marks a camera entity as belonging to a render texture (it is never the main camera).
#[derive(Clone, Copy, Debug)]
pub struct RenderTextureCamera(pub RenderTextureId);

#[derive(Clone, Debug, Serialize)]
pub struct RenderTextureOpts {
    /// Re-render every frame (live view) or only when requested (portraits).
    pub every_frame: bool,
    /// Background where nothing was drawn becomes transparent (alpha 0). UI panels drawn into the
    /// texture become transparent too.
    pub transparent: bool,
    pub sky: bool,
    pub fog: bool,
    pub bloom: bool,
    pub shadows: bool,
    /// Linear background colour when `sky` is off and not transparent.
    pub clear_color: Vec3,
    pub exposure: f32,
}

impl Default for RenderTextureOpts {
    fn default() -> Self {
        Self { every_frame: true, transparent: false, sky: true, fog: true, bloom: true, shadows: true, clear_color: Vec3::new(0.1, 0.1, 0.12), exposure: 1.0 }
    }
}

impl RenderTextureOpts {
    /// Rendered once (and on request), no sky/fog, transparent background.
    pub fn portrait() -> Self {
        Self { every_frame: false, transparent: true, sky: false, fog: false, ..Self::default() }
    }
    pub fn live() -> Self {
        Self::default()
    }
}

pub(crate) struct RenderTexture {
    pub size: (u32, u32),
    pub texture: TextureId,
    pub gpu: GpuTexture,
    pub view0: GpuTextureView,
    pub mips: u32,
    pub targets: PostTargets,
    pub camera: Option<Entity>,
    pub opts: RenderTextureOpts,
    pub pending: bool,
    pub renders: u64,
    pub label: String,
    /// Blob URL of the last PNG export (shared with the async export task).
    pub export: Rc<RefCell<Option<String>>>,
    pub exporting: Rc<RefCell<bool>>,
}

impl RenderTexture {
    pub fn due(&self) -> bool {
        self.camera.is_some() && (self.opts.every_frame || self.pending)
    }

    /// Settings for this texture's render (the main settings with this texture's overrides).
    pub fn apply(&self, s: &mut RenderSettings) {
        s.sky = self.opts.sky && !self.opts.transparent;
        if !self.opts.fog {
            s.fog_density = 0.0;
        }
        s.bloom &= self.opts.bloom;
        s.shadows &= self.opts.shadows;
        s.clear_color = self.opts.clear_color;
        s.exposure = self.opts.exposure;
        s.vignette = 0.0;
        s.camera_jitter = Vec3::ZERO;
        s.depth_jitter = 0.0;
        s.reverse_draw_order = false;
        s.debug_view = 0;
    }

    pub fn describe(&self, id: usize) -> serde_json::Value {
        serde_json::json!({
            "id": id, "label": self.label, "size": [self.size.0, self.size.1], "texture": self.texture.0,
            "camera": self.camera.map(|c| c.id()), "opts": self.opts, "renders": self.renders,
            "pending": self.pending, "url": *self.export.borrow(), "exporting": *self.exporting.borrow(),
        })
    }
}

fn js_err(e: wasm_bindgen::JsValue) -> String {
    format!("{e:?}")
}

/// Reads the texture back and turns it into a PNG blob URL (async; result in `export`).
pub(crate) fn start_export(device: &GpuDevice, queue: &GpuQueue, rt: &RenderTexture) -> Result<(), String> {
    if *rt.exporting.borrow() {
        return Ok(());
    }
    let (w, h) = rt.size;
    let row = (w * 4).div_ceil(256) * 256;
    let buf = device
        .create_buffer(&GpuBufferDescriptor::new(row * h, gpu_buffer_usage::MAP_READ | gpu_buffer_usage::COPY_DST))
        .map_err(js_err)?;
    let enc = device.create_command_encoder();
    let src = GpuTexelCopyTextureInfo::new(&rt.gpu);
    let dst = GpuTexelCopyBufferInfo::new(&buf);
    dst.set_bytes_per_row(row);
    dst.set_rows_per_image(h);
    let size: Vec<js_sys::Number> = [w, h].iter().map(|x| js_sys::Number::from(*x)).collect();
    enc.copy_texture_to_buffer_with_u32_sequence(&src, &dst, &size).map_err(js_err)?;
    queue.submit(&[enc.finish()]);
    let (export, exporting, transparent) = (rt.export.clone(), rt.exporting.clone(), rt.opts.transparent);
    *exporting.borrow_mut() = true;
    let label = rt.label.clone();
    let promise = buf.map_async(gpu_map_mode::READ);
    wasm_bindgen_futures::spawn_local(async move {
        let r: Result<String, String> = async {
            wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(wasm_bindgen::JsValue::from(promise))).await.map_err(js_err)?;
            let mapped = js_sys::Uint8Array::new(&buf.get_mapped_range().map_err(js_err)?.into()).to_vec();
            buf.unmap();
            let mut px = vec![0u8; (w * h * 4) as usize];
            for y in 0..h as usize {
                let s = y * row as usize;
                px[y * w as usize * 4..(y + 1) * w as usize * 4].copy_from_slice(&mapped[s..s + w as usize * 4]);
            }
            if !transparent {
                for a in px.iter_mut().skip(3).step_by(4) {
                    *a = 255;
                }
            }
            let data = ImageData::new_with_u8_clamped_array_and_sh(wasm_bindgen::Clamped(&px), w, h).map_err(js_err)?;
            let canvas = OffscreenCanvas::new(w, h).map_err(js_err)?;
            let ctx: OffscreenCanvasRenderingContext2d = canvas.get_context("2d").map_err(js_err)?.ok_or("no 2d context")?.unchecked_into();
            ctx.put_image_data(&data, 0, 0).map_err(js_err)?;
            let blob: Blob = wasm_bindgen_futures::JsFuture::from(canvas.convert_to_blob().map_err(js_err)?).await.map_err(js_err)?.unchecked_into();
            let url = Url::create_object_url_with_blob(&blob).map_err(js_err)?;
            Ok(url)
        }
        .await;
        match r {
            Ok(url) => {
                if let Some(old) = export.borrow_mut().replace(url) {
                    let _ = Url::revoke_object_url(&old);
                }
                crate::log_info!("render texture '{label}' exported");
            }
            Err(e) => crate::log_error!("render texture '{label}' export: {e}"),
        }
        *exporting.borrow_mut() = false;
    });
    Ok(())
}

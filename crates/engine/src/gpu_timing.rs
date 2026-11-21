//! GPU time per render pass via WebGPU timestamp queries (`timestamp-query` feature; Chrome
//! quantizes timestamps to ~0.1 ms). Two timestamps per pass, resolved into a buffer and read back
//! asynchronously; `FrameStats::gpu_ms` shows the last completed measurement (1–3 frames old).
//! While a readback is in flight, frames are not timed (no stalls).

use serde::Serialize;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::JsCast;
use web_sys::*;

pub(crate) const FEATURE: &str = "timestamp-query";
/// Pass slots: cull (compute), shadow, opaque, blend, hiz (compute).
pub(crate) const CULL: usize = 0;
pub(crate) const SHADOW: usize = 1;
pub(crate) const OPAQUE: usize = 2;
pub(crate) const BLEND: usize = 3;
pub(crate) const HIZ: usize = 4;
/// Bloom + composite (begin on the first post pass, end on the last).
pub(crate) const POST: usize = 5;
/// gpu_slow test pass (`render {gpu_slow_ms}`).
pub(crate) const SLOW: usize = 6;
const N: usize = 7;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct GpuTimes {
    pub cull: f32,
    pub shadow: f32,
    pub opaque: f32,
    pub blend: f32,
    pub hiz: f32,
    /// Bloom + depth darkening + tonemap.
    pub post: f32,
    /// gpu_slow test pass (`render {gpu_slow_ms}`; 0 = off).
    pub slow: f32,
    /// Sum of the measured passes (ms).
    pub total: f32,
    /// First pass begin to last pass end (ms): also contains what runs between the timed passes
    /// (texture copies, resolves, state transitions), so it is the better "whole frame" number.
    pub frame: f32,
    /// Begin and end of each pass in ms since the first pass began, in the order cull, shadow,
    /// opaque, blend, hiz, post (latest measurement, not smoothed; 0 = pass not timed). Gaps
    /// between one pass's end and the next one's begin are untimed GPU work.
    pub timeline: [[f32; 2]; 7],
    /// Number of completed measurements so far.
    pub samples: u32,
}

pub(crate) struct GpuTiming {
    qs: GpuQuerySet,
    resolve: GpuBuffer,
    read: GpuBuffer,
    pending: Rc<Cell<bool>>,
    pub last: Rc<RefCell<GpuTimes>>,
    active: bool,
    used: [bool; N],
}

impl GpuTiming {
    pub fn new(device: &GpuDevice) -> Result<Self, String> {
        let d = GpuQuerySetDescriptor::new((N * 2) as u32, GpuQueryType::Timestamp);
        d.set_label("pass-timestamps");
        let qs = device.create_query_set(&d).map_err(|e| format!("{e:?}"))?;
        let size = (N * 2 * 8) as u32;
        let resolve = device
            .create_buffer(&GpuBufferDescriptor::new(size, gpu_buffer_usage::QUERY_RESOLVE | gpu_buffer_usage::COPY_SRC))
            .map_err(|e| format!("{e:?}"))?;
        let read = device
            .create_buffer(&GpuBufferDescriptor::new(size, gpu_buffer_usage::MAP_READ | gpu_buffer_usage::COPY_DST))
            .map_err(|e| format!("{e:?}"))?;
        Ok(Self { qs, resolve, read, pending: Rc::new(Cell::new(false)), last: Rc::new(RefCell::new(GpuTimes::default())), active: false, used: [false; N] })
    }

    /// Start of a frame: time it only if the previous readback finished.
    pub fn begin_frame(&mut self) {
        self.active = !self.pending.get();
        self.used = [false; N];
    }

    pub fn render_writes(&mut self, slot: usize) -> Option<GpuRenderPassTimestampWrites> {
        if !self.active {
            return None;
        }
        self.used[slot] = true;
        let w = GpuRenderPassTimestampWrites::new(&self.qs);
        w.set_beginning_of_pass_write_index((slot * 2) as u32);
        w.set_end_of_pass_write_index((slot * 2 + 1) as u32);
        Some(w)
    }

    /// Split writes for a sequence of passes: (begin-only for the first, end-only for the last).
    pub fn split_writes(&mut self, slot: usize) -> Option<(GpuRenderPassTimestampWrites, GpuRenderPassTimestampWrites)> {
        if !self.active {
            return None;
        }
        self.used[slot] = true;
        let b = GpuRenderPassTimestampWrites::new(&self.qs);
        b.set_beginning_of_pass_write_index((slot * 2) as u32);
        let e = GpuRenderPassTimestampWrites::new(&self.qs);
        e.set_end_of_pass_write_index((slot * 2 + 1) as u32);
        Some((b, e))
    }

    pub fn compute_writes(&mut self, slot: usize) -> Option<GpuComputePassTimestampWrites> {
        if !self.active {
            return None;
        }
        self.used[slot] = true;
        let w = GpuComputePassTimestampWrites::new(&self.qs);
        w.set_beginning_of_pass_write_index((slot * 2) as u32);
        w.set_end_of_pass_write_index((slot * 2 + 1) as u32);
        Some(w)
    }

    /// After the last pass: resolve the queries and copy them for readback.
    pub fn encode_resolve(&self, enc: &GpuCommandEncoder) {
        if !self.active || !self.used.iter().any(|u| *u) {
            return;
        }
        enc.resolve_query_set_with_u32(&self.qs, 0, (N * 2) as u32, &self.resolve, 0);
        let _ = enc.copy_buffer_to_buffer_with_u32_and_u32(&self.resolve, 0, &self.read, 0);
    }

    /// After `queue.submit`: map the readback buffer asynchronously.
    pub fn start_readback(&self) {
        if !self.active || !self.used.iter().any(|u| *u) {
            return;
        }
        self.pending.set(true);
        let (buf, pending, last, used) = (self.read.clone(), self.pending.clone(), self.last.clone(), self.used);
        let promise = buf.map_async(gpu_map_mode::READ);
        wasm_bindgen_futures::spawn_local(async move {
            if wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise.unchecked_into::<wasm_bindgen::JsValue>())).await.is_ok() {
                if let Ok(range) = buf.get_mapped_range() {
                    let words = js_sys::Uint32Array::new(&range);
                    let ts = |i: usize| words.get_index((i * 2) as u32) as f64 + words.get_index((i * 2 + 1) as u32) as f64 * 4_294_967_296.0;
                    let mut ms = [0.0f32; N];
                    let (mut first, mut last_end) = (f64::MAX, 0.0f64);
                    for (slot, m) in ms.iter_mut().enumerate() {
                        if used[slot] {
                            let (a, b) = (ts(slot * 2), ts(slot * 2 + 1));
                            if b >= a && a > 0.0 {
                                *m = ((b - a) / 1.0e6) as f32;
                                first = first.min(a);
                                last_end = last_end.max(b);
                            }
                        }
                    }
                    let span = if last_end > first { ((last_end - first) / 1.0e6) as f32 } else { 0.0 };
                    let mut l = last.borrow_mut();
                    // light smoothing (timestamps are quantized)
                    let k = if l.samples == 0 { 1.0 } else { 0.2 };
                    let mix = |old: f32, new: f32| old + (new - old) * k;
                    l.cull = mix(l.cull, ms[CULL]);
                    l.shadow = mix(l.shadow, ms[SHADOW]);
                    l.opaque = mix(l.opaque, ms[OPAQUE]);
                    l.blend = mix(l.blend, ms[BLEND]);
                    l.hiz = mix(l.hiz, ms[HIZ]);
                    l.post = mix(l.post, ms[POST]);
                    l.slow = if used[SLOW] { mix(l.slow, ms[SLOW]) } else { 0.0 };
                    l.total = l.cull + l.shadow + l.opaque + l.blend + l.hiz + l.post + l.slow;
                    l.frame = mix(l.frame, span);
                    for slot in 0..N {
                        let (a, b) = (ts(slot * 2), ts(slot * 2 + 1));
                        l.timeline[slot] = if used[slot] && b >= a && a >= first { [((a - first) / 1.0e6) as f32, ((b - first) / 1.0e6) as f32] } else { [0.0; 2] };
                    }
                    l.samples += 1;
                }
                buf.unmap();
            }
            pending.set(false);
        });
    }
}

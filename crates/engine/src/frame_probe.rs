//! `frame_probe`: reads back one *real* frame of the renderer, to find out on a device whose picture stays
//! empty where the data goes wrong (made for a PowerVR B-series phone, Chrome 158, white canvas with every
//! shader running fine in `tools/draw-probe.mjs`).
//!
//! On the next main-view frame it encodes, before the submit:
//! - **uniform / storage echoes**: a compute shader reads the engine's own buffers through the engine's own
//!   struct declarations (`Globals` via `g`, the cull `Params` via `p`, one `Instance` record) and writes every
//!   field back as words at its WGSL layout offset, three times: (a) arrays read with a runtime index (as the
//!   engine does), (b) arrays unrolled with constant indices, (c) the raw buffer as `array<vec4u>`. Each is
//!   compared word by word with a plain copy of the buffer: a mismatch names the field the GPU reads wrong.
//! - the centre pixel of the HDR target (after the scene passes) and, when the page was opened with
//!   `?frame_probe=1` (canvas configured with COPY_SRC), of the canvas.
//! The result (with the cull counters of that frame) arrives a few frames later: `frame_probe` again, and as
//! two log lines (`frame probe: <summary>`, `frame probe result: <json>`). With `?frame_probe=1` the probe
//! also arms itself once, 180 main frames after start (no API call needed on a phone).
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::*;

/// One field of a WGSL struct with its byte offset (uniform/storage layout rules for the types the engine uses).
#[derive(Clone, Debug)]
pub(crate) struct Field {
    pub name: String,
    pub ty: String,
    pub offset: u32,
}

fn type_layout(ty: &str) -> Option<(u32, u32)> {
    let ty = ty.trim();
    Some(match ty {
        "f32" | "u32" | "i32" | "atomic<u32>" => (4, 4),
        "vec2f" | "vec2u" | "vec2i" => (8, 8),
        "vec3f" | "vec3u" | "vec3i" => (12, 16),
        "vec4f" | "vec4u" | "vec4i" => (16, 16),
        "mat4x4f" => (64, 16),
        _ => {
            let (t, n) = array_of(ty)?;
            let (s, a) = type_layout(&t)?;
            let stride = s.div_ceil(a) * a;
            (stride * n, a)
        }
    })
}

/// `array<T, N>` -> (T, N)
fn array_of(ty: &str) -> Option<(String, u32)> {
    let inner = ty.strip_prefix("array<")?.strip_suffix('>')?;
    let i = inner.rfind(',')?;
    Some((inner[..i].trim().to_string(), inner[i + 1..].trim().parse().ok()?))
}

/// Fields of `struct name { .. }` in `src` with offsets, and the struct size.
pub(crate) fn struct_layout(src: &str, name: &str) -> Option<(Vec<Field>, u32)> {
    let clean: String = src.lines().map(|l| l.split("//").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
    let start = clean.find(&format!("struct {name} {{"))?;
    let body_start = start + clean[start..].find('{')? + 1;
    let body_end = body_start + clean[body_start..].find('}')?;
    let mut fields = Vec::new();
    let (mut off, mut align) = (0u32, 4u32);
    let mut depth = 0i32;
    let mut cur = String::new();
    let mut parts = Vec::new();
    for ch in clean[body_start..body_end].chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            _ => {}
        }
        if ch == ',' && depth == 0 {
            parts.push(std::mem::take(&mut cur));
        } else {
            cur.push(ch);
        }
    }
    parts.push(cur);
    for p in parts {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        let (n, t) = p.split_once(':')?;
        let (size, a) = type_layout(t)?;
        off = off.div_ceil(a) * a;
        fields.push(Field { name: n.trim().to_string(), ty: t.trim().to_string(), offset: off });
        off += size;
        align = align.max(a);
    }
    Some((fields, off.div_ceil(align) * align))
}

/// WGSL statements writing value `e` of type `ty` as u32 words at word offset `w` (an expression) + `base`.
fn emit(out: &mut String, e: &str, ty: &str, w: &str, dynamic: bool, depth: u32) {
    let ty = ty.trim();
    let scalar = |out: &mut String, e: &str, ty: &str, w: &str| {
        let v = match ty {
            "u32" => e.to_string(),
            _ => format!("bitcast<u32>({e})"),
        };
        out.push_str(&format!("fp_out[{w}] = {v};\n"));
    };
    match ty {
        "f32" | "u32" | "i32" => scalar(out, e, ty, w),
        "vec2f" | "vec3f" | "vec4f" | "vec2u" | "vec3u" | "vec4u" | "vec2i" | "vec3i" | "vec4i" => {
            let n: usize = ty[3..4].parse().unwrap_or(4);
            let st = match &ty[4..] {
                "u" => "u32",
                "i" => "i32",
                _ => "f32",
            };
            for (i, c) in ["x", "y", "z", "w"].iter().take(n).enumerate() {
                scalar(out, &format!("({e}).{c}"), st, &format!("{w} + {i}u"));
            }
        }
        "mat4x4f" => {
            for c in 0..4 {
                emit(out, &format!("({e})[{c}]"), "vec4f", &format!("{w} + {}u", c * 4), dynamic, depth);
            }
        }
        _ => {
            if let Some((t, n)) = array_of(ty) {
                let (s, a) = type_layout(&t).unwrap_or((16, 16));
                let stride = s.div_ceil(a) * a / 4;
                if dynamic {
                    let k = format!("fp_k{depth}");
                    out.push_str(&format!("for (var {k} = 0u; {k} < {n}u; {k}++) {{\n"));
                    emit(out, &format!("({e})[{k}]"), &t, &format!("{w} + {k} * {stride}u"), dynamic, depth + 1);
                    out.push_str("}\n");
                } else {
                    for i in 0..n {
                        emit(out, &format!("({e})[{i}]"), &t, &format!("{w} + {}u", i * stride), dynamic, depth + 1);
                    }
                }
            }
        }
    }
}

/// Echo shader: `prefix` declares the struct and its binding; `access` is the expression of the struct value
/// (`g`, `p`, `instances[3]`); `raw_decl` declares `fp_raw` over the same buffer, `raw_base` its first vec4.
fn echo_wgsl(prefix: &str, fields: &[Field], words: u32, access: &str, raw_decl: &str, raw_base: u32) -> String {
    let mut s = String::new();
    s.push_str(prefix);
    s.push_str("\n@group(1) @binding(0) var<storage, read_write> fp_out: array<u32>;\n");
    s.push_str(raw_decl);
    s.push_str("\n@compute @workgroup_size(1) fn fp_echo() {\n");
    for (region, dynamic) in [(0u32, true), (1, false)] {
        for f in fields {
            emit(&mut s, &format!("{access}.{}", f.name), &f.ty, &format!("{}u", region * words + f.offset / 4), dynamic, 0);
        }
    }
    s.push_str(&format!(
        "for (var i = 0u; i < {}u; i++) {{ let v = fp_raw[{raw_base}u + i]; let o = {}u + i * 4u; fp_out[o] = v.x; fp_out[o + 1u] = v.y; fp_out[o + 2u] = v.z; fp_out[o + 3u] = v.w; }}\n}}\n",
        words / 4,
        2 * words
    ));
    s
}

/// One echo target of this probe run.
pub(crate) struct EchoSpec<'a> {
    pub label: &'static str,
    /// shader source declaring the struct + its binding at group 0 (common.wgsl, cull.wgsl)
    pub prefix: &'a str,
    pub struct_name: &'static str,
    pub access: String,
    /// the engine buffer read by the echo (bound at group 0 binding `binding` like in the engine)
    pub buffer: &'a GpuBuffer,
    pub binding: u32,
    pub uniform: bool,
    /// byte offset of the record inside `buffer` (instance slot * record size)
    pub byte_offset: u32,
}

struct EchoRun {
    label: &'static str,
    fields: Vec<Field>,
    words: u32,
    out: GpuBuffer,  // 3 regions (dyn, const, raw) + the plain copy, MAP_READ copy target below
    read: GpuBuffer, // MAP_READ: 4 * words u32
}

/// Encodes the echoes + pixel copies into `enc`; returns what `finish` needs after the submit.
pub(crate) struct Pending {
    runs: Vec<EchoRun>,
    pixels: Vec<(&'static str, GpuBuffer, GpuTextureFormat)>,
    /// 16 x 9 grid of samples per texture (one 256-byte row per sample)
    grids: Vec<(&'static str, GpuBuffer, GpuTextureFormat)>,
    pub info: serde_json::Value,
    errors: Vec<String>,
}

pub(crate) fn encode(device: &GpuDevice, enc: &GpuCommandEncoder, specs: &[EchoSpec], pixels: &[(&'static str, &GpuTexture, (u32, u32))], info: serde_json::Value) -> Pending {
    let mut p = Pending { runs: Vec::new(), pixels: Vec::new(), grids: Vec::new(), info, errors: Vec::new() };
    let buf = |size: u32, usage: u32| device.create_buffer(&GpuBufferDescriptor::new(size.max(16), usage)).map_err(|e| format!("{e:?}"));
    for s in specs {
        let Some((fields, size)) = struct_layout(s.prefix, s.struct_name) else {
            p.errors.push(format!("{}: struct {} not parsed", s.label, s.struct_name));
            continue;
        };
        let words = size / 4;
        let raw_decl = if s.uniform {
            format!("@group(1) @binding(1) var<uniform> fp_raw: array<vec4u, {}>;\n", words / 4)
        } else {
            "@group(1) @binding(1) var<storage, read> fp_raw: array<vec4u>;\n".to_string()
        };
        let raw_base = if s.uniform { 0 } else { s.byte_offset / 16 };
        let code = echo_wgsl(s.prefix, &fields, words, &s.access, &raw_decl, raw_base);
        let run = (|| -> Result<EchoRun, String> {
            let module = device.create_shader_module(&GpuShaderModuleDescriptor::new(&code));
            module.set_label(&format!("frame-probe-{}", s.label));
            crate::debug::report_shader_errors(&module, &format!("frame_probe {}", s.label));
            let stage = GpuProgrammableStage::new(&module);
            stage.set_entry_point("fp_echo");
            let auto = JsValue::from_str("auto");
            let d = GpuComputePipelineDescriptor::new(auto.unchecked_ref::<GpuPipelineLayout>(), &stage);
            let pipe = device.create_compute_pipeline(&d);
            let out = buf(words * 3 * 4, gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_SRC)?;
            let read = buf(words * 4 * 4, gpu_buffer_usage::MAP_READ | gpu_buffer_usage::COPY_DST)?;
            let g0 = GpuBindGroupEntry::new_with_gpu_buffer(s.binding, s.buffer);
            let bg0 = device.create_bind_group(&GpuBindGroupDescriptor::new(&[g0], &pipe.get_bind_group_layout(0)));
            let raw = GpuBufferBinding::new(s.buffer);
            if s.uniform {
                raw.set_size(size);
            }
            let bg1 = device.create_bind_group(&GpuBindGroupDescriptor::new(
                &[GpuBindGroupEntry::new_with_gpu_buffer(0, &out), GpuBindGroupEntry::new_with_gpu_buffer_binding(1, &raw)],
                &pipe.get_bind_group_layout(1),
            ));
            let pass = enc.begin_compute_pass();
            pass.set_label(&format!("frame-probe-{}", s.label));
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, Some(&bg0));
            pass.set_bind_group(1, Some(&bg1));
            pass.dispatch_workgroups(1);
            pass.end();
            enc.copy_buffer_to_buffer_with_u32_and_u32_and_u32(&out, 0, &read, 0, words * 3 * 4).map_err(|e| format!("{e:?}"))?;
            enc.copy_buffer_to_buffer_with_u32_and_u32_and_u32(s.buffer, s.byte_offset, &read, words * 3 * 4, words * 4).map_err(|e| format!("{e:?}"))?;
            Ok(EchoRun { label: s.label, fields, words, out, read })
        })();
        match run {
            Ok(r) => p.runs.push(r),
            Err(e) => p.errors.push(format!("{}: {e}", s.label)),
        }
    }
    for (label, tex, (x, y)) in pixels {
        let Ok(b) = buf(256, gpu_buffer_usage::MAP_READ | gpu_buffer_usage::COPY_DST) else { continue };
        let src = GpuTexelCopyTextureInfo::new(tex);
        src.set_origin(&crate::renderer::nums(&[*x, *y]));
        let dst = GpuTexelCopyBufferInfo::new(&b);
        dst.set_bytes_per_row(256);
        match enc.copy_texture_to_buffer_with_u32_sequence(&src, &dst, &crate::renderer::nums(&[1, 1])) {
            Ok(()) => p.pixels.push((label, b, tex.format())),
            Err(e) => p.errors.push(format!("pixel {label}: {e:?}")),
        }
        // the whole picture at a glance: 16 x 9 samples at the cell centres (washed out vs empty vs fine)
        let (w, h) = (tex.width(), tex.height());
        let Ok(g) = buf(GRID_W * GRID_H * 256, gpu_buffer_usage::MAP_READ | gpu_buffer_usage::COPY_DST) else { continue };
        let mut ok = true;
        for j in 0..GRID_H {
            for i in 0..GRID_W {
                let src = GpuTexelCopyTextureInfo::new(tex);
                src.set_origin(&crate::renderer::nums(&[((2 * i + 1) * w / (2 * GRID_W)).min(w - 1), ((2 * j + 1) * h / (2 * GRID_H)).min(h - 1)]));
                let dst = GpuTexelCopyBufferInfo::new(&g);
                dst.set_bytes_per_row(256);
                dst.set_offset((j * GRID_W + i) * 256);
                ok &= enc.copy_texture_to_buffer_with_u32_sequence(&src, &dst, &crate::renderer::nums(&[1, 1])).is_ok();
            }
        }
        if ok {
            p.grids.push((grid_label(label), g, tex.format()));
        }
    }
    p
}

const GRID_W: u32 = 16;
const GRID_H: u32 = 9;

fn grid_label(l: &str) -> &'static str {
    match l {
        "hdr_centre" => "hdr_grid",
        "canvas_centre" => "canvas_grid",
        _ => "grid",
    }
}

/// One sample as `rrggbb` (HDR: linear value x 255 clamped, so 0.5 = `7f`; `+` appended when a channel is > 1).
fn sample_hex(bytes: &[u8], fmt: &GpuTextureFormat) -> (String, f32) {
    let (r, g, b) = match fmt {
        GpuTextureFormat::Rgba16float => {
            let h = |i: usize| f16(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]));
            (h(0), h(1), h(2))
        }
        GpuTextureFormat::Bgra8unorm => (bytes[2] as f32 / 255.0, bytes[1] as f32 / 255.0, bytes[0] as f32 / 255.0),
        _ => (bytes[0] as f32 / 255.0, bytes[1] as f32 / 255.0, bytes[2] as f32 / 255.0),
    };
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let over = if r.max(g).max(b) > 1.0 { "+" } else { "" };
    (format!("{:02x}{:02x}{:02x}{over}", q(r), q(g), q(b)), 0.2126 * r + 0.7152 * g + 0.0722 * b)
}

fn f16(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    s * match e {
        0 => m * 2f32.powi(-24),
        31 => {
            if m == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

async fn map(b: &GpuBuffer) -> Result<Vec<u8>, String> {
    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(JsValue::from(b.map_async(gpu_map_mode::READ))))
        .await
        .map_err(|e| format!("{e:?}"))?;
    let range = b.get_mapped_range().map_err(|e| format!("{e:?}"))?;
    let v = js_sys::Uint8Array::new(&range).to_vec();
    b.unmap();
    Ok(v)
}

/// Field containing word `w` (for the report).
fn field_at(fields: &[Field], w: u32) -> String {
    let byte = w * 4;
    match fields.iter().rev().find(|f| f.offset <= byte) {
        Some(f) => format!("{}+{}", f.name, (byte - f.offset) / 4),
        None => format!("word {w}"),
    }
}

/// After the submit: reads everything back and stores the JSON report in `out` (also logged).
pub(crate) fn finish(p: Pending, out: Rc<RefCell<Option<serde_json::Value>>>) {
    wasm_bindgen_futures::spawn_local(async move {
        let mut echoes = serde_json::Map::new();
        let mut summary = Vec::new();
        for r in &p.runs {
            let bytes = match map(&r.read).await {
                Ok(b) => b,
                Err(e) => {
                    echoes.insert(r.label.into(), serde_json::json!({ "error": e }));
                    continue;
                }
            };
            let w = |i: u32| u32::from_le_bytes(bytes[(i * 4) as usize..(i * 4 + 4) as usize].try_into().unwrap());
            let n = r.words;
            let (mut bad_dyn, mut bad_const, mut bad_raw) = (0, 0, 0);
            let mut first = Vec::new();
            for i in 0..n {
                let (a, b, c, buf) = (w(i), w(n + i), w(2 * n + i), w(3 * n + i));
                let (dd, dc, dr) = (a != buf, b != buf, c != buf);
                bad_dyn += dd as u32;
                bad_const += dc as u32;
                bad_raw += dr as u32;
                if (dd || dc || dr) && first.len() < 16 {
                    first.push(serde_json::json!({
                        "field": field_at(&r.fields, i), "word": i,
                        "buffer": f32::from_bits(buf), "runtime_index": f32::from_bits(a), "const_index": f32::from_bits(b), "raw": f32::from_bits(c),
                        "buffer_bits": format!("{buf:#010x}"), "runtime_bits": format!("{a:#010x}"),
                    }));
                }
            }
            let ok = bad_dyn + bad_const + bad_raw == 0;
            summary.push(format!("{} {}", r.label, if ok { "ok".to_string() } else { format!("WRONG dyn {bad_dyn} const {bad_const} raw {bad_raw} of {n} words") }));
            echoes.insert(
                r.label.into(),
                serde_json::json!({ "ok": ok, "words": n, "wrong_runtime_index": bad_dyn, "wrong_const_index": bad_const, "wrong_raw": bad_raw, "first": first }),
            );
            r.out.destroy();
            r.read.destroy();
        }
        let mut pixels = serde_json::Map::new();
        for (label, b, fmt) in &p.pixels {
            let v = match map(b).await {
                Ok(bytes) => match fmt {
                    GpuTextureFormat::Rgba16float => {
                        let h = |i: usize| f16(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]));
                        serde_json::json!([h(0), h(1), h(2), h(3)])
                    }
                    GpuTextureFormat::Bgra8unorm => serde_json::json!({ "rgba": [bytes[2], bytes[1], bytes[0], bytes[3]] }),
                    _ => serde_json::json!({ "bytes": [bytes[0], bytes[1], bytes[2], bytes[3]] }),
                },
                Err(e) => serde_json::json!({ "error": e }),
            };
            summary.push(format!("{label} {v}"));
            pixels.insert((*label).into(), v);
            b.destroy();
        }
        for (label, b, fmt) in &p.grids {
            let v = match map(b).await {
                Ok(bytes) => {
                    let (mut rows, mut lum) = (Vec::new(), Vec::new());
                    for j in 0..GRID_H {
                        let row: Vec<String> = (0..GRID_W)
                            .map(|i| {
                                let (hex, l) = sample_hex(&bytes[((j * GRID_W + i) * 256) as usize..], fmt);
                                lum.push(l);
                                hex
                            })
                            .collect();
                        rows.push(row.join(" "));
                    }
                    let mean = lum.iter().sum::<f32>() / lum.len() as f32;
                    let (min, max) = lum.iter().fold((f32::MAX, f32::MIN), |(a, b), &l| (a.min(l), b.max(l)));
                    summary.push(format!("{label} luminance min {min:.3} mean {mean:.3} max {max:.3}"));
                    serde_json::json!({ "rows": rows, "luminance": { "min": min, "mean": mean, "max": max } })
                }
                Err(e) => serde_json::json!({ "error": e }),
            };
            pixels.insert((*label).into(), v);
            b.destroy();
        }
        let mut res = p.info.clone();
        res["echo"] = serde_json::Value::Object(echoes);
        res["pixels"] = serde_json::Value::Object(pixels);
        res["errors"] = serde_json::json!(p.errors);
        crate::log_info!("frame probe: {} {}", summary.join(", "), if p.errors.is_empty() { String::new() } else { format!("errors {:?}", p.errors) });
        crate::log_info!("frame probe result: {res}");
        *out.borrow_mut() = Some(res);
    });
}

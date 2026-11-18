//! Debug mode — toggled with the ` key (the key left of 1: `Backquote`, on German layouts `^`;
//! also Shift+´ which types a backtick). While it is on, the engine:
//!
//! * logs every key and mouse-button press/release (real DOM events *and* AI-injected input),
//! * logs the camera pose once per second (wall clock),
//! * shows a small DEBUG badge with the camera position and the log file,
//! * appends everything as JSON lines to `logs/debug-YYYYMMDD-HHMMSS.log` in the project root
//!   (POSTed in batches to the dev server's `/__debuglog`; one file per activation).
//!
//! Line format (one JSON object per line):
//! `{"t":12345.6,"step":800,"frame":812,"kind":"key","code":"KeyW","key":"w","down":true,"source":"dom"}`
//! `{"t":13000.1,"step":839,"frame":851,"kind":"camera","pos":[x,y,z],"forward":[x,y,z],"yaw_deg":..,"pitch_deg":..}`
//! plus `debug_mode` on/off markers. AI: `engine.call("debug_mode", {on?})` toggles and reports.

use crate::log_warn;
use glam::{Mat4, Vec3};
use serde_json::json;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::JsCast;

/// Is this keyboard event the debug-mode toggle?
pub fn is_toggle(code: &str, key: &str, shift: bool) -> bool {
    code == "Backquote" || key == "`" || (code == "Equal" && shift && key == "Dead")
}

#[derive(Default)]
struct Shared {
    sent_lines: u64,
    last_error: Option<String>,
    in_flight: bool,
}

#[derive(Default)]
pub struct DebugMode {
    /// Last Ctrl+click inspection result shown in the badge.
    pub(crate) toast: Option<String>,
    on: bool,
    file: Option<String>,
    pending: Vec<String>,
    last_cam_ms: f64,
    last_perf_ms: f64,
    /// frame intervals (ms) since the last perf line
    frame_dts: Vec<f32>,
    last_flush_ms: f64,
    lines: u64,
    keys: u64,
    last_cam: Option<[f64; 3]>,
    shared: Rc<RefCell<Shared>>,
}

impl DebugMode {
    pub fn on(&self) -> bool {
        self.on
    }

    pub fn file(&self) -> Option<&str> {
        self.file.as_deref()
    }

    pub fn set(&mut self, on: bool, now: f64, step: u64, frame: u64) {
        if on == self.on {
            return;
        }
        if on {
            let d = js_sys::Date::new_0();
            self.file = Some(format!(
                "debug-{:04}{:02}{:02}-{:02}{:02}{:02}.log",
                d.get_full_year(),
                d.get_month() + 1,
                d.get_date(),
                d.get_hours(),
                d.get_minutes(),
                d.get_seconds()
            ));
            self.on = true;
            self.last_cam_ms = f64::NEG_INFINITY; // log the camera immediately
            self.push(json!({"t": (now * 10.0).round() / 10.0, "step": step, "frame": frame, "kind": "debug_mode", "on": true,
                "date": String::from(d.to_iso_string()), "user_agent": web_sys::window().and_then(|w| w.navigator().user_agent().ok())}));
            crate::log_info!(
                "debug mode ON -> logs/{}",
                self.file.as_deref().unwrap_or("?")
            );
        } else {
            self.push(
                json!({"t": (now * 10.0).round() / 10.0, "step": step, "frame": frame, "kind": "debug_mode", "on": false}),
            );
            self.flush();
            self.on = false;
            crate::log_info!(
                "debug mode OFF ({} lines in logs/{})",
                self.lines,
                self.file.as_deref().unwrap_or("?")
            );
        }
    }

    pub fn toggle(&mut self, now: f64, step: u64, frame: u64) {
        let on = !self.on;
        self.set(on, now, step, frame);
    }

    fn push(&mut self, v: serde_json::Value) {
        self.pending.push(v.to_string());
        self.lines += 1;
    }

    /// Any other event line (e.g. `{"kind":"pick", ...}` from Ctrl+click inspection).
    pub fn record_event(&mut self, now: f64, step: u64, frame: u64, mut v: serde_json::Value) {
        if !self.on {
            return;
        }
        v["t"] = json!((now * 10.0).round() / 10.0);
        v["step"] = json!(step);
        v["frame"] = json!(frame);
        self.push(v);
    }

    /// Key or mouse-button edge. `code` is `KeyboardEvent.code` or `Mouse0..2`.
    #[allow(clippy::too_many_arguments)]
    pub fn record_input(
        &mut self,
        now: f64,
        step: u64,
        frame: u64,
        code: &str,
        key: &str,
        down: bool,
        source: &str,
    ) {
        if !self.on {
            return;
        }
        self.keys += 1;
        self.push(json!({"t": (now * 10.0).round() / 10.0, "step": step, "frame": frame, "kind": "key", "code": code, "key": key, "down": down, "source": source}));
    }

    /// Per frame: camera line every second, flush every 500 ms (or when 64 lines are queued).
    pub(crate) fn tick(&mut self, now: f64, step: u64, frame: u64, camera: Option<Mat4>) {
        if !self.on {
            return;
        }
        if now - self.last_cam_ms >= 1000.0 {
            self.last_cam_ms = if self.last_cam_ms.is_finite() {
                self.last_cam_ms + 1000.0
            } else {
                now
            };
            if now - self.last_cam_ms > 1000.0 {
                self.last_cam_ms = now; // tab was throttled: don't burst
            }
            if let Some(m) = camera {
                let pos = m.w_axis.truncate();
                let fwd = (-m.z_axis.truncate()).normalize_or(Vec3::NEG_Z);
                let yaw = fwd.x.atan2(-fwd.z).to_degrees();
                let pitch = fwd.y.clamp(-1.0, 1.0).asin().to_degrees();
                let r = |v: f32| ((v as f64) * 1000.0).round() / 1000.0 + 0.0; // f64: no f32 noise; +0.0 drops -0.0
                self.last_cam = Some([r(pos.x), r(pos.y), r(pos.z)]);
                self.push(json!({"t": (now * 10.0).round() / 10.0, "step": step, "frame": frame, "kind": "camera",
                    "pos": [r(pos.x), r(pos.y), r(pos.z)], "forward": [r(fwd.x), r(fwd.y), r(fwd.z)],
                    "yaw_deg": r(yaw), "pitch_deg": r(pitch)}));
            }
        }
        if !self.pending.is_empty()
            && (now - self.last_flush_ms >= 500.0 || self.pending.len() >= 64)
        {
            self.last_flush_ms = now;
            self.flush();
        }
    }

    /// POSTs queued lines to the dev server (skipped while a request is in flight).
    fn flush(&mut self) {
        let Some(file) = self.file.clone() else {
            return;
        };
        if self.pending.is_empty() || self.shared.borrow().in_flight {
            return;
        }
        let lines = std::mem::take(&mut self.pending);
        let n = lines.len() as u64;
        let body = lines.join("\n") + "\n";
        let Some(win) = web_sys::window() else { return };
        let init = web_sys::RequestInit::new();
        init.set_method("POST");
        init.set_body(&wasm_bindgen::JsValue::from_str(&body));
        let _ = js_sys::Reflect::set(&init, &"keepalive".into(), &true.into()); // survives page unload
        let promise = win.fetch_with_str_and_init(&format!("/__debuglog?file={file}"), &init);
        let shared = self.shared.clone();
        shared.borrow_mut().in_flight = true;
        wasm_bindgen_futures::spawn_local(async move {
            let res = wasm_bindgen_futures::JsFuture::from(promise).await;
            let mut s = shared.borrow_mut();
            s.in_flight = false;
            match res.map(|r| r.unchecked_into::<web_sys::Response>()) {
                Ok(r) if r.ok() => {
                    s.sent_lines += n;
                    s.last_error = None;
                }
                Ok(r) => {
                    let msg = format!(
                        "dev server answered {} (is `node tools/dev.mjs serve/watch` running?)",
                        r.status()
                    );
                    if s.last_error.as_deref() != Some(msg.as_str()) {
                        log_warn!("debug log: {msg}");
                    }
                    s.last_error = Some(msg);
                }
                Err(e) => {
                    let msg = format!("POST failed: {e:?}");
                    if s.last_error.is_none() {
                        log_warn!("debug log: {msg}");
                    }
                    s.last_error = Some(msg);
                }
            }
        });
    }

    /// Performance line once per second while on: fps, frame-time percentiles, CPU phases, GPU passes.
    pub(crate) fn record_perf(&mut self, now: f64, step: u64, frame: u64, frame_dt_ms: f32, snapshot: impl FnOnce() -> serde_json::Value) {
        if !self.on {
            self.frame_dts.clear();
            return;
        }
        self.frame_dts.push(frame_dt_ms);
        if now - self.last_perf_ms < 1000.0 {
            return;
        }
        self.last_perf_ms = now;
        let mut d = std::mem::take(&mut self.frame_dts);
        d.sort_by(|a, b| a.total_cmp(b));
        let pct = |p: f32| d.get(((d.len() as f32 - 1.0) * p).round().max(0.0) as usize).copied().unwrap_or(0.0);
        let r = |v: f32| (v * 100.0).round() / 100.0;
        let mut v = snapshot();
        v["t"] = json!((now * 10.0).round() / 10.0);
        v["step"] = json!(step);
        v["frame"] = json!(frame);
        v["kind"] = json!("perf");
        v["frames"] = json!(d.len());
        v["frame_ms"] = json!({"p50": r(pct(0.5)), "p95": r(pct(0.95)), "max": r(pct(1.0))});
        v["slow_frames_over_20ms"] = json!(d.iter().filter(|x| **x > 20.0).count());
        self.push(v);
    }

    /// Small HUD badge (empty when off).
    /// `perf`: live frame-rate text shown in the badge (fps, frame/GPU ms, resolution scale).
    pub(crate) fn badge_html(&self, perf: &str) -> String {
        if !self.on {
            return String::new();
        }
        let cam = self
            .last_cam
            .map(|c| format!("{:.1} {:.1} {:.1}", c[0], c[1], c[2]))
            .unwrap_or_default();
        format!(
            "<div class='panel debugmode'><b style='color:#f87171'>● DEBUG</b> {perf}<br>cam {cam} · {} inputs · logs/{}<br><small>Ctrl+click an object: copy its info</small>{}</div>",
            self.keys,
            self.file.as_deref().unwrap_or("?"),
            self.toast.as_ref().map(|t| format!("<br><b style='color:#93c5fd'>{}</b>", t.replace('<', "&lt;"))).unwrap_or_default()
        )
    }

    /// AI snapshot.
    pub fn describe(&self) -> serde_json::Value {
        let s = self.shared.borrow();
        json!({
            "on": self.on,
            "file": self.file.as_ref().map(|f| format!("logs/{f}")),
            "lines": self.lines,
            "sent_lines": s.sent_lines,
            "pending": self.pending.len(),
            "inputs_logged": self.keys,
            "last_camera": self.last_cam,
            "last_error": s.last_error,
            "toggle_key": "` (code Backquote; German layout: ^ key or Shift+´)",
        })
    }
}

//! Frame pacing: one place decides how often a frame is drawn and how much GPU work may queue.
//!
//! * **Cadence** (`auto_interval`): frames are drawn on every Nth display refresh (N = 1 at the full
//!   rate). The controller starts at the fastest rate the cap (`max_fps`, an optional upper limit)
//!   allows, steps to a slower cadence as soon as frames miss their slot and back to a faster one only
//!   with evidence of headroom. With GPU timestamps the evidence is the measured GPU frame time; without
//!   them (PowerVR, ...) it is the load: the drawn triangles must be clearly below the load the faster
//!   cadence failed at (remembered per cadence for the session), and a failed probe doubles the wait.
//! * **Back-pressure**: without timestamps, a frame is not submitted while the oldest unfinished one is
//!   more than `max_frames_in_flight - 1` frame slots behind its expected completion, so the GPU queue
//!   (and the compositor's, which drops late frames: "teleporting") cannot grow.
//!
//! Completion is observed through `onSubmittedWorkDone`, which Chrome resolves 1-2 refreshes after the GPU
//! finished, with some jitter. The *baseline* (lowest latency of the last 10 s: GPU time + resolve delay)
//! and the *jitter* (75th percentile above it in healthy windows) are learned, so only the excess over
//! them counts as queueing.

use serde::Serialize;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

#[derive(Default)]
pub(crate) struct Shared {
    /// Submit times of frames not resolved yet (in order).
    pending: VecDeque<f64>,
    /// (submit, resolve) of frames resolved since the last drain.
    resolved: Vec<(f64, f64)>,
}

impl Shared {
    pub(crate) fn resolve(&mut self, submit: f64, now: f64) {
        if let Some(p) = self.pending.iter().position(|&t| t == submit) {
            self.pending.remove(p);
        }
        self.resolved.push((submit, now));
    }
}

/// `stats.render.pacing`.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PacingStats {
    /// "timestamps" (GPU frame time measured), "latency" (submit -> done only) or "off" (`auto_frame_cap` false).
    pub mode: &'static str,
    /// Refreshes per drawn frame in use: max(cap_interval, auto_interval).
    pub interval: u32,
    /// Cadence chosen by the controller.
    pub auto_interval: u32,
    /// Cadence `max_fps` allows at most (1 = no cap).
    pub cap_interval: u32,
    /// Display refresh interval (ms).
    pub vsync_ms: f32,
    /// Sustained GPU frame time (ms): timestamps, or completion-to-completion while GPU-bound; 0 = not
    /// known (frames fit their slot).
    pub gpu_frame_ms: f32,
    /// Lowest submit -> done latency of the last 10 s (GPU time + Chrome's resolve delay).
    pub baseline_ms: f32,
    /// Learned resolve jitter above the baseline in healthy windows (75th percentile).
    pub jitter_ms: f32,
    /// How far the oldest unfinished frame is behind its expected completion (ms, 0 = no backlog).
    pub queue_ms: f32,
    /// Frames submitted and not resolved yet.
    pub queued: u32,
    /// Frames that finished more than 3/4 of a slot late (total).
    pub late: u64,
    /// Frames not submitted because of back-pressure (total).
    pub skipped: u64,
    /// Missed (late + skipped) share of the last window, %.
    pub window_miss_pct: f32,
    pub steps_down: u32,
    pub steps_up: u32,
    /// Healthy time needed before the next step to a faster cadence (s).
    pub wait_s: f32,
    /// Per cadence N (index N): average drawn triangles when cadence N failed (0 = never failed).
    pub failed_triangles: Vec<u64>,
}

/// What the renderer passes in once per frame.
pub(crate) struct Inputs {
    pub enabled: bool,
    pub timestamps: Option<f32>,
    /// Load signature for the failure memory (drawn triangles; `gpu_slow_ms` counts as 100k per ms).
    pub triangles: u64,
    pub scale: f32,
    pub min_scale: f32,
    pub auto_resolution: bool,
    pub min_fps: f32,
}

const MAX_DIV: u32 = 6;

pub(crate) struct FramePacing {
    pub(crate) shared: Rc<RefCell<Shared>>,
    lat: VecDeque<(f64, f32)>,
    last_latency: f32,
    win_start: f64,
    win_frames: u32,
    win_late: u32,
    win_skip: u32,
    win_excess: Vec<f32>,
    /// Raw submit -> done latencies of the window.
    win_lat: Vec<f32>,
    win_resolves: Vec<f64>,
    win_tris: f64,
    win_gpu: f32,
    win_gpu_n: u32,
    ok_since: Option<f64>,
    last_change: f64,
    wait_ms: f64,
    failed_tris: [f64; MAX_DIV as usize + 1],
    /// (cadence or scale probe, value, time): a step towards more work; failing within 4 s doubles the wait.
    probe: Option<(bool, f32, f64)>,
    scale_ceiling: f32,
    /// Average drawn triangles when the latency path last lowered the scale.
    scale_fail_tris: f64,
    started: f64,
    pub(crate) divisor: u32,
    pub(crate) cap: u32,
    pub(crate) vsync_ms: f32,
    jitter: f32,
    pub(crate) stats: PacingStats,
}

impl Default for FramePacing {
    fn default() -> Self {
        Self {
            shared: Default::default(),
            lat: VecDeque::new(),
            last_latency: 0.0,
            win_start: 0.0,
            win_frames: 0,
            win_late: 0,
            win_skip: 0,
            win_excess: vec![],
            win_lat: vec![],
            win_resolves: vec![],
            win_tris: 0.0,
            win_gpu: 0.0,
            win_gpu_n: 0,
            ok_since: None,
            last_change: -1e9,
            wait_ms: 5000.0,
            failed_tris: [0.0; MAX_DIV as usize + 1],
            probe: None,
            scale_ceiling: 1.0,
            scale_fail_tris: 0.0,
            started: 0.0,
            divisor: 1,
            cap: 1,
            vsync_ms: 0.0,
            jitter: -1.0,
            stats: PacingStats::default(),
        }
    }
}

impl FramePacing {
    fn v(&self) -> f32 {
        if self.vsync_ms > 0.0 { self.vsync_ms } else { 16.7 }
    }

    /// Refreshes per drawn frame in use.
    pub(crate) fn interval(&self) -> u32 {
        self.divisor.max(self.cap).max(1)
    }

    fn slot(&self) -> f32 {
        self.interval() as f32 * self.v()
    }

    fn baseline(&self) -> f32 {
        self.lat.iter().map(|l| l.1).fold(f32::MAX, f32::min)
    }

    pub(crate) fn last_latency(&self) -> f32 {
        self.last_latency
    }

    pub(crate) fn queued(&self) -> u32 {
        self.shared.borrow().pending.len() as u32
    }

    /// Takes the completions that arrived since the last call.
    pub(crate) fn drain(&mut self, now: f64) {
        let done: Vec<(f64, f64)> = std::mem::take(&mut self.shared.borrow_mut().resolved);
        for (s, r) in done {
            let l = (r - s) as f32;
            self.last_latency = l;
            self.lat.push_back((r, l));
            let base = self.baseline();
            let excess = (l - base).max(0.0);
            self.win_excess.push(excess);
            self.win_lat.push(l);
            self.win_resolves.push(r);
            if self.lat.len() >= 20 && self.jitter >= 0.0 && excess > self.jitter + 0.75 * self.slot() {
                self.win_late += 1;
                self.stats.late += 1;
            }
        }
        while self.lat.front().is_some_and(|f| now - f.0 > 10000.0) && self.lat.len() > 20 {
            self.lat.pop_front();
        }
        if self.jitter < 0.0 && self.lat.len() >= 20 {
            self.jitter = 2.0 * self.slot(); // lenient until healthy windows taught the real value
        }
    }

    /// How far the oldest unfinished frame is behind its expected completion (ms).
    fn queue_ms(&self, now: f64) -> f32 {
        if self.lat.len() < 20 {
            return 0.0;
        }
        let oldest = self.shared.borrow().pending.front().copied();
        oldest.map_or(0.0, |t| ((now - t) as f32 - self.baseline() - self.jitter.max(0.0)).max(0.0))
    }

    /// Back-pressure: true = do not submit this frame (the oldest unfinished frame is more than
    /// `max_in_flight - 1` slots behind its expected completion: baseline + learned jitter).
    pub(crate) fn should_skip(&mut self, now: f64, max_in_flight: u32) -> bool {
        let queued = self.queued();
        if max_in_flight == 0 {
            return false;
        }
        let skip = queued >= 64 || self.queue_ms(now) > self.slot() * (max_in_flight.max(1) - 1) as f32 + 0.25 * self.v();
        if skip {
            self.win_skip += 1;
            self.stats.skipped += 1;
        }
        skip
    }

    /// A frame was submitted at `t`.
    pub(crate) fn submitted(&mut self, t: f64) {
        self.shared.borrow_mut().pending.push_back(t);
    }

    /// A promise that never resolves: forget the pending frames.
    pub(crate) fn forget_pending(&mut self) {
        self.shared.borrow_mut().pending.clear();
    }

    /// Once per drawn frame (after render). Returns a new render scale when the latency path decided one.
    pub(crate) fn frame(&mut self, now: f64, inp: Inputs) -> Option<f32> {
        self.win_frames += 1;
        self.win_tris += inp.triangles as f64;
        if let Some(g) = inp.timestamps {
            self.win_gpu += g;
            self.win_gpu_n += 1;
        }
        let v = self.v();
        self.stats.mode = if !inp.enabled { "off" } else if inp.timestamps.is_some() { "timestamps" } else { "latency" };
        self.stats.cap_interval = self.cap;
        self.stats.auto_interval = self.divisor;
        self.stats.interval = self.interval();
        self.stats.vsync_ms = self.vsync_ms;
        self.stats.baseline_ms = if self.lat.is_empty() { 0.0 } else { self.baseline() };
        self.stats.jitter_ms = self.jitter.max(0.0);
        self.stats.queue_ms = self.queue_ms(now);
        self.stats.queued = self.queued();
        self.stats.wait_s = (self.wait_ms / 1000.0) as f32;
        self.stats.failed_triangles = self.failed_tris.iter().map(|&t| t as u64).collect();
        if self.win_start == 0.0 {
            self.win_start = now;
            self.started = now;
        }
        if now - self.win_start < 500.0 || self.win_frames < 8 {
            return None;
        }
        // --- one window (~0.5 s) ---
        let n = self.win_frames;
        let miss = self.win_late + self.win_skip;
        let miss_pct = miss as f32 / (n + self.win_skip) as f32 * 100.0;
        let avg_tris = self.win_tris / n as f64;
        let ts = (self.win_gpu_n > 0).then(|| self.win_gpu / self.win_gpu_n as f32);
        let bound = miss >= 2 && self.win_resolves.len() >= 3;
        let gpu_est = match ts {
            Some(g) => g,
            None if bound => {
                let r = &self.win_resolves;
                ((r[r.len() - 1] - r[0]) / (r.len() - 1) as f64) as f32
            }
            None => 0.0,
        };
        let mut ex = std::mem::take(&mut self.win_excess);
        let mut wl = std::mem::take(&mut self.win_lat);
        wl.sort_by(f32::total_cmp);
        // median submit -> done latency: an upper bound of the GPU frame time while nothing queues
        let med_lat = wl.get(wl.len() / 2).copied().unwrap_or(f32::MAX);
        let slot = self.slot();
        let over = match ts {
            Some(g) => g > 0.92 * slot || self.win_skip >= 2,
            None => miss >= 2 && miss_pct > 8.0,
        };
        // healthy windows teach the resolve jitter (latency path)
        if !over && ex.len() >= 4 && self.jitter >= 0.0 {
            ex.sort_by(f32::total_cmp);
            let p75 = ex[ex.len() * 3 / 4];
            self.jitter = (self.jitter * 0.8 + p75 * 0.2).min(4.0 * slot);
        }
        self.stats.window_miss_pct = miss_pct;
        self.stats.gpu_frame_ms = gpu_est;
        self.win_start = now;
        self.win_frames = 0;
        self.win_late = 0;
        self.win_skip = 0;
        self.win_resolves.clear();
        self.win_tris = 0.0;
        self.win_gpu = 0.0;
        self.win_gpu_n = 0;
        if !inp.enabled {
            self.divisor = 1;
            return None;
        }
        // startup: shader compiles, uploads and the first scene load are not the sustained load
        if now - self.started < 3000.0 {
            return None;
        }
        let d = self.interval();
        // a change reallocates targets / shifts the cadence: its own hiccup is not a verdict
        if now - self.last_change < 1000.0 {
            return None;
        }
        if over {
            self.ok_since = None;
            if let Some((_, _, t)) = self.probe.take() {
                if now - t < 4000.0 {
                    self.wait_ms = (self.wait_ms * 2.0).min(120000.0);
                }
            }
            let need = if gpu_est > 0.0 { ((gpu_est * 1.1) / v).ceil() as u32 } else { d + 1 };
            let need = need.clamp(d + 1, MAX_DIV);
            let next_fps = 1000.0 / (need as f32 * v);
            // latency path: resolution is the second lever, used before the rate drops below min_fps
            if ts.is_none() && inp.auto_resolution && inp.scale > inp.min_scale + 1e-3 && next_fps < inp.min_fps {
                // GPU time scales with pixels: aim at 90 % of the slot in one step when the time is known
                let k = if gpu_est > 0.0 { (0.9 * slot / gpu_est).sqrt().min(0.85) } else { 0.85 };
                let next = ((inp.scale * k / 0.05).floor() * 0.05).min(inp.scale - 0.05).max(inp.min_scale);
                self.scale_ceiling = self.scale_ceiling.min(inp.scale);
                self.scale_fail_tris = avg_tris;
                self.last_change = now;
                crate::log_info!("frame pacing: scale {:.2} -> {next:.2} ({miss_pct:.0}% missed at {:.1} fps)", inp.scale, 1000.0 / slot);
                return Some(next);
            }
            if d >= MAX_DIV {
                return None;
            }
            // the measured GPU time rules out every cadence from d up to need - 1: remember them all, so
            // stepping back does not probe a rate that is known to fail at this load
            for k in d..need {
                let f = &mut self.failed_tris[k as usize];
                if *f <= 0.0 || avg_tris < *f {
                    *f = avg_tris.max(1.0);
                }
            }
            self.divisor = need;
            self.stats.steps_down += 1;
            self.last_change = now;
            crate::log_info!(
                "frame pacing: every {need} refreshes ({:.1} fps) - {miss_pct:.0}% missed at {:.1} fps, gpu {}, {:.0}k triangles",
                1000.0 / (need as f32 * v),
                1000.0 / slot,
                if gpu_est > 0.0 { format!("~{gpu_est:.1} ms") } else { "unknown".into() },
                avg_tris / 1000.0
            );
            return None;
        }
        let ok_since = *self.ok_since.get_or_insert(now);
        if let Some((_, _, t)) = self.probe {
            if now - t > 4000.0 {
                self.probe = None; // the step held
            }
        }
        if now - ok_since < self.wait_ms || now - self.last_change < self.wait_ms {
            return None;
        }
        if d > self.cap.max(1) {
            // with a measured GPU time: straight to the fastest cadence that fits; else one step
            // (the median latency is an upper bound of the GPU time without timestamps)
            let bound = ts.or((wl.len() >= 4).then_some(med_lat));
            let fast = bound.and_then(|g| (self.cap.max(1)..d).find(|&k| g < 0.85 * k as f32 * v)).unwrap_or(d - 1);
            let fast_slot = fast as f32 * v;
            let failed = self.failed_tris[fast as usize];
            let lat_fits = wl.len() >= 4 && med_lat < 0.85 * fast_slot;
            let evidence = match ts {
                Some(g) => g < 0.85 * fast_slot,
                None => lat_fits || failed <= 0.0 || avg_tris < failed * 0.7,
            };
            if !evidence {
                return None;
            }
            self.divisor = fast;
            self.stats.steps_up += 1;
            self.last_change = now;
            self.ok_since = Some(now);
            self.probe = Some((true, fast as f32, now));
            crate::log_info!(
                "frame pacing: back to every {fast} refreshes ({:.1} fps) - {}",
                1000.0 / fast_slot,
                match ts {
                    Some(g) => format!("gpu {g:.1} ms < 85 % of {fast_slot:.1} ms"),
                    None if lat_fits => format!("median latency {med_lat:.1} ms < 85 % of {fast_slot:.1} ms"),
                    None if failed > 0.0 => format!("{:.0}k triangles < 70 % of the {:.0}k it failed at", avg_tris / 1000.0, failed / 1000.0),
                    None => "never failed at that rate".into(),
                }
            );
            return None;
        }
        // full rate holds: the latency path raises the resolution again, below where it failed
        // the median latency bounds the GPU time from above: the largest scale it predicts to fit; without
        // that evidence one 0.05 step, not above the scale that failed unless the load dropped
        let fits = |k: f32| wl.len() >= 4 && med_lat * (k / inp.scale).powi(2) < 0.8 * slot;
        let lighter = self.scale_fail_tris > 0.0 && avg_tris < self.scale_fail_tris * 0.7;
        let first = (inp.scale + 0.05).min(1.0);
        if ts.is_none() && inp.auto_resolution && inp.scale < 1.0 && (fits(first) || lighter || first <= self.scale_ceiling + 1e-3) {
            let mut next = first;
            while next + 0.05 <= 1.0 + 1e-3 && fits(next + 0.05) {
                next += 0.05;
            }
            if fits(next) || lighter {
                self.scale_ceiling = self.scale_ceiling.max(next);
            }
            self.last_change = now;
            self.ok_since = Some(now);
            self.probe = Some((false, next, now));
            crate::log_info!("frame pacing: scale {:.2} -> {next:.2} (no misses for {:.0} s)", inp.scale, self.wait_ms / 1000.0);
            return Some(next);
        }
        None
    }
}

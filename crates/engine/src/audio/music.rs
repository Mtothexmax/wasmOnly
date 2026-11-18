//! Streaming music. Long tracks are played through an `<audio>` element (`HtmlAudioElement`):
//! Chrome downloads them progressively with HTTP range requests and starts playing as soon as
//! enough is buffered — nothing is decoded up front, nothing blocks the game. The element is
//! wired into Web Audio (`MediaElementAudioSourceNode -> track gain -> music bus -> master`), so
//! tracks crossfade, have their own volume bus and obey the focus pause.
//!
//! ```ignore
//! ctx.audio.play_music("assets/music/theme.mp3", MusicOpts::default());          // fade in 1.5 s, loop
//! ctx.audio.prefetch_music("assets/music/boss.mp3");                              // start buffering early
//! ctx.audio.play_music("assets/music/boss.mp3", MusicOpts { fade: 3.0, ..Default::default() }); // crossfade
//! ctx.audio.set_music_volume(0.5);
//! ```
//! AI: `engine.call("music")` shows every track with state (`buffering`/`playing`/`paused`/
//! `blocked`/`error`), position, duration and how much is buffered; transitions are logged.

use crate::{log_info, log_warn};
use serde::Serialize;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use web_sys::{AudioContext, GainNode, HtmlAudioElement, MediaElementAudioSourceNode};

#[derive(Clone, Copy, Debug, Serialize)]
pub struct MusicOpts {
    /// Fade-in seconds (and fade-out of the previous track = crossfade).
    pub fade: f32,
    pub volume: f32,
    pub looped: bool,
}

impl Default for MusicOpts {
    fn default() -> Self {
        Self { fade: 1.5, volume: 1.0, looped: true }
    }
}

struct Track {
    id: u32,
    url: String,
    el: HtmlAudioElement,
    node: MediaElementAudioSourceNode,
    gain: GainNode,
    volume: f32,
    /// Should be playing (false after `stop`/explicit pause).
    wants_play: bool,
    /// `play()` was rejected (autoplay policy) — retried on the next user gesture.
    blocked: Rc<Cell<bool>>,
    play_error: Rc<RefCell<Option<String>>>,
    fade_out_end: Option<f64>,
    last_state: &'static str,
}

impl Track {
    fn state(&self) -> &'static str {
        if self.el.error().is_some() {
            "error"
        } else if self.blocked.get() {
            "blocked"
        } else if !self.wants_play {
            "stopped"
        } else if self.el.paused() {
            "paused"
        } else if self.el.ready_state() < 3 {
            "buffering" // HAVE_FUTURE_DATA not reached: waiting for the network
        } else {
            "playing"
        }
    }

    fn buffered_end(&self) -> f64 {
        let b = self.el.buffered();
        let mut end = 0.0f64;
        for i in 0..b.length() {
            if let (Ok(s), Ok(e)) = (b.start(i), b.end(i)) {
                if s <= self.el.current_time() + 0.5 {
                    end = end.max(e);
                }
            }
        }
        end
    }

    fn play(&self) {
        let Ok(p) = self.el.play() else { return };
        let blocked = self.blocked.clone();
        let err = self.play_error.clone();
        let url = self.url.clone();
        wasm_bindgen_futures::spawn_local(async move {
            match wasm_bindgen_futures::JsFuture::from(p).await {
                Ok(_) => blocked.set(false),
                Err(e) => {
                    let msg = js_sys::Reflect::get(&e, &"name".into()).ok().and_then(|v| v.as_string()).unwrap_or_default();
                    if msg == "NotAllowedError" {
                        blocked.set(true); // autoplay policy: retry on the next gesture
                    } else if msg != "AbortError" {
                        log_warn!("music {url}: play() failed: {e:?}");
                        *err.borrow_mut() = Some(format!("{e:?}"));
                    }
                }
            }
        });
    }
}

#[derive(Default)]
pub struct Music {
    tracks: Vec<Track>,
    prefetched: HashMap<String, HtmlAudioElement>,
    next_id: u32,
    volume: f32,
}

fn new_element(url: &str) -> Option<HtmlAudioElement> {
    let el = HtmlAudioElement::new_with_src(url).ok()?;
    el.set_preload("auto");
    Some(el)
}

impl Music {
    pub(crate) fn new() -> Self {
        Self { volume: 1.0, ..Default::default() }
    }

    /// Starts buffering `url` without playing it (e.g. the next area's music).
    pub(crate) fn prefetch(&mut self, url: &str) {
        if self.prefetched.contains_key(url) || self.tracks.iter().any(|t| t.url == url && t.fade_out_end.is_none()) {
            return;
        }
        if let Some(el) = new_element(url) {
            el.load();
            self.prefetched.insert(url.to_string(), el);
        }
    }

    pub(crate) fn current_url(&self) -> Option<&str> {
        self.tracks.iter().rev().find(|t| t.fade_out_end.is_none()).map(|t| t.url.as_str())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn play(&mut self, ctx: &AudioContext, bus: &GainNode, url: &str, o: MusicOpts, may_start: bool) -> Option<u32> {
        let now = ctx.current_time();
        // same track already active: just adjust volume / loop
        if let Some(t) = self.tracks.iter_mut().rev().find(|t| t.fade_out_end.is_none() && t.url == url) {
            t.volume = o.volume;
            t.el.set_loop(o.looped);
            ramp(&t.gain, now, o.volume, o.fade);
            if !t.wants_play {
                t.wants_play = true;
                if may_start {
                    t.play();
                }
            }
            return Some(t.id);
        }
        self.stop(now, o.fade);
        let el = self.prefetched.remove(url).or_else(|| new_element(url))?;
        el.set_loop(o.looped);
        let node = ctx.create_media_element_source(&el).ok()?;
        let gain = ctx.create_gain().ok()?;
        gain.gain().set_value(0.0);
        node.connect_with_audio_node(&gain).ok()?;
        gain.connect_with_audio_node(bus).ok()?;
        ramp(&gain, now, o.volume, o.fade);
        let id = self.next_id;
        self.next_id += 1;
        let t = Track {
            id,
            url: url.to_string(),
            el,
            node,
            gain,
            volume: o.volume,
            wants_play: true,
            blocked: Rc::new(Cell::new(false)),
            play_error: Rc::new(RefCell::new(None)),
            fade_out_end: None,
            last_state: "",
        };
        if may_start {
            t.play();
        }
        log_info!("music: {url} requested (fade {:.1}s, streaming)", o.fade);
        self.tracks.push(t);
        Some(id)
    }

    /// Fades out every active track over `fade` seconds.
    pub(crate) fn stop(&mut self, now: f64, fade: f32) {
        for t in self.tracks.iter_mut().filter(|t| t.fade_out_end.is_none()) {
            ramp(&t.gain, now, 0.0, fade);
            t.fade_out_end = Some(now + fade.max(0.0) as f64);
        }
    }

    /// Focus / explicit pause handling: pause or resume elements that want to play.
    pub(crate) fn set_running(&mut self, running: bool) {
        for t in &self.tracks {
            if !t.wants_play {
                continue;
            }
            if running {
                if t.el.paused() {
                    t.play();
                }
            } else {
                let _ = t.el.pause();
            }
        }
    }

    /// Retry tracks blocked by the autoplay policy (call on a user gesture).
    pub(crate) fn retry_blocked(&mut self) {
        for t in self.tracks.iter().filter(|t| t.blocked.get() && t.wants_play) {
            t.play();
        }
    }

    pub(crate) fn update(&mut self, now: f64) {
        self.tracks.retain(|t| {
            let done = t.fade_out_end.is_some_and(|end| now >= end);
            if done {
                let _ = t.el.pause();
                t.el.remove_attribute("src").ok();
                t.el.load(); // abort the download
                t.node.disconnect().ok();
                t.gain.disconnect().ok();
                log_info!("music: {} stopped", t.url);
            }
            !done
        });
        for t in &mut self.tracks {
            let s = t.state();
            if s != t.last_state {
                if t.last_state.is_empty() || s != "paused" {
                    log_info!("music: {} {} -> {} (t={:.1}s, buffered {:.1}s)", t.url, if t.last_state.is_empty() { "new" } else { t.last_state }, s, t.el.current_time(), t.buffered_end());
                }
                t.last_state = s;
            }
        }
    }

    pub(crate) fn set_volume(&mut self, v: f32) {
        self.volume = v;
    }

    pub(crate) fn describe(&self) -> serde_json::Value {
        let r = |x: f64| if x.is_finite() { (x * 100.0).round() / 100.0 } else { -1.0 };
        serde_json::json!({
            "volume": self.volume,
            "current": self.current_url(),
            "tracks": self.tracks.iter().map(|t| {
                let dur = t.el.duration();
                let buf = t.buffered_end();
                serde_json::json!({
                    "id": t.id, "url": t.url, "state": t.state(),
                    "time": r(t.el.current_time()), "duration": if dur.is_finite() { serde_json::json!(r(dur)) } else { serde_json::Value::Null },
                    "buffered_s": r(buf), "buffered_pct": if dur.is_finite() && dur > 0.0 { serde_json::json!((buf / dur * 1000.0).round() / 10.0) } else { serde_json::Value::Null },
                    "ready_state": t.el.ready_state(), "network_state": t.el.network_state(),
                    "volume": t.volume, "loop": t.el.loop_(), "fading_out": t.fade_out_end.is_some(),
                    "error": t.el.error().map(|e| format!("MediaError code {}", e.code())).or_else(|| t.play_error.borrow().clone()),
                })
            }).collect::<Vec<_>>(),
            "prefetched": self.prefetched.iter().map(|(u, el)| serde_json::json!({"url": u, "ready_state": el.ready_state(),
                "duration": if el.duration().is_finite() { serde_json::json!(r(el.duration())) } else { serde_json::Value::Null }})).collect::<Vec<_>>(),
        })
    }
}

fn ramp(g: &GainNode, now: f64, to: f32, secs: f32) {
    let p = g.gain();
    let _ = p.cancel_scheduled_values(now);
    let _ = p.set_value_at_time(p.value(), now);
    if secs <= 0.0 {
        let _ = p.set_value_at_time(to, now);
    } else {
        let _ = p.linear_ramp_to_value_at_time(to, now + secs as f64);
    }
}

//! Audio: Web Audio API (the browser is the platform). One `AudioContext`, a master gain with two
//! buses (sfx, music), per-voice `AudioBufferSourceNode -> GainNode -> [PannerNode (HRTF)] -> sfx`
//! and streamed music tracks (`<audio>` element -> gain -> music bus), see [`music`].
//!
//! * Sounds come from files (`ctx.load_sound("assets/x.ogg")`, any format Chrome decodes) or from
//!   samples (`ctx.audio.add_samples(name, &samples, rate)`; see [`synth`] for procedural
//!   effects — no asset files needed for prototypes).
//! * 3D: `Play::new().at(pos)` or `.follow(entity)`; the listener follows the camera each frame.
//! * Autoplay policy: the context starts suspended and resumes on the first key/pointer event.
//!   The debug Chrome (`bridge.mjs launch`) runs with `--autoplay-policy=no-user-gesture-required`.
//! * AI: `engine.call("audio")` shows context state, sounds, active voices and the last plays
//!   (with simulation step), so "did the shot make a sound?" is an assertion, not a guess.

use crate::scene::GlobalTransform;
use crate::{log_info, log_warn};
use glam::{Mat4, Vec3};
use hecs::{Entity, World};
use serde::Serialize;
use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    AudioBuffer, AudioBufferSourceNode, AudioContext, AudioNode, BiquadFilterType,
    DistanceModelType, GainNode, OverSampleType, PannerNode, PanningModelType,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct SoundId(pub u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct VoiceId(pub u32);

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SoundState {
    Loading,
    Ready,
    Failed(String),
}

struct Sound {
    name: String,
    buffer: Option<AudioBuffer>,
    state: SoundState,
    plays: u32,
}

/// Playback-time effects, built from Web Audio nodes per voice:
/// `source -> [WaveShaper] -> [high-pass] -> [low-pass] -> [band-pass] -> gain -> out`
/// plus parallel sends `gain -> Delay(+feedback) -> out` (echo) and `gain -> Convolver -> out`
/// (reverb), then `out -> [StereoPanner] -> [Panner 3D] -> master`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Fx {
    pub lowpass: Option<f32>,
    pub highpass: Option<f32>,
    /// (centre Hz, Q)
    pub bandpass: Option<(f32, f32)>,
    /// (delay seconds, feedback 0..0.95, wet mix 0..1)
    pub echo: Option<(f32, f32, f32)>,
    /// (decay seconds, wet mix 0..1)
    pub reverb: Option<(f32, f32)>,
    /// Waveshaper drive (0 = off, 1..50 typical).
    pub distortion: Option<f32>,
    /// Stereo pan -1..1 (2D sounds).
    pub pan: Option<f32>,
    /// Pitch offset in cents (100 = one semitone), independent of `pitch`.
    pub detune: f32,
}

impl Fx {
    fn is_empty(&self) -> bool {
        *self == Fx::default()
    }
    /// Seconds the effect tail rings after the source ends.
    fn tail(&self) -> f64 {
        let echo = self
            .echo
            .map(|(d, fb, _)| d as f64 * (0.001f64.ln() / (fb.clamp(0.01, 0.95) as f64).ln()))
            .unwrap_or(0.0);
        let rev = self.reverb.map(|(t, _)| t as f64).unwrap_or(0.0);
        echo.max(rev)
    }
}

/// Playback parameters (builder).
#[derive(Clone, Copy, Debug, Default)]
pub struct Play {
    pub fx: Fx,
    pub volume: f32,
    pub pitch: f32,
    pub looped: bool,
    pub at: Option<Vec3>,
    pub follow: Option<Entity>,
    /// Distance (world units) at which a 3D sound starts to attenuate.
    pub ref_distance: f32,
}

impl Play {
    pub fn new() -> Self {
        Self {
            volume: 1.0,
            pitch: 1.0,
            looped: false,
            at: None,
            follow: None,
            ref_distance: 3.0,
            fx: Fx::default(),
        }
    }
    pub fn volume(mut self, v: f32) -> Self {
        self.volume = v;
        self
    }
    /// Playback rate (1 = original; 2 = one octave up).
    pub fn pitch(mut self, p: f32) -> Self {
        self.pitch = p;
        self
    }
    pub fn looped(mut self) -> Self {
        self.looped = true;
        self
    }
    /// Positional (3D) sound at a fixed world position.
    pub fn at(mut self, p: Vec3) -> Self {
        self.at = Some(p);
        self
    }
    /// Positional sound that follows an entity's `GlobalTransform`.
    pub fn follow(mut self, e: Entity) -> Self {
        self.follow = Some(e);
        self
    }
    pub fn ref_distance(mut self, d: f32) -> Self {
        self.ref_distance = d;
        self
    }
    /// Low-pass filter (Hz): muffled / distant / underwater.
    pub fn lowpass(mut self, hz: f32) -> Self {
        self.fx.lowpass = Some(hz);
        self
    }
    /// High-pass filter (Hz): thin / radio / telephone.
    pub fn highpass(mut self, hz: f32) -> Self {
        self.fx.highpass = Some(hz);
        self
    }
    /// Band-pass (centre Hz, Q): walkie-talkie, resonant timbre.
    pub fn bandpass(mut self, hz: f32, q: f32) -> Self {
        self.fx.bandpass = Some((hz, q));
        self
    }
    /// Echo: delay seconds and feedback 0..0.95 (wet mix 0.5).
    pub fn echo(mut self, delay: f32, feedback: f32) -> Self {
        self.fx.echo = Some((delay, feedback, 0.5));
        self
    }
    /// Reverb with a generated impulse response of `seconds` (wet mix 0.4).
    pub fn reverb(mut self, seconds: f32) -> Self {
        self.fx.reverb = Some((seconds, 0.4));
        self
    }
    /// Waveshaper distortion (1..50).
    pub fn distort(mut self, drive: f32) -> Self {
        self.fx.distortion = Some(drive);
        self
    }
    /// Stereo pan -1 (left) .. 1 (right) for 2D sounds.
    pub fn pan(mut self, x: f32) -> Self {
        self.fx.pan = Some(x.clamp(-1.0, 1.0));
        self
    }
    /// Pitch offset in cents (100 = one semitone).
    pub fn detune(mut self, cents: f32) -> Self {
        self.fx.detune = cents;
        self
    }
}

struct Voice {
    id: VoiceId,
    sound: SoundId,
    src: AudioBufferSourceNode,
    panner: Option<PannerNode>,
    follow: Option<Entity>,
    pos: Option<Vec3>,
    ended: Rc<Cell<bool>>,
    /// Output bus of the voice graph; disconnected when the voice (and its effect tail) is done.
    out: GainNode,
    tail: f64,
    ended_at: Option<f64>,
    fx: Fx,
    _on_end: Closure<dyn FnMut()>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PlayEvent {
    pub sound: String,
    pub voice: u32,
    pub step: u64,
    pub frame: u64,
    pub pos: Option<[f32; 3]>,
    pub volume: f32,
    pub pitch: f32,
    #[serde(skip_serializing_if = "Fx::is_empty")]
    pub fx: Fx,
    /// Not played: one-shot triggered while the tab was unfocused (see `pause_on_blur`).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub suppressed: bool,
}

pub struct Audio {
    ctx: Option<AudioContext>,
    master: Option<GainNode>,
    /// Sound effects bus (all `play` voices) and music bus (streamed tracks), both -> master.
    sfx_bus: Option<GainNode>,
    music_bus: Option<GainNode>,
    sfx_volume: f32,
    music_volume: f32,
    music: music::Music,
    sounds: Vec<Sound>,
    by_name: HashMap<String, SoundId>,
    voices: Vec<Voice>,
    next_voice: u32,
    volume: f32,
    muted: bool,
    /// Tab/window has focus (window focus/blur + document visibility).
    focused: bool,
    /// Suspend the AudioContext while the tab is unfocused (default on); resume on focus.
    pause_on_blur: bool,
    pub listener: Vec3,
    /// Last 32 plays (newest last).
    pub history: VecDeque<PlayEvent>,
    pub plays_total: u64,
    pub(crate) step: u64,
    pub(crate) frame: u64,
    /// Reverb impulse responses by length in ms.
    impulses: HashMap<u32, AudioBuffer>,
}

impl Default for Audio {
    fn default() -> Self {
        let ctx = AudioContext::new()
            .map_err(|e| log_warn!("AudioContext unavailable: {e:?}"))
            .ok();
        let master = ctx.as_ref().and_then(|c| {
            let g = c.create_gain().ok()?;
            g.connect_with_audio_node(&c.destination()).ok()?;
            Some(g)
        });
        let bus = |c: &AudioContext| {
            let g = c.create_gain().ok()?;
            g.connect_with_audio_node(master.as_ref()?).ok()?;
            Some(g)
        };
        let sfx_bus = ctx.as_ref().and_then(bus);
        let music_bus = ctx.as_ref().and_then(bus);
        let a = Self {
            ctx,
            master,
            sfx_bus,
            music_bus,
            sfx_volume: 1.0,
            music_volume: 1.0,
            music: music::Music::new(),
            sounds: vec![],
            by_name: HashMap::new(),
            voices: vec![],
            next_voice: 0,
            volume: 1.0,
            muted: false,
            focused: web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.has_focus().ok())
                .unwrap_or(true),
            pause_on_blur: true,
            listener: Vec3::ZERO,
            history: VecDeque::new(),
            plays_total: 0,
            step: 0,
            frame: 0,
            impulses: HashMap::new(),
        };
        if !a.focused {
            a.suspend(); // page loaded in a background tab
        }
        a
    }
}

impl Audio {
    /// "running", "suspended" (waiting for a user gesture), "closed" or "unavailable".
    pub fn state(&self) -> String {
        match &self.ctx {
            Some(c) => format!("{:?}", c.state()).to_lowercase(),
            None => "unavailable".into(),
        }
    }

    /// Resumes a suspended context (called on user gestures; safe to call any time).
    /// Does nothing while the tab is unfocused and `pause_on_blur` is on.
    pub fn resume(&mut self) {
        if self.pause_on_blur && !self.focused {
            return;
        }
        if let Some(c) = &self.ctx {
            if c.state() == web_sys::AudioContextState::Suspended {
                let _ = c.resume();
            }
        }
        self.music.retry_blocked();
    }

    fn may_play(&self) -> bool {
        !(self.pause_on_blur && !self.focused)
    }

    // --- Music (streamed) ----------------------------------------------------------------------

    /// Streams `url` as music (crossfades from the current track). Returns immediately; the track
    /// starts as soon as Chrome has buffered enough. Same URL again = keep playing, adjust volume.
    pub fn play_music(&mut self, url: &str, opts: MusicOpts) -> Option<u32> {
        let ctx = self.ctx.clone()?;
        let bus = self.music_bus.clone()?;
        let may = self.may_play();
        self.music.play(&ctx, &bus, url, opts, may)
    }

    /// Fades the current music out.
    pub fn stop_music(&mut self, fade: f32) {
        let now = self.ctx.as_ref().map(|c| c.current_time()).unwrap_or(0.0);
        self.music.stop(now, fade);
    }

    /// Starts downloading `url` in the background so a later `play_music` starts instantly.
    pub fn prefetch_music(&mut self, url: &str) {
        self.music.prefetch(url);
    }

    pub fn current_music(&self) -> Option<&str> {
        self.music.current_url()
    }

    pub fn set_music_volume(&mut self, v: f32) {
        self.music_volume = v.max(0.0);
        self.music.set_volume(self.music_volume);
        if let Some(g) = &self.music_bus {
            g.gain().set_value(self.music_volume);
        }
    }
    pub fn music_volume(&self) -> f32 {
        self.music_volume
    }
    pub fn set_sfx_volume(&mut self, v: f32) {
        self.sfx_volume = v.max(0.0);
        if let Some(g) = &self.sfx_bus {
            g.gain().set_value(self.sfx_volume);
        }
    }
    pub fn sfx_volume(&self) -> f32 {
        self.sfx_volume
    }

    /// AI snapshot of the music player.
    pub fn describe_music(&self) -> serde_json::Value {
        self.music.describe()
    }

    fn suspend(&self) {
        if let Some(c) = &self.ctx {
            if c.state() == web_sys::AudioContextState::Running {
                let _ = c.suspend();
            }
        }
    }

    /// Called by the engine on window focus/blur and visibility changes. Suspending the
    /// AudioContext freezes its clock, so every voice (incl. loops and echo tails) continues
    /// exactly where it stopped when focus returns.
    pub fn set_focused(&mut self, focused: bool) {
        if self.focused == focused {
            return;
        }
        self.focused = focused;
        if !self.pause_on_blur {
            return;
        }
        if focused {
            self.resume();
        } else {
            self.suspend();
        }
        self.music.set_running(focused);
        log_info!(
            "audio {} (tab {})",
            if focused { "resumed" } else { "suspended" },
            if focused { "focused" } else { "lost focus" }
        );
    }

    pub fn focused(&self) -> bool {
        self.focused
    }

    /// Turn the pause-on-focus-loss behaviour on/off (e.g. for background music players).
    pub fn set_pause_on_blur(&mut self, on: bool) {
        self.pause_on_blur = on;
        if on && !self.focused {
            self.suspend();
            self.music.set_running(false);
        } else {
            self.resume();
            self.music.set_running(true);
        }
    }
    pub fn pause_on_blur(&self) -> bool {
        self.pause_on_blur
    }

    pub fn set_volume(&mut self, v: f32) {
        self.volume = v.max(0.0);
        self.apply_master();
    }
    pub fn volume(&self) -> f32 {
        self.volume
    }
    pub fn set_muted(&mut self, m: bool) {
        self.muted = m;
        self.apply_master();
    }
    pub fn muted(&self) -> bool {
        self.muted
    }
    fn apply_master(&self) {
        if let Some(g) = &self.master {
            g.gain()
                .set_value(if self.muted { 0.0 } else { self.volume });
        }
    }

    pub fn find(&self, name: &str) -> Option<SoundId> {
        self.by_name.get(name).copied()
    }

    fn register(&mut self, name: &str, state: SoundState, buffer: Option<AudioBuffer>) -> SoundId {
        if let Some(id) = self.find(name) {
            let s = &mut self.sounds[id.0 as usize];
            s.state = state;
            s.buffer = buffer;
            return id;
        }
        self.sounds.push(Sound {
            name: name.into(),
            buffer,
            state,
            plays: 0,
        });
        let id = SoundId(self.sounds.len() as u32 - 1);
        self.by_name.insert(name.into(), id);
        id
    }

    /// Mono sound from samples in [-1, 1].
    pub fn add_samples(&mut self, name: &str, samples: &[f32], sample_rate: f32) -> SoundId {
        let buffer = self.ctx.as_ref().and_then(|c| {
            let b = c
                .create_buffer(1, samples.len().max(1) as u32, sample_rate)
                .ok()?;
            b.copy_to_channel(samples, 0).ok()?;
            Some(b)
        });
        let state = if buffer.is_some() {
            SoundState::Ready
        } else {
            SoundState::Failed("no audio context".into())
        };
        self.register(name, state, buffer)
    }

    /// Renders an [`synth::Sfx`] and registers it under `name`.
    pub fn add_sfx(&mut self, name: &str, sfx: &synth::Sfx) -> SoundId {
        self.add_samples(name, &sfx.render(), synth::RATE)
    }

    /// Registers a synth preset under its preset name (e.g. "laser"); `seed` != 0 mutates it.
    pub fn add_preset(&mut self, preset: synth::SfxPreset, seed: u32) -> SoundId {
        let base = synth::Sfx::preset(preset);
        let sfx = if seed == 0 {
            base
        } else {
            base.mutate(seed, 1.0)
        };
        self.add_sfx(preset.name(), &sfx)
    }

    fn impulse(&mut self, ctx: &AudioContext, seconds: f32) -> Option<AudioBuffer> {
        let key = (seconds.clamp(0.05, 8.0) * 1000.0) as u32;
        if let Some(b) = self.impulses.get(&key) {
            return Some(b.clone());
        }
        let rate = ctx.sample_rate();
        let len = (key as f32 / 1000.0 * rate) as usize;
        let b = ctx.create_buffer(2, len.max(1) as u32, rate).ok()?;
        let mut seed = 0x9E37_79B9u32;
        for ch in 0..2 {
            let data: Vec<f32> = (0..len)
                .map(|i| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let w = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
                    w * (1.0 - i as f32 / len as f32).powf(3.0)
                })
                .collect();
            b.copy_to_channel(&data, ch).ok()?;
        }
        self.impulses.insert(key, b.clone());
        Some(b)
    }

    /// Plays a sound. Returns `None` if it is not loaded (yet) or audio is unavailable.
    pub fn play(&mut self, id: SoundId, p: Play) -> Option<VoiceId> {
        let ctx = self.ctx.clone()?;
        let master = self.sfx_bus.clone()?; // voices go to the sfx bus
        let sound = self.sounds.get(id.0 as usize)?;
        let Some(buffer) = sound.buffer.clone() else {
            log_warn!("sound '{}' not ready ({:?})", sound.name, sound.state);
            return None;
        };
        // Unfocused tab: drop one-shots instead of queueing them in the suspended context (they
        // would all fire at once on refocus). Loops still start and continue on refocus.
        if self.pause_on_blur && !self.focused && !p.looped {
            let ev = PlayEvent {
                sound: sound.name.clone(),
                voice: u32::MAX,
                step: self.step,
                frame: self.frame,
                pos: p.at.map(|v| v.into()),
                volume: p.volume,
                pitch: p.pitch,
                fx: p.fx,
                suppressed: true,
            };
            self.push_history(ev);
            return None;
        }
        let src = ctx.create_buffer_source().ok()?;
        src.set_buffer(Some(&buffer));
        src.set_loop(p.looped);
        src.playback_rate().set_value(p.pitch.max(0.01));
        src.detune().set_value(p.fx.detune);
        let fx = p.fx;
        // insert chain: source -> [shaper] -> [hp] -> [lp] -> [bp] -> gain
        let mut head: AudioNode = src.clone().unchecked_into();
        if let Some(drive) = fx.distortion.filter(|d| *d > 0.0) {
            let ws = ctx.create_wave_shaper().ok()?;
            let mut curve: Vec<f32> = (0..1024)
                .map(|i| {
                    let x = i as f32 / 511.5 - 1.0;
                    (1.0 + drive) * x / (1.0 + drive * x.abs())
                })
                .collect();
            ws.set_curve_opt_f32_slice(Some(&mut curve));
            ws.set_oversample(OverSampleType::N4x);
            head.connect_with_audio_node(&ws).ok()?;
            head = ws.unchecked_into();
        }
        let filters = [
            fx.highpass.map(|f| (BiquadFilterType::Highpass, f, 0.707)),
            fx.lowpass.map(|f| (BiquadFilterType::Lowpass, f, 0.707)),
            fx.bandpass.map(|(f, q)| (BiquadFilterType::Bandpass, f, q)),
        ];
        for (kind, freq, q) in filters.into_iter().flatten() {
            let bq = ctx.create_biquad_filter().ok()?;
            bq.set_type(kind);
            bq.frequency().set_value(freq);
            bq.q().set_value(q);
            head.connect_with_audio_node(&bq).ok()?;
            head = bq.unchecked_into();
        }
        let gain = ctx.create_gain().ok()?;
        gain.gain().set_value(p.volume.max(0.0));
        head.connect_with_audio_node(&gain).ok()?;
        let out = ctx.create_gain().ok()?;
        gain.connect_with_audio_node(&out).ok()?;
        if let Some((delay, feedback, mix)) = fx.echo {
            let d = ctx.create_delay_with_max_delay_time(5.0).ok()?;
            d.delay_time().set_value(delay.clamp(0.01, 5.0));
            let fb = ctx.create_gain().ok()?;
            fb.gain().set_value(feedback.clamp(0.0, 0.95));
            let wet = ctx.create_gain().ok()?;
            wet.gain().set_value(mix);
            gain.connect_with_audio_node(&d).ok()?;
            d.connect_with_audio_node(&fb).ok()?;
            fb.connect_with_audio_node(&d).ok()?;
            d.connect_with_audio_node(&wet).ok()?;
            wet.connect_with_audio_node(&out).ok()?;
        }
        if let Some((secs, mix)) = fx.reverb {
            if let Some(ir) = self.impulse(&ctx, secs) {
                let cv = ctx.create_convolver().ok()?;
                cv.set_buffer(Some(&ir));
                let wet = ctx.create_gain().ok()?;
                wet.gain().set_value(mix);
                gain.connect_with_audio_node(&cv).ok()?;
                cv.connect_with_audio_node(&wet).ok()?;
                wet.connect_with_audio_node(&out).ok()?;
            }
        }
        let mut tail_node: AudioNode = out.clone().unchecked_into();
        if let Some(x) = fx.pan {
            let sp = ctx.create_stereo_panner().ok()?;
            sp.pan().set_value(x);
            tail_node.connect_with_audio_node(&sp).ok()?;
            tail_node = sp.unchecked_into();
        }
        let gain = tail_node;
        let sound = self.sounds.get_mut(id.0 as usize)?;
        let positional = p.at.is_some() || p.follow.is_some();
        let panner = if positional {
            let pn = ctx.create_panner().ok()?;
            pn.set_panning_model(PanningModelType::Hrtf);
            pn.set_distance_model(DistanceModelType::Inverse);
            pn.set_ref_distance(p.ref_distance.max(0.01) as f64);
            pn.set_max_distance(500.0);
            pn.set_rolloff_factor(1.0);
            if let Some(at) = p.at {
                set_panner(&pn, at);
            }
            gain.connect_with_audio_node(&pn).ok()?;
            pn.connect_with_audio_node(&master).ok()?;
            Some(pn)
        } else {
            gain.connect_with_audio_node(&master).ok()?;
            None
        };
        let ended = Rc::new(Cell::new(false));
        let flag = ended.clone();
        let on_end = Closure::<dyn FnMut()>::new(move || flag.set(true));
        web_sys::AudioScheduledSourceNode::set_onended(&src, Some(on_end.as_ref().unchecked_ref()));
        web_sys::AudioScheduledSourceNode::start(&src).ok()?;
        sound.plays += 1;
        let name = sound.name.clone();
        let vid = VoiceId(self.next_voice);
        self.next_voice += 1;
        self.plays_total += 1;
        self.push_history(PlayEvent {
            sound: name,
            voice: vid.0,
            step: self.step,
            frame: self.frame,
            pos: p.at.map(|v| v.into()),
            volume: p.volume,
            pitch: p.pitch,
            fx,
            suppressed: false,
        });
        self.voices.push(Voice {
            id: vid,
            sound: id,
            src,
            panner,
            follow: p.follow,
            pos: p.at,
            ended,
            out,
            tail: fx.tail(),
            ended_at: None,
            fx,
            _on_end: on_end,
        });
        Some(vid)
    }

    fn push_history(&mut self, ev: PlayEvent) {
        self.history.push_back(ev);
        if self.history.len() > 32 {
            self.history.pop_front();
        }
    }

    /// Plays by name (convenience for debug commands and quick game code).
    pub fn play_named(&mut self, name: &str, p: Play) -> Option<VoiceId> {
        let id = self.find(name)?;
        self.play(id, p)
    }

    pub fn stop(&mut self, v: VoiceId) {
        if let Some(voice) = self.voices.iter().find(|x| x.id == v) {
            let _ = web_sys::AudioScheduledSourceNode::stop(&voice.src);
        }
    }

    pub fn stop_all(&mut self) {
        for v in &self.voices {
            let _ = web_sys::AudioScheduledSourceNode::stop(&v.src);
        }
    }

    /// Per frame: listener = camera, positional voices follow their entities, finished voices drop.
    pub(crate) fn update(&mut self, world: &World, camera: Option<Mat4>) {
        if let Some(c) = &self.ctx {
            let now = c.current_time();
            self.music.update(now);
        }
        if let (Some(ctx), Some(cam)) = (&self.ctx, camera) {
            let l = ctx.listener();
            let p = cam.w_axis.truncate();
            let fwd = -cam.z_axis.truncate().normalize_or(Vec3::NEG_Z);
            let up = cam.y_axis.truncate().normalize_or(Vec3::Y);
            l.set_position(p.x as f64, p.y as f64, p.z as f64);
            l.set_orientation(
                fwd.x as f64,
                fwd.y as f64,
                fwd.z as f64,
                up.x as f64,
                up.y as f64,
                up.z as f64,
            );
            self.listener = p;
        }
        for v in &mut self.voices {
            if let (Some(e), Some(pn)) = (v.follow, &v.panner) {
                if let Ok(g) = world.get::<&GlobalTransform>(e) {
                    let p = g.translation();
                    set_panner(pn, p);
                    v.pos = Some(p);
                }
            }
        }
        let now = self.ctx.as_ref().map(|c| c.current_time()).unwrap_or(0.0);
        self.voices.retain_mut(|v| {
            if !v.ended.get() {
                return true;
            }
            let t = *v.ended_at.get_or_insert(now);
            let done = now >= t + v.tail;
            if done {
                v.out.disconnect().ok();
                if let Some(pn) = &v.panner {
                    pn.disconnect().ok();
                }
            }
            !done
        });
    }

    /// AI snapshot.
    pub fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "state": self.state(),
            "volume": self.volume,
            "muted": self.muted,
            "focused": self.focused,
            "time": self.ctx.as_ref().map(|c| (c.current_time() * 1000.0).round() / 1000.0),
            "pause_on_blur": self.pause_on_blur,
            "listener": [self.listener.x, self.listener.y, self.listener.z],
            "sounds": self.sounds.iter().enumerate().map(|(i, s)| serde_json::json!({
                "id": i, "name": s.name, "state": s.state, "plays": s.plays,
                "duration": s.buffer.as_ref().map(|b| (b.duration() * 1000.0).round() / 1000.0),
            })).collect::<Vec<_>>(),
            "voices": self.voices.iter().map(|v| serde_json::json!({
                "id": v.id.0, "sound": self.sounds.get(v.sound.0 as usize).map(|s| s.name.clone()),
                "pos": v.pos.map(|p| [p.x, p.y, p.z]),
                "distance": v.pos.map(|p| p.distance(self.listener)),
                "fx": v.fx, "ended": v.ended.get(),
            })).collect::<Vec<_>>(),
            "presets": synth::SfxPreset::ALL.iter().map(|p| p.name()).collect::<Vec<_>>(),
            "plays_total": self.plays_total,
            "sfx_volume": self.sfx_volume,
            "music_volume": self.music_volume,
            "music": self.music.describe(),
            "history": self.history,
        })
    }
}

fn set_panner(pn: &PannerNode, p: Vec3) {
    pn.position_x().set_value(p.x);
    pn.position_y().set_value(p.y);
    pn.position_z().set_value(p.z);
}

/// Loads and decodes an audio file (fetch + `decodeAudioData`). Registered under its URL.
pub(crate) fn request(ctx: &mut crate::app::Ctx, url: &str) -> SoundId {
    if let Some(id) = ctx.audio.find(url) {
        return id;
    }
    let id = ctx.audio.register(url, SoundState::Loading, None);
    let Some(actx) = ctx.audio.ctx.clone() else {
        ctx.audio
            .register(url, SoundState::Failed("no audio context".into()), None);
        return id;
    };
    let me = ctx.me.clone();
    let url = url.to_string();
    wasm_bindgen_futures::spawn_local(async move {
        let t0 = crate::log::now_ms();
        let result: Result<AudioBuffer, String> = async {
            let bytes = crate::gltf_loader::fetch_bytes(&url).await?;
            let arr = js_sys::Uint8Array::from(&bytes[..]);
            let promise = actx
                .decode_audio_data(&arr.buffer())
                .map_err(|e| format!("decode: {e:?}"))?;
            let buf = wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| format!("decode {url}: {e:?}"))?;
            Ok(buf.unchecked_into())
        }
        .await;
        let Some(rc) = me.upgrade() else { return };
        let Ok(mut c) = rc.try_borrow_mut() else {
            return;
        };
        match result {
            Ok(b) => {
                log_info!(
                    "sound loaded: {url} ({:.2} s, {:.0} ms)",
                    b.duration(),
                    crate::log::now_ms() - t0
                );
                c.audio.register(&url, SoundState::Ready, Some(b));
            }
            Err(e) => {
                crate::log_error!("sound {url} failed: {e}");
                c.audio.register(&url, SoundState::Failed(e), None);
            }
        }
    });
    id
}

pub mod music;
pub mod synth;

pub use music::MusicOpts;

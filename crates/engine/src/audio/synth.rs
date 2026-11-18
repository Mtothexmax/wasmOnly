//! Deterministic sound-effect synthesizer (sfxr-style). Pure Rust, no assets, same `Sfx` = same
//! samples, so sounds are reproducible and diffable by agents.
//!
//! * Oscillators: sine, square (duty), saw, triangle, white / pink / brown noise.
//! * Timbre: extra `partials` (frequency multiplier, amplitude) — harmonic (2, 3, 4…) for organ-like
//!   tones, inharmonic (2.76, 5.4…) for bells — plus `duty`, filters and `bitcrush`.
//! * Pitch: `freq` -> `freq_end` slide, vibrato, one arpeggio jump.
//! * ADSR envelope, one-pole low/high-pass, volume.
//! * [`SfxPreset`]: coin, laser, explosion, powerup, hit, jump, blip, click, select, alarm, thud,
//!   chime, bell and seamless loops wind, rain, engine. `Sfx::mutate(seed)` gives variations.
//!
//! Playback-time effects (echo, reverb, distortion, filters, pan, detune) are Web Audio nodes:
//! see `Play` in the parent module.

use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;

/// Sample rate used by the synthesizer.
pub const RATE: f32 = 44_100.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Wave {
    Sine,
    Square,
    Saw,
    Triangle,
    /// White noise; `freq` sets the sample-and-hold rate (low = grainy, high = hiss).
    Noise,
    /// Pink noise (-3 dB/octave): wind, surf, soft ambience.
    Pink,
    /// Brown noise (-6 dB/octave): rumble, explosions, distant thunder.
    Brown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sfx {
    pub wave: Wave,
    /// Start frequency (Hz).
    pub freq: f32,
    /// End frequency (Hz) — exponential slide over the whole sound.
    pub freq_end: f32,
    /// Square duty cycle 0..1.
    pub duty: f32,
    pub attack: f32,
    pub decay: f32,
    /// Sustain level 0..1.
    pub sustain: f32,
    pub sustain_time: f32,
    pub release: f32,
    /// Vibrato depth in semitones and rate in Hz.
    pub vibrato_depth: f32,
    pub vibrato_rate: f32,
    /// Frequency multiplier applied after `arp_time` seconds (0 = off).
    pub arp_mult: f32,
    pub arp_time: f32,
    /// Additional sine partials (frequency multiplier, amplitude) — shapes the timbre.
    pub partials: Vec<(f32, f32)>,
    /// White-noise mix 0..1 on top of the oscillator.
    pub noise_mix: f32,
    /// One-pole filters in Hz (0 = off).
    pub lowpass: f32,
    pub highpass: f32,
    /// Quantize to this many bits (0 = off). 4..8 = retro.
    pub bitcrush: u32,
    pub volume: f32,
    pub seed: u32,
}

impl Default for Sfx {
    fn default() -> Self {
        Self {
            wave: Wave::Square,
            freq: 440.0,
            freq_end: 440.0,
            duty: 0.5,
            attack: 0.005,
            decay: 0.05,
            sustain: 0.6,
            sustain_time: 0.1,
            release: 0.2,
            vibrato_depth: 0.0,
            vibrato_rate: 6.0,
            arp_mult: 0.0,
            arp_time: 0.0,
            partials: vec![],
            noise_mix: 0.0,
            lowpass: 0.0,
            highpass: 0.0,
            bitcrush: 0,
            volume: 0.6,
            seed: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SfxPreset {
    Coin,
    Laser,
    Explosion,
    Powerup,
    Hit,
    Jump,
    Blip,
    Click,
    Select,
    Alarm,
    Thud,
    Chime,
    Bell,
    /// Seamless loop.
    Wind,
    /// Seamless loop.
    Rain,
    /// Seamless loop.
    Engine,
}

impl SfxPreset {
    pub const ALL: [SfxPreset; 16] = [
        SfxPreset::Coin,
        SfxPreset::Laser,
        SfxPreset::Explosion,
        SfxPreset::Powerup,
        SfxPreset::Hit,
        SfxPreset::Jump,
        SfxPreset::Blip,
        SfxPreset::Click,
        SfxPreset::Select,
        SfxPreset::Alarm,
        SfxPreset::Thud,
        SfxPreset::Chime,
        SfxPreset::Bell,
        SfxPreset::Wind,
        SfxPreset::Rain,
        SfxPreset::Engine,
    ];

    pub fn name(self) -> &'static str {
        match self {
            SfxPreset::Coin => "coin",
            SfxPreset::Laser => "laser",
            SfxPreset::Explosion => "explosion",
            SfxPreset::Powerup => "powerup",
            SfxPreset::Hit => "hit",
            SfxPreset::Jump => "jump",
            SfxPreset::Blip => "blip",
            SfxPreset::Click => "click",
            SfxPreset::Select => "select",
            SfxPreset::Alarm => "alarm",
            SfxPreset::Thud => "thud",
            SfxPreset::Chime => "chime",
            SfxPreset::Bell => "bell",
            SfxPreset::Wind => "wind",
            SfxPreset::Rain => "rain",
            SfxPreset::Engine => "engine",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == s)
    }

    /// Loops are rendered with a flat envelope and a length that wraps cleanly; play them with
    /// `Play::new().looped()`.
    pub fn is_loop(self) -> bool {
        matches!(self, SfxPreset::Wind | SfxPreset::Rain | SfxPreset::Engine)
    }
}

impl Sfx {
    pub fn preset(p: SfxPreset) -> Sfx {
        let d = Sfx::default();
        let flat = |secs: f32| Sfx {
            attack: 0.0,
            decay: 0.0,
            sustain: 1.0,
            sustain_time: secs,
            release: 0.0,
            ..Sfx::default()
        };
        match p {
            SfxPreset::Coin => Sfx {
                freq: 988.0,
                freq_end: 988.0,
                arp_mult: 1.335,
                arp_time: 0.07,
                attack: 0.002,
                sustain_time: 0.09,
                release: 0.25,
                ..d
            },
            SfxPreset::Laser => Sfx {
                wave: Wave::Saw,
                freq: 1400.0,
                freq_end: 160.0,
                attack: 0.001,
                decay: 0.02,
                sustain_time: 0.04,
                release: 0.14,
                lowpass: 7000.0,
                ..d
            },
            SfxPreset::Explosion => Sfx {
                wave: Wave::Brown,
                freq: 0.0,
                freq_end: 0.0,
                noise_mix: 0.35,
                attack: 0.003,
                decay: 0.1,
                sustain: 0.5,
                sustain_time: 0.1,
                release: 0.9,
                lowpass: 1400.0,
                volume: 0.9,
                ..d
            },
            SfxPreset::Powerup => Sfx {
                duty: 0.25,
                freq: 300.0,
                freq_end: 900.0,
                vibrato_depth: 0.4,
                vibrato_rate: 14.0,
                sustain_time: 0.25,
                release: 0.25,
                ..d
            },
            SfxPreset::Hit => Sfx {
                wave: Wave::Noise,
                freq: 1800.0,
                freq_end: 300.0,
                attack: 0.001,
                decay: 0.03,
                sustain_time: 0.02,
                release: 0.12,
                lowpass: 3500.0,
                ..d
            },
            SfxPreset::Jump => Sfx {
                freq: 260.0,
                freq_end: 720.0,
                sustain_time: 0.03,
                release: 0.16,
                volume: 0.5,
                ..d
            },
            SfxPreset::Blip => Sfx {
                freq: 880.0,
                freq_end: 880.0,
                attack: 0.001,
                sustain_time: 0.04,
                release: 0.05,
                volume: 0.45,
                ..d
            },
            SfxPreset::Click => Sfx {
                wave: Wave::Triangle,
                freq: 2000.0,
                freq_end: 1200.0,
                noise_mix: 0.1,
                attack: 0.0005,
                decay: 0.0,
                sustain_time: 0.01,
                release: 0.035,
                ..d
            },
            SfxPreset::Select => Sfx {
                wave: Wave::Sine,
                freq: 660.0,
                freq_end: 660.0,
                arp_mult: 1.5,
                arp_time: 0.07,
                sustain_time: 0.1,
                release: 0.15,
                partials: vec![(2.0, 0.3)],
                ..d
            },
            SfxPreset::Alarm => Sfx {
                freq: 600.0,
                freq_end: 600.0,
                vibrato_depth: 4.0,
                vibrato_rate: 5.0,
                sustain_time: 0.8,
                release: 0.1,
                lowpass: 3000.0,
                volume: 0.4,
                ..d
            },
            SfxPreset::Thud => Sfx {
                wave: Wave::Sine,
                freq: 140.0,
                freq_end: 45.0,
                noise_mix: 0.3,
                attack: 0.002,
                decay: 0.05,
                sustain_time: 0.03,
                release: 0.18,
                lowpass: 900.0,
                volume: 0.8,
                ..d
            },
            SfxPreset::Chime => Sfx {
                wave: Wave::Sine,
                freq: 880.0,
                freq_end: 880.0,
                arp_mult: 1.5,
                arp_time: 0.14,
                sustain_time: 0.2,
                release: 0.5,
                partials: vec![(2.0, 0.25), (3.0, 0.1)],
                ..d
            },
            SfxPreset::Bell => Sfx {
                wave: Wave::Sine,
                freq: 520.0,
                freq_end: 520.0,
                attack: 0.001,
                decay: 0.3,
                sustain: 0.35,
                sustain_time: 0.1,
                release: 1.6,
                partials: vec![(2.76, 0.5), (5.4, 0.25), (8.93, 0.12)],
                ..d
            },
            SfxPreset::Wind => Sfx {
                wave: Wave::Pink,
                lowpass: 650.0,
                highpass: 90.0,
                volume: 0.9,
                ..flat(4.0)
            },
            SfxPreset::Rain => Sfx {
                wave: Wave::Noise,
                freq: 22_050.0,
                freq_end: 22_050.0,
                highpass: 1500.0,
                lowpass: 9000.0,
                volume: 0.35,
                ..flat(2.0)
            },
            // 66 cycles of 55 Hz: the loop ends exactly on a period boundary.
            SfxPreset::Engine => Sfx {
                wave: Wave::Saw,
                freq: 55.0,
                freq_end: 55.0,
                partials: vec![(2.0, 0.3), (0.5, 0.4)],
                lowpass: 420.0,
                volume: 0.7,
                ..flat(66.0 / 55.0)
            },
        }
    }

    /// Total length in seconds.
    pub fn duration(&self) -> f32 {
        (self.attack + self.decay + self.sustain_time + self.release).max(0.01)
    }

    /// Deterministic variation (like sfxr "mutate"): small random changes to pitch, slide,
    /// envelope and timbre. `amount` 0..1.
    pub fn mutate(&self, seed: u32, amount: f32) -> Sfx {
        let mut r = Lcg(seed.wrapping_mul(2_654_435_761).max(1));
        let mut j = |x: f32, rel: f32| x * (1.0 + (r.next() * 2.0 - 1.0) * rel * amount);
        let mut s = self.clone();
        s.freq = j(s.freq, 0.25);
        s.freq_end = j(s.freq_end, 0.25);
        s.duty = j(s.duty, 0.3).clamp(0.05, 0.95);
        s.sustain_time = j(s.sustain_time, 0.3);
        s.release = j(s.release, 0.3);
        s.vibrato_rate = j(s.vibrato_rate, 0.3);
        if s.lowpass > 0.0 {
            s.lowpass = j(s.lowpass, 0.3);
        }
        s.seed = seed;
        s
    }

    fn env(&self, t: f32) -> f32 {
        let (a, d, st, r) = (self.attack, self.decay, self.sustain_time, self.release);
        if t < a {
            t / a.max(1e-6)
        } else if t < a + d {
            1.0 - (1.0 - self.sustain) * (t - a) / d.max(1e-6)
        } else if t < a + d + st {
            self.sustain
        } else {
            let k = ((t - a - d - st) / r.max(1e-6)).min(1.0);
            self.sustain * (1.0 - k) * (1.0 - k)
        }
    }

    /// Renders mono samples at [`RATE`].
    pub fn render(&self) -> Vec<f32> {
        let dur = self.duration();
        let n = (dur * RATE) as usize;
        let mut out = Vec::with_capacity(n);
        let mut rng = Lcg(self.seed.max(1));
        let mut phase = 0.0f32;
        let mut partial_phase = vec![0.0f32; self.partials.len()];
        let mut held = 0.0f32;
        let mut hold_phase = 1.0f32;
        let (mut b0, mut b1, mut b2, mut b3, mut b4, mut b5, mut b6) =
            (0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let mut brown = 0.0f32;
        let mut lp = 0.0f32;
        let mut hp_lp = 0.0f32;
        let lp_a = if self.lowpass > 0.0 {
            1.0 - (-TAU * self.lowpass / RATE).exp()
        } else {
            1.0
        };
        let hp_a = if self.highpass > 0.0 {
            1.0 - (-TAU * self.highpass / RATE).exp()
        } else {
            0.0
        };
        let ratio = if self.freq > 0.0 && self.freq_end > 0.0 {
            self.freq_end / self.freq
        } else {
            1.0
        };
        for i in 0..n {
            let t = i as f32 / RATE;
            let mut f = self.freq * ratio.powf(t / dur);
            if self.arp_mult > 0.0 && t >= self.arp_time {
                f *= self.arp_mult;
            }
            if self.vibrato_depth > 0.0 {
                f *= 2f32.powf(self.vibrato_depth * (TAU * self.vibrato_rate * t).sin() / 12.0);
            }
            phase = (phase + f / RATE).fract();
            let white = rng.next() * 2.0 - 1.0;
            let osc = match self.wave {
                Wave::Sine => (phase * TAU).sin(),
                Wave::Square => {
                    if phase < self.duty {
                        1.0
                    } else {
                        -1.0
                    }
                }
                Wave::Saw => phase * 2.0 - 1.0,
                Wave::Triangle => 1.0 - 4.0 * (phase - 0.5).abs(),
                Wave::Noise => {
                    hold_phase += f / RATE;
                    if hold_phase >= 1.0 {
                        hold_phase = hold_phase.fract();
                        held = white;
                    }
                    held
                }
                Wave::Pink => {
                    // Paul Kellet's refined pink filter
                    b0 = 0.99886 * b0 + white * 0.0555179;
                    b1 = 0.99332 * b1 + white * 0.0750759;
                    b2 = 0.96900 * b2 + white * 0.153_852;
                    b3 = 0.86650 * b3 + white * 0.3104856;
                    b4 = 0.55000 * b4 + white * 0.5329522;
                    b5 = -0.7616 * b5 - white * 0.0168980;
                    let p = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
                    b6 = white * 0.115926;
                    p * 0.11
                }
                Wave::Brown => {
                    brown = (brown + white * 0.02) / 1.02;
                    brown * 3.5
                }
            };
            let mut s = osc;
            for (k, (mult, amp)) in self.partials.iter().enumerate() {
                partial_phase[k] = (partial_phase[k] + f * mult / RATE).fract();
                s += (partial_phase[k] * TAU).sin() * amp;
            }
            s = s * (1.0 - self.noise_mix) + white * self.noise_mix;
            lp += lp_a * (s - lp);
            s = lp;
            if hp_a > 0.0 {
                hp_lp += hp_a * (s - hp_lp);
                s -= hp_lp;
            }
            s *= self.env(t) * self.volume;
            if self.bitcrush > 0 {
                let q = (1u32 << self.bitcrush.min(16)) as f32 * 0.5;
                s = (s * q).round() / q;
            }
            out.push(s.clamp(-1.0, 1.0));
        }
        out
    }
}

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Peak and RMS of rendered samples (AI-side check that a sound is not silent / clipping).
pub fn levels(samples: &[f32]) -> (f32, f32) {
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt();
    (peak, rms)
}

// --- Backwards-compatible one-liners -----------------------------------------------------------

/// Pitch sweep (`f0` -> `f1` Hz) with quadratic decay; `noise` 0..1 white-noise mix,
/// `square` 0..1 blends sine -> square.
pub fn sweep(f0: f32, f1: f32, seconds: f32, noise: f32, square: f32) -> Vec<f32> {
    let n = (seconds * RATE) as usize;
    let mut out = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    let mut rng = Lcg(0x1234_5678);
    for i in 0..n {
        let t = i as f32 / n.max(1) as f32;
        let f = f0 * (f1 / f0).powf(t);
        phase += f / RATE;
        let s = (phase * TAU).sin();
        let tone = s * (1.0 - square) + s.signum() * square;
        let wn = rng.next() * 2.0 - 1.0;
        let env = (1.0 - t).powf(2.0) * (t * 200.0).min(1.0);
        out.push((tone * (1.0 - noise) + wn * noise) * env * 0.6);
    }
    out
}

pub fn thud() -> Vec<f32> {
    Sfx::preset(SfxPreset::Thud).render()
}
pub fn zap() -> Vec<f32> {
    Sfx::preset(SfxPreset::Laser).render()
}
pub fn jump() -> Vec<f32> {
    Sfx::preset(SfxPreset::Jump).render()
}
pub fn click() -> Vec<f32> {
    Sfx::preset(SfxPreset::Click).render()
}
pub fn chime() -> Vec<f32> {
    Sfx::preset(SfxPreset::Chime).render()
}

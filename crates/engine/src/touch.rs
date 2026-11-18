//! Touch input and gestures (Android Chrome; also any touchscreen PC). Every finger is tracked by
//! its pointer id; the engine turns them into game-ready gestures:
//!
//! * **joystick** — a finger that starts in the left 40 % of the canvas becomes a virtual stick
//!   (`axis`, folded into `Input::move_axis()`); a ring + knob is drawn while it is held.
//! * **look** — fingers starting elsewhere drag the camera (`look_delta`, CSS px per frame).
//! * **pinch** — two non-joystick fingers: `pinch` = log zoom factor this frame (> 0 = spread).
//! * **tap** — short touch (< 250 ms, < 12 px movement): `taps` positions this frame. A tap on a
//!   world panel clicks it and is removed from `taps`.
//! * **hold** — touch held > 450 ms without moving: `holds_started` this frame, `Touch::held`.
//! * **swipe** — fast flick (> 60 px within 300 ms): `swipes` (direction, px/s) this frame.
//!
//! Real touches arrive as pointer events (`pointerType == "touch"`); AI agents inject the same
//! path with `engine.call("input", {touch: {id, x, y, phase: "start"|"move"|"end"}})`.

use glam::Vec2;
use serde::Serialize;

const TAP_MS: f64 = 250.0;
const TAP_PX: f32 = 12.0;
const HOLD_MS: f64 = 450.0;
const SWIPE_PX: f32 = 60.0;
const SWIPE_MS: f64 = 300.0;
/// Joystick radius in CSS px.
pub const STICK_RADIUS: f32 = 60.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TouchRole {
    Joystick,
    Look,
}

#[derive(Clone, Debug, Serialize)]
pub struct Touch {
    pub id: i32,
    pub pos: Vec2,
    pub start: Vec2,
    #[serde(skip)]
    pub start_ms: f64,
    /// Largest distance from the start position so far (px).
    pub travel: f32,
    pub role: TouchRole,
    pub held: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Swipe {
    pub dir: Vec2,
    pub speed: f32,
    pub from: Vec2,
}

#[derive(Default, Debug, Serialize)]
pub struct Gestures {
    pub touches: Vec<Touch>,
    /// Virtual stick, x = right, y = forward (up on screen), length <= 1.
    pub axis: Vec2,
    pub look_delta: Vec2,
    pub pinch: f32,
    pub taps: Vec<Vec2>,
    pub holds_started: Vec<Vec2>,
    pub swipes: Vec<Swipe>,
    /// A touch was ever seen (show touch HUD, hide mouse hints).
    pub touch_device: bool,
    #[serde(skip)]
    pinch_dist: Option<f32>,
}

impl Gestures {
    pub(crate) fn start(&mut self, id: i32, pos: Vec2, now: f64, canvas_w: f32) {
        self.touch_device = true;
        self.touches.retain(|t| t.id != id);
        let has_stick = self.touches.iter().any(|t| t.role == TouchRole::Joystick);
        let role = if !has_stick && pos.x < canvas_w * 0.4 { TouchRole::Joystick } else { TouchRole::Look };
        self.touches.push(Touch { id, pos, start: pos, start_ms: now, travel: 0.0, role, held: false });
        self.pinch_dist = None;
    }

    pub(crate) fn move_to(&mut self, id: i32, pos: Vec2) {
        let lookers = self.touches.iter().filter(|t| t.role == TouchRole::Look).count();
        let Some(t) = self.touches.iter_mut().find(|t| t.id == id) else { return };
        let d = pos - t.pos;
        t.pos = pos;
        t.travel = t.travel.max((pos - t.start).length());
        if t.role == TouchRole::Look && lookers == 1 {
            self.look_delta += d;
        }
        if lookers >= 2 {
            let looks: Vec<Vec2> = self.touches.iter().filter(|t| t.role == TouchRole::Look).take(2).map(|t| t.pos).collect();
            let dist = looks[0].distance(looks[1]).max(1.0);
            if let Some(prev) = self.pinch_dist {
                self.pinch += (dist / prev).ln();
            }
            self.pinch_dist = Some(dist);
        }
    }

    pub(crate) fn end(&mut self, id: i32, pos: Vec2, now: f64) {
        let Some(i) = self.touches.iter().position(|t| t.id == id) else { return };
        let t = self.touches.remove(i);
        let dt = now - t.start_ms;
        let travel = t.travel.max((pos - t.start).length());
        if dt < TAP_MS && travel < TAP_PX {
            self.taps.push(pos);
        } else if dt < SWIPE_MS && travel > SWIPE_PX {
            let v = pos - t.start;
            self.swipes.push(Swipe { dir: v.normalize_or_zero(), speed: v.length() / (dt.max(1.0) as f32 / 1000.0), from: t.start });
        }
        self.pinch_dist = None;
    }

    /// Per frame (before the game update): joystick axis and hold detection.
    pub(crate) fn update(&mut self, now: f64) {
        self.axis = Vec2::ZERO;
        for t in &mut self.touches {
            if t.role == TouchRole::Joystick {
                let v = (t.pos - t.start) / STICK_RADIUS;
                let v = if v.length() > 1.0 { v.normalize() } else { v };
                self.axis = Vec2::new(v.x, -v.y);
            } else if !t.held && t.travel < TAP_PX && now - t.start_ms > HOLD_MS {
                t.held = true;
                self.holds_started.push(t.pos);
            }
        }
    }

    pub(crate) fn end_frame(&mut self) {
        self.look_delta = Vec2::ZERO;
        self.pinch = 0.0;
        self.taps.clear();
        self.holds_started.clear();
        self.swipes.clear();
    }

    pub(crate) fn release_all(&mut self) {
        self.touches.clear();
        self.axis = Vec2::ZERO;
        self.pinch_dist = None;
    }

    /// The joystick finger (for drawing the ring/knob).
    pub fn stick(&self) -> Option<&Touch> {
        self.touches.iter().find(|t| t.role == TouchRole::Joystick)
    }
}

//! Input state. Browser events are collected into this struct; the game reads it.
//! Routing rule: DOM UI gets events first. Keyboard events whose target is an editable element are
//! ignored; pointer events only arrive here when they hit the canvas (the UI overlay root has
//! `pointer-events: none`, interactive UI elements opt in with `pointer-events: auto`).
//!
//! AI tooling can inject input via `engine.call("input", {...})` — identical code path.

use glam::Vec2;
use serde::Serialize;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
}

impl MouseButton {
    pub fn from_dom(b: i16) -> Option<Self> {
        match b {
            0 => Some(Self::Left),
            1 => Some(Self::Middle),
            2 => Some(Self::Right),
            _ => None,
        }
    }
}

#[derive(Default, Debug, Serialize)]
pub struct Input {
    /// `KeyboardEvent.code` values currently held (e.g. "KeyW", "Space", "ArrowLeft").
    pub keys: HashSet<String>,
    pressed: HashSet<String>,
    released: HashSet<String>,
    pub buttons: HashSet<MouseButton>,
    buttons_pressed: HashSet<MouseButton>,
    /// Presses since the last fixed step: `key_pressed` inside `fixed_update` reads these, so a tap
    /// that starts and ends between two fixed steps (short tap, paused simulation, high-refresh
    /// display) is still seen exactly once by the simulation.
    #[serde(skip)]
    fixed_pressed: HashSet<String>,
    #[serde(skip)]
    fixed_released: HashSet<String>,
    #[serde(skip)]
    fixed_buttons_pressed: HashSet<MouseButton>,
    #[serde(skip)]
    in_fixed: bool,
    /// Pointer position in CSS pixels relative to the canvas.
    pub pointer: Vec2,
    /// Accumulated movement since the last frame (CSS pixels; uses movementX/Y, works under pointer lock).
    pub pointer_delta: Vec2,
    pub wheel: f32,
    pub pointer_locked: bool,
    /// True while the pointer (or the screen centre under pointer lock) is over a world panel.
    pub pointer_over_panel: bool,
    /// Gamepad 0: left stick, right stick (deadzone applied).
    pub gamepad_left: Vec2,
    pub gamepad_right: Vec2,
    pub gamepad_buttons: Vec<bool>,
    /// Touch fingers and gestures (joystick, look, pinch, tap, hold, swipe), see [`crate::touch`].
    pub touch: crate::touch::Gestures,
}

impl Input {
    pub fn key(&self, code: &str) -> bool {
        self.keys.contains(code)
    }
    /// The key went down — in `update`: during this frame; in `fixed_update`: since the previous
    /// fixed step. Works in both places; a press is never lost or seen twice by the same callback.
    pub fn key_pressed(&self, code: &str) -> bool {
        if self.in_fixed { self.fixed_pressed.contains(code) } else { self.pressed.contains(code) }
    }
    pub fn key_released(&self, code: &str) -> bool {
        if self.in_fixed { self.fixed_released.contains(code) } else { self.released.contains(code) }
    }
    /// Held now, or pressed since the last fixed step (in `fixed_update`): use for actions like
    /// jump/fire so taps shorter than a step still count.
    pub fn key_down_or_pressed(&self, code: &str) -> bool {
        self.key(code) || self.key_pressed(code)
    }
    pub fn button(&self, b: MouseButton) -> bool {
        self.buttons.contains(&b)
    }
    pub fn button_pressed(&self, b: MouseButton) -> bool {
        if self.in_fixed { self.fixed_buttons_pressed.contains(&b) } else { self.buttons_pressed.contains(&b) }
    }
    pub(crate) fn begin_fixed(&mut self) {
        self.in_fixed = true;
    }
    /// After each fixed step: the step has seen these edges.
    pub(crate) fn end_fixed(&mut self) {
        self.in_fixed = false;
        self.fixed_pressed.clear();
        self.fixed_released.clear();
        self.fixed_buttons_pressed.clear();
    }
    /// WASD/arrow keys + gamepad left stick + touch joystick as a vector (x = right, y = forward).
    pub fn move_axis(&self) -> Vec2 {
        let mut v = Vec2::ZERO;
        if self.key("KeyW") || self.key("ArrowUp") {
            v.y += 1.0;
        }
        if self.key("KeyS") || self.key("ArrowDown") {
            v.y -= 1.0;
        }
        if self.key("KeyD") || self.key("ArrowRight") {
            v.x += 1.0;
        }
        if self.key("KeyA") || self.key("ArrowLeft") {
            v.x -= 1.0;
        }
        v += Vec2::new(self.gamepad_left.x, -self.gamepad_left.y);
        v += self.touch.axis;
        if v.length_squared() > 1.0 {
            v.normalize()
        } else {
            v
        }
    }

    pub fn key_down(&mut self, code: &str) {
        if self.keys.insert(code.to_string()) {
            self.pressed.insert(code.to_string());
            self.fixed_pressed.insert(code.to_string());
        }
    }
    pub fn key_up(&mut self, code: &str) {
        if self.keys.remove(code) {
            self.released.insert(code.to_string());
            self.fixed_released.insert(code.to_string());
        }
    }
    pub fn button_down(&mut self, b: MouseButton) {
        if self.buttons.insert(b) {
            self.buttons_pressed.insert(b);
            self.fixed_buttons_pressed.insert(b);
        }
    }
    pub fn button_up(&mut self, b: MouseButton) {
        self.buttons.remove(&b);
    }
    /// Removes this frame's press edge (used when a world panel consumed the click).
    pub(crate) fn consume_press(&mut self, b: MouseButton) {
        self.buttons_pressed.remove(&b);
    }
    /// Called by the engine at the end of each frame.
    pub(crate) fn end_frame(&mut self) {
        self.pressed.clear();
        self.released.clear();
        self.buttons_pressed.clear();
        self.pointer_delta = Vec2::ZERO;
        self.wheel = 0.0;
        self.touch.end_frame();
    }
    pub(crate) fn release_all(&mut self) {
        let keys: Vec<String> = self.keys.drain().collect();
        self.fixed_released.extend(keys.iter().cloned());
        self.released.extend(keys);
        self.buttons.clear();
        self.touch.release_all();
    }

    pub(crate) fn poll_gamepad(&mut self) {
        let Some(nav) = web_sys::window().map(|w| w.navigator()) else {
            return;
        };
        let Ok(pads) = nav.get_gamepads() else { return };
        let pad = pads
            .iter()
            .find_map(|p| wasm_bindgen::JsCast::dyn_into::<web_sys::Gamepad>(p).ok());
        let Some(pad) = pad else { return };
        let axes: Vec<f32> = pad
            .axes()
            .iter()
            .map(|a| a.as_f64().unwrap_or(0.0) as f32)
            .collect();
        let dz = |v: Vec2| if v.length() < 0.15 { Vec2::ZERO } else { v };
        self.gamepad_left = dz(Vec2::new(
            *axes.first().unwrap_or(&0.0),
            *axes.get(1).unwrap_or(&0.0),
        ));
        self.gamepad_right = dz(Vec2::new(
            *axes.get(2).unwrap_or(&0.0),
            *axes.get(3).unwrap_or(&0.0),
        ));
        self.gamepad_buttons = pad
            .buttons()
            .iter()
            .map(|b| {
                wasm_bindgen::JsCast::dyn_into::<web_sys::GamepadButton>(b)
                    .map(|b| b.pressed())
                    .unwrap_or(false)
            })
            .collect();
    }
}

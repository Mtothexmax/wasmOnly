//! UI bridge: the game's UI is plain HTML/CSS in a DOM overlay above the canvas.
//!
//! * Engine -> DOM: `ui.set("score", "<b>12</b>")` fills `<... data-slot="score">` (created on
//!   demand in the HUD). Writes are cached, so calling it every frame is cheap.
//! * DOM -> Engine: any click on an element with `data-action="name"` queues `"name"`;
//!   the game drains them with `ui.take_actions()`.
//! * World-anchored elements: the [`WorldLabel`] component; positioned via CSS transform per frame.

use crate::scene::{GlobalTransform, WorldLabel};
use glam::{Mat4, Vec3};
use hecs::{Entity, World};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Document, Element, HtmlElement};

pub struct Ui {
    doc: Document,
    pub root: HtmlElement,
    hud: HtmlElement,
    slots: HashMap<String, (HtmlElement, String)>,
    labels: HashMap<Entity, (HtmlElement, String)>,
    actions: Rc<RefCell<Vec<String>>>,
    stick: Option<(HtmlElement, HtmlElement)>,
}

impl Ui {
    pub(crate) fn new(doc: &Document) -> Result<Ui, JsValue> {
        let root: HtmlElement = match doc.get_element_by_id("ui") {
            Some(e) => e.dyn_into()?,
            None => {
                let e: HtmlElement = doc.create_element("div")?.dyn_into()?;
                e.set_id("ui");
                doc.body().ok_or("no body")?.append_child(&e)?;
                e
            }
        };
        let hud: HtmlElement = match root.query_selector(".hud")? {
            Some(e) => e.dyn_into()?,
            None => {
                let e: HtmlElement = doc.create_element("div")?.dyn_into()?;
                e.set_class_name("hud");
                root.append_child(&e)?;
                e
            }
        };
        let actions = Rc::new(RefCell::new(Vec::new()));
        let a = actions.clone();
        let on_click = Closure::<dyn FnMut(web_sys::Event)>::new(move |ev: web_sys::Event| {
            let target = ev.target().and_then(|t| t.dyn_into::<Element>().ok());
            if let Some(el) = target.and_then(|t| t.closest("[data-action]").ok().flatten()) {
                if let Some(name) = el.get_attribute("data-action") {
                    a.borrow_mut().push(name);
                }
            }
        });
        root.add_event_listener_with_callback("click", on_click.as_ref().unchecked_ref())?;
        on_click.forget();
        Ok(Ui {
            doc: doc.clone(),
            root,
            hud,
            slots: HashMap::new(),
            labels: HashMap::new(),
            actions,
            stick: None,
        })
    }

    /// Sets the inner HTML of HUD slot `slot` (created on first use).
    pub fn set(&mut self, slot: &str, html: &str) {
        if let Some((el, cache)) = self.slots.get_mut(slot) {
            if cache != html {
                el.set_inner_html(html);
                *cache = html.to_string();
            }
            return;
        }
        let existing = self
            .root
            .query_selector(&format!("[data-slot=\"{slot}\"]"))
            .ok()
            .flatten();
        let el: HtmlElement = match existing {
            Some(e) => e.unchecked_into(),
            None => {
                let e: HtmlElement = self.doc.create_element("div").unwrap().unchecked_into();
                let _ = e.set_attribute("data-slot", slot);
                let _ = self.hud.append_child(&e);
                e
            }
        };
        el.set_inner_html(html);
        self.slots.insert(slot.to_string(), (el, html.to_string()));
    }

    /// Clicked `data-action` names since the last call.
    pub fn take_actions(&mut self) -> Vec<String> {
        std::mem::take(&mut *self.actions.borrow_mut())
    }

    /// Queues an action as if clicked (used by the AI bridge).
    pub fn push_action(&mut self, name: &str) {
        self.actions.borrow_mut().push(name.to_string());
    }

    /// Snapshot of HUD slot contents (for AI inspection).
    pub fn slots(&self) -> HashMap<String, String> {
        self.slots
            .iter()
            .map(|(k, (_, v))| (k.clone(), v.clone()))
            .collect()
    }

    pub(crate) fn sync_labels(&mut self, world: &World, view_proj: Mat4, css_w: f32, css_h: f32) {
        let mut seen: Vec<Entity> = vec![];
        for (e, label, g) in world
            .query::<(Entity, &WorldLabel, &GlobalTransform)>()
            .iter()
        {
            seen.push(e);
            let entry = self.labels.entry(e).or_insert_with(|| {
                let el: HtmlElement = self.doc.create_element("div").unwrap().unchecked_into();
                let _ = self.root.append_child(&el);
                (el, String::from("\u{0}"))
            });
            let class = format!("world-label {}", label.class);
            if entry.0.class_name() != class {
                entry.0.set_class_name(&class);
            }
            if entry.1 != label.html {
                entry.0.set_inner_html(&label.html);
                entry.1 = label.html.clone();
            }
            let p: Vec3 = g.translation() + label.offset;
            let clip = view_proj * p.extend(1.0);
            let style = entry.0.style();
            if clip.w <= 0.01 {
                let _ = style.set_property("display", "none");
                continue;
            }
            let ndc = clip.truncate() / clip.w;
            let x = (ndc.x * 0.5 + 0.5) * css_w;
            let y = (1.0 - (ndc.y * 0.5 + 0.5)) * css_h;
            let _ = style.set_property("display", "block");
            let _ = style.set_property(
                "transform",
                &format!("translate({x:.1}px, {y:.1}px) translate(-50%, -100%)"),
            );
        }
        self.labels.retain(|e, (el, _)| {
            let keep = seen.contains(e);
            if !keep {
                el.remove();
            }
            keep
        });
    }

    /// Draws the virtual joystick (ring at the touch start, knob under the finger).
    pub(crate) fn sync_stick(&mut self, stick: Option<&crate::touch::Touch>) {
        if self.stick.is_none() {
            if stick.is_none() {
                return;
            }
            let mk = |class: &str| -> Option<HtmlElement> {
                let e: HtmlElement = self.doc.create_element("div").ok()?.unchecked_into();
                e.set_class_name(class);
                self.root.append_child(&e).ok()?;
                Some(e)
            };
            let (Some(ring), Some(knob)) = (mk("touch-stick"), mk("touch-knob")) else { return };
            self.stick = Some((ring, knob));
        }
        let Some((ring, knob)) = &self.stick else { return };
        match stick {
            Some(t) => {
                let r = crate::touch::STICK_RADIUS;
                let d = t.pos - t.start;
                let d = if d.length() > r { d.normalize() * r } else { d };
                let _ = ring.style().set_property("display", "block");
                let _ = knob.style().set_property("display", "block");
                let _ = ring.style().set_property("transform", &format!("translate({:.1}px, {:.1}px) translate(-50%, -50%)", t.start.x, t.start.y));
                let _ = knob.style().set_property("transform", &format!("translate({:.1}px, {:.1}px) translate(-50%, -50%)", t.start.x + d.x, t.start.y + d.y));
            }
            None => {
                let _ = ring.style().set_property("display", "none");
                let _ = knob.style().set_property("display", "none");
            }
        }
    }

    pub fn label_count(&self) -> usize {
        self.labels.len()
    }
}

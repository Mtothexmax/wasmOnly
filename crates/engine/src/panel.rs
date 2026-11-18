//! World panels: clickable 2D UI on 3D surfaces.
//!
//! A [`Panel`] is a small retained widget list (labels, buttons, bars, rects) in panel pixels.
//! The engine draws it with Canvas2D into an `OffscreenCanvas`, uploads it to a GPU texture and
//! shows it on a quad (XY plane, facing +Z; the entity's scale = size in world units).
//!
//! Input: every frame the engine casts the pointer ray (screen centre under pointer lock)
//! against all panels, converts the hit to panel pixels via the quad's UV and hit-tests the
//! widgets. A left click on a button queues its `action` in the same queue as DOM HUD buttons
//! (`ctx.ui.take_actions()`), and the click is consumed (`input.button_pressed(Left)` is false).
//! Physics colliders in front of a panel occlude it.
//!
//! AI: `engine.call("panels")` lists every panel, widget rect and its on-screen position;
//! `panel_click {id, widget}` clicks through the same code path.

use crate::physics::Physics;
use crate::renderer::{Material, MaterialId, Renderer, TextureId};
use crate::scene::{GlobalTransform, Name, Parent};
use crate::{log_info, log_warn};
use glam::{Mat4, Vec2, Vec3};
use hecs::{Entity, World};
use serde::Serialize;
use wasm_bindgen::JsCast;
use web_sys::{OffscreenCanvas, OffscreenCanvasRenderingContext2d};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct PanelId(pub u32);

/// Component on the entity that shows a panel.
#[derive(Clone, Copy, Debug)]
pub struct WorldPanel(pub PanelId);

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum WidgetKind {
    Label {
        size: f32,
        color: String,
        align: String,
    },
    Button {
        action: String,
        color: String,
        bg: String,
        hover_bg: String,
    },
    Bar {
        value: f32,
        color: String,
        bg: String,
    },
    Rect {
        color: String,
        radius: f32,
    },
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Widget {
    pub id: String,
    /// x, y, w, h in panel pixels (origin top-left).
    pub rect: [f32; 4],
    pub text: String,
    pub visible: bool,
    #[serde(flatten)]
    pub kind: WidgetKind,
}

/// Retained 2D UI description. Build with the chained helpers, change later via
/// `ctx.panels.get_mut(id)` (marks it for redraw).
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Panel {
    pub width: u32,
    pub height: u32,
    pub background: String,
    pub radius: f32,
    pub widgets: Vec<Widget>,
}

impl Panel {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            background: "#1e293b".into(),
            radius: 24.0,
            widgets: vec![],
        }
    }
    pub fn background(mut self, css: &str) -> Self {
        self.background = css.into();
        self
    }
    fn push(mut self, id: &str, rect: [f32; 4], text: &str, kind: WidgetKind) -> Self {
        self.widgets.push(Widget {
            id: id.into(),
            rect,
            text: text.into(),
            visible: true,
            kind,
        });
        self
    }
    /// Text. `size` in px; `\n` starts a new line.
    pub fn label(self, id: &str, rect: [f32; 4], text: &str, size: f32) -> Self {
        self.push(
            id,
            rect,
            text,
            WidgetKind::Label {
                size,
                color: "#e2e8f0".into(),
                align: "left".into(),
            },
        )
    }
    /// Clickable button; a click queues `action` (read it with `ctx.ui.take_actions()`).
    pub fn button(self, id: &str, rect: [f32; 4], text: &str, action: &str) -> Self {
        self.push(
            id,
            rect,
            text,
            WidgetKind::Button {
                action: action.into(),
                color: "#111827".into(),
                bg: "#fbbf24".into(),
                hover_bg: "#fde68a".into(),
            },
        )
    }
    /// Progress bar, `value` in 0..1.
    pub fn bar(self, id: &str, rect: [f32; 4], value: f32, color: &str) -> Self {
        self.push(
            id,
            rect,
            "",
            WidgetKind::Bar {
                value,
                color: color.into(),
                bg: "#334155".into(),
            },
        )
    }
    pub fn rect(self, id: &str, rect: [f32; 4], color: &str) -> Self {
        self.push(
            id,
            rect,
            "",
            WidgetKind::Rect {
                color: color.into(),
                radius: 12.0,
            },
        )
    }

    pub fn widget_mut(&mut self, id: &str) -> Option<&mut Widget> {
        self.widgets.iter_mut().find(|w| w.id == id)
    }
    pub fn widget(&self, id: &str) -> Option<&Widget> {
        self.widgets.iter().find(|w| w.id == id)
    }
    pub fn set_text(&mut self, id: &str, text: &str) {
        if let Some(w) = self.widget_mut(id) {
            if w.text != text {
                w.text = text.into();
            }
        }
    }
    pub fn set_value(&mut self, id: &str, v: f32) {
        if let Some(Widget {
            kind: WidgetKind::Bar { value, .. },
            ..
        }) = self.widget_mut(id)
        {
            *value = v.clamp(0.0, 1.0);
        }
    }
    pub fn set_visible(&mut self, id: &str, visible: bool) {
        if let Some(w) = self.widget_mut(id) {
            w.visible = visible;
        }
    }
    /// Topmost visible button containing the panel-pixel point.
    pub fn button_at(&self, p: Vec2) -> Option<&Widget> {
        self.widgets.iter().rev().find(|w| {
            w.visible
                && matches!(w.kind, WidgetKind::Button { .. })
                && p.x >= w.rect[0]
                && p.y >= w.rect[1]
                && p.x <= w.rect[0] + w.rect[2]
                && p.y <= w.rect[1] + w.rect[3]
        })
    }
}

struct Slot {
    entity: Entity,
    panel: Panel,
    drawn: Option<Panel>,
    drawn_hover: Option<String>,
    hover: Option<String>,
    canvas: OffscreenCanvas,
    ctx2d: OffscreenCanvasRenderingContext2d,
    texture: TextureId,
    material: MaterialId,
    clicks: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct PanelClick {
    pub panel: u32,
    pub widget: String,
    pub action: String,
    pub frame: u64,
}

#[derive(Default)]
pub struct Panels {
    slots: Vec<Slot>,
    /// Most recent clicks (newest last, max 32) — for AI inspection.
    pub history: Vec<PanelClick>,
    /// Panel + panel-pixel point under the pointer this frame.
    pub hovered: Option<(PanelId, Vec2)>,
}

impl Panels {
    pub(crate) fn create(
        &mut self,
        renderer: &mut Renderer,
        entity: Entity,
        panel: Panel,
    ) -> Result<(PanelId, MaterialId), String> {
        let canvas =
            OffscreenCanvas::new(panel.width, panel.height).map_err(|e| format!("{e:?}"))?;
        let ctx2d: OffscreenCanvasRenderingContext2d = canvas
            .get_context("2d")
            .map_err(|e| format!("{e:?}"))?
            .ok_or("OffscreenCanvas 2d context unavailable")?
            .unchecked_into();
        let texture = renderer.create_texture(panel.width, panel.height, "panel")?;
        let material = renderer.add_material(Material {
            name: "panel".into(),
            texture: Some(texture),
            ui: true,
            ..Default::default()
        });
        let id = PanelId(self.slots.len() as u32);
        self.slots.push(Slot {
            entity,
            panel,
            drawn: None,
            drawn_hover: None,
            hover: None,
            canvas,
            ctx2d,
            texture,
            material,
            clicks: 0,
        });
        Ok((id, material))
    }

    pub fn get(&self, id: PanelId) -> Option<&Panel> {
        self.slots.get(id.0 as usize).map(|s| &s.panel)
    }
    /// Mutable access; the panel is redrawn at the end of the frame if anything changed.
    pub fn get_mut(&mut self, id: PanelId) -> Option<&mut Panel> {
        self.slots.get_mut(id.0 as usize).map(|s| &mut s.panel)
    }
    pub fn entity(&self, id: PanelId) -> Option<Entity> {
        self.slots.get(id.0 as usize).map(|s| s.entity)
    }
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Ray vs. every panel quad. Returns (panel, panel-pixel point, distance).
    pub fn raycast(
        &self,
        world: &World,
        physics: &Physics,
        origin: Vec3,
        dir: Vec3,
    ) -> Option<(PanelId, Vec2, f32)> {
        let mut best: Option<(PanelId, Vec2, f32)> = None;
        for (i, s) in self.slots.iter().enumerate() {
            let Ok(g) = world.get::<&GlobalTransform>(s.entity) else {
                continue;
            };
            let inv: Mat4 = g.0.inverse();
            let o = inv.transform_point3(origin);
            let d = inv.transform_vector3(dir);
            if d.z.abs() < 1e-6 || o.z <= 0.0 {
                continue; // parallel, or looking at the back side
            }
            let t_local = -o.z / d.z;
            if t_local <= 0.0 {
                continue;
            }
            let p = o + d * t_local;
            if p.x.abs() > 0.5 || p.y.abs() > 0.5 {
                continue;
            }
            let world_hit = g.0.transform_point3(p);
            let t = (world_hit - origin).length();
            if best.as_ref().is_some_and(|b| b.2 <= t) {
                continue;
            }
            let px = Vec2::new(
                (p.x + 0.5) * s.panel.width as f32,
                (0.5 - p.y) * s.panel.height as f32,
            );
            best = Some((PanelId(i as u32), px, t));
        }
        let (id, px, t) = best?;
        // Occlusion by physics geometry in front of the panel (ignore the panel's own hierarchy).
        if let Some(h) = physics.raycast(origin, dir, t) {
            let own = self.slots[id.0 as usize].entity;
            let is_own = h
                .entity
                .is_some_and(|e| e == own || is_descendant(world, e, own));
            if !is_own && h.distance < t - 0.02 {
                return None;
            }
        }
        Some((id, px, t))
    }

    /// Hover + click handling for this frame. Returns true if a click hit a panel button.
    pub(crate) fn pointer(
        &mut self,
        world: &World,
        physics: &Physics,
        ray: Option<(Vec3, Vec3)>,
        clicked: bool,
        frame: u64,
    ) -> Option<PanelClick> {
        let hit = ray.and_then(|(o, d)| self.raycast(world, physics, o, d));
        self.hovered = hit.map(|(id, p, _)| (id, p));
        for (i, s) in self.slots.iter_mut().enumerate() {
            s.hover = match hit {
                Some((id, p, _)) if id.0 as usize == i => {
                    s.panel.button_at(p).map(|w| w.id.clone())
                }
                _ => None,
            };
        }
        let (id, p, _) = hit?;
        if !clicked {
            return None;
        }
        self.click_at(id, p, frame)
    }

    /// Click at a panel-pixel point (shared by pointer input and `panel_click`).
    pub fn click_at(&mut self, id: PanelId, p: Vec2, frame: u64) -> Option<PanelClick> {
        let s = self.slots.get_mut(id.0 as usize)?;
        let w = s.panel.button_at(p)?;
        let WidgetKind::Button { action, .. } = &w.kind else {
            return None;
        };
        let click = PanelClick {
            panel: id.0,
            widget: w.id.clone(),
            action: action.clone(),
            frame,
        };
        s.clicks += 1;
        log_info!(
            "panel {} click '{}' -> action '{}'",
            id.0,
            click.widget,
            click.action
        );
        self.history.push(click.clone());
        if self.history.len() > 32 {
            self.history.remove(0);
        }
        Some(click)
    }

    /// Redraws changed panels and uploads them.
    pub(crate) fn flush(&mut self, renderer: &mut Renderer) {
        for s in &mut self.slots {
            if s.drawn.as_ref() == Some(&s.panel) && s.drawn_hover == s.hover {
                continue;
            }
            draw(&s.ctx2d, &s.panel, s.hover.as_deref());
            match s.canvas.transfer_to_image_bitmap() {
                Ok(bmp) => {
                    if let Err(e) = renderer.update_texture(s.texture, &bmp) {
                        log_warn!("panel upload: {e}");
                    }
                    bmp.close();
                }
                Err(e) => log_warn!("panel transfer: {e:?}"),
            }
            s.drawn = Some(s.panel.clone());
            s.drawn_hover = s.hover.clone();
        }
    }

    /// AI snapshot: panels, widgets, hover, on-screen widget centres (CSS px, if visible).
    pub fn describe(&self, world: &World, view_proj: Mat4, css: Vec2) -> serde_json::Value {
        let list: Vec<serde_json::Value> = self
            .slots
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let g = world
                    .get::<&GlobalTransform>(s.entity)
                    .map(|g| g.0)
                    .unwrap_or(Mat4::IDENTITY);
                let widgets: Vec<serde_json::Value> = s
                    .panel
                    .widgets
                    .iter()
                    .map(|w| {
                        let c = Vec2::new(w.rect[0] + w.rect[2] * 0.5, w.rect[1] + w.rect[3] * 0.5);
                        let local = Vec3::new(
                            c.x / s.panel.width as f32 - 0.5,
                            0.5 - c.y / s.panel.height as f32,
                            0.0,
                        );
                        let clip = view_proj * g.transform_point3(local).extend(1.0);
                        let screen = (clip.w > 0.01).then(|| {
                            let n = clip.truncate() / clip.w;
                            [
                                ((n.x * 0.5 + 0.5) * css.x).round(),
                                ((0.5 - n.y * 0.5) * css.y).round(),
                            ]
                        });
                        let mut v = serde_json::to_value(w).unwrap_or_default();
                        v["screen"] = serde_json::json!(screen);
                        v
                    })
                    .collect();
                serde_json::json!({
                    "id": i, "entity": s.entity.id(),
                    "name": world.get::<&Name>(s.entity).map(|n| n.0.clone()).unwrap_or_default(),
                    "size": [s.panel.width, s.panel.height], "hover": s.hover, "clicks": s.clicks,
                    "texture": s.texture.0, "material": s.material.0, "widgets": widgets,
                })
            })
            .collect();
        serde_json::json!({"panels": list, "history": self.history,
            "hovered": self.hovered.map(|(id, p)| serde_json::json!({"panel": id.0, "px": [p.x.round(), p.y.round()]}))})
    }
}

fn is_descendant(world: &World, mut e: Entity, root: Entity) -> bool {
    for _ in 0..64 {
        match world.get::<&Parent>(e) {
            Ok(p) if p.0 == root => return true,
            Ok(p) => e = p.0,
            Err(_) => return false,
        }
    }
    false
}

fn rounded(c: &OffscreenCanvasRenderingContext2d, r: [f32; 4], radius: f32) {
    c.begin_path();
    let _ = c.round_rect_with_f64(
        r[0] as f64,
        r[1] as f64,
        r[2] as f64,
        r[3] as f64,
        radius as f64,
    );
}

fn draw(c: &OffscreenCanvasRenderingContext2d, p: &Panel, hover: Option<&str>) {
    let (w, h) = (p.width as f64, p.height as f64);
    c.clear_rect(0.0, 0.0, w, h);
    c.set_fill_style_str(&p.background);
    rounded(c, [0.0, 0.0, p.width as f32, p.height as f32], p.radius);
    c.fill();
    for wd in p.widgets.iter().filter(|w| w.visible) {
        let [x, y, ww, hh] = wd.rect;
        match &wd.kind {
            WidgetKind::Rect { color, radius } => {
                c.set_fill_style_str(color);
                rounded(c, wd.rect, *radius);
                c.fill();
            }
            WidgetKind::Bar { value, color, bg } => {
                c.set_fill_style_str(bg);
                rounded(c, wd.rect, hh * 0.5);
                c.fill();
                if *value > 0.0 {
                    c.set_fill_style_str(color);
                    rounded(c, [x, y, (ww * value).max(hh), hh], hh * 0.5);
                    c.fill();
                }
            }
            WidgetKind::Label { size, color, align } => {
                c.set_fill_style_str(color);
                c.set_font(&format!("600 {size}px system-ui, 'Segoe UI', sans-serif"));
                c.set_text_baseline("top");
                c.set_text_align(align);
                let tx = match align.as_str() {
                    "center" => x + ww * 0.5,
                    "right" => x + ww,
                    _ => x,
                };
                for (i, line) in wd.text.lines().enumerate() {
                    let _ = c.fill_text_with_max_width(
                        line,
                        tx as f64,
                        (y + i as f32 * size * 1.25) as f64,
                        ww as f64,
                    );
                }
            }
            WidgetKind::Button {
                color,
                bg,
                hover_bg,
                ..
            } => {
                let hot = hover == Some(wd.id.as_str());
                c.set_fill_style_str(if hot { hover_bg } else { bg });
                rounded(c, wd.rect, 14.0);
                c.fill();
                c.set_fill_style_str(color);
                let size = (hh * 0.42).round();
                c.set_font(&format!("700 {size}px system-ui, 'Segoe UI', sans-serif"));
                c.set_text_align("center");
                c.set_text_baseline("middle");
                let _ = c.fill_text_with_max_width(
                    &wd.text,
                    (x + ww * 0.5) as f64,
                    (y + hh * 0.5) as f64,
                    (ww - 8.0) as f64,
                );
            }
        }
    }
}

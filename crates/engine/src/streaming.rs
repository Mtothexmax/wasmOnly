//! World streaming and floating origin.
//!
//! **Streaming layers**: the world is cut into square cells; a layer's generator builds a cell's
//! content when the camera comes within `load_radius` and the cell's scene is unloaded beyond
//! `unload_radius` (hysteresis). Each loaded cell is its own scene (`stream:<layer>:<x>:<z>`), so
//! everything spawned in the generator — entities, bodies, static colliders, particles, glTF that
//! finishes loading later — goes away with it. Cells load nearest first, `budget` per frame.
//!
//! ```ignore
//! ctx.add_stream_layer("props", 128.0, 400.0, |ctx, cell| {
//!     let min = cell.min;                                   // absolute world XZ of the cell corner
//!     for k in 0..20 { /* spawn at ctx.local(DVec3::new(...)) */ }
//!     // big static content: ctx.batch_static(root, 64.0)
//! });
//! ```
//!
//! **Floating origin**: positions stay f32 near the camera; `ctx.origin` (f64) is where the local
//! origin sits in the world. When the camera gets farther than `origin_shift_distance` from it,
//! everything is moved back (`ctx.shift_origin`) — transforms of root entities, physics bodies
//! and static colliders, particles, motion bases, the third-person camera. Use `ctx.local(abs)` /
//! `ctx.to_absolute(local)` to convert; terrain, streaming and grass work in absolute coordinates.
//!
//! AI API: `world` (origin, layers with loaded cells and counters), `{shift: [x,y,z]}`,
//! `{origin_shift_distance: m}`, `{layer, radius?}`.

use crate::app::Ctx;
use glam::{DVec3, Vec3};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::rc::Rc;

/// A streaming cell: integer coordinates + absolute bounds.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Cell {
    pub x: i32,
    pub z: i32,
    /// Absolute world position of the cell's minimum corner (y = 0).
    pub min: DVec3,
    pub size: f64,
}

impl Cell {
    pub fn center(&self) -> DVec3 {
        self.min + DVec3::new(self.size * 0.5, 0.0, self.size * 0.5)
    }
    /// Deterministic seed for procedural content of this cell.
    pub fn seed(&self, salt: u32) -> u64 {
        let h = (self.x as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (self.z as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ salt as u64;
        h ^ (h >> 29)
    }
}

type CellFn = Rc<dyn Fn(&mut Ctx, Cell)>;

pub struct StreamLayer {
    pub name: String,
    pub cell: f64,
    pub load_radius: f64,
    pub unload_radius: f64,
    /// Cells built per frame at most.
    pub budget: u32,
    pub enabled: bool,
    generator: CellFn,
    loaded: HashMap<(i32, i32), String>,
    pub loads: u64,
    pub unloads: u64,
    pub last_build_ms: f32,
}

pub struct Streaming {
    pub layers: Vec<StreamLayer>,
    /// Shift the origin when the camera is farther than this from it (m, 0 = never).
    pub origin_shift_distance: f32,
    /// Shifts snap to multiples of this (m), keeping coordinates tidy.
    pub origin_shift_snap: f64,
    pub shifts: u32,
}

impl Default for Streaming {
    fn default() -> Self {
        Self { layers: vec![], origin_shift_distance: 1500.0, origin_shift_snap: 256.0, shifts: 0 }
    }
}

impl Ctx {
    /// Local (f32, render/physics) position of an absolute world position.
    pub fn local(&self, abs: DVec3) -> Vec3 {
        (abs - self.origin).as_vec3()
    }

    /// Registers a streaming layer (see module docs). Replaces a layer with the same name.
    pub fn add_stream_layer(&mut self, name: &str, cell: f32, load_radius: f32, generator: impl Fn(&mut Ctx, Cell) + 'static) {
        self.remove_stream_layer(name);
        self.streaming.layers.push(StreamLayer {
            name: name.to_string(),
            cell: cell.max(1.0) as f64,
            load_radius: load_radius as f64,
            unload_radius: (load_radius * 1.2 + cell * 0.5) as f64,
            budget: 2,
            enabled: true,
            generator: Rc::new(generator),
            loaded: HashMap::new(),
            loads: 0,
            unloads: 0,
            last_build_ms: 0.0,
        });
    }

    /// Unloads every cell of a layer and removes it.
    pub fn remove_stream_layer(&mut self, name: &str) {
        if let Some(i) = self.streaming.layers.iter().position(|l| l.name == name) {
            let l = self.streaming.layers.remove(i);
            for (_, scene) in l.loaded {
                self.unload_scene(&scene);
            }
        }
    }

    /// Absolute camera position (main camera).
    pub fn camera_absolute(&self) -> Option<DVec3> {
        crate::scene::main_camera_pose(&self.world).map(|g| self.origin + g.w_axis.truncate().as_dvec3())
    }

    /// Per frame: floating origin, then streaming layers.
    pub(crate) fn update_streaming(&mut self) {
        let Some(cam) = self.camera_absolute() else { return };
        // floating origin
        let local = cam - self.origin;
        let d = self.streaming.origin_shift_distance as f64;
        if d > 0.0 && (local.x * local.x + local.z * local.z).sqrt() > d {
            let snap = self.streaming.origin_shift_snap.max(1.0);
            let delta = DVec3::new((local.x / snap).round() * snap, 0.0, (local.z / snap).round() * snap);
            self.shift_origin(delta);
        }
        for li in 0..self.streaming.layers.len() {
            let (name, cell, lr, ur, budget, enabled, gen) = {
                let l = &self.streaming.layers[li];
                (l.name.clone(), l.cell, l.load_radius, l.unload_radius, l.budget, l.enabled, l.generator.clone())
            };
            if !enabled {
                continue;
            }
            let (cx0, cx1) = (((cam.x - lr) / cell).floor() as i32, ((cam.x + lr) / cell).floor() as i32);
            let (cz0, cz1) = (((cam.z - lr) / cell).floor() as i32, ((cam.z + lr) / cell).floor() as i32);
            let dist = |x: i32, z: i32| {
                let c = DVec3::new((x as f64 + 0.5) * cell, 0.0, (z as f64 + 0.5) * cell);
                ((c.x - cam.x).powi(2) + (c.z - cam.z).powi(2)).sqrt()
            };
            let mut want = vec![];
            for z in cz0..=cz1 {
                for x in cx0..=cx1 {
                    let dd = dist(x, z);
                    if dd <= lr && !self.streaming.layers[li].loaded.contains_key(&(x, z)) {
                        want.push((dd, x, z));
                    }
                }
            }
            want.sort_by(|a, b| a.0.total_cmp(&b.0));
            let drop: Vec<((i32, i32), String)> = self.streaming.layers[li].loaded.iter().filter(|(k, _)| dist(k.0, k.1) > ur).map(|(k, s)| (*k, s.clone())).collect();
            for (k, scene) in drop {
                self.unload_scene(&scene);
                let l = &mut self.streaming.layers[li];
                l.loaded.remove(&k);
                l.unloads += 1;
            }
            for (_, x, z) in want.into_iter().take(budget.max(1) as usize) {
                let scene = format!("stream:{name}:{x}:{z}");
                let prev = self.scenes.current().to_string();
                self.begin_scene(&scene);
                let t0 = crate::log::now_ms();
                gen(self, Cell { x, z, min: DVec3::new(x as f64 * cell, 0.0, z as f64 * cell), size: cell });
                let ms = (crate::log::now_ms() - t0) as f32;
                self.begin_scene(&prev);
                let l = &mut self.streaming.layers[li];
                l.loaded.insert((x, z), scene);
                l.loads += 1;
                l.last_build_ms = ms;
            }
        }
    }

    /// Moves the local origin by `delta` (absolute): everything in the world is translated by
    /// `-delta` so it stays where it is in absolute terms. Positions near the camera keep full f32
    /// precision however far the camera travels.
    pub fn shift_origin(&mut self, delta: DVec3) {
        let d = delta.as_vec3();
        if d == Vec3::ZERO {
            return;
        }
        self.origin += delta;
        for (t, m) in self.world.query_mut::<hecs::Without<(&mut crate::scene::Transform, Option<&mut crate::batch::Motion>), &crate::scene::Parent>>() {
            t.translation -= d;
            if let Some(m) = m {
                m.shift(d);
            }
        }
        let pw = &mut self.physics.world;
        for (_, b) in pw.bodies.iter_mut() {
            let p = b.translation();
            b.set_translation(p - d, false);
            if b.is_kinematic() {
                let mut next = *b.position();
                next.translation -= d;
                b.set_next_kinematic_position(next);
            }
        }
        for (_, c) in pw.colliders.iter_mut() {
            if c.parent().is_none() {
                let p = c.translation();
                c.set_translation(p - d);
            }
        }
        for ps in self.world.query_mut::<&mut crate::particles::ParticleSystem>() {
            ps.shift(d);
        }
        for c in self.world.query_mut::<&mut crate::character::ThirdPersonCamera>() {
            if let Some(p) = c.pivot.as_mut() {
                *p -= d;
            }
        }
        self.renderer.invalidate_history();
        crate::scene::propagate_transforms(&mut self.world);
        self.streaming.shifts += 1;
        crate::log_info!("origin shifted by [{:.0}, {:.0}, {:.0}] -> origin [{:.0}, {:.0}, {:.0}]", delta.x, delta.y, delta.z, self.origin.x, self.origin.y, self.origin.z);
    }

    /// AI API `world`.
    pub(crate) fn world_command(&mut self, a: &Value) -> Result<Value, String> {
        if let Some(v) = a["shift"].as_array() {
            let f = |i: usize| v.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0);
            self.shift_origin(DVec3::new(f(0), f(1), f(2)));
        }
        if let Some(d) = a["origin_shift_distance"].as_f64() {
            self.streaming.origin_shift_distance = d as f32;
        }
        if let Some(name) = a["layer"].as_str() {
            let l = self.streaming.layers.iter_mut().find(|l| l.name == name).ok_or_else(|| format!("no stream layer '{name}'"))?;
            if let Some(r) = a["radius"].as_f64() {
                l.load_radius = r;
                l.unload_radius = r * 1.2 + l.cell * 0.5;
            }
            if let Some(e) = a["enabled"].as_bool() {
                l.enabled = e;
            }
        }
        let cam = self.camera_absolute();
        let layers: Vec<Value> = self
            .streaming
            .layers
            .iter()
            .map(|l| {
                let mut cells: Vec<[i32; 2]> = l.loaded.keys().map(|k| [k.0, k.1]).collect();
                cells.sort();
                json!({"name": l.name, "cell": l.cell, "load_radius": l.load_radius, "unload_radius": l.unload_radius, "enabled": l.enabled,
                       "loaded": cells.len(), "cells": cells, "loads": l.loads, "unloads": l.unloads, "last_build_ms": l.last_build_ms})
            })
            .collect();
        Ok(json!({
            "origin": [self.origin.x, self.origin.y, self.origin.z],
            "camera_absolute": cam.map(|c| [c.x, c.y, c.z]),
            "origin_shift_distance": self.streaming.origin_shift_distance,
            "shifts": self.streaming.shifts,
            "layers": layers,
            "entities": self.world.len(),
            "meshes": self.renderer.mesh_count(),
            "terrain_chunks": self.terrain.as_ref().map(|t| t.chunks.len()),
        }))
    }
}

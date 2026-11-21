//! Particle systems: CPU-simulated in the fixed step (deterministic: own seeded RNG per emitter),
//! drawn as camera-facing quads in the blend pass (instanced, one draw per blend mode). Additive
//! particles with HDR colours (> 1) glow through bloom.
//!
//! ```ignore
//! ctx.spawn_particles("fire", Vec3::new(0., 0.2, 0.), ParticleEmitter::preset(ParticlePreset::Fire));
//! let burst = ParticleEmitter { burst: 80, rate: 0.0, one_shot: true, ..ParticleEmitter::preset(ParticlePreset::Sparks) };
//! ctx.spawn_particles("hit", point, burst);                 // despawns itself when done
//! ```
//!
//! Seeds come from `ctx.particle_rng`, not `ctx.rng`: adding an emitter never shifts the game's
//! random rolls. `spawn_particles_seeded` fixes the seed explicitly. `ParticleEmitter::layer` is the
//! render layer (a camera draws it when bit `layer` is set in `Camera::layers`).
//! One-shot presets (`Splash`) set `one_shot: true`; a continuous variant needs `one_shot: false`
//! as well as a `rate`, or it despawns once its first particles die.
//!
//! AI API: `particles` lists emitters with live counts; `particles {preset, pos}` spawns one.

use crate::app::{Ctx, Rng};
use crate::scene::{GlobalTransform, Name, Transform};
use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3, Vec4};
use hecs::{Entity, World};
use serde::Serialize;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ParticleRaw {
    pub pos_size: [f32; 4],
    pub color: [f32; 4],
    pub params: [f32; 4],
    pub vel: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ParticleShape {
    /// Soft round dot.
    Dot,
    Square,
    /// Stretched along the velocity (sparks, rain).
    Spark,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ParticlePreset {
    Fire,
    Smoke,
    Sparks,
    Fireflies,
    Dust,
    Magic,
    Rain,
    Splash,
}

impl ParticlePreset {
    pub const ALL: [ParticlePreset; 8] = [Self::Fire, Self::Smoke, Self::Sparks, Self::Fireflies, Self::Dust, Self::Magic, Self::Rain, Self::Splash];
    pub fn name(self) -> &'static str {
        match self {
            Self::Fire => "fire",
            Self::Smoke => "smoke",
            Self::Sparks => "sparks",
            Self::Fireflies => "fireflies",
            Self::Dust => "dust",
            Self::Magic => "magic",
            Self::Rain => "rain",
            Self::Splash => "splash",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == s)
    }
}

/// Emitter settings. Colours are linear RGB (values > 1 glow with bloom when `additive`).
#[derive(Clone, Debug, Serialize)]
pub struct ParticleEmitter {
    pub enabled: bool,
    /// Particles per second.
    pub rate: f32,
    /// Particles emitted at once on the first step.
    pub burst: u32,
    pub max: u32,
    /// Seconds (min, max).
    pub lifetime: (f32, f32),
    /// Initial speed (min, max) along `direction` within the cone `spread` (radians).
    pub speed: (f32, f32),
    pub direction: Vec3,
    pub spread: f32,
    /// Spawn inside a box of these half extents around the emitter.
    pub area: Vec3,
    pub gravity: Vec3,
    /// Velocity damping per second.
    pub drag: f32,
    /// Random velocity change per second (flicker, swirl).
    pub turbulence: f32,
    /// Size at birth and death (m).
    pub size: (f32, f32),
    pub color_start: Vec4,
    pub color_end: Vec4,
    /// Opacity fade-in fraction of the lifetime.
    pub fade_in: f32,
    pub additive: bool,
    pub shape: ParticleShape,
    /// Spark stretch per m/s.
    pub stretch: f32,
    /// Spin speed range (rad/s).
    pub spin: f32,
    /// Despawn the entity once nothing is left to emit and all particles died.
    pub one_shot: bool,
    /// Particles move with the emitter (true) or stay where they were emitted (false).
    pub local_space: bool,
    /// Render layer 0..31 (like `MeshRenderer::layer`): drawn only by cameras with bit `layer` in `layers`.
    pub layer: u8,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self::preset(ParticlePreset::Dust)
    }
}

impl ParticleEmitter {
    pub fn preset(p: ParticlePreset) -> Self {
        let base = Self {
            enabled: true,
            rate: 20.0,
            burst: 0,
            max: 400,
            lifetime: (1.0, 2.0),
            speed: (0.5, 1.0),
            direction: Vec3::Y,
            spread: 0.5,
            area: Vec3::ZERO,
            gravity: Vec3::ZERO,
            drag: 0.5,
            turbulence: 0.0,
            size: (0.2, 0.2),
            color_start: Vec4::ONE,
            color_end: Vec4::new(1.0, 1.0, 1.0, 0.0),
            fade_in: 0.1,
            additive: false,
            shape: ParticleShape::Dot,
            stretch: 0.0,
            spin: 0.0,
            one_shot: false,
            local_space: false,
            layer: 0,
        };
        match p {
            ParticlePreset::Fire => Self {
                rate: 35.0,
                lifetime: (0.5, 1.0),
                speed: (0.8, 1.6),
                spread: 0.35,
                area: Vec3::new(0.15, 0.0, 0.15),
                gravity: Vec3::new(0.0, 1.5, 0.0),
                drag: 1.5,
                turbulence: 2.0,
                size: (0.45, 0.1),
                color_start: Vec4::new(1.4, 0.55, 0.12, 0.7),
                color_end: Vec4::new(0.5, 0.06, 0.01, 0.0),
                additive: true,
                spin: 2.0,
                ..base
            },
            ParticlePreset::Smoke => Self {
                rate: 10.0,
                lifetime: (2.5, 4.0),
                speed: (0.4, 0.8),
                spread: 0.3,
                area: Vec3::splat(0.2),
                gravity: Vec3::new(0.3, 0.4, 0.0),
                drag: 0.6,
                turbulence: 0.4,
                size: (0.5, 2.2),
                color_start: Vec4::new(0.35, 0.33, 0.32, 0.55),
                color_end: Vec4::new(0.6, 0.6, 0.62, 0.0),
                fade_in: 0.25,
                spin: 0.6,
                ..base
            },
            ParticlePreset::Sparks => Self {
                rate: 25.0,
                lifetime: (0.4, 0.9),
                speed: (3.0, 7.0),
                spread: 0.8,
                gravity: Vec3::new(0.0, -9.81, 0.0),
                drag: 0.4,
                size: (0.12, 0.05),
                color_start: Vec4::new(6.0, 3.5, 1.2, 1.0),
                color_end: Vec4::new(2.0, 0.4, 0.05, 0.0),
                fade_in: 0.0,
                additive: true,
                shape: ParticleShape::Spark,
                stretch: 0.25,
                ..base
            },
            ParticlePreset::Fireflies => Self {
                rate: 4.0,
                max: 60,
                lifetime: (4.0, 8.0),
                speed: (0.1, 0.3),
                spread: std::f32::consts::PI,
                area: Vec3::new(3.0, 0.8, 3.0),
                drag: 0.3,
                turbulence: 1.2,
                size: (0.09, 0.09),
                color_start: Vec4::new(2.5, 4.0, 0.8, 1.0),
                color_end: Vec4::new(1.5, 3.0, 0.5, 0.0),
                fade_in: 0.3,
                additive: true,
                ..base
            },
            ParticlePreset::Dust => Self {
                rate: 6.0,
                max: 150,
                lifetime: (3.0, 6.0),
                speed: (0.05, 0.15),
                spread: std::f32::consts::PI,
                area: Vec3::new(4.0, 1.5, 4.0),
                drag: 0.2,
                turbulence: 0.15,
                size: (0.03, 0.03),
                color_start: Vec4::new(1.0, 0.95, 0.85, 0.5),
                color_end: Vec4::new(1.0, 0.95, 0.85, 0.0),
                fade_in: 0.4,
                ..base
            },
            ParticlePreset::Magic => Self {
                rate: 40.0,
                lifetime: (0.8, 1.6),
                speed: (0.3, 1.0),
                spread: std::f32::consts::PI,
                area: Vec3::splat(0.3),
                gravity: Vec3::new(0.0, 0.6, 0.0),
                drag: 1.0,
                turbulence: 1.5,
                size: (0.15, 0.0),
                color_start: Vec4::new(1.2, 0.5, 4.0, 1.0),
                color_end: Vec4::new(0.3, 1.5, 3.0, 0.0),
                additive: true,
                ..base
            },
            ParticlePreset::Rain => Self {
                rate: 600.0,
                max: 3000,
                lifetime: (0.8, 1.0),
                speed: (11.0, 13.0),
                direction: Vec3::new(0.08, -1.0, 0.0),
                spread: 0.03,
                area: Vec3::new(14.0, 0.0, 14.0),
                drag: 0.0,
                size: (0.04, 0.04),
                color_start: Vec4::new(0.7, 0.75, 0.85, 0.35),
                color_end: Vec4::new(0.7, 0.75, 0.85, 0.35),
                fade_in: 0.05,
                shape: ParticleShape::Spark,
                stretch: 0.12,
                ..base
            },
            // one-shot burst: `..preset(Splash)` with a rate also needs `one_shot: false`
            ParticlePreset::Splash => Self {
                rate: 0.0,
                burst: 40,
                one_shot: true,
                lifetime: (0.4, 0.8),
                speed: (1.5, 3.5),
                spread: 0.7,
                gravity: Vec3::new(0.0, -9.81, 0.0),
                drag: 0.5,
                size: (0.1, 0.04),
                color_start: Vec4::new(0.8, 0.9, 1.0, 0.8),
                color_end: Vec4::new(0.8, 0.9, 1.0, 0.0),
                fade_in: 0.0,
                ..base
            },
        }
    }

    pub fn burst(mut self, n: u32) -> Self {
        self.burst = n;
        self.rate = 0.0;
        self.one_shot = true;
        self
    }
}

#[derive(Clone, Copy, Debug)]
struct Particle {
    pos: Vec3,
    vel: Vec3,
    age: f32,
    life: f32,
    rot: f32,
    spin: f32,
    seed: f32,
}

/// Component: emitter settings + live particles.
pub struct ParticleSystem {
    pub emitter: ParticleEmitter,
    particles: Vec<Particle>,
    acc: f32,
    burst_done: bool,
    rng: Rng,
    /// Emitter world position last step (local_space: particles follow its movement).
    last_origin: Option<Vec3>,
}

impl ParticleSystem {
    pub fn new(emitter: ParticleEmitter, seed: u64) -> Self {
        Self { emitter, particles: vec![], acc: 0.0, burst_done: false, rng: Rng(seed | 1), last_origin: None }
    }
    pub fn count(&self) -> usize {
        self.particles.len()
    }
    pub fn finished(&self) -> bool {
        self.emitter.one_shot && self.burst_done && self.emitter.rate <= 0.0 && self.particles.is_empty()
    }
    /// Origin shift: live particles and the last emitter position move with the world.
    pub(crate) fn shift(&mut self, d: Vec3) {
        for p in &mut self.particles {
            p.pos -= d;
        }
        if let Some(o) = self.last_origin.as_mut() {
            *o -= d;
        }
    }

    /// Restart a one-shot burst.
    pub fn trigger(&mut self) {
        self.burst_done = false;
    }

    fn emit(&mut self, origin: Vec3, rot: Quat) {
        let e = &self.emitter;
        if self.particles.len() as u32 >= e.max {
            return;
        }
        let r = &mut self.rng;
        let off = Vec3::new(r.range(-1.0, 1.0), r.range(-1.0, 1.0), r.range(-1.0, 1.0)) * e.area;
        let dir = rot * e.direction.normalize_or(Vec3::Y);
        // random direction in a cone around `dir`
        let (a, b) = (dir.any_orthonormal_pair().0, dir.any_orthonormal_pair().1);
        let cos_t = 1.0 - r.f32() * (1.0 - e.spread.min(std::f32::consts::PI).cos());
        let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
        let phi = r.f32() * std::f32::consts::TAU;
        let d = dir * cos_t + (a * phi.cos() + b * phi.sin()) * sin_t;
        let speed = r.range(e.speed.0, e.speed.1);
        let life = r.range(e.lifetime.0, e.lifetime.1).max(0.01);
        let p = Particle {
            pos: origin + rot * off,
            vel: d * speed,
            age: 0.0,
            life,
            rot: r.range(0.0, std::f32::consts::TAU),
            spin: r.range(-e.spin, e.spin),
            seed: r.f32(),
        };
        self.particles.push(p);
    }

    pub(crate) fn step(&mut self, origin: Vec3, rot: Quat, dt: f32) {
        if self.emitter.local_space {
            if let Some(last) = self.last_origin {
                let d = origin - last;
                for p in &mut self.particles {
                    p.pos += d;
                }
            }
        }
        self.last_origin = Some(origin);
        if self.emitter.enabled {
            if !self.burst_done {
                for _ in 0..self.emitter.burst {
                    self.emit(origin, rot);
                }
                self.burst_done = true;
            }
            self.acc += self.emitter.rate * dt;
            while self.acc >= 1.0 {
                self.acc -= 1.0;
                self.emit(origin, rot);
            }
        }
        let e = &self.emitter;
        let damp = (-e.drag * dt).exp();
        let turb = e.turbulence;
        let r = &mut self.rng;
        for p in &mut self.particles {
            p.age += dt;
            if turb > 0.0 {
                p.vel += Vec3::new(r.range(-1.0, 1.0), r.range(-1.0, 1.0), r.range(-1.0, 1.0)) * turb * dt * 3.0;
            }
            p.vel = (p.vel + e.gravity * dt) * damp;
            p.pos += p.vel * dt;
            p.rot += p.spin * dt;
        }
        self.particles.retain(|p| p.age < p.life);
    }

    fn write(&self, out: &mut Vec<ParticleRaw>) {
        let e = &self.emitter;
        let shape = match e.shape {
            ParticleShape::Dot => 0.0,
            ParticleShape::Square => 1.0,
            ParticleShape::Spark => 2.0,
        };
        for p in &self.particles {
            let t = (p.age / p.life).clamp(0.0, 1.0);
            let mut c = e.color_start.lerp(e.color_end, t);
            if e.fade_in > 0.0 {
                c.w *= (t / e.fade_in).min(1.0);
            }
            if e.additive {
                // fireflies etc.: slight per-particle flicker
                c.w *= 0.85 + 0.15 * (p.age * 9.0 + p.seed * 40.0).sin();
            }
            let size = e.size.0 + (e.size.1 - e.size.0) * t;
            out.push(ParticleRaw {
                pos_size: [p.pos.x, p.pos.y, p.pos.z, size],
                color: c.into(),
                params: [p.rot, shape, e.stretch, 0.0],
                vel: [p.vel.x, p.vel.y, p.vel.z, 0.0],
            });
        }
    }
}

/// Steps every particle system (fixed step) and despawns finished one-shots. Emitters on a layer
/// that no camera draws (canvas or render texture) are not simulated: they wait where they are.
pub(crate) fn step_all(ctx: &mut Ctx, dt: f32) {
    step_where(ctx, dt, None);
}

/// Steps only the emitters of one scene (the top screen while it pauses the screens below).
pub(crate) fn step_scene(ctx: &mut Ctx, dt: f32, scene: u32) {
    step_where(ctx, dt, Some(scene));
}

fn step_where(ctx: &mut Ctx, dt: f32, scene: Option<u32>) {
    let drawn = ctx.world.query::<&crate::scene::Camera>().iter().fold(0u32, |m, c| m | c.layers);
    // entities spawned this frame are tagged later, with the current scene
    let current = ctx.scenes.current_id();
    let mut done = vec![];
    for (e, ps, t, g, tag) in ctx.world.query_mut::<(Entity, &mut ParticleSystem, &Transform, Option<&GlobalTransform>, Option<&crate::screens::SceneTag>)>() {
        if drawn & (1u32 << (ps.emitter.layer & 31)) == 0 {
            continue;
        }
        if let Some(s) = scene {
            if tag.map_or(current, |t| t.0) != s {
                continue;
            }
        }
        let (origin, rot) = match g {
            Some(g) => {
                let (_, r, p) = g.0.to_scale_rotation_translation();
                (p, r)
            }
            None => (t.translation, t.rotation),
        };
        ps.step(origin, rot, dt);
        if ps.finished() {
            done.push(e);
        }
    }
    for e in done {
        ctx.despawn(e);
    }
}

/// Instance data for drawing: (additive, alpha-blended sorted far -> near).
pub(crate) fn collect(world: &World, cam: Vec3, layers: u32) -> (Vec<ParticleRaw>, Vec<ParticleRaw>) {
    let (mut add, mut alpha) = (vec![], vec![]);
    for ps in world.query::<&ParticleSystem>().iter() {
        if layers & (1u32 << (ps.emitter.layer & 31)) == 0 {
            continue;
        }
        if ps.emitter.additive {
            ps.write(&mut add);
        } else {
            ps.write(&mut alpha);
        }
    }
    let d = |p: &ParticleRaw| Vec3::new(p.pos_size[0], p.pos_size[1], p.pos_size[2]).distance_squared(cam);
    alpha.sort_by(|a, b| d(b).total_cmp(&d(a)));
    (add, alpha)
}

impl Ctx {
    /// Spawns a particle emitter at `pos`. Parent it (`Parent`) to make it follow an entity.
    /// The seed comes from `ctx.particle_rng` (separate from `ctx.rng`, so game rolls stay put).
    pub fn spawn_particles(&mut self, name: &str, pos: Vec3, emitter: ParticleEmitter) -> Entity {
        let seed = self.particle_rng.next_u64();
        self.spawn_particles_seeded(name, pos, emitter, seed)
    }

    /// [`Ctx::spawn_particles`] with an explicit seed (same seed = same particles).
    pub fn spawn_particles_seeded(&mut self, name: &str, pos: Vec3, emitter: ParticleEmitter, seed: u64) -> Entity {
        self.world.spawn((Transform::from_translation(pos), Name(name.to_string()), ParticleSystem::new(emitter, seed)))
    }

    /// AI API `particles`.
    pub(crate) fn particles_command(&mut self, a: &serde_json::Value) -> Result<serde_json::Value, String> {
        use serde_json::json;
        if let Some(name) = a["preset"].as_str() {
            let p = ParticlePreset::from_name(name).ok_or_else(|| format!("unknown preset '{name}' ({:?})", ParticlePreset::ALL.map(|p| p.name())))?;
            let pos = a["pos"].as_array().map(|v| Vec3::new(v[0].as_f64().unwrap_or(0.0) as f32, v[1].as_f64().unwrap_or(0.0) as f32, v[2].as_f64().unwrap_or(0.0) as f32)).unwrap_or(Vec3::ZERO);
            let mut em = ParticleEmitter::preset(p);
            if let Some(b) = a["burst"].as_u64() {
                em = em.burst(b as u32);
            }
            if let Some(l) = a["layer"].as_u64() {
                em.layer = l.min(31) as u8;
            }
            let ename = a["name"].as_str().unwrap_or(name).to_string();
            let e = match a["seed"].as_u64() {
                Some(seed) => self.spawn_particles_seeded(&ename, pos, em, seed),
                None => self.spawn_particles(&ename, pos, em),
            };
            return Ok(json!({"id": e.id()}));
        }
        let list: Vec<serde_json::Value> = self
            .world
            .query::<(Entity, &ParticleSystem, Option<&Name>, &Transform)>()
            .iter()
            .filter(|(_, _, n, _)| a["name"].as_str().is_none_or(|f| n.is_some_and(|n| n.0 == f)))
            .map(|(e, ps, n, t)| json!({"id": e.id(), "name": n.map(|n| n.0.clone()), "pos": t.translation.to_array(), "count": ps.count(), "emitter": ps.emitter}))
            .collect();
        let total: usize = self.world.query::<&ParticleSystem>().iter().map(|p| p.count()).sum();
        Ok(json!({"emitters": list, "particles": total, "presets": ParticlePreset::ALL.map(|p| p.name())}))
    }
}

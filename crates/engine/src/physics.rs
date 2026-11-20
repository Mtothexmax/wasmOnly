//! Physics: rapier3d as a fixed module. Bodies are linked to entities through the [`Body`]
//! component; rapier's `user_data` holds the entity bits for the reverse lookup (raycasts, contacts).
//!
//! Collision detection for game logic:
//! - **Events**: after every fixed step the set of touching collider pairs (solid contacts, sensor
//!   overlaps, character-controller touches) is compared with the previous step; every pair that
//!   starts or stops touching becomes a [`CollisionEvent`] (`collision_events()`, cleared at the start
//!   of each frame; `call collisions` keeps the last 256). No collider flags are needed.
//! - **Triggers**: sensor colliders (`add_trigger`, `Ctx::spawn_trigger`) report enter/exit
//!   (`entered`, `exited`) and who is inside right now (`inside`), also for character controllers.
//! - **Queries**: `touching`, `overlap_sphere`, `overlap_box`, `sphere_cast`, `raycast`.

pub use rapier3d as rapier;

use crate::scene::{GlobalTransform, Parent, Transform};
use glam::{Quat, Vec3};
use hecs::{Entity, World};
use rapier3d::prelude::*;
use std::collections::{BTreeMap, VecDeque};

/// Links an entity to a rapier rigid body. For dynamic bodies rapier drives the `Transform`;
/// for kinematic bodies the `Transform` drives rapier. Bodies must be root entities (no `Parent`).
#[derive(Clone, Copy, Debug)]
pub struct Body(pub RigidBodyHandle);

pub struct Physics {
    pub world: PhysicsWorld,
    /// Static colliders added since the last scene tagging (see `screens::Scenes`).
    pub(crate) new_statics: Vec<ColliderHandle>,
    /// Pairs touching after the last step, keyed by (collider, collider) in a fixed order.
    touching: BTreeMap<(u64, u64), Touch>,
    /// Events of the fixed steps of the current frame.
    events: Vec<CollisionEvent>,
    /// The last 256 events (AI API).
    history: VecDeque<CollisionEvent>,
    /// Character-controller touches found this step (the controller is kinematic, rapier reports none).
    pub(crate) probe_hits: Vec<(ColliderHandle, ColliderHandle)>,
    /// Fixed steps simulated so far.
    pub steps: u64,
}

/// Two colliders started or stopped touching (solid contact, or overlap when one is a sensor).
/// `a`/`b` are the entities of the colliders (None = static collider without an entity).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CollisionEvent {
    pub a: Option<Entity>,
    pub b: Option<Entity>,
    pub a_collider: ColliderHandle,
    pub b_collider: ColliderHandle,
    pub started: bool,
    /// One of the two is a sensor (trigger zone): an overlap, no contact force.
    pub sensor: bool,
    /// Fixed step the change happened in.
    pub step: u64,
}

impl CollisionEvent {
    /// The other side if `e` is one of the two.
    pub fn other(&self, e: Entity) -> Option<Option<Entity>> {
        if self.a == Some(e) {
            Some(self.b)
        } else if self.b == Some(e) {
            Some(self.a)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy)]
struct Touch {
    a: Option<Entity>,
    b: Option<Entity>,
    a_collider: ColliderHandle,
    b_collider: ColliderHandle,
    sensor: bool,
}

fn key(h: ColliderHandle) -> u64 {
    let (i, g) = h.into_raw_parts();
    (g as u64) << 32 | i as u64
}

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub other: Option<Entity>,
    /// Points from the other collider towards this body.
    pub normal: Vec3,
    pub depth: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    pub entity: Option<Entity>,
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
}

impl Default for Physics {
    fn default() -> Self {
        Self::new()
    }
}

impl Physics {
    pub fn new() -> Self {
        let mut world = PhysicsWorld::new();
        world.gravity = Vec3::new(0.0, -9.81, 0.0);
        Self { world, new_statics: vec![], touching: BTreeMap::new(), events: vec![], history: VecDeque::new(), probe_hits: vec![], steps: 0 }
    }

    pub fn set_gravity(&mut self, g: Vec3) {
        self.world.gravity = g;
    }

    /// Creates a body + collider for `entity`, starting at the entity's current `Transform`.
    pub fn add(
        &mut self,
        world: &mut World,
        entity: Entity,
        body: RigidBodyBuilder,
        collider: ColliderBuilder,
    ) -> RigidBodyHandle {
        let t = world
            .get::<&Transform>(entity)
            .map(|t| *t)
            .unwrap_or_default();
        let body = body
            .pose(Pose::from_parts(t.translation, t.rotation))
            .user_data(entity.to_bits().get() as u128)
            .build();
        let collider = collider.user_data(entity.to_bits().get() as u128).build();
        let (h, _) = self.world.insert(body, collider);
        let _ = world.insert_one(entity, Body(h));
        h
    }

    /// Adds a static collider not attached to any body (e.g. level geometry).
    pub fn add_static(&mut self, collider: ColliderBuilder, at: Vec3) -> ColliderHandle {
        let h = self.world.colliders.insert(collider.translation(at).build());
        self.new_statics.push(h);
        h
    }

    /// Trigger zone on `entity`: a sensor that follows the entity's `Transform` (kinematic body). It
    /// blocks nothing and reports everything that enters, stays inside or leaves it, including
    /// character controllers and other kinematic bodies (`entered`, `inside`, `exited`).
    pub fn add_trigger(&mut self, world: &mut World, entity: Entity, collider: ColliderBuilder) -> RigidBodyHandle {
        // moving things only (characters and NPCs are kinematic): static level geometry such as
        // the ground would otherwise be "inside" every trigger that touches it
        let types = ActiveCollisionTypes::DYNAMIC_KINEMATIC | ActiveCollisionTypes::KINEMATIC_KINEMATIC;
        self.add(world, entity, RigidBodyBuilder::kinematic_position_based(), collider.sensor(true).active_collision_types(types))
    }

    /// Static trigger zone without an entity (events carry its collider handle, `a`/`b` = None).
    pub fn add_static_trigger(&mut self, collider: ColliderBuilder, at: Vec3) -> ColliderHandle {
        self.add_static(collider.sensor(true).active_collision_types(ActiveCollisionTypes::DYNAMIC_FIXED | ActiveCollisionTypes::KINEMATIC_FIXED), at)
    }

    /// Collision events of this frame's fixed steps (cleared when the next frame starts).
    pub fn collision_events(&self) -> &[CollisionEvent] {
        &self.events
    }

    /// The last (up to 256) events, oldest first.
    pub fn recent_events(&self) -> impl Iterator<Item = &CollisionEvent> {
        self.history.iter()
    }

    /// Entities that entered the trigger (or started touching `entity`) this frame.
    pub fn entered(&self, entity: Entity) -> Vec<Entity> {
        self.events.iter().filter(|e| e.started).filter_map(|e| e.other(entity).flatten()).collect()
    }

    /// Entities that left the trigger (or stopped touching `entity`) this frame.
    pub fn exited(&self, entity: Entity) -> Vec<Entity> {
        self.events.iter().filter(|e| !e.started).filter_map(|e| e.other(entity).flatten()).collect()
    }

    /// Everything touching `entity` right now (solid contacts and sensor overlaps, after the last step).
    pub fn touching(&self, entity: Entity) -> Vec<Entity> {
        let mut v: Vec<Entity> = self
            .touching
            .values()
            .filter_map(|t| if t.a == Some(entity) { t.b } else if t.b == Some(entity) { t.a } else { None })
            .collect();
        v.dedup();
        v
    }

    /// Entities inside the trigger `entity` right now (sensor overlaps only).
    pub fn inside(&self, entity: Entity) -> Vec<Entity> {
        let mut v: Vec<Entity> = self
            .touching
            .values()
            .filter(|t| t.sensor)
            .filter_map(|t| if t.a == Some(entity) { t.b } else if t.b == Some(entity) { t.a } else { None })
            .collect();
        v.dedup();
        v
    }

    pub fn is_touching(&self, a: Entity, b: Entity) -> bool {
        self.touching.values().any(|t| (t.a == Some(a) && t.b == Some(b)) || (t.a == Some(b) && t.b == Some(a)))
    }

    /// Entities whose colliders overlap a sphere (`sensors`: include trigger zones).
    pub fn overlap_sphere(&self, center: Vec3, radius: f32, sensors: bool) -> Vec<Entity> {
        self.overlap(&Ball::new(radius), Pose::from_parts(center, Quat::IDENTITY), sensors)
    }

    /// Entities whose colliders overlap a box (half extents, rotation).
    pub fn overlap_box(&self, center: Vec3, half: Vec3, rotation: Quat, sensors: bool) -> Vec<Entity> {
        self.overlap(&Cuboid::new(half), Pose::from_parts(center, rotation), sensors)
    }

    fn overlap(&self, shape: &dyn Shape, pose: Pose, sensors: bool) -> Vec<Entity> {
        let filter = if sensors { QueryFilter::default() } else { QueryFilter::default().exclude_sensors() };
        let mut v: Vec<Entity> = self.world.intersect_shape(pose, shape, filter).filter_map(|(h, _)| self.entity_of_collider(h)).collect();
        v.sort_by_key(|e| e.to_bits());
        v.dedup();
        v
    }

    /// First solid collider a sphere of `radius` hits moving from `origin` along `dir` (thick ray).
    pub fn sphere_cast(&self, origin: Vec3, dir: Vec3, radius: f32, max_dist: f32) -> Option<RayHit> {
        self.sphere_cast_excluding(origin, dir, radius, max_dist, &[])
    }

    /// [`Physics::sphere_cast`] that ignores the colliders of `exclude` (the caster itself).
    pub fn sphere_cast_excluding(&self, origin: Vec3, dir: Vec3, radius: f32, max_dist: f32, exclude: &[Entity]) -> Option<RayHit> {
        let dir = dir.normalize_or_zero();
        let opts = rapier3d::parry::query::ShapeCastOptions::with_max_time_of_impact(max_dist);
        let pred = |_: ColliderHandle, c: &Collider| Entity::from_bits(c.user_data as u64).is_none_or(|e| !exclude.contains(&e));
        let mut filter = QueryFilter::default().exclude_sensors();
        if !exclude.is_empty() {
            filter.predicate = Some(&pred);
        }
        let (h, hit) = self.world.cast_shape(&Pose::from_parts(origin, Quat::IDENTITY), dir, &Ball::new(radius), opts, filter)?;
        let toi = hit.time_of_impact;
        Some(RayHit { entity: self.entity_of_collider(h), distance: toi, point: origin + dir * toi, normal: hit.normal2 })
    }

    /// Readable name of a collider for logs: `name#id`, or the shape and position of a static one.
    pub fn collider_label(&self, world: &World, h: ColliderHandle) -> String {
        if let Some(e) = self.entity_of_collider(h) {
            let n = world.get::<&crate::scene::Name>(e).map(|n| n.0.clone()).unwrap_or_default();
            return format!("{n}#{}", e.id());
        }
        match self.world.colliders.get(h) {
            Some(c) => {
                let p = c.position().translation;
                let kind = if c.is_sensor() { "static trigger" } else { "static collider" };
                format!("{kind} {:?} at ({:.1}, {:.1}, {:.1})", c.shape().shape_type(), p.x, p.y, p.z)
            }
            None => "removed collider".into(),
        }
    }

    /// Clears the per-frame event list (called before the frame's fixed steps).
    pub(crate) fn begin_frame(&mut self) {
        self.events.clear();
    }

    /// Diffs the touching pairs against the last step and records the changes.
    fn update_events(&mut self) {
        let mut now: BTreeMap<(u64, u64), Touch> = BTreeMap::new();
        let mut add = |phys: &Physics, h1: ColliderHandle, h2: ColliderHandle, sensor: bool| {
            let (h1, h2) = if key(h1) <= key(h2) { (h1, h2) } else { (h2, h1) };
            now.entry((key(h1), key(h2))).or_insert(Touch { a: phys.entity_of_collider(h1), b: phys.entity_of_collider(h2), a_collider: h1, b_collider: h2, sensor });
        };
        for p in self.world.narrow_phase.contact_pairs() {
            if p.has_any_active_contact() {
                add(self, p.collider1, p.collider2, false);
            }
        }
        for (h1, h2, on) in self.world.narrow_phase.intersection_pairs() {
            if on {
                add(self, h1, h2, true);
            }
        }
        for (h1, h2) in std::mem::take(&mut self.probe_hits) {
            if h1 != h2 {
                add(self, h1, h2, false);
            }
        }
        let step = self.steps;
        let push = |ev: CollisionEvent, events: &mut Vec<CollisionEvent>, history: &mut VecDeque<CollisionEvent>| {
            events.push(ev);
            if history.len() == 256 {
                history.pop_front();
            }
            history.push_back(ev);
        };
        for (k, t) in &now {
            if !self.touching.contains_key(k) {
                push(CollisionEvent { a: t.a, b: t.b, a_collider: t.a_collider, b_collider: t.b_collider, started: true, sensor: t.sensor, step }, &mut self.events, &mut self.history);
            }
        }
        for (k, t) in &self.touching {
            if !now.contains_key(k) {
                push(CollisionEvent { a: t.a, b: t.b, a_collider: t.a_collider, b_collider: t.b_collider, started: false, sensor: t.sensor, step }, &mut self.events, &mut self.history);
            }
        }
        self.touching = now;
    }

    pub fn remove(&mut self, world: &World, entity: Entity) {
        if let Ok(b) = world.get::<&Body>(entity) {
            self.world.remove_body_with_colliders(b.0, true);
        }
    }

    pub fn body(&self, world: &World, entity: Entity) -> Option<&RigidBody> {
        let h = world.get::<&Body>(entity).ok()?.0;
        self.world.bodies.get(h)
    }

    pub fn body_mut(&mut self, world: &World, entity: Entity) -> Option<&mut RigidBody> {
        let h = world.get::<&Body>(entity).ok()?.0;
        self.world.bodies.get_mut(h)
    }

    pub fn entity_of_collider(&self, h: ColliderHandle) -> Option<Entity> {
        let c = self.world.colliders.get(h)?;
        Entity::from_bits(c.user_data as u64)
    }

    /// Touching contacts of `entity`'s body. `normal` points from the other collider *towards*
    /// this body (a floor below gives roughly +Y), `depth` > 0 means penetration.
    pub fn contacts(&self, world: &World, entity: Entity) -> Vec<Contact> {
        let Some(rb) = self.body(world, entity) else { return vec![] };
        let mut out = vec![];
        for &ch in rb.colliders() {
            for pair in self.world.contact_pairs_with(ch) {
                if !pair.has_any_active_contact() {
                    continue;
                }
                let (me_first, other) = if pair.collider1 == ch { (true, pair.collider2) } else { (false, pair.collider1) };
                for m in pair.manifolds() {
                    let Some(deepest) = m.points.iter().map(|p| p.dist).reduce(f32::min) else { continue };
                    if deepest > 0.02 {
                        continue; // predicted contact, not touching yet
                    }
                    let n = if me_first { -m.data.normal } else { m.data.normal };
                    out.push(Contact { other: self.entity_of_collider(other), normal: n, depth: -deepest });
                }
            }
        }
        out
    }

    /// Standing on something: any touching contact whose normal is within `max_slope_deg` of +Y.
    /// Robust on edges, steps and in gaps, where a single ray straight down misses.
    pub fn grounded(&self, world: &World, entity: Entity, max_slope_deg: f32) -> bool {
        let min_y = max_slope_deg.to_radians().cos();
        self.contacts(world, entity).iter().any(|c| c.normal.y >= min_y)
    }

    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RayHit> {
        self.raycast_excluding(origin, dir, max_dist, &[])
    }

    /// [`Physics::raycast`] that ignores the colliders of `exclude`: a ray from inside or next to
    /// the hero's capsule does not hit the hero at distance 0.
    pub fn raycast_excluding(&self, origin: Vec3, dir: Vec3, max_dist: f32, exclude: &[Entity]) -> Option<RayHit> {
        let dir = dir.normalize_or_zero();
        let ray = Ray::new(origin, dir);
        let pred = |_: ColliderHandle, c: &Collider| Entity::from_bits(c.user_data as u64).is_none_or(|e| !exclude.contains(&e));
        let mut filter = QueryFilter::default();
        if !exclude.is_empty() {
            filter.predicate = Some(&pred);
        }
        let (h, hit) = self.world.cast_ray_and_get_normal(&ray, max_dist, true, filter)?;
        Some(RayHit {
            entity: self.entity_of_collider(h),
            distance: hit.time_of_impact,
            point: origin + dir * hit.time_of_impact,
            normal: hit.normal,
        })
    }

    pub fn body_count(&self) -> usize {
        self.world.bodies.len()
    }

    /// One fixed step: kinematic bodies <- Transform, simulate, Transform <- dynamic bodies.
    pub fn step(&mut self, world: &mut World, dt: f32) {
        for (t, b) in world
            .query::<(&Transform, &Body)>()
            .without::<&Parent>()
            .iter()
        {
            if let Some(rb) = self.world.bodies.get_mut(b.0) {
                if rb.is_kinematic() {
                    rb.set_next_kinematic_position(Pose::from_parts(t.translation, t.rotation));
                }
            }
        }
        self.world.integration_parameters.dt = dt;
        self.world.step();
        self.steps += 1;
        self.update_events();
        for (t, b, g) in world
            .query::<(&mut Transform, &Body, Option<&mut GlobalTransform>)>()
            .without::<&Parent>()
            .iter()
        {
            if let Some(rb) = self.world.bodies.get(b.0) {
                if rb.is_dynamic() {
                    let p = rb.position();
                    t.translation = p.translation;
                    t.rotation = p.rotation;
                    if let Some(g) = g {
                        g.0 = t.matrix();
                    }
                }
            }
        }
    }
}

impl crate::app::Ctx {
    /// Spawns an invisible box trigger zone (half extents) at `t`: doors, tall grass, area exits.
    /// Read it with `ctx.physics.entered(e)`, `inside(e)`, `exited(e)`.
    pub fn spawn_trigger(&mut self, name: &str, half: Vec3, t: Transform) -> Entity {
        let e = self.world.spawn((t, crate::scene::Name(name.to_string())));
        self.physics.add_trigger(&mut self.world, e, ColliderBuilder::cuboid(half.x, half.y, half.z));
        e
    }

    /// AI API `collisions`: recent events, or what touches / is inside an entity.
    pub(crate) fn collisions_command(&mut self, a: &serde_json::Value) -> Result<serde_json::Value, String> {
        use serde_json::json;
        // entity by id or by name
        let ent = |v: &serde_json::Value| -> Option<Entity> {
            if let Some(name) = v.as_str() {
                return self.world.query::<(Entity, &crate::scene::Name)>().iter().find(|(_, n)| n.0 == name).map(|(e, _)| e);
            }
            let id = v.as_u64()? as u32;
            self.world.iter().map(|r| r.entity()).find(|e| e.id() == id)
        };
        let label = |h: ColliderHandle, e: Option<Entity>| json!({"id": e.map(|e| e.id()), "name": self.physics.collider_label(&self.world, h)});
        if let Some(e) = ent(&a["touching"]) {
            let v: Vec<_> = self.physics.touching(e).iter().map(|o| json!({"id": o.id(), "name": self.world.get::<&crate::scene::Name>(*o).map(|n| n.0.clone()).unwrap_or_default()})).collect();
            return Ok(json!({"entity": e.id(), "touching": v}));
        }
        if let Some(e) = ent(&a["inside"]) {
            let v: Vec<_> = self.physics.inside(e).iter().map(|o| json!({"id": o.id(), "name": self.world.get::<&crate::scene::Name>(*o).map(|n| n.0.clone()).unwrap_or_default()})).collect();
            return Ok(json!({"trigger": e.id(), "inside": v}));
        }
        let since = a["since"].as_u64().unwrap_or(0);
        let only = ent(&a["entity"]);
        let limit = a["limit"].as_u64().unwrap_or(64) as usize;
        let mut list: Vec<serde_json::Value> = self
            .physics
            .recent_events()
            .filter(|e| e.step >= since && only.is_none_or(|o| e.a == Some(o) || e.b == Some(o)))
            .map(|e| json!({"step": e.step, "started": e.started, "sensor": e.sensor, "a": label(e.a_collider, e.a), "b": label(e.b_collider, e.b)}))
            .collect();
        let n = list.len();
        if n > limit {
            list.drain(..n - limit);
        }
        let started = list.iter().filter(|e| e["started"] == true).count();
        Ok(json!({"step": self.physics.steps, "touching_pairs": self.physics.touching.len(), "frame_events": self.physics.events.len(),
            "started": started, "stopped": list.len() - started, "events": list}))
    }

    /// AI API `overlap`: entities overlapping a sphere `{pos, radius}` or box `{pos, half}`.
    pub(crate) fn overlap_command(&mut self, a: &serde_json::Value) -> Result<serde_json::Value, String> {
        use serde_json::json;
        let v3 = |v: &serde_json::Value| -> Option<Vec3> { let a = v.as_array()?; Some(Vec3::new(a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32, a.get(2)?.as_f64()? as f32)) };
        let pos = v3(&a["pos"]).ok_or("pos: [x,y,z] required")?;
        let sensors = a["sensors"].as_bool().unwrap_or(false);
        let hits = if let Some(h) = v3(&a["half"]) {
            self.physics.overlap_box(pos, h, Quat::IDENTITY, sensors)
        } else {
            self.physics.overlap_sphere(pos, a["radius"].as_f64().unwrap_or(1.0) as f32, sensors)
        };
        let v: Vec<_> = hits.iter().map(|o| json!({"id": o.id(), "name": self.world.get::<&crate::scene::Name>(*o).map(|n| n.0.clone()).unwrap_or_default()})).collect();
        Ok(json!({"count": v.len(), "entities": v}))
    }
}

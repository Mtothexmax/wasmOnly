//! Physics: rapier3d as a fixed module. Bodies are linked to entities through the [`Body`]
//! component; rapier's `user_data` holds the entity bits for the reverse lookup (raycasts, contacts).

pub use rapier3d as rapier;

use crate::scene::{GlobalTransform, Parent, Transform};
use glam::Vec3;
use hecs::{Entity, World};
use rapier3d::prelude::*;

/// Links an entity to a rapier rigid body. For dynamic bodies rapier drives the `Transform`;
/// for kinematic bodies the `Transform` drives rapier. Bodies must be root entities (no `Parent`).
#[derive(Clone, Copy, Debug)]
pub struct Body(pub RigidBodyHandle);

pub struct Physics {
    pub world: PhysicsWorld,
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
        Self { world }
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
        self.world
            .colliders
            .insert(collider.translation(at).build())
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
        let dir = dir.normalize_or_zero();
        let ray = Ray::new(origin, dir);
        let (h, hit) =
            self.world
                .cast_ray_and_get_normal(&ray, max_dist, true, QueryFilter::default())?;
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

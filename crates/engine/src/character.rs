//! Character controller (rapier's `KinematicCharacterController`) and a third-person camera.
//!
//! **Character.** `ctx.spawn_character(name, pos, CharacterController::default())` creates a capsule
//! with a kinematic body. Each fixed step the game sets what the character *wants*
//! (`wish` = horizontal direction, `run`, `jump`), and the engine moves it before physics:
//! sliding along walls, climbing slopes up to `max_slope_deg`, auto-stepping onto ledges up to
//! `step_height`, snapping down to the ground, gravity, coyote time and jump buffering, pushing
//! dynamic bodies. The result is in `grounded`, `velocity`, `sliding`, `collisions`.
//!
//! ```ignore
//! // fixed_update:
//! let yaw = ctx.world.get::<&ThirdPersonCamera>(cam).map(|c| c.yaw).unwrap_or(0.0);
//! let mut cc = ctx.world.get::<&mut CharacterController>(hero).unwrap();
//! cc.drive(ctx.input.move_axis(), yaw);      // camera-relative WASD / stick / touch joystick
//! cc.run = ctx.input.key("ShiftLeft");
//! cc.jump = ctx.input.key_down_or_pressed("Space"); // held or tapped since the last step: one jump per press
//! ```
//!
//! **Third-person camera.** Put [`ThirdPersonCamera`] on the camera entity. After `Game::update` the
//! engine orbits it around the target: mouse (pointer lock or right-drag), touch look and gamepad
//! right stick turn it, wheel and pinch zoom; a sphere cast pulls it in front of walls; position
//! and zoom are smoothed.

use crate::app::Ctx;
use crate::input::MouseButton;
use crate::physics::rapier::control::{CharacterAutostep, CharacterCollision, CharacterLength, KinematicCharacterController};
use crate::physics::rapier::prelude::*;
use crate::physics::Body;
use crate::scene::{Name, Transform};
use glam::{Quat, Vec2, Vec3};
use hecs::Entity;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct CharacterController {
    // ---- shape + tuning ----
    pub radius: f32,
    /// Half height of the capsule's cylinder part (total height = 2 * (half_height + radius)).
    pub half_height: f32,
    /// Walk speed in m/s.
    pub speed: f32,
    pub run_multiplier: f32,
    /// Initial upward speed of a jump (m/s). Jump height = v² / (2 g).
    pub jump_speed: f32,
    /// Multiplies world gravity.
    pub gravity_scale: f32,
    pub max_slope_deg: f32,
    /// Ledges up to this height are stepped onto (m).
    pub step_height: f32,
    /// Keeps the character glued to the ground when walking down slopes/stairs (m).
    pub snap_to_ground: f32,
    /// 0 = no steering in the air, 1 = full.
    pub air_control: f32,
    /// How fast horizontal velocity reaches the target (1/s).
    pub acceleration: f32,
    /// Seconds after leaving the ground in which a jump still works.
    pub coyote_time: f32,
    /// Seconds a jump press is remembered before landing.
    pub jump_buffer: f32,
    /// Turn the entity to face the movement direction (rad/s; 0 = off).
    pub turn_speed: f32,
    /// Mass used to push dynamic bodies (0 = don't push).
    pub push_mass: f32,
    // ---- input (set by the game every fixed step) ----
    /// Desired horizontal direction (length <= 1).
    #[serde(skip)]
    pub wish: Vec3,
    #[serde(skip)]
    pub run: bool,
    /// Jump button held; a jump fires on the press edge.
    #[serde(skip)]
    pub jump: bool,
    // ---- state ----
    pub velocity: Vec3,
    pub grounded: bool,
    pub sliding: bool,
    pub yaw: f32,
    /// Colliding entities of the last step.
    #[serde(skip)]
    pub collisions: Vec<Entity>,
    pub jumps: u32,
    /// Seconds the character has wanted to move but could not (see the "stuck" log warning).
    pub blocked_time: f32,
    pub stuck_reports: u32,
    /// Times the step-up retry got the character over a low edge / seam.
    pub step_ups: u32,
    #[serde(skip)]
    #[doc(hidden)]
    pub warn_cooldown: f32,
    #[serde(skip)]
    #[doc(hidden)]
    pub airborne_presses: u32,
    /// internal state
    #[serde(skip)]
    #[doc(hidden)]
    pub since_ground: f32,
    /// internal state
    #[serde(skip)]
    #[doc(hidden)]
    pub since_jump_press: f32,
    /// internal state
    #[serde(skip)]
    #[doc(hidden)]
    pub jump_was_held: bool,
}

impl Default for CharacterController {
    fn default() -> Self {
        Self {
            radius: 0.35,
            half_height: 0.5,
            speed: 5.0,
            run_multiplier: 1.7,
            jump_speed: 6.5,
            gravity_scale: 1.8,
            max_slope_deg: 50.0,
            step_height: 0.35,
            snap_to_ground: 0.3,
            air_control: 0.35,
            acceleration: 14.0,
            coyote_time: 0.12,
            jump_buffer: 0.12,
            turn_speed: 12.0,
            push_mass: 2.0,
            wish: Vec3::ZERO,
            run: false,
            jump: false,
            velocity: Vec3::ZERO,
            grounded: false,
            sliding: false,
            yaw: 0.0,
            collisions: vec![],
            jumps: 0,
            blocked_time: 0.0,
            stuck_reports: 0,
            step_ups: 0,
            warn_cooldown: 0.0,
            airborne_presses: 0,
            since_ground: 10.0,
            since_jump_press: 10.0,
            jump_was_held: false,
        }
    }
}

impl CharacterController {
    /// Camera-relative input: `axis` (x = right, y = forward, e.g. `input.move_axis()`) turned by
    /// the camera's yaw into a world direction.
    pub fn drive(&mut self, axis: Vec2, camera_yaw: f32) {
        let a = if axis.length() > 1.0 { axis.normalize() } else { axis };
        let fwd = Vec3::new(-camera_yaw.sin(), 0.0, -camera_yaw.cos());
        let right = Vec3::new(fwd.z * -1.0, 0.0, fwd.x);
        self.wish = right * a.x + fwd * a.y;
    }

    /// Feet position relative to the entity's origin (the capsule centre).
    pub fn foot_offset(&self) -> f32 {
        self.half_height + self.radius
    }
}

impl Ctx {
    /// Spawns a capsule character at `pos` (feet position) with a kinematic body. Attach a mesh as a
    /// child or insert a `MeshRenderer` on the returned entity.
    pub fn spawn_character(&mut self, name: &str, feet: Vec3, cc: CharacterController) -> Entity {
        let center = feet + Vec3::Y * cc.foot_offset();
        let (hh, r) = (cc.half_height, cc.radius);
        let e = self.world.spawn((Transform::from_translation(center), Name(name.to_string())));
        self.physics.add(
            &mut self.world,
            e,
            RigidBodyBuilder::kinematic_position_based(),
            ColliderBuilder::capsule_y(hh, r).friction(0.0),
        );
        let _ = self.world.insert_one(e, cc);
        e
    }
}

/// Moves every character (before the physics step).
pub(crate) fn step_characters(ctx: &mut Ctx, dt: f32) {
    let gravity = ctx.physics.world.gravity;
    let list: Vec<Entity> = ctx.world.query::<(Entity, &CharacterController, &Body)>().iter().map(|(e, ..)| e).collect();
    // characters API + debug log snapshot use `describe_characters`
    for e in list {
        let (Ok(mut cc), Ok(body), Ok(mut t)) = (
            ctx.world.get::<&mut CharacterController>(e),
            ctx.world.get::<&Body>(e),
            ctx.world.get::<&mut Transform>(e),
        ) else {
            continue;
        };
        let cc = &mut *cc;
        let handle = body.0;
        let pw = &mut ctx.physics.world;
        let Some(collider) = pw.bodies.get(handle).and_then(|b| b.colliders().first().copied()) else { continue };
        let shape = pw.colliders[collider].shared_shape().clone();

        // jump timing (edge, buffer, coyote)
        let pressed = cc.jump && !cc.jump_was_held;
        cc.jump_was_held = cc.jump;
        cc.since_jump_press = if pressed { 0.0 } else { cc.since_jump_press + dt };
        cc.since_ground = if cc.grounded { 0.0 } else { cc.since_ground + dt };

        // horizontal velocity towards the wish
        let wish = Vec3::new(cc.wish.x, 0.0, cc.wish.z);
        let wish = if wish.length() > 1.0 { wish.normalize() } else { wish };
        let target = wish * cc.speed * if cc.run { cc.run_multiplier } else { 1.0 };
        let control = if cc.grounded { 1.0 } else { cc.air_control };
        let k = (1.0 - (-cc.acceleration * control * dt).exp()).clamp(0.0, 1.0);
        let h = Vec3::new(cc.velocity.x, 0.0, cc.velocity.z);
        let h = h + (target - h) * k;
        // standing still: the exponential approach never reaches 0 exactly; stop below 5 cm/s
        let idle = cc.grounded && wish.length_squared() < 1e-6;
        let h = if idle && h.length() < 0.05 { Vec3::ZERO } else { h };

        // vertical
        let mut vy = cc.velocity.y;
        if cc.grounded && vy < 0.0 {
            vy = -0.5; // keep contact
        }
        let jumps_before = cc.jumps;
        if cc.since_jump_press <= cc.jump_buffer && cc.since_ground <= cc.coyote_time && vy <= 0.5 {
            vy = cc.jump_speed;
            cc.since_jump_press = 10.0;
            cc.since_ground = 10.0;
            cc.jumps += 1;
        }
        vy += gravity.y * cc.gravity_scale * dt;
        vy = vy.max(-50.0);

        let mut kcc = KinematicCharacterController::default();
        kcc.max_slope_climb_angle = cc.max_slope_deg.to_radians();
        kcc.min_slope_slide_angle = (cc.max_slope_deg + 5.0).to_radians();
        kcc.offset = CharacterLength::Absolute(0.02);
        kcc.autostep = (cc.step_height > 0.0).then(|| CharacterAutostep {
            max_height: CharacterLength::Absolute(cc.step_height),
            min_width: CharacterLength::Absolute(cc.radius * 0.5),
            include_dynamic_bodies: false,
        });
        // no snapping while going up (jumps)
        kcc.snap_to_ground = (vy <= 0.0 && cc.snap_to_ground > 0.0).then_some(CharacterLength::Absolute(cc.snap_to_ground));

        let filter = QueryFilter::default().exclude_rigid_body(handle).exclude_sensors();
        let mut hits: Vec<CharacterCollision> = vec![];
        // Rising (jump): move up on its own first, with slope limits off. rapier's KCC treats upward
        // motion along a wall as climbing a too-steep slope and removes it — jumping next to a
        // crate, wall or post did nothing. Only a real ceiling may stop the rise now.
        let mut rise = 0.0f32;
        let mut ceiling = false;
        if vy > 0.0 {
            let mut up = KinematicCharacterController::default();
            up.offset = kcc.offset;
            up.max_slope_climb_angle = std::f32::consts::FRAC_PI_2;
            up.min_slope_slide_angle = std::f32::consts::FRAC_PI_2;
            up.autostep = None;
            up.snap_to_ground = None;
            let want = vy * dt;
            let qp = pw.broad_phase.as_query_pipeline(pw.narrow_phase.query_dispatcher(), &pw.bodies, &pw.colliders, filter);
            let m = up.move_shape(dt, &qp, &*shape, &Pose::from_parts(t.translation, Quat::IDENTITY), Vec3::Y * want, |c| hits.push(c));
            rise = m.translation.y;
            ceiling = rise < want * 0.5;
            t.translation += m.translation;
        }
        let desired = Vec3::new(h.x, if vy > 0.0 { 0.0 } else { vy }, h.z) * dt;
        let pos = Pose::from_parts(t.translation, Quat::IDENTITY);
        let mut moved = {
            let qp = pw.broad_phase.as_query_pipeline(pw.narrow_phase.query_dispatcher(), &pw.bodies, &pw.colliders, filter);
            kcc.move_shape(dt, &qp, &*shape, &pos, desired, |c| hits.push(c))
        };
        // Step-up retry: a grounded character that wants to move but makes (almost) no horizontal
        // progress has met a low edge or a seam between boxes. Its round bottom touches such an edge
        // at a slope-like angle, so rapier's autostep (walls only) never triggers and certain
        // approach positions end in zero motion (seen as a 1-in-3 stall on 5 cm plates). Retry as:
        // lift by step_height (no slope limits), move horizontally, drop back with ground snapping.
        // The better of the two results wins; deterministic for a given state.
        let want_h = Vec3::new(desired.x, 0.0, desired.z);
        let got_h = Vec3::new(moved.translation.x, 0.0, moved.translation.z);
        if cc.step_height > 0.0 && vy <= 0.0 && cc.since_ground <= cc.coyote_time && want_h.length() > 1e-5 && got_h.length() < want_h.length() * 0.5 {
            let qp = pw.broad_phase.as_query_pipeline(pw.narrow_phase.query_dispatcher(), &pw.bodies, &pw.colliders, filter);
            let mut up = KinematicCharacterController::default();
            up.offset = kcc.offset;
            up.max_slope_climb_angle = std::f32::consts::FRAC_PI_2;
            up.min_slope_slide_angle = std::f32::consts::FRAC_PI_2;
            up.autostep = None;
            up.snap_to_ground = None;
            let lift = up.move_shape(dt, &qp, &*shape, &pos, Vec3::Y * cc.step_height, |_| {});
            let p1 = Pose::from_parts(t.translation + lift.translation, Quat::IDENTITY);
            let mut flat = kcc.clone();
            flat.snap_to_ground = None;
            flat.autostep = None;
            let side = flat.move_shape(dt, &qp, &*shape, &p1, want_h, |_| {});
            let side_h = Vec3::new(side.translation.x, 0.0, side.translation.z);
            // only a real step: at least half the wanted distance, and it must land on ground
            // (walls stay walls, ledges higher than step_height are not climbed)
            if side_h.length() >= want_h.length() * 0.5 && side_h.length() > got_h.length() + 1e-4 {
                let p2 = Pose::from_parts(p1.translation + side.translation, Quat::IDENTITY);
                let mut down = kcc.clone();
                down.autostep = None;
                down.snap_to_ground = Some(CharacterLength::Absolute(cc.step_height + cc.snap_to_ground.max(0.05)));
                let mut drop_hits = vec![];
                let drop = down.move_shape(dt, &qp, &*shape, &p2, Vec3::NEG_Y * (lift.translation.y + 0.01), |c| drop_hits.push(c));
                let climbed = lift.translation.y + drop.translation.y;
                if drop.grounded && climbed <= cc.step_height + 0.01 {
                    moved.translation = lift.translation + side.translation + drop.translation;
                    moved.grounded = true;
                    moved.is_sliding_down_slope = false;
                    hits.extend(drop_hits);
                    cc.step_ups += 1;
                }
            }
        }
        if cc.push_mass > 0.0 && !hits.is_empty() {
            let mut qpm = pw.broad_phase.as_query_pipeline_mut(pw.narrow_phase.query_dispatcher(), &mut pw.bodies, &mut pw.colliders, filter);
            kcc.solve_character_collision_impulses(dt, &mut qpm, &*shape, cc.push_mass, &hits);
        }
        // idle on the ground: drop sub-millimetre corrections of the collision solver (contact
        // offset, depenetration), so a standing character and its follow camera stay exactly still
        if idle && h == Vec3::ZERO && cc.jumps == jumps_before && !moved.is_sliding_down_slope && moved.grounded && moved.translation.length() < 0.002 {
            moved.translation = Vec3::ZERO;
        }
        t.translation += moved.translation;
        cc.grounded = moved.grounded && vy <= 0.0;
        cc.sliding = moved.is_sliding_down_slope;
        let actual = moved.translation / dt.max(1e-6);
        cc.velocity = Vec3::new(actual.x, if cc.grounded || ceiling { 0.0 } else { vy }, actual.z);
        cc.collisions = hits.iter().filter_map(|c| ctx.physics.entity_of_collider(c.handle)).collect();

        // Self-diagnosis: input that has no effect is logged (with what blocks it), so a "jump /
        // walk does nothing" report shows its cause in the log instead of needing a repro.
        let want_speed = Vec3::new(target.x, 0.0, target.z).length();
        let got_speed = Vec3::new(actual.x, 0.0, actual.z).length();
        if want_speed > 0.5 && got_speed < want_speed * 0.1 {
            cc.blocked_time += dt;
        } else {
            cc.blocked_time = 0.0;
        }
        cc.warn_cooldown = (cc.warn_cooldown - dt).max(0.0);
        let names = |ctx_w: &hecs::World, list: &[Entity]| -> String {
            let mut v: Vec<String> = list.iter().take(3).map(|e| format!("{}#{}", ctx_w.get::<&crate::scene::Name>(*e).map(|n| n.0.clone()).unwrap_or_default(), e.id())).collect();
            v.dedup();
            if v.is_empty() { "nothing (no collision reported)".into() } else { v.join(", ") }
        };
        if cc.blocked_time > 0.5 && cc.warn_cooldown <= 0.0 {
            cc.warn_cooldown = 3.0;
            cc.stuck_reports += 1;
            crate::log_warn!("character #{} stuck: wants {:.1} m/s, moves {:.2} m/s, blocked by {} (grounded={})", e.id(), want_speed, got_speed, names(&ctx.world, &cc.collisions), cc.grounded);
        }
        if ceiling && cc.warn_cooldown <= 0.0 {
            cc.warn_cooldown = 3.0;
            crate::log_warn!("character #{} jump blocked after {:.2} m by {}", e.id(), rise, names(&ctx.world, &cc.collisions));
        }
        if pressed && cc.jumps == jumps_before && cc.since_ground > cc.coyote_time + cc.jump_buffer && cc.warn_cooldown <= 0.0 {
            // pressed while airborne: normal, but if it repeats the player is likely stuck in the air
            cc.airborne_presses += 1;
            if cc.airborne_presses >= 3 {
                cc.warn_cooldown = 3.0;
                cc.airborne_presses = 0;
                crate::log_warn!("character #{} jump pressed while not grounded {:.1} s (velocity y {:.2}, touching {})", e.id(), cc.since_ground, cc.velocity.y, names(&ctx.world, &cc.collisions));
            }
        } else if cc.grounded {
            cc.airborne_presses = 0;
        }
        if cc.turn_speed > 0.0 && wish.length_squared() > 0.01 {
            let want = (-wish.x).atan2(-wish.z);
            let mut d = want - cc.yaw;
            d = (d + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
            cc.yaw += d * (1.0 - (-cc.turn_speed * dt).exp());
            t.rotation = Quat::from_rotation_y(cc.yaw);
        }
    }
}

/// Orbit camera following `target`. Put it on the camera entity (with `Camera` + `Transform`).
#[derive(Clone, Debug, Serialize)]
pub struct ThirdPersonCamera {
    #[serde(skip)]
    pub target: Option<Entity>,
    /// Look-at point above the target's origin.
    pub pivot_offset: Vec3,
    /// Radians; 0 looks along -Z.
    pub yaw: f32,
    /// Radians; positive looks down at the target.
    pub pitch: f32,
    pub distance: f32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub min_pitch: f32,
    pub max_pitch: f32,
    /// Radians per CSS pixel of mouse movement.
    pub sensitivity: f32,
    /// Mouse look needs the right button (or pointer lock). false = any mouse movement turns.
    pub require_button: bool,
    /// Follow smoothing rate (1/s); higher = stiffer.
    pub follow_rate: f32,
    /// Keep the camera in front of colliders between pivot and camera.
    pub collide: bool,
    /// Sphere radius of the collision cast.
    pub probe_radius: f32,
    /// Current (collision-shortened, smoothed) distance.
    pub current_distance: f32,
    /// internal state
    #[serde(skip)]
    #[doc(hidden)]
    pub pivot: Option<Vec3>,
    /// internal state
    #[serde(skip)]
    #[doc(hidden)]
    pub want_distance: f32,
}

impl Default for ThirdPersonCamera {
    fn default() -> Self {
        Self {
            target: None,
            pivot_offset: Vec3::new(0.0, 0.8, 0.0),
            yaw: 0.0,
            pitch: 0.35,
            distance: 7.0,
            min_distance: 2.0,
            max_distance: 20.0,
            min_pitch: -0.4,
            max_pitch: 1.3,
            sensitivity: 0.005,
            require_button: true,
            follow_rate: 12.0,
            collide: true,
            probe_radius: 0.25,
            current_distance: 7.0,
            pivot: None,
            want_distance: 7.0,
        }
    }
}

impl ThirdPersonCamera {
    pub fn following(target: Entity) -> Self {
        Self { target: Some(target), ..Default::default() }
    }
    pub fn forward_flat(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos())
    }
}

/// Updates every third-person camera (after `Game::update`).
pub(crate) fn update_cameras(ctx: &mut Ctx, dt: f32) {
    let cams: Vec<Entity> = ctx.world.query::<(Entity, &ThirdPersonCamera)>().iter().map(|(e, _)| e).collect();
    for cam in cams {
        let Ok(mut c) = ctx.world.get::<&mut ThirdPersonCamera>(cam).map(|c| (*c).clone()) else { continue };
        let Some(target) = c.target.filter(|t| ctx.world.contains(*t)) else { continue };
        // roots: Transform is current (physics/characters write it); children: last GlobalTransform
        let tpos = if ctx.world.get::<&crate::scene::Parent>(target).is_ok() {
            ctx.world.get::<&crate::scene::GlobalTransform>(target).map(|g| g.translation()).unwrap_or_default()
        } else {
            ctx.world.get::<&Transform>(target).map(|t| t.translation).unwrap_or_default()
        };

        // input
        let inp = &ctx.input;
        let mouse_look = inp.pointer_locked || !c.require_button || inp.button(MouseButton::Right);
        let mut look = if mouse_look && !inp.pointer_over_panel { inp.pointer_delta } else { Vec2::ZERO };
        look += inp.touch.look_delta;
        look += inp.gamepad_right * 600.0 * dt; // stick: ~3 rad/s at full tilt with default sensitivity
        c.yaw -= look.x * c.sensitivity;
        c.pitch = (c.pitch + look.y * c.sensitivity).clamp(c.min_pitch, c.max_pitch);
        if c.want_distance <= 0.0 {
            c.want_distance = c.distance;
        }
        c.want_distance *= (1.0 + inp.wheel * 0.001).clamp(0.5, 2.0);
        if inp.touch.pinch != 0.0 {
            c.want_distance /= (1.0 + inp.touch.pinch).clamp(0.5, 2.0);
        }
        c.want_distance = c.want_distance.clamp(c.min_distance, c.max_distance);
        c.distance += (c.want_distance - c.distance) * (1.0 - (-10.0 * dt).exp());

        // follow
        let want_pivot = tpos + c.pivot_offset;
        let pivot = match c.pivot {
            Some(p) => p + (want_pivot - p) * (1.0 - (-c.follow_rate * dt).exp()),
            None => want_pivot,
        };
        c.pivot = Some(pivot);
        let dir = Vec3::new(c.yaw.sin() * c.pitch.cos(), c.pitch.sin(), c.yaw.cos() * c.pitch.cos());

        // collision: shape cast from pivot towards the camera, ignoring the target's body
        let mut dist = c.distance;
        if c.collide {
            let pw = &ctx.physics.world;
            let mut filter = QueryFilter::default().exclude_sensors();
            if let Ok(b) = ctx.world.get::<&Body>(target) {
                filter = filter.exclude_rigid_body(b.0);
            }
            let ball = Ball::new(c.probe_radius);
            let qp = pw.broad_phase.as_query_pipeline(pw.narrow_phase.query_dispatcher(), &pw.bodies, &pw.colliders, filter);
            let opts = crate::physics::rapier::parry::query::ShapeCastOptions { max_time_of_impact: c.distance, stop_at_penetration: false, ..Default::default() };
            if let Some((_, hit)) = qp.cast_shape(&Pose::from_parts(pivot, Quat::IDENTITY), dir, &ball, opts) {
                dist = hit.time_of_impact.max(0.3);
            }
        }
        // pull in fast, ease back out
        let rate = if dist < c.current_distance { 30.0 } else { 4.0 };
        c.current_distance += (dist - c.current_distance) * (1.0 - (-rate * dt).exp());
        let eye = pivot + dir * c.current_distance;
        if let Ok(mut t) = ctx.world.get::<&mut Transform>(cam) {
            *t = Transform::from_translation(eye).looking_at(pivot, Vec3::Y).with_scale(t.scale);
        }
        if let Ok(mut slot) = ctx.world.get::<&mut ThirdPersonCamera>(cam) {
            *slot = c;
        }
    }
}

/// JSON snapshot of every character (AI API `characters`, debug-mode log lines).
pub fn describe_characters(world: &hecs::World) -> serde_json::Value {
    let list: Vec<serde_json::Value> = world
        .query::<(Entity, &CharacterController, &Transform)>()
        .iter()
        .map(|(e, cc, t)| {
            serde_json::json!({"id": e.id(), "pos": t.translation.to_array(), "grounded": cc.grounded, "velocity": cc.velocity.to_array(),
                "jumps": cc.jumps, "jump_held": cc.jump, "wish": cc.wish.to_array(), "blocked_time": cc.blocked_time, "stuck_reports": cc.stuck_reports, "step_ups": cc.step_ups,
                "touching": cc.collisions.iter().map(|c| c.id()).collect::<Vec<_>>()})
        })
        .collect();
    serde_json::Value::Array(list)
}

impl Ctx {
    /// AI API `characters`: all character controllers; `{id, pos}` places one (feet stay above `pos`).
    pub(crate) fn characters_command(&mut self, a: &serde_json::Value) -> Result<serde_json::Value, String> {
        if let (Some(id), Some(p)) = (a["id"].as_u64(), a["pos"].as_array()) {
            let e = self.world.query::<(Entity, &CharacterController)>().iter().map(|(e, _)| e).find(|e| e.id() == id as u32).ok_or("no character with that id")?;
            let v = |i: usize| p.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
            if let Ok(mut t) = self.world.get::<&mut Transform>(e) {
                t.translation = Vec3::new(v(0), v(1), v(2));
            }
            if let Ok(mut cc) = self.world.get::<&mut CharacterController>(e) {
                cc.velocity = Vec3::ZERO;
            }
        }
        Ok(serde_json::json!({"characters": describe_characters(&self.world)}))
    }
}

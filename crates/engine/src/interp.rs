//! Render interpolation for objects moved by the fixed simulation step.
//!
//! Physics and character controllers advance in fixed 1/60 s steps. A display at 57, 60, 75 or
//! 144 Hz then gets 0, 1 or 2 steps per frame, so a moving ball or walker visibly jumps (and the
//! camera following it judders) even though the frame rate looks fine. Here every dynamic body
//! and character controller is drawn between its last two simulated poses, at the fraction of the
//! next step that has already elapsed (`accumulator / fixed_dt`). Rendering is one step (16.7 ms)
//! behind the simulation, the standard trade-off.
//!
//! The simulation never sees the blended pose: the true pose is restored before each fixed step.
//! When game code moves such an entity outside the fixed step (a teleport in `update`), the new
//! pose is taken as is (no blend from the old place). While paused the true pose is shown, so
//! `pause` + `step` tests read exact values.
//! Opt out per entity: `NoInterpolation`; globally: `ctx.interpolation = false`.

use crate::app::Ctx;
use crate::physics::Body;
use crate::scene::{Parent, Transform};
use glam::{Quat, Vec3};
use hecs::Entity;

/// Last two simulated poses + the pose that was displayed (to detect outside edits).
#[derive(Clone, Copy, Debug)]
pub struct Interpolated {
    prev: (Vec3, Quat),
    curr: (Vec3, Quat),
    shown: (Vec3, Quat),
}

/// Marker: draw this entity at its exact simulated pose.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoInterpolation;

fn pose(t: &Transform) -> (Vec3, Quat) {
    (t.translation, t.rotation)
}

/// Start of a fixed step: put the true pose back (if the blended one is shown) and remember it as
/// the "previous" pose of this step.
pub(crate) fn before_step(ctx: &mut Ctx) {
    if !ctx.interpolation {
        return;
    }
    tag_new(ctx);
    let restore = ctx.interp_shown;
    ctx.interp_shown = false;
    for (t, i) in ctx.world.query_mut::<(&mut Transform, &mut Interpolated)>() {
        if restore {
            if pose(t) == i.shown {
                t.translation = i.curr.0;
                t.rotation = i.curr.1;
            } else {
                i.curr = pose(t); // moved by game code: take it
            }
        }
        i.prev = pose(t);
    }
}

/// After the frame's fixed steps: blend between the last two poses by the elapsed fraction of
/// the next step.
pub(crate) fn after_steps(ctx: &mut Ctx, stepped: bool) {
    if !ctx.interpolation {
        return;
    }
    tag_new(ctx);
    let alpha = if ctx.time.paused { 1.0 } else { (ctx.time.accumulator / ctx.time.fixed_dt).clamp(0.0, 1.0) };
    let was_shown = ctx.interp_shown;
    for (t, i) in ctx.world.query_mut::<(&mut Transform, &mut Interpolated)>() {
        if stepped {
            i.curr = pose(t);
        } else if was_shown && pose(t) != i.shown {
            // moved by game code between steps: snap
            i.curr = pose(t);
            i.prev = pose(t);
        } else if !was_shown {
            i.curr = pose(t);
        }
        // a jump of more than 2 m in one step is a teleport, not motion
        let (p, r) = if (i.curr.0 - i.prev.0).length_squared() > 4.0 { i.curr } else { (i.prev.0.lerp(i.curr.0, alpha), i.prev.1.slerp(i.curr.1, alpha)) };
        t.translation = p;
        t.rotation = r;
        i.shown = (p, r);
    }
    ctx.interp_shown = true;
}

/// Adds `Interpolated` to new dynamic bodies and character controllers (root entities only).
fn tag_new(ctx: &mut Ctx) {
    let mut new: Vec<(Entity, (Vec3, Quat))> = vec![];
    for (e, t, b) in ctx
        .world
        .query::<(Entity, &Transform, &Body)>()
        .without::<&Interpolated>()
        .without::<&NoInterpolation>()
        .without::<&Parent>()
        .iter()
    {
        if ctx.physics.world.bodies.get(b.0).is_some_and(|rb| rb.is_dynamic()) {
            new.push((e, pose(t)));
        }
    }
    for (e, t, _) in ctx
        .world
        .query::<(Entity, &Transform, &crate::character::CharacterController)>()
        .without::<&Interpolated>()
        .without::<&NoInterpolation>()
        .without::<&Parent>()
        .iter()
    {
        new.push((e, pose(t)));
    }
    for (e, p) in new {
        let _ = ctx.world.insert_one(e, Interpolated { prev: p, curr: p, shown: p });
    }
}

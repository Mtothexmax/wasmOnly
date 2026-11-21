//! Skeletal animation, animation clips and morph targets (from glTF).
//!
//! * **Clips** (`AnimationClip`) animate node translation/rotation/scale and morph weights with
//!   STEP, LINEAR (slerp for rotations) or CUBICSPLINE interpolation.
//! * **[`Animator`]** sits on a glTF model root: `play(name, fade)` crossfades between clips,
//!   with speed and looping. Time advances in the fixed step (deterministic, follows `pause`/`step`
//!   and `time_scale`); poses are applied once per frame before transforms propagate.
//! * **Skinning + morph targets** run on the CPU after transform propagation, written into a
//!   per-instance vertex buffer ([`Deformer`]). Shadows, culling, picking and the z-fighting tools
//!   see the deformed mesh like any other. Fine for tens of characters with a few thousand
//!   vertices each.
//!
//! ```ignore
//! let hero = ctx.load_gltf("assets/critter.glb", Transform::default());
//! // later (once loaded): engine.call("animation", {id}) lists clips
//! ctx.play_animation(hero, "wave", PlayOpts::looped().fade(0.25));
//! ctx.set_morph_weight(head, 0, 0.8);   // manual morph weight (when no clip drives it)
//! ```
//!
//! AI API: `animation {id}` shows clips, playing state and morph weights; `animation {id, play,
//! fade?, speed?, looped?}` plays a clip; `{id, stop:true}` stops.

use crate::app::Ctx;
use crate::mesh::Vertex;
use crate::renderer::MeshId;
use crate::scene::{GlobalTransform, Transform};
use glam::{Mat4, Quat, Vec3};
use hecs::{Entity, World};
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Interp {
    Step,
    Linear,
    CubicSpline,
}

#[derive(Clone, Debug)]
pub enum ChannelValues {
    Translation(Vec<Vec3>),
    Rotation(Vec<Quat>),
    Scale(Vec<Vec3>),
    /// Flattened: keyframe-major, `n` weights per keyframe.
    Weights(Vec<f32>, usize),
}

#[derive(Clone, Debug)]
pub struct Channel {
    /// glTF node index (mapped to an entity by the Animator).
    pub node: usize,
    pub interp: Interp,
    pub times: Vec<f32>,
    /// For CUBICSPLINE: [in_tangent, value, out_tangent] per keyframe.
    pub values: ChannelValues,
}

#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<Channel>,
}

/// Locate `t` in the keyframe times: (index of the key before, fraction to the next, key span).
fn locate(times: &[f32], t: f32) -> (usize, f32, f32) {
    if times.len() < 2 || t <= times[0] {
        return (0, 0.0, 0.0);
    }
    let last = times.len() - 1;
    if t >= times[last] {
        return (last, 0.0, 0.0);
    }
    let i = times.partition_point(|&k| k <= t).saturating_sub(1).min(last - 1);
    let span = times[i + 1] - times[i];
    let f = if span > 0.0 { (t - times[i]) / span } else { 0.0 };
    (i, f, span)
}

fn hermite<T>(p0: T, m0: T, p1: T, m1: T, f: f32, span: f32) -> T
where
    T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T> + Copy,
{
    let (f2, f3) = (f * f, f * f * f);
    p0 * (2.0 * f3 - 3.0 * f2 + 1.0) + m0 * ((f3 - 2.0 * f2 + f) * span) + p1 * (-2.0 * f3 + 3.0 * f2) + m1 * ((f3 - f2) * span)
}

/// Sampled value of one channel.
enum Sample {
    T(Vec3),
    R(Quat),
    S(Vec3),
    W(Vec<f32>),
}

impl Channel {
    fn sample(&self, t: f32) -> Sample {
        let (i, f, span) = locate(&self.times, t);
        let cubic = self.interp == Interp::CubicSpline;
        let next = (i + 1).min(self.times.len().saturating_sub(1));
        // index of the value (CUBICSPLINE stores 3 per key)
        let v = |k: usize| if cubic { k * 3 + 1 } else { k };
        let step = self.interp == Interp::Step || f == 0.0;
        macro_rules! vec_sample {
            ($vals:expr) => {{
                let vals = $vals;
                if step {
                    vals[v(i)]
                } else if cubic {
                    hermite(vals[v(i)], vals[i * 3 + 2], vals[v(next)], vals[next * 3], f, span)
                } else {
                    vals[v(i)].lerp(vals[v(next)], f)
                }
            }};
        }
        match &self.values {
            ChannelValues::Translation(vals) => Sample::T(vec_sample!(vals)),
            ChannelValues::Scale(vals) => Sample::S(vec_sample!(vals)),
            ChannelValues::Rotation(vals) => {
                let q = if step {
                    vals[v(i)]
                } else if cubic {
                    let h = |q: Quat| glam::Vec4::from(q);
                    let r = hermite(h(vals[v(i)]), h(vals[i * 3 + 2]), h(vals[v(next)]), h(vals[next * 3]), f, span);
                    Quat::from_vec4(r)
                } else {
                    vals[v(i)].slerp(vals[v(next)], f)
                };
                Sample::R(q.normalize())
            }
            ChannelValues::Weights(vals, n) => {
                let n = *n;
                let get = |k: usize, j: usize| vals.get(k * n + j).copied().unwrap_or(0.0);
                let w = (0..n)
                    .map(|j| {
                        if step {
                            get(v(i), j)
                        } else if cubic {
                            hermite(get(v(i), j), get(i * 3 + 2, j), get(v(next), j), get(next * 3, j), f, span)
                        } else {
                            get(v(i), j) + (get(v(next), j) - get(v(i), j)) * f
                        }
                    })
                    .collect();
                Sample::W(w)
            }
        }
    }
}

/// Options for [`Animator::play`].
#[derive(Clone, Copy, Debug)]
pub struct PlayOpts {
    pub looped: bool,
    pub speed: f32,
    /// Crossfade duration from the current clip (s).
    pub fade: f32,
}

impl Default for PlayOpts {
    fn default() -> Self {
        Self { looped: true, speed: 1.0, fade: 0.2 }
    }
}

impl PlayOpts {
    pub fn looped() -> Self {
        Self::default()
    }
    pub fn once() -> Self {
        Self { looped: false, ..Self::default() }
    }
    pub fn fade(mut self, s: f32) -> Self {
        self.fade = s;
        self
    }
    pub fn speed(mut self, s: f32) -> Self {
        self.speed = s;
        self
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Playing {
    pub clip: usize,
    pub time: f32,
    pub speed: f32,
    pub looped: bool,
}

impl Playing {
    fn local_time(&self, dur: f32) -> f32 {
        if dur <= 0.0 {
            0.0
        } else if self.looped {
            self.time.rem_euclid(dur)
        } else {
            self.time.clamp(0.0, dur)
        }
    }
}

/// Component on a glTF model root: its clips and what is playing.
pub struct Animator {
    pub clips: Arc<Vec<AnimationClip>>,
    /// glTF node index -> entity.
    pub nodes: Vec<Entity>,
    /// glTF node index -> entities with a Deformer (the node's mesh primitives), for weight channels.
    pub node_deformers: Vec<Vec<Entity>>,
    pub current: Option<Playing>,
    /// Clip fading out: (state, fade elapsed, fade duration).
    pub previous: Option<(Playing, f32, f32)>,
    /// Rest pose (node transforms at instantiation) for nodes a clip does not animate.
    rest: Vec<Transform>,
}

impl Animator {
    pub(crate) fn new(clips: Arc<Vec<AnimationClip>>, nodes: Vec<Entity>, node_deformers: Vec<Vec<Entity>>, rest: Vec<Transform>) -> Self {
        Self { clips, nodes, node_deformers, current: None, previous: None, rest }
    }

    pub fn clip_index(&self, name: &str) -> Option<usize> {
        self.clips.iter().position(|c| c.name == name)
    }

    pub fn clip_names(&self) -> Vec<&str> {
        self.clips.iter().map(|c| c.name.as_str()).collect()
    }

    /// Starts `name` (crossfading from the current clip). Returns false if there is no such clip.
    pub fn play(&mut self, name: &str, o: PlayOpts) -> bool {
        let Some(i) = self.clip_index(name) else { return false };
        if let Some(c) = &self.current {
            if c.clip == i && c.looped == o.looped {
                if let Some(c) = self.current.as_mut() {
                    c.speed = o.speed;
                }
                return true; // already playing
            }
        }
        let next = Playing { clip: i, time: 0.0, speed: o.speed, looped: o.looped };
        self.previous = match (self.current.take(), o.fade > 0.0) {
            (Some(c), true) => Some((c, 0.0, o.fade)),
            _ => None,
        };
        self.current = Some(next);
        true
    }

    pub fn stop(&mut self) {
        self.current = None;
        self.previous = None;
    }

    /// The current non-looping clip reached its end.
    pub fn finished(&self) -> bool {
        self.current.as_ref().is_some_and(|c| !c.looped && c.time >= self.clips[c.clip].duration)
    }

    pub fn current_name(&self) -> Option<&str> {
        self.current.as_ref().map(|c| self.clips[c.clip].name.as_str())
    }

    fn advance(&mut self, dt: f32) {
        if let Some(c) = self.current.as_mut() {
            c.time += dt * c.speed;
        }
        if let Some((p, e, d)) = self.previous.as_mut() {
            p.time += dt * p.speed;
            *e += dt;
            if *e >= *d {
                self.previous = None;
            }
        }
    }
}

/// CPU source of a deformable primitive (shared by all instances of a model).
#[derive(Debug)]
pub struct DeformSource {
    pub base: Vec<Vertex>,
    /// Per vertex: 4 joint indices into the skin's joint list + weights.
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    /// Morph targets: (position deltas, normal deltas or empty).
    pub targets: Vec<(Vec<[f32; 3]>, Vec<[f32; 3]>)>,
}

/// Component on a mesh entity whose vertices are computed every frame.
pub struct Deformer {
    pub src: Arc<DeformSource>,
    /// Joint entities + inverse bind matrices (None: morph targets only).
    pub skin: Option<(Vec<Entity>, Arc<Vec<Mat4>>)>,
    /// Morph target weights (animated by weight channels or set by the game).
    pub weights: Vec<f32>,
    /// This instance's own GPU mesh.
    pub mesh: MeshId,
    out: Vec<Vertex>,
    /// Skip work when nothing changed since the last upload.
    last_key: Option<Vec<f32>>,
}

impl Deformer {
    pub(crate) fn new(src: Arc<DeformSource>, skin: Option<(Vec<Entity>, Arc<Vec<Mat4>>)>, weights: Vec<f32>, mesh: MeshId) -> Self {
        Self { src, skin, weights, mesh, out: vec![], last_key: None }
    }
}

/// Fixed step: advance clip time.
pub(crate) fn advance_all(world: &mut World, dt: f32) {
    for a in world.query_mut::<&mut Animator>() {
        a.advance(dt);
    }
}

fn apply_sample(world: &World, a: &Animator, ch: &Channel, s: Sample, w: f32, pose: &mut [Option<Transform>]) {
    let Some(&e) = a.nodes.get(ch.node) else { return };
    let base = pose[ch.node].unwrap_or(a.rest[ch.node]);
    let mut t = base;
    match s {
        Sample::T(v) => t.translation = base.translation.lerp(v, w),
        Sample::R(q) => t.rotation = base.rotation.slerp(q, w),
        Sample::S(v) => t.scale = base.scale.lerp(v, w),
        Sample::W(ws) => {
            for &d in a.node_deformers.get(ch.node).map(|v| v.as_slice()).unwrap_or(&[]) {
                if let Ok(mut def) = world.get::<&mut Deformer>(d) {
                    for (k, x) in ws.iter().enumerate() {
                        if let Some(dst) = def.weights.get_mut(k) {
                            // w = 1 must give x exactly: `dst + (x - dst)` can differ in the last bit and
                            // flip between two values, re-deforming (and re-uploading) the mesh every frame
                            *dst = if w >= 1.0 { *x } else { *dst + (x - *dst) * w };
                        }
                    }
                }
            }
            let _ = e;
            return;
        }
    }
    pose[ch.node] = Some(t);
}

/// Per frame (after `Game::update`, before transform propagation): write the sampled pose into
/// the node transforms (crossfading previous -> current).
pub(crate) fn apply_poses(world: &mut World) {
    let roots: Vec<Entity> = world.query::<(Entity, &Animator)>().iter().map(|(e, _)| e).collect();
    for r in roots {
        let Ok(a) = world.get::<&Animator>(r) else { continue };
        let Some(cur) = a.current.clone() else { continue };
        let mut pose: Vec<Option<Transform>> = vec![None; a.nodes.len()];
        let fade_w = a.previous.as_ref().map(|(_, e, d)| (e / d).clamp(0.0, 1.0)).unwrap_or(1.0);
        if let Some((prev, _, _)) = &a.previous {
            let clip = &a.clips[prev.clip];
            let t = prev.local_time(clip.duration);
            for ch in &clip.channels {
                apply_sample(world, &a, ch, ch.sample(t), 1.0, &mut pose);
            }
        }
        let clip = &a.clips[cur.clip];
        let t = cur.local_time(clip.duration);
        for ch in &clip.channels {
            apply_sample(world, &a, ch, ch.sample(t), if a.previous.is_some() { fade_w } else { 1.0 }, &mut pose);
        }
        let writes: Vec<(Entity, Transform)> = pose.iter().enumerate().filter_map(|(i, p)| p.map(|t| (a.nodes[i], t))).collect();
        drop(a);
        for (e, t) in writes {
            if let Ok(mut dst) = world.get::<&mut Transform>(e) {
                *dst = t;
            }
        }
    }
}

/// Per frame (after transform propagation): skin + morph every deformer and upload its vertices.
pub(crate) fn deform_all(ctx: &mut Ctx) {
    let list: Vec<Entity> = ctx.world.query::<(Entity, &Deformer)>().iter().map(|(e, _)| e).collect();
    for e in list {
        let mesh_m = ctx.world.get::<&GlobalTransform>(e).map(|g| g.0).unwrap_or(Mat4::IDENTITY);
        let Ok(mut d) = ctx.world.get::<&mut Deformer>(e) else { continue };
        // joint matrices in the mesh's local space
        let mut mats: Vec<Mat4> = vec![];
        if let Some((joints, ibm)) = &d.skin {
            let inv = mesh_m.inverse();
            for (k, j) in joints.iter().enumerate() {
                let g = ctx.world.get::<&GlobalTransform>(*j).map(|g| g.0).unwrap_or(Mat4::IDENTITY);
                mats.push(inv * g * ibm.get(k).copied().unwrap_or(Mat4::IDENTITY));
            }
        }
        // change detection: joint matrices + weights
        let mut key: Vec<f32> = d.weights.clone();
        for m in &mats {
            key.extend_from_slice(&m.to_cols_array());
        }
        if d.last_key.as_ref() == Some(&key) {
            continue;
        }
        let src = d.src.clone();
        let n = src.base.len();
        let mut out = std::mem::take(&mut d.out);
        out.clear();
        out.extend_from_slice(&src.base);
        // morph targets
        for (t, (dp, dn)) in src.targets.iter().enumerate() {
            let w = d.weights.get(t).copied().unwrap_or(0.0);
            if w == 0.0 {
                continue;
            }
            for i in 0..n.min(dp.len()) {
                for c in 0..3 {
                    out[i].pos[c] += dp[i][c] * w;
                }
                if let Some(nv) = dn.get(i) {
                    for c in 0..3 {
                        out[i].normal[c] += nv[c] * w;
                    }
                }
            }
        }
        // skinning
        if !mats.is_empty() {
            for i in 0..n {
                let (js, ws) = (src.joints.get(i).copied().unwrap_or([0; 4]), src.weights.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]));
                let mut m = Mat4::ZERO;
                let mut wsum = 0.0;
                for k in 0..4 {
                    if ws[k] > 0.0 {
                        m += *mats.get(js[k] as usize).unwrap_or(&Mat4::IDENTITY) * ws[k];
                        wsum += ws[k];
                    }
                }
                if wsum <= 0.0 {
                    continue;
                }
                if (wsum - 1.0).abs() > 1e-3 {
                    m *= 1.0 / wsum;
                }
                let p = m.transform_point3(Vec3::from(out[i].pos));
                let nn = m.transform_vector3(Vec3::from(out[i].normal)).normalize_or(Vec3::Y);
                out[i].pos = p.into();
                out[i].normal = nn.into();
            }
        } else if !src.targets.is_empty() {
            for v in out.iter_mut() {
                v.normal = Vec3::from(v.normal).normalize_or(Vec3::Y).into();
            }
        }
        let mesh = d.mesh;
        d.last_key = Some(key);
        ctx.renderer.update_mesh_vertices(mesh, &out);
        d.out = out;
    }
}

impl Ctx {
    /// Plays clip `name` on the model rooted at `root` (the entity `load_gltf` returned).
    /// False if the model has no such clip (or has not finished loading).
    pub fn play_animation(&mut self, root: Entity, name: &str, opts: PlayOpts) -> bool {
        match self.world.get::<&mut Animator>(root) {
            Ok(mut a) => a.play(name, opts),
            Err(_) => false,
        }
    }

    /// Sets morph target `index` of every deformable mesh below `e` (including `e`).
    pub fn set_morph_weight(&mut self, e: Entity, index: usize, w: f32) {
        for d in crate::scene::descendants(&self.world, e) {
            if let Ok(mut def) = self.world.get::<&mut Deformer>(d) {
                if let Some(x) = def.weights.get_mut(index) {
                    *x = w;
                }
            }
        }
    }

    /// AI API `animation`.
    pub(crate) fn animation_command(&mut self, a: &Value) -> Result<Value, String> {
        let describe = |world: &World, e: Entity| -> Value {
            let anim = world.get::<&Animator>(e).ok();
            let deformers: Vec<Value> = crate::scene::descendants(world, e)
                .into_iter()
                .filter_map(|d| {
                    world.get::<&Deformer>(d).ok().map(|def| {
                        json!({"entity": d.id(), "skinned": def.skin.is_some(), "joints": def.skin.as_ref().map_or(0, |s| s.0.len()),
                               "vertices": def.src.base.len(), "morph_targets": def.src.targets.len(), "weights": def.weights})
                    })
                })
                .collect();
            json!({
                "id": e.id(),
                "clips": anim.as_ref().map(|a| a.clips.iter().map(|c| json!({"name": c.name, "duration": c.duration, "channels": c.channels.len()})).collect::<Vec<_>>()),
                "playing": anim.as_ref().and_then(|a| a.current.as_ref().map(|c| json!({"clip": a.clips[c.clip].name, "time": c.time, "speed": c.speed, "looped": c.looped}))),
                "fading_from": anim.as_ref().and_then(|a| a.previous.as_ref().map(|(p, e, d)| json!({"clip": a.clips[p.clip].name, "elapsed": e, "duration": d}))),
                "finished": anim.as_ref().map(|a| a.finished()),
                "deformers": deformers,
            })
        };
        let Some(id) = a["id"].as_u64() else {
            let list: Vec<Value> = self.world.query::<(Entity, &Animator)>().iter().map(|(e, _)| e).collect::<Vec<_>>().into_iter().map(|e| describe(&self.world, e)).collect();
            return Ok(json!({"animators": list}));
        };
        let e = self.world.iter().map(|r| r.entity()).find(|e| e.id() == id as u32).ok_or(format!("no entity with id {id}"))?;
        if a["stop"].as_bool() == Some(true) {
            if let Ok(mut an) = self.world.get::<&mut Animator>(e) {
                an.stop();
            }
        }
        if let Some(name) = a["play"].as_str() {
            let o = PlayOpts {
                looped: a["looped"].as_bool().unwrap_or(true),
                speed: a["speed"].as_f64().unwrap_or(1.0) as f32,
                fade: a["fade"].as_f64().unwrap_or(0.2) as f32,
            };
            if !self.play_animation(e, name, o) {
                return Err(format!("entity {id} has no clip '{name}' (or no Animator yet)"));
            }
        }
        if let (Some(i), Some(w)) = (a["morph"].as_u64(), a["weight"].as_f64()) {
            self.set_morph_weight(e, i as usize, w as f32);
        }
        Ok(describe(&self.world, e))
    }
}

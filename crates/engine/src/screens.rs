//! Screens (title, menu, gameplay, pause, ...) as a stack, and scenes as groups of entities that
//! are loaded and unloaded together.
//!
//! **Scenes.** Every entity belongs to a scene ([`SceneTag`]). `ctx.begin_scene("level1")` makes
//! "level1" current: everything spawned from then on (also glTF nodes, physics bodies, static
//! colliders from `physics.add_static`) is tagged with it. `ctx.unload_scene("level1")` despawns it
//! all again. Entities spawned while no scene is current belong to `"persistent"` and stay.
//!
//! **Screens.** [`ScreenStack`] implements [`Game`], so a game can be a stack of [`Screen`]s:
//!
//! ```ignore
//! engine::run(ScreenStack::new(MyData::default(), Box::new(Title))
//!     .register("title", || Box::new(Title))       // optional: lets the AI API jump to screens
//!     .register("play", || Box::new(Play::default())));
//!
//! impl Screen<MyData> for Title {
//!     fn name(&self) -> &'static str { "title" }
//!     fn enter(&mut self, ctx: &mut Ctx, _d: &mut MyData) { /* spawn the title scene */ }
//!     fn update(&mut self, ctx: &mut Ctx, _d: &mut MyData, _dt: f32) -> Trans<MyData> {
//!         if ctx.input.key_pressed("Enter") { Trans::Switch(Box::new(Play::default())) } else { Trans::None }
//!     }
//! }
//! ```
//!
//! Each screen gets its own scene, named like the screen: whatever it spawns in `enter`/`update` is
//! unloaded when it exits (`Switch`/`Pop`), unless [`Screen::keep_scene`] says otherwise. A pushed
//! overlay (pause menu) covers the screen below: only the top screen updates, and with
//! [`Screen::pauses_below`] (default true) the simulation (fixed steps + physics) is paused.
//!
//! AI API: `screens` shows the stack and scenes; `screens {switch|push: name}` / `{pop: true}`
//! changes screens through the registered constructors. Other commands go to the top screen first.

use crate::app::{Ctx, Game};
use crate::physics::rapier::prelude::ColliderHandle;
use hecs::Entity;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Scene membership of an entity (index into [`Scenes`]). 0 = "persistent".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneTag(pub u32);

pub struct Scenes {
    names: Vec<String>,
    current: u32,
    statics: HashMap<u32, Vec<ColliderHandle>>,
    loaded_at: HashMap<u32, u64>,
}

impl Default for Scenes {
    fn default() -> Self {
        Self { names: vec!["persistent".into()], current: 0, statics: HashMap::new(), loaded_at: HashMap::new() }
    }
}

impl Scenes {
    /// Scene name of a tag.
    pub fn name_of(&self, tag: SceneTag) -> &str {
        self.names.get(tag.0 as usize).map(|s| s.as_str()).unwrap_or("?")
    }

    pub fn current(&self) -> &str {
        &self.names[self.current as usize]
    }
    pub fn id(&self, name: &str) -> Option<u32> {
        self.names.iter().position(|n| n == name).map(|i| i as u32)
    }
    pub(crate) fn current_id(&self) -> u32 {
        self.current
    }

    pub(crate) fn id_or_new(&mut self, name: &str) -> u32 {
        self.id(name).unwrap_or_else(|| {
            self.names.push(name.to_string());
            (self.names.len() - 1) as u32
        })
    }

    /// Tags every untagged entity (and new static colliders) with the current scene.
    pub(crate) fn tag_new(&mut self, world: &mut hecs::World, physics: &mut crate::physics::Physics) {
        let fresh: Vec<Entity> = world.query::<hecs::Without<Entity, &SceneTag>>().iter().collect();
        for e in fresh {
            // children inherit their parent's scene (glTF nodes arriving after a scene switch)
            let tag = world
                .get::<&crate::scene::Parent>(e)
                .ok()
                .and_then(|p| world.get::<&SceneTag>(p.0).ok().map(|t| *t))
                .unwrap_or(SceneTag(self.current));
            let _ = world.insert_one(e, tag);
        }
        let new_statics = std::mem::take(&mut physics.new_statics);
        if !new_statics.is_empty() {
            self.statics.entry(self.current).or_default().extend(new_statics);
        }
    }

    /// (scene, entity count, static colliders) for the AI API.
    pub fn describe(&self, world: &hecs::World) -> Value {
        let mut counts = vec![0usize; self.names.len()];
        for t in world.query::<&SceneTag>().iter() {
            if let Some(c) = counts.get_mut(t.0 as usize) {
                *c += 1;
            }
        }
        let list: Vec<Value> = self
            .names
            .iter()
            .enumerate()
            .filter(|(i, _)| counts[*i] > 0 || *i as u32 == self.current || self.statics.get(&(*i as u32)).is_some_and(|s| !s.is_empty()))
            .map(|(i, n)| {
                json!({"name": n, "entities": counts[i], "static_colliders": self.statics.get(&(i as u32)).map_or(0, |s| s.len()),
                       "loaded_at_step": self.loaded_at.get(&(i as u32))})
            })
            .collect();
        json!({"current": self.current(), "scenes": list})
    }
}

impl Ctx {
    /// Makes `name` the current scene; entities spawned from now on belong to it.
    pub fn begin_scene(&mut self, name: &str) {
        self.scenes.tag_new(&mut self.world, &mut self.physics);
        let id = self.scenes.id_or_new(name);
        self.scenes.loaded_at.entry(id).or_insert(self.time.step);
        self.scenes.current = id;
    }

    /// Back to the "persistent" scene (nothing spawned from now on is unloaded automatically).
    pub fn end_scene(&mut self) {
        self.scenes.tag_new(&mut self.world, &mut self.physics);
        self.scenes.current = 0;
    }

    /// Despawns every entity of scene `name` (with children and physics bodies) and its static
    /// colliders. Returns the number of entities removed. The current scene falls back to
    /// "persistent" if it was `name`.
    pub fn unload_scene(&mut self, name: &str) -> usize {
        self.scenes.tag_new(&mut self.world, &mut self.physics);
        let Some(id) = self.scenes.id(name) else { return 0 };
        if id == 0 {
            crate::log_warn!("unload_scene: the persistent scene cannot be unloaded");
            return 0;
        }
        let doomed: Vec<Entity> = self.world.query::<(Entity, &SceneTag)>().iter().filter(|(_, t)| t.0 == id).map(|(e, _)| e).collect();
        // one pass over the hierarchy for all of them (despawn() per entity would rebuild the
        // child map every time: O(n²) for big streamed cells)
        let all = crate::scene::descendants_of(&self.world, &doomed);
        let n = all.len();
        for e in all {
            self.physics.remove(&self.world, e);
            let _ = self.world.despawn(e);
        }
        for h in self.scenes.statics.remove(&id).unwrap_or_default() {
            self.physics.world.remove_collider(h);
        }
        self.scenes.loaded_at.remove(&id);
        if self.scenes.current == id {
            self.scenes.current = 0;
        }
        crate::log_info!("scene '{name}' unloaded ({n} entities)");
        n
    }

    /// Assets still loading (glTF models). A loading screen waits for 0.
    pub fn assets_pending(&self) -> usize {
        self.models.pending.len()
    }
}

/// What the top screen wants after its update.
pub enum Trans<D> {
    None,
    /// Cover the current screen with a new one (pause menu, dialog).
    Push(Box<dyn Screen<D>>),
    /// Remove the top screen; the one below resumes.
    Pop,
    /// Replace the top screen (title -> play), unloading its scene.
    Switch(Box<dyn Screen<D>>),
    /// Clear the whole stack and start over with this screen.
    Reset(Box<dyn Screen<D>>),
}

/// One screen. `D` is data shared by all screens (save state, settings, score...).
#[allow(unused_variables)]
pub trait Screen<D>: 'static {
    fn name(&self) -> &'static str;
    fn enter(&mut self, ctx: &mut Ctx, data: &mut D) {}
    fn exit(&mut self, ctx: &mut Ctx, data: &mut D) {}
    /// Another screen was pushed on top of this one.
    fn pause(&mut self, ctx: &mut Ctx, data: &mut D) {}
    /// The screen above was popped.
    fn resume(&mut self, ctx: &mut Ctx, data: &mut D) {}
    fn fixed_update(&mut self, ctx: &mut Ctx, data: &mut D) {}
    fn update(&mut self, ctx: &mut Ctx, data: &mut D, dt: f32) -> Trans<D> {
        Trans::None
    }
    /// For overlays: pause the simulation of the screens below while this one is on top. The
    /// overlay's own particle emitters (its scene) keep simulating; the AI API `pause` stops them too.
    fn pauses_below(&self) -> bool {
        true
    }
    /// Keep this screen's entities when it exits (default: unload its scene).
    fn keep_scene(&self) -> bool {
        false
    }
    fn debug_state(&self, ctx: &Ctx, data: &D) -> Value {
        Value::Null
    }
    fn debug_command(&mut self, ctx: &mut Ctx, data: &mut D, name: &str, args: &Value) -> Option<Result<Value, String>> {
        None
    }
    fn debug_commands(&self) -> Vec<(&'static str, &'static str)> {
        vec![]
    }
}

type Factory<D> = Box<dyn Fn() -> Box<dyn Screen<D>>>;

pub struct ScreenStack<D: 'static> {
    pub data: D,
    stack: Vec<Box<dyn Screen<D>>>,
    initial: Option<Box<dyn Screen<D>>>,
    registry: Vec<(&'static str, Factory<D>)>,
    /// `ctx.time.paused` before overlays paused it, per stack depth.
    paused_before: Vec<bool>,
    history: Vec<(u64, String)>,
}

impl<D: 'static> ScreenStack<D> {
    pub fn new(data: D, initial: Box<dyn Screen<D>>) -> Self {
        Self { data, stack: vec![], initial: Some(initial), registry: vec![], paused_before: vec![], history: vec![] }
    }

    /// Named constructor so the AI API (`screens {switch: name}`) can open this screen.
    pub fn register(mut self, name: &'static str, f: impl Fn() -> Box<dyn Screen<D>> + 'static) -> Self {
        self.registry.push((name, Box::new(f)));
        self
    }

    pub fn top_name(&self) -> Option<&'static str> {
        self.stack.last().map(|s| s.name())
    }

    fn note(&mut self, ctx: &Ctx, what: String) {
        crate::log_info!("screens: {what}");
        self.history.push((ctx.time.step, what));
        if self.history.len() > 32 {
            self.history.remove(0);
        }
    }

    fn enter_top(&mut self, ctx: &mut Ctx) {
        let s = self.stack.last_mut().unwrap();
        ctx.begin_scene(s.name());
        s.enter(ctx, &mut self.data);
        ctx.scenes.tag_new(&mut ctx.world, &mut ctx.physics);
    }

    fn exit_top(&mut self, ctx: &mut Ctx) {
        let mut s = self.stack.pop().unwrap();
        ctx.begin_scene(s.name());
        s.exit(ctx, &mut self.data);
        if !s.keep_scene() {
            ctx.unload_scene(s.name());
        }
        if s.pauses_below() {
            if let Some(p) = self.paused_before.pop() {
                ctx.time.paused = p;
            }
        }
    }

    pub fn apply(&mut self, ctx: &mut Ctx, t: Trans<D>) {
        match t {
            Trans::None => {}
            Trans::Push(s) => {
                self.note(ctx, format!("push {}", s.name()));
                if let Some(top) = self.stack.last_mut() {
                    ctx.begin_scene(top.name());
                    top.pause(ctx, &mut self.data);
                }
                if s.pauses_below() && !self.stack.is_empty() {
                    self.paused_before.push(ctx.time.paused);
                    ctx.time.paused = true;
                }
                self.stack.push(s);
                self.enter_top(ctx);
            }
            Trans::Pop => {
                if self.stack.len() <= 1 {
                    crate::log_warn!("screens: pop on the last screen ignored");
                    return;
                }
                self.note(ctx, format!("pop {}", self.top_name().unwrap_or("")));
                self.exit_top(ctx);
                let top = self.stack.last_mut().unwrap();
                ctx.begin_scene(top.name());
                top.resume(ctx, &mut self.data);
            }
            Trans::Switch(s) => {
                self.note(ctx, format!("switch {} -> {}", self.top_name().unwrap_or("-"), s.name()));
                if !self.stack.is_empty() {
                    self.exit_top(ctx);
                }
                self.stack.push(s);
                self.enter_top(ctx);
            }
            Trans::Reset(s) => {
                self.note(ctx, format!("reset -> {}", s.name()));
                while !self.stack.is_empty() {
                    self.exit_top(ctx);
                }
                self.stack.push(s);
                self.enter_top(ctx);
            }
        }
        // the top screen's particles keep running while it pauses the screens below
        ctx.time.live_scene = match self.stack.last() {
            Some(top) if self.stack.len() > 1 && top.pauses_below() && ctx.time.paused => Some(ctx.scenes.id_or_new(top.name())),
            _ => None,
        };
    }

    fn make(&self, name: &str) -> Result<Box<dyn Screen<D>>, String> {
        self.registry
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, f)| f())
            .ok_or_else(|| format!("no screen '{name}' registered; known: {:?}", self.registry.iter().map(|r| r.0).collect::<Vec<_>>()))
    }
}

impl<D: 'static> Game for ScreenStack<D> {
    fn init(&mut self, ctx: &mut Ctx) {
        ctx.end_scene(); // whatever exists now (camera, lights from elsewhere) is persistent
        if let Some(s) = self.initial.take() {
            self.apply(ctx, Trans::Switch(s));
        }
    }

    fn fixed_update(&mut self, ctx: &mut Ctx) {
        if let Some(s) = self.stack.last_mut() {
            s.fixed_update(ctx, &mut self.data);
        }
    }

    fn update(&mut self, ctx: &mut Ctx, dt: f32) {
        let Some(s) = self.stack.last_mut() else { return };
        let t = s.update(ctx, &mut self.data, dt);
        self.apply(ctx, t);
    }

    fn debug_commands(&self) -> Vec<(&'static str, &'static str)> {
        let mut v = vec![("screens", "Screen stack + scenes. args: {switch?: name, push?: name, pop?: true, reset?: name}")];
        if let Some(s) = self.stack.last() {
            v.extend(s.debug_commands());
        }
        v
    }

    fn inactive_debug_commands(&self) -> Vec<(&'static str, &'static str, &'static str)> {
        // registered screens other than the top one (constructors only build the struct;
        // nothing happens until on_enter)
        let top = self.top_name();
        let mut v = vec![];
        for (n, f) in &self.registry {
            if Some(*n) == top {
                continue;
            }
            for (c, d) in f().debug_commands() {
                v.push((c, d, *n));
            }
        }
        v
    }

    fn debug_command(&mut self, ctx: &mut Ctx, name: &str, a: &Value) -> Option<Result<Value, String>> {
        if name == "screens" {
            let r = (|| -> Result<Value, String> {
                if let Some(n) = a["switch"].as_str() {
                    let s = self.make(n)?;
                    self.apply(ctx, Trans::Switch(s));
                } else if let Some(n) = a["push"].as_str() {
                    let s = self.make(n)?;
                    self.apply(ctx, Trans::Push(s));
                } else if let Some(n) = a["reset"].as_str() {
                    let s = self.make(n)?;
                    self.apply(ctx, Trans::Reset(s));
                } else if a["pop"].as_bool() == Some(true) {
                    self.apply(ctx, Trans::Pop);
                }
                Ok(json!({
                    "stack": self.stack.iter().map(|s| s.name()).collect::<Vec<_>>(),
                    "top": self.top_name(),
                    "registered": self.registry.iter().map(|r| r.0).collect::<Vec<_>>(),
                    "paused": ctx.time.paused,
                    "history": self.history.iter().map(|(s, w)| json!({"step": s, "what": w})).collect::<Vec<_>>(),
                    "scenes": ctx.scenes.describe(&ctx.world),
                }))
            })();
            return Some(r);
        }
        let s = self.stack.last_mut()?;
        s.debug_command(ctx, &mut self.data, name, a)
    }

    fn debug_state(&self, ctx: &Ctx) -> Value {
        let top = self.stack.last().map(|s| s.debug_state(ctx, &self.data)).unwrap_or(Value::Null);
        json!({"screen": self.top_name(), "stack": self.stack.iter().map(|s| s.name()).collect::<Vec<_>>(), "state": top})
    }
}

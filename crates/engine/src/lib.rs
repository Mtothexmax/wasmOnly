//! wasmOnly engine — a Chrome-only 3D engine for `wasm32-unknown-unknown`.
//!
//! Design rules (see README.md / AGENTS.md):
//! * One way per task: WebGPU + WGSL, glTF (.glb) only, hecs ECS, rapier physics.
//! * The browser is the platform: images via `createImageBitmap`, UI via DOM, audio via Web Audio.
//! * AI-first: every piece of runtime state is reachable through `window.engine.call(...)`
//!   (see [`debug`]), logs are structured and buffered, simulation is deterministic and steppable.
//!
//! Game code implements [`Game`] and calls [`run`].

pub mod animation;
pub mod app;
pub mod audio;
pub mod batch;
pub mod character;
pub mod debug;
pub mod debugmode;
pub mod geom;
pub mod gltf_loader;
pub mod gpu_timing;
pub(crate) mod gpu_cull;
pub mod input;
pub mod interp;
pub mod jobs;
pub mod lights;
pub mod log;
pub mod mesh;
pub mod panel;
pub mod particles;
pub mod physics;
pub(crate) mod post;
pub mod render_texture;
pub mod renderer;
pub mod save;
pub mod screens;
pub mod sky;
pub mod scene;
pub mod streaming;
pub mod terrain;
pub mod texgen;
pub mod touch;
pub mod ui;
pub mod vegetation;
pub mod water;
pub mod zfight;

pub use app::{run, Ctx, Game, Time};
pub use glam;
pub use hecs;
pub use serde_json;

/// Everything a game usually needs.
pub mod prelude {
    pub use crate::animation::{Animator, PlayOpts};
    pub use crate::app::{run, Ctx, Game, Time};
    pub use crate::batch::Motion;
    pub use crate::geom::{MeshBuilder, Opening};
    pub use crate::gltf_loader::GltfOpts;
    pub use crate::audio::synth::{self, Sfx, SfxPreset, Wave};
    pub use crate::audio::{Fx, MusicOpts, Play, SoundId, VoiceId};
    pub use crate::character::{CharacterController, ThirdPersonCamera};
    pub use crate::input::{Input, MouseButton};
    pub use crate::lights::SpotLight;
    pub use crate::water::Ocean;
    pub use crate::panel::{Panel, PanelId, WorldPanel};
    pub use crate::particles::{ParticleEmitter, ParticlePreset, ParticleShape, ParticleSystem};
    pub use crate::physics::{rapier, Body, Physics};
    pub use crate::render_texture::{RenderTextureCamera, RenderTextureId, RenderTextureOpts};
    pub use crate::renderer::{AlphaMode, Lod, Material, MaterialId, MeshId, TextureId};
    pub use crate::scene::{
        Camera, GlobalTransform, MeshRenderer, Name, Parent, PointLight, RenderLayer, Transform, WorldLabel,
    };
    pub use crate::screens::{SceneTag, Screen, ScreenStack, Trans};
    pub use crate::terrain::{NoiseParams, ScatterRule, SplatRules, Terrain, TerrainDesc};
    pub use crate::streaming::Cell;
    pub use crate::texgen::{TexGen, TexPreset};
    pub use glam::DVec3;
    pub use crate::{log_error, log_info, log_warn};
    pub use glam::{EulerRot, Mat4, Quat, Vec2, Vec3, Vec4};
    pub use hecs::{Entity, World};
    pub use serde_json::{json, Value};
}

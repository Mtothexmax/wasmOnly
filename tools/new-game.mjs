#!/usr/bin/env node
// Scaffolds a standalone game on the wasmOnly engine in its own folder / repository.
//
//   node tools/new-game.mjs <dir> [--name my-game] [--engine-git [--rev <commit|tag>]]
//
// Creates <dir> with: Cargo.toml (engine as a path dependency to this checkout, or as a git
// dependency with --engine-git), the pinned toolchain + wasm flags, a ScreenStack game skeleton
// (title + play screen, terrain-free ground, character controller, third-person camera, save game,
// debug_state, a game command), assets/, tests/smoke.txt, .gitignore, .mcp.json (bridge MCP server)
// and CLAUDE.md/AGENTS.md telling an agent where the engine docs and tools are.
//
// Then:  node <engine>/tools/dev.mjs watch --game-dir <dir>     (build + serve + live reload)
//        node <engine>/tools/bridge.mjs launch                   (debug Chrome)
//        node <engine>/tools/bridge.mjs run <dir>/tests/smoke.txt
// Existing files are never overwritten.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const flag = (n) => args.includes(n);
const dir = args.find((a, i) => !a.startsWith('--') && !(i > 0 && ['--name', '--rev'].includes(args[i - 1])));
if (!dir) {
  console.error('usage: node tools/new-game.mjs <dir> [--name my-game] [--engine-git [--rev <commit|tag>]]');
  process.exit(2);
}
const target = path.resolve(dir);
const name = (opt('--name') ?? path.basename(target)).toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '') || 'game';
const fwd = (p) => p.split(path.sep).join('/');
const enginePath = fwd(path.relative(target, path.join(ROOT, 'crates', 'engine')));
const engineAbs = fwd(ROOT);
const ws = fs.readFileSync(path.join(ROOT, 'Cargo.toml'), 'utf8');
const dep = (n) => (ws.match(new RegExp(`^${n}\\s*=\\s*(.+)$`, 'm')) ?? [])[1];
const engineDep = flag('--engine-git')
  ? `{ git = "https://github.com/Mtothexmax/wasmOnly", ${opt('--rev') ? `rev = "${opt('--rev')}"` : 'branch = "main"'} }`
  : `{ path = "${enginePath}" }`;

const files = {
  'Cargo.toml': `[package]
name = "${name}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
engine = ${engineDep}
# must match the installed wasm-bindgen CLI exactly (same pin as the engine)
wasm-bindgen = ${dep('wasm-bindgen')}
glam = ${dep('glam')}
serde = ${dep('serde')}
serde_json = ${dep('serde_json')}

# standalone: not part of another workspace
[workspace]

[profile.dev]
opt-level = 1
debug = true

[profile.dev.package."*"]
opt-level = 3
debug = false

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
`,
  'rust-toolchain.toml': fs.readFileSync(path.join(ROOT, 'rust-toolchain.toml'), 'utf8'),
  '.cargo/config.toml': fs.readFileSync(path.join(ROOT, '.cargo', 'config.toml'), 'utf8'),
  '.gitignore': '/target\n/dist\n/logs\n/tests/shots\n/.chrome-debug-profile\n',
  'assets/.gitkeep': '',
  '.mcp.json': JSON.stringify({ mcpServers: { wasmonly: { command: 'node', args: [`${engineAbs}/tools/bridge.mjs`, 'mcp'] } } }, null, 2) + '\n',
  'src/lib.rs': `//! ${name} — a game on the wasmOnly engine. Start: CLAUDE.md.

use engine::physics::rapier::prelude::{ColliderBuilder, RigidBodyBuilder};
use engine::prelude::*;
use engine::renderer::Material;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn main() {
    engine::run(
        ScreenStack::new(Data::default(), Box::new(Title))
            .register("title", || Box::new(Title))
            .register("play", || Box::new(PlayScreen::default())),
    );
}

/// Saved progress (localStorage slot "progress").
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Progress {
    plays: u32,
}

/// Data shared by all screens.
#[derive(Default)]
struct Data {
    progress: Progress,
    loaded: bool,
}

impl Data {
    fn ensure(&mut self, ctx: &mut Ctx) {
        if !self.loaded {
            self.loaded = true;
            ctx.save.set_namespace("${name}", 1);
            self.progress = ctx.save.read("progress").unwrap_or_default();
        }
    }
}

struct Title;

impl Screen<Data> for Title {
    fn name(&self) -> &'static str {
        "title"
    }
    fn enter(&mut self, ctx: &mut Ctx, data: &mut Data) {
        data.ensure(ctx);
        ctx.world.spawn((Transform::from_xyz(0.0, 2.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y), Camera::default()));
        ctx.ui.set("menu", &format!("<div class='panel menu'><h1>${name}</h1><div>Rounds: {}</div><button data-action='play'>Play (Enter)</button></div>", data.progress.plays));
    }
    fn exit(&mut self, ctx: &mut Ctx, _data: &mut Data) {
        ctx.ui.set("menu", "");
    }
    fn update(&mut self, ctx: &mut Ctx, _data: &mut Data, _dt: f32) -> Trans<Data> {
        let clicked = ctx.ui.take_actions().iter().any(|a| a == "play");
        if clicked || ctx.input.key_pressed("Enter") {
            return Trans::Switch(Box::new(PlayScreen::default()));
        }
        Trans::None
    }
}

#[derive(Default)]
struct PlayScreen {
    hero: Option<Entity>,
    camera: Option<Entity>,
}

impl Screen<Data> for PlayScreen {
    fn name(&self) -> &'static str {
        "play"
    }

    fn enter(&mut self, ctx: &mut Ctx, data: &mut Data) {
        data.ensure(ctx);
        data.progress.plays += 1;
        let _ = ctx.save.write("progress", &data.progress);
        // ground
        let grass = ctx.generate_texture(&TexGen::new(TexPreset::Grass).size(256).seed(1));
        let ground_mat = ctx.renderer.add_material(Material { texture: grass, ..Material::color(0xffffff).rough(0.95) });
        let ground = ctx.spawn_mesh("ground", ctx.renderer.builtin.cube, ground_mat, Transform::from_xyz(0.0, -0.5, 0.0).with_scale(Vec3::new(60.0, 1.0, 60.0)));
        ctx.physics.add(&mut ctx.world, ground, RigidBodyBuilder::fixed(), ColliderBuilder::cuboid(30.0, 0.5, 30.0));
        // a few boxes to walk around and jump on
        for k in 0..5 {
            let b = ctx.spawn_cube("box", Vec3::new(-4.0 + k as f32 * 2.0, 0.5, -4.0), Vec3::ONE, 0xe9c46a);
            ctx.physics.add(&mut ctx.world, b, RigidBodyBuilder::fixed(), ColliderBuilder::cuboid(0.5, 0.5, 0.5));
        }
        // hero: character controller + a capsule-ish body (replace with a glTF model as a child)
        let hero = ctx.spawn_character("hero", Vec3::new(0.0, 0.0, 4.0), CharacterController::default());
        let body = ctx.spawn_sphere("hero-body", Vec3::ZERO, 0.4, 0x2a9d8f);
        let _ = ctx.world.insert_one(body, Parent(hero));
        if let Ok(mut t) = ctx.world.get::<&mut Transform>(body) {
            *t = Transform::default().with_scale(Vec3::new(0.7, 1.7, 0.7));
        }
        self.hero = Some(hero);
        self.camera = Some(ctx.world.spawn((Transform::from_xyz(0.0, 4.0, 10.0), Camera::default(), ThirdPersonCamera::following(hero))));
        ctx.ui.set("help", "<div class='panel'><kbd>WASD</kbd> walk · <kbd>Space</kbd> jump · <kbd>Right drag</kbd> camera · <kbd>Esc</kbd> menu</div>");
    }

    fn exit(&mut self, ctx: &mut Ctx, _data: &mut Data) {
        ctx.ui.set("help", "");
    }

    fn fixed_update(&mut self, ctx: &mut Ctx, _data: &mut Data) {
        // fixed_update: held keys (input.key); taps/edges (key_pressed, key_down_or_pressed) are latched per step
        let yaw = self.camera.and_then(|c| ctx.world.get::<&ThirdPersonCamera>(c).ok().map(|c| c.yaw)).unwrap_or(0.0);
        let (axis, jump, run) = (ctx.input.move_axis(), ctx.input.key_down_or_pressed("Space"), ctx.input.key("ShiftLeft"));
        if let Some(mut cc) = self.hero.and_then(|h| ctx.world.get::<&mut CharacterController>(h).ok()) {
            cc.drive(axis, yaw);
            cc.jump = jump;
            cc.run = run;
        }
    }

    fn update(&mut self, ctx: &mut Ctx, _data: &mut Data, _dt: f32) -> Trans<Data> {
        if ctx.input.key_pressed("Escape") {
            return Trans::Switch(Box::new(Title));
        }
        Trans::None
    }

    fn debug_commands(&self) -> Vec<(&'static str, &'static str)> {
        vec![("teleport", "Move the hero. args: {pos: [x, y, z]}")]
    }

    fn debug_command(&mut self, ctx: &mut Ctx, _data: &mut Data, name: &str, a: &Value) -> Option<Result<Value, String>> {
        match name {
            "teleport" => {
                let p = a["pos"].as_array()?;
                let v = |i: usize| p.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                if let Some(mut t) = self.hero.and_then(|h| ctx.world.get::<&mut Transform>(h).ok()) {
                    t.translation = Vec3::new(v(0), v(1), v(2));
                }
                Some(Ok(json!({"pos": [v(0), v(1), v(2)]})))
            }
            _ => None,
        }
    }

    /// Everything an agent needs to assert progress (\`call game_state\` -> {screen, stack, state}).
    fn debug_state(&self, ctx: &Ctx, data: &Data) -> Value {
        let hero = self.hero.and_then(|h| {
            let t = ctx.world.get::<&Transform>(h).ok()?.translation;
            let cc = ctx.world.get::<&CharacterController>(h).ok()?;
            Some(json!({"pos": t.to_array(), "grounded": cc.grounded, "jumps": cc.jumps}))
        });
        json!({"hero": hero, "progress": data.progress})
    }
}
`,
  'tests/smoke.txt': `# Smoke test — run with: node ${engineAbs}/tools/bridge.mjs run tests/smoke.txt
reload
expect game_state.screen == "title"
action play
wait_for game_state.screen == "play" 3000
pause
step 60
expect game_state.state.hero.grounded == true
key Space
step 12
expect game_state.state.hero.jumps == 1
step 60
expect game_state.state.hero.grounded == true
resume
frames 10
screenshot shots/play.png
`,
  'CLAUDE.md': `# ${name}

A game on the **wasmOnly** engine (Rust -> wasm32, WebGPU, Chrome only, AI-first).
Read AGENTS.md here first, then the engine's own docs:

- ${engineAbs}/AGENTS.md — dev loop, AI runtime API (every state is JSON via \`window.engine.call\`), code map, rules
- ${engineAbs}/.claude/README.md — skills: engine loop, verify gameplay, new game (Game/Ctx/Screen API,
  open world, generators, animation, particles, ...), debug rendering, assets
- ${engineAbs}/HANDOFF.md — engine overview for a new project, porting notes (three.js -> wasmOnly)
- Reference game: ${engineAbs}/games/demo (title / play / pause screens, open world in src/world.rs)
`,
  'AGENTS.md': `# AGENTS.md — ${name}

Engine checkout: \`${engineAbs}\` (tools + docs live there; the engine crate is ${flag('--engine-git') ? 'a git dependency' : `a path dependency: ${enginePath}`}).

## Loop
\`\`\`bash
node ${engineAbs}/tools/dev.mjs watch --game-dir .          # build + serve http://localhost:8080 + rebuild/reload (keep running)
node ${engineAbs}/tools/bridge.mjs launch                   # debug Chrome (CDP :9222), muted
node ${engineAbs}/tools/bridge.mjs reload                   # after a rebuild: errors/warnings as JSON
node ${engineAbs}/tools/bridge.mjs call help                # every engine + game command
node ${engineAbs}/tools/bridge.mjs run tests/smoke.txt      # scripted test (input, step, expect, screenshot)
node ${engineAbs}/tools/bridge.mjs check                    # build check: errors, z-fighting
node ${engineAbs}/tools/bridge.mjs perf                     # frame times, CPU phases, GPU passes, verdict
\`\`\`
MCP: \`.mcp.json\` registers the bridge as server \`wasmonly\` (engine_call, reload, screenshot, run_script, perf, check).

## Rules
- Verify with JSON (\`game_state\`, \`entity\`, \`logs\`) and test scripts, not screenshots alone.
- Put checks you will run again into \`tests/*.txt\`.
- \`debug_state\` must expose what a test needs; add game commands in \`debug_command\`.
- Input: \`input.key\` = held; \`key_pressed\` = edge (in \`fixed_update\` latched since the last step, so taps are never lost);
  jump/fire-style actions: \`key_down_or_pressed\`. Every action needs a scripted test with a short tap (\`key Space\` + \`step\`).
- When the user reports a bug: ask them to press \` (on German keyboards ^) before and after reproducing it; read the
  newest \`logs/debug-*.log\` (keys, camera, per-second perf + characters + game_state).
- Engine bugs/missing features: fix them in the engine checkout (${engineAbs}) so every game benefits;
  keep game-specific content (generators, levels) in this repo.
`,
};

let wrote = 0;
for (const [rel, content] of Object.entries(files)) {
  const p = path.join(target, rel);
  if (fs.existsSync(p)) {
    console.log(JSON.stringify({ skipped: rel, reason: 'exists' }));
    continue;
  }
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, content);
  wrote++;
}
console.log(JSON.stringify({ created: fwd(target), name, files: wrote, engine: flag('--engine-git') ? 'git' : enginePath,
  next: [`node ${engineAbs}/tools/dev.mjs watch --game-dir ${fwd(target)}`, `node ${engineAbs}/tools/bridge.mjs launch`, `node ${engineAbs}/tools/bridge.mjs run ${fwd(target)}/tests/smoke.txt`] }));

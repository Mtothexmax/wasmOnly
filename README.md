# wasmOnly — Chrome-only 3D-Engine in Rust/WASM, AI-first

Eine Engine, die ausschließlich für `wasm32-unknown-unknown` baut und in Chrome läuft
(WebGPU über `web-sys`, WGSL, hecs, rapier). Sie ist „AI-first“: Jeder Laufzeitzustand ist als
JSON über `window.engine.call(...)` abfragbar und steuerbar, und eine Chrome-Debug-Bridge
(CDP) macht das per CLI und als MCP-Server für Agenten nutzbar. Anleitungen für Agenten liegen in
[`.claude/`](.claude/README.md).

## Schnellstart

Voraussetzungen: `rustup` (die Toolchain ist in `rust-toolchain.toml` auf **1.99.0** festgelegt und
wird automatisch installiert), `wasm-bindgen-cli` **in exakt der Version aus `Cargo.toml`**
(aktuell 0.2.129), Node ≥ 22 und Chrome. Für das optionale Thread-Backend kommt ein festgelegtes
Nightly dazu (siehe [Threads](#threads-optional-nightly)).

```bash
node tools/make-sample-assets.mjs   # einmalig: tree.glb, crate.glb, rock.obj erzeugen
node tools/dev.mjs watch            # bauen, http://localhost:8080 ausliefern, bei Änderungen neu bauen und neu laden
node tools/bridge.mjs launch        # Chrome mit Remote-Debugging (Port 9222) öffnen, stumm (--sound für Ton)
node tools/bridge.mjs call stats    # Engine abfragen
```

Langsames Netz testen: `curl -X POST "http://localhost:8080/__throttle?kbps=48&match=assets/music"` drosselt
nur passende Pfade (`kbps=0` hebt es auf). Der Dev-Server beantwortet Range-Requests, Musik streamt also
wie auf einem echten Webserver.

Debug-Modus: <kbd>`</kbd> bzw. <kbd>^</kbd> drücken. Ab dann landen Tastendrücke, Mausklicks und die
Kameraposition (jede Sekunde) in `logs/debug-*.log` im Projektordner; nochmal drücken beendet die Datei.

Debugging in den DevTools: Der Dev-Build enthält DWARF für Engine- und Spielcode. Mit der
Chrome-Erweiterung „C/C++ DevTools Support (DWARF)“ setzt du Breakpoints direkt in `.rs`-Dateien.

## Entscheidungen und Stand

| Bereich | Entscheidung | Stand |
|---|---|---|
| Target | `wasm32-unknown-unknown`, `wasm-bindgen`, `web-sys` | ✅ |
| SIMD | `+simd128` für den ganzen Build (glam und rapier nutzen es) | ✅ |
| Grafik | WebGPU direkt über `web-sys` (unstable cfg), WGSL inkl. Compute-Shader | ✅ |
| Rendering | Forward-PBR (GGX), 4× MSAA, ACES-Tonemapping | ✅ |
| Transparenz | `AlphaMode::Opaque / Mask(cutoff) / Blend` und `double_sided` (auch aus glTF). Mask = Alpha-Test mit ausgestanzten Schatten, batcht und wird auf der GPU gecullt wie deckende Geometrie. Blend = eigener Durchgang nach allem Deckenden, pro Instanz von hinten nach vorne sortiert, ohne Tiefenschreiben | ✅ |
| Licht | 1 Richtungslicht mit Shadow-Map (3×3 PCF, texel-snapped), bis zu 16 Punktlichter, Hemisphären-Ambient | ✅ · Clustered Lighting, CSM, Spots und IBL fehlen |
| Culling/Draws | GPU-driven: Ein Compute-Shader macht Frustum- und Hi-Z-Occlusion-Culling pro Instanz und schreibt die Argumente für `drawIndexedIndirect`. Fallback: CPU-Frustum-Culling mit direkten Draws | ✅ |
| Instancing | Ein Storage-Buffer für alle Instanzen, ein Draw pro (Textur, Mesh) | ✅ (5000 Würfel: 60 fps, 7 Draw Calls) |
| LOD | `Lod`-Komponente, Distanz relativ zur Objektgröße. Automatische Stufen per Vertex-Clustering für glTF-Meshes ab 256 Dreiecken; die eingebaute Kugel hat drei Stufen | ✅ |
| Szene | hecs-ECS, `Transform` + `Parent` → `GlobalTransform` | ✅ |
| Assets | Laufzeit nur glTF 2.0 (`.glb`), Bilder über `createImageBitmap`, Cache pro URL, alles asynchron. Mipmaps werden auf der GPU erzeugt, 8× anisotrope Filterung | ✅ · KTX2 und Textur-Streaming fehlen |
| Fremdformate | `tools/asset-cook` (russimp/Assimp, nativ): OBJ/FBX/DAE/3DS/… → `.glb` | ✅ |
| Physik | rapier3d 0.36 als festes Modul, `Body`-Komponente, Raycasts, CCD | ✅ |
| GUI/HUD | DOM-Overlay: HUD-Slots, `data-action`-Klicks, `WorldLabel` (weltverankert) | ✅ |
| 3D-GUI | **World Panels**: 2D-Widgets (Label, Button, Balken, Rechteck) per Canvas2D in einer Textur auf einem 3D-Quad. Klicks per Strahl und UV, Verdeckung durch Physik, gleiche Aktions-Queue wie das HUD | ✅ · SVG→Textur (resvg) fehlt |
| Input | Tastatur, Pointer, Rad, Pointer Lock, Gamepad; DOM-UI und World Panels haben Vorrang | ✅ |
| Touch (Android) | Mehrere Finger über Pointer Events. Gesten: virtueller Joystick (linke Bildschirmseite, fließt in `move_axis()`), Umschauen, Pinch-Zoom, Tippen (klickt World Panels), Halten, Wischen. Joystick wird gezeichnet. iOS ist bewusst kein Ziel | ✅ |
| Audio | Web Audio: Master-Gain, 3D-Sound über `PannerNode` (HRTF), Listener folgt der Kamera, Dateien per `decodeAudioData`. Verliert der Tab den Fokus, wird der AudioContext angehalten und läuft bei Fokus an derselben Stelle weiter (abschaltbar: `set_pause_on_blur(false)`) | ✅ |
| Musik | **Streaming**: lange Tracks laufen über ein `<audio>`-Element mit HTTP-Range-Requests, starten sobald genug gepuffert ist und blockieren nie das Spiel. Überblendung, Prefetch, eigener Musik- und SFX-Bus mit getrennter Lautstärke, Autoplay-Retry, Fokus-Pause. `ctx.audio.play_music(url, MusicOpts)` | ✅ |
| Prozedurales Audio | Deterministischer Synth (`engine::audio::synth`): Sinus/Rechteck/Sägezahn/Dreieck, weißes/rosa/braunes Rauschen, ADSR, Pitch-Slide, Vibrato, Arpeggio, Obertöne (Klangfarbe), Bitcrush. 16 Presets (coin, laser, explosion, powerup, …, Schleifen wind/rain/engine). Effekte beim Abspielen als Web-Audio-Nodes: Tief-/Hoch-/Bandpass, Echo mit Feedback, Hall (Convolver), Verzerrung, Pitch/Detune, Stereo-Pan | ✅ |
| Prozedurale Texturen | `engine::texgen` mit dem `noise`-Crate: 18 nahtlos kachelnde Presets (noise, clouds, marble, wood, stone, grass, sand, dirt, lava, water, bricks, tiles, metal, rust, camo, checker und mit Alpha: blades, leaves), parallel über das Job-System | ✅ |
| Animation | Skinning, Clips, Morph Targets | ❌ |
| Threads | Optionales Backend: Web-Worker-Pool (rayon über `wasm-bindgen-rayon`), Shared Memory (SharedArrayBuffer), Atomics. Die Job-API `engine::jobs` ist in beiden Builds gleich | ✅ · Asset-Aufbereitung in Workern fehlt |
| Build | `tools/dev.mjs` (cargo + wasm-bindgen, `wasm-opt` falls vorhanden). Toolchains festgelegt: Stable 1.99.0, Thread-Backend nightly-2026-11-01 | ✅ |
| AI-Tooling | `window.engine.call`, strukturierte Logs, deterministisches Stepping, CDP-Bridge, MCP, `.claude/`-Skills | ✅ |
| Debug-Modus | Taste <kbd>`</kbd> (Taste links neben 1, auf deutschen Tastaturen <kbd>^</kbd> oder Shift+´): protokolliert jeden Tasten- und Mausdruck und einmal pro Sekunde die Kamera als JSON-Zeilen in `logs/debug-JJJJMMTT-hhmmss.log` (über den Dev-Server), zeigt ein DEBUG-Badge | ✅ |

## Architektur

```
games/<spiel>  (Rust, implementiert engine::Game)
   │
crates/engine ── app (Loop, Ctx) · scene · input · physics · gltf_loader · ui · panel · audio · jobs · debug (AI-API) · log
   │
renderer ─────── WebGPU: [Compute: Culling] → Shadow-Pass → PBR-Pass (MSAA, indirect) → [Compute: Hi-Z]
   │
Browser ──────── web-sys: WebGPU · DOM · OffscreenCanvas · Web Audio · fetch · createImageBitmap · Gamepad · Worker
```

Frame: `requestAnimationFrame` → fester Schritt mit 1/60 s (`Game::fixed_update`, danach
rapier) → World Panels (Hover, Klicks → Aktionen) → `Game::update(dt)` → Transforms propagieren →
Panels neu zeichnen → Audio-Listener → rendern → `WorldLabel`s positionieren.

## Ein Spiel schreiben

```rust
use engine::prelude::*;
use engine::physics::rapier::prelude::*;

#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() { engine::run(MyGame::default()); }

#[derive(Default)] struct MyGame;

impl Game for MyGame {
    fn init(&mut self, ctx: &mut Ctx) {
        ctx.world.spawn((Transform::from_xyz(0., 3., 8.).looking_at(Vec3::ZERO, Vec3::Y), Camera::default()));
        let b = ctx.spawn_cube("box", Vec3::new(0., 4., 0.), Vec3::ONE, 0xff8800);
        ctx.physics.add(&mut ctx.world, b, RigidBodyBuilder::dynamic(), ColliderBuilder::cuboid(0.5, 0.5, 0.5));
        ctx.load_gltf("assets/tree.glb", Transform::from_xyz(2., 0., 0.));
        ctx.ui.set("title", "<div class='panel'>Hallo</div>");

        // anklickbare 3D-GUI: Klick auf "Start" landet als Aktion "start" in ctx.ui.take_actions()
        let panel = Panel::new(512, 256).label("t", [24., 20., 460., 40.], "Hallo 3D", 32.).button("go", [24., 140., 220., 70.], "Start", "start");
        ctx.spawn_panel("menu", panel, Transform::from_xyz(0., 2., 0.).with_scale(Vec3::new(2., 1., 1.)));

        // Sound ohne Asset-Dateien
        ctx.audio.add_samples("zap", &synth::zap(), synth::RATE);
        ctx.audio.play_named("zap", Play::new().at(Vec3::ZERO));
    }
}
```

Das Beispiel `games/demo` („Tumble“) nutzt alle Systeme: Ball steuern, Kistenturm von der Insel
stoßen, schießen, HUD-Buttons, ein Terminal als World Panel (Neustart, Schatten, Zeitlupe, Ton),
3D-Sound mit Effekten (Laser mit Echo, Aufprall mit Hall, Wind als Ambience), prozedurale Texturen für
Boden, Klippe und Metall, glTF-Bäume, gekochte Felsen und dynamische glTF-Kisten.

## Threads (optional, Nightly)

Der Engine-Kern baut auf Stable. Nur das Thread-Backend braucht Nightly, weil die Standardbibliothek
mit `+atomics,+bulk-memory` neu gebaut werden muss (`-Z build-std`). Die Version ist in `Cargo.toml`
unter `[workspace.metadata.wasmonly] threads-toolchain` festgelegt:

```bash
rustup toolchain install nightly-2026-11-01 --profile minimal --component rust-src --target wasm32-unknown-unknown
node tools/dev.mjs build --threads     # oder: watch --threads; Ausgabe in target/threads
```

Das Ergebnis nutzt gemeinsamen Speicher (SharedArrayBuffer, braucht COOP/COEP, die der Dev-Server
sendet) und startet vor `Game::init` einen Worker-Pool mit `navigator.hardwareConcurrency` Threads.
Spielcode benutzt `engine::jobs::{par_map, par_for_each_mut}`: ohne Backend seriell, mit Backend
parallel, mit identischem Ergebnis. `stats.jobs` zeigt, was aktiv ist. Das Feature `threads-physics`
lässt zusätzlich rapier parallel rechnen; das ist schneller, aber die Reihenfolge der Kontakte ist
zwischen Läufen nicht garantiert.

## Asset-Cooker (russimp)

Assimp ist C++ und lässt sich nicht nach `wasm32-unknown-unknown` bauen. Deshalb läuft russimp
als natives Build-Werkzeug, und die Engine sieht zur Laufzeit nur `.glb`:

```bash
cargo run --release --manifest-path tools/asset-cook/Cargo.toml -- games/demo/assets-src/rock.obj games/demo/assets/rock.glb
cargo run --release --manifest-path tools/asset-cook/Cargo.toml -- --batch games/demo/assets-src games/demo/assets
```

`russimp-sys` braucht beim Bauen **libclang** (für bindgen): LLVM installieren
(`winget install LLVM.LLVM`) oder `LIBCLANG_PATH` auf ein Verzeichnis mit `libclang.dll` setzen.
Das Feature `prebuilt` lädt die vorgebaute Assimp-Bibliothek herunter, CMake ist dafür nicht nötig.

`russimp-sys` 2.0.2 bringt bindgen 0.63 mit. Mit libclang ab Version 16 erzeugt das leere
`aiNode`/`aiScene`-Structs. Deshalb liegt unter `tools/asset-cook/vendor/russimp-sys` eine Kopie,
die nur bindgen auf 0.72 anhebt (eingebunden über `[patch.crates-io]`).

## AI-API (Auszug)

`engine.call("help")` listet alle Befehle. Die wichtigsten: `stats`, `entities`, `entity`, `set`,
`spawn`, `despawn`, `pause`, `resume`, `step`, `time_scale`, `seed`, `input`, `logs`, `render`,
`camera`, `raycast`, `pick`, `ui`, `debug_mode`, `music`, `panels`, `panel_click`, `audio`, `audio_play`, `audio_stop`, `audio_set`, `sfx`, `texgen`,
`assets`, `game_state` und die eigenen Befehle des Spiels. Siehe [AGENTS.md](AGENTS.md).

---
name: cook-assets
description: Convert OBJ, FBX, DAE, 3DS, PLY, STL and other Assimp formats to .glb with the native russimp cooker, and load them in a game. Use when adding 3D models or when a model fails to load.
---

# Asset cooking (russimp / Assimp → .glb)

The engine loads **only `.glb`** at runtime. Assimp is C++ and cannot target wasm32, so
`tools/asset-cook` is a native build-time tool (own Cargo workspace, never part of the wasm build).

## Convert

```bash
# one file
cargo run --release --manifest-path tools/asset-cook/Cargo.toml -- games/demo/assets-src/rock.obj games/demo/assets/rock.glb
# a whole folder (assets-src -> assets)
cargo run --release --manifest-path tools/asset-cook/Cargo.toml -- --batch games/demo/assets-src games/demo/assets
# options: --scale <f>
```

Output is one JSON line per file:
`{"input":..,"ok":true,"output":..,"stats":{"triangles":108,"materials":2,"textures":0,"warnings":[]}}`.
Check `ok` and `warnings`.

Keep sources in `games/<game>/assets-src/`, cooked files in `games/<game>/assets/` (copied to `dist/assets/`).

## Load and check

```rust
let e = ctx.load_gltf("assets/rock.glb", Transform::from_xyz(0.0, 0.0, 0.0));
ctx.physics.add(&mut ctx.world, e, RigidBodyBuilder::fixed(), ColliderBuilder::ball(0.8));
```

```bash
node tools/bridge.mjs reload
node tools/bridge.mjs call assets      # models_ready lists the url; models_failed must be {}
node tools/bridge.mjs call entities '{"name":"rock"}'
```

## Build requirements / known issues

- Needs **libclang** (bindgen). Installed at `C:\Program Files\LLVM\bin` (`winget install LLVM.LLVM`).
  If not found: set `LIBCLANG_PATH="C:/Program Files/LLVM/bin"`.
- `russimp` uses the `prebuilt` feature (downloads Assimp, no CMake).
- `tools/asset-cook/vendor/russimp-sys` is a patched copy: upstream pins bindgen 0.63, which with
  libclang ≥ 16 emits empty (`_address: u8`) `aiNode`/`aiScene`/`aiFile`/`aiFileIO` and `russimp`
  fails to compile. The copy only bumps bindgen to 0.72. Do not remove the `[patch.crates-io]`.
- Running the built exe directly (not via `cargo run`) needs the Assimp DLL from
  `tools/asset-cook/target/release/build/russimp-sys-*/out/dylib/bin` on `PATH`.
- Textures: embedded textures are written into the `.glb`; external ones must sit next to the source file.

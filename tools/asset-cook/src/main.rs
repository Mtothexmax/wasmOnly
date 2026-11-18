//! asset-cook — native build-time tool.
//!
//! Imports any format Assimp understands (OBJ, FBX, DAE, 3DS, PLY, STL, ...) via `russimp`
//! and writes a single self-contained `.glb` (glTF 2.0 binary) that the WASM engine loads.
//!
//! Why native: Assimp is C++ and does not build for `wasm32-unknown-unknown`. Following the
//! "one way per task" rule, the runtime only ever sees `.glb`; every other format is cooked here.
//!
//! Usage:
//!   cargo run --release -- <input> <output.glb> [--scale 1.0] [--name Model]
//!   cargo run --release -- --batch <input_dir> <output_dir>     (all known extensions)
//!
//! Output is machine-readable: one JSON line per cooked file on stdout (AI-first).

use russimp::material::{DataContent, Material, PropertyTypeInfo, TextureType};
use russimp::node::Node;
use russimp::scene::{PostProcess, Scene};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::rc::Rc;

const EXTENSIONS: &[&str] = &["obj", "fbx", "dae", "3ds", "ply", "stl", "blend", "x", "lwo", "ase", "off"];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: asset-cook <input> <output.glb> [--scale S]\n       asset-cook --batch <in_dir> <out_dir> [--scale S]");
        std::process::exit(2);
    }
    let scale = flag_value(&args, "--scale").map(|s| s.parse::<f32>().expect("--scale expects a number")).unwrap_or(1.0);
    let positional: Vec<&String> = {
        let mut out = vec![];
        let mut skip = false;
        for a in &args {
            if skip { skip = false; continue; }
            if a == "--scale" { skip = true; continue; }
            if a.starts_with("--") { continue; }
            out.push(a);
        }
        out
    };

    let mut failed = false;
    if args.iter().any(|a| a == "--batch") {
        let (input, output) = (Path::new(positional[0]), Path::new(positional[1]));
        std::fs::create_dir_all(output).expect("create output dir");
        for entry in std::fs::read_dir(input).expect("read input dir") {
            let p = entry.unwrap().path();
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
            if EXTENSIONS.contains(&ext.as_str()) {
                let out = output.join(p.file_stem().unwrap()).with_extension("glb");
                failed |= !report(&p, &out, cook(&p, &out, scale));
            }
        }
    } else {
        let (input, output) = (PathBuf::from(positional[0]), PathBuf::from(positional[1]));
        failed = !report(&input, &output, cook(&input, &output, scale));
    }
    std::process::exit(if failed { 1 } else { 0 });
}

fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1))
}

fn report(input: &Path, output: &Path, r: Result<Value, String>) -> bool {
    match r {
        Ok(stats) => {
            println!("{}", json!({"ok": true, "input": input, "output": output, "stats": stats}));
            true
        }
        Err(e) => {
            println!("{}", json!({"ok": false, "input": input, "error": e}));
            false
        }
    }
}

/// Builds the glTF JSON + one binary buffer.
struct GlbWriter {
    bin: Vec<u8>,
    buffer_views: Vec<Value>,
    accessors: Vec<Value>,
    images: Vec<Value>,
    textures: Vec<Value>,
}

impl GlbWriter {
    fn push_view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while self.bin.len() % 4 != 0 { self.bin.push(0); }
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        let mut v = json!({"buffer": 0, "byteOffset": offset, "byteLength": bytes.len()});
        if let Some(t) = target { v["target"] = json!(t); }
        self.buffer_views.push(v);
        self.buffer_views.len() - 1
    }

    fn push_vec3(&mut self, data: &[[f32; 3]], with_bounds: bool) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes())).collect();
        let view = self.push_view(&bytes, Some(34962));
        let mut acc = json!({"bufferView": view, "componentType": 5126, "count": data.len(), "type": "VEC3"});
        if with_bounds && !data.is_empty() {
            let mut min = [f32::MAX; 3];
            let mut max = [f32::MIN; 3];
            for v in data { for i in 0..3 { min[i] = min[i].min(v[i]); max[i] = max[i].max(v[i]); } }
            acc["min"] = json!(min);
            acc["max"] = json!(max);
        }
        self.accessors.push(acc);
        self.accessors.len() - 1
    }

    fn push_vec2(&mut self, data: &[[f32; 2]]) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes())).collect();
        let view = self.push_view(&bytes, Some(34962));
        self.accessors.push(json!({"bufferView": view, "componentType": 5126, "count": data.len(), "type": "VEC2"}));
        self.accessors.len() - 1
    }

    fn push_indices(&mut self, data: &[u32]) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        let view = self.push_view(&bytes, Some(34963));
        self.accessors.push(json!({"bufferView": view, "componentType": 5125, "count": data.len(), "type": "SCALAR"}));
        self.accessors.len() - 1
    }

    fn push_image(&mut self, bytes: &[u8], mime: &str) -> usize {
        let view = self.push_view(bytes, None);
        self.images.push(json!({"bufferView": view, "mimeType": mime}));
        self.textures.push(json!({"source": self.images.len() - 1, "sampler": 0}));
        self.textures.len() - 1
    }
}

fn cook(input: &Path, output: &Path, scale: f32) -> Result<Value, String> {
    let scene = Scene::from_file(
        input.to_str().ok_or("non-utf8 path")?,
        vec![
            PostProcess::Triangulate,
            PostProcess::GenerateSmoothNormals,
            PostProcess::JoinIdenticalVertices,
            PostProcess::ImproveCacheLocality,
            PostProcess::FlipUVs, // glTF has its UV origin top-left
            PostProcess::SortByPrimitiveType,
            PostProcess::ValidateDataStructure,
        ],
    )
    .map_err(|e| format!("assimp: {e:?}"))?;

    let base_dir = input.parent().unwrap_or(Path::new("."));
    let mut w = GlbWriter { bin: vec![], buffer_views: vec![], accessors: vec![], images: vec![], textures: vec![] };

    // Materials
    let mut warnings: Vec<String> = vec![];
    let materials: Vec<Value> = scene
        .materials
        .iter()
        .map(|m| material_to_gltf(m, base_dir, &mut w, &scene, &mut warnings))
        .collect();

    // Meshes (one glTF mesh per Assimp mesh; triangles only)
    let mut meshes = vec![];
    let mut tri_count = 0usize;
    for mesh in &scene.meshes {
        let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| [v.x * scale, v.y * scale, v.z * scale]).collect();
        let normals: Vec<[f32; 3]> = mesh.normals.iter().map(|v| [v.x, v.y, v.z]).collect();
        let indices: Vec<u32> = mesh.faces.iter().filter(|f| f.0.len() == 3).flat_map(|f| f.0.iter().copied()).collect();
        if indices.is_empty() {
            warnings.push(format!("mesh '{}' has no triangles (points/lines skipped)", mesh.name));
            meshes.push(json!({"name": mesh.name, "primitives": []}));
            continue;
        }
        tri_count += indices.len() / 3;
        let mut attributes = json!({ "POSITION": w.push_vec3(&positions, true) });
        if normals.len() == positions.len() {
            attributes["NORMAL"] = json!(w.push_vec3(&normals, false));
        }
        if let Some(Some(uvs)) = mesh.texture_coords.first() {
            let uv: Vec<[f32; 2]> = uvs.iter().map(|v| [v.x, v.y]).collect();
            attributes["TEXCOORD_0"] = json!(w.push_vec2(&uv));
        }
        let idx = w.push_indices(&indices);
        meshes.push(json!({
            "name": mesh.name,
            "primitives": [{ "attributes": attributes, "indices": idx, "material": mesh.material_index, "mode": 4 }]
        }));
    }
    // glTF forbids empty primitive arrays -> drop those meshes and remap.
    let mut remap = vec![None; meshes.len()];
    let mut kept = vec![];
    for (i, m) in meshes.into_iter().enumerate() {
        if !m["primitives"].as_array().unwrap().is_empty() {
            remap[i] = Some(kept.len());
            kept.push(m);
        }
    }

    // Node hierarchy
    let mut nodes: Vec<Value> = vec![];
    let root = scene.root.clone().ok_or("scene has no root node")?;
    let root_idx = push_node(&root, &mut nodes, &remap);

    let mut gltf = json!({
        "asset": {"version": "2.0", "generator": "wasmOnly asset-cook (russimp/Assimp)"},
        "scene": 0,
        "scenes": [{"nodes": [root_idx]}],
        "nodes": nodes,
        "meshes": kept,
        "materials": materials,
        "accessors": w.accessors,
        "bufferViews": w.buffer_views,
        "buffers": [{"byteLength": 0}],
    });
    while w.bin.len() % 4 != 0 { w.bin.push(0); }
    gltf["buffers"][0]["byteLength"] = json!(w.bin.len());
    if !w.images.is_empty() {
        gltf["images"] = json!(w.images);
        gltf["textures"] = json!(w.textures);
        gltf["samplers"] = json!([{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]);
    }

    write_glb(output, &gltf, &w.bin)?;
    Ok(json!({
        "meshes": scene.meshes.len(), "triangles": tri_count, "materials": scene.materials.len(),
        "textures": w.images.len(), "bytes": std::fs::metadata(output).map(|m| m.len()).unwrap_or(0),
        "warnings": warnings,
    }))
}

fn push_node(node: &Rc<Node>, nodes: &mut Vec<Value>, mesh_remap: &[Option<usize>]) -> usize {
    let idx = nodes.len();
    let t = &node.transformation; // row-major (a = row 0)
    let matrix = [t.a1, t.b1, t.c1, t.d1, t.a2, t.b2, t.c2, t.d2, t.a3, t.b3, t.c3, t.d3, t.a4, t.b4, t.c4, t.d4];
    nodes.push(json!({"name": node.name, "matrix": matrix}));
    let mut children: Vec<usize> = vec![];
    // glTF: one mesh per node -> extra Assimp meshes become child nodes.
    let node_meshes: Vec<usize> = node.meshes.iter().filter_map(|&m| mesh_remap[m as usize]).collect();
    if let Some((&first, rest)) = node_meshes.split_first() {
        nodes[idx]["mesh"] = json!(first);
        for &m in rest {
            nodes.push(json!({"name": format!("{}_mesh{}", node.name, m), "mesh": m}));
            children.push(nodes.len() - 1);
        }
    }
    for child in node.children.borrow().iter() {
        children.push(push_node(child, nodes, mesh_remap));
    }
    if !children.is_empty() {
        nodes[idx]["children"] = json!(children);
    }
    idx
}

fn prop_floats(m: &Material, key: &str) -> Option<Vec<f32>> {
    m.properties.iter().find(|p| p.key == key).and_then(|p| match &p.data {
        PropertyTypeInfo::FloatArray(f) => Some(f.clone()),
        _ => None,
    })
}

fn prop_string(m: &Material, key: &str) -> Option<String> {
    m.properties.iter().find(|p| p.key == key).and_then(|p| match &p.data {
        PropertyTypeInfo::String(s) => Some(s.clone()),
        _ => None,
    })
}

fn material_to_gltf(m: &Material, base_dir: &Path, w: &mut GlbWriter, scene: &Scene, warnings: &mut Vec<String>) -> Value {
    let name = prop_string(m, "?mat.name").unwrap_or_default();
    let color = prop_floats(m, "$clr.base").or_else(|| prop_floats(m, "$clr.diffuse")).unwrap_or(vec![0.8, 0.8, 0.8, 1.0]);
    let opacity = prop_floats(m, "$mat.opacity").and_then(|v| v.first().copied()).unwrap_or(1.0);
    let metallic = prop_floats(m, "$mat.metallicFactor").and_then(|v| v.first().copied()).unwrap_or(0.0);
    let roughness = prop_floats(m, "$mat.roughnessFactor").and_then(|v| v.first().copied()).unwrap_or_else(|| {
        // Phong shininess -> rough approximation
        prop_floats(m, "$mat.shininess").and_then(|v| v.first().copied()).map(|s| (2.0 / (s + 2.0)).sqrt().clamp(0.05, 1.0)).unwrap_or(0.8)
    });
    let emissive = prop_floats(m, "$clr.emissive").unwrap_or(vec![0.0, 0.0, 0.0]);
    let alpha = color.get(3).copied().unwrap_or(1.0) * opacity;

    let mut pbr = json!({
        "baseColorFactor": [color[0], color[1], color[2], alpha],
        "metallicFactor": metallic,
        "roughnessFactor": roughness,
    });

    let tex = m.textures.get(&TextureType::BaseColor).or_else(|| m.textures.get(&TextureType::Diffuse));
    if let Some(tex) = tex {
        let tex = tex.borrow();
        let image = match &tex.data {
            // Embedded compressed image (e.g. FBX/GLB embedded PNG/JPG)
            DataContent::Bytes(bytes) if !bytes.is_empty() => Some((bytes.clone(), mime_from_hint(&tex.ach_format_hint, bytes))),
            _ => {
                let fname = tex.filename.replace('\\', "/");
                if let Some(embedded) = fname.strip_prefix('*') {
                    // "*N" references scene.textures; russimp already resolved those into tex.data above.
                    warnings.push(format!("embedded texture {embedded} has unsupported raw texel data"));
                    None
                } else {
                    let p = base_dir.join(&fname);
                    match std::fs::read(&p) {
                        Ok(bytes) => {
                            let mime = mime_from_hint(p.extension().and_then(|e| e.to_str()).unwrap_or(""), &bytes);
                            Some((bytes, mime))
                        }
                        Err(e) => {
                            warnings.push(format!("texture '{}' not found: {e}", p.display()));
                            None
                        }
                    }
                }
            }
        };
        if let Some((bytes, mime)) = image {
            if mime == "image/png" || mime == "image/jpeg" || mime == "image/webp" {
                pbr["baseColorTexture"] = json!({"index": w.push_image(&bytes, mime)});
            } else {
                warnings.push(format!("texture of material '{name}' has unsupported type ({mime}); convert to PNG/JPG/WebP"));
            }
        }
    }
    let _ = scene;
    let mut mat = json!({"name": name, "pbrMetallicRoughness": pbr});
    if emissive.iter().take(3).any(|&e| e > 0.0) {
        mat["emissiveFactor"] = json!([emissive[0].min(1.0), emissive[1].min(1.0), emissive[2].min(1.0)]);
    }
    if alpha < 0.999 {
        mat["alphaMode"] = json!("BLEND");
    }
    mat
}

fn mime_from_hint(hint: &str, bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) { return "image/png"; }
    if bytes.starts_with(&[0xFF, 0xD8]) { return "image/jpeg"; }
    if bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" { return "image/webp"; }
    match hint.to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "application/octet-stream",
    }
}

fn write_glb(path: &Path, gltf: &Value, bin: &[u8]) -> Result<(), String> {
    let mut json_bytes = serde_json::to_vec(gltf).map_err(|e| e.to_string())?;
    while json_bytes.len() % 4 != 0 { json_bytes.push(b' '); }
    let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(bin);
    if let Some(dir) = path.parent() { std::fs::create_dir_all(dir).map_err(|e| e.to_string())?; }
    std::fs::write(path, out).map_err(|e| format!("write {}: {e}", path.display()))
}

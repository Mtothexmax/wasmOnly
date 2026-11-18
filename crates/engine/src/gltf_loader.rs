//! glTF 2.0 loading (`.glb`, or `.gltf` with external/embedded buffers). Fetched with `fetch`,
//! images decoded by the browser (`createImageBitmap`), uploaded once and cached per URL.
//! Other formats (OBJ, FBX, ...) are converted offline by `tools/asset-cook` (russimp/Assimp).

use crate::app::Ctx;
use crate::mesh::{MeshData, Vertex};
use crate::renderer::{AlphaMode, Lod, Material, MaterialId, MeshId, TextureId};
use crate::scene::{MeshRenderer, Name, Parent, Transform};
use glam::{Quat, Vec3, Vec4};
use hecs::Entity;
use js_sys::{Array, Uint8Array};
use std::collections::HashMap;
use wasm_bindgen::JsCast;
use web_sys::{Blob, BlobPropertyBag, ImageBitmap, Response};

/// glTF meshes with at least this many triangles get automatic LODs (vertex clustering).
pub const AUTO_LOD_MIN_TRIANGLES: usize = 256;

/// GPU-resident model, instantiable any number of times.
#[derive(Clone, Debug)]
pub struct ModelTemplate {
    pub url: String,
    pub nodes: Vec<NodeTemplate>,
    pub roots: Vec<usize>,
    pub triangles: usize,
}

#[derive(Clone, Debug)]
pub struct NodeTemplate {
    pub name: String,
    pub transform: Transform,
    /// (mesh, material, automatic LOD chain for meshes >= AUTO_LOD_MIN_TRIANGLES)
    pub prims: Vec<(MeshId, MaterialId, Option<Lod>)>,
    pub children: Vec<usize>,
}

#[derive(Default)]
pub struct ModelCache {
    pub ready: HashMap<String, ModelTemplate>,
    /// url -> roots waiting for the load to finish
    pub pending: HashMap<String, Vec<Entity>>,
    pub failed: HashMap<String, String>,
}

/// CPU-side result of parsing (before GPU upload).
struct Parsed {
    meshes: Vec<Vec<(MeshData, Option<usize>)>>,
    materials: Vec<(Material, Option<usize>)>,
    images: Vec<Option<ImageBitmap>>,
    nodes: Vec<(String, Transform, Option<usize>, Vec<usize>)>,
    roots: Vec<usize>,
}

pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    let win = web_sys::window().ok_or("no window")?;
    let resp: Response = win
        .fetch_with_str(url)
        .await
        .map_err(|e| format!("fetch {url}: {e:?}"))?
        .unchecked_into();
    if !resp.ok() {
        return Err(format!("fetch {url}: HTTP {}", resp.status()));
    }
    let buf = resp
        .array_buffer()
        .map_err(|e| format!("{e:?}"))?
        .await
        .map_err(|e| format!("{e:?}"))?;
    Ok(Uint8Array::new(&buf).to_vec())
}

async fn decode_image(bytes: &[u8], mime: &str) -> Result<ImageBitmap, String> {
    let arr = Array::new();
    arr.push(&Uint8Array::from(bytes));
    let opts = BlobPropertyBag::new();
    opts.set_type(mime);
    let blob =
        Blob::new_with_u8_array_sequence_and_options(&arr, &opts).map_err(|e| format!("{e:?}"))?;
    let win = web_sys::window().ok_or("no window")?;
    let bmp = win
        .create_image_bitmap_with_blob(&blob)
        .map_err(|e| format!("{e:?}"))?
        .await
        .map_err(|e| format!("createImageBitmap: {e:?}"))?;
    bmp.dyn_into::<ImageBitmap>()
        .map_err(|_| "not an ImageBitmap".to_string())
}

fn resolve_url(base: &str, rel: &str) -> String {
    match base.rfind('/') {
        Some(i) => format!("{}{}", &base[..=i], rel),
        None => rel.to_string(),
    }
}

async fn parse(url: &str, bytes: Vec<u8>) -> Result<Parsed, String> {
    let gltf = gltf::Gltf::from_slice(&bytes).map_err(|e| format!("glTF parse: {e}"))?;
    let doc = &gltf.document;

    // Buffers: GLB BIN chunk or external files. (data: URIs are not supported -> use .glb)
    let mut buffers: Vec<Vec<u8>> = vec![];
    for b in doc.buffers() {
        match b.source() {
            gltf::buffer::Source::Bin => {
                buffers.push(gltf.blob.clone().ok_or("GLB without BIN chunk")?)
            }
            gltf::buffer::Source::Uri(uri) if uri.starts_with("data:") => {
                return Err("data: URIs are not supported, export as .glb".into())
            }
            gltf::buffer::Source::Uri(uri) => {
                buffers.push(fetch_bytes(&resolve_url(url, uri)).await?)
            }
        }
    }

    let mut images = vec![];
    for img in doc.images() {
        let decoded = match img.source() {
            gltf::image::Source::View { view, mime_type } => {
                let buf = &buffers[view.buffer().index()];
                let slice = &buf[view.offset()..view.offset() + view.length()];
                decode_image(slice, mime_type).await
            }
            gltf::image::Source::Uri { uri, mime_type } => {
                let mime = mime_type.unwrap_or(if uri.ends_with(".png") {
                    "image/png"
                } else {
                    "image/jpeg"
                });
                match fetch_bytes(&resolve_url(url, uri)).await {
                    Ok(b) => decode_image(&b, mime).await,
                    Err(e) => Err(e),
                }
            }
        };
        match decoded {
            Ok(b) => images.push(Some(b)),
            Err(e) => {
                crate::log_warn!("{url}: image {} failed: {e}", img.index());
                images.push(None);
            }
        }
    }

    let materials = doc
        .materials()
        .map(|m| {
            let pbr = m.pbr_metallic_roughness();
            let tex = pbr
                .base_color_texture()
                .map(|t| t.texture().source().index());
            let mat = Material {
                name: m.name().unwrap_or("").to_string(),
                base_color: Vec4::from(pbr.base_color_factor()),
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor(),
                emissive: Vec3::from(m.emissive_factor()) * m.emissive_strength().unwrap_or(1.0),
                unlit: false,
                alpha_mode: match m.alpha_mode() {
                    gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                    gltf::material::AlphaMode::Mask => AlphaMode::Mask(m.alpha_cutoff().unwrap_or(0.5)),
                    gltf::material::AlphaMode::Blend => AlphaMode::Blend,
                },
                double_sided: m.double_sided(),
                ..Default::default()
            };
            (mat, tex)
        })
        .collect();

    let mut meshes = vec![];
    for mesh in doc.meshes() {
        let mut prims = vec![];
        for p in mesh.primitives() {
            if p.mode() != gltf::mesh::Mode::Triangles {
                crate::log_warn!(
                    "{url}: mesh '{}' primitive with mode {:?} skipped",
                    mesh.name().unwrap_or(""),
                    p.mode()
                );
                continue;
            }
            let reader = p.reader(|b| buffers.get(b.index()).map(|v| v.as_slice()));
            let Some(positions) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = positions.collect();
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
            let uvs: Option<Vec<[f32; 2]>> =
                reader.read_tex_coords(0).map(|t| t.into_f32().collect());
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|i| i.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());
            let mut md = MeshData {
                vertices: positions
                    .iter()
                    .enumerate()
                    .map(|(i, p)| Vertex {
                        pos: *p,
                        normal: normals
                            .as_ref()
                            .and_then(|n| n.get(i))
                            .copied()
                            .unwrap_or([0.0, 1.0, 0.0]),
                        uv: uvs
                            .as_ref()
                            .and_then(|u| u.get(i))
                            .copied()
                            .unwrap_or([0.0, 0.0]),
                    })
                    .collect(),
                indices,
            };
            if normals.is_none() {
                compute_normals(&mut md);
            }
            prims.push((md, p.material().index()));
        }
        meshes.push(prims);
    }

    let nodes = doc
        .nodes()
        .map(|n| {
            let (t, r, s) = n.transform().decomposed();
            let tr = Transform {
                translation: Vec3::from(t),
                rotation: Quat::from_array(r),
                scale: Vec3::from(s),
            };
            (
                n.name().unwrap_or("").to_string(),
                tr,
                n.mesh().map(|m| m.index()),
                n.children().map(|c| c.index()).collect(),
            )
        })
        .collect();
    let scene = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .ok_or("glTF has no scene")?;
    let roots = scene.nodes().map(|n| n.index()).collect();
    Ok(Parsed {
        meshes,
        materials,
        images,
        nodes,
        roots,
    })
}

fn compute_normals(md: &mut MeshData) {
    let mut acc = vec![Vec3::ZERO; md.vertices.len()];
    for t in md.indices.as_chunks::<3>().0 {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let (pa, pb, pc) = (
            Vec3::from(md.vertices[a].pos),
            Vec3::from(md.vertices[b].pos),
            Vec3::from(md.vertices[c].pos),
        );
        let n = (pb - pa).cross(pc - pa);
        acc[a] += n;
        acc[b] += n;
        acc[c] += n;
    }
    for (v, n) in md.vertices.iter_mut().zip(acc) {
        v.normal = n.normalize_or(Vec3::Y).into();
    }
}

fn upload(ctx: &mut Ctx, url: &str, p: Parsed) -> ModelTemplate {
    let mut tex_ids: Vec<Option<TextureId>> = vec![];
    for (i, img) in p.images.iter().enumerate() {
        tex_ids.push(img.as_ref().and_then(|b| {
            match ctx.renderer.add_texture_bitmap(b, &format!("{url}#img{i}")) {
                Ok(t) => Some(t),
                Err(e) => {
                    crate::log_warn!("{url}: texture upload failed: {e}");
                    None
                }
            }
        }));
    }
    let mat_ids: Vec<MaterialId> = p
        .materials
        .into_iter()
        .map(|(mut m, img)| {
            m.texture = img.and_then(|i| tex_ids.get(i).copied().flatten());
            ctx.renderer.add_material(m)
        })
        .collect();
    let default_mat = ctx.renderer.builtin.default_material;
    let mut triangles = 0;
    let mesh_prims: Vec<Vec<(MeshId, MaterialId, Option<Lod>)>> = p
        .meshes
        .iter()
        .enumerate()
        .map(|(mi, prims)| {
            prims
                .iter()
                .enumerate()
                .map(|(pi, (md, mat))| {
                    triangles += md.indices.len() / 3;
                    let name = format!("{url}#mesh{mi}.{pi}");
                    let (id, lod) = if md.indices.len() / 3 >= AUTO_LOD_MIN_TRIANGLES {
                        let (id, lod) = ctx.renderer.add_mesh_with_lods(&name, md, &[0.5, 0.25]);
                        (id, (lod.meshes.len() > 1).then_some(lod))
                    } else {
                        (ctx.renderer.add_mesh(&name, md), None)
                    };
                    (
                        id,
                        mat.and_then(|m| mat_ids.get(m).copied())
                            .unwrap_or(default_mat),
                        lod,
                    )
                })
                .collect()
        })
        .collect();
    let nodes = p
        .nodes
        .into_iter()
        .map(|(name, transform, mesh, children)| NodeTemplate {
            name,
            transform,
            prims: mesh.map(|m| mesh_prims[m].clone()).unwrap_or_default(),
            children,
        })
        .collect();
    ModelTemplate {
        url: url.to_string(),
        nodes,
        roots: p.roots,
        triangles,
    }
}

/// Spawns the template's node tree as children of `root`.
pub fn instantiate(ctx: &mut Ctx, root: Entity, model: &ModelTemplate) {
    fn spawn_node(ctx: &mut Ctx, parent: Entity, model: &ModelTemplate, idx: usize) {
        let n = &model.nodes[idx];
        let e = ctx
            .world
            .spawn((n.transform, Name(n.name.clone()), Parent(parent)));
        let attach = |ctx: &mut Ctx,
                      target: Entity,
                      (mesh, mat, lod): &(MeshId, MaterialId, Option<Lod>)| {
            let _ = ctx.world.insert_one(target, MeshRenderer::new(*mesh, *mat));
            if let Some(l) = lod {
                let _ = ctx.world.insert_one(target, l.clone());
            }
        };
        if n.prims.len() == 1 {
            attach(ctx, e, &n.prims[0]);
        } else {
            for p in &n.prims {
                let child = ctx.world.spawn((Transform::default(), Parent(e)));
                attach(ctx, child, p);
            }
        }
        for &c in &n.children {
            spawn_node(ctx, e, model, c);
        }
    }
    if !ctx.world.contains(root) {
        return; // despawned while loading
    }
    for &r in &model.roots {
        spawn_node(ctx, root, model, r);
    }
}

/// Starts loading `url` (or reuses the cache) and attaches the model under `root` when ready.
pub(crate) fn request(ctx: &mut Ctx, url: &str, root: Entity) {
    if let Some(m) = ctx.models.ready.get(url).cloned() {
        instantiate(ctx, root, &m);
        return;
    }
    if let Some(err) = ctx.models.failed.get(url) {
        crate::log_error!("model {url} previously failed: {err}");
        return;
    }
    if let Some(waiting) = ctx.models.pending.get_mut(url) {
        waiting.push(root);
        return;
    }
    ctx.models.pending.insert(url.to_string(), vec![root]);
    let me = ctx.me.clone();
    let url = url.to_string();
    wasm_bindgen_futures::spawn_local(async move {
        let t0 = crate::log::now_ms();
        let result = match fetch_bytes(&url).await {
            Ok(bytes) => parse(&url, bytes).await,
            Err(e) => Err(e),
        };
        let Some(ctx_rc) = me.upgrade() else { return };
        let Ok(mut ctx) = ctx_rc.try_borrow_mut() else {
            crate::log_error!("model {url}: engine busy when load finished");
            return;
        };
        let waiting = ctx.models.pending.remove(&url).unwrap_or_default();
        match result {
            Ok(parsed) => {
                let model = upload(&mut ctx, &url, parsed);
                crate::log_info!(
                    "model loaded: {url} ({} nodes, {} triangles, {:.0} ms)",
                    model.nodes.len(),
                    model.triangles,
                    crate::log::now_ms() - t0
                );
                for root in waiting {
                    instantiate(&mut ctx, root, &model);
                }
                ctx.models.ready.insert(url, model);
            }
            Err(e) => {
                crate::log_error!("model {url} failed: {e}");
                ctx.models.failed.insert(url, e);
            }
        }
    });
}

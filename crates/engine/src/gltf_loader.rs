//! glTF 2.0 loading (`.glb`, or `.gltf` with external/embedded buffers). Fetched with `fetch`,
//! images decoded by the browser (`createImageBitmap`), uploaded once and cached per URL.
//! Other formats (OBJ, FBX, ...) are converted offline by `tools/asset-cook` (russimp/Assimp).

use crate::app::Ctx;
use crate::mesh::{MeshData, Vertex};
use crate::renderer::{AlphaMode, Lod, Material, MaterialId, MeshId, TextureId};
use crate::scene::{MeshRenderer, Name, Parent, Transform};
use crate::animation::{AnimationClip, Animator, Channel, ChannelValues, DeformSource, Deformer, Interp};
use glam::{Mat4, Quat, Vec3, Vec4};
use std::sync::Arc;
use hecs::Entity;
use js_sys::{Array, Uint8Array};
use std::collections::HashMap;
use wasm_bindgen::JsCast;
use web_sys::{Blob, BlobPropertyBag, ImageBitmap, Response};

/// glTF meshes with at least this many triangles get automatic LODs (vertex clustering).
pub const AUTO_LOD_MIN_TRIANGLES: usize = 256;

/// Per-load options for glTF models ([`Ctx::load_gltf_with`]); `ctx.gltf_defaults` applies to
/// plain `load_gltf`.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct GltfOpts {
    /// Attach the automatic LOD chain (vertex-clustered levels) to big primitives. Off = always
    /// full detail (small detailed models: characters, figures, props seen from far cameras).
    pub auto_lod: bool,
    /// Multiplies this model's LOD switch distances, impostor included (on top of
    /// `render.lod_distance_scale`; bigger = detail kept farther).
    pub lod_distance: f32,
    /// Last LOD level = an impostor: one camera-facing card (2 triangles) baked from
    /// `impostor_views` azimuths at the first instantiation (cached per model + bake settings).
    /// Static models only (skinned / morphed models keep their meshes).
    pub impostor: bool,
    /// Switch distance to the card in multiples of the model's bounding radius (default 60: a
    /// model ~20 px in radius at 60° FOV / 1350 px), times `lod_distance` and the render scale.
    /// Auto-LOD levels that would start beyond it are dropped.
    pub impostor_distance: f32,
    /// Baked azimuths (default 8; 4 for symmetric props, 12-16 for distinctive ones).
    pub impostor_views: u32,
    /// Texels across one view (default 128).
    pub impostor_res: u32,
}

impl Default for GltfOpts {
    fn default() -> Self {
        Self { auto_lod: true, lod_distance: 1.0, impostor: false, impostor_distance: 60.0, impostor_views: 8, impostor_res: 128 }
    }
}

/// Marks the impostor card entity of a glTF instance (child of the model root).
#[derive(Clone, Copy, Debug)]
pub struct ImpostorCard;

/// GPU-resident model, instantiable any number of times.
#[derive(Clone, Debug)]
pub struct ModelTemplate {
    pub url: String,
    pub nodes: Vec<NodeTemplate>,
    pub roots: Vec<usize>,
    pub triangles: usize,
    /// (joint node indices, inverse bind matrices)
    pub skins: Vec<(Vec<usize>, Arc<Vec<Mat4>>)>,
    pub clips: Arc<Vec<AnimationClip>>,
}

#[derive(Clone, Debug)]
pub struct NodeTemplate {
    pub name: String,
    pub transform: Transform,
    /// (mesh, material, automatic LOD chain for meshes >= AUTO_LOD_MIN_TRIANGLES)
    pub prims: Vec<(MeshId, MaterialId, Option<Lod>)>,
    /// Per prim: CPU source when skinned or morphed (each instance gets its own GPU mesh).
    pub deform: Vec<Option<Arc<DeformSource>>>,
    pub skin: Option<usize>,
    /// Default morph weights of the node's mesh.
    pub morph_weights: Vec<f32>,
    pub children: Vec<usize>,
}

#[derive(Default)]
pub struct ModelCache {
    pub ready: HashMap<String, ModelTemplate>,
    /// url -> roots waiting for the load to finish
    pub pending: HashMap<String, Vec<Entity>>,
    pub failed: HashMap<String, String>,
    /// (url, views, res) -> baked impostor (or the reason it could not be baked).
    pub impostors: HashMap<(String, u32, u32), Result<crate::impostor::Impostor, String>>,
}

/// CPU-side result of parsing (before GPU upload).
struct Parsed {
    meshes: Vec<Vec<(MeshData, Option<usize>, Option<DeformSource>)>>,
    mesh_weights: Vec<Vec<f32>>,
    materials: Vec<(Material, Option<usize>)>,
    images: Vec<Option<ImageBitmap>>,
    /// (name, transform, mesh, children, skin)
    nodes: Vec<(String, Transform, Option<usize>, Vec<usize>, Option<usize>)>,
    roots: Vec<usize>,
    skins: Vec<(Vec<usize>, Vec<Mat4>)>,
    clips: Vec<AnimationClip>,
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

pub(crate) async fn decode_image(bytes: &[u8], mime: &str) -> Result<ImageBitmap, String> {
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
            // COLOR_0 is linear: stored as sRGB bytes like every other vertex colour
            let colors: Option<Vec<[u8; 4]>> = reader.read_colors(0).map(|c| {
                c.into_rgba_f32()
                    .map(|c| {
                        let s = |x: f32| (if x <= 0.0031308 { x * 12.92 } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }).clamp(0.0, 1.0);
                        [(s(c[0]) * 255.0).round() as u8, (s(c[1]) * 255.0).round() as u8, (s(c[2]) * 255.0).round() as u8, (c[3].clamp(0.0, 1.0) * 255.0).round() as u8]
                    })
                    .collect()
            });
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
                        color: colors.as_ref().and_then(|c| c.get(i)).copied().unwrap_or(Vertex::WHITE),
                    })
                    .collect(),
                indices,
            };
            if normals.is_none() {
                compute_normals(&mut md);
            }
            // skinning attributes + morph targets -> CPU deformation source
            let joints: Option<Vec<[u16; 4]>> = reader.read_joints(0).map(|j| j.into_u16().collect());
            let weights: Option<Vec<[f32; 4]>> = reader.read_weights(0).map(|w| w.into_f32().collect());
            let targets: Vec<(Vec<[f32; 3]>, Vec<[f32; 3]>)> = reader
                .read_morph_targets()
                .map(|(pos, nor, _)| (pos.map(|i| i.collect()).unwrap_or_default(), nor.map(|i| i.collect()).unwrap_or_default()))
                .collect();
            let deform = (joints.is_some() && weights.is_some() || !targets.is_empty()).then(|| DeformSource {
                base: md.vertices.clone(),
                joints: joints.unwrap_or_default(),
                weights: weights.unwrap_or_default(),
                targets,
            });
            prims.push((md, p.material().index(), deform));
        }
        meshes.push(prims);
    }
    let mesh_weights: Vec<Vec<f32>> = doc.meshes().map(|m| m.weights().map(|w| w.to_vec()).unwrap_or_default()).collect();

    let skins: Vec<(Vec<usize>, Vec<Mat4>)> = doc
        .skins()
        .map(|sk| {
            let joints: Vec<usize> = sk.joints().map(|j| j.index()).collect();
            let ibm: Vec<Mat4> = sk
                .reader(|b| buffers.get(b.index()).map(|v| v.as_slice()))
                .read_inverse_bind_matrices()
                .map(|it| it.map(|m| Mat4::from_cols_array_2d(&m)).collect())
                .unwrap_or_else(|| vec![Mat4::IDENTITY; joints.len()]);
            (joints, ibm)
        })
        .collect();

    let mut clips = vec![];
    for (ai, anim) in doc.animations().enumerate() {
        let mut channels = vec![];
        let mut duration = 0.0f32;
        for ch in anim.channels() {
            let r = ch.reader(|b| buffers.get(b.index()).map(|v| v.as_slice()));
            let (Some(inputs), Some(outputs)) = (r.read_inputs(), r.read_outputs()) else { continue };
            let times: Vec<f32> = inputs.collect();
            duration = duration.max(times.last().copied().unwrap_or(0.0));
            use gltf::animation::util::ReadOutputs;
            let values = match outputs {
                ReadOutputs::Translations(t) => ChannelValues::Translation(t.map(Vec3::from).collect()),
                ReadOutputs::Rotations(q) => ChannelValues::Rotation(q.into_f32().map(Quat::from_array).collect()),
                ReadOutputs::Scales(s) => ChannelValues::Scale(s.map(Vec3::from).collect()),
                ReadOutputs::MorphTargetWeights(w) => {
                    let all: Vec<f32> = w.into_f32().collect();
                    let keys = times.len().max(1) * if ch.sampler().interpolation() == gltf::animation::Interpolation::CubicSpline { 3 } else { 1 };
                    let n = all.len() / keys;
                    ChannelValues::Weights(all, n)
                }
            };
            channels.push(Channel {
                node: ch.target().node().index(),
                interp: match ch.sampler().interpolation() {
                    gltf::animation::Interpolation::Step => Interp::Step,
                    gltf::animation::Interpolation::Linear => Interp::Linear,
                    gltf::animation::Interpolation::CubicSpline => Interp::CubicSpline,
                },
                times,
                values,
            });
        }
        clips.push(AnimationClip { name: anim.name().map(|n| n.to_string()).unwrap_or_else(|| format!("clip{ai}")), duration, channels });
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
                n.skin().map(|s| s.index()),
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
        mesh_weights,
        materials,
        images,
        nodes,
        roots,
        skins,
        clips,
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
    let mut mesh_deform: Vec<Vec<Option<Arc<DeformSource>>>> = vec![];
    let mesh_prims: Vec<Vec<(MeshId, MaterialId, Option<Lod>)>> = p
        .meshes
        .into_iter()
        .enumerate()
        .map(|(mi, prims)| {
            let mut deforms = vec![];
            let out = prims
                .into_iter()
                .enumerate()
                .map(|(pi, (md, mat, deform))| {
                    triangles += md.indices.len() / 3;
                    let name = format!("{url}#mesh{mi}.{pi}");
                    let deformable = deform.is_some();
                    deforms.push(deform.map(Arc::new));
                    let (id, lod) = if !deformable && md.indices.len() / 3 >= AUTO_LOD_MIN_TRIANGLES {
                        let (id, lod) = ctx.renderer.add_mesh_with_lods(&name, &md, &[0.5, 0.25]);
                        (id, (lod.meshes.len() > 1).then_some(lod))
                    } else {
                        (ctx.renderer.add_mesh(&name, &md), None)
                    };
                    (
                        id,
                        mat.and_then(|m| mat_ids.get(m).copied())
                            .unwrap_or(default_mat),
                        lod,
                    )
                })
                .collect();
            mesh_deform.push(deforms);
            out
        })
        .collect();
    let nodes = p
        .nodes
        .into_iter()
        .map(|(name, transform, mesh, children, skin)| NodeTemplate {
            name,
            transform,
            prims: mesh.map(|m| mesh_prims[m].clone()).unwrap_or_default(),
            deform: mesh.map(|m| mesh_deform[m].clone()).unwrap_or_default(),
            skin,
            morph_weights: mesh.and_then(|m| p.mesh_weights.get(m).cloned()).unwrap_or_default(),
            children,
        })
        .collect();
    if !p.clips.is_empty() || !p.skins.is_empty() {
        crate::log_info!("{url}: {} skins, {} clips {:?}", p.skins.len(), p.clips.len(), p.clips.iter().map(|c| c.name.as_str()).collect::<Vec<_>>());
    }
    ModelTemplate {
        url: url.to_string(),
        nodes,
        roots: p.roots,
        triangles,
        skins: p.skins.into_iter().map(|(j, m)| (j, Arc::new(m))).collect(),
        clips: Arc::new(p.clips),
    }
}

/// Spawns the template's node tree as children of `root`. Skinned / morphed primitives get their
/// own GPU mesh and a [`Deformer`]; models with clips get an [`Animator`] on `root`.
pub fn instantiate(ctx: &mut Ctx, root: Entity, model: &ModelTemplate) {
    if !ctx.world.contains(root) {
        return; // despawned while loading
    }
    let layer = ctx.world.get::<&crate::scene::RenderLayer>(root).map(|l| l.0).unwrap_or(0);
    let opts = ctx.world.get::<&GltfOpts>(root).map(|o| *o).unwrap_or(ctx.gltf_defaults);
    let imp = if opts.impostor { impostor_for(ctx, model, &opts) } else { None };
    // switch distance to the card in model-root units (prims divide by their scale below the root)
    let imp_d = imp.map(|i| i.radius * opts.impostor_distance.max(1.0) * opts.lod_distance.max(0.01));
    let n_nodes = model.nodes.len();
    let mut node_entity: Vec<Option<Entity>> = vec![None; n_nodes];
    let mut node_prims: Vec<Vec<Entity>> = vec![vec![]; n_nodes];
    // spawn the hierarchy
    let mut stack: Vec<(usize, Entity, f32)> = model.roots.iter().rev().map(|&r| (r, root, 1.0)).collect();
    while let Some((idx, parent, parent_scale)) = stack.pop() {
        let n = &model.nodes[idx];
        let chain_scale = parent_scale * n.transform.scale.abs().max_element().max(1e-6);
        let e = ctx.world.spawn((n.transform, Name(n.name.clone()), Parent(parent)));
        node_entity[idx] = Some(e);
        for (k, (mesh, mat, lod)) in n.prims.iter().enumerate() {
            let target = if n.prims.len() == 1 { e } else { ctx.world.spawn((Transform::default(), Parent(e))) };
            let deform = n.deform.get(k).cloned().flatten();
            let mesh = match &deform {
                // own copy: every instance deforms differently
                Some(src) => {
                    let md = MeshData { vertices: src.base.clone(), indices: ctx.renderer.mesh_indices(*mesh).unwrap_or_default() };
                    ctx.renderer.add_mesh(&format!("{}#deform", ctx.renderer.mesh(*mesh).map(|m| m.name.clone()).unwrap_or_default()), &md)
                }
                None => *mesh,
            };
            let mut mr = MeshRenderer::new(mesh, *mat);
            mr.layer = layer;
            let _ = ctx.world.insert_one(target, mr);
            let mut lod_c = match (lod, &deform, opts.auto_lod) {
                (Some(l), None, true) => {
                    let mut l = l.clone();
                    for d in l.distances.iter_mut() {
                        *d *= opts.lod_distance.max(0.01);
                    }
                    Some(l)
                }
                _ => None,
            };
            if let (Some(d_root), None) = (imp_d, &deform) {
                // the meshes stop where the card starts: drop levels that would begin beyond it
                let d = d_root / chain_scale;
                let mut l = lod_c.take().unwrap_or(Lod { meshes: vec![mesh], distances: vec![], cull_beyond: false });
                let keep = (1 + l.distances.iter().take_while(|&&x| x < d).count()).min(l.meshes.len());
                l.meshes.truncate(keep);
                l.distances.truncate(keep - 1);
                l.distances.push(d);
                l.cull_beyond = true;
                lod_c = Some(l);
            }
            if let Some(l) = lod_c {
                let _ = ctx.world.insert_one(target, l);
            }
            if let Some(src) = deform {
                let weights = if n.morph_weights.len() == src.targets.len() { n.morph_weights.clone() } else { vec![0.0; src.targets.len()] };
                // skin joints are resolved after the whole tree exists
                let _ = ctx.world.insert_one(target, Deformer::new(src, None, weights, mesh));
                node_prims[idx].push(target);
            }
        }
        for &c in n.children.iter().rev() {
            stack.push((c, e, chain_scale));
        }
    }
    if let (Some(i), Some(d)) = (imp, imp_d) {
        let mut mr = MeshRenderer::new(i.mesh, i.material);
        mr.layer = layer;
        let card = ctx.world.spawn((
            Transform::from_translation(i.pivot),
            Name("impostor".into()),
            Parent(root),
            mr,
            Lod { meshes: vec![i.empty, i.mesh], distances: vec![d], cull_beyond: false },
            ImpostorCard,
            crate::zfight::NoZCheck,
        ));
        let _ = card;
    }
    // resolve skins
    for (idx, n) in model.nodes.iter().enumerate() {
        let Some(si) = n.skin else { continue };
        let Some((joints, ibm)) = model.skins.get(si) else { continue };
        let js: Vec<Entity> = joints.iter().filter_map(|&j| node_entity.get(j).copied().flatten()).collect();
        for &pe in &node_prims[idx] {
            if let Ok(mut d) = ctx.world.get::<&mut Deformer>(pe) {
                if !d.src.joints.is_empty() {
                    d.skin = Some((js.clone(), ibm.clone()));
                }
            }
        }
    }
    if !model.clips.is_empty() {
        let nodes: Vec<Entity> = node_entity.iter().map(|e| e.unwrap_or(root)).collect();
        let rest: Vec<Transform> = model.nodes.iter().map(|n| n.transform).collect();
        let _ = ctx.world.insert_one(root, Animator::new(model.clips.clone(), nodes, node_prims, rest));
    }
}

/// The model's impostor for these bake settings (baked on first use, cached per model).
fn impostor_for(ctx: &mut Ctx, model: &ModelTemplate, opts: &GltfOpts) -> Option<crate::impostor::Impostor> {
    let key = (model.url.clone(), opts.impostor_views.clamp(1, 16), opts.impostor_res.clamp(32, 512));
    if let Some(r) = ctx.models.impostors.get(&key) {
        return r.as_ref().ok().copied();
    }
    let r = (|| {
        if model.nodes.iter().any(|n| n.deform.iter().any(|d| d.is_some())) || !model.skins.is_empty() {
            return Err("skinned / morphed model".to_string());
        }
        // every primitive with its node -> model-root matrix
        let mut parts = vec![];
        let mut stack: Vec<(usize, Mat4)> = model.roots.iter().map(|&r| (r, Mat4::IDENTITY)).collect();
        while let Some((idx, parent)) = stack.pop() {
            let n = &model.nodes[idx];
            let m = parent * n.transform.matrix();
            for (mesh, mat, _) in &n.prims {
                parts.push((*mesh, *mat, m));
            }
            stack.extend(n.children.iter().map(|&c| (c, m)));
        }
        ctx.renderer.bake_impostor(&model.url, &parts, key.1, key.2)
    })();
    if let Err(e) = &r {
        crate::log_warn!("{}: no impostor ({e})", model.url);
    }
    ctx.models.impostors.insert(key, r.clone());
    r.ok()
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

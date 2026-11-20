//! GPU-driven scene of the main view (child module of `renderer`).
//!
//! Every drawn or shadow-casting entity keeps, next to its instance record (`InstancePool`), a
//! persistent object record in a GPU buffer: bounding sphere, LOD table and the draw group of every
//! LOD level, for the main view and for shadows. Per frame:
//!
//!   CPU: `GpuScene::sync` compares each entity with what its slot holds (transform, mesh, material
//!        + version, opacity, LOD component) and rebuilds only what changed; slots of entities that
//!        are gone are cleared. Blended entities (a handful) are sorted on the CPU as before.
//!   GPU: `cull.wgsl` (`reset` + `cull`) picks the LOD level, tests the camera frustum and the Hi-Z
//!        pyramid and appends every survivor's slot to its draw group's slice of the id buffer — for
//!        the main view and for every shadow cascade rendered this frame — then each group is one
//!        `drawIndexedIndirect`.
//!
//! A draw group is (pipeline, texture, mesh) for the main view and (alpha-tested, texture, mesh) for
//! shadows. Its slice of the id buffer has room for all its members, so the indirect args only
//! change when group membership changes (spawn, despawn, material or mesh swap), not when the
//! camera moves. Draw order: groups sorted per pipeline by the nearest visible instance, read back
//! from the GPU (1-3 frames old).

use super::*;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

/// LOD levels an object can have on the GPU path (longer chains are cut).
pub(super) const LEVELS: usize = 8;
pub(super) const NONE: u32 = u32::MAX;
/// Shadow passes the culling pass can fill at once (current + fading set of cascades).
pub(super) const PASSES: usize = 2 * MAX_CASCADES;
/// Shadow id lists: per pass q the static (or all) casters in list q, the dynamic ones in q + PASSES
/// (static shadow cache: see `RenderSettings::shadow_static_cache`).
pub(super) const SHADOW_LISTS: usize = 2 * PASSES;
/// Frames without a change after which a dynamic caster counts as static again (10 s: a hero or NPC
/// that pauses briefly must not move into the static copy and out again, two full re-renders).
const DYN_FRAMES: u64 = 600;
const N_COUNTERS: usize = 12;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(super) struct GpuObject {
    sphere: [f32; 4],
    origin: [f32; 4],
    lod: [f32; LEVELS],
    groups: [u32; LEVELS],
    sgroups: [u32; LEVELS],
    info: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct SceneParams {
    pub planes: [[f32; 4]; 6],
    pub prev_view_proj: [[f32; 4]; 4],
    pub cam: [f32; 4],
    pub hiz_size: [f32; 2],
    pub count: u32,
    pub flags: u32,
    pub hiz_mips: u32,
    pub layers: u32,
    pub n_groups: u32,
    pub n_sgroups: u32,
    pub pass_mask: u32,
    pub n_args: u32,
    /// Bit q: pass q is split (static casters -> list q, dynamic -> list q + PASSES).
    pub pass_split: u32,
    /// Bit q: a split pass also needs its static list this frame (static cache re-rendered).
    pub static_mask: u32,
    pub pass_lv: [[[f32; 4]; 4]; PASSES],
    pub pass_box: [[f32; 4]; PASSES],
}

/// The LOD component as last seen (change detection).
#[derive(Clone, PartialEq)]
struct LodKey {
    meshes: Vec<u32>,
    distances: Vec<f32>,
    cull_beyond: bool,
}

impl LodKey {
    fn matches(a: &Option<LodKey>, b: Option<&Lod>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(k), Some(l)) => {
                k.cull_beyond == l.cull_beyond
                    && k.distances == l.distances
                    && k.meshes.len() == l.meshes.len()
                    && k.meshes.iter().zip(&l.meshes).all(|(a, b)| *a == b.0)
            }
            _ => false,
        }
    }
}

/// What a blended entity needs for its per-frame CPU sort.
#[derive(Clone)]
struct BlendObj {
    meshes: [u32; LEVELS],
    double_sided: bool,
    tex: u32,
    water: bool,
}

#[derive(Clone, Default)]
struct SlotInfo {
    active: bool,
    seen: u64,
    // inputs, compared every frame
    model: Mat4,
    mesh: u32,
    material: u32,
    mat_ver: u32,
    opacity: u32,
    cast: bool,
    layer: u8,
    epoch: u32,
    lod: Option<LodKey>,
    mesh_ver: u64,
    /// Second slot of a faded caster (its unfaded shadow record), NONE otherwise.
    shadow_slot: u32,
    /// Frame of the last move / deformation (dynamic caster while recent).
    changed_frame: u64,
    // derived
    obj: GpuObject,
    main: bool,
    tris: u64,
    blend: Option<Box<BlendObj>>,
    blend_pos: u32,
}

/// Draw groups of one kind with their member counts and id-buffer slices.
#[derive(Default)]
struct Groups {
    map: crate::scene::FxMap<(u8, u32, u32), u32>,
    keys: Vec<(u8, u32, u32)>,
    members: Vec<u32>,
    /// member counts the current layout was made for
    laid: Vec<u32>,
    first: Vec<u32>,
    total: u32,
    touched: bool,
    /// Conservative bounds of the members (grown on every change, rebuilt once a second):
    /// AABB of their spheres, AABB of their origins, world-space LOD distance band (before bias).
    bmin: Vec<Vec3>,
    bmax: Vec<Vec3>,
    omin: Vec<Vec3>,
    omax: Vec<Vec3>,
    band: Vec<(f32, f32)>,
}

const EMPTY_BAND: (f32, f32) = (f32::MAX, 0.0);

impl Groups {
    fn acquire(&mut self, key: (u8, u32, u32)) -> u32 {
        let id = *self.map.entry(key).or_insert_with(|| {
            self.keys.push(key);
            self.members.push(0);
            self.laid.push(0);
            self.first.push(0);
            self.bmin.push(Vec3::splat(f32::MAX));
            self.bmax.push(Vec3::splat(f32::MIN));
            self.omin.push(Vec3::splat(f32::MAX));
            self.omax.push(Vec3::splat(f32::MIN));
            self.band.push(EMPTY_BAND);
            self.keys.len() as u32 - 1
        });
        self.members[id as usize] += 1;
        self.touched = true;
        id
    }

    fn release(&mut self, id: u32) {
        if id != NONE {
            let m = &mut self.members[id as usize];
            *m = m.saturating_sub(1);
            self.touched = true;
        }
    }

    /// True if member counts differ from the current layout.
    fn changed(&mut self) -> bool {
        std::mem::take(&mut self.touched) && self.members != self.laid
    }

    fn expand(&mut self, g: u32, sphere: &[f32; 4], origin: Vec3, band: (f32, f32)) {
        let g = g as usize;
        let (c, r) = (Vec3::new(sphere[0], sphere[1], sphere[2]), sphere[3]);
        self.bmin[g] = self.bmin[g].min(c - r);
        self.bmax[g] = self.bmax[g].max(c + r);
        self.omin[g] = self.omin[g].min(origin);
        self.omax[g] = self.omax[g].max(origin);
        self.band[g] = (self.band[g].0.min(band.0), self.band[g].1.max(band.1));
    }

    fn reset_bounds(&mut self) {
        self.bmin.fill(Vec3::splat(f32::MAX));
        self.bmax.fill(Vec3::splat(f32::MIN));
        self.omin.fill(Vec3::splat(f32::MAX));
        self.omax.fill(Vec3::splat(f32::MIN));
        self.band.fill(EMPTY_BAND);
    }

    /// Could a member of group `g` be at its LOD level for this camera? (distance band vs the
    /// distance range of the members' origins)
    fn lod_possible(&self, g: usize, cam: Vec3, lod_bias: f32) -> bool {
        let (lo, hi) = self.band[g];
        let (a, b) = (self.omin[g], self.omax[g]);
        let near = (cam.clamp(a, b) - cam).length();
        let far = (cam - a).abs().max((cam - b).abs()).length();
        near <= hi * lod_bias * 1.001 + 1e-3 && far >= lo * lod_bias * 0.999 - 1e-3
    }

    /// Sphere around the members' AABB.
    fn sphere(&self, g: usize) -> (Vec3, f32) {
        let (a, b) = (self.bmin[g], self.bmax[g]);
        ((a + b) * 0.5, (b - a).length() * 0.5)
    }

    fn layout(&mut self) {
        let mut at = 0u32;
        for (i, m) in self.members.iter().enumerate() {
            self.first[i] = at;
            at += m;
        }
        self.total = at;
        self.laid.clone_from(&self.members);
    }
}

/// CPU side of the GPU-driven scene.
#[derive(Default)]
pub(super) struct GpuScene {
    infos: Vec<SlotInfo>,
    main: Groups,
    shadow: Groups,
    keys: crate::scene::FxMap<(u32, u32, u32), u32>,
    /// Material bind-group keys (base, normal, height); group keys hold an index into this list.
    pub key_list: Vec<(u32, u32, u32)>,
    blend_slots: Vec<u32>,
    /// Object slots to upload.
    dirty: Vec<u32>,
    /// The object buffer was reallocated: upload every slot.
    pub upload_all: bool,
    pub args: Vec<DrawArgs>,
    pub args_dirty: bool,
    /// Bounding spheres of shadow casters that changed this frame (old and new), for the shadow cache.
    pub shadow_changes: Vec<[f32; 4]>,
    /// The part of `shadow_changes` that touches the static casters (static shadow cache).
    pub shadow_static_changes: Vec<[f32; 4]>,
    /// Everything changed (first frame, shadows toggled): no cached shadow layer is valid.
    pub shadow_reset: bool,
    /// Casters currently flagged dynamic (info.y bit 2), checked for expiry every frame.
    dyn_slots: Vec<u32>,
    /// Slots whose change touched the static shadow copies this frame (`gpu_scene` shows them).
    static_change_slots: Vec<(u32, &'static str)>,
    /// What the current note_shadow calls belong to (for `static_changes`).
    note_why: &'static str,
    /// Slots changed this frame and why (for `gpu_scene`: what keeps uploads / shadows busy).
    changed: Vec<(u32, &'static str)>,
    /// Triangles (LOD 0) and count of the main-view objects.
    pub main_tris: u64,
    /// `main_tris` split by render layer (a camera's `layers` mask selects which count).
    main_tris_layer: [u64; 32],
    pub main_count: u32,
    epoch: u32,
    shadows_on: Option<bool>,
    bounds: (Vec3, Vec3),
    bounds_frame: u64,
    fallback: Material,
    warned_lod: bool,
}

/// World-space distance band (before `lod_bias`) in which an object shows LOD level `l`.
fn lod_band(o: &GpuObject, l: usize) -> (f32, f32) {
    let n = (o.info[0] & 0xff) as usize;
    let nd = ((o.info[0] >> 8) & 0xff) as usize;
    let cull_beyond = (o.info[0] >> 16) & 1 == 1;
    let scale = 1.0 / o.origin[3].max(1e-12);
    // level l needs d > every earlier switch distance, and d <= its own (or beyond all of them)
    let lo = o.lod[..l.min(nd)].iter().fold(0.0f32, |a, &b| a.max(b)) * scale;
    let hi = if l + 1 == n && !cull_beyond {
        f32::MAX
    } else if l < nd {
        o.lod[l] * scale
    } else {
        f32::MAX
    };
    (lo, hi)
}

fn scale_of(m: &Mat4) -> f32 {
    m.x_axis.truncate().length().max(m.y_axis.truncate().length()).max(m.z_axis.truncate().length())
}

fn mesh_version(meshes: &[GpuMesh], mr: &MeshRenderer, lod: Option<&Lod>) -> u64 {
    let v = |m: MeshId| meshes.get(m.0 as usize).map_or(0, |g| g.version) as u64;
    match lod {
        None => v(mr.mesh),
        Some(l) => l.meshes.iter().fold(0u64, |h, m| h.rotate_left(13) ^ v(*m)),
    }
}

impl GpuScene {
    /// LOD-0 triangles of the main-view objects on the layers in `mask` (a camera's `layers`).
    pub fn main_tris_for(&self, mask: u32) -> u64 {
        if mask == u32::MAX {
            return self.main_tris;
        }
        (0..32).filter(|l| mask & (1u32 << l) != 0).map(|l| self.main_tris_layer[l]).sum()
    }

    pub fn group_count(&self) -> (u32, u32) {
        (self.main.keys.len() as u32, self.shadow.keys.len() as u32)
    }

    /// Ids per shadow pass, and the total id-buffer size the GPU writes (main + all shadow passes).
    pub fn id_layout(&self) -> (u32, u32) {
        (self.shadow.total, self.main.total + SHADOW_LISTS as u32 * self.shadow.total)
    }

    pub fn slots(&self) -> u32 {
        self.infos.len() as u32
    }

    pub fn scene_bounds(&self) -> (Vec3, Vec3) {
        self.bounds
    }

    fn key(&mut self, k: (u32, u32, u32)) -> u32 {
        *self.keys.entry(k).or_insert_with(|| {
            self.key_list.push(k);
            self.key_list.len() as u32 - 1
        })
    }

    fn ensure(&mut self, slot: usize) {
        if self.infos.len() <= slot {
            self.infos.resize_with(slot + 1, || SlotInfo { shadow_slot: NONE, ..Default::default() });
        }
    }

    /// Grows the bounds of every group slot `i` belongs to.
    fn expand_groups(&mut self, i: usize) {
        let o = self.infos[i].obj;
        let origin = Vec3::new(o.origin[0], o.origin[1], o.origin[2]);
        for l in 0..LEVELS {
            if o.groups[l] == NONE && o.sgroups[l] == NONE {
                continue;
            }
            let band = lod_band(&o, l);
            if o.groups[l] != NONE {
                self.main.expand(o.groups[l], &o.sphere, origin, band);
            }
            if o.sgroups[l] != NONE {
                self.shadow.expand(o.sgroups[l], &o.sphere, origin, band);
            }
        }
    }

    fn note_shadow(&mut self, i: usize) {
        let o = &self.infos[i].obj;
        if o.info[1] & 2 != 0 {
            self.shadow_changes.push(o.sphere);
            if o.info[1] & 4 == 0 {
                self.shadow_static_changes.push(o.sphere);
                self.static_change_slots.push((i as u32, self.note_why));
            }
        }
    }

    /// Bounding spheres of the casters currently drawn into the dynamic shadow lists.
    pub fn dynamic_spheres(&self) -> impl Iterator<Item = [f32; 4]> + '_ {
        self.dyn_slots.iter().map(|&i| &self.infos[i as usize]).filter(|f| f.active && f.obj.info[1] & 6 == 6).map(|f| f.obj.sphere)
    }

    /// A caster moved or deformed. Moved again within half a second, it renders into the dynamic
    /// shadow list from now on (until it stays still for `DYN_FRAMES`); a single move (placement,
    /// teleport) just updates the static copy. Call after noting its old shape, so the static cache
    /// drops it.
    fn make_dynamic(&mut self, i: usize, frame: u64) {
        let inf = &mut self.infos[i];
        if inf.obj.info[1] & 2 == 0 {
            return;
        }
        let recent = inf.changed_frame != 0 && frame.saturating_sub(inf.changed_frame) <= 30;
        inf.changed_frame = frame;
        if recent && inf.obj.info[1] & 4 == 0 {
            inf.obj.info[1] |= 4;
            self.dyn_slots.push(i as u32);
            self.dirty.push(i as u32);
        }
    }

    /// Removes slot `i`'s group memberships and draw flags (it stays allocated).
    fn clear(&mut self, i: usize) {
        self.note_shadow(i);
        let inf = &mut self.infos[i];
        for g in inf.obj.groups {
            self.main.release(g);
        }
        for g in inf.obj.sgroups {
            self.shadow.release(g);
        }
        inf.obj.groups = [NONE; LEVELS];
        inf.obj.sgroups = [NONE; LEVELS];
        inf.obj.info[1] = 0;
        self.main_tris -= inf.tris;
        self.main_tris_layer[(inf.layer & 31) as usize] -= inf.tris;
        self.main_count -= inf.main as u32;
        inf.tris = 0;
        inf.main = false;
        if inf.blend.take().is_some() {
            let pos = inf.blend_pos as usize;
            self.blend_slots.swap_remove(pos);
            if let Some(&moved) = self.blend_slots.get(pos) {
                self.infos[moved as usize].blend_pos = pos as u32;
            }
        }
        self.dirty.push(i as u32);
    }

    fn deactivate(&mut self, i: usize) {
        if !self.infos[i].active {
            return;
        }
        self.clear(i);
        self.infos[i].active = false;
        let ss = std::mem::replace(&mut self.infos[i].shadow_slot, NONE);
        if ss != NONE {
            self.deactivate(ss as usize);
        }
    }

    /// Brings the scene up to date with the world. `pool.frame` must already be this frame's.
    #[allow(clippy::too_many_arguments)]
    pub fn sync(&mut self, world: &World, pool: &mut InstancePool, materials: &[Material], mat_ver: &[u32], meshes: &[GpuMesh], shadows_on: bool) {
        let frame = pool.frame;
        self.shadow_changes.clear();
        self.shadow_static_changes.clear();
        self.static_change_slots.clear();
        self.changed.clear();
        if self.shadows_on != Some(shadows_on) {
            self.shadows_on = Some(shadows_on);
            self.epoch += 1;
            self.shadow_reset = true;
        }
        for (ent, g, mr, lod) in world.query::<(hecs::Entity, &GlobalTransform, &MeshRenderer, Option<&Lod>)>().iter() {
            let ek = ent.to_bits().get();
            let mv = mesh_version(meshes, mr, lod);
            let mat_v = mat_ver.get(mr.material.0 as usize).copied().unwrap_or(0);
            if let Some(&s) = pool.slot_of.get(&(ek, false)) {
                let i = s as usize;
                if let Some(inf) = self.infos.get(i).filter(|f| f.active) {
                    let same_inputs = inf.epoch == self.epoch
                        && inf.mesh == mr.mesh.0
                        && inf.material == mr.material.0
                        && inf.mat_ver == mat_v
                        && inf.opacity == mr.opacity.to_bits()
                        && inf.cast == mr.cast_shadows
                        && inf.layer == mr.layer
                        && LodKey::matches(&inf.lod, lod);
                    if same_inputs && inf.model == g.0 {
                        let ss = inf.shadow_slot;
                        let mesh_changed = inf.mesh_ver != mv;
                        pool.seen[i] = frame;
                        self.infos[i].seen = frame;
                        if ss != NONE {
                            pool.seen[ss as usize] = frame;
                            self.infos[ss as usize].seen = frame;
                        }
                        if mesh_changed {
                            self.changed.push((s, "mesh deformed"));
                            self.note_why = "mesh deformed";
                            // deformed mesh (skinning, morphs): same draw, new shadow
                            self.infos[i].mesh_ver = mv;
                            for k in [i as u32, ss] {
                                if k != NONE {
                                    self.note_shadow(k as usize);
                                    self.make_dynamic(k as usize, frame);
                                    self.note_shadow(k as usize);
                                }
                            }
                        }
                        continue;
                    }
                    if same_inputs {
                        self.changed.push((s, "moved"));
                        self.note_why = "moved";
                        // moved only: new record + sphere, same groups
                        self.moved(ek, i, &g.0, mr, mv, pool, materials, meshes, shadows_on, frame);
                        continue;
                    }
                }
            }
            self.note_why = "rebuilt";
            self.rebuild(ek, &g.0, mr, lod, mv, mat_v, pool, materials, meshes, shadows_on, frame);
            if let Some(&s) = pool.slot_of.get(&(ek, false)) {
                self.changed.push((s, "rebuilt"));
            }
        }
        // dynamic casters that stayed still long enough go back into the static shadow cache, in
        // batches (every 30 frames: one re-render of the static copies instead of one per caster)
        let mut k = if frame % 30 == 0 { 0 } else { self.dyn_slots.len() };
        while k < self.dyn_slots.len() {
            let i = self.dyn_slots[k] as usize;
            let inf = &self.infos[i];
            if !inf.active || inf.obj.info[1] & 4 == 0 {
                self.dyn_slots.swap_remove(k);
            } else if frame.saturating_sub(inf.changed_frame) > DYN_FRAMES {
                self.note_why = "expired";
                self.infos[i].obj.info[1] &= !4;
                self.note_shadow(i);
                self.dirty.push(i as u32);
                self.dyn_slots.swap_remove(k);
            } else {
                k += 1;
            }
        }
        // entities that are gone (despawned, MeshRenderer removed)
        for i in 0..self.infos.len() {
            if self.infos[i].active && self.infos[i].seen != frame {
                self.changed.push((i as u32, "removed"));
                self.note_why = "removed";
                self.deactivate(i);
            }
        }
        if self.main.changed() | self.shadow.changed() {
            self.relayout(meshes);
        }
        // opaque scene bounds (water SSR): grown on change, recomputed once a second
        if frame >= self.bounds_frame + 60 || self.bounds.0.x > self.bounds.1.x {
            self.bounds_frame = frame;
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for inf in self.infos.iter().filter(|f| f.active && f.main) {
                let s = inf.obj.sphere;
                let (c, r) = (Vec3::new(s[0], s[1], s[2]), s[3]);
                lo = lo.min(c - r);
                hi = hi.max(c + r);
            }
            self.bounds = (lo, hi);
            // group bounds only grow between rebuilds (moving members): shrink them again
            self.main.reset_bounds();
            self.shadow.reset_bounds();
            for i in 0..self.infos.len() {
                if self.infos[i].active {
                    self.expand_groups(i);
                }
            }
        }
    }

    fn sphere(model: &Mat4, scale: f32, levels: &[u32], meshes: &[GpuMesh]) -> [f32; 4] {
        let mut out: Option<(Vec3, f32)> = None;
        for &m in levels {
            let Some(gm) = meshes.get(m as usize).filter(|g| g.index_count > 0) else { continue };
            let c = model.transform_point3(gm.bounds_center);
            let r = gm.bounds_radius * scale;
            out = Some(match out {
                None => (c, r),
                Some((c0, r0)) => (c0, r0.max(c.distance(c0) + r)),
            });
        }
        let (c, r) = out.unwrap_or((model.w_axis.truncate(), 0.0));
        [c.x, c.y, c.z, r]
    }

    fn levels(&mut self, mr: &MeshRenderer, lod: Option<&Lod>) -> ([u32; LEVELS], usize, [f32; LEVELS], usize, bool) {
        let mut ms = [NONE; LEVELS];
        let mut ds = [0.0f32; LEVELS];
        match lod {
            None => {
                ms[0] = mr.mesh.0;
                (ms, 1, ds, 0, false)
            }
            Some(l) => {
                if (l.meshes.len() > LEVELS || l.distances.len() > LEVELS) && !self.warned_lod {
                    self.warned_lod = true;
                    crate::log_warn!("Lod with {} meshes / {} distances: the GPU path uses the first {LEVELS}", l.meshes.len(), l.distances.len());
                }
                let n = l.meshes.len().min(LEVELS);
                let nd = l.distances.len().min(LEVELS);
                for k in 0..n {
                    ms[k] = l.meshes[k].0;
                }
                ds[..nd].copy_from_slice(&l.distances[..nd]);
                (ms, n, ds, nd, l.cull_beyond)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn moved(&mut self, ek: u64, i: usize, model: &Mat4, mr: &MeshRenderer, mv: u64, pool: &mut InstancePool, materials: &[Material], meshes: &[GpuMesh], shadows_on: bool, frame: u64) {
        let mat = materials.get(mr.material.0 as usize).unwrap_or(&self.fallback);
        let c = classify(model, mr, mat, shadows_on);
        pool.put((ek, false), &c.raw);
        self.note_shadow(i);
        self.make_dynamic(i, frame);
        let scale = scale_of(model);
        let o = model.w_axis.truncate();
        let ss = self.infos[i].shadow_slot;
        let levels: Vec<u32> = match &self.infos[i].lod {
            Some(l) => l.meshes.iter().copied().take(LEVELS).collect(),
            None => vec![mr.mesh.0],
        };
        let sphere = Self::sphere(model, scale, &levels, meshes);
        let origin = [o.x, o.y, o.z, 1.0 / scale.max(1e-4)];
        {
            let inf = &mut self.infos[i];
            inf.model = *model;
            inf.mesh_ver = mv;
            inf.seen = frame;
            inf.obj.sphere = sphere;
            inf.obj.origin = origin;
        }
        self.note_shadow(i);
        self.expand_groups(i);
        if self.infos[i].main {
            self.bounds.0 = self.bounds.0.min(Vec3::new(sphere[0], sphere[1], sphere[2]) - sphere[3]);
            self.bounds.1 = self.bounds.1.max(Vec3::new(sphere[0], sphere[1], sphere[2]) + sphere[3]);
        }
        self.dirty.push(i as u32);
        if ss != NONE {
            if let Some(r) = &c.shadow_raw {
                pool.put((ek, true), r);
            }
            let s = ss as usize;
            self.note_shadow(s);
            self.make_dynamic(s, frame);
            self.infos[s].seen = frame;
            self.infos[s].obj.sphere = sphere;
            self.infos[s].obj.origin = origin;
            self.note_shadow(s);
            self.expand_groups(s);
            self.dirty.push(ss);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rebuild(&mut self, ek: u64, model: &Mat4, mr: &MeshRenderer, lod: Option<&Lod>, mv: u64, mat_v: u32, pool: &mut InstancePool, materials: &[Material], meshes: &[GpuMesh], shadows_on: bool, frame: u64) {
        let mat = materials.get(mr.material.0 as usize).unwrap_or(&self.fallback);
        let c = classify(model, mr, mat, shadows_on);
        let s = pool.put((ek, false), &c.raw) as usize;
        self.ensure(s);
        let old_shadow = if self.infos[s].active {
            self.clear(s);
            self.infos[s].shadow_slot
        } else {
            NONE
        };
        let scale = scale_of(model);
        let o = model.w_axis.truncate();
        let (ms, n, ds, nd, cull_beyond) = self.levels(mr, lod);
        let sphere = Self::sphere(model, scale, &ms[..n], meshes);
        let tex = self.key(c.tex_key);
        let valid = |m: u32| meshes.get(m as usize).filter(|g| g.index_count > 0);
        let main = !c.blend;
        let mut groups = [NONE; LEVELS];
        let mut tris = 0u64;
        if main {
            let pipe = (if c.mask || c.ui { 2 } else { 0 }) + c.double_sided as u8;
            for l in 0..n {
                if let Some(gm) = valid(ms[l]) {
                    groups[l] = self.main.acquire((pipe, tex, ms[l]));
                    if tris == 0 {
                        tris = (gm.index_count / 3) as u64;
                    }
                }
            }
        }
        let mut sgroups = [NONE; LEVELS];
        if c.casts {
            let stex = if c.mask { tex } else { 0 };
            for l in 0..n {
                if valid(ms[l]).is_some() {
                    sgroups[l] = self.shadow.acquire((c.mask as u8, stex, ms[l]));
                }
            }
        }
        let shadow_here = c.casts && c.shadow_raw.is_none();
        let info_x = n as u32 | (nd as u32) << 8 | (cull_beyond as u32) << 16 | (lod.is_some() as u32) << 17 | ((mr.layer & 31) as u32) << 24;
        let origin = [o.x, o.y, o.z, 1.0 / scale.max(1e-4)];
        let obj = GpuObject {
            sphere,
            origin,
            lod: ds,
            groups,
            sgroups: if shadow_here { sgroups } else { [NONE; LEVELS] },
            info: [info_x, main as u32 | (shadow_here as u32) << 1, 0, 0],
        };
        let blend = (c.blend && !c.hidden).then(|| Box::new(BlendObj { meshes: ms, double_sided: c.double_sided, tex, water: c.water }));
        let mut new_shadow = NONE;
        if let Some(r) = &c.shadow_raw {
            let ss = pool.put((ek, true), r) as usize;
            self.ensure(ss);
            if self.infos[ss].active {
                self.clear(ss);
            }
            let si = &mut self.infos[ss];
            si.active = true;
            si.seen = frame;
            si.shadow_slot = NONE;
            si.obj = GpuObject { sphere, origin, lod: ds, groups: [NONE; LEVELS], sgroups, info: [info_x, 2, 0, 0] };
            self.dirty.push(ss as u32);
            self.note_shadow(ss);
            self.expand_groups(ss);
            new_shadow = ss as u32;
        }
        if old_shadow != NONE && old_shadow != new_shadow {
            self.deactivate(old_shadow as usize);
        }
        let is_blend = blend.is_some();
        let inf = &mut self.infos[s];
        *inf = SlotInfo {
            active: true,
            seen: frame,
            model: *model,
            mesh: mr.mesh.0,
            material: mr.material.0,
            mat_ver: mat_v,
            opacity: mr.opacity.to_bits(),
            cast: mr.cast_shadows,
            layer: mr.layer,
            epoch: self.epoch,
            lod: lod.map(|l| LodKey { meshes: l.meshes.iter().map(|m| m.0).collect(), distances: l.distances.clone(), cull_beyond: l.cull_beyond }),
            mesh_ver: mv,
            shadow_slot: new_shadow,
            changed_frame: 0,
            obj,
            main,
            tris,
            blend,
            blend_pos: self.blend_slots.len() as u32,
        };
        if is_blend {
            self.blend_slots.push(s as u32);
        }
        self.main_tris += tris;
        self.main_tris_layer[(mr.layer & 31) as usize] += tris;
        self.main_count += main as u32;
        if main {
            self.bounds.0 = self.bounds.0.min(Vec3::new(sphere[0], sphere[1], sphere[2]) - sphere[3]);
            self.bounds.1 = self.bounds.1.max(Vec3::new(sphere[0], sphere[1], sphere[2]) + sphere[3]);
        }
        self.note_shadow(s);
        self.expand_groups(s);
        self.dirty.push(s as u32);
    }

    /// New id-buffer slices for every group (membership changed): rebuilds the indirect args.
    fn relayout(&mut self, meshes: &[GpuMesh]) {
        self.main.layout();
        self.shadow.layout();
        let ic = |m: u32| meshes.get(m as usize).map_or(0, |g| g.index_count);
        let mut args = Vec::with_capacity(self.main.keys.len() + SHADOW_LISTS * self.shadow.keys.len());
        for (k, &first) in self.main.keys.iter().zip(&self.main.first) {
            args.push(DrawArgs { index_count: ic(k.2), instance_count: 0, first_index: 0, base_vertex: 0, first_instance: first });
        }
        for p in 0..SHADOW_LISTS as u32 {
            let base = self.main.total + p * self.shadow.total;
            for (k, &first) in self.shadow.keys.iter().zip(&self.shadow.first) {
                args.push(DrawArgs { index_count: ic(k.2), instance_count: 0, first_index: 0, base_vertex: 0, first_instance: base + first });
            }
        }
        self.args = args;
        self.args_dirty = true;
    }

    /// Object slot ranges to upload (merged across small gaps).
    pub fn take_dirty(&mut self) -> Vec<(usize, usize)> {
        if std::mem::take(&mut self.upload_all) {
            self.dirty.clear();
            return if self.infos.is_empty() { vec![] } else { vec![(0, self.infos.len())] };
        }
        let mut d = std::mem::take(&mut self.dirty);
        d.sort_unstable();
        d.dedup();
        let mut ranges: Vec<(usize, usize)> = vec![];
        for &s in &d {
            let s = s as usize;
            match ranges.last_mut() {
                Some(r) if s <= r.1 + 4 => r.1 = s + 1,
                _ => ranges.push((s, s + 1)),
            }
        }
        if ranges.len() > 32 {
            ranges = vec![(ranges[0].0, ranges[ranges.len() - 1].1)];
        }
        d.clear();
        self.dirty = d;
        ranges
    }

    pub fn objects(&self, a: usize, b: usize) -> Vec<GpuObject> {
        self.infos[a..b].iter().map(|f| f.obj).collect()
    }

    /// Main groups to draw: (pipeline, texture, mesh, group id), per pipeline nearest first.
    /// Groups whose members can be neither in the frustum nor at their LOD level are skipped
    /// (conservative bounds: the GPU appended nothing to them).
    pub fn draw_order(&self, near: &[u32], reverse: bool, planes: &[Vec4; 6], cam: Vec3, lod_bias: f32) -> Vec<(u8, u32, u32, u32)> {
        let mut v: Vec<(u8, u32, u32, u32)> = (0..self.main.keys.len())
            .filter(|&g| self.main.members[g] > 0)
            .filter(|&g| {
                let (a, b) = (self.main.bmin[g], self.main.bmax[g]);
                // AABB vs plane: the corner furthest along the normal
                planes.iter().all(|p| {
                    let n = p.truncate();
                    n.dot(Vec3::select(n.cmpge(Vec3::ZERO), b, a)) + p.w >= 0.0
                }) && self.main.lod_possible(g, cam, lod_bias)
            })
            .map(|g| {
                let k = self.main.keys[g];
                (k.0, k.1, k.2, g as u32)
            })
            .collect();
        let d = |g: u32| near.get(g as usize).map_or(0.0, |&b| if b == NONE { f32::MAX } else { f32::from_bits(b) });
        v.sort_by(|a, b| a.0.cmp(&b.0).then(d(a.3).total_cmp(&d(b.3))).then(a.3.cmp(&b.3)));
        if reverse {
            v.reverse();
        }
        v
    }

    /// Shadow groups to draw: (alpha-tested, texture, mesh, group id).
    pub fn shadow_groups(&self, cam: Vec3, lod_bias: f32) -> Vec<(bool, u32, u32, u32)> {
        let mut v: Vec<(bool, u32, u32, u32)> = (0..self.shadow.keys.len())
            .filter(|&g| self.shadow.members[g] > 0 && self.shadow.lod_possible(g, cam, lod_bias))
            .map(|g| {
                let k = self.shadow.keys[g];
                (k.0 == 1, k.1, k.2, g as u32)
            })
            .collect();
        v.sort_by_key(|g| (g.0, g.1));
        v
    }

    /// Main draw groups: (pipeline, texture key, mesh, members).
    pub fn group_members(&self) -> Vec<(u8, u32, u32, u32)> {
        self.main.keys.iter().zip(&self.main.members).map(|(k, &m)| (k.0, k.1, k.2, m)).collect()
    }

    /// Per shadow group: has a dynamic caster (the dynamic overlay of a split pass draws only these).
    pub fn dynamic_shadow_groups(&self) -> Vec<bool> {
        let mut v = vec![false; self.shadow.keys.len()];
        for &i in &self.dyn_slots {
            let inf = &self.infos[i as usize];
            if inf.active && inf.obj.info[1] & 6 == 6 {
                for &g in &inf.obj.sgroups {
                    if let Some(f) = v.get_mut(g as usize) {
                        *f = true;
                    }
                }
            }
        }
        v
    }

    /// Can a member of shadow group `g` touch this cascade's light box? (same test as cull.wgsl)
    pub fn shadow_group_in_box(&self, g: u32, lv: &Mat4, half: f32, depth: f32) -> bool {
        let (c, r) = self.shadow.sphere(g as usize);
        let p = lv.transform_point3(c);
        p.x.abs() <= half + r && p.y.abs() <= half + r && -p.z >= -r && -p.z <= depth + r
    }

    /// Blended entities in view: (view depth, double_sided, texture, mesh, slot), unsorted; and
    /// whether one of them is water.
    pub fn blended(&self, planes: &[Vec4; 6], cam: Vec3, fwd: Vec3, lod_bias: f32, layers: u32, meshes: &[GpuMesh]) -> (Vec<(f32, bool, u32, u32, u32)>, bool) {
        let mut out = vec![];
        let mut water = false;
        for &s in &self.blend_slots {
            let inf = &self.infos[s as usize];
            let Some(b) = &inf.blend else { continue };
            if layers & (1u32 << (inf.layer & 31)) == 0 {
                continue;
            }
            let o = &inf.obj;
            let n = (o.info[0] & 0xff) as usize;
            let nd = ((o.info[0] >> 8) & 0xff) as usize;
            let level = if nd == 0 && (o.info[0] >> 17) & 1 == 0 {
                Some(0)
            } else {
                let d = (Vec3::new(o.origin[0], o.origin[1], o.origin[2]) - cam).length() * o.origin[3] / lod_bias;
                match (0..nd).find(|&k| d <= o.lod[k]) {
                    Some(k) => (k < n).then_some(k),
                    None => ((o.info[0] >> 16) & 1 == 0).then(|| n.checked_sub(1)).flatten(),
                }
            };
            let Some(level) = level else { continue };
            let mesh = b.meshes[level];
            let Some(gm) = meshes.get(mesh as usize).filter(|g| g.index_count > 0) else { continue };
            // the drawn mesh's own bounds, as on the CPU path
            let m = &inf.model;
            let c = m.transform_point3(gm.bounds_center);
            let r = gm.bounds_radius * scale_of(m);
            if !planes.iter().all(|p| p.truncate().dot(c) + p.w >= -r) {
                continue;
            }
            water |= b.water;
            out.push(((c - cam).dot(fwd), b.double_sided, b.tex, mesh, s));
        }
        (out, water)
    }

    /// State for the `gpu_scene` command.
    pub fn describe(&self, owner: &[(u64, bool)]) -> serde_json::Value {
        let changed: Vec<serde_json::Value> = self
            .changed
            .iter()
            .take(24)
            .map(|&(s, why)| {
                let id = owner.get(s as usize).and_then(|o| hecs::Entity::from_bits(o.0)).map(|e| e.id());
                serde_json::json!({"slot": s, "entity": id, "why": why})
            })
            .collect();
        serde_json::json!({
            "changed_count": self.changed.len(),
            "changed": changed,
            "shadow_changes": self.shadow_changes.len(),
            "dynamic_casters": self.dyn_slots.len(),
            "static_changes": self.static_change_slots.iter().take(24).map(|&(s, why)| {
                let id = owner.get(s as usize).and_then(|o| hecs::Entity::from_bits(o.0)).map(|e| e.id());
                serde_json::json!({"slot": s, "entity": id, "why": why})
            }).collect::<Vec<_>>(),
            "slots": self.infos.len(),
            "active": self.infos.iter().filter(|f| f.active).count(),
            "main_objects": self.main_count,
            "blended": self.blend_slots.len(),
            "main_groups": self.main.members.iter().filter(|&&m| m > 0).count(),
            "shadow_groups": self.shadow.members.iter().filter(|&&m| m > 0).count(),
            "ids_main": self.main.total,
            "ids_per_shadow_pass": self.shadow.total,
        })
    }
}

// --- GPU side ---------------------------------------------------------------------------------

/// Last completed readback: counters + nearest distance per main group.
#[derive(Default)]
pub(super) struct Readback {
    pub counters: [u32; N_COUNTERS],
    pub near: Vec<u32>,
}

pub(super) struct SceneGpu {
    reset_pipeline: GpuComputePipeline,
    cull_pipeline: GpuComputePipeline,
    bgl: GpuBindGroupLayout,
    params_buf: GpuBuffer,
    objects_buf: GpuBuffer,
    objects_cap: u32,
    pub args_buf: GpuBuffer,
    args_cap: u32,
    near_buf: GpuBuffer,
    near_cap: u32,
    counters_buf: GpuBuffer,
    readback_buf: GpuBuffer,
    readback_cap: u32,
    readback_pending: Rc<Cell<bool>>,
    /// main groups in the readback in flight
    readback_groups: Cell<u32>,
    pub last: Rc<RefCell<Readback>>,
    bg: Option<(GpuBindGroup, u32, u32)>,
}

const OBJ_SIZE: u32 = std::mem::size_of::<GpuObject>() as u32;

impl SceneGpu {
    fn buf(device: &GpuDevice, size: u32, usage: u32, label: &str) -> Result<GpuBuffer, String> {
        let b = device.create_buffer(&GpuBufferDescriptor::new(size.max(16), usage)).map_err(js_err)?;
        b.set_label(label);
        Ok(b)
    }

    pub fn new(device: &GpuDevice) -> Result<Self, String> {
        let cs = gpu_shader_stage::COMPUTE;
        let buf_entry = |binding: u32, ty: GpuBufferBindingType| {
            let e = GpuBindGroupLayoutEntry::new(binding, cs);
            let b = GpuBufferBindingLayout::new();
            b.set_type(ty);
            e.set_buffer(&b);
            e
        };
        let hiz = GpuBindGroupLayoutEntry::new(5, cs);
        let t = GpuTextureBindingLayout::new();
        t.set_sample_type(GpuTextureSampleType::UnfilterableFloat);
        hiz.set_texture(&t);
        let bgl = device
            .create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&[
                buf_entry(0, GpuBufferBindingType::Uniform),
                buf_entry(1, GpuBufferBindingType::ReadOnlyStorage),
                buf_entry(2, GpuBufferBindingType::Storage),
                buf_entry(3, GpuBufferBindingType::Storage),
                buf_entry(4, GpuBufferBindingType::Storage),
                hiz,
                buf_entry(6, GpuBufferBindingType::Storage),
            ]))
            .map_err(js_err)?;
        let module = device.create_shader_module(&GpuShaderModuleDescriptor::new(include_str!("shaders/cull.wgsl")));
        module.set_label("cull.wgsl");
        crate::debug::report_shader_errors(&module, "cull.wgsl");
        let pipeline = |entry: &str| {
            let pl = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&[JsNullable::wrap(bgl.clone())]));
            let stage = GpuProgrammableStage::new(&module);
            stage.set_entry_point(entry);
            let d = GpuComputePipelineDescriptor::new(&pl, &stage);
            d.set_label(entry);
            device.create_compute_pipeline(&d)
        };
        let storage = gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST;
        Ok(Self {
            reset_pipeline: pipeline("reset"),
            cull_pipeline: pipeline("cull"),
            bgl,
            params_buf: Self::buf(device, std::mem::size_of::<SceneParams>() as u32, gpu_buffer_usage::UNIFORM | gpu_buffer_usage::COPY_DST, "scene-params")?,
            objects_buf: Self::buf(device, 1024 * OBJ_SIZE, storage, "scene-objects")?,
            objects_cap: 1024,
            args_buf: Self::buf(device, 256 * 20, storage | gpu_buffer_usage::INDIRECT, "scene-args")?,
            args_cap: 256,
            near_buf: Self::buf(device, 256 * 4, storage | gpu_buffer_usage::COPY_SRC, "scene-near")?,
            near_cap: 256,
            counters_buf: Self::buf(device, N_COUNTERS as u32 * 4, storage | gpu_buffer_usage::COPY_SRC, "scene-counters")?,
            readback_buf: Self::buf(device, 1024, gpu_buffer_usage::MAP_READ | gpu_buffer_usage::COPY_DST, "scene-readback")?,
            readback_cap: 1024,
            readback_pending: Rc::new(Cell::new(false)),
            readback_groups: Cell::new(0),
            last: Rc::new(RefCell::new(Readback::default())),
            bg: None,
        })
    }

    /// Grows the buffers; true if the object buffer was replaced (upload every slot).
    pub fn ensure(&mut self, device: &GpuDevice, objects: u32, args: u32, groups: u32) -> Result<bool, String> {
        let storage = gpu_buffer_usage::STORAGE | gpu_buffer_usage::COPY_DST;
        let mut replaced = false;
        if objects > self.objects_cap {
            self.objects_cap = objects.next_power_of_two();
            self.objects_buf = Self::buf(device, self.objects_cap * OBJ_SIZE, storage, "scene-objects")?;
            replaced = true;
            self.bg = None;
        }
        if args > self.args_cap {
            self.args_cap = args.next_power_of_two();
            self.args_buf = Self::buf(device, self.args_cap * 20, storage | gpu_buffer_usage::INDIRECT, "scene-args")?;
            self.bg = None;
        }
        if groups > self.near_cap {
            self.near_cap = groups.next_power_of_two();
            self.near_buf = Self::buf(device, self.near_cap * 4, storage | gpu_buffer_usage::COPY_SRC, "scene-near")?;
            self.bg = None;
        }
        Ok(replaced)
    }

    pub fn write_objects(&self, queue: &GpuQueue, at: usize, objs: &[GpuObject]) -> Result<(), String> {
        queue.write_buffer_with_u32_and_u8_slice(&self.objects_buf, at as u32 * OBJ_SIZE, bytemuck::cast_slice(objs)).map_err(js_err)
    }

    pub fn write_args(&self, queue: &GpuQueue, args: &[DrawArgs]) -> Result<(), String> {
        if args.is_empty() {
            return Ok(());
        }
        queue.write_buffer_with_u32_and_u8_slice(&self.args_buf, 0, bytemuck::cast_slice(args)).map_err(js_err)
    }

    /// Encodes reset + cull. `ids_gen` / `hiz_gen` identify the id buffer and Hi-Z texture.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(&mut self, device: &GpuDevice, queue: &GpuQueue, enc: &GpuCommandEncoder, ids: &GpuBuffer, ids_gen: u32, hiz: &GpuTextureView, hiz_gen: u32, params: &SceneParams, ts: Option<GpuComputePassTimestampWrites>) -> Result<(), String> {
        queue.write_buffer_with_u32_and_u8_slice(&self.params_buf, 0, bytemuck::bytes_of(params)).map_err(js_err)?;
        if !matches!(&self.bg, Some((_, a, b)) if *a == ids_gen && *b == hiz_gen) {
            let g = device.create_bind_group(&GpuBindGroupDescriptor::new(
                &[
                    GpuBindGroupEntry::new_with_gpu_buffer(0, &self.params_buf),
                    GpuBindGroupEntry::new_with_gpu_buffer(1, &self.objects_buf),
                    GpuBindGroupEntry::new_with_gpu_buffer(2, &self.args_buf),
                    GpuBindGroupEntry::new_with_gpu_buffer(3, ids),
                    GpuBindGroupEntry::new_with_gpu_buffer(4, &self.counters_buf),
                    GpuBindGroupEntry::new_with_gpu_texture_view(5, hiz),
                    GpuBindGroupEntry::new_with_gpu_buffer(6, &self.near_buf),
                ],
                &self.bgl,
            ));
            self.bg = Some((g, ids_gen, hiz_gen));
        }
        let bg = &self.bg.as_ref().unwrap().0;
        let pass = crate::gpu_cull::compute_pass(enc, ts);
        pass.set_label("cull");
        pass.set_bind_group(0, Some(bg));
        pass.set_pipeline(&self.reset_pipeline);
        pass.dispatch_workgroups(params.n_args.max(params.n_groups).max(N_COUNTERS as u32).div_ceil(64));
        if params.count > 0 {
            pass.set_pipeline(&self.cull_pipeline);
            pass.dispatch_workgroups(params.count.div_ceil(64));
        }
        pass.end();
        Ok(())
    }

    /// Copies counters + nearest distances for async readback (skipped while one is in flight).
    pub fn encode_readback(&mut self, device: &GpuDevice, enc: &GpuCommandEncoder, n_groups: u32) -> bool {
        if self.readback_pending.get() {
            return false;
        }
        let need = (N_COUNTERS as u32 + n_groups) * 4;
        if need > self.readback_cap {
            self.readback_cap = need.next_power_of_two();
            match Self::buf(device, self.readback_cap, gpu_buffer_usage::MAP_READ | gpu_buffer_usage::COPY_DST, "scene-readback") {
                Ok(b) => self.readback_buf = b,
                Err(_) => return false,
            }
        }
        if enc.copy_buffer_to_buffer_with_u32_and_u32_and_u32(&self.counters_buf, 0, &self.readback_buf, 0, N_COUNTERS as u32 * 4).is_err() {
            return false;
        }
        if n_groups > 0 && enc.copy_buffer_to_buffer_with_u32_and_u32_and_u32(&self.near_buf, 0, &self.readback_buf, N_COUNTERS as u32 * 4, n_groups * 4).is_err() {
            return false;
        }
        self.readback_groups.set(n_groups);
        true
    }

    /// Call after `queue.submit` if `encode_readback` returned true.
    pub fn start_readback(&self) {
        self.readback_pending.set(true);
        let buf = self.readback_buf.clone();
        let pending = self.readback_pending.clone();
        let out = self.last.clone();
        let n_groups = self.readback_groups.get();
        let promise = buf.map_async_with_u32_and_u32(gpu_map_mode::READ, 0, (N_COUNTERS as u32 + n_groups) * 4);
        wasm_bindgen_futures::spawn_local(async move {
            if wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise.unchecked_into::<wasm_bindgen::JsValue>())).await.is_ok() {
                if let Ok(range) = buf.get_mapped_range_with_u32_and_u32(0, (N_COUNTERS as u32 + n_groups) * 4) {
                    let arr = js_sys::Uint32Array::new(&range);
                    let mut o = out.borrow_mut();
                    for (i, v) in o.counters.iter_mut().enumerate() {
                        *v = arr.get_index(i as u32);
                    }
                    o.near = (0..n_groups).map(|g| arr.get_index(N_COUNTERS as u32 + g)).collect();
                }
                buf.unmap();
            }
            pending.set(false);
        });
    }
}

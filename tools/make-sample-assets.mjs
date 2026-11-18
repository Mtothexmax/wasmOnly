#!/usr/bin/env node
// Generates the demo's sample assets (procedural, no downloads):
//   games/demo/assets/tree.glb   — low-poly tree, 1 mesh with 2 primitives/materials
//   games/demo/assets/crate.glb  — textured cube (embedded PNG -> exercises createImageBitmap path)
//   games/demo/assets-src/rock.obj — an OBJ to try tools/asset-cook (russimp) with
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'games', 'demo', 'assets');
const SRC = path.join(ROOT, 'games', 'demo', 'assets-src');
fs.mkdirSync(OUT, { recursive: true });
fs.mkdirSync(SRC, { recursive: true });

// ---------- PNG -------------------------------------------------------------------------------
const CRC = new Uint32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc32 = (buf) => { let c = 0xffffffff; for (const b of buf) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
function png(w, h, pixel) {
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 4 + 1)] = 0;
    for (let x = 0; x < w; x++) raw.set(pixel(x, y), y * (w * 4 + 1) + 1 + x * 4);
  }
  const chunk = (type, data) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 6;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

// ---------- GLB writer --------------------------------------------------------------------------
class Glb {
  constructor() { this.bin = []; this.len = 0; this.views = []; this.accessors = []; this.json = { asset: { version: '2.0', generator: 'wasmOnly make-sample-assets' } }; }
  view(buf, target) {
    const pad = (4 - (this.len % 4)) % 4; if (pad) { this.bin.push(Buffer.alloc(pad)); this.len += pad; }
    const v = { buffer: 0, byteOffset: this.len, byteLength: buf.length }; if (target) v.target = target;
    this.bin.push(buf); this.len += buf.length; this.views.push(v); return this.views.length - 1;
  }
  floats(arr, type, n, bounds) {
    const view = this.view(Buffer.from(new Float32Array(arr).buffer), 34962);
    const a = { bufferView: view, componentType: 5126, count: arr.length / n, type };
    if (bounds) { a.min = [0, 1, 2].map((i) => Math.min(...arr.filter((_, k) => k % 3 === i))); a.max = [0, 1, 2].map((i) => Math.max(...arr.filter((_, k) => k % 3 === i))); }
    this.accessors.push(a); return this.accessors.length - 1;
  }
  indices(arr) {
    const view = this.view(Buffer.from(new Uint32Array(arr).buffer), 34963);
    this.accessors.push({ bufferView: view, componentType: 5125, count: arr.length, type: 'SCALAR' }); return this.accessors.length - 1;
  }
  prim({ pos, nrm, uv, idx }, material) {
    const attributes = { POSITION: this.floats(pos, 'VEC3', 3, true), NORMAL: this.floats(nrm, 'VEC3', 3) };
    if (uv) attributes.TEXCOORD_0 = this.floats(uv, 'VEC2', 2);
    return { attributes, indices: this.indices(idx), material };
  }
  write(file) {
    const binBuf = Buffer.concat(this.bin); const pad = (4 - (binBuf.length % 4)) % 4; const bin = Buffer.concat([binBuf, Buffer.alloc(pad)]);
    Object.assign(this.json, { accessors: this.accessors, bufferViews: this.views, buffers: [{ byteLength: bin.length }] });
    let json = Buffer.from(JSON.stringify(this.json)); json = Buffer.concat([json, Buffer.alloc((4 - (json.length % 4)) % 4, 0x20)]);
    const header = Buffer.alloc(12); header.write('glTF', 0); header.writeUInt32LE(2, 4); header.writeUInt32LE(12 + 8 + json.length + 8 + bin.length, 8);
    const ch = (len, type) => { const b = Buffer.alloc(8); b.writeUInt32LE(len, 0); b.write(type, 4); return b; };
    fs.writeFileSync(file, Buffer.concat([header, ch(json.length, 'JSON'), json, ch(bin.length, 'BIN\0'), bin]));
    console.log(JSON.stringify({ wrote: path.relative(ROOT, file), bytes: fs.statSync(file).size }));
  }
}

// ---------- geometry (flat-shaded, non-indexed for a low-poly look) ----------------------------
function flat(tris) { // tris: [[a,b,c], ...] with points [x,y,z]
  const pos = [], nrm = [], idx = [];
  for (const [a, b, c] of tris) {
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const l = Math.hypot(...n) || 1; n = n.map((x) => x / l);
    for (const p of [a, b, c]) { idx.push(pos.length / 3); pos.push(...p); nrm.push(...n); }
  }
  return { pos, nrm, idx };
}
function cylinder(r0, r1, y0, y1, seg, jitter = 0) {
  const tris = []; let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * jitter;
  const ring = (r, y) => Array.from({ length: seg }, (_, i) => { const a = (i / seg) * Math.PI * 2; return [Math.cos(a) * r + rnd(), y, Math.sin(a) * r + rnd()]; });
  const lo = ring(r0, y0), hi = r1 > 0 ? ring(r1, y1) : null, tip = [0, y1, 0];
  for (let i = 0; i < seg; i++) {
    const j = (i + 1) % seg;
    if (hi) { tris.push([lo[i], hi[i], hi[j]], [lo[i], hi[j], lo[j]]); } else tris.push([lo[i], tip, lo[j]]);
    tris.push([lo[j], [0, y0, 0], lo[i]]); // bottom cap
  }
  return tris;
}

// tree.glb
{
  const g = new Glb();
  const trunk = g.prim(flat(cylinder(0.22, 0.16, 0, 1.4, 7)), 0);
  const leaves = g.prim(flat([...cylinder(1.25, 0, 1.0, 2.6, 8, 0.15), ...cylinder(0.95, 0, 1.9, 3.4, 8, 0.12)]), 1);
  Object.assign(g.json, {
    scene: 0, scenes: [{ nodes: [0] }], nodes: [{ name: 'tree', mesh: 0 }],
    meshes: [{ name: 'tree', primitives: [trunk, leaves] }],
    materials: [
      { name: 'bark', pbrMetallicRoughness: { baseColorFactor: [0.23, 0.13, 0.07, 1], metallicFactor: 0, roughnessFactor: 0.95 } },
      { name: 'leaves', pbrMetallicRoughness: { baseColorFactor: [0.11, 0.33, 0.12, 1], metallicFactor: 0, roughnessFactor: 0.8 } },
    ],
  });
  g.write(path.join(OUT, 'tree.glb'));
}

// crate.glb (textured)
{
  const S = 128;
  const tex = png(S, S, (x, y) => {
    const border = x < 10 || y < 10 || x >= S - 10 || y >= S - 10;
    const diag = Math.abs(x - y) < 7 || Math.abs(x - (S - 1 - y)) < 7;
    const plank = Math.floor(y / 27) % 2;
    const grain = Math.sin(x * 0.35 + y * 0.05) * 10 + ((x * 7 + y * 13) % 9);
    if (border || diag) return [115 + grain, 72 + grain * 0.6, 36, 255];
    return [190 + grain - plank * 18, 138 + grain * 0.7 - plank * 14, 74, 255];
  });
  const g = new Glb();
  const faces = [[[1, 0, 0], [0, 0, -1], [0, 1, 0]], [[-1, 0, 0], [0, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]];
  const pos = [], nrm = [], uv = [], idx = [];
  for (const [n, u, v] of faces) {
    const base = pos.length / 3;
    for (const [su, sv] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      pos.push(...[0, 1, 2].map((i) => (n[i] + u[i] * su + v[i] * sv) * 0.5)); nrm.push(...n); uv.push((su + 1) / 2, 1 - (sv + 1) / 2);
    }
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  const prim = g.prim({ pos, nrm, uv, idx }, 0);
  const img = g.view(tex);
  Object.assign(g.json, {
    scene: 0, scenes: [{ nodes: [0] }], nodes: [{ name: 'crate', mesh: 0 }],
    meshes: [{ name: 'crate', primitives: [prim] }],
    materials: [{ name: 'wood', pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicFactor: 0, roughnessFactor: 0.75 } }],
    images: [{ bufferView: img, mimeType: 'image/png' }], textures: [{ source: 0, sampler: 0 }],
    samplers: [{ magFilter: 9729, minFilter: 9729 }],
  });
  g.write(path.join(OUT, 'crate.glb'));
}

// critter.glb — skinned body (3 joints) + head with 2 morph targets + 3 clips
// (LINEAR rotations + morph weights, CUBICSPLINE translation, STEP scale). Tests engine::animation.
{
  const g = new Glb();
  const data = (arr, type, comp = 5126, minmax = false) => {
    const Typed = comp === 5123 ? Uint16Array : Float32Array;
    const view = g.view(Buffer.from(new Typed(arr).buffer));
    const n = { SCALAR: 1, VEC3: 3, VEC4: 4, MAT4: 16 }[type];
    const a = { bufferView: view, componentType: comp, count: arr.length / n, type };
    if (minmax) { a.min = [Math.min(...arr)]; a.max = [Math.max(...arr)]; }
    g.accessors.push(a); return g.accessors.length - 1;
  };
  // body: smooth cylinder r 0.3, y 0..1.8, weights blend between joints at y = 0, 0.6, 1.2
  const seg = 12, rings = 18, H = 1.8, pos = [], nrm = [], idx = [], jnt = [], wgt = [];
  for (let r = 0; r <= rings; r++) {
    const y = (r / rings) * H, t = Math.min(y / 0.6, 2.0), i = Math.min(Math.floor(t), 2), f = i === 2 ? 0 : t - i;
    for (let s = 0; s <= seg; s++) {
      const a = (s / seg) * Math.PI * 2; const rad = 0.3 * (1 - 0.25 * (y / H));
      pos.push(Math.cos(a) * rad, y, Math.sin(a) * rad); nrm.push(Math.cos(a), 0, Math.sin(a));
      jnt.push(i, Math.min(i + 1, 2), 0, 0); wgt.push(1 - f, f, 0, 0);
    }
  }
  for (let r = 0; r < rings; r++) for (let s = 0; s < seg; s++) {
    const a = r * (seg + 1) + s, b = a + 1, c = a + seg + 1, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  const body = { attributes: { POSITION: g.floats(pos, 'VEC3', 3, true), NORMAL: g.floats(nrm, 'VEC3', 3), JOINTS_0: data(jnt, 'VEC4', 5123), WEIGHTS_0: data(wgt, 'VEC4') }, indices: g.indices(idx), material: 0 };
  // head: sphere r 0.35 + morph targets "squash" and "spikes"
  const lat = 12, lon = 16, hp = [], hn = [], hi = [], sq = [], sp = [];
  for (let i = 0; i <= lat; i++) for (let j = 0; j <= lon; j++) {
    const t = (i / lat) * Math.PI, p = (j / lon) * Math.PI * 2;
    const n = [Math.sin(t) * Math.cos(p), Math.cos(t), Math.sin(t) * Math.sin(p)];
    hp.push(...n.map((x) => x * 0.35)); hn.push(...n);
    sq.push(n[0] * 0.12, -n[1] * 0.16, n[2] * 0.12);
    const spike = Math.max(0, Math.sin(t * 4) * Math.sin(p * 4)) * 0.18; sp.push(...n.map((x) => x * spike));
  }
  for (let i = 0; i < lat; i++) for (let j = 0; j < lon; j++) {
    const a = i * (lon + 1) + j, b = a + 1, c = a + lon + 1, d = c + 1;
    hi.push(a, b, c, b, d, c);
  }
  const head = { attributes: { POSITION: g.floats(hp, 'VEC3', 3, true), NORMAL: g.floats(hn, 'VEC3', 3) }, indices: g.indices(hi), material: 1,
    targets: [{ POSITION: g.floats(sq, 'VEC3', 3, true) }, { POSITION: g.floats(sp, 'VEC3', 3, true) }] };
  const ibm = data([...[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], ...[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,-0.6,0,1], ...[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,-1.2,0,1]], 'MAT4');
  const qz = (a) => [0, 0, Math.sin(a / 2), Math.cos(a / 2)], qy = (a) => [0, Math.sin(a / 2), 0, Math.cos(a / 2)];
  const anims = [];
  const sampler = (times, values, type, interpolation = 'LINEAR') => ({ input: data(times, 'SCALAR', 5126, true), output: data(values, type), interpolation });
  { // wave: spine + neck sway (LINEAR quaternions), head squash (LINEAR weights)
    const T = [0, 0.5, 1, 1.5, 2];
    const s = [sampler(T, [0, 0.35, 0, -0.35, 0].flatMap(qz), 'VEC4'), sampler(T, [0, 0.5, 0, -0.5, 0].flatMap(qz), 'VEC4'), sampler(T, [0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 'SCALAR')];
    anims.push({ name: 'wave', samplers: s, channels: [{ sampler: 0, target: { node: 2, path: 'rotation' } }, { sampler: 1, target: { node: 3, path: 'rotation' } }, { sampler: 2, target: { node: 5, path: 'weights' } }] });
  }
  { // bounce: hip jump (CUBICSPLINE: in-tangent, value, out-tangent), head squashes on landing (STEP scale)
    const T = [0, 0.4, 0.8];
    const tr = [0, 0, 0, 0, 0, 0, 0, 2.5, 0, /**/ 0, 0, 0, 0, 0.5, 0, 0, 0, 0, /**/ 0, -2.5, 0, 0, 0, 0, 0, 0, 0];
    const s = [sampler(T, tr, 'VEC3', 'CUBICSPLINE'), sampler([0, 0.4, 0.7], [1.25, 0.8, 1.25, 1, 1, 1, 1.25, 0.8, 1.25], 'VEC3', 'STEP')];
    anims.push({ name: 'bounce', samplers: s, channels: [{ sampler: 0, target: { node: 1, path: 'translation' } }, { sampler: 1, target: { node: 5, path: 'scale' } }] });
  }
  { // idle: slow turn + spikes pulse
    const T = [0, 1.5, 3];
    const s = [sampler(T, [0, 0.5, 0].flatMap(qy), 'VEC4'), sampler(T, [0, 0, 0, 0.8, 0, 0], 'SCALAR')];
    anims.push({ name: 'idle', samplers: s, channels: [{ sampler: 0, target: { node: 1, path: 'rotation' } }, { sampler: 1, target: { node: 5, path: 'weights' } }] });
  }
  Object.assign(g.json, {
    scene: 0, scenes: [{ nodes: [0] }],
    nodes: [
      { name: 'critter', children: [1, 4] },
      { name: 'hip', children: [2] },
      { name: 'spine', translation: [0, 0.6, 0], children: [3] },
      { name: 'neck', translation: [0, 0.6, 0], children: [5] },
      { name: 'body', mesh: 0, skin: 0 },
      { name: 'head', translation: [0, 0.75, 0], mesh: 1 },
    ],
    meshes: [{ name: 'body', primitives: [body] }, { name: 'head', primitives: [head], weights: [0, 0], extras: { targetNames: ['squash', 'spikes'] } }],
    skins: [{ name: 'spine', joints: [1, 2, 3], inverseBindMatrices: ibm, skeleton: 1 }],
    animations: anims,
    materials: [
      { name: 'skin', pbrMetallicRoughness: { baseColorFactor: [0.2, 0.55, 0.65, 1], metallicFactor: 0, roughnessFactor: 0.6 } },
      { name: 'glow', pbrMetallicRoughness: { baseColorFactor: [1, 0.55, 0.2, 1], metallicFactor: 0, roughnessFactor: 0.4 }, emissiveFactor: [1, 0.45, 0.12], extensions: { KHR_materials_emissive_strength: { emissiveStrength: 1.3 } } },
    ],
    extensionsUsed: ['KHR_materials_emissive_strength'],
  });
  g.write(path.join(OUT, 'critter.glb'));
}

// rock.obj (source asset for tools/asset-cook)
{
  let seed = 3; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const lat = 6, lon = 9, v = [], f = [];
  for (let i = 0; i <= lat; i++) for (let j = 0; j < lon; j++) {
    const t = (i / lat) * Math.PI, p = (j / lon) * Math.PI * 2, r = 0.8 + rnd() * 0.35;
    v.push([Math.sin(t) * Math.cos(p) * r * 1.2, Math.cos(t) * r * 0.7, Math.sin(t) * Math.sin(p) * r]);
  }
  for (let i = 0; i < lat; i++) for (let j = 0; j < lon; j++) {
    const a = i * lon + j + 1, b = i * lon + ((j + 1) % lon) + 1, c = a + lon, d = b + lon;
    f.push([a, b, d], [a, d, c]);
  }
  const obj = ['# procedural rock — cook with: cargo run --release --manifest-path tools/asset-cook/Cargo.toml -- games/demo/assets-src/rock.obj games/demo/assets/rock.glb',
    'mtllib rock.mtl', 'o rock', ...v.map((p) => `v ${p.map((x) => x.toFixed(4)).join(' ')}`), 'usemtl stone', 's 1', ...f.map((t) => `f ${t.join(' ')}`)].join('\n');
  fs.writeFileSync(path.join(SRC, 'rock.obj'), obj + '\n');
  fs.writeFileSync(path.join(SRC, 'rock.mtl'), 'newmtl stone\nKd 0.45 0.43 0.40\nNs 10\n');
  console.log(JSON.stringify({ wrote: 'games/demo/assets-src/rock.obj' }));
}

// skybox-night.png: equirectangular night sky (2:1) — Milky Way band and nebula tint. Blended
// in by `render.skybox_night_blend` after `ctx.load_skybox("assets/skybox-night.png")`.
{
  const W = 2048, H = 1024;
  const hash = (x, y) => { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); };
  const noise = (x, y) => {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  };
  const fbm = (x, y) => { let s = 0, a = 0.5; for (let o = 0; o < 6; o++) { s += a * noise(x, y); x *= 2.03; y *= 2.03; a *= 0.5; } return s; };
  const img = new Float32Array(W * H * 3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const lon = (x / W) * Math.PI * 2, lat = Math.PI / 2 - (y / H) * Math.PI;
    const d = [Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)];
    // galactic plane tilted 60 degrees
    const n = [0, Math.cos(1.05), Math.sin(1.05)];
    const g = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
    const band = Math.exp(-g * g * 18);
    const u = (x / W) * 8, v = (y / H) * 4;
    const cloud = fbm(u * 2, v * 2), dust = fbm(u * 5 + 3, v * 5);
    const glow = band * (0.35 + 0.65 * cloud) * (1 - 0.7 * Math.max(0, dust - 0.45) * 2.2);
    const neb = Math.max(0, fbm(u * 1.3 + 9, v * 1.3) - 0.55) * 1.4;
    const i = (y * W + x) * 3;
    img[i] = 0.006 + glow * 0.07 + neb * 0.025;
    img[i + 1] = 0.008 + glow * 0.065 + neb * 0.01;
    img[i + 2] = 0.018 + glow * 0.09 + neb * 0.04;
  }
  // no stars in the image: the engine's procedural stars stay pixel-sharp on top of any skybox
  const srgb = (l) => Math.round(255 * Math.min(1, l <= 0.0031308 ? l * 12.92 : 1.055 * Math.pow(l, 1 / 2.4) - 0.055));
  fs.writeFileSync(path.join(OUT, 'skybox-night.png'), png(W, H, (x, y) => { const i = (y * W + x) * 3; return [srgb(img[i]), srgb(img[i + 1]), srgb(img[i + 2]), 255]; }));
  console.log(JSON.stringify({ wrote: 'games/demo/assets/skybox-night.png' }));
}

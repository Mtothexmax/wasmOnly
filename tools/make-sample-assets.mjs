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

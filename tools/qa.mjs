// Automated visual QA for agents (used by tools/bridge.mjs): screenshots can't show flicker and
// eyeballing is unreliable, so these checks return numbers + names instead of pictures.
//
//   check   — reload, wait for assets, z-fighting test, engine warnings/errors (+ optional flicker)
//   flicker — N screenshots alternating draw order (exact depth ties flip) and a perturbed depth
//             mapping (camera near plane; near-ties flip), CPU draw path, frozen shader time, hidden
//             HUD. Pixel positions stay identical, so only z-fighting changes between frames. Regions are named via
//             the engine's `pick`, a heatmap PNG is written.
//
// Zero npm dependencies: PNG is decoded/encoded here with node:zlib.

import zlib from 'node:zlib';
import fs from 'node:fs';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------------------------
// Minimal PNG (8-bit RGB/RGBA, non-interlaced) — enough for CDP screenshots.

export function decodePng(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error('not a PNG');
  let off = 8, w = 0, h = 0, ct = 0;
  const idat = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off);
    const type = buf.toString('ascii', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4);
      if (data[8] !== 8 || data[12] !== 0) throw new Error('only 8-bit non-interlaced PNG');
      ct = data[9];
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    off += 12 + len;
  }
  const bpp = ct === 6 ? 4 : ct === 2 ? 3 : 0;
  if (!bpp) throw new Error(`PNG color type ${ct} not supported`);
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = w * bpp;
  const out = Buffer.alloc(w * h * 4);
  const prev = Buffer.alloc(stride), cur = Buffer.alloc(stride);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)];
    raw.copy(cur, 0, y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? cur[x - bpp] : 0, b = prev[x], c = x >= bpp ? prev[x - bpp] : 0;
      let v = cur[x];
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      cur[x] = v & 255;
    }
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      out[o] = cur[x * bpp]; out[o + 1] = cur[x * bpp + 1]; out[o + 2] = cur[x * bpp + 2]; out[o + 3] = bpp === 4 ? cur[x * bpp + 3] : 255;
    }
    cur.copy(prev);
  }
  return { w, h, rgba: out };
}

const CRC = new Uint32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
function crc32(b) { let c = 0xffffffff; for (const x of b) c = CRC[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
export function encodePng(w, h, rgb) {
  const raw = Buffer.alloc(h * (w * 3 + 1));
  for (let y = 0; y < h; y++) rgb.copy(raw, y * (w * 3 + 1) + 1, y * w * 3, (y + 1) * w * 3);
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

// ---------------------------------------------------------------------------------------------

const frames = (s, n = 3) => s.eval(`new Promise(r => { let k = 0; const f = () => (++k >= ${n} ? r(1) : requestAnimationFrame(f)); requestAnimationFrame(f); })`);
const result = (r) => (r && r.ok ? r.result : null);

/**
 * Flicker detection. opts: {frames=6, jitter=0.2 (relative near-plane change), threshold=48 (0-255 channel range),
 * cell=16 px, out=heatmap path}.
 */
export async function flicker(s, opts = {}) {
  const n = opts.frames ?? 6, j = opts.jitter ?? 0.2, thr = opts.threshold ?? 48, cell = opts.cell ?? 16;
  const wasPaused = result(await s.call('stats'))?.time?.paused ?? false;
  const wasGpu = result(await s.call('render'))?.gpu_culling ?? true;
  await s.call('pause');
  await s.call('render', { freeze_time: true, camera_jitter: [0, 0, 0], depth_jitter: 0, gpu_culling: false });
  const dpr = await s.eval(`(() => { const u = document.getElementById('ui'); window.__qaUi = u ? u.style.visibility : ''; if (u) u.style.visibility = 'hidden'; const e = document.getElementById('engine-error'); return devicePixelRatio; })()`);
  const imgs = [];
  try {
    // settle: let paused physics, hidden HUD and render state reach the screen; drop one capture
    await frames(s, 5);
    await s.screenshot();
    for (let k = 0; k < n; k++) {
      await s.call('render', { depth_jitter: n > 1 ? (k / (n - 1) - 0.5) * 2 * j : 0, reverse_draw_order: k % 2 === 1 });
      await frames(s, 3);
      imgs.push(decodePng(Buffer.from(await s.screenshot(), 'base64')));
    }
  } finally {
    await s.call('render', { freeze_time: false, camera_jitter: [0, 0, 0], depth_jitter: 0, reverse_draw_order: false, gpu_culling: wasGpu });
    await s.eval(`(() => { const u = document.getElementById('ui'); if (u) u.style.visibility = window.__qaUi ?? ''; return 1; })()`);
    if (!wasPaused) await s.call('resume');
  }
  const { w, h } = imgs[0];
  const mask = new Uint8Array(w * h);
  let count = 0;
  for (let p = 0; p < w * h; p++) {
    let range = 0;
    for (let ch = 0; ch < 3; ch++) {
      let mn = 255, mx = 0;
      for (const im of imgs) { const v = im.rgba[p * 4 + ch]; if (v < mn) mn = v; if (v > mx) mx = v; }
      range = Math.max(range, mx - mn);
    }
    if (range >= thr) { mask[p] = 1; count++; }
  }
  // regions: cells with a meaningful number of flickering pixels (isolated pixels = edge noise)
  const cw = Math.ceil(w / cell), chh = Math.ceil(h / cell);
  const cells = [];
  for (let cy = 0; cy < chh; cy++) for (let cx = 0; cx < cw; cx++) {
    let c = 0;
    for (let y = cy * cell; y < Math.min(h, (cy + 1) * cell); y++) for (let x = cx * cell; x < Math.min(w, (cx + 1) * cell); x++) c += mask[y * w + x];
    if (c >= cell * cell * 0.08) cells.push({ cx, cy, c });
  }
  cells.sort((a, b) => b.c - a.c);
  const regions = [];
  for (const c of cells.slice(0, 12)) {
    const x = (c.cx + 0.5) * cell / dpr, y = (c.cy + 0.5) * cell / dpr;
    const pick = result(await s.call('pick', { x, y, mesh: true }));
    regions.push({ x: Math.round(x), y: Math.round(y), pixels: c.c, entity: pick?.hit ? pick.entity : null, depth_ties: pick?.depth_ties ?? [] });
  }
  let heatmap = null;
  if (opts.out) {
    const rgb = Buffer.alloc(w * h * 3);
    const base = imgs[0].rgba;
    for (let p = 0; p < w * h; p++) {
      const l = (base[p * 4] * 0.3 + base[p * 4 + 1] * 0.59 + base[p * 4 + 2] * 0.11) * 0.45;
      rgb[p * 3] = mask[p] ? 255 : l; rgb[p * 3 + 1] = mask[p] ? 40 : l; rgb[p * 3 + 2] = mask[p] ? 40 : l;
    }
    fs.writeFileSync(opts.out, encodePng(w, h, rgb));
    heatmap = opts.out;
  }
  return { frames: n, jitter: j, threshold: thr, size: [w, h], flicker_pixels: count, flicker_ratio: +(count / (w * h)).toFixed(6), flicker_cells: cells.length, regions, heatmap };
}

/** Build check: reload, wait for assets, z-fighting, engine log. opts: {flicker, strict, out}. */
export async function check(s, opts = {}) {
  const r = await s.reload();
  const res = { ready: r.ready, ready_ms: r.ms };
  if (!r.ready) return { ok: false, ...res, error: r.fatal ?? 'engine not ready' };
  for (let t = 0; t < 100; t++) {
    const a = result(await s.call('assets'));
    if (a && a.models_pending.length === 0) break;
    await sleep(100);
  }
  await frames(s, 3);
  const z = result(await s.call('zfight', { limit: 10 }));
  const logs = result(await s.call('logs', { level: 'warn', limit: 200 }))?.entries ?? [];
  res.zfight = { pairs: z?.pairs ?? -1, worst: z?.worst ?? [] };
  res.errors = logs.filter((e) => e.level === 'error').map((e) => e.msg);
  res.warnings = logs.filter((e) => e.level === 'warn').map((e) => e.msg);
  if (opts.flicker) res.flicker = await flicker(s, { out: opts.out });
  const flickerBad = res.flicker ? res.flicker.flicker_cells > 0 : false;
  res.ok = res.zfight.pairs === 0 && res.errors.length === 0 && !flickerBad && (!opts.strict || res.warnings.length === 0);
  return res;
}

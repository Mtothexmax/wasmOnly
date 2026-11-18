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

/**
 * Performance capture. opts: {ms=3000, profile=false (Chrome CPU sampling profile), top=15}.
 * Returns frame-time distribution, engine CPU phases, GPU pass times and a verdict
 * (vsync-limited / gpu-bound / cpu-bound / outside-engine).
 */
export async function perf(s, opts = {}) {
  const ms = opts.ms ?? 3000, top = opts.top ?? 15;
  if (opts.profile) {
    await s.send('Profiler.enable');
    await s.send('Profiler.setSamplingInterval', { interval: 250 });
    await s.send('Profiler.start');
  }
  const sample = await s.eval(`new Promise(res => {
    const dts = [], cpu = [], ph = [], gpu = [];
    let last = performance.now(); const end = last + ${ms};
    const f = (t) => {
      dts.push(t - last); last = t;
      const st = engine.call('stats').result;
      cpu.push(st.time.frame_ms); ph.push(st.time.phases); if (st.render.gpu_ms) gpu.push(st.render.gpu_ms);
      if (t < end) requestAnimationFrame(f); else res({ dts: dts.slice(1), cpu, ph, gpu, st,
        dpr: devicePixelRatio, css: [innerWidth, innerHeight], focus: document.hasFocus(), visible: document.visibilityState });
    };
    requestAnimationFrame(f);
  })`);
  const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
  const r2 = (x) => Math.round(x * 100) / 100;
  const d = [...sample.dts].sort((a, b) => a - b);
  const pct = (p) => d[Math.min(d.length - 1, Math.round((d.length - 1) * p))] ?? 0;
  const phases = {};
  for (const k of ['game_fixed', 'physics', 'update', 'engine', 'render']) phases[k] = r2(avg(sample.ph.map((p) => p[k])));
  const gpuLast = sample.gpu.at(-1) ?? null;
  const gpu = gpuLast ? Object.fromEntries(['cull', 'shadow', 'opaque', 'blend', 'hiz', 'total'].map((k) => [k, r2(gpuLast[k])])) : null;
  const res = {
    frames: d.length,
    fps: r2(1000 / avg(sample.dts)),
    frame_ms: { p50: r2(pct(0.5)), p95: r2(pct(0.95)), p99: r2(pct(0.99)), max: r2(pct(1)) },
    slow_frames_over_20ms: sample.dts.filter((x) => x > 20).length,
    engine_cpu_ms: r2(avg(sample.cpu)),
    cpu_phases: phases,
    gpu_ms: gpu,
    canvas: [sample.st.render.width, sample.st.render.height], dpr: sample.dpr, css: sample.css,
    draw_calls: sample.st.render.draw_calls, triangles: sample.st.render.triangles,
    focus: sample.focus, visibility: sample.visible,
  };
  // verdict. Display interval = the most common short interval (rounded to 0.5 ms); averages hide
  // stutter, so the share of frames that missed a refresh decides "smooth" vs "stutter".
  const p50 = res.frame_ms.p50;
  const bins = new Map();
  for (const x of d) if (x > 4 && x < 40) { const k = Math.round(x * 2) / 2; bins.set(k, (bins.get(k) ?? 0) + 1); }
  const mode = [...bins.entries()].sort((x, y) => y[1] - x[1])[0]?.[0] ?? 16.67;
  const near = d.filter((x) => Math.abs(x - mode) < 1);
  const vsync = near.length ? avg(near) : mode;
  const missed = d.filter((x) => x > vsync * 1.5).length, missedPct = r2((100 * missed) / Math.max(1, d.length));
  res.display_hz = r2(1000 / vsync);
  res.missed_frames_pct = missedPct;
  const gpuT = gpu?.total ?? 0, cpuT = res.engine_cpu_ms;
  const budget = vsync;
  const bigGpu = gpu ? Object.entries(gpu).filter(([k]) => !['total', 'samples'].includes(k)).sort((x, y) => y[1] - x[1])[0] : null;
  const bigCpu = Object.entries(phases).sort((x, y) => y[1] - x[1])[0];
  const gpuTxt = gpu ? `GPU ${r2(gpuT)} ms (biggest '${bigGpu[0]}' ${bigGpu[1]} ms at ${res.canvas.join('x')}, MSAA 4x)` : 'GPU unknown';
  const cpuTxt = `engine CPU ${cpuT} ms (biggest '${bigCpu[0]}' ${bigCpu[1]} ms)`;
  if (missedPct <= 2) res.verdict = `smooth: ${missedPct}% missed frames at ~${res.display_hz} Hz; ${gpuTxt}, ${cpuTxt}; nothing to fix`;
  else if (gpuT > budget * 0.75 && gpuT >= cpuT) res.verdict = `gpu-bound: ${gpuTxt} vs a ${r2(budget)} ms budget; ${missedPct}% missed frames`;
  else if (cpuT > budget * 0.75) res.verdict = `cpu-bound: ${cpuTxt} vs a ${r2(budget)} ms budget; ${missedPct}% missed frames`;
  else if (gpuT > budget * 0.25 && gpuT >= cpuT)
    res.verdict = `stutter, likely GPU spikes: ${missedPct}% missed frames; average ${gpuTxt} is ${Math.round((100 * gpuT) / budget)}% of the budget, so slower frames (compositor, other GPU work, thermal/power state) push it over — reduce GPU load (resolution, MSAA, biggest pass) and re-measure`;
  else res.verdict = `stutter outside the engine: ${missedPct}% missed frames but ${gpuTxt}, ${cpuTxt} — browser/compositor, GC, other tabs or a background window (focus=${res.focus}); run with --profile`;
  if (!gpu) res.gpu_note = 'GPU timing unavailable (adapter lacks timestamp-query)';

  if (opts.profile) {
    const { profile } = await s.send('Profiler.stop');
    const self = new Map();
    let total = 0;
    for (const n of profile.nodes) {
      const cf = n.callFrame;
      const name = cf.functionName || `(anonymous ${cf.url.split('/').pop()}:${cf.lineNumber})`;
      const key = `${name}  [${cf.url ? cf.url.split('/').pop() : ''}]`;
      self.set(key, (self.get(key) ?? 0) + (n.hitCount ?? 0));
      total += n.hitCount ?? 0;
    }
    res.cpu_profile = {
      samples: total,
      top_self: [...self.entries()].sort((a, b) => b[1] - a[1]).slice(0, top).map(([k, v]) => ({ fn: k, pct: r2((v / total) * 100) })),
    };
  }
  return res;
}

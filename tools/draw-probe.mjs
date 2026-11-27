#!/usr/bin/env node
// Draw probe for drivers that accept a pipeline but silently draw / dispatch nothing (PowerVR B-series,
// Android 10, Chrome 158: no validation error, no device loss, the draw just vanishes). gpu-probe.mjs only
// checks that pipelines *compile*; this page checks that they *run*, by readback.
//
//   node tools/draw-probe.mjs <out.html>
//
// Open the page on the device (e.g. serve it next to the game). Per test it reports
//   ran      the stage executed (a probe atomic in a storage buffer was incremented / the colour arrived)
//   DROPPED  the pipeline was valid, no error was raised, and nothing executed: the driver bug
//   ERROR    WebGPU validation / compile error (a bug in the probe or a real rejection)
// Sections:
//   1. calibration: trivial vertex / fragment / compute stages (must say "ran", else the probe itself is hit)
//   2. micro: one construct each (let array[k], v[k], uniform array<mat4x4f>[k], loops, atomics, ...)
//      in a fragment and a compute shader; k is a runtime value
//   3. engine: every engine entry point (pbr, shadow, sky, sky_lut, particles, post, cull, hiz, mipgen,
//      impostor) with the real shader source, dummy zero bindings. A vertex stage is wrapped (the real
//      vs runs, then its position is overridden with a full-screen triangle unless a uniform says otherwise,
//      so nothing is optimised away) and drawn with a constant fragment shader; a fragment / compute
//      stage gets a probe atomicAdd as its first statement and a full-screen triangle / one workgroup.
//   4. bisection: for every DROPPED engine entry, each function it reaches is stubbed (zero return) one at
//      a time; the stubs that make it run name the function holding the construct the driver drops.
// Results are shown on the page and POSTed as JSON to /__report and /__log (kind "drawprobe").
// URL options: ?only=<substring> (tests whose name contains it), ?bisect=0, ?micro=0, ?engine=0.
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const sh = (f) => readFileSync(join(root, 'crates/engine/src/shaders', f), 'utf8').replaceAll('\r\n', '\n');
const out = process.argv[2];
if (!out) { console.error('usage: node tools/draw-probe.mjs <out.html>'); process.exit(2); }

// modules exactly as the engine assembles them, for 1x MSAA (renderer::ms_src)
const ms1 = (s) => s.replaceAll('texture_depth_multisampled_2d', 'texture_depth_2d');
const common = sh('common.wgsl');
const MODS = {
  pbr: ms1(`${common}\n${sh('pbr.wgsl')}`),
  shadow: `${common}\n${sh('shadow.wgsl')}`,
  sky: ms1(`${common}\n${sh('sky.wgsl')}`),
  sky_lut: `${common}\n${sh('sky_lut.wgsl')}`,
  particles: ms1(`${common}\n${sh('particles.wgsl')}`),
  post: ms1(sh('post.wgsl')),
  cull: sh('cull.wgsl'),
  hiz: ms1(sh('hiz.wgsl')).replace('s < 4;', 's < 1;'),
  mipgen: sh('mipgen.wgsl'),
  impostor: sh('impostor.wgsl'),
};
// [module, entry, kind, fragment constants]
const ENGINE = [
  ['pbr', 'vs_main', 'vertex'], ['pbr', 'fs_main', 'fragment', { ALPHA_TEST: 0 }], ['pbr', 'fs_main', 'fragment', { ALPHA_TEST: 1 }],
  ['pbr', 'fs_depth', 'fragment'], ['pbr', 'fs_blend', 'fragment'], ['pbr', 'vs_grass', 'vertex'], ['pbr', 'fs_grass', 'fragment'],
  ['shadow', 'vs_shadow', 'vertex'], ['shadow', 'vs_shadow_mask', 'vertex'], ['shadow', 'fs_shadow_mask', 'fragment'],
  ['sky', 'vs_sky', 'vertex'], ['sky', 'fs_sky', 'fragment'], ['sky_lut', 'vs_lut', 'vertex'], ['sky_lut', 'fs_lut', 'fragment'],
  ['particles', 'vs_particle', 'vertex'], ['particles', 'fs_additive', 'fragment'], ['particles', 'fs_alpha', 'fragment'],
  ['post', 'vs_full', 'vertex'], ['post', 'fs_prefilter', 'fragment'], ['post', 'fs_down', 'fragment'], ['post', 'fs_up', 'fragment'],
  ['post', 'vs_composite', 'vertex'], ['post', 'fs_composite', 'fragment'], ['post', 'fs_composite', 'fragment', { FXAA: 1 }],
  ['cull', 'reset', 'compute'], ['cull', 'cull', 'compute'], ['hiz', 'init', 'compute'], ['hiz', 'down', 'compute'],
  ['mipgen', 'vs', 'vertex'], ['mipgen', 'fs', 'fragment'],
  ['impostor', 'vs_bake', 'vertex'], ['impostor', 'fs_bake', 'fragment'], ['impostor', 'vs_full', 'vertex'], ['impostor', 'fs_dilate', 'fragment'],
];
// micro constructs: [name, module-scope declarations, body (sets r from runtime k = 1), where]
const MICRO = [
  ['plain', '', 'r = probe_u.y;'],
  ['let_array', '', 'let a = array<f32, 4>(1.0, 2.0, 3.0, 4.0); r = a[k];'],
  ['const_array', 'const CA = array<f32, 4>(1.0, 2.0, 3.0, 4.0);', 'r = CA[k];'],
  ['var_array_write', '', 'var a = array<f32, 4>(); a[k] = 2.0; r = a[1];'],
  ['private_array', 'var<private> pa: array<f32, 4>;', 'pa[k] = 2.0; r = pa[1];'],
  ['vec_index', '', 'let v = vec4f(1.0, 2.0, 3.0, 4.0); r = v[k];'],
  ['vec_write', '', 'var v = vec4f(); v[k] = 2.0; r = v.y;'],
  ['mat_column', '', 'let m = mat4x4f(vec4f(1.0), vec4f(2.0), vec4f(3.0), vec4f(4.0)); r = m[k].x;'],
  ['uniform_vec_comp', '', 'r = probe_u[k];'],
  ['uniform_arr_vec4', '@group(0) @binding(20) var<uniform> u_arr: array<vec4f, 4>;', 'r = u_arr[k].x;'],
  ['uniform_arr_mat4', '@group(0) @binding(21) var<uniform> u_mats: array<mat4x4f, 4>;', 'r = u_mats[k][0].x;'],
  ['uniform_mat4_mul', '@group(0) @binding(21) var<uniform> u_mats: array<mat4x4f, 4>;', 'r = (u_mats[k] * vec4f(1.0)).x;'],
  ['uniform_struct_arr', 'struct US { a: vec4f, m: array<mat4x4f, 4>, s: vec4f, }\n@group(0) @binding(22) var<uniform> u_st: US;', 'r = u_st.m[k][1].y + u_st.s.x;'],
  ['storage_arr', '@group(0) @binding(23) var<storage, read> s_arr: array<vec4f>;', 'r = s_arr[k].x;'],
  ['storage_struct', 'struct SS { m: mat4x4f, v: vec4f, }\n@group(0) @binding(24) var<storage, read> s_st: array<SS>;', 'r = (s_st[k].m * s_st[k].v).x;'],
  ['storage_vec_comp', '@group(0) @binding(23) var<storage, read> s_arr: array<vec4f>;', 'r = s_arr[0][k];'],
  ['loop_break', '', 'var i = 0u; loop { if (i >= k + 2u || f32(i) > probe_u.w + 8.0) { break; } r += 1.0; i++; }'],
  ['for_continue', '', 'for (var i = 0u; i < 4u; i++) { if (i == k) { continue; } r += f32(i); }'],
  ['switch_u32', '', 'switch k { case 0u: { r = 1.0; } case 1u: { r = 2.0; } default: { r = 3.0; } }'],
  ['tex_load', '@group(0) @binding(25) var t2: texture_2d<f32>;', 'r = textureLoad(t2, vec2i(i32(k), 0), 0).x;'],
  ['tex_dims', '@group(0) @binding(25) var t2: texture_2d<f32>;', 'r = f32(textureDimensions(t2).x + k);'],
  ['tex_sample_level', '@group(0) @binding(25) var t2: texture_2d<f32>;\n@group(0) @binding(29) var smp: sampler;', 'r = textureSampleLevel(t2, smp, vec2f(probe_u.y * 0.3), 0.0).x;'],
  ['tex_sample_grad', '@group(0) @binding(25) var t2: texture_2d<f32>;\n@group(0) @binding(29) var smp: sampler;', 'r = textureSampleGrad(t2, smp, vec2f(probe_u.y * 0.3), vec2f(0.01), vec2f(0.01)).x;'],
  ['depth_load', '@group(0) @binding(26) var td: texture_depth_2d;', 'r = textureLoad(td, vec2i(i32(k), 0), 0);'],
  ['shadow_compare', '@group(0) @binding(27) var tda: texture_depth_2d_array;\n@group(0) @binding(28) var scmp: sampler_comparison;', 'r = textureSampleCompareLevel(tda, scmp, vec2f(0.5), i32(k), 0.5);'],
  ['bitcast', '', 'r = bitcast<f32>(k + 0x3f800000u);'],
  ['atomic_min_rt', '@group(0) @binding(19) var<storage, read_write> s_at: array<atomic<u32>>;', 'atomicMin(&s_at[k], 5u); r = f32(atomicLoad(&s_at[0]));'],
  ['ptr_param', 'fn pf(p: ptr<function, vec4f>) { (*p).y = 2.0; }', 'var v = vec4f(); pf(&v); r = v.y;'],
  ['struct_return', 'struct SR { a: vec3f, b: f32, }\nfn sr(x: f32) -> SR { var o: SR; o.a = vec3f(x); o.b = x * 2.0; return o; }', 'let s = sr(probe_u.y); r = s.a.x + s.b;'],
  ['select_vec_bool', '', 'r = select(vec4f(1.0), vec4f(2.0), vec4<bool>(k == 0u, k == 1u, k == 2u, k == 3u)).y;'],
  ['pack4x8', '', 'r = f32(pack4x8unorm(vec4f(probe_u.y * 0.1)));'],
  ['override_bool', 'override OB: bool = true;', 'r = select(1.0, 2.0, OB);'],
  ['matrix_inverse_ops', '', 'let m = mat3x3f(vec3f(probe_u.y), vec3f(1.0), vec3f(2.0)); r = (transpose(m) * vec3f(1.0)).z;'],
  ['dpdx', '', 'r = dpdx(probe_u.y * 2.0) + fwidth(probe_u.z);', 'fragment'],
  ['discard', '', 'if (probe_u.y > 5.0) { discard; } r = 1.0;', 'fragment'],
  ['workgroup_arr', 'var<workgroup> wg: array<u32, 64>;', 'wg[k] = 3u; workgroupBarrier(); r = f32(wg[1]);', 'compute'],
];

const page = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Draw probe</title><body style="font:13px monospace;background:#111;color:#ddd;padding:10px">
<h3 style="margin:4px 0">wasmOnly draw probe</h3><pre id="o" style="white-space:pre-wrap"></pre><script type="module">
const MODS = ${JSON.stringify(MODS)};
const ENGINE = ${JSON.stringify(ENGINE)};
const MICRO = ${JSON.stringify(MICRO)};
const q = new URLSearchParams(location.search);
const ONLY = q.get('only');
const o = document.getElementById('o');
const line = (s) => { o.textContent += s + '\\n'; };
const post = (data) => {
  const body = JSON.stringify({ t: Math.round(performance.now()), kind: 'drawprobe', data });
  for (const url of ['/__report', '/__log']) { try { fetch(url, { method: 'POST', body }).catch(() => {}); } catch {} }
};

// ---- WGSL parsing (just enough for the engine's shaders) ----
const strip = (s) => s.replace(/\\/\\/[^\\n]*/g, (m) => ' '.repeat(m.length)); // comments out, offsets kept
function splitTop(s, sep) {
  const out = []; let d = 0, cur = '';
  for (const ch of s) {
    if ('(<['.includes(ch)) d++; else if (')>]'.includes(ch)) d--;
    if (ch === sep && d === 0) { out.push(cur); cur = ''; } else cur += ch;
  }
  if (cur.trim()) out.push(cur);
  return out.map((x) => x.trim()).filter(Boolean);
}
function parseParam(p) {
  const attrs = []; let s = p.trim(), m;
  while ((m = s.match(/^@\\w+(\\s*\\([^)]*\\))?\\s*/))) { attrs.push(m[0].trim()); s = s.slice(m[0].length); }
  const i = s.indexOf(':');
  return i < 0 ? { attrs, name: '', type: '' } : { attrs, name: s.slice(0, i).trim(), type: s.slice(i + 1).trim() };
}
const stripAttrs = (t) => t.replace(/@\\w+(\\s*\\([^)]*\\))?/g, '').trim();
function findFn(src, name) {
  const c = strip(src);
  const re = new RegExp('((?:@\\\\w+(?:\\\\s*\\\\([^)]*\\\\))?\\\\s*)*)fn\\\\s+' + name + '\\\\s*\\\\(', 'g');
  const m = re.exec(c);
  if (!m) return null;
  let i = m.index + m[0].length, d = 1; const ps = i;
  while (d > 0) { if (c[i] === '(') d++; else if (c[i] === ')') d--; i++; }
  const open = c.indexOf('{', i);
  let ret = src.slice(i, open).trim(); ret = ret.startsWith('->') ? ret.slice(2).trim() : '';
  let j = open, dd = 0;
  for (; j < c.length; j++) { if (c[j] === '{') dd++; else if (c[j] === '}' && --dd === 0) break; }
  return { attrStart: m.index, params: src.slice(ps, i - 1), ret, open, close: j };
}
function structFields(src, T) {
  if (!/^\\w+$/.test(T)) return null;
  const c = strip(src);
  const m = new RegExp('struct\\\\s+' + T + '\\\\s*\\\\{').exec(c);
  if (!m) return null;
  const end = c.indexOf('}', m.index);
  return splitTop(c.slice(m.index + m[0].length, end), ',').map(parseParam).filter((f) => f.name);
}
const isPos = (f) => f.attrs.some((a) => /builtin\\(\\s*position/.test(a));
const isVi = (f) => f.attrs.some((a) => /builtin\\(\\s*vertex_index/.test(a));
function fnNames(src) { return [...strip(src).matchAll(/\\bfn\\s+(\\w+)\\s*\\(/g)].map((m) => m[1]); }
function reachable(src, entry) {
  const names = new Set(fnNames(src)); const seen = new Set(); const todo = [entry];
  while (todo.length) {
    const f = findFn(src, todo.pop()); if (!f) continue;
    const body = strip(src).slice(f.open, f.close);
    for (const n of names) if (!seen.has(n) && n !== entry && new RegExp('\\\\b' + n + '\\\\s*\\\\(').test(body)) { seen.add(n); todo.push(n); }
  }
  return [...seen];
}
function stubFn(src, name) {
  const f = findFn(src, name); if (!f) return src;
  const rt = stripAttrs(f.ret);
  return src.slice(0, f.open) + (rt ? '{ return ' + rt + '(); }' : '{ }') + src.slice(f.close + 1);
}
const PROBE = \`
@group(0) @binding(30) var<uniform> probe_u: vec4f;
@group(0) @binding(31) var<storage, read_write> probe_out: array<atomic<u32>, 4>;
fn probe_full(i: u32) -> vec4f { let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(p * 2.0 - 1.0, 0.5, 1.0); }
@fragment fn probe_fs() -> @location(0) vec4f { atomicAdd(&probe_out[1], 1u); return vec4f(0.25, 0.5, 0.75, 1.0); }
\`;
// vertex: real vs runs as <entry>__orig, the new entry overrides its position (unless probe_u.x >= 0.5)
function wrapVertex(src, entry) {
  const f = findFn(src, entry);
  const ps = splitTop(f.params, ',').map(parseParam);
  let vi = ps.find(isVi)?.name;
  if (!vi) for (const p of ps) { const q = (structFields(src, p.type) || []).find(isVi); if (q) { vi = p.name + '.' + q.name; break; } }
  const extra = vi ? '' : (ps.length ? ', ' : '') + '@builtin(vertex_index) probe_vi: u32';
  vi = vi || 'probe_vi';
  const rt = stripAttrs(f.ret);
  const pos = /builtin\\(\\s*position/.test(f.ret) ? 'o' : 'o.' + structFields(src, rt).find(isPos).name;
  const orig = 'fn ' + entry + '__orig(' + ps.map((p) => p.name + ': ' + p.type).join(', ') + ') -> ' + rt + ' ';
  return src.slice(0, f.attrStart) + orig + src.slice(f.open) + PROBE +
    '\\n@vertex fn ' + entry + '(' + f.params + extra + ') -> ' + f.ret + ' {\\n  var o = ' + entry + '__orig(' + ps.map((p) => p.name).join(', ') +
    ');\\n  if (probe_u.x < 0.5) { ' + pos + ' = probe_full(' + vi + '); }\\n  return o;\\n}\\n';
}
// fragment: probe atomic first, a full-screen probe_vs that outputs the fs input struct
function wrapFragment(src, entry) {
  const f = findFn(src, entry);
  const ps = splitTop(f.params, ',').map(parseParam);
  const sp = ps.find((p) => structFields(src, p.type));
  let vs;
  if (sp) {
    const q = structFields(src, sp.type).find(isPos);
    vs = '@vertex fn probe_vs(@builtin(vertex_index) i: u32) -> ' + sp.type + ' { var o: ' + sp.type + '; o.' + q.name + ' = probe_full(i); return o; }';
  } else {
    vs = '@vertex fn probe_vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return probe_full(i); }';
  }
  return src.slice(0, f.open + 1) + ' atomicAdd(&probe_out[0], 1u); ' + src.slice(f.open + 1) + PROBE + '\\n' + vs + '\\n';
}
function wrapCompute(src, entry) {
  const f = findFn(src, entry);
  return src.slice(0, f.open + 1) + ' atomicAdd(&probe_out[0], 1u); ' + src.slice(f.open + 1) + PROBE;
}
// fragment outputs -> colour target formats
function fragTargets(src, entry) {
  const f = findFn(src, entry);
  if (!f.ret) return [];
  const fmt = (t) => /u32|vec\\du/.test(t) ? 'rgba32uint' : /i32|vec\\di/.test(t) ? 'rgba32sint' : 'rgba16float';
  const loc = (a) => +((a.join(' ').match(/location\\(\\s*(\\d+)/) || [])[1] ?? -1);
  if (/@location/.test(f.ret)) { const n = loc([f.ret]); return Array.from({ length: n + 1 }, (_, i) => (i === n ? { format: fmt(f.ret) } : null)); }
  const fs = structFields(src, stripAttrs(f.ret)) || [];
  const t = []; for (const x of fs) { const n = loc(x.attrs); if (n >= 0) t[n] = { format: fmt(x.type) }; }
  return Array.from({ length: t.length }, (_, i) => t[i] ?? null);
}
// vertex inputs -> vertex buffers (one per location, zero data)
const VF = { f32: ['float32', 4], vec2f: ['float32x2', 8], vec3f: ['float32x3', 12], vec4f: ['float32x4', 16], u32: ['uint32', 4],
  vec2u: ['uint32x2', 8], vec3u: ['uint32x3', 12], vec4u: ['uint32x4', 16], i32: ['sint32', 4], vec2i: ['sint32x2', 8], vec4i: ['sint32x4', 16] };
const normT = (t) => t.replace(/vec(\\d)<\\s*(f32|u32|i32)\\s*>/, (_, n, s) => 'vec' + n + s[0]);
function vertexInputs(src, entry) {
  const f = findFn(src, entry); const locs = [];
  for (const p of splitTop(f.params, ',').map(parseParam)) {
    const fs = structFields(src, p.type);
    for (const x of fs || [p]) { const m = x.attrs.join(' ').match(/location\\(\\s*(\\d+)/); if (m) locs.push([+m[1], normT(x.type)]); }
  }
  return locs.map(([loc, t]) => ({ arrayStride: (VF[t] || VF.vec4f)[1], attributes: [{ format: (VF[t] || VF.vec4f)[0], offset: 0, shaderLocation: loc }] }));
}
// @group/@binding declarations -> explicit bind group layouts + zero resources
function bindings(src) {
  return [...strip(src).matchAll(/@group\\(\\s*(\\d+)\\s*\\)\\s*@binding\\(\\s*(\\d+)\\s*\\)\\s*var(?:<([^>]*)>)?\\s+(\\w+)\\s*:\\s*([^;]+);/g)]
    .map((m) => ({ group: +m[1], binding: +m[2], space: (m[3] || '').replace(/\\s/g, ''), name: m[4], type: m[5].trim() }));
}

// ---- device ----
const a = await navigator.gpu?.requestAdapter();
if (!a) { line('no WebGPU adapter'); post({ error: 'no adapter' }); throw 0; }
const d = await a.requestDevice();
const info = a.info ? [a.info.vendor, a.info.architecture, a.info.device, a.info.description].join(' | ') : '?';
line('adapter: ' + info); line('ua: ' + navigator.userAgent);
post({ adapter: info, ua: navigator.userAgent });
let lost = null;
d.lost.then((l) => { lost = l.message || l.reason; line('DEVICE LOST: ' + lost); post({ lost }); });
d.onuncapturederror = (e) => post({ uncaptured: e.error.message });
const U_SIZE = Math.min(65536, d.limits.maxUniformBufferBindingSize);
const GS = GPUShaderStage, BU = GPUBufferUsage, TU = GPUTextureUsage;
const dimOf = (s) => ({ '1d': '1d', '2d': '2d', '2d_array': '2d-array', '3d': '3d', cube: 'cube', cube_array: 'cube-array' })[s];
function texture(dim, format, samples = 1, extraUsage = 0) {
  const size = { '1d': [4], '2d': [4, 4, 1], '2d-array': [4, 4, 4], cube: [4, 4, 6], 'cube-array': [4, 4, 12], '3d': [4, 4, 4] }[dim];
  const usage = TU.TEXTURE_BINDING | TU.COPY_DST | extraUsage | (samples > 1 ? TU.RENDER_ATTACHMENT : 0);
  const t = d.createTexture({ size, format, usage, sampleCount: samples, dimension: dim === '1d' ? '1d' : dim === '3d' ? '3d' : '2d' });
  return t.createView({ dimension: dim });
}
function resourceFor(b, compute, probeU, probeOut) {
  const visAll = compute ? GS.COMPUTE : GS.VERTEX | GS.FRAGMENT;
  const visW = compute ? GS.COMPUTE : GS.FRAGMENT;
  const e = { binding: b.binding };
  let r, t = b.type, m;
  if (b.space.startsWith('uniform')) {
    e.visibility = visAll; e.buffer = { type: 'uniform' };
    r = { buffer: b.name === 'probe_u' ? probeU : d.createBuffer({ size: U_SIZE, usage: BU.UNIFORM }) };
  } else if (b.space.startsWith('storage')) {
    const rw = b.space.includes('read_write');
    e.visibility = rw ? visW : visAll; e.buffer = { type: rw ? 'storage' : 'read-only-storage' };
    r = { buffer: b.name === 'probe_out' ? probeOut : d.createBuffer({ size: 1 << 20, usage: BU.STORAGE }) };
  } else if (/^sampler_comparison/.test(t)) {
    e.visibility = visAll; e.sampler = { type: 'comparison' }; r = d.createSampler({ compare: 'less' });
  } else if (/^sampler/.test(t)) {
    e.visibility = visAll; e.sampler = { type: 'filtering' }; r = d.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  } else if ((m = t.match(/^texture_storage_(\\w+?)\\s*<\\s*(\\w+)\\s*,\\s*(\\w+)\\s*>/))) {
    const dim = dimOf(m[1]); const access = { write: 'write-only', read: 'read-only', read_write: 'read-write' }[m[3]];
    e.visibility = access === 'read-only' ? visAll : visW; e.storageTexture = { format: m[2], access, viewDimension: dim };
    r = texture(dim, m[2], 1, TU.STORAGE_BINDING);
  } else if ((m = t.match(/^texture_depth_multisampled_2d/))) {
    e.visibility = visAll; e.texture = { sampleType: 'depth', multisampled: true }; r = texture('2d', 'depth32float', 4);
  } else if ((m = t.match(/^texture_depth_(2d_array|2d|cube_array|cube)/))) {
    const dim = dimOf(m[1]); e.visibility = visAll; e.texture = { sampleType: 'depth', viewDimension: dim }; r = texture(dim, 'depth32float');
  } else if ((m = t.match(/^texture_multisampled_2d\\s*<\\s*(\\w+)/))) {
    e.visibility = visAll; e.texture = { sampleType: 'unfilterable-float', multisampled: true }; r = texture('2d', 'rgba16float', 4);
  } else if ((m = t.match(/^texture_(1d|2d_array|2d|3d|cube_array|cube)\\s*<\\s*(\\w+)/))) {
    const dim = dimOf(m[1]); const st = { f32: 'float', u32: 'uint', i32: 'sint' }[m[2]];
    e.visibility = visAll; e.texture = { sampleType: st, viewDimension: dim };
    r = texture(dim, { float: 'rgba16float', uint: 'rgba32uint', sint: 'rgba32sint' }[st]);
  } else throw new Error('binding type not handled: ' + t);
  return [e, r];
}
// text of the given entry points and every function they reach (a module may declare one binding
// number twice for different entry points, e.g. impostor.wgsl)
function usedText(code, entries) {
  const c = strip(code); let t = '';
  for (const e of entries) for (const n of [e, ...reachable(code, e)]) { const f = findFn(code, n); if (f) t += c.slice(f.attrStart, f.close + 1) + '\\n'; }
  return t;
}
function setup(code, compute, entries) {
  const probeU = d.createBuffer({ size: 16, usage: BU.UNIFORM | BU.COPY_DST });
  d.queue.writeBuffer(probeU, 0, new Float32Array([0, 1, 2, 3]));
  const probeOut = d.createBuffer({ size: 16, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST });
  const used = usedText(code, entries);
  const seen = new Set();
  const bs = bindings(code).filter((b) => {
    if (!new RegExp('\\\\b' + b.name + '\\\\b').test(used) && !b.name.startsWith('probe_')) return false;
    const k = b.group + ':' + b.binding; if (seen.has(k)) return false; seen.add(k); return true;
  });
  const ng = bs.reduce((n, b) => Math.max(n, b.group + 1), 0);
  const layouts = [], groups = [];
  for (let g = 0; g < ng; g++) {
    const ents = bs.filter((b) => b.group === g).map((b) => resourceFor(b, compute, probeU, probeOut));
    const bgl = d.createBindGroupLayout({ entries: ents.map((x) => x[0]) });
    layouts.push(bgl);
    groups.push(d.createBindGroup({ layout: bgl, entries: ents.map((x, i) => ({ binding: x[0].binding, resource: x[1] })) }));
  }
  return { layout: d.createPipelineLayout({ bindGroupLayouts: layouts }), groups, probeOut };
}
const pick = (code, consts) => Object.fromEntries(Object.entries(consts || {}).filter(([k]) => new RegExp('override\\\\s+' + k + '\\\\b').test(code)));
const timeout = (p, ms) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout ' + ms + ' ms')), ms))]);

// one test: kind vertex | fragment | compute; returns 'ran' | 'DROPPED' | 'ERROR ...'
async function runTest(kind, src, entry, consts) {
  if (lost) return 'ERROR device lost';
  let code;
  try { code = kind === 'vertex' ? wrapVertex(src, entry) : kind === 'fragment' ? wrapFragment(src, entry) : wrapCompute(src, entry); }
  catch (e) { return 'ERROR probe wrap: ' + e.message; }
  d.pushErrorScope('out-of-memory'); d.pushErrorScope('internal'); d.pushErrorScope('validation');
  let res = null, rb, rbc;
  try {
    const module = d.createShaderModule({ code });
    const ci = await module.getCompilationInfo();
    const errs = ci.messages.filter((m) => m.type === 'error');
    if (errs.length) throw new Error('WGSL ' + errs.slice(0, 2).map((m) => m.lineNum + ': ' + m.message).join('; '));
    const compute = kind === 'compute';
    const entries = kind === 'compute' ? [entry] : kind === 'vertex' ? [entry, 'probe_fs'] : ['probe_vs', entry];
    const { layout, groups, probeOut } = setup(code, compute, entries);
    const enc = d.createCommandEncoder();
    let color = null;
    if (compute) {
      const p = await d.createComputePipelineAsync({ layout, compute: { module, entryPoint: entry } });
      const pass = enc.beginComputePass(); pass.setPipeline(p); groups.forEach((g, i) => pass.setBindGroup(i, g));
      pass.dispatchWorkgroups(1); pass.end();
    } else {
      const vertex = kind === 'vertex' ? { module, entryPoint: entry, buffers: vertexInputs(code, entry) } : { module, entryPoint: 'probe_vs' };
      const fsEntry = kind === 'vertex' ? 'probe_fs' : entry;
      const targets = kind === 'vertex' ? [{ format: 'rgba8unorm' }] : fragTargets(code, entry);
      const p = await d.createRenderPipelineAsync({
        layout, vertex, primitive: { topology: 'triangle-list', cullMode: 'none' },
        fragment: { module, entryPoint: fsEntry, targets, constants: kind === 'fragment' ? pick(code, consts) : {} },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
      });
      const views = targets.map((t) => t && d.createTexture({ size: [4, 4], format: t.format, usage: TU.RENDER_ATTACHMENT | TU.COPY_SRC }));
      const depth = d.createTexture({ size: [4, 4], format: 'depth24plus', usage: TU.RENDER_ATTACHMENT });
      const pass = enc.beginRenderPass({
        colorAttachments: views.map((t) => t && { view: t.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }),
        depthStencilAttachment: { view: depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 1 },
      });
      pass.setPipeline(p); groups.forEach((g, i) => pass.setBindGroup(i, g));
      if (kind === 'vertex') vertex.buffers.forEach((b, i) => pass.setVertexBuffer(i, d.createBuffer({ size: b.arrayStride * 16, usage: BU.VERTEX })));
      pass.draw(3, 1); pass.end();
      if (kind === 'vertex') {
        rbc = d.createBuffer({ size: 256 * 4, usage: BU.MAP_READ | BU.COPY_DST });
        enc.copyTextureToBuffer({ texture: views[0] }, { buffer: rbc, bytesPerRow: 256 }, [4, 4]);
      }
    }
    rb = d.createBuffer({ size: 16, usage: BU.MAP_READ | BU.COPY_DST });
    enc.copyBufferToBuffer(probeOut, 0, rb, 0, 16);
    d.queue.submit([enc.finish()]);
  } catch (e) { res = 'ERROR ' + String(e.message || e).split('\\n')[0]; }
  const ve = await d.popErrorScope(), ie = await d.popErrorScope(), oe = await d.popErrorScope();
  const se = ve || ie || oe;
  if (se && !res) res = 'ERROR ' + se.message.split('\\n')[0];
  if (res) return res;
  try {
    await timeout(rb.mapAsync(GPUMapMode.READ), 5000);
    const c = new Uint32Array(rb.getMappedRange().slice(0)); rb.unmap();
    let px = null;
    if (rbc) { await timeout(rbc.mapAsync(GPUMapMode.READ), 5000); px = [...new Uint8Array(rbc.getMappedRange().slice(0, 4))]; rbc.unmap(); }
    const ran = kind === 'vertex' ? c[1] > 0 || (px && px[0] > 40 && px[0] < 90 && px[2] > 160) : c[0] > 0;
    return ran ? 'ran' : 'DROPPED' + (px ? ' (px ' + px.join(',') + ')' : '');
  } catch (e) { return 'ERROR readback ' + e.message; }
}

const results = [];
async function test(group, name, kind, src, entry, consts) {
  if (ONLY && !name.includes(ONLY)) return null;
  const t0 = performance.now();
  const res = await runTest(kind, src, entry, consts);
  const r = { group, name, kind, res, ms: Math.round(performance.now() - t0) };
  results.push(r);
  line((res === 'ran' ? '  ' : res.startsWith('DROPPED') ? '!!' : '??') + ' ' + group.padEnd(7) + name.padEnd(42) + res + '  ' + r.ms + 'ms');
  post(r);
  return res;
}

line('\\n-- 1. calibration (must all say "ran") --');
const CAL = '@vertex fn t_vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(p * 2.0 - 1.0, 0.5, 1.0); }\\n' +
  '@fragment fn t_fs() -> @location(0) vec4f { return vec4f(1.0); }\\n@compute @workgroup_size(1) fn t_cs() { }\\n';
await test('cal', 'vertex', 'vertex', CAL, 't_vs');
await test('cal', 'fragment', 'fragment', CAL, 't_fs');
await test('cal', 'compute', 'compute', CAL, 't_cs');

if (q.get('micro') !== '0') {
  line('\\n-- 2. micro constructs (k = 1 at runtime) --');
  for (const [name, decl, body, where] of MICRO) {
    if (where !== 'compute') {
      const fsrc = decl + '\\n@fragment fn t_fs() -> @location(0) vec4f { let k = u32(probe_u.y); var r = 0.0; ' + body + ' return vec4f(r); }\\n';
      await test('micro', 'fs:' + name, 'fragment', fsrc, 't_fs');
    }
    if (where !== 'fragment') {
      const csrc = decl + '\\n@compute @workgroup_size(1) fn t_cs() { let k = u32(probe_u.y); var r = 0.0; ' + body + ' if (r == 12345.0) { atomicAdd(&probe_out[3], 1u); } }\\n';
      await test('micro', 'cs:' + name, 'compute', csrc, 't_cs');
    }
  }
}

const dropped = [];
// ?force_bisect=pbr:fs_main runs the bisection for that entry even if it ran (checks that every stub compiles)
for (const fb of (q.get('force_bisect') || '').split(',').filter(Boolean)) {
  const [mod, entry] = fb.split(':'); const e = ENGINE.find((x) => x[0] === mod && x[1] === entry);
  if (e) dropped.push([mod, entry, e[2], e[3], fb + ' (forced)']);
}
if (q.get('engine') !== '0') {
  line('\\n-- 3. engine entry points (real shaders, zero data) --');
  for (const [mod, entry, kind, consts] of ENGINE) {
    const name = mod + ':' + entry + (consts ? ' ' + JSON.stringify(consts) : '');
    const res = await test('engine', name, kind, MODS[mod], entry, consts);
    if (res && res.startsWith('DROPPED')) dropped.push([mod, entry, kind, consts, name]);
  }
}

if (q.get('bisect') !== '0' && dropped.length) {
  line('\\n-- 4. bisection: stub one function at a time (ran = that function holds the construct) --');
  for (const [mod, entry, kind, consts, name] of dropped) {
    const src = MODS[mod];
    const fns = reachable(src, entry);
    line(name + ': ' + fns.length + ' functions reachable');
    const all = fns.reduce((s, f) => stubFn(s, f), src);
    const ra = await test('bisect', name + ' ALL stubbed', kind, all, entry, consts);
    if (ra !== 'ran' && !name.endsWith('(forced)')) { line('   -> still not running with every callee stubbed: the construct is in ' + entry + ' itself'); continue; }
    const hits = [];
    for (const f of fns) {
      const r = await test('bisect', name + ' stub ' + f, kind, stubFn(src, f), entry, consts);
      if (r === 'ran') hits.push(f);
    }
    line('   -> runs when stubbing: ' + (hits.join(', ') || '(no single function: several places)'));
    // several places: keep one function at a time with every other callee stubbed
    const alone = [];
    if (!hits.length) {
      for (const f of fns) {
        const r = await test('bisect', name + ' only ' + f, kind, fns.filter((x) => x !== f).reduce((s, x) => stubFn(s, x), src), entry, consts);
        if (r.startsWith('DROPPED')) alone.push(f);
      }
      line('   -> dropped with only this function left: ' + (alone.join(', ') || '(none alone: a combination)'));
    }
    post({ bisect: name, hits, alone });
  }
}
const sum = results.filter((r) => r.res !== 'ran').map((r) => r.group + ':' + r.name + '=' + r.res.split(' ')[0]);
line('\\ndone. not ran: ' + (sum.length ? '\\n' + sum.join('\\n') : 'none'));
post({ done: true, not_ran: sum });
</script>`;
writeFileSync(out, page);
console.log(`wrote ${out} (${ENGINE.length} engine entries, ${MICRO.length} micro constructs)`);

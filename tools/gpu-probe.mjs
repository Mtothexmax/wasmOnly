#!/usr/bin/env node
// Shader probe for devices whose driver rejects an engine pipeline (e.g. VK_ERROR_UNKNOWN on PowerVR):
// writes one self-contained HTML page that compiles the PBR / grass pipelines in many reduced variants
// (parts of the shader stubbed out) and reports which compile. One page load on the device shows which
// construct the driver chokes on.
//
//   node tools/gpu-probe.mjs <out.html>
//
// Open the page on the device. Results are shown on the page and POSTed as JSON lines to /__log
// (if the server has that endpoint, e.g. a LAN test server), kind "probe".
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const sh = (f) => readFileSync(join(root, 'crates/engine/src/shaders', f), 'utf8');
const src = `${sh('common.wgsl')}\n${sh('pbr.wgsl')}`;
const out = process.argv[2];
if (!out) { console.error('usage: node tools/gpu-probe.mjs <out.html>'); process.exit(2); }

// Replace the body of `fn name(...) ... { ... }` (brace-balanced).
function stub(s, name, body) {
  const i = s.indexOf(`fn ${name}(`);
  if (i < 0) throw new Error(`fn ${name} not found`);
  const open = s.indexOf('{', i);
  let depth = 0, j = open;
  for (; j < s.length; j++) {
    if (s[j] === '{') depth++;
    else if (s[j] === '}' && --depth === 0) break;
  }
  return s.slice(0, open) + `{ ${body} }` + s.slice(j + 1);
}
const T = {
  no_override: (s) => s.replace('override ALPHA_TEST: bool = true;', 'const ALPHA_TEST: bool = false;'),
  no_shadow: (s) => stub(s, 'shadow_factor', 'return 1.0;'),
  no_cascade: (s) => stub(s, 'cascade_shadow', 'return 1.0;'),
  one_cascade: (s) => stub(s, 'shadow_set_factor', 'return cascade_shadow(world, n, 0u, prev);'),
  no_lights: (s) => stub(s, 'cluster_lights', 'return vec3f(0.0);'),
  no_fog: (s) => stub(s, 'finish', 'return sanitize(c);'),
  no_terrain: (s) => stub(s, 'terrain_albedo', 'return vec3f(1.0);'),
  no_parallax: (s) => stub(s, 'parallax_uv', 'return uv;'),
  no_debug: (s) => stub(stub(s, 'debug_extra', 'return vec4f(0.0);'), 'debug_out', 'return s.color;'),
  trivial_fs: (s) => stub(s, 'fs_main', 'return vec4f(i.world, 1.0);'),
  trivial_vs: (s) => stub(s, 'vs_main', 'var o: VsOut; o.clip = vec4f(v.pos, 1.0); o.world = v.pos; o.normal = v.normal; o.uv = v.uv; o.inst = v.inst; return o;'),
  no_gerstner: (s) => stub(s, 'gerstner', 'var w: Wave; w.disp = vec3f(0.0); w.n = vec3f(0.0, 1.0, 0.0); w.crest = 0.0; return w;'),
  grass_trivial_fs: (s) => stub(s, 'fs_grass', 'return vec4f(i.world, 1.0);'),
};
const apply = (names) => names.reduce((s, n) => T[n](s), src);
const variants = [
  ['main', 'full', []],
  ['main', 'no_override', ['no_override']],
  ['main', 'trivial_fs', ['trivial_fs']],
  ['main', 'trivial_fs+trivial_vs', ['trivial_fs', 'trivial_vs']],
  ['main', 'no_shadow', ['no_shadow']],
  ['main', 'no_cascade', ['no_cascade']],
  ['main', 'one_cascade', ['one_cascade']],
  ['main', 'no_lights', ['no_lights']],
  ['main', 'no_fog', ['no_fog']],
  ['main', 'no_terrain', ['no_terrain']],
  ['main', 'no_parallax', ['no_parallax']],
  ['main', 'no_debug', ['no_debug']],
  ['main', 'no_gerstner', ['no_gerstner']],
  ['main', 'no_shadow+no_lights', ['no_shadow', 'no_lights']],
  ['main', 'no_shadow+no_lights+no_fog', ['no_shadow', 'no_lights', 'no_fog']],
  ['main', 'all_stubs', ['no_override', 'no_shadow', 'no_lights', 'no_fog', 'no_terrain', 'no_parallax', 'no_debug', 'no_gerstner']],
  ['grass', 'full', []],
  ['grass', 'trivial_fs', ['grass_trivial_fs']],
  ['grass', 'no_shadow', ['no_shadow']],
  ['grass', 'no_cascade', ['no_cascade']],
  ['grass', 'no_lights', ['no_lights']],
  ['grass', 'no_fog', ['no_fog']],
  ['grass', 'no_shadow+no_lights+no_fog', ['no_shadow', 'no_lights', 'no_fog']],
].map(([kind, name, t]) => ({ kind, name, code: apply(t) }));

const page = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>GPU probe</title><body style="font:14px monospace;background:#111;color:#ddd;padding:12px">
<h3>wasmOnly shader probe</h3><pre id="o"></pre><script type="module">
const V = ${JSON.stringify(variants)};
const o = document.getElementById('o');
const log = (kind, data) => { try { fetch('/__log', { method: 'POST', body: JSON.stringify({ t: Math.round(performance.now()), kind, data }) }); } catch {} };
const line = (s) => { o.textContent += s + '\\n'; };
const a = await navigator.gpu?.requestAdapter();
if (!a) { line('no adapter'); log('probe', 'no adapter'); throw 0; }
const d = await a.requestDevice();
const info = a.info ? a.info.vendor + ' ' + a.info.architecture : '?';
line('adapter: ' + info); log('probe', { adapter: info, ua: navigator.userAgent });
d.onuncapturederror = (e) => log('probe-uncaptured', e.error.message);
const vbuf = { arrayStride: 32, attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 0 }, { format: 'float32x3', offset: 12, shaderLocation: 1 }, { format: 'float32x2', offset: 24, shaderLocation: 2 }] };
const results = [];
for (const v of V) {
  const t0 = performance.now();
  let res;
  try {
    const module = d.createShaderModule({ code: v.code });
    const ci = await module.getCompilationInfo();
    const errs = ci.messages.filter((m) => m.type === 'error');
    if (errs.length) throw new Error('WGSL: ' + errs.map((m) => m.lineNum + ': ' + m.message).join('; '));
    const desc = {
      label: v.kind + ':' + v.name, layout: 'auto',
      vertex: v.kind === 'main' ? { module, entryPoint: 'vs_main', buffers: [vbuf] } : { module, entryPoint: 'vs_grass' },
      fragment: { module, entryPoint: v.kind === 'main' ? 'fs_main' : 'fs_grass', targets: [{ format: 'rgba16float' }],
                  constants: v.kind === 'main' && v.code.includes('override ALPHA_TEST') ? { ALPHA_TEST: 0 } : {} },
      primitive: { topology: 'triangle-list', cullMode: v.kind === 'main' ? 'back' : 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: 4 },
    };
    d.pushErrorScope('internal'); d.pushErrorScope('validation');
    await d.createRenderPipelineAsync(desc);
    const ve = await d.popErrorScope(); const ie = await d.popErrorScope();
    res = ve || ie ? 'FAIL ' + (ve || ie).message.split('\\n')[0] : 'ok';
  } catch (e) { res = 'FAIL ' + String(e.message || e).split('\\n')[0]; }
  const r = { kind: v.kind, name: v.name, res, ms: Math.round(performance.now() - t0) };
  results.push(r); line(r.kind.padEnd(6) + r.name.padEnd(30) + r.res + '  ' + r.ms + 'ms'); log('probe', r);
}
log('probe-done', results.map((r) => r.kind + ':' + r.name + '=' + (r.res === 'ok' ? 'ok' : 'FAIL')).join(' '));
line('done');
</script>`;
writeFileSync(out, page);
console.log(`wrote ${out} (${variants.length} variants)`);

#!/usr/bin/env node
// Shader probe for devices whose driver rejects an engine pipeline (e.g. VK_ERROR_UNKNOWN on PowerVR):
// writes one self-contained HTML page that compiles the PBR / grass pipelines in many reduced variants
// (parts of the shader stubbed out) and reports which compile. One page load on the device shows which
// construct the driver chokes on.
//
//   node tools/gpu-probe.mjs <out.html> [kind:transform+transform ...]   (kind = main | grass)
//
// Open the page on the device. Results are shown on the page and POSTed as JSON lines to /__log
// (if the server has that endpoint, e.g. a LAN test server), kind "probe".
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const sh = (f) => readFileSync(join(root, 'crates/engine/src/shaders', f), 'utf8').replaceAll('\r\n', '\n');
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
  no_cascade: (s) => stub(s, 'cascade_coord', 'return vec4f(0.0);'),
  no_lights: (s) => stub(s, 'cluster_lights', 'return vec3f(0.0);'),
  no_fog: (s) => stub(s, 'finish', 'return sanitize(c);'),
  no_terrain: (s) => stub(s, 'terrain_albedo', 'return vec3f(1.0);'),
  no_parallax: (s) => stub(s, 'parallax_uv', 'return uv;'),
  no_debug: (s) => stub(stub(s, 'debug_extra', 'return vec4f(0.0);'), 'debug_out', 'return s.color;'),
  no_water: (s) => stub(s, 'water', 'return vec4f(0.0);'),
  no_ssr: (s) => stub(s, 'ssr', 'return vec4f(0.0);'),
  trivial_fs: (s) => stub(s, 'fs_main', 'return vec4f(i.world, 1.0);'),
  trivial_vs: (s) => stub(s, 'vs_main', 'var o: VsOut; o.clip = vec4f(v.pos, 1.0); o.world = v.pos; o.normal = v.normal; o.uv = v.uv; o.inst = v.inst; return o;'),
  no_gerstner: (s) => stub(s, 'gerstner', 'var w: Wave; w.disp = vec3f(0.0); w.n = vec3f(0.0, 1.0, 0.0); w.crest = 0.0; return w;'),
  grass_trivial_fs: (s) => stub(s, 'fs_grass', 'return vec4f(i.world, 1.0);'),
  // shadow internals
  one_tap: (s) => s.replace(/    sum \+= w[ab]\.x \* w[ab]\.y \* textureSampleCompareLevel\([^;]*;\n/g, ''),
  no_cmp: (s) => s.replace(/textureSampleCompareLevel\(shadow_map, shadow_sampler, ([^;]*?), layer, z\)/g,
    'select(0.0, 1.0, z <= textureLoad(shadow_map, vec2i(clamp($1, vec2f(0.0), vec2f(0.999)) * size), layer, 0))'),
  cascade_noret: (s) => s.replace('if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || ndc.z > 1.0) { return 1.0; }',
    'let outside = any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || ndc.z > 1.0;').replace('return sum / 9.0;', 'return select(sum / 9.0, 1.0, outside);'),
  // fog internals
  fog_nolut: (s) => s.replace(/c = textureSampleLevel\(sky_lut, sky_sampler,[^;]*;/, 'c = g.sky_zenith.rgb;'),
  fog_cheap: (s) => stub(s, 'fog_color', 'return g.fog_color.rgb;'),
  fog_noret: (s) => stub(s, 'apply_fog', 'let f = fog_amount(world); var c = color; if (f > 0.0) { c = mix(color, fog_color(world), f); } return c;'),
  lights_once: (s) => s.replace('for (var k = 0u; k < cl.y; k++) {', 'for (var k = 0u; k < min(cl.y, 1u); k++) {'),
};
const ALL = ['no_override', 'no_shadow', 'no_lights', 'no_fog', 'no_terrain', 'no_parallax', 'no_debug', 'no_gerstner'];
T.only_shadow = (s) => ALL.filter((n) => n !== 'no_shadow').reduce((a, n) => T[n](a), s);
T.only_lights = (s) => ALL.filter((n) => n !== 'no_lights').reduce((a, n) => T[n](a), s);
T.only_fog = (s) => ALL.filter((n) => n !== 'no_fog').reduce((a, n) => T[n](a), s);
const apply = (names) => names.reduce((s, n) => T[n](s), src);
// Optional variant list on the command line: kind:transform+transform ... (e.g. grass:no_fog+one_tap).
// kind = main (fs_main, opaque) | mask (fs_main, ALPHA_TEST) | blend (fs_blend, water) | grass
const custom = process.argv.slice(3).map((a) => { const [kind, t] = a.split(':'); return [kind, t, t === 'full' ? [] : t.split('+')]; });
const variants = (custom.length ? custom : [
  ['main', 'full', []],
  ['mask', 'full', []],
  ['blend', 'full', []],
  ['grass', 'full', []],
  ['main', 'trivial_fs+trivial_vs', ['trivial_fs', 'trivial_vs']],
  ['main', 'trivial_fs', ['trivial_fs']],
  ['main', 'no_override', ['no_override']],
  ['main', 'no_shadow', ['no_shadow']],
  ['main', 'no_cascade', ['no_cascade']],
  ['main', 'no_lights', ['no_lights']],
  ['main', 'no_fog', ['no_fog']],
  ['main', 'no_terrain', ['no_terrain']],
  ['main', 'no_parallax', ['no_parallax']],
  ['main', 'only_shadow', ['only_shadow']],
  ['main', 'only_lights', ['only_lights']],
  ['main', 'only_fog', ['only_fog']],
  ['blend', 'no_water', ['no_water']],
  ['blend', 'no_ssr', ['no_ssr']],
  ['grass', 'no_shadow', ['no_shadow']],
  ['grass', 'no_lights', ['no_lights']],
  ['grass', 'no_fog', ['no_fog']],
]).map(([kind, name, t]) => ({ kind, name, code: apply(t) }));

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
const vbuf = { arrayStride: 36, attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 0 }, { format: 'float32x3', offset: 12, shaderLocation: 1 }, { format: 'float32x2', offset: 24, shaderLocation: 2 }, { format: 'unorm8x4', offset: 32, shaderLocation: 3 }] };
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
      vertex: v.kind === 'grass' ? { module, entryPoint: 'vs_grass' } : { module, entryPoint: 'vs_main', buffers: [vbuf] },
      fragment: { module, entryPoint: { main: 'fs_main', mask: 'fs_main', blend: 'fs_blend', grass: 'fs_grass' }[v.kind],
                  targets: [v.kind === 'blend' ? { format: 'rgba16float', blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } } : { format: 'rgba16float' }],
                  constants: v.kind !== 'grass' && v.code.includes('override ALPHA_TEST') ? { ALPHA_TEST: v.kind === 'main' ? 0 : 1 } : {} },
      primitive: { topology: 'triangle-list', cullMode: v.kind === 'grass' ? 'none' : 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: v.kind !== 'blend', depthCompare: 'less' },
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

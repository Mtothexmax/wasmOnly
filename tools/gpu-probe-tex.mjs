#!/usr/bin/env node
// Texture probe for devices where textured materials render black (no WebGPU error): writes one page
// that uploads a known colour the way the engine does and reads every stage back.
//   upload:  copyExternalImageToTexture (ImageBitmap from a PNG blob, like glTF images) | writeTexture
//   format:  rgba8unorm-srgb | rgba8unorm
//   then:    mip generation with a render pass per level (as renderer.rs generate_mips)
//   check:   readback of mip levels 0, 1, 3, 6 and a sample through the material sampler
//            (linear + mipmap linear + anisotropy 8, as renderer.rs) with textureSampleGrad/Level
//
//   node tools/gpu-probe-tex.mjs <out.html>
// Results on the page and POSTed to /__log (kind "texprobe"). Expected texel: about (200,150,50,255).
import { writeFileSync } from 'node:fs';

const out = process.argv[2];
if (!out) { console.error('usage: node tools/gpu-probe-tex.mjs <out.html>'); process.exit(2); }

const page = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>GPU texture probe</title><body style="font:13px monospace;background:#111;color:#ddd;padding:12px">
<h3>wasmOnly texture probe</h3><pre id="o"></pre><script type="module">
const o = document.getElementById('o');
const log = (kind, data) => { try { fetch('/__log', { method: 'POST', body: JSON.stringify({ t: Math.round(performance.now()), kind, data }) }); } catch {} };
const line = (s) => { o.textContent += s + '\\n'; };
const a = await navigator.gpu.requestAdapter();
const d = await a.requestDevice();
const info = a.info ? a.info.vendor + ' ' + a.info.architecture : '?';
line('adapter: ' + info); log('texprobe', { adapter: info });
d.onuncapturederror = (e) => { line('uncaptured: ' + e.error.message); log('texprobe-uncaptured', e.error.message); };
const S = 64, MIPS = 7;
// source: a PNG blob decoded by createImageBitmap (the glTF path)
const cv = new OffscreenCanvas(S, S); const cx = cv.getContext('2d');
cx.fillStyle = 'rgb(200,150,50)'; cx.fillRect(0, 0, S, S);
const blob = await cv.convertToBlob({ type: 'image/png' });
const bmp = await createImageBitmap(blob);
const raw = new Uint8Array(S * S * 4); for (let i = 0; i < S * S; i++) raw.set([200, 150, 50, 255], i * 4);

const mipModule = d.createShaderModule({ code: \`
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VsOut; o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0); o.uv = vec2f(p.x, 1.0 - p.y); return o;
}
@fragment fn fs(i: VsOut) -> @location(0) vec4f { return textureSampleLevel(src, samp, i.uv, 0.0); }
@group(0) @binding(2) var<uniform> lod: vec4f;
@fragment fn fs_grad(i: VsOut) -> @location(0) vec4f {
  let g = vec2f(lod.x / 64.0, 0.0);
  return textureSampleGrad(src, samp, i.uv * 3.0, g, g.yx);
}
@fragment fn fs_level(i: VsOut) -> @location(0) vec4f { return textureSampleLevel(src, samp, i.uv, lod.x); }
\` });
const linear = d.createSampler({ magFilter: 'linear', minFilter: 'linear' });
const material = d.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 8, addressModeU: 'repeat', addressModeV: 'repeat' });
const mipPipe = {};
for (const f of ['rgba8unorm-srgb', 'rgba8unorm']) mipPipe[f] = d.createRenderPipeline({ layout: 'auto', vertex: { module: mipModule, entryPoint: 'vs' }, fragment: { module: mipModule, entryPoint: 'fs', targets: [{ format: f }] } });

async function readLevel(tex, level) {
  const w = Math.max(1, S >> level);
  const buf = d.createBuffer({ size: 256 * w, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = d.createCommandEncoder();
  e.copyTextureToBuffer({ texture: tex, mipLevel: level }, { buffer: buf, bytesPerRow: 256 }, [w, w]);
  d.queue.submit([e.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const px = new Uint8Array(buf.getMappedRange()).slice((w >> 1) * 256 + (w >> 1) * 4, (w >> 1) * 256 + (w >> 1) * 4 + 4);
  buf.unmap(); buf.destroy();
  return [...px].join(',');
}
async function sampleThrough(tex, entry, lodv) {
  const target = d.createTexture({ size: [4, 4], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const p = d.createRenderPipeline({ layout: 'auto', vertex: { module: mipModule, entryPoint: 'vs' }, fragment: { module: mipModule, entryPoint: entry, targets: [{ format: 'rgba8unorm' }] } });
  const ub = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  d.queue.writeBuffer(ub, 0, new Float32Array([lodv, 0, 0, 0]));
  const bg = d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: material }, { binding: 2, resource: { buffer: ub } }] });
  const e = d.createCommandEncoder();
  const pass = e.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [1, 0, 1, 1] }] });
  pass.setPipeline(p); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
  const buf = d.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  e.copyTextureToBuffer({ texture: target }, { buffer: buf, bytesPerRow: 256 }, [4, 4]);
  d.queue.submit([e.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const px = new Uint8Array(buf.getMappedRange()).slice(256 * 2 + 8, 256 * 2 + 12);
  buf.unmap();
  return [...px].join(',');
}
const results = [];
for (const format of ['rgba8unorm-srgb', 'rgba8unorm']) for (const upload of ['external', 'writeTexture']) for (const mipgen of [true, false]) {
  const r = { format, upload, mipgen };
  try {
    d.pushErrorScope('validation'); d.pushErrorScope('internal');
    const tex = d.createTexture({ size: [S, S], format, mipLevelCount: MIPS, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    if (upload === 'external') d.queue.copyExternalImageToTexture({ source: bmp }, { texture: tex }, [S, S]);
    else d.queue.writeTexture({ texture: tex }, raw, { bytesPerRow: S * 4 }, [S, S]);
    if (mipgen) {
      const e = d.createCommandEncoder();
      for (let l = 1; l < MIPS; l++) {
        const bg = d.createBindGroup({ layout: mipPipe[format].getBindGroupLayout(0), entries: [{ binding: 0, resource: tex.createView({ baseMipLevel: l - 1, mipLevelCount: 1 }) }, { binding: 1, resource: linear }] });
        const pass = e.beginRenderPass({ colorAttachments: [{ view: tex.createView({ baseMipLevel: l, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store' }] });
        pass.setPipeline(mipPipe[format]); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
      }
      d.queue.submit([e.finish()]);
    }
    for (const l of [0, 1, 3, 6]) r['mip' + l] = await readLevel(tex, l);
    r.grad_lod0 = await sampleThrough(tex, 'fs_grad', 1);
    r.grad_far = await sampleThrough(tex, 'fs_grad', 40);
    r.level3 = await sampleThrough(tex, 'fs_level', 3);
    const ie = await d.popErrorScope(); const ve = await d.popErrorScope();
    if (ie || ve) r.error = (ie || ve).message.split('\\n')[0];
  } catch (e) { r.error = String(e.message || e).split('\\n')[0]; }
  results.push(r); log('texprobe', r);
  line(JSON.stringify(r));
}
log('texprobe-done', results.length);
line('done (expected ~200,150,50,255 everywhere; srgb readbacks are the stored bytes)');
</script>`;
writeFileSync(out, page);
console.log(`wrote ${out}`);

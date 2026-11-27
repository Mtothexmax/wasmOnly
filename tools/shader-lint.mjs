#!/usr/bin/env node
// WGSL lint for driver bugs the WebGPU validation does not catch.
//
//   node tools/shader-lint.mjs [dir|file ...]     (default: crates/engine/src/shaders)
//
// Rule `dynamic-array-index`: a let/var/const array (function-local or module const) indexed with
// a runtime value. A PowerVR B-series driver (Android 10, Chrome 158, 2026-11-27) silently drops
// every draw whose shader contains that: no validation error, no device loss, nothing drawn.
// Storage/uniform buffers (`var<storage>`, struct fields) and vectors are not affected.
// Constant indices (`WAVE_LEN[3]`) are const-expressions and fine. Fix: bit maths (`(i << 1u) & 2u`),
// `select()` chains, or calls with constant indices.
// Exit code 1 when something is found; `dev.mjs build` runs it first.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function files(p) {
  if (!fs.existsSync(p)) return [];
  if (fs.statSync(p).isFile()) return p.endsWith('.wgsl') ? [p] : [];
  return fs.readdirSync(p, { withFileTypes: true }).flatMap((e) =>
    e.name === 'target' || e.name === 'node_modules' || e.name.startsWith('.') ? [] : files(path.join(p, e.name)));
}

/** Findings for one WGSL source: [{line, name, index, text}]. */
export function lintSource(src) {
  const code = src.replace(/\/\/[^\n]*/g, (m) => ' '.repeat(m.length)); // keep offsets
  const names = new Set();
  // `let x = array<..>(..)`, `const X = array(..)`, `var x: array<T, N>` (not var<storage>/var<uniform>)
  for (const m of code.matchAll(/\b(?:let|const|var)\s+(\w+)\s*(?::\s*array\s*<[^=;]*>)?\s*=\s*array\b/g)) names.add(m[1]);
  for (const m of code.matchAll(/\bvar\s+(\w+)\s*:\s*array\s*</g)) names.add(m[1]);
  const out = [];
  for (const name of names) {
    const re = new RegExp(`(?<![\\w.])${name}\\s*\\[`, 'g');
    for (const m of code.matchAll(re)) {
      // the index expression up to the matching ]
      let depth = 1, i = m.index + m[0].length, j = i;
      while (j < code.length && depth > 0) { if (code[j] === '[') depth++; else if (code[j] === ']') depth--; j++; }
      const index = code.slice(i, j - 1).trim();
      if (/^\d+[ui]?$/.test(index)) continue;
      const line = code.slice(0, m.index).split('\n').length;
      out.push({ line, name, index, text: src.split('\n')[line - 1].trim() });
    }
  }
  return out.sort((a, b) => a.line - b.line);
}

export function lint(paths) {
  const found = [];
  for (const f of paths.flatMap(files)) {
    for (const h of lintSource(fs.readFileSync(f, 'utf8'))) found.push({ file: path.relative(ROOT, f), ...h });
  }
  return found;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const found = lint(args.length ? args.map((a) => path.resolve(a)) : [path.join(ROOT, 'crates/engine/src/shaders')]);
  for (const h of found) {
    console.log(`${h.file}:${h.line}: dynamic-array-index: ${h.name}[${h.index}] (PowerVR drops the draw; use select/bit maths)\n    ${h.text}`);
  }
  console.log(JSON.stringify({ shader_lint: found.length ? 'error' : 'ok', findings: found.length }));
  process.exit(found.length ? 1 : 0);
}

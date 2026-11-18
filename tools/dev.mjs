#!/usr/bin/env node
// Build + serve + watch for the wasmOnly engine. Zero npm dependencies (Node >= 22).
//
//   node tools/dev.mjs build [--game demo] [--release] [--threads]   -> dist/
//     --threads: threads backend (Web Worker pool, shared memory, atomics). Uses the pinned nightly
//     from Cargo.toml [workspace.metadata.wasmonly] threads-toolchain + build-std; output in target/threads.
//   node tools/dev.mjs serve [--port 8080]               -> static server for dist/ (+ /__events live reload)
//   node tools/dev.mjs watch [--game demo] [--port 8080] -> build, serve, rebuild + reload on change
//
// POST /__throttle?kbps=64&match=assets/music slows responses whose path contains `match` (test
// streaming/slow networks; kbps=0 turns it off). GET /__throttle shows the setting.
//
// POST /__debuglog?file=<name>.log appends the request body (JSON lines from the engine's debug
// mode, toggled with the ` key) to logs/<name>.log in the project root.
//
// Output is line-oriented; build results end with a single JSON line (AI-first):
//   {"build":"ok","game":"demo","profile":"dev","ms":1234,"wasm_bytes":...}

import { spawnSync, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'dist');
const args = process.argv.slice(2);
const cmd = args[0] ?? 'watch';
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const flag = (name) => args.includes(name);
const game = opt('--game', 'demo');
const release = flag('--release');
const threads = flag('--threads');

// Pinned toolchains (reproducible builds): stable in rust-toolchain.toml, threads nightly in Cargo.toml.
function threadsToolchain() {
  const m = fs.readFileSync(path.join(ROOT, 'Cargo.toml'), 'utf8').match(/threads-toolchain\s*=\s*"([^"]+)"/);
  if (!m) throw new Error('Cargo.toml: [workspace.metadata.wasmonly] threads-toolchain missing');
  return m[1];
}
const THREAD_RUSTFLAGS = [
  '--cfg=web_sys_unstable_apis', '-C', 'target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128',
  // shared, imported memory + TLS exports: wasm-bindgen instantiates the same module per worker
  '-C', 'link-arg=--shared-memory', '-C', 'link-arg=--import-memory', '-C', 'link-arg=--max-memory=2147483648',
  '-C', 'link-arg=--export=__wasm_init_tls', '-C', 'link-arg=--export=__tls_size',
  '-C', 'link-arg=--export=__tls_align', '-C', 'link-arg=--export=__tls_base',
].join(' ');
const port = Number(opt('--port', 8080));

function which(bin) {
  const r = spawnSync(process.platform === 'win32' ? 'where' : 'which', [bin], { encoding: 'utf8' });
  return r.status === 0 ? r.stdout.split(/\r?\n/)[0].trim() : null;
}

function copyDir(src, dst) {
  if (!fs.existsSync(src)) return;
  fs.mkdirSync(dst, { recursive: true });
  for (const e of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, e.name), d = path.join(dst, e.name);
    e.isDirectory() ? copyDir(s, d) : fs.copyFileSync(s, d);
  }
}

function checkBindgenVersion() {
  const lock = fs.readFileSync(path.join(ROOT, 'Cargo.lock'), 'utf8');
  const m = lock.match(/name = "wasm-bindgen"\r?\nversion = "([^"]+)"/);
  const cli = execFileSync('wasm-bindgen', ['--version'], { encoding: 'utf8' }).trim().split(' ')[1];
  if (m && m[1] !== cli) {
    throw new Error(`wasm-bindgen CLI ${cli} != crate ${m[1]}. Fix: cargo install wasm-bindgen-cli --version ${m[1]}`);
  }
}

export function build() {
  const t0 = Date.now();
  const profile = release ? 'release' : 'debug';
  const cargoArgs = ['build', '-p', game, '--target', 'wasm32-unknown-unknown'];
  if (release) cargoArgs.push('--release');
  const env = { ...process.env };
  let targetDir = path.join(ROOT, 'target');
  let toolchain = null;
  if (threads) {
    toolchain = threadsToolchain();
    cargoArgs.unshift(`+${toolchain}`);
    cargoArgs.push('-Z', 'build-std=panic_abort,std', '--features', 'engine/threads');
    targetDir = path.join(ROOT, 'target', 'threads');
    env.CARGO_TARGET_DIR = targetDir;
    env.CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS = THREAD_RUSTFLAGS;
  }
  console.log(`> cargo ${cargoArgs.join(' ')}`);
  const r = spawnSync('cargo', cargoArgs, { cwd: ROOT, stdio: 'inherit', env });
  if (r.status !== 0) {
    console.log(JSON.stringify({ build: 'error', stage: 'cargo', game, profile }));
    return false;
  }
  checkBindgenVersion();
  const wasm = path.join(targetDir, 'wasm32-unknown-unknown', profile, `${game}.wasm`);
  const pkg = path.join(DIST, 'pkg');
  fs.rmSync(pkg, { recursive: true, force: true });
  const bg = ['--target', 'web', '--out-dir', pkg, '--out-name', 'game', '--no-typescript'];
  if (!release) bg.push('--keep-debug'); // DWARF for Chrome DevTools (C/C++ DevTools Support extension)
  bg.push(wasm);
  console.log(`> wasm-bindgen ${bg.join(' ')}`);
  const b = spawnSync('wasm-bindgen', bg, { cwd: ROOT, stdio: 'inherit' });
  if (b.status !== 0) {
    console.log(JSON.stringify({ build: 'error', stage: 'wasm-bindgen', game, profile }));
    return false;
  }
  const out = path.join(pkg, 'game_bg.wasm');
  if (release) {
    if (which('wasm-opt')) {
      console.log('> wasm-opt -O3');
      spawnSync('wasm-opt', ['-O3', '--enable-simd', '--enable-bulk-memory', '--enable-nontrapping-float-to-int', '-o', out, out], { stdio: 'inherit' });
    } else {
      console.log('(wasm-opt not found — skipping; install binaryen for smaller release builds)');
    }
  }
  fs.copyFileSync(path.join(ROOT, 'web', 'index.html'), path.join(DIST, 'index.html'));
  copyDir(path.join(ROOT, 'games', game, 'assets'), path.join(DIST, 'assets'));
  const info = { build: 'ok', game, profile, threads, toolchain: toolchain ?? 'rust-toolchain.toml', ms: Date.now() - t0, wasm_bytes: fs.statSync(out).size };
  console.log(JSON.stringify(info));
  return true;
}

const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json',
  '.glb': 'model/gltf-binary', '.gltf': 'model/gltf+json', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp',
  '.svg': 'image/svg+xml', '.css': 'text/css', '.ktx2': 'image/ktx2', '.mp3': 'audio/mpeg', '.ogg': 'audio/ogg', '.wav': 'audio/wav' };
const clients = new Set();
let throttle = { kbps: 0, match: '' };

/** Pipes a file stream to the response; while the throttle matches `urlPath` each chunk is
 *  rate-limited (checked per chunk, so lifting the throttle speeds up running downloads too). */
function sendStream(stream, res, urlPath) {
  stream.on('data', (chunk) => {
    // resume only when every reason to wait is over: socket drained AND throttle delay elapsed
    let waits = 0;
    const done = () => { if (--waits === 0) stream.resume(); };
    if (!res.write(chunk)) { waits++; res.once('drain', done); }
    if (throttle.kbps && urlPath.includes(throttle.match)) { waits++; setTimeout(done, (chunk.length / (throttle.kbps * 128)) * 1000); }
    if (waits > 0) stream.pause();
  });
  stream.on('end', () => res.end());
  res.on('close', () => stream.destroy());
}

export function serve() {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    if (url.pathname === '/__events') {
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
      res.write(': hi\n\n');
      clients.add(res);
      req.on('close', () => clients.delete(res));
      return;
    }
    if (url.pathname === '/__throttle') {
      if (req.method === 'POST') {
        throttle = { kbps: Number(url.searchParams.get('kbps') ?? 0), match: url.searchParams.get('match') ?? '' };
        console.log(JSON.stringify({ throttle }));
      }
      res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(throttle));
      return;
    }
    if (url.pathname === '/__debuglog' && req.method === 'POST') {
      const name = (url.searchParams.get('file') ?? 'debug.log').replace(/[^A-Za-z0-9._-]/g, '_');
      if (!name.endsWith('.log')) { res.writeHead(400).end('file must end in .log'); return; }
      const dir = path.join(ROOT, 'logs');
      const chunks = [];
      req.on('data', (c) => chunks.push(c));
      req.on('end', () => {
        fs.mkdirSync(dir, { recursive: true });
        const file = path.join(dir, name);
        fs.appendFileSync(file, Buffer.concat(chunks));
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ file: path.relative(ROOT, file).split(path.sep).join('/') }));
      });
      return;
    }
    let p = path.normalize(path.join(DIST, decodeURIComponent(url.pathname)));
    if (!p.startsWith(DIST)) { res.writeHead(403).end(); return; }
    if (fs.existsSync(p) && fs.statSync(p).isDirectory()) p = path.join(p, 'index.html');
    fs.stat(p, (err, st) => {
      if (err || !st.isFile()) { res.writeHead(404, { 'Content-Type': 'text/plain' }).end('not found'); return; }
      const headers = {
        'Content-Type': MIME[path.extname(p)] ?? 'application/octet-stream', 'Cache-Control': 'no-store',
        // Cross-origin isolation: enables SharedArrayBuffer (threads backend).
        'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp',
        // Range requests: <audio> streams music progressively and can seek/loop without a full download.
        'Accept-Ranges': 'bytes',
      };
      const m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? '');
      if (m && (m[1] !== '' || m[2] !== '')) {
        let start = m[1] === '' ? Math.max(0, st.size - Number(m[2])) : Number(m[1]);
        let end = m[1] !== '' && m[2] !== '' ? Math.min(Number(m[2]), st.size - 1) : st.size - 1;
        if (start >= st.size || start > end) {
          res.writeHead(416, { ...headers, 'Content-Range': `bytes */${st.size}` }).end();
          return;
        }
        res.writeHead(206, { ...headers, 'Content-Range': `bytes ${start}-${end}/${st.size}`, 'Content-Length': end - start + 1 });
        sendStream(fs.createReadStream(p, { start, end, highWaterMark: 16 * 1024 }), res, url.pathname);
        return;
      }
      res.writeHead(200, { ...headers, 'Content-Length': st.size });
      if (req.method === 'HEAD') { res.end(); return; }
      sendStream(fs.createReadStream(p, { highWaterMark: 16 * 1024 }), res, url.pathname);
    });
  });
  server.listen(port, '127.0.0.1', () => console.log(JSON.stringify({ serve: 'ok', url: `http://localhost:${port}/` })));
  return server;
}

function reloadClients() {
  for (const c of clients) c.write('event: reload\ndata: 1\n\n');
}

function watch() {
  build();
  serve();
  let timer = null, building = false, again = false;
  const trigger = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (building) { again = true; return; }
      building = true;
      const ok = build();
      building = false;
      if (ok) reloadClients();
      if (again) { again = false; trigger(); }
    }, 200);
  };
  for (const dir of ['crates', 'games', 'web']) {
    fs.watch(path.join(ROOT, dir), { recursive: true }, (_ev, file) => {
      if (file && !/target|[\\/]\.|~$/.test(file)) trigger();
    });
  }
  console.log('watching crates/ games/ web/ ...');
}

if (cmd === 'build') process.exit(build() ? 0 : 1);
else if (cmd === 'serve') serve();
else if (cmd === 'watch') watch();
else { console.error('usage: node tools/dev.mjs build|serve|watch [--game demo] [--release] [--port 8080]'); process.exit(2); }

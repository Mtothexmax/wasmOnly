#!/usr/bin/env node
// Chrome debug bridge — drives a running game over the Chrome DevTools Protocol (CDP).
// Zero npm dependencies (Node >= 22: global fetch + WebSocket).
//
// CLI (every command prints JSON on stdout):
//   node tools/bridge.mjs launch [--url http://localhost:8080/] [--headless] [--sound]   start Chrome with CDP on :9222
//     (muted by default so agents don't play audio at the user while developing; --sound = audible)
//   node tools/bridge.mjs status                     Chrome + page + engine status
//   node tools/bridge.mjs call <cmd> ['<json args>'] window.engine.call(cmd, args)   e.g. call stats
//   node tools/bridge.mjs eval '<js expression>'     evaluate in the page (awaits promises)
//   node tools/bridge.mjs screenshot [out.png]       PNG of the page
//   node tools/bridge.mjs reload                     reload page, wait for engine-ready, report errors
//   node tools/bridge.mjs wait [--timeout 20000]     wait until window.engine.ready
//   node tools/bridge.mjs console [--ms 3000]        stream console/exception events for N ms
//   node tools/bridge.mjs check [--flicker] [--strict] [--out heat.png]
//                                                    build check: reload, assets, z-fighting, log errors
//                                                    (exit 1 on problems; --strict also fails on warnings)
//   node tools/bridge.mjs flicker [--frames 6] [--jitter 0.2] [--threshold 48] [--out heat.png]
//                                                    screenshots with flipped draw order + perturbed depth -> flickering regions
//   node tools/bridge.mjs perf [--ms 3000] [--profile] [--top 15] frame times, engine CPU phases, GPU pass times,
//                                                    verdict (vsync/gpu/cpu-bound); --profile adds top CPU functions
//   node tools/bridge.mjs run <script.txt|.json> [--out dir] [--keep-going]
//   node tools/bridge.mjs run --commands "key Space; step 30; expect game_state.jumps == 1; screenshot a.png"
//                                                    scripted test run (see tools/script.mjs), exit 1 on failure
//   node tools/bridge.mjs mcp                        run as MCP server (stdio) — see .mcp.json
//
// Options: --port 9222 (CDP port), --url <page url to target/open>, --profile-dir <dir> (launch)
//   Second agent / second game at the same time: its own Chrome, e.g.
//   node tools/bridge.mjs launch --port 9223 --url http://localhost:8091/   (then pass the same --port/--url
//   to every command, or set CDP_PORT=9223 GAME_URL=http://localhost:8091/ in the environment)

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { check, flicker, perf } from './qa.mjs';
import { runScript } from './script.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const flag = (n) => argv.includes(n);
const CDP_PORT = Number(opt('--port', process.env.CDP_PORT ?? 9222));
const PAGE_URL = opt('--url', process.env.GAME_URL ?? 'http://localhost:8080/');
const CDP = `http://127.0.0.1:${CDP_PORT}`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------------------------------------
function findChrome() {
  const c = [process.env.CHROME,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    path.join(os.homedir(), 'AppData/Local/Google/Chrome/Application/chrome.exe'),
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium'];
  return c.find((p) => p && fs.existsSync(p));
}

async function cdpVersion() {
  try { return await (await fetch(`${CDP}/json/version`)).json(); } catch { return null; }
}

async function launch(url = PAGE_URL, headless = false) {
  const existing = await cdpVersion();
  if (existing) {
    await openPage(url);
    return { launched: false, reason: 'chrome already listening', browser: existing.Browser, url };
  }
  const chrome = findChrome();
  if (!chrome) throw new Error('Chrome not found; set CHROME=<path to chrome executable>');
  // one profile per CDP port: two agents can each run their own debug Chrome (no focus fights:
  // a tab of the other session taking focus would blur this page)
  const profile = path.resolve(opt('--profile-dir') ?? path.join(ROOT, CDP_PORT === 9222 ? '.chrome-debug-profile' : `.chrome-debug-profile-${CDP_PORT}`));
  const args = [`--remote-debugging-port=${CDP_PORT}`, `--user-data-dir=${profile}`, '--no-first-run', '--autoplay-policy=no-user-gesture-required', '--no-default-browser-check',
    // keep rAF running when the window is occluded/in background, so agents see live frames
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
    '--window-size=1280,800'];
  // Muted output: Web Audio and <audio> still run (all audio API checks work), nothing is audible.
  if (!flag('--sound')) args.push('--mute-audio');
  if (headless) args.push('--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan');
  args.push(url);
  const child = spawn(chrome, args, { detached: true, stdio: 'ignore' });
  child.unref();
  for (let i = 0; i < 50; i++) {
    const v = await cdpVersion();
    if (v) return { launched: true, browser: v.Browser, url, cdp: CDP, profile };
    await sleep(200);
  }
  throw new Error(`Chrome started but CDP did not answer on ${CDP}`);
}

async function openPage(url) {
  const list = await (await fetch(`${CDP}/json/list`)).json();
  if (list.some((t) => t.type === 'page' && t.url.startsWith(url))) return;
  await fetch(`${CDP}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' });
}

async function pickTarget() {
  let list;
  try { list = await (await fetch(`${CDP}/json/list`)).json(); }
  catch { throw new Error(`no Chrome with CDP on ${CDP} — run: node tools/bridge.mjs launch`); }
  const pages = list.filter((t) => t.type === 'page');
  const origin = new URL(PAGE_URL).origin;
  const t = pages.find((p) => p.url.startsWith(origin)) ?? null;
  if (!t) throw new Error(`no tab with ${origin} open (tabs: ${pages.map((p) => p.url).join(', ') || 'none'}) — run: node tools/bridge.mjs launch`);
  return t;
}

class Session {
  static async open() {
    const t = await pickTarget();
    const s = new Session();
    s.target = t;
    s.ws = new WebSocket(t.webSocketDebuggerUrl);
    s.nextId = 1; s.pending = new Map(); s.listeners = new Set();
    s.ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && s.pending.has(msg.id)) {
        const { resolve, reject } = s.pending.get(msg.id); s.pending.delete(msg.id);
        msg.error ? reject(new Error(`${msg.error.message} ${msg.error.data ?? ''}`)) : resolve(msg.result);
      } else if (msg.method) for (const l of s.listeners) l(msg);
    };
    await new Promise((res, rej) => { s.ws.onopen = res; s.ws.onerror = () => rej(new Error('CDP websocket failed')); });
    s.ws.onclose = () => { s.closed = true; for (const { reject } of s.pending.values()) reject(new Error('CDP connection closed')); };
    return s;
  }
  send(method, params = {}) {
    const id = this.nextId++;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { if (this.pending.has(id)) { this.pending.delete(id); reject(new Error(`CDP ${method} timed out`)); } }, 30000);
      timer.unref();
      this.pending.set(id, { resolve: (v) => { clearTimeout(timer); resolve(v); }, reject: (e) => { clearTimeout(timer); reject(e); } });
    });
  }
  async eval(expression) {
    const r = await this.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, userGesture: true });
    if (r.exceptionDetails) {
      const d = r.exceptionDetails;
      throw new Error(d.exception?.description ?? d.text ?? 'evaluation failed');
    }
    return r.result.value;
  }
  async call(name, args) {
    const expr = `(() => { if (!window.engine?.call) return { ok: false, error: window.engine?.fatal ? 'engine failed: ' + window.engine.fatal : 'engine not ready (window.engine.call missing)' };
      return window.engine.call(${JSON.stringify(name)}, ${JSON.stringify(args ?? null)}); })()`;
    return this.eval(expr);
  }
  async screenshot() {
    await this.send('Page.bringToFront').catch(() => {});
    const r = await this.send('Page.captureScreenshot', { format: 'png' });
    return r.data;
  }
  async waitReady(timeout = 20000) {
    const t0 = Date.now();
    while (Date.now() - t0 < timeout) {
      const s = await this.eval('({ready: !!window.engine?.ready, fatal: window.engine?.fatal ?? null})').catch(() => null);
      if (s?.fatal) return { ready: false, fatal: s.fatal, ms: Date.now() - t0 };
      if (s?.ready) return { ready: true, ms: Date.now() - t0 };
      await sleep(150);
    }
    return { ready: false, timeout: true, ms: Date.now() - t0 };
  }
  async reload() {
    // Mark the old page, reload, then wait until a *new* page (no mark) reports engine-ready.
    // No page lifecycle events: streamed <audio> (music) delays 'load' by design.
    await this.eval('window.__bridgeOldPage = true').catch(() => {});
    const t0 = Date.now();
    await this.send('Page.reload', { ignoreCache: true });
    while (Date.now() - t0 < 20000) {
      const fresh = await this.eval('window.__bridgeOldPage === undefined').catch(() => false);
      if (fresh) break;
      await sleep(50);
    }
    const ready = await this.waitReady();
    const errors = ready.ready ? (await this.call('logs', { level: 'warn' }))?.result?.entries ?? [] : [];
    return { ...ready, ms: Date.now() - t0, warnings_and_errors: errors };
  }
  close() { try { this.ws.close(); } catch {} }
}

async function consoleStream(ms) {
  const s = await Session.open();
  s.listeners.add((m) => {
    if (m.method === 'Runtime.consoleAPICalled') {
      const text = m.params.args.map((a) => a.value ?? a.description ?? a.type).join(' ');
      console.log(JSON.stringify({ type: 'console', level: m.params.type, text }));
    } else if (m.method === 'Runtime.exceptionThrown') {
      const d = m.params.exceptionDetails;
      console.log(JSON.stringify({ type: 'exception', text: d.exception?.description ?? d.text }));
    } else if (m.method === 'Log.entryAdded') {
      console.log(JSON.stringify({ type: 'log', level: m.params.entry.level, source: m.params.entry.source, text: m.params.entry.text }));
    }
  });
  await s.send('Runtime.enable');
  await s.send('Log.enable');
  await sleep(ms);
  s.close();
}

// ------------------------------------------------------------------------------------------------
// MCP server (stdio, newline-delimited JSON-RPC 2.0)
const TOOLS = [
  { name: 'launch_chrome', description: 'Start Chrome with remote debugging and open the game page (no-op if already running).',
    inputSchema: { type: 'object', properties: { url: { type: 'string', description: `default ${PAGE_URL}` }, headless: { type: 'boolean' } } } },
  { name: 'status', description: 'Chrome/page/engine status (ready, fatal error, basic stats).', inputSchema: { type: 'object', properties: {} } },
  { name: 'engine_call', description: "Call the in-game AI API: window.engine.call(name, args). Use name='help' to list all commands (stats, entities, entity, set, spawn, despawn, pause, resume, step, input, logs, render, camera, raycast, pick, ui, assets, game_state, plus game commands).",
    inputSchema: { type: 'object', properties: { name: { type: 'string' }, args: { type: 'object' } }, required: ['name'] } },
  { name: 'screenshot', description: 'PNG screenshot of the game page.', inputSchema: { type: 'object', properties: {} } },
  { name: 'reload', description: 'Reload the page (e.g. after a rebuild) and wait for engine-ready; returns warnings/errors from the engine log.', inputSchema: { type: 'object', properties: {} } },
  { name: 'check', description: 'Build check: reload, wait for assets, z-fighting test, engine warnings/errors (optionally a flicker scan). Returns ok + details.',
    inputSchema: { type: 'object', properties: { flicker: { type: 'boolean' }, strict: { type: 'boolean' } } } },
  { name: 'flicker', description: 'Flicker scan: several screenshots alternating draw order and depth mapping (time frozen, HUD hidden, pixels stay put); returns flickering regions with the entity under each (z-fighting, shadow acne, shimmering).',
    inputSchema: { type: 'object', properties: { frames: { type: 'number' }, threshold: { type: 'number' } } } },
  { name: 'perf', description: 'Performance capture: frame-time percentiles, engine CPU phases, GPU time per pass, verdict (smooth / gpu-bound / cpu-bound / stutter); profile=true adds the top CPU functions (Chrome sampling profiler).',
    inputSchema: { type: 'object', properties: { ms: { type: 'number' }, profile: { type: 'boolean' } } } },
  { name: 'run_script', description: 'Scripted test run: a list of steps executed in order — reload, pause/resume, step N (deterministic), frames N, wait ms, key/down/up/hold <Code>, mouse dx dy, click x y, action <name>, screen <name>, call <cmd> {json}, wait_for <cmd.path> <op> <value> [ms], expect <cmd.path> <op> <value>, screenshot <file>, log <text>. `script` is text (lines or ";"-separated) or an array of {verb: arg}. Returns a pass/fail report per step plus engine warnings/errors that appeared.',
    inputSchema: { type: 'object', properties: { script: { description: 'text or array of steps' }, keep_going: { type: 'boolean' }, out_dir: { type: 'string' } }, required: ['script'] } },
  { name: 'eval_js', description: 'Evaluate a JavaScript expression in the page (promises awaited, result returned by value).',
    inputSchema: { type: 'object', properties: { expression: { type: 'string' } }, required: ['expression'] } },
];

async function status() {
  const v = await cdpVersion();
  if (!v) return { chrome: null, hint: 'node tools/bridge.mjs launch' };
  let page = null, engine = null;
  try {
    const s = await Session.open();
    page = s.target.url;
    engine = await s.eval(`({ready: !!window.engine?.ready, fatal: window.engine?.fatal ?? null, version: window.engine?.version ?? null})`);
    if (engine.ready) engine.stats = (await s.call('stats'))?.result;
    s.close();
  } catch (e) { page = { error: e.message }; }
  return { chrome: v.Browser, cdp: CDP, page, engine };
}

async function withSession(fn, front = false) {
  const s = await Session.open();
  // long-running commands need live frames: a hidden tab (another tab in front) gets no rAF
  if (front) await s.send('Page.bringToFront').catch(() => {});
  try { return await fn(s); } finally { s.close(); }
}

async function runTool(name, a = {}) {
  switch (name) {
    case 'launch_chrome': return text(await launch(a.url ?? PAGE_URL, !!a.headless));
    case 'status': return text(await status());
    case 'engine_call': return text(await withSession((s) => s.call(a.name, a.args)));
    case 'screenshot': return { content: [{ type: 'image', data: await withSession((s) => s.screenshot()), mimeType: 'image/png' }] };
    case 'reload': return text(await withSession((s) => s.reload()));
    case 'eval_js': return text(await withSession((s) => s.eval(a.expression)));
    case 'run_script': return text(await withSession((s) => runScript(s, a.script, { stopOnFail: !a.keep_going, outDir: a.out_dir ?? ROOT }), true));
    case 'perf': return text(await withSession((s) => perf(s, { ms: a.ms, profile: !!a.profile }), true));
    case 'check': return text(await withSession((s) => check(s, { flicker: !!a.flicker, strict: !!a.strict }), true));
    case 'flicker': return text(await withSession((s) => flicker(s, { frames: a.frames, threshold: a.threshold }), true));
    default: throw new Error(`unknown tool ${name}`);
  }
}
const text = (v) => ({ content: [{ type: 'text', text: typeof v === 'string' ? v : JSON.stringify(v, null, 2) }] });

function mcp() {
  let buf = '';
  const reply = (id, result, error) => process.stdout.write(JSON.stringify(error ? { jsonrpc: '2.0', id, error } : { jsonrpc: '2.0', id, result }) + '\n');
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', async (chunk) => {
    buf += chunk;
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
      if (!line) continue;
      let msg; try { msg = JSON.parse(line); } catch { continue; }
      if (msg.id === undefined) continue; // notifications
      try {
        if (msg.method === 'initialize') {
          reply(msg.id, { protocolVersion: msg.params?.protocolVersion ?? '2025-06-18', capabilities: { tools: {} },
            serverInfo: { name: 'wasmonly-bridge', version: '0.1.0' },
            instructions: 'Drives the wasmOnly game in Chrome via CDP. Start with launch_chrome (dev server must run: node tools/dev.mjs watch), then engine_call {name:"help"}.' });
        } else if (msg.method === 'tools/list') reply(msg.id, { tools: TOOLS });
        else if (msg.method === 'tools/call') {
          try { reply(msg.id, await runTool(msg.params.name, msg.params.arguments ?? {})); }
          catch (e) { reply(msg.id, { content: [{ type: 'text', text: `error: ${e.message}` }], isError: true }); }
        } else if (msg.method === 'ping') reply(msg.id, {});
        else reply(msg.id, null, { code: -32601, message: `method not found: ${msg.method}` });
      } catch (e) { reply(msg.id, null, { code: -32603, message: e.message }); }
    }
  });
}

// ------------------------------------------------------------------------------------------------
const out = (v) => console.log(JSON.stringify(v, null, 2));
const positional = argv.filter((a, i) => !a.startsWith('--') && !(i > 0 && argv[i - 1].startsWith('--') && !['--headless', '--release', '--sound', '--flicker', '--strict', '--profile', '--keep-going'].includes(argv[i - 1])));
const [cmd, ...rest] = positional;

try {
  switch (cmd) {
    case 'launch': out(await launch(PAGE_URL, flag('--headless'))); break;
    case 'status': out(await status()); break;
    case 'call': {
      const args = rest[1] ? JSON.parse(rest[1]) : null;
      const r = await withSession((s) => s.call(rest[0] ?? 'help', args));
      out(r);
      if (r && r.ok === false) process.exitCode = 1;
      break;
    }
    case 'eval': out(await withSession((s) => s.eval(rest.join(' ')))); break;
    case 'screenshot': {
      const file = rest[0] ?? path.join(ROOT, 'screenshot.png');
      const data = await withSession((s) => s.screenshot());
      fs.writeFileSync(file, Buffer.from(data, 'base64'));
      out({ file: path.resolve(file), bytes: fs.statSync(file).size });
      break;
    }
    case 'reload': { const r = await withSession((s) => s.reload()); out(r); if (!r.ready) process.exitCode = 1; break; }
    case 'wait': { const r = await withSession((s) => s.waitReady(Number(opt('--timeout', 20000)))); out(r); if (!r.ready) process.exitCode = 1; break; }
    case 'console': await consoleStream(Number(opt('--ms', 3000))); break;
    case 'check': {
      const r = await withSession((s) => check(s, { flicker: flag('--flicker'), strict: flag('--strict'), out: opt('--out') }), true);
      out(r);
      if (!r.ok) process.exitCode = 1;
      break;
    }
    case 'perf': {
      out(await withSession((s) => perf(s, { ms: Number(opt('--ms', 3000)), profile: flag('--profile'), top: Number(opt('--top', 15)) }), true));
      break;
    }
    case 'flicker': {
      const num = (n, d) => (opt(n) !== undefined ? Number(opt(n)) : d);
      const r = await withSession((s) => flicker(s, { frames: num('--frames', 6), jitter: num('--jitter', 0.2), threshold: num('--threshold', 48), out: opt('--out') }), true);
      out(r);
      if (r.flicker_cells > 0) process.exitCode = 1;
      break;
    }
    case 'run': {
      // node tools/bridge.mjs run tests/jump.txt   |   run --commands "key Space; step 30; expect game_state.jumps == 1"
      const src = opt('--commands') ?? fs.readFileSync(rest[0], 'utf8');
      const outDir = opt('--out') ?? (rest[0] ? path.dirname(path.resolve(rest[0])) : process.cwd());
      const r = await withSession((s) => runScript(s, src, { stopOnFail: !flag('--keep-going'), outDir }), true);
      out(r);
      if (!r.passed) process.exitCode = 1;
      break;
    }
    case 'mcp': mcp(); break;
    default:
      console.error('usage: node tools/bridge.mjs launch|status|call|eval|screenshot|reload|wait|console|check|flicker|perf|run|mcp  (see header of this file)');
      process.exitCode = 2;
  }
} catch (e) {
  out({ error: e.message });
  process.exitCode = 1;
}

// Scripted test runs: a list of steps (input, wait, step, screenshot, expectations) executed
// against the running game through the CDP session — one command instead of a chain of
// `bridge call` / `bridge screenshot` invocations. Used by `bridge run` and the MCP tool `run_script`.
//
// Text form (one command per line or separated by ";", `#` starts a comment):
//
//   reload                              reload the page, wait for engine-ready (+ assets)
//   pause | resume                      simulation
//   step 60                             60 fixed steps (1/60 s each), synchronous + deterministic
//   frames 3                            wait 3 rendered frames
//   wait 500                            wait 500 ms real time
//   key Space                           tap: down, 2 frames, up (edge input, read in Game::update)
//   down KeyW | up KeyW                 hold / release (read in fixed_update while held)
//   hold KeyW 30                        down, step 30, up
//   mouse 40 0                          relative mouse motion (dx dy)
//   click 640 360                       left click at CSS pixel (pointer moves there first)
//   action reset                        HUD/panel action (same path as data-action buttons)
//   screen play                         switch screen (ScreenStack games: screens {switch})
//   call game_state                     any engine/game command; `call render {"bloom":false}`
//   wait_for game_state.grounded == true 3000   poll until true (timeout ms, default 5000)
//   expect game_state.score >= 1        assertion (ops: == != < <= > >= ~= (abs diff < 0.01), contains)
//   screenshot shots/after-jump.png     PNG (directory created)
//   log some text                       note in the report
//
// Paths: `<command>.<field>.<index>...`, e.g. `entity{"id":5}.pos.1` passes args in braces.
// JSON form: [{"key":"Space"}, {"step":60}, {"expect":"game_state.jumps == 1"}, ...] — each object
// has one verb as key and its argument string/number/object as value.
//
// Report: {passed, failures, steps:[{i, cmd, ok, ms, value?, error?}], screenshots, new_log}; the
// engine's warnings/errors produced during the run are in `new_log`.

import fs from 'node:fs';
import path from 'node:path';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const frames = (s, n = 3) => s.eval(`new Promise(r => { let k = 0; const f = () => (++k >= ${n} ? r(1) : requestAnimationFrame(f)); requestAnimationFrame(f); })`);

/** Text or JSON-array script -> [{verb, arg, src}] */
export function parseScript(src) {
  if (Array.isArray(src)) {
    return src.map((o) => {
      const [verb, arg] = Object.entries(o)[0] ?? [];
      return { verb, arg: typeof arg === 'string' || typeof arg === 'number' ? String(arg) : arg === true ? '' : JSON.stringify(arg), src: JSON.stringify(o) };
    });
  }
  const t = String(src).trim();
  if (t.startsWith('[')) return parseScript(JSON.parse(t));
  const out = [];
  for (let line of t.split(/\r?\n|;(?![^{]*})/)) {
    line = line.replace(/(^|\s)#.*$/, '').trim();
    if (!line) continue;
    const m = line.match(/^(\S+)\s*(.*)$/);
    out.push({ verb: m[1].toLowerCase(), arg: m[2].trim(), src: line });
  }
  return out;
}

/** "game_state.score" / 'entity{"id":3}.pos.1' -> {cmd, args, path[]} */
function parsePath(p) {
  const m = p.match(/^([a-z_][a-z0-9_]*)(\{.*?\})?(?:\.(.*))?$/i);
  if (!m) throw new Error(`bad path '${p}'`);
  return { cmd: m[1], args: m[2] ? JSON.parse(m[2]) : undefined, path: m[3] ? m[3].split('.') : [] };
}

function dig(v, path) {
  for (const k of path) {
    if (v == null) return undefined;
    v = Array.isArray(v) && /^-?\d+$/.test(k) ? v[(Number(k) + v.length) % v.length] : v[k];
  }
  return v;
}

function parseValue(s) {
  try { return JSON.parse(s); } catch { return s; }
}

const OPS = {
  '==': (a, b) => JSON.stringify(a) === JSON.stringify(b),
  '!=': (a, b) => JSON.stringify(a) !== JSON.stringify(b),
  '<': (a, b) => a < b, '<=': (a, b) => a <= b, '>': (a, b) => a > b, '>=': (a, b) => a >= b,
  '~=': (a, b) => Math.abs(a - b) < 0.01,
  contains: (a, b) => (Array.isArray(a) ? a.some((x) => JSON.stringify(x) === JSON.stringify(b)) : String(a).includes(String(b))),
};

/** "game_state.score >= 1 [timeout]" -> {path, op, value, timeout} */
function parseCond(arg) {
  const m = arg.match(/^(\S+)\s+(==|!=|<=|>=|<|>|~=|contains)\s+(.+?)(?:\s+(\d+))?$/);
  if (!m) throw new Error(`condition must be '<path> <op> <value>', got '${arg}'`);
  return { path: parsePath(m[1]), op: m[2], value: parseValue(m[3]), timeout: m[4] ? Number(m[4]) : undefined };
}

async function engine(s, name, args) {
  const r = await s.call(name, args);
  if (!r || r.ok === false) throw new Error(`${name}: ${r?.error ?? 'no response'}`);
  return r.result;
}

async function evalCond(s, c) {
  const v = dig(await engine(s, c.path.cmd, c.path.args), c.path.path);
  return { ok: OPS[c.op](v, c.value), value: v };
}

/**
 * Runs `script` (text, JSON text or array). opts: {stopOnFail=true, outDir} (relative screenshot
 * paths resolve against outDir, default cwd).
 */
export async function runScript(s, script, opts = {}) {
  const steps = parseScript(script);
  const stopOnFail = opts.stopOnFail ?? true;
  const outDir = opts.outDir ?? process.cwd();
  const report = { passed: true, failures: 0, steps: [], screenshots: [] };
  let logSeq = 0;
  try { logSeq = (await engine(s, 'stats'))?.log_seq ?? 0; } catch {}
  // page marker: a reload we did not ask for (dev-server live reload after a rebuild) invalidates the run
  const runId = `run-${Date.now()}-${Math.random()}`;
  const mark = () => s.eval(`window.__scriptRun = ${JSON.stringify(runId)}`).catch(() => {});
  const marked = () => s.eval(`window.__scriptRun === ${JSON.stringify(runId)}`).catch(() => false);
  await mark();
  const t0 = Date.now();
  for (let i = 0; i < steps.length; i++) {
    const { verb, arg, src } = steps[i];
    const st = { i, cmd: src, ok: true };
    const ts = Date.now();
    try {
      if (verb !== 'reload' && !(await marked())) {
        report.page_reloaded = true;
        throw new Error('the page was reloaded during the run (dev-server live reload after a rebuild?) — rerun the script');
      }
      switch (verb) {
        case 'reload': {
          const r = await s.reload();
          if (!r.ready) throw new Error(r.fatal ?? 'engine not ready');
          for (let k = 0; k < 100; k++) { const a = await engine(s, 'assets'); if (!a.models_pending?.length) break; await sleep(100); }
          try { logSeq = (await engine(s, 'stats'))?.log_seq ?? logSeq; } catch {}
          await mark();
          break;
        }
        case 'pause': await engine(s, 'pause'); break;
        case 'resume': await engine(s, 'resume'); break;
        case 'step': st.value = await engine(s, 'step', { n: Number(arg || 1) }); await frames(s, 1); break;
        case 'frames': await frames(s, Number(arg || 1)); break;
        case 'wait': await sleep(Number(arg || 100)); break;
        case 'key': {
          const code = arg;
          await engine(s, 'input', { key: code, down: true });
          await frames(s, 2);
          await engine(s, 'input', { key: code, down: false });
          await frames(s, 1);
          break;
        }
        case 'down': await engine(s, 'input', { key: arg, down: true }); break;
        case 'up': await engine(s, 'input', { key: arg, down: false }); break;
        case 'hold': {
          const [code, n] = arg.split(/\s+/);
          await engine(s, 'input', { key: code, down: true });
          await frames(s, 1);
          await engine(s, 'step', { n: Number(n || 30) });
          await engine(s, 'input', { key: code, down: false });
          await frames(s, 1);
          break;
        }
        case 'mouse': {
          const [dx, dy] = arg.split(/\s+/).map(Number);
          await engine(s, 'input', { dx, dy });
          await frames(s, 1);
          break;
        }
        case 'click': {
          const [x, y] = arg.split(/\s+/).map(Number);
          if (Number.isFinite(x)) await engine(s, 'input', { x, y });
          await engine(s, 'input', { button: 0, down: true });
          await frames(s, 2);
          await engine(s, 'input', { button: 0, down: false });
          await frames(s, 1);
          break;
        }
        case 'action': await engine(s, 'input', { action: arg }); await frames(s, 2); break;
        case 'screen': st.value = (await engine(s, 'screens', { switch: arg })).stack; await frames(s, 2); break;
        case 'call': {
          const m = arg.match(/^(\S+)\s*(.*)$/);
          st.value = await engine(s, m[1], m[2] ? JSON.parse(m[2]) : undefined);
          break;
        }
        case 'expect': {
          const c = parseCond(arg);
          const r = await evalCond(s, c);
          st.value = r.value;
          if (!r.ok) throw new Error(`expected ${c.op} ${JSON.stringify(c.value)}, got ${JSON.stringify(r.value)}`);
          break;
        }
        case 'wait_for': {
          const c = parseCond(arg);
          const until = Date.now() + (c.timeout ?? 5000);
          let r;
          for (;;) {
            r = await evalCond(s, c);
            if (r.ok || Date.now() > until) break;
            await frames(s, 2);
          }
          st.value = r.value;
          if (!r.ok) throw new Error(`timed out: ${c.op} ${JSON.stringify(c.value)}, last ${JSON.stringify(r.value)}`);
          break;
        }
        case 'screenshot': {
          const file = path.resolve(outDir, arg || `step-${i}.png`);
          fs.mkdirSync(path.dirname(file), { recursive: true });
          await frames(s, 2);
          fs.writeFileSync(file, Buffer.from(await s.screenshot(), 'base64'));
          report.screenshots.push(file);
          st.value = file;
          break;
        }
        case 'log': st.value = arg; break;
        default: throw new Error(`unknown command '${verb}' (reload pause resume step frames wait key down up hold mouse click action screen call expect wait_for screenshot log)`);
      }
    } catch (e) {
      st.ok = false;
      st.error = e.message;
      report.passed = false;
      report.failures++;
    }
    st.ms = Date.now() - ts;
    report.steps.push(st);
    if (!st.ok && (stopOnFail || report.page_reloaded)) {
      report.stopped_at = i;
      break;
    }
  }
  report.ms = Date.now() - t0;
  try {
    const l = await engine(s, 'logs', { since: logSeq, level: 'warn', limit: 50 });
    report.new_log = (l.entries ?? []).map((e) => `${e.level}: ${e.msg}`);
  } catch {}
  return report;
}

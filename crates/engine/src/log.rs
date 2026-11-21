//! Structured logging. Every entry goes to the browser console *and* into a ring buffer that
//! AI tooling reads via `engine.call("logs", {since})` — no console scraping required.

use serde::Serialize;
use std::cell::RefCell;
use std::collections::VecDeque;

const CAPACITY: usize = 10_000;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Clone, Debug, Serialize)]
pub struct Entry {
    pub seq: u64,
    /// Milliseconds since page load (performance.now()).
    pub t: f64,
    /// Fixed simulation step at which the entry was written.
    pub step: u64,
    pub level: Level,
    pub msg: String,
}

struct LogState {
    entries: VecDeque<Entry>,
    next_seq: u64,
    step: u64,
}

thread_local! {
    static LOG: RefCell<LogState> = const { RefCell::new(LogState { entries: VecDeque::new(), next_seq: 0, step: 0 }) };
}

pub(crate) fn set_step(step: u64) {
    LOG.with(|l| l.borrow_mut().step = step);
}

pub fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

pub fn write(level: Level, msg: String) {
    let text = format!("[engine] {msg}");
    match level {
        Level::Debug => web_sys::console::debug_1(&text.as_str().into()),
        Level::Info => web_sys::console::log_1(&text.as_str().into()),
        Level::Warn => web_sys::console::warn_1(&text.as_str().into()),
        Level::Error => web_sys::console::error_1(&text.as_str().into()),
    }
    LOG.with(|l| {
        let mut l = l.borrow_mut();
        let seq = l.next_seq;
        l.next_seq += 1;
        let step = l.step;
        if l.entries.len() >= CAPACITY {
            l.entries.pop_front();
        }
        l.entries.push_back(Entry {
            seq,
            t: now_ms(),
            step,
            level,
            msg,
        });
    });
}

/// Seq of the oldest entry still in the ring buffer (older ones were dropped).
pub fn oldest_seq() -> u64 {
    LOG.with(|l| l.borrow().entries.front().map_or(l.borrow().next_seq, |e| e.seq))
}

/// Entries with `seq >= since` and `level >= min_level`; the newest `limit` of them.
pub fn entries_since(since: u64, min_level: Level, limit: usize) -> Vec<Entry> {
    LOG.with(|l| {
        let l = l.borrow();
        let v: Vec<Entry> = l
            .entries
            .iter()
            .filter(|e| e.seq >= since && e.level >= min_level)
            .cloned()
            .collect();
        let skip = v.len().saturating_sub(limit);
        v.into_iter().skip(skip).collect()
    })
}

pub fn next_seq() -> u64 {
    LOG.with(|l| l.borrow().next_seq)
}

#[macro_export]
macro_rules! log_info { ($($t:tt)*) => { $crate::log::write($crate::log::Level::Info, format!($($t)*)) } }
#[macro_export]
macro_rules! log_warn { ($($t:tt)*) => { $crate::log::write($crate::log::Level::Warn, format!($($t)*)) } }
#[macro_export]
macro_rules! log_error { ($($t:tt)*) => { $crate::log::write($crate::log::Level::Error, format!($($t)*)) } }
#[macro_export]
macro_rules! log_debug { ($($t:tt)*) => { $crate::log::write($crate::log::Level::Debug, format!($($t)*)) } }

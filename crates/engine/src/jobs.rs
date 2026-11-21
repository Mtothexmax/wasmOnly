//! Job system: data-parallel helpers with one API for both builds.
//!
//! * **Stable build (default):** everything runs on the main thread, in order.
//! * **`threads` feature (pinned nightly backend, `node tools/dev.mjs build --threads`):**
//!   wasm32 with `+atomics,+bulk-memory` and a *shared* `WebAssembly.Memory`
//!   (SharedArrayBuffer, needs COOP/COEP — the dev server sends them). A rayon pool runs on
//!   Web Workers via `wasm-bindgen-rayon`; Atomics (`memory.atomic.wait/notify`) synchronise it.
//!   The pool is started by the engine before `Game::init`, sized to `navigator.hardwareConcurrency`.
//!
//! Results are identical in both builds (`par_map` keeps order), so deterministic stepping and
//! AI assertions do not depend on the thread count. Small inputs (< `MIN_PARALLEL`) stay serial.

/// Below this many items the helpers run serially (worker hand-off costs more than it saves).
pub const MIN_PARALLEL: usize = 256;

#[cfg(feature = "threads")]
mod imp {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub static THREADS: AtomicUsize = AtomicUsize::new(1);

    pub fn threads() -> usize {
        THREADS.load(Ordering::Relaxed)
    }

    pub fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
        par_map_min(super::MIN_PARALLEL, items, f)
    }

    pub fn par_map_min<T: Sync, R: Send>(
        min: usize,
        items: &[T],
        f: impl Fn(&T) -> R + Sync + Send,
    ) -> Vec<R> {
        if items.len() < min.max(2) || threads() <= 1 {
            return items.iter().map(f).collect();
        }
        items.par_iter().map(f).collect()
    }

    pub fn par_for_each_mut<T: Send>(items: &mut [T], f: impl Fn(&mut T) + Sync + Send) {
        if items.len() < super::MIN_PARALLEL || threads() <= 1 {
            items.iter_mut().for_each(f);
            return;
        }
        items.par_iter_mut().for_each(f);
    }

    pub async fn init() -> Result<usize, String> {
        let n = web_sys::window()
            .map(|w| w.navigator().hardware_concurrency() as usize)
            .unwrap_or(4)
            .clamp(2, 16);
        let promise = wasm_bindgen_rayon::init_thread_pool(n);
        wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|e| format!("initThreadPool: {e:?}"))?;
        THREADS.store(n, Ordering::Relaxed);
        Ok(n)
    }
}

#[cfg(not(feature = "threads"))]
mod imp {
    pub fn threads() -> usize {
        1
    }
    pub fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
        items.iter().map(f).collect()
    }
    pub fn par_map_min<T: Sync, R: Send>(
        _min: usize,
        items: &[T],
        f: impl Fn(&T) -> R + Sync + Send,
    ) -> Vec<R> {
        items.iter().map(f).collect()
    }
    pub fn par_for_each_mut<T: Send>(items: &mut [T], f: impl Fn(&mut T) + Sync + Send) {
        items.iter_mut().for_each(f);
    }
    pub async fn init() -> Result<usize, String> {
        Ok(1)
    }
}

/// Worker threads available to the job system (1 = serial build).
pub fn threads() -> usize {
    imp::threads()
}

/// `items.iter().map(f).collect()`, in parallel when the threads backend is active. Order is kept.
pub fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    imp::par_map(items, f)
}

/// Like [`par_map`] but parallel from `min` items on — for few, expensive items (e.g. texture bands).
pub fn par_map_min<T: Sync, R: Send>(
    min: usize,
    items: &[T],
    f: impl Fn(&T) -> R + Sync + Send,
) -> Vec<R> {
    imp::par_map_min(min, items, f)
}

/// `items.iter_mut().for_each(f)`, in parallel when the threads backend is active.
pub fn par_for_each_mut<T: Send>(items: &mut [T], f: impl Fn(&mut T) + Sync + Send) {
    imp::par_for_each_mut(items, f)
}

/// True when compiled with the threads backend (shared memory + atomics).
pub const fn threads_backend() -> bool {
    cfg!(feature = "threads")
}

pub(crate) async fn init() -> Result<usize, String> {
    imp::init().await
}

/// Wall-clock budget for one frame's share of deferred work (see [`JobQueue`]).
///
/// ```ignore
/// let budget = FrameBudget::new(2.0);                // 2 ms this frame
/// while let Some(chunk) = self.build_queue.pop(&budget) {
///     self.build_chunk(ctx, chunk);                    // game data and ctx are both reachable
/// }
/// ```
pub struct FrameBudget {
    start: f64,
    ms: f64,
    taken: std::cell::Cell<u32>,
}

impl FrameBudget {
    /// Starts the clock now. `ms <= 0` = one item per frame.
    pub fn new(ms: f32) -> Self {
        Self { start: crate::log::now_ms(), ms: ms as f64, taken: std::cell::Cell::new(0) }
    }
    pub fn elapsed_ms(&self) -> f32 {
        (crate::log::now_ms() - self.start) as f32
    }
    /// True while time is left. The first item of a budget always runs (progress every frame,
    /// even when one item is larger than the whole budget).
    pub fn has_time(&self) -> bool {
        self.taken.get() == 0 || self.elapsed_ms() < self.ms as f32
    }
    /// Items handed out under this budget so far.
    pub fn taken(&self) -> u32 {
        self.taken.get()
    }
}

/// FIFO of deferred work items processed under a per-frame [`FrameBudget`] (chunk builds,
/// mesh generation, path finding, ...). The queue holds plain data (`T`), the game runs it, so a
/// job can reach both the game's own state and `Ctx`. Single-threaded and deterministic: the
/// same items run in the same order; only how many per frame depends on the machine.
/// `stats()` is JSON for a game's `debug_state`.
pub struct JobQueue<T> {
    items: std::collections::VecDeque<T>,
    /// Items finished in total.
    pub done: u64,
    /// Items run in the last frame that ran any, and the time they took (ms).
    pub last_frame_items: u32,
    pub last_frame_ms: f32,
    /// Longest single frame share so far (ms): over the budget means one item is too big.
    pub max_frame_ms: f32,
}

impl<T> Default for JobQueue<T> {
    fn default() -> Self {
        Self { items: Default::default(), done: 0, last_frame_items: 0, last_frame_ms: 0.0, max_frame_ms: 0.0 }
    }
}

impl<T> JobQueue<T> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, item: T) {
        self.items.push_back(item);
    }
    /// Runs before everything already queued (urgent work, e.g. the chunk under the player).
    pub fn push_front(&mut self, item: T) {
        self.items.push_front(item);
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    pub fn clear(&mut self) {
        self.items.clear();
    }
    /// Drops queued items that are no longer wanted (e.g. chunks that left the load radius).
    pub fn retain(&mut self, f: impl FnMut(&T) -> bool) {
        self.items.retain(f);
    }
    /// Re-orders the queue (e.g. nearest chunk first after the player moved).
    pub fn sort_by_key<K: Ord>(&mut self, f: impl FnMut(&T) -> K) {
        self.items.make_contiguous().sort_by_key(f);
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }
    /// Next item while `budget` has time left, else `None` (the rest waits for the next frame).
    pub fn pop(&mut self, budget: &FrameBudget) -> Option<T> {
        self.account(budget);
        if !budget.has_time() {
            return None;
        }
        let item = self.items.pop_front()?;
        budget.taken.set(budget.taken.get() + 1);
        self.done += 1;
        Some(item)
    }
    /// Runs items with `f` until the budget is used up; returns how many ran.
    pub fn run(&mut self, budget_ms: f32, mut f: impl FnMut(T)) -> u32 {
        let b = FrameBudget::new(budget_ms);
        let mut n = 0;
        while let Some(item) = self.pop(&b) {
            f(item);
            n += 1;
        }
        self.account(&b);
        n
    }
    // stats of the budget in use (the final `pop` of a frame sees all its items)
    fn account(&mut self, b: &FrameBudget) {
        if b.taken() == 0 {
            return;
        }
        let ms = b.elapsed_ms();
        self.last_frame_items = b.taken();
        self.last_frame_ms = ms;
        self.max_frame_ms = self.max_frame_ms.max(ms);
    }
    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({"queued": self.items.len(), "done": self.done, "last_frame_items": self.last_frame_items,
            "last_frame_ms": self.last_frame_ms, "max_frame_ms": self.max_frame_ms})
    }
}

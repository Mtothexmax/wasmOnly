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

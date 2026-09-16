//! Lazily-built, idle-evicting pool of ONNX sessions.
//!
//! Both model pools (embedder, cross-encoder) share the same shape: a
//! small fixed number of sessions that are not thread-safe, handed out
//! one caller at a time. Before this module they were built eagerly on
//! first use and lived until the process exited, so a MemoryPilot that
//! had answered one `recall` an hour ago still held ~300 MB of model
//! weights. Sessions are now built on demand and dropped again after
//! `MEMORYPILOT_MODEL_IDLE_SECS` (default 600) without a caller; the next
//! caller rebuilds them (~0.3 s for the int8 models, whose weights are
//! memory-mapped from the on-disk cache). The disk query cache in front
//! of the embedder means a repeated `recall` usually does not wake it.
//!
//! Set `MEMORYPILOT_MODEL_IDLE_SECS=0` to keep sessions resident forever
//! (the pre-4.5 behaviour), e.g. for latency benchmarks.

use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

struct State<T> {
    /// Sessions that are built and not currently handed out.
    idle: Vec<T>,
    /// Sessions currently handed out to callers.
    in_use: usize,
    /// Last time a session was returned (or the pool was created).
    last_used: Instant,
}

pub struct IdlePool<T: Send + 'static> {
    name: &'static str,
    capacity: usize,
    build: fn() -> T,
    state: Mutex<State<T>>,
    notify: Condvar,
}

/// Idle time after which every session of a pool is dropped. `0`
/// disables eviction.
pub fn idle_timeout() -> Option<Duration> {
    let secs = std::env::var("MEMORYPILOT_MODEL_IDLE_SECS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(600);
    (secs > 0).then(|| Duration::from_secs(secs))
}

impl<T: Send + 'static> IdlePool<T> {
    /// `capacity` sessions at most, each built with `build` when first
    /// needed. Must be stored in a `static` (the eviction thread and the
    /// guards hold a `&'static` to it).
    pub fn new(name: &'static str, capacity: usize, build: fn() -> T) -> Self {
        Self {
            name,
            capacity: capacity.max(1),
            build,
            state: Mutex::new(State {
                idle: Vec::with_capacity(capacity),
                in_use: 0,
                last_used: Instant::now(),
            }),
            notify: Condvar::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Start the background evictor for this pool. Call once, right after
    /// the pool is first created; a no-op when eviction is disabled.
    pub fn spawn_evictor(&'static self) {
        let Some(timeout) = idle_timeout() else {
            return;
        };
        let poll = timeout.min(Duration::from_secs(30)).max(Duration::from_secs(1));
        let _ = std::thread::Builder::new()
            .name(format!("mp-{}-evict", self.name))
            .spawn(move || loop {
                std::thread::sleep(poll);
                self.evict_if_idle(timeout);
            });
    }

    fn evict_if_idle(&self, timeout: Duration) {
        let dropped = {
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            if state.in_use > 0 || state.idle.is_empty() || state.last_used.elapsed() < timeout {
                return;
            }
            std::mem::take(&mut state.idle)
        };
        let count = dropped.len();
        // Drop outside the lock: tearing down an ONNX session is not free.
        drop(dropped);
        crate::embedding::release_freed_memory();
        eprintln!(
            "[MemoryPilot] {} idle for {}s — released {} session{} to free memory; the next call rebuilds it.",
            self.name,
            timeout.as_secs(),
            count,
            if count == 1 { "" } else { "s" }
        );
    }

    /// Hand out one session, building it if the pool has spare capacity
    /// and nothing idle, otherwise waiting for one to come back.
    pub fn acquire(&'static self) -> PoolGuard<T> {
        let mut state = self.state.lock().expect("model pool poisoned");
        loop {
            if let Some(session) = state.idle.pop() {
                state.in_use += 1;
                return PoolGuard {
                    pool: self,
                    session: Some(session),
                };
            }
            if state.in_use < self.capacity {
                // Reserve the slot, then build without holding the lock so
                // other callers can keep using sessions that already exist.
                state.in_use += 1;
                drop(state);
                let session = (self.build)();
                return PoolGuard {
                    pool: self,
                    session: Some(session),
                };
            }
            state = self.notify.wait(state).expect("model pool wait poisoned");
        }
    }

    /// Number of sessions currently built (idle or handed out).
    #[cfg(test)]
    pub fn loaded(&self) -> usize {
        let state = self.state.lock().expect("model pool poisoned");
        state.idle.len() + state.in_use
    }
}

pub struct PoolGuard<T: Send + 'static> {
    pool: &'static IdlePool<T>,
    session: Option<T>,
}

impl<T: Send + 'static> PoolGuard<T> {
    pub fn get(&mut self) -> &mut T {
        self.session.as_mut().expect("pool guard already released")
    }
}

impl<T: Send + 'static> Drop for PoolGuard<T> {
    fn drop(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };
        if let Ok(mut state) = self.pool.state.lock() {
            state.idle.push(session);
            state.in_use = state.in_use.saturating_sub(1);
            state.last_used = Instant::now();
            self.pool.notify.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static BUILDS: AtomicUsize = AtomicUsize::new(0);
    fn build_counter() -> usize {
        BUILDS.fetch_add(1, Ordering::SeqCst)
    }
    static POOL: std::sync::OnceLock<IdlePool<usize>> = std::sync::OnceLock::new();
    fn pool() -> &'static IdlePool<usize> {
        POOL.get_or_init(|| IdlePool::new("test", 2, build_counter))
    }

    #[test]
    fn builds_lazily_reuses_and_evicts() {
        let pool = pool();
        assert_eq!(pool.loaded(), 0, "nothing built before first acquire");
        {
            let mut a = pool.acquire();
            let mut b = pool.acquire();
            assert_ne!(*a.get(), *b.get(), "two callers get two sessions");
            assert_eq!(pool.loaded(), 2);
        }
        let before = BUILDS.load(Ordering::SeqCst);
        drop(pool.acquire());
        assert_eq!(BUILDS.load(Ordering::SeqCst), before, "idle session reused, not rebuilt");

        std::thread::sleep(Duration::from_millis(20));
        pool.evict_if_idle(Duration::from_millis(10));
        assert_eq!(pool.loaded(), 0, "idle sessions dropped after the timeout");

        let _held = pool.acquire();
        pool.evict_if_idle(Duration::ZERO);
        assert_eq!(pool.loaded(), 1, "a session in use is never evicted");
    }
}

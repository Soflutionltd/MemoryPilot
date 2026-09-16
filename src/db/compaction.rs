use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use super::Database;

static LAST_COMPACT: OnceLock<Mutex<std::time::Instant>> = OnceLock::new();
static LAST_ROLLUP: OnceLock<Mutex<std::time::Instant>> = OnceLock::new();
/// One rollup thread at a time.
static ROLLUP_RUNNING: AtomicBool = AtomicBool::new(false);
/// Re-entrancy guard: GC and capsule compaction write their merged rows
/// through `add_memory`, which calls back into `maybe_auto_compact`.
static COMPACTING: AtomicBool = AtomicBool::new(false);

/// How often the hierarchical episodic rollup is allowed to run on the
/// write path. The rollup is idempotent (INSERT OR IGNORE by content
/// hash), so this only bounds the cost of the *first* pass after new
/// memories land; subsequent passes are near-free.
const ROLLUP_INTERVAL_SECS: u64 = 600;
/// Don't bother building episodes until there is enough raw material for
/// at least a couple of interval rollups.
const ROLLUP_MIN_MEMORIES: i64 = 8;

/// Rewind the compaction debounce so the next `add_memory` runs GC +
/// capsule compaction immediately. Test-only: the clock is process-wide.
#[cfg(test)]
pub(super) fn arm_compaction_for_tests() {
    let last = LAST_COMPACT.get_or_init(|| Mutex::new(std::time::Instant::now()));
    if let Ok(mut timestamp) = last.lock() {
        *timestamp = std::time::Instant::now() - std::time::Duration::from_secs(600);
    }
    COMPACTING.store(false, Ordering::SeqCst);
}

pub(super) fn maybe_auto_compact(db: &Database) {
    maybe_roll_up_episodes(db);
    maybe_compact(db);
    db.maybe_consolidate();
}

/// Throttled, best-effort hierarchical episodic rollup. Runs at most once
/// per `ROLLUP_INTERVAL_SECS`, on its own thread with its own database
/// handle, and is a no-op when `MEMORYPILOT_EPISODIC=off`.
///
/// It must not run on the write path: every new episode costs one
/// embedding (~0.2 s), and a corpus with months of history has hundreds
/// of hourly buckets to materialise the first time — that used to stall
/// `add_memory` for minutes (and, before v4.5, crash the server on French
/// text). Now `add_memory` only kicks the thread and returns.
fn maybe_roll_up_episodes(db: &Database) {
    if std::env::var("MEMORYPILOT_EPISODIC")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "off" | "0" | "false"))
        .unwrap_or(false)
    {
        return;
    }

    let last = LAST_ROLLUP.get_or_init(|| {
        Mutex::new(std::time::Instant::now() - std::time::Duration::from_secs(ROLLUP_INTERVAL_SECS))
    });
    let Ok(mut timestamp) = last.lock() else {
        return;
    };
    if timestamp.elapsed() < std::time::Duration::from_secs(ROLLUP_INTERVAL_SECS) {
        return;
    }

    let memory_count: i64 = db
        .conn
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap_or(0);
    if memory_count < ROLLUP_MIN_MEMORIES {
        return;
    }

    *timestamp = std::time::Instant::now();
    drop(timestamp);

    let Some(path) = super::embed_db_path().map(|path| path.to_path_buf()) else {
        return;
    };
    if ROLLUP_RUNNING.swap(true, Ordering::SeqCst) {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("mp-episodic-rollup".into())
        .spawn(move || {
            match Database::open_maintenance(&path).and_then(|maintenance| maintenance.roll_up_episodes()) {
                Ok(report) => {
                    let created = report.interval_created + report.daily_created + report.longterm_created;
                    if created > 0 {
                        eprintln!(
                            "[MemoryPilot] Episodic rollup: {} interval, {} daily, {} long-term episodes built from {} memories.",
                            report.interval_created, report.daily_created, report.longterm_created, report.memories_scanned
                        );
                    }
                }
                Err(error) => eprintln!("[MemoryPilot] episodic rollup skipped: {}", error),
            }
            ROLLUP_RUNNING.store(false, Ordering::SeqCst);
        });
    if spawned.is_err() {
        ROLLUP_RUNNING.store(false, Ordering::SeqCst);
    }
}

fn maybe_compact(db: &Database) {
    if COMPACTING.load(Ordering::SeqCst) {
        return;
    }
    let last = LAST_COMPACT.get_or_init(|| {
        Mutex::new(std::time::Instant::now() - std::time::Duration::from_secs(600))
    });

    {
        // Decide and stamp under the lock, then release it: `run_gc` and
        // `compact_to_capsules` insert their merged rows via `add_memory`,
        // which re-enters this function — holding a std `Mutex` across
        // that call deadlocked the server (hit in v4.5.0 once the
        // episodic rollup stopped panicking ahead of it).
        let Ok(mut timestamp) = last.lock() else {
            return;
        };
        if timestamp.elapsed() < std::time::Duration::from_secs(300) {
            return;
        }
        *timestamp = std::time::Instant::now();
    }

    let memory_count: i64 = db
        .conn
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap_or(0);
    let threshold = crate::gc::AUTO_COMPACT_THRESHOLD;
    if (memory_count as usize) < threshold {
        return;
    }

    if COMPACTING.swap(true, Ordering::SeqCst) {
        return;
    }
    let config = crate::gc::GcConfig::default();
    let gc_report = db.run_gc(&config, false).ok();

    let still_high = memory_count as usize >= threshold + (threshold / 2);
    let gc_did_little = gc_report
        .as_ref()
        .map(|report| report.memories_compressed == 0 && report.expired_removed == 0)
        .unwrap_or(true);

    if still_high || gc_did_little {
        let _ = db.compact_to_capsules(14, 2);
    }
    COMPACTING.store(false, Ordering::SeqCst);
}

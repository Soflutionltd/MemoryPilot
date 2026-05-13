use std::sync::{Mutex, OnceLock};

use super::Database;

static LAST_COMPACT: OnceLock<Mutex<std::time::Instant>> = OnceLock::new();

pub(super) fn maybe_auto_compact(db: &Database) {
    let last = LAST_COMPACT.get_or_init(|| {
        Mutex::new(std::time::Instant::now() - std::time::Duration::from_secs(600))
    });

    let Ok(mut timestamp) = last.lock() else {
        return;
    };
    if timestamp.elapsed() < std::time::Duration::from_secs(300) {
        return;
    }

    let memory_count: i64 = db
        .conn
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap_or(0);
    let threshold = crate::gc::AUTO_COMPACT_THRESHOLD;
    if (memory_count as usize) < threshold {
        return;
    }

    *timestamp = std::time::Instant::now();

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
}

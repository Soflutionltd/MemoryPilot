use rusqlite::Connection;
use std::path::Path;
use std::sync::Mutex;

/// How long a connection waits for another one to finish switching a new
/// database to WAL.
const WAL_SWITCH_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

pub(super) fn configure_connection(conn: &Connection) -> Result<(), String> {
    // Switching a new database to WAL needs it exclusively, and SQLite
    // answers "locked" at once, without the busy handler, when another
    // connection is opening the same file: a server opens several at start.
    let started = std::time::Instant::now();
    loop {
        match conn.execute_batch("PRAGMA journal_mode = WAL;") {
            Ok(()) => break,
            Err(error) if is_locked(&error) && started.elapsed() < WAL_SWITCH_WAIT => {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(error) => return Err(format!("Pragma: {}", error)),
        }
    }
    conn.execute_batch(
        "
        PRAGMA synchronous = NORMAL;
        PRAGMA cache_size = -8000;
        PRAGMA foreign_keys = ON;
    ",
    )
    .map_err(|error| format!("Pragma: {}", error))
}

fn is_locked(error: &rusqlite::Error) -> bool {
    matches!(
        error.sqlite_error_code(),
        Some(rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked)
    )
}

pub(super) fn open_read_pool(path: &Path, size: usize) -> Result<Vec<Mutex<Connection>>, String> {
    let mut read_pool = Vec::with_capacity(size);
    for _ in 0..size {
        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|error| format!("Read pool open: {}", error))?;
        let _ = conn.execute_batch("PRAGMA cache_size = -4000;");
        read_pool.push(Mutex::new(conn));
    }
    Ok(read_pool)
}

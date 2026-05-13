use rusqlite::Connection;
use std::path::Path;
use std::sync::Mutex;

pub(super) fn configure_connection(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA cache_size = -8000;
        PRAGMA foreign_keys = ON;
    ",
    )
    .map_err(|error| format!("Pragma: {}", error))
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

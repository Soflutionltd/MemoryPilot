//! Storage contract for memory rows.
//!
//! `Database` keeps the logic (dedup, vault sealing, embedding jobs, links,
//! scope); a `MemoryStore` only persists rows and keeps its own indexes in
//! step (full-text entry, vector index). SQLite implements it in
//! `sqlite_store.rs`; the server mode's Postgres store implements the same
//! contract.

use super::Memory;

pub trait MemoryStore {
    /// Stores a new memory and its full-text entry. `content_hash` is the
    /// hash of `memory.content`; the embedding is computed later.
    fn insert_memory(&self, memory: &Memory, content_hash: &str) -> Result<(), String>;

    /// One memory as tools see it (credentials masked).
    fn fetch_memory(&self, id: &str) -> Result<Option<Memory>, String>;

    /// The stored content of a memory, sealed label included for a
    /// credential (never the masked copy).
    fn stored_content(&self, id: &str) -> Result<String, String>;

    /// Rewrites the mutable fields of an existing memory (content, kind,
    /// tags, importance, expiry, metadata, updated_at) and its full-text
    /// entry. With `content_changed`, the stored embedding is dropped and
    /// `content_hash` replaced.
    fn write_memory(&self, memory: &Memory, content_hash: &str, content_changed: bool) -> Result<(), String>;

    /// Deletes a memory and its index entries. False when it did not exist.
    fn remove_memory(&self, id: &str) -> Result<bool, String>;

    /// Memories filtered by project and kind, newest update first, with the
    /// total count before paging.
    fn list_memory_rows(
        &self,
        project: Option<&str>,
        kind: Option<&str>,
        exclude_kind: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Memory>, i64), String>;

    /// A memory of the same project (`None`: no project) whose content has
    /// this hash.
    fn find_by_content_hash(&self, content_hash: &str, project: Option<&str>) -> Option<Memory>;

    /// The most recently updated memories of a project (`None`: no project).
    fn recent_memories(&self, project: Option<&str>, limit: usize) -> Result<Vec<Memory>, String>;
}

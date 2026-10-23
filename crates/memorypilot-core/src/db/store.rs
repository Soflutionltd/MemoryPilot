//! Storage contract for memory rows.
//!
//! `Database` keeps the logic (dedup, vault sealing, embedding jobs, links,
//! scope); a `MemoryStore` only persists rows and keeps its own indexes in
//! step (full-text entry, vector index), a `GraphStore` the entities and
//! links. SQLite implements it in
//! `sqlite_store.rs`; the server mode's Postgres store implements the same
//! contract.

use super::Memory;
use std::collections::{HashMap, HashSet};

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

/// What the write path keeps beside the rows: project names, the entities
/// extracted from each memory and the links derived from shared entities.
/// The extraction and linking rules stay in `Database`.
pub trait GraphStore {
    /// Registers a project name (already canonical) when it is unknown.
    fn ensure_project(&self, name: &str) -> Result<(), String>;

    /// Replaces the (kind, value) entities of a memory the store may change.
    fn replace_entities(&self, memory_id: &str, entities: &[(String, String)]) -> Result<(), String>;

    /// (id, kind) of other memories carrying this exact entity, at most
    /// `limit`: in `project` or without a project (any project when
    /// `any_project`); with no `project`, only memories without one.
    fn memories_with_entity(
        &self,
        kind: &str,
        value: &str,
        exclude_id: &str,
        project: Option<&str>,
        any_project: bool,
        limit: usize,
    ) -> Result<Vec<(String, String)>, String>;

    /// Replaces the links of a memory, in both directions, with these
    /// (source, target, relation) links.
    fn replace_links(&self, memory_id: &str, links: &[(String, String, String)]) -> Result<(), String>;
}

/// What search, ranking and recall read. The pipeline (fusion, boosts,
/// reranking, budgets) stays in `Database`; a store answers these queries
/// with its own indexes (FTS5 and an in-process ANN for SQLite, tsvector and
/// pgvector for Postgres).
pub trait RecallStore {
    /// Memories matching a full-text query in FTS5 syntax (as built by
    /// `fts::fts5_query_variants`: quoted prefix terms, a phrase or `NEAR`),
    /// best match first, at most `limit`. Other stores translate it.
    fn full_text_candidates(
        &self,
        fts_query: &str,
        project: Option<&str>,
        kind: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Memory>, String>;

    /// Memories of the project and kind (restricted to `only_ids` when
    /// given), each with its similarity to `query_embedding`, or `None`
    /// while its embedding is not computed yet.
    fn vector_candidates(
        &self,
        query_embedding: &[f32],
        project: Option<&str>,
        kind: Option<&str>,
        only_ids: Option<&[String]>,
    ) -> Result<Vec<(Memory, Option<f32>)>, String>;

    /// Nearest memory ids from an approximate index, best first. Empty when
    /// the store has none.
    fn approximate_neighbours(&self, query_embedding: &[f32], limit: usize) -> Vec<(String, f32)>;

    /// Number of vectors in the approximate index (0 without one). A store
    /// may stop counting at [`super::ANN_BYPASS_THRESHOLD`], all search needs.
    fn approximate_index_len(&self) -> usize;

    /// (target id, relation) of the links pointing at these memories; every
    /// link with `None`.
    fn link_relations(&self, target_ids: Option<&[&String]>) -> Vec<(String, String)>;

    /// Terms related to a lowercased word: objects of current triples
    /// whose subject it is (up to 3), subjects of those whose object it is
    /// (up to 3), then entities that co-occur with it (up to 4).
    fn related_terms(&self, word: &str) -> Vec<String>;

    /// Pairs of these memories connected by a link or a shared non-date entity.
    fn adjacency_pairs(&self, ids: &[String]) -> HashSet<(String, String)>;

    /// Per memory: (current, closed) knowledge triples it is the source of.
    fn triple_counts(&self, ids: &[&String]) -> HashMap<String, (i64, i64)>;

    /// Stored vectors of these memories, decoded; memories without one
    /// (or with another dimension) are absent.
    fn memory_vectors(&self, ids: &[&String]) -> HashMap<String, Vec<f32>>;

    /// Raw `date` entity values of these memories.
    fn memory_date_values(&self, ids: &[&String]) -> HashMap<String, Vec<String>>;

    /// These memory ids plus those reachable within `depth` link hops
    /// (either direction).
    fn graph_neighbours(&self, ids: &[String], depth: u32) -> Result<HashSet<String>, String>;

    /// Root path recorded for a project.
    fn project_path(&self, project: &str) -> Option<String>;

    /// Distinct entity values of these kinds in a project's memories.
    fn project_entities(&self, project: &str, kinds: &[&str], limit: usize) -> Vec<String>;

    /// Contents (credentials masked) of a project's memories updated in the
    /// last `days` days, newest first.
    fn recent_project_contents(&self, project: &str, days: u32, limit: usize) -> Vec<String>;

    /// Unexpired memories of importance `min_importance` or more, most
    /// important then most recent first.
    fn important_memories(&self, min_importance: i32, limit: usize) -> Result<Vec<Memory>, String>;

    /// Memories tagged pinned, most important then most recent first.
    fn pinned_memories(&self) -> Result<Vec<Memory>, String>;

    /// Number of memories, of one kind when given.
    fn memory_count(&self, kind: Option<&str>) -> i64;

    /// Number of known projects.
    fn project_count(&self) -> i64;

    /// Drops expired memories before a search. A store may throttle it.
    fn expire_memories(&self);

    /// Counts one access for each memory a search returned. A store may
    /// defer the write.
    fn record_access(&self, ids: &[String]);
}

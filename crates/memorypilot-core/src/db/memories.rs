//! Memory rows: add (with dedup), update, delete, get, list, bulk import, v1 migration.

use super::*;

impl Database {
    // ─── DEDUP ────────────────────────────────────────

    /// Normalize text for comparison: lowercase, collapse whitespace, strip punctuation.
    pub(super) fn normalize(text: &str) -> String {
        text.to_lowercase()
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == ' ' {
                    c
                } else {
                    ' '
                }
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Jaccard similarity between two normalized strings (word-level).
    pub(super) fn similarity(a: &str, b: &str) -> f64 {
        let a_words: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let b_words: std::collections::HashSet<&str> = b.split_whitespace().collect();
        if a_words.is_empty() && b_words.is_empty() {
            return 1.0;
        }
        let intersection = a_words.intersection(&b_words).count() as f64;
        let union = a_words.union(&b_words).count() as f64;
        if union == 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    // ─── CRUD ────────────────────────────────────────

    /// Add memory with dedup check. Returns (memory, was_merged).
    pub fn add_memory(
        &self,
        content: &str,
        kind: &str,
        project: Option<&str>,
        tags: &[String],
        source: &str,
        importance: i32,
        expires_at: Option<&str>,
        metadata: Option<&serde_json::Value>,
        scope: &MemoryScope,
    ) -> Result<(Memory, bool), String> {
        self.add_memory_with_id(
            None, content, kind, project, tags, source, importance, expires_at, metadata, scope,
        )
    }

    /// Same as [`Self::add_memory`] but lets the caller pin the row's
    /// primary key. Used by benchmarks (and only by benchmarks) so that
    /// the memory id is reproducible across runs — without it, the
    /// random UUID drives the deterministic id-based tie-break we now
    /// apply in `search`, which produced visible run-to-run variance
    /// on `--benchmark-fr`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_memory_with_id(
        &self,
        explicit_id: Option<&str>,
        content: &str,
        kind: &str,
        project: Option<&str>,
        tags: &[String],
        source: &str,
        importance: i32,
        expires_at: Option<&str>,
        metadata: Option<&serde_json::Value>,
        scope: &MemoryScope,
    ) -> Result<(Memory, bool), String> {
        let outcome = add_memory_to(
            self, explicit_id, content, kind, project, tags, source, importance, expires_at, metadata, scope,
        )?;
        let memory = self.after_write(&outcome)?;
        if !outcome.merged {
            // Auto-compaction: trigger GC when memory count exceeds threshold
            self.maybe_auto_compact();
        }
        Ok((memory, outcome.merged))
    }

    /// What the local store schedules after a write: the embedding of new
    /// content, and sealing a credential now (the server runs for days, a
    /// credential must not wait for a restart in clear). Returns the memory
    /// as stored, sealing included.
    fn after_write(&self, outcome: &WriteOutcome) -> Result<Memory, String> {
        let memory = &outcome.memory;
        // A credential is sealed by the store before it is written: embed what
        // was stored (the masked label), never the secret.
        let stored_content = if memory.kind == "credential" {
            self.stored_content(&memory.id)?
        } else {
            memory.content.clone()
        };
        if outcome.content_changed {
            queue_embedding_job(
                &memory.id,
                &stored_content,
                &Self::embedding_input(&memory.kind, memory.project.as_deref(), &memory.created_at, &stored_content),
            );
        }
        if memory.kind == "credential" {
            // Rows written by older versions, or with the vault off then on.
            self.seal_credentials()?;
            return self
                .get_memory(&memory.id)?
                .ok_or_else(|| format!("Memory {} vanished while being sealed", memory.id));
        }
        Ok(memory.clone())
    }

    /// Trigger auto-compaction if memory count exceeds threshold.
    /// Debounced: runs max once per 5 minutes.
    pub(super) fn maybe_auto_compact(&self) {
        compaction::maybe_auto_compact(self);
    }

    /// Full update with all fields.
    pub fn update_memory_full(
        &self,
        id: &str,
        content: Option<&str>,
        kind: Option<&str>,
        tags: Option<&[String]>,
        importance: Option<i32>,
        expires_at: Option<&str>,
        metadata: Option<&serde_json::Value>,
    ) -> Result<Option<Memory>, String> {
        let Some(outcome) = update_memory_in(self, id, content, kind, tags, importance, expires_at, metadata)? else {
            return Ok(None);
        };
        Ok(Some(self.after_write(&outcome)?))
    }

    pub fn delete_memory(&self, id: &str) -> Result<bool, String> {
        self.remove_memory(id)
    }

    pub fn get_memory(&self, id: &str) -> Result<Option<Memory>, String> {
        self.fetch_memory(id)
    }

    // ─── BULK ADD ─────────────────────────────────────

    /// Add multiple memories in one call, with dedup per item. Returns (added, merged, skipped).
    pub fn add_memories_bulk(
        &self,
        items: &[BulkItem],
    ) -> Result<(Vec<Memory>, usize, usize), String> {
        let mut added: Vec<Memory> = Vec::new();
        let mut merged = 0usize;
        let mut skipped = 0usize;
        for item in items {
            if item.content.trim().is_empty() {
                skipped += 1;
                continue;
            }
            let tags: Vec<String> = item.tags.clone().unwrap_or_default();
            let imp = item.importance.unwrap_or(3);
            let exp = item.expires_at.as_deref();
            match self.add_memory(
                &item.content,
                &item.kind,
                item.project.as_deref(),
                &tags,
                &item.source,
                imp,
                exp,
                item.metadata.as_ref(),
                &item.scope(),
            ) {
                Ok((mem, was_merged)) => {
                    if was_merged {
                        merged += 1;
                    } else {
                        added.push(mem);
                    }
                }
                Err(_) => {
                    skipped += 1;
                }
            }
        }
        Ok((added, merged, skipped))
    }

    // ─── LIST ─────────────────────────────────────────

    pub fn list_memories(
        &self,
        project: Option<&str>,
        kind: Option<&str>,
        exclude_kind: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<Memory>, i64), String> {
        let _ = self.cleanup_expired();
        let canonical_project = Self::canonical_project(project);
        self.list_memory_rows(canonical_project.as_deref(), kind, exclude_kind, limit, offset)
    }

    // ─── IMPORT / MIGRATE ─────────────────────────────

    pub fn import_batch(
        &self,
        memories: &[(String, String, Option<String>, Vec<String>, String)],
    ) -> Result<usize, String> {
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| format!("Tx: {}", e))?;
        let mut count = 0;
        for (content, kind, project, tags, source) in memories {
            let exists: bool = tx
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM memories WHERE content=?1)",
                    params![content],
                    |r| r.get(0),
                )
                .unwrap_or(false);
            if exists {
                continue;
            }
            let canonical_project = project.as_deref().and_then(Self::canonical_project_name);
            let id = Uuid::new_v4().to_string();
            let now = Utc::now().to_rfc3339();
            let tags_json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
            let emb = crate::embedding::embed_text(content);
            let emb_blob = crate::embedding::vec_to_blob(&emb);
            tx.execute(
                "INSERT INTO memories (id,content,kind,project,tags,source,importance,embedding,created_at,updated_at,access_count) VALUES (?1,?2,?3,?4,?5,?6,3,?7,?8,?9,0)",
                params![id, content, kind, canonical_project.as_deref(), tags_json, source, emb_blob, now, now],
            ).map_err(|e| format!("Import: {}", e))?;
            let rowid = tx.last_insert_rowid();
            let fts_content = Self::fts_index_content(content);
            tx.execute(
                "INSERT INTO memories_fts (rowid,content,tags,kind,project) VALUES (?1,?2,?3,?4,?5)",
                params![rowid, fts_content, tags_json, kind, canonical_project.as_deref().unwrap_or("")],
            ).map_err(|e| format!("FTS: {}", e))?;
            if let Some(p) = canonical_project.as_deref() {
                let _ = tx.execute(
                    "INSERT OR IGNORE INTO projects (name,path,created_at) VALUES (?1,'',?2)",
                    params![p, now],
                );
            }
            count += 1;
        }
        tx.commit().map_err(|e| format!("Commit: {}", e))?;
        Ok(count)
    }

    pub fn migrate_from_v1(&self) -> Result<usize, String> {
        let v1_dir = dirs::home_dir().ok_or("No home")?.join(DB_DIR);
        let mut batch: Vec<(String, String, Option<String>, Vec<String>, String)> = Vec::new();

        // Load global.json
        let global_path = v1_dir.join("global.json");
        if global_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&global_path) {
                if let Ok(store) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(memories) = store.get("memories").and_then(|v| v.as_array()) {
                        for m in memories {
                            parse_v1_memory(m, None, &mut batch);
                        }
                    }
                }
            }
        }
        // Load projects/*.json
        let projects_dir = v1_dir.join("projects");
        if projects_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&projects_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) != Some("json") {
                        continue;
                    }
                    let proj_name = path
                        .file_stem()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        if let Ok(store) = serde_json::from_str::<serde_json::Value>(&content) {
                            if let Some(memories) = store.get("memories").and_then(|v| v.as_array())
                            {
                                for m in memories {
                                    parse_v1_memory(m, Some(proj_name.clone()), &mut batch);
                                }
                            }
                        }
                    }
                }
            }
        }
        self.import_batch(&batch)
    }
}

/// What a write did, so the store's owner can schedule what follows it
/// (embedding, vault sealing, compaction).
#[derive(Debug, Clone)]
pub struct WriteOutcome {
    pub memory: Memory,
    /// Folded into an existing near-duplicate instead of inserted.
    pub merged: bool,
    /// The stored content changed: its embedding must be computed again.
    pub content_changed: bool,
}

/// A near-duplicate in the same project: same content hash, or a Jaccard
/// match among the 200 most recent memories.
fn find_duplicate<S: MemoryStore + ?Sized>(
    store: &S,
    content: &str,
    project: Option<&str>,
) -> Result<Option<Memory>, String> {
    if let Some(mem) = store.find_by_content_hash(&content_hash(content), project) {
        return Ok(Some(mem));
    }
    let norm = Database::normalize(content);
    for mem in store.recent_memories(project, 200)? {
        let mem_norm = Database::normalize(&mem.content);
        if Database::similarity(&norm, &mem_norm) >= DEDUP_THRESHOLD {
            return Ok(Some(mem));
        }
    }
    Ok(None)
}

/// How many replaced texts a memory keeps in `metadata.previous_versions`.
const MAX_PREVIOUS_VERSIONS: usize = 5;

/// `metadata` with the existing text appended to `previous_versions` (newest last).
fn with_previous_version(metadata: Option<serde_json::Value>, existing: &Memory) -> serde_json::Value {
    let mut meta = match metadata {
        Some(value @ serde_json::Value::Object(_)) => value,
        Some(other) => serde_json::json!({ "previous_metadata": other }),
        None => serde_json::json!({}),
    };
    let mut versions = meta
        .get("previous_versions")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    versions.push(serde_json::json!({ "content": existing.content, "until": Utc::now().to_rfc3339() }));
    let excess = versions.len().saturating_sub(MAX_PREVIOUS_VERSIONS);
    versions.drain(..excess);
    meta["previous_versions"] = serde_json::Value::Array(versions);
    meta
}

/// Links are derived data: a failure leaves the memory stored, unlinked
/// until its next write, and is printed.
fn report_link_failure(rebuilt: Result<(), String>, memory_id: &str) {
    if let Err(error) = rebuilt {
        eprintln!("[MemoryPilot] Links of memory {} not rebuilt: {}", memory_id, error);
    }
}

/// Adds a memory to any store: a secret value forces the credential kind,
/// a near-duplicate is merged into instead (latest content unless it adds
/// nothing, highest importance, union of tags), a new memory gets its project, entities and
/// links. Embedding and sealing are left to the caller ([`WriteOutcome`]).
#[allow(clippy::too_many_arguments)]
pub fn add_memory_to<S: MemoryStore + GraphStore + ?Sized>(
    store: &S,
    explicit_id: Option<&str>,
    content: &str,
    kind: &str,
    project: Option<&str>,
    tags: &[String],
    source: &str,
    importance: i32,
    expires_at: Option<&str>,
    metadata: Option<&serde_json::Value>,
    scope: &MemoryScope,
) -> Result<WriteOutcome, String> {
    // A secret value is sealed and hidden from safe recall whatever kind the caller chose.
    let forced_credential = kind != "credential" && vault::contains_secret_value(content);
    let kind = if forced_credential { "credential" } else { kind };
    let canonical_project = Database::canonical_project(project);
    let scoped_metadata = Database::apply_scope_to_metadata(metadata, scope);
    if let Some(existing) = find_duplicate(store, content, canonical_project.as_deref())? {
        // Merge. The new text replaces the old one unless it adds nothing to
        // it (all its words already there): a near-duplicate that changes a
        // figure, a date or a name ("30 jours" -> "60 jours") is a correction,
        // and the latest statement wins. The replaced text is kept in
        // `metadata.previous_versions`, so nothing is lost.
        // A credential is read masked: keeping it means not rewriting it.
        let new_norm = Database::normalize(content);
        let old_norm = Database::normalize(&existing.content);
        let adds_nothing = {
            let old_words: std::collections::HashSet<&str> = old_norm.split_whitespace().collect();
            new_norm.split_whitespace().all(|w| old_words.contains(w))
        };
        let replaces = existing.kind != "credential" && content != existing.content && !adds_nothing;
        let new_content = if replaces {
            Some(content)
        } else if existing.kind == "credential" {
            None
        } else {
            Some(existing.content.as_str())
        };
        let new_importance = importance.max(existing.importance);
        let mut merged_tags: Vec<String> = existing.tags.clone();
        for t in tags {
            if !merged_tags.contains(t) {
                merged_tags.push(t.clone());
            }
        }
        let mut merged_metadata = Database::merge_metadata(existing.metadata.as_ref(), scoped_metadata.as_ref());
        if replaces {
            merged_metadata = Some(with_previous_version(merged_metadata, &existing));
        }
        let updated = update_memory_in(
            store,
            &existing.id,
            new_content,
            forced_credential.then_some("credential"),
            Some(&merged_tags),
            Some(new_importance),
            expires_at,
            merged_metadata.as_ref(),
        )?;
        return Ok(match updated {
            Some(outcome) => WriteOutcome { merged: true, ..outcome },
            None => WriteOutcome { memory: existing, merged: true, content_changed: false },
        });
    }

    let id = explicit_id
        .map(|value| value.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let now = Utc::now().to_rfc3339();
    let mem = Memory {
        id,
        content: content.into(),
        kind: kind.into(),
        project: canonical_project,
        tags: tags.to_vec(),
        source: source.into(),
        importance: importance.clamp(1, 5),
        expires_at: expires_at.map(String::from),
        created_at: now.clone(),
        updated_at: now,
        metadata: scoped_metadata,
        last_accessed_at: None,
        access_count: 0,
    };
    store.insert_memory(&mem, &content_hash(content))?;
    if let Some(proj) = mem.project.as_deref() {
        if let Err(error) = store.ensure_project(proj) {
            eprintln!("[MemoryPilot] Project {} not registered: {}", proj, error);
        }
    }
    report_link_failure(rebuild_memory_links(store, &mem), &mem.id);
    Ok(WriteOutcome { memory: mem, merged: false, content_changed: true })
}

/// Rewrites the given fields of a memory in any store, guarding a
/// credential's secret against its masked copy; `None` when the memory
/// does not exist. Embedding and sealing are left to the caller.
#[allow(clippy::too_many_arguments)]
pub fn update_memory_in<S: MemoryStore + GraphStore + ?Sized>(
    store: &S,
    id: &str,
    content: Option<&str>,
    kind: Option<&str>,
    tags: Option<&[String]>,
    importance: Option<i32>,
    expires_at: Option<&str>,
    metadata: Option<&serde_json::Value>,
) -> Result<Option<WriteOutcome>, String> {
    let mut existing = match store.fetch_memory(id)? {
        Some(m) => m,
        None => return Ok(None),
    };
    if existing.kind == "credential" {
        // Reads are masked: work on what is stored, never on the masked copy.
        existing.content = store.stored_content(id)?;
    }
    let sealed = existing.kind == "credential" && existing.content.starts_with(vault::SEAL_PREFIX);
    if let (Some(text), true) = (content, existing.kind == "credential") {
        let masked = text.contains("••••");
        // Sealed: a 🔒 label edit keeps the ciphertext. Not sealed: the stored text is the secret.
        let loses_secret = if sealed {
            masked && !text.starts_with(vault::SEAL_PREFIX)
        } else {
            masked || text.starts_with(vault::SEAL_PREFIX)
        };
        if loses_secret {
            return Err(format!(
                "Memory {} is a credential and this text is its masked copy (••••): saving it would replace the secret with the mask. Send the full text with the real values{}.",
                id,
                if sealed { ", or keep the 🔒 prefix to edit the label only" } else { "" }
            ));
        }
    }
    let now = Utc::now().to_rfc3339();
    let requested_kind = kind.unwrap_or(&existing.kind);
    if existing.kind == "credential" && requested_kind != "credential" && content.is_none() {
        // Reads would then return the value: that is a reveal without confirmation.
        return Err(format!(
            "Memory {} is a credential: changing its kind alone would expose its value. Send the new text with it (reveal_secret first if needed, the user confirms).",
            id
        ));
    }
    let new_content = content.unwrap_or(existing.content.as_str());
    let new_kind = if requested_kind != "credential" && vault::contains_secret_value(new_content) {
        "credential"
    } else {
        requested_kind
    };
    let new_tags = tags
        .map(|t| t.to_vec())
        .unwrap_or_else(|| existing.tags.clone());
    let new_imp = importance.unwrap_or(existing.importance).clamp(1, 5);
    let new_exp = if expires_at.is_some() {
        expires_at.map(String::from)
    } else {
        existing.expires_at.clone()
    };
    let mut new_metadata = metadata.cloned().or_else(|| existing.metadata.clone());
    if new_kind == "credential" && metadata.is_some() {
        // Replacing the metadata must not drop the ciphertext of the secret.
        if let Some(old) = existing.metadata.as_ref().and_then(|m| m.as_object()) {
            let target = new_metadata.get_or_insert_with(|| serde_json::json!({}));
            if !target.is_object() {
                *target = serde_json::json!({ "previous_metadata": target.clone() });
            }
            if let Some(object) = target.as_object_mut() {
                for key in ["enc", "sealed"] {
                    if let Some(value) = old.get(key) {
                        object.insert(key.into(), value.clone());
                    }
                }
            }
        }
    }
    if new_kind != "credential" {
        // No longer a credential: drop the ciphertext of the old secret.
        if let Some(object) = new_metadata.as_mut().and_then(|m| m.as_object_mut()) {
            object.remove("enc");
            object.remove("sealed");
        }
    }
    // Leaving the vault swaps the stored masked label for the clear text: hash and embedding must follow.
    let leaves_vault = existing.kind == "credential" && new_kind != "credential";
    let content_changed = (content.is_some() && new_content != existing.content) || leaves_vault;

    let mem = Memory {
        id: id.into(),
        content: new_content.into(),
        kind: new_kind.into(),
        project: existing.project,
        tags: new_tags,
        source: existing.source,
        importance: new_imp,
        expires_at: new_exp,
        created_at: existing.created_at,
        updated_at: now,
        metadata: new_metadata,
        last_accessed_at: existing.last_accessed_at,
        access_count: existing.access_count,
    };
    store.write_memory(&mem, &content_hash(&mem.content), content_changed)?;
    report_link_failure(rebuild_memory_links(store, &mem), &mem.id);
    Ok(Some(WriteOutcome { memory: mem, merged: false, content_changed }))
}

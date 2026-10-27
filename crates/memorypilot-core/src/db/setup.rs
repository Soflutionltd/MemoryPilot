//! Schema creation and upgrades.

use super::*;

impl Database {
    pub(super) fn init_schema(&self) -> Result<(), String> {
        self.conn.execute_batch("
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'fact',
                project TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                source TEXT NOT NULL DEFAULT 'cursor',
                importance INTEGER NOT NULL DEFAULT 3,
                expires_at TEXT,
                metadata TEXT,
                embedding BLOB,
                content_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT,
                access_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS memory_links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL DEFAULT 'relates_to',
                valid_from TEXT,
                valid_to TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id);
            CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id);

            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id TEXT NOT NULL,
                entity_kind TEXT NOT NULL,
                entity_value TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_entities_value ON memory_entities(entity_value);
            CREATE INDEX IF NOT EXISTS idx_entities_memory ON memory_entities(memory_id);

            CREATE TABLE IF NOT EXISTS knowledge_triples (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                confidence REAL DEFAULT 1.0,
                source_memory_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_triples_subject ON knowledge_triples(subject);
            CREATE INDEX IF NOT EXISTS idx_triples_object ON knowledge_triples(object);
            CREATE INDEX IF NOT EXISTS idx_triples_valid ON knowledge_triples(valid_from, valid_to);

            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                project TEXT,
                level TEXT NOT NULL DEFAULT 'interval',
                title TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL,
                embedding BLOB,
                start_at TEXT NOT NULL,
                end_at TEXT NOT NULL,
                salience REAL NOT NULL DEFAULT 0.0,
                memory_count INTEGER NOT NULL DEFAULT 0,
                source_ids TEXT NOT NULL DEFAULT '[]',
                entities TEXT NOT NULL DEFAULT '[]',
                child_ids TEXT NOT NULL DEFAULT '[]',
                content_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_project ON episodes(project);
            CREATE INDEX IF NOT EXISTS idx_episodes_level ON episodes(level);
            CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(start_at, end_at);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_hash ON episodes(content_hash) WHERE content_hash IS NOT NULL;

            CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
            CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
            CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_expires ON memories(expires_at) WHERE expires_at IS NOT NULL;

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, tags, kind, project,
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TABLE IF NOT EXISTS projects (
                name TEXT PRIMARY KEY,
                path TEXT NOT NULL DEFAULT '',
                description TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories_archive (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                kind TEXT NOT NULL,
                project TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                source TEXT NOT NULL DEFAULT '',
                importance INTEGER NOT NULL DEFAULT 3,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived_at TEXT NOT NULL,
                reason TEXT NOT NULL,
                replaced_by TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_archive_replaced_by ON memories_archive(replaced_by);
        ").map_err(|e| format!("Schema: {}", e))
    }

    /// Upgrade schema for existing databases (add new columns if missing).
    pub(super) fn upgrade_schema(&self) -> Result<(), String> {
        // Check if importance column exists
        let has_importance: bool = self
            .conn
            .prepare("SELECT importance FROM memories LIMIT 0")
            .is_ok();
        if !has_importance {
            let _ = self.conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN importance INTEGER NOT NULL DEFAULT 3;
                 ALTER TABLE memories ADD COLUMN expires_at TEXT;",
            );
        }
        // v3.0 columns
        let has_embedding: bool = self
            .conn
            .prepare("SELECT embedding FROM memories LIMIT 0")
            .is_ok();
        if !has_embedding {
            let _ = self.conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN embedding BLOB;
                 ALTER TABLE memories ADD COLUMN last_accessed_at TEXT;
                 ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;",
            );
        }
        // v4.0: knowledge_triples + temporal columns
        let _ = self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memory_links (
                 source_id TEXT NOT NULL,
                 target_id TEXT NOT NULL,
                 relation_type TEXT NOT NULL DEFAULT 'relates_to',
                 valid_from TEXT,
                 valid_to TEXT,
                 confidence REAL DEFAULT 1.0,
                 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                 PRIMARY KEY (source_id, target_id),
                 FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                 FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
             );
             CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id);
             CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id);
             CREATE TABLE IF NOT EXISTS memory_entities (
                 memory_id TEXT NOT NULL,
                 entity_kind TEXT NOT NULL,
                 entity_value TEXT NOT NULL,
                 valid_from TEXT,
                 valid_to TEXT,
                 FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
             );
             CREATE INDEX IF NOT EXISTS idx_entities_value ON memory_entities(entity_value);
             CREATE INDEX IF NOT EXISTS idx_entities_memory ON memory_entities(memory_id);
             CREATE TABLE IF NOT EXISTS knowledge_triples (
                 id TEXT PRIMARY KEY,
                 subject TEXT NOT NULL,
                 predicate TEXT NOT NULL,
                 object TEXT NOT NULL,
                 valid_from TEXT,
                 valid_to TEXT,
                 confidence REAL DEFAULT 1.0,
                 source_memory_id TEXT,
                 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                 FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE SET NULL
             );
             CREATE INDEX IF NOT EXISTS idx_triples_subject ON knowledge_triples(subject);
             CREATE INDEX IF NOT EXISTS idx_triples_object ON knowledge_triples(object);
             CREATE INDEX IF NOT EXISTS idx_triples_valid ON knowledge_triples(valid_from, valid_to);"
        );
        // v4.0.1: content_hash column
        let has_content_hash: bool = self
            .conn
            .prepare("SELECT content_hash FROM memories LIMIT 0")
            .is_ok();
        if !has_content_hash {
            let _ = self
                .conn
                .execute_batch("ALTER TABLE memories ADD COLUMN content_hash TEXT;");
        }
        // v4.3: hierarchical episodic memory. Additive — existing rows
        // are untouched; the table is created empty and populated lazily
        // by the rollup pass (`Database::roll_up_episodes`).
        let _ = self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS episodes (
                 id TEXT PRIMARY KEY,
                 project TEXT,
                 level TEXT NOT NULL DEFAULT 'interval',
                 title TEXT NOT NULL DEFAULT '',
                 summary TEXT NOT NULL,
                 embedding BLOB,
                 start_at TEXT NOT NULL,
                 end_at TEXT NOT NULL,
                 salience REAL NOT NULL DEFAULT 0.0,
                 memory_count INTEGER NOT NULL DEFAULT 0,
                 source_ids TEXT NOT NULL DEFAULT '[]',
                 entities TEXT NOT NULL DEFAULT '[]',
                 child_ids TEXT NOT NULL DEFAULT '[]',
                 content_hash TEXT,
                 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
             );
             CREATE INDEX IF NOT EXISTS idx_episodes_project ON episodes(project);
             CREATE INDEX IF NOT EXISTS idx_episodes_level ON episodes(level);
             CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(start_at, end_at);
             CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_hash ON episodes(content_hash) WHERE content_hash IS NOT NULL;",
        );
        Ok(())
    }
}

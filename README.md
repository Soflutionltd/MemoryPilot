<p align="center">
  <img src="static/banner.png" alt="MemoryPilot" width="900"/>
</p>

<p align="center">
  <strong>The most advanced MCP memory server. Period.</strong><br>
  <sub>Hybrid search (BM25 + TF-IDF RRF) · GraphRAG · Chunked RAG · Auto-Linting (Self-Healing) · Project brain · Single binary</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/v3.2-latest-green" alt="v3.2"/>
  <img src="https://img.shields.io/badge/language-Rust-orange" alt="Rust"/>
  <img src="https://img.shields.io/badge/search-Hybrid_RRF-blueviolet" alt="Hybrid RRF"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"/>
  <img src="https://img.shields.io/badge/binary-2.4MB-yellow" alt="Binary size"/>
</p>

---

## Why

AI coding assistants forget everything between sessions. MemoryPilot gives them persistent, searchable memory with project awareness, semantic understanding, and automatic knowledge organization.

**vs every other MCP memory server:**

| Feature | MemoryPilot v3.2 | MCP Memory (Node.js) | Other Rust/Python servers |
|---------|-----------------|----------------------|--------------------------|
| Search | Hybrid BM25 + TF-IDF RRF fusion | Unranked filter | BM25 only |
| GraphRAG | Auto entity extraction + graph traversal | No | No |
| Chunked RAG | Transcript auto-chunking (Zero context bloat) | No | No |
| Self-Healing | Background auto-linting loop | No | No |
| Garbage collection | Heuristic merge + scoring | No | TTL only |
| Project brain (<1500 tokens) | Yes | No | No |
| File watcher context boost | Yes | No | No |
| Deduplication | Jaccard 85% threshold | No | Basic exact match |
| Memory types | 10 types, importance 1-5 | 1 type | 2-3 types |
| Startup | 1-2 ms | 50-100 ms | 5-20 ms |
| Binary | 2.4 MB, zero deps | 200 MB+ (node_modules) | 5-50 MB |
| Storage | SQLite WAL + FTS5 | JSON files | SQLite basic |

## The 6 Pillars

### 1. Hybrid Search (BM25 + TF-IDF RRF)

Every memory gets a 384-dimension TF-IDF embedding vector on insert. Search runs both BM25 full-text and cosine similarity in parallel, then merges results with Reciprocal Rank Fusion. This catches semantic matches that keyword search misses.

Results are boosted by importance weighting, knowledge graph link density, and file watcher context.

### 2. GraphRAG

Every memory is automatically analyzed for entities: technologies, file paths, components, projects. Entities are stored in a dedicated table. Memories sharing entities are auto-linked with inferred relationship types (resolves, implements, depends_on, deprecates...).

When searching, `MemoryPilot` traverses the knowledge graph (GraphRAG) from the top matches to pull in related context (e.g., finding the architecture decision that led to a specific bug fix).

### 3. Chunked RAG (Transcripts)

You can save full conversation transcripts without polluting the LLM context window. The `add_transcript` tool automatically chunks large texts into ~2000 character blocks and links them together. These chunks are excluded from auto-loading on startup (`recall`), but are perfectly searchable via Vector embeddings.

### 4. Self-Healing (Auto-Linter)

MemoryPilot watches your files. When you save a Rust (`cargo check`), Svelte (`svelte-check`), or TS (`tsc`) file, it lints it in the background. If it finds a compilation error, it automatically creates a `bug` memory with the exact stack trace. Your AI agent instantly knows what's broken without you having to copy-paste the terminal output.

### 5. Garbage Collection

Old, low-importance memories are scored for cleanup candidacy. Groups of related stale memories are merged into condensed summaries using heuristic keyword extraction. Orphaned links and entities are cleaned. DB is vacuumed after significant deletions.

### 6. Project Brain

One tool call returns a dense JSON snapshot of a project under 1500 tokens: tech stack, architecture decisions, active bugs, recent changes, key components. Perfect for injecting into a new conversation.

## Install

```bash
git clone https://github.com/Soflution1/MemoryPilot.git
cd MemoryPilot
cargo build --release
cp target/release/MemoryPilot ~/.local/bin/
chmod +x ~/.local/bin/MemoryPilot
xattr -cr ~/.local/bin/MemoryPilot  # macOS only
```

### Cursor Integration (Zero-Config)

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "MemoryPilot": {
      "command": "/Users/you/.local/bin/MemoryPilot"
    }
  }
}
```

**That's it.** MemoryPilot automatically injects a dynamic System Prompt into Cursor and Claude Desktop on startup. The AI will proactively act as your autonomous secretary, calling the `add_memory` tool in the background to store your architecture decisions, API keys, and bug fixes without you having to write any `.cursorrules` or prompt it manually.

Or use via [McpHub](https://github.com/Soflution1/McpHub) for SSE transport with all your other MCP servers.

### First run

```bash
# If upgrading from v1 (JSON files):
MemoryPilot --migrate

# Compute embeddings for existing memories:
MemoryPilot --backfill
```

## MCP Tools (20)

| Tool | Description |
|------|-------------|
| **`recall`** | Start here. Loads all context in one shot: project memories, preferences, critical facts, patterns, decisions, global prompt. |
| **`get_project_brain`** | Instant project summary (<1500 tokens): tech stack, architecture, bugs, recent changes, components. |
| **`search_memory`** | Hybrid BM25 + TF-IDF RRF search, boosted by importance, graph links, and file watcher context. |
| **`get_file_context`** | Memories related to recently modified files in working directory. |
| `add_memory` | Store with auto-dedup (Jaccard 85%), auto entity extraction, auto graph linking. Importance 1-5, TTL. |
| `add_memories` | Bulk add multiple memories in one call with per-item dedup. |
| `get_memory` | Retrieve by ID. |
| `update_memory` | Update content, kind, tags, importance, TTL. |
| `delete_memory` | Delete by ID (cascades to entities and links). |
| `list_memories` | List with project/kind filters and pagination. |
| `get_project_context` | Full project context with preferences and patterns. |
| `register_project` | Register project with filesystem path for auto-detection. |
| `list_projects` | List projects with memory counts. |
| `get_stats` | DB statistics: totals, by kind, by project, DB size. |
| `get_global_prompt` | Auto-discover GLOBAL_PROMPT.md from ~/.MemoryPilot/ or project root. |
| `export_memories` | Export as JSON or Markdown with importance stars. |
| `set_config` | Set config values (e.g. global_prompt_path). |
| `run_gc` | Garbage collection: merge old memories, clean orphans, vacuum. Supports dry_run. |
| `cleanup_expired` | Remove expired TTL memories. |
| `migrate_v1` | Import from v1 JSON files. |

### Memory Types

`fact` · `preference` · `decision` · `pattern` · `snippet` · `bug` · `credential` · `todo` · `note`

Each memory has importance (1-5), optional TTL, tags, project scope, and auto-generated embedding + entity links.

## CLI

```bash
MemoryPilot              # Start MCP stdio server
MemoryPilot --backfill   # Compute missing TF-IDF embeddings
MemoryPilot --migrate    # Import v1 JSON data to SQLite
MemoryPilot --version    # Show version
MemoryPilot --help       # Show help
```

## Architecture

```
src/main.rs        — CLI + MCP stdio server loop + file watcher init
src/db.rs          — SQLite engine: hybrid search, CRUD, graph, GC, brain, recall
src/tools.rs       — 20 MCP tool definitions + handlers
src/protocol.rs    — JSON-RPC types
src/embedding.rs   — TF-IDF 384-dim vectors, cosine similarity, RRF fusion
src/graph.rs       — Entity extraction (tech, files, components) + relation inference
src/gc.rs          — GC scoring, heuristic memory merging, stopwords
src/watcher.rs     — File system watcher with keyword extraction for search boost
```

### Database Schema

```sql
memories        — id, content, kind, project, tags, importance, embedding (BLOB),
                  expires_at, last_accessed_at, access_count, metadata
memories_fts    — FTS5 virtual table (content, tags, kind, project)
memory_entities — memory_id, entity_kind, entity_value
memory_links    — source_id, target_id, relation_type (CASCADE delete)
projects        — name, path, description
config          — key/value store
```

## Performance

| Metric | Value |
|--------|-------|
| Binary size | 2.4 MB |
| Startup | 1-2 ms |
| Search (866 memories) | <1 ms (hybrid RRF) |
| RAM | ~5 MB |
| Embedding generation | <0.1 ms per memory |
| Storage overhead | ~1.5 KB per embedding (384 × 4 bytes) |
| Runtime dependencies | **None** |

## Storage

- Database: `~/.MemoryPilot/memory.db`
- Global prompt: `~/.MemoryPilot/GLOBAL_PROMPT.md`

## License

MIT — Built by [SOFLUTION LTD](https://soflution.com)

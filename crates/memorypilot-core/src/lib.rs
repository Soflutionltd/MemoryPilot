#![recursion_limit = "256"]
//! MemoryPilot core: storage, hybrid search, recall, knowledge graph, vault.
//! The MCP server (`MemoryPilot` binary) is a client of this library.

pub mod ann;
pub mod chunker;
pub mod code_chunker;
pub mod connectors;
pub mod db;
pub mod diversity;
pub mod embedding;
pub mod facts;
pub mod fts;
pub mod gc;
pub mod graph;
pub mod onepassword;
pub mod onnx_external;
pub mod pool;
pub mod reader;
pub mod reranking;
pub mod session_capsule;
pub mod session_export;
pub mod session_fusion;
pub mod splitter;
pub mod stemming;
pub mod telemetry;
pub mod temporal;
pub mod text;
pub mod tokenizer;
pub mod verify;
pub mod working_memory;

/// Global prompt file contents, keyed by path, invalidated on mtime change.
pub(crate) static PROMPT_CACHE: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, (std::time::SystemTime, String)>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

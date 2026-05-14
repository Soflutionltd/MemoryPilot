#![recursion_limit = "256"]
mod ann;
mod chunker;
mod code_chunker;
/// MemoryPilot v4.0 — God-Tier MCP memory server.
/// Hybrid search (BM25 + fastembed RRF), Temporal Knowledge Graph, GC, Project Brain, File Watcher, HTTP server.
/// (c) SOFLUTION LTD — Apache 2.0 License
mod db;
mod embedding;
mod fts;
mod gc;
mod graph;
#[cfg(feature = "http")]
mod http;
mod protocol;
mod reranking;
mod session_capsule;
mod session_export;
mod session_fusion;
mod splitter;
mod tools;
mod watcher;
mod working_memory;

use protocol::{JsonRpcRequest, JsonRpcResponse};
use serde_json::json;
use std::io::{self, BufRead, Write};

use std::sync::{Arc, Mutex, OnceLock};

pub static WATCHER_STATE: OnceLock<Arc<Mutex<watcher::FileWatcherState>>> = OnceLock::new();
pub static PROMPT_CACHE: std::sync::LazyLock<
    Mutex<std::collections::HashMap<String, (std::time::SystemTime, String)>>,
> = std::sync::LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

const VERSION: &str = env!("CARGO_PKG_VERSION");
const SERVER_NAME: &str = "MemoryPilot";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--version" || a == "-v") {
        println!("MemoryPilot v{}", VERSION);
        return;
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    if args.iter().any(|a| a == "--migrate") {
        run_migrate();
        return;
    }
    if args.iter().any(|a| a == "--backfill-force") {
        run_backfill_force();
        return;
    }
    if args.iter().any(|a| a == "--backfill") {
        run_backfill();
        return;
    }
    if args.iter().any(|a| a == "--benchmark-recall") {
        run_benchmark_recall(&args);
        return;
    }
    if args.iter().any(|a| a == "--benchmark-search") {
        run_benchmark_search(&args);
        return;
    }
    if args.iter().any(|a| a == "--benchmark-longmemeval") {
        run_benchmark_longmemeval(&args);
        return;
    }
    if args.iter().any(|a| a == "--benchmark-fr") {
        run_benchmark_fr(&args);
        return;
    }
    if args.iter().any(|a| a == "--benchmark-latency") {
        run_benchmark_latency(&args);
        return;
    }
    if args.iter().any(|a| a == "--benchmark-concurrency") {
        run_benchmark_concurrency(&args);
        return;
    }
    #[cfg(feature = "http")]
    {
        if let Some(pos) = args.iter().position(|a| a == "--http") {
            let port = args
                .get(pos + 1)
                .and_then(|v| v.parse::<u16>().ok())
                .unwrap_or(7437);
            run_http_server(port);
            return;
        }
    }
    run_mcp_server();
}

fn run_mcp_server() {
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(state) = watcher::start_watcher(&cwd.to_string_lossy()) {
            let _ = WATCHER_STATE.set(state);
        }
    }

    let db = match db::Database::open() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("DB error: {}", e);
            std::process::exit(1);
        }
    };
    let db_arc = std::sync::Arc::new(db);

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            Ok(_) => continue,
            Err(_) => break,
        };
        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::error(None, -32700, format!("Parse: {}", e));
                let _ = writeln!(out, "{}", serde_json::to_string(&resp).unwrap());
                let _ = out.flush();
                continue;
            }
        };
        if request.id.is_none() {
            // JSON-RPC notification — no response expected
            let _ = handle_request(&db_arc, &request);
            continue;
        }
        let response = handle_request(&db_arc, &request);
        let _ = writeln!(out, "{}", serde_json::to_string(&response).unwrap());
        let _ = out.flush();
    }
}

fn handle_request(db: &db::Database, req: &JsonRpcRequest) -> JsonRpcResponse {
    match req.method.as_str() {
        "initialize" => JsonRpcResponse::success(
            req.id.clone(),
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": SERVER_NAME, "version": VERSION },
                "instructions": "CRITICAL WORKFLOW:\n1. Always call 'recall' tool at the start of a conversation.\n2. DURING the conversation, you MUST proactively call 'add_memory' to store any new architecture decision, convention, or significant bug fix. Do NOT ask the user for permission — act as an autonomous technical secretary.\n3. NEVER store secrets, passwords, API keys, or tokens in memory. Use environment variables or secret managers for credentials."
            }),
        ),
        "notifications/initialized" => JsonRpcResponse::success(req.id.clone(), json!({})),
        "tools/list" => JsonRpcResponse::success(req.id.clone(), tools::tool_definitions()),
        "tools/call" => {
            let name = req
                .params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args = req.params.get("arguments").cloned().unwrap_or(json!({}));
            JsonRpcResponse::success(req.id.clone(), tools::handle_tool_call(db, name, &args))
        }
        "ping" => JsonRpcResponse::success(req.id.clone(), json!({})),
        _ => JsonRpcResponse::error(req.id.clone(), -32601, format!("Unknown: {}", req.method)),
    }
}
fn run_migrate() {
    let db = match db::Database::open() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("DB error: {}", e);
            std::process::exit(1);
        }
    };
    match db.migrate_from_v1() {
        Ok(n) => println!("✓ Migrated {} memories from v1 JSON to SQLite.", n),
        Err(e) => {
            eprintln!("✗ Failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_backfill() {
    eprintln!("Embedding engine: fastembed (multilingual-e5-small, 384-dim)");
    let db = match db::Database::open() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("DB error: {}", e);
            std::process::exit(1);
        }
    };
    match db.backfill_embeddings() {
        Ok(n) => println!("✓ Generated embeddings for {} memories (missing only).", n),
        Err(e) => {
            eprintln!("✗ Failed: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "http")]
fn run_http_server(port: u16) {
    let db = match db::Database::open() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("DB error: {}", e);
            std::process::exit(1);
        }
    };
    let db_arc = std::sync::Arc::new(db);
    http::start_http_server(db_arc, port);
}

fn run_backfill_force() {
    eprintln!("Embedding engine: fastembed (multilingual-e5-small, 384-dim) (force overwrite ALL)");
    let db = match db::Database::open() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("DB error: {}", e);
            std::process::exit(1);
        }
    };
    match db.backfill_embeddings_force() {
        Ok(n) => println!("✓ Regenerated embeddings for ALL {} memories.", n),
        Err(e) => {
            eprintln!("✗ Failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_benchmark_recall(args: &[String]) {
    let db = match db::Database::open() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("DB error: {}", e);
            std::process::exit(1);
        }
    };
    let scenario_limit = args
        .windows(2)
        .find(|window| window[0] == "--scenario-limit")
        .and_then(|window| window[1].parse::<usize>().ok())
        .unwrap_or(12);
    match db.benchmark_recall(scenario_limit) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into())
        ),
        Err(error) => {
            eprintln!("✗ Benchmark failed: {}", error);
            std::process::exit(1);
        }
    }
}

fn run_benchmark_search(args: &[String]) {
    let db = match db::Database::open() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("DB error: {}", e);
            std::process::exit(1);
        }
    };
    let scenario_limit = args
        .windows(2)
        .find(|window| window[0] == "--scenario-limit")
        .and_then(|window| window[1].parse::<usize>().ok())
        .unwrap_or(20);
    match db.benchmark_search(scenario_limit) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into())
        ),
        Err(error) => {
            eprintln!("✗ Search benchmark failed: {}", error);
            std::process::exit(1);
        }
    }
}

fn run_benchmark_longmemeval(args: &[String]) {
    eprintln!("[LongMemEval] Starting benchmark runner...");
    eprintln!("Embedding engine: fastembed (multilingual-e5-small, 384-dim)");
    let dataset_path = args
        .windows(2)
        .find(|w| w[0] == "--benchmark-longmemeval")
        .map(|w| w[1].as_str())
        .unwrap_or("benchmarks/longmemeval_s_cleaned.json");
    let limit = args
        .windows(2)
        .find(|w| w[0] == "--limit")
        .and_then(|w| w[1].parse::<usize>().ok());
    eprintln!(
        "[LongMemEval] Dataset: {} (limit: {:?})",
        dataset_path, limit
    );
    let min_r5 = args
        .windows(2)
        .find(|w| w[0] == "--min-r5")
        .and_then(|w| parse_percent(&w[1]));
    match db::Database::benchmark_longmemeval(dataset_path, limit) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into())
            );
            if let Some(minimum) = min_r5 {
                let actual = report
                    .get("metrics")
                    .and_then(|metrics| metrics.get("recall_at_5"))
                    .and_then(|value| value.as_str())
                    .and_then(parse_percent);
                match actual {
                    Some(value) if value + f64::EPSILON >= minimum => {
                        eprintln!(
                            "[LongMemEval] Guard passed: R@5 {:.1}% >= {:.1}%",
                            value, minimum
                        );
                    }
                    Some(value) => {
                        eprintln!(
                            "✗ LongMemEval guard failed: R@5 {:.1}% < {:.1}%",
                            value, minimum
                        );
                        std::process::exit(2);
                    }
                    None => {
                        eprintln!("✗ LongMemEval guard failed: missing R@5 metric");
                        std::process::exit(2);
                    }
                }
            }
        }
        Err(error) => {
            eprintln!("✗ LongMemEval benchmark failed: {}", error);
            std::process::exit(1);
        }
    }
}

fn parse_percent(value: &str) -> Option<f64> {
    value.trim().trim_end_matches('%').parse::<f64>().ok()
}

fn run_benchmark_fr(args: &[String]) {
    eprintln!("[BenchFR] Starting French retrieval benchmark...");
    eprintln!("Embedding engine: fastembed (multilingual-e5-small, 384-dim)");
    let min_r5 = args
        .windows(2)
        .find(|w| w[0] == "--min-r5")
        .and_then(|w| parse_percent(&w[1]));
    match db::Database::benchmark_fr() {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into())
            );
            if let Some(minimum) = min_r5 {
                let actual = report
                    .get("metrics")
                    .and_then(|metrics| metrics.get("recall_at_5"))
                    .and_then(|value| value.as_str())
                    .and_then(parse_percent);
                match actual {
                    Some(value) if value + f64::EPSILON >= minimum => {
                        eprintln!("[BenchFR] Guard passed: R@5 {:.1}% >= {:.1}%", value, minimum);
                    }
                    Some(value) => {
                        eprintln!(
                            "✗ BenchFR guard failed: R@5 {:.1}% < {:.1}%",
                            value, minimum
                        );
                        std::process::exit(2);
                    }
                    None => {
                        eprintln!("✗ BenchFR guard failed: missing R@5 metric");
                        std::process::exit(2);
                    }
                }
            }
        }
        Err(error) => {
            eprintln!("✗ BenchFR failed: {}", error);
            std::process::exit(1);
        }
    }
}

fn run_benchmark_latency(args: &[String]) {
    use std::time::Instant;

    let queries_count = args
        .windows(2)
        .find(|w| w[0] == "--queries")
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(50);
    let seed_memories = args
        .windows(2)
        .find(|w| w[0] == "--seed-memories")
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(5_000);
    let custom_db = args
        .windows(2)
        .find(|w| w[0] == "--db")
        .map(|w| w[1].clone());

    eprintln!("[Latency] Starting bench (queries={}, seed_memories={})", queries_count, seed_memories);

    let tmp_dir = std::env::temp_dir().join(format!("memorypilot-latency-{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).ok();
    let db_path = match &custom_db {
        Some(p) => std::path::PathBuf::from(p),
        None => tmp_dir.join("bench.db"),
    };
    let needs_seed = custom_db.is_none() || !db_path.exists();

    if needs_seed {
        eprintln!("[Latency] Seeding {} synthetic memories at {}...", seed_memories, db_path.display());
        let seed_start = Instant::now();
        let db = match db::Database::open_at(&db_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("✗ open_at failed: {}", e);
                std::process::exit(1);
            }
        };
        let topics = [
            "rust async runtime", "tokio mutex deadlock", "sqlite wal checkpoint",
            "fastembed onnx model", "knowledge graph entity link", "stripe webhook signature",
            "supabase rls policy", "svelte runes derived", "tailwind responsive grid",
            "cloudflare wrangler deploy", "embedding cosine similarity", "bm25 ranking function",
            "hnsw ann index", "tree sitter rust parser", "ttl garbage collection",
        ];
        let scope = db::MemoryScope::default();
        for i in 0..seed_memories {
            let topic = &topics[i % topics.len()];
            // Synthesize content unique enough to defeat the dedup pass.
            let nonce_a: u64 = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1);
            let nonce_b: u64 = 0xBF58_476D_1CE4_E5B9u64.wrapping_mul(i as u64 + 7);
            let content = format!(
                "Memory #{} about {} (run-id {:016x}-{:016x}): unique entry detailing how {} \
                 interacts with subsystem #{} under condition {}, batch {}, slot {}, scenario {}.",
                i,
                topic,
                nonce_a,
                nonce_b,
                topic,
                i % 257,
                i % 91,
                i / 500,
                i % 13,
                i.wrapping_mul(31)
            );
            let _ = db.add_memory(
                &content,
                "note",
                Some("bench"),
                &[topic.to_string(), format!("batch-{}", i / 500)],
                "bench-latency",
                3,
                None,
                None,
                &scope,
            );
        }
        let seed_ms = seed_start.elapsed().as_millis();
        eprintln!("[Latency] Seed done in {} ms. Closing.", seed_ms);
        drop(db);
        // Wait briefly for async embed worker so the warm-up has real data.
        eprintln!("[Latency] Waiting 8s for async embedding worker to populate vectors...");
        std::thread::sleep(std::time::Duration::from_secs(8));
    }

    // Optional flag: clear persisted ANN to force a cold warm-up (useful to
    // benchmark warm sync vs warm async startup paths).
    if args.iter().any(|a| a == "--clear-ann") {
        let mut ann_file = db_path.clone();
        let ann_name = ann_file
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| format!("{}.ann.usearch", n))
            .unwrap_or_default();
        ann_file.set_file_name(&ann_name);
        let _ = std::fs::remove_file(&ann_file);
        let mut keys_file = ann_file.clone();
        keys_file.set_file_name(format!("{}.keys.json", ann_name));
        let _ = std::fs::remove_file(&keys_file);
        eprintln!("[Latency] Cleared persisted ANN at {}", ann_file.display());
    }

    eprintln!("[Latency] Measuring open_at() startup...");
    let open_start = Instant::now();
    let db = match db::Database::open_at(&db_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("✗ open_at failed: {}", e);
            std::process::exit(1);
        }
    };
    let open_ms = open_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[Latency] open_at: {:.2} ms", open_ms);

    let queries: Vec<String> = (0..queries_count)
        .map(|i| {
            let topics = [
                "rust async runtime tokio", "sqlite wal mode performance",
                "fastembed multilingual model", "knowledge graph entity",
                "stripe webhook validation", "supabase row level security",
                "svelte 5 runes pattern", "tailwind grid responsive",
                "hnsw approximate nearest neighbor", "bm25 fts5 ranking",
                "embedding cosine vector search", "tree sitter parser",
            ];
            format!("{} variant {}", topics[i % topics.len()], i)
        })
        .collect();

    eprintln!("[Latency] Cold pass ({} queries, embed cache MISS expected)...", queries.len());
    let mut cold_times = Vec::with_capacity(queries.len());
    for q in &queries {
        let t = Instant::now();
        let _ = db.search(q, 10, None, None, None, None);
        cold_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    eprintln!("[Latency] Warm pass (same queries, embed cache HIT expected)...");
    let mut warm_times = Vec::with_capacity(queries.len());
    for q in &queries {
        let t = Instant::now();
        let _ = db.search(q, 10, None, None, None, None);
        warm_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let cold_stats = percentiles(&mut cold_times);
    let warm_stats = percentiles(&mut warm_times);
    let speedup = if warm_stats.p50 > 0.0 {
        cold_stats.p50 / warm_stats.p50
    } else {
        0.0
    };

    let report = serde_json::json!({
        "config": {
            "queries": queries_count,
            "seed_memories": if needs_seed { seed_memories } else { 0 },
            "db_path": db_path.display().to_string(),
            "user_provided_db": custom_db.is_some(),
            "seeded_this_run": needs_seed,
        },
        "open_at_ms": format!("{:.2}", open_ms),
        "cold_search_ms": {
            "p50": format!("{:.2}", cold_stats.p50),
            "p95": format!("{:.2}", cold_stats.p95),
            "p99": format!("{:.2}", cold_stats.p99),
            "avg": format!("{:.2}", cold_stats.avg),
        },
        "warm_search_ms": {
            "p50": format!("{:.2}", warm_stats.p50),
            "p95": format!("{:.2}", warm_stats.p95),
            "p99": format!("{:.2}", warm_stats.p99),
            "avg": format!("{:.2}", warm_stats.avg),
        },
        "cache_speedup_x": format!("{:.2}", speedup),
    });

    println!("{}", serde_json::to_string_pretty(&report).unwrap());

    if custom_db.is_none() {
        std::fs::remove_dir_all(&tmp_dir).ok();
    }
}

fn run_benchmark_concurrency(args: &[String]) {
    use std::sync::Arc;
    use std::time::Instant;

    let client_count = args
        .windows(2)
        .find(|w| w[0] == "--clients")
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(8);
    let queries_per_client = args
        .windows(2)
        .find(|w| w[0] == "--queries-per-client")
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(50);
    let seed_memories = args
        .windows(2)
        .find(|w| w[0] == "--seed-memories")
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(2_000);

    eprintln!(
        "[Concurrency] Starting bench (clients={}, queries_per_client={}, seed={})",
        client_count, queries_per_client, seed_memories
    );

    let tmp_dir = std::env::temp_dir().join(format!(
        "memorypilot-concurrency-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp_dir).ok();
    let db_path = tmp_dir.join("bench.db");

    eprintln!("[Concurrency] Seeding {} synthetic memories...", seed_memories);
    let seed_start = Instant::now();
    {
        let db = match db::Database::open_at(&db_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("✗ open_at failed: {}", e);
                std::process::exit(1);
            }
        };
        let topics = [
            "rust async runtime", "tokio mutex deadlock", "sqlite wal checkpoint",
            "fastembed onnx model", "knowledge graph entity", "stripe webhook",
            "supabase rls policy", "svelte runes", "tailwind grid",
            "cloudflare wrangler", "embedding cosine", "bm25 ranking",
            "hnsw ann index", "tree sitter parser", "ttl garbage collection",
        ];
        let scope = db::MemoryScope::default();
        for i in 0..seed_memories {
            let topic = &topics[i % topics.len()];
            let nonce: u64 = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1);
            let content = format!(
                "Memory #{} about {} (uniq {:016x}): detailed scenario {} variant {}.",
                i, topic, nonce, i % 257, i.wrapping_mul(31)
            );
            let _ = db.add_memory(&content, "note", Some("bench"), &[topic.to_string()],
                "bench-concurrency", 3, None, None, &scope);
        }
        eprintln!("[Concurrency] Seed done in {:.1}s", seed_start.elapsed().as_secs_f64());
        eprintln!("[Concurrency] Waiting for embed worker to fully drain...");
        let wait_start = Instant::now();
        loop {
            let probe = match rusqlite::Connection::open(&db_path) {
                Ok(c) => c,
                Err(_) => break,
            };
            let _ = probe.busy_timeout(std::time::Duration::from_secs(2));
            let pending: i64 = probe
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE embedding IS NULL",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(-1);
            if pending == 0 {
                break;
            }
            if wait_start.elapsed().as_secs() > 600 {
                eprintln!("[Concurrency] Timeout: {} embeddings still pending", pending);
                break;
            }
            if wait_start.elapsed().as_millis() % 5000 < 250 && pending > 0 {
                eprintln!("  ... {} pending after {:.0}s", pending, wait_start.elapsed().as_secs_f64());
            }
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        eprintln!(
            "[Concurrency] Embed worker drained in {:.1}s.",
            wait_start.elapsed().as_secs_f64()
        );
    }

    let queries: Arc<Vec<String>> = Arc::new(
        (0..queries_per_client * 4)
            .map(|i| {
                let topics = [
                    "rust async tokio", "sqlite wal performance",
                    "fastembed multilingual", "knowledge graph",
                    "stripe webhook validation", "supabase row level security",
                    "svelte 5 runes pattern", "tailwind responsive",
                    "hnsw approximate nearest neighbor", "bm25 fts5",
                    "embedding cosine search", "tree sitter parser",
                ];
                format!("{} variant {}", topics[i % topics.len()], i)
            })
            .collect(),
    );

    // Warm up the lazy fastembed ONNX model and the in-process ANN index on
    // the main thread so the first search of every worker does not eat the
    // one-time init/warm costs.
    eprintln!("[Concurrency] Warming up embedding model and ANN...");
    {
        let warm_db = match db::Database::open_at(&db_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("✗ warm open_at failed: {}", e);
                std::process::exit(1);
            }
        };
        for q in queries.iter().take(10) {
            let _ = warm_db.search(q, 10, None, None, None, None);
        }
        // Give the detached ANN warm-up thread time to persist its index
        // before workers open their own handles.
        std::thread::sleep(std::time::Duration::from_secs(3));
    }

    eprintln!(
        "[Concurrency] Launching {} clients × {} queries in parallel (each opens its own DB handle)...",
        client_count, queries_per_client
    );
    let bench_start = Instant::now();
    let mut handles = Vec::with_capacity(client_count);
    for client_id in 0..client_count {
        let queries_handle = Arc::clone(&queries);
        let qpc = queries_per_client;
        let db_path_clone = db_path.clone();
        handles.push(std::thread::spawn(move || {
            // Each worker opens its own Database handle (mirrors a multi-worker
            // HTTP server where every request handler holds a connection).
            let db = match db::Database::open_at(&db_path_clone) {
                Ok(d) => d,
                Err(_) => return (Vec::new(), qpc),
            };
            let mut latencies = Vec::with_capacity(qpc);
            let mut errors = 0usize;
            for i in 0..qpc {
                let q_index = (client_id * qpc + i) % queries_handle.len();
                let q = &queries_handle[q_index];
                let t = Instant::now();
                match db.search(q, 10, None, None, None, None) {
                    Ok(_) => {
                        // Skip the first 5 queries per worker so we measure
                        // steady-state and not the open_at + ANN warmup tail.
                        if i >= 5 {
                            latencies.push(t.elapsed().as_secs_f64() * 1000.0);
                        }
                    }
                    Err(_) => errors += 1,
                }
            }
            (latencies, errors)
        }));
    }

    let mut all_latencies: Vec<f64> = Vec::with_capacity(client_count * queries_per_client);
    let mut total_errors = 0usize;
    for handle in handles {
        let (latencies, errors) = handle.join().unwrap_or((Vec::new(), 0));
        all_latencies.extend(latencies);
        total_errors += errors;
    }
    let bench_ms = bench_start.elapsed().as_secs_f64() * 1000.0;

    let stats = percentiles(&mut all_latencies);
    let total_queries = client_count * queries_per_client;
    let throughput = total_queries as f64 / (bench_ms / 1000.0);

    let report = serde_json::json!({
        "config": {
            "clients": client_count,
            "queries_per_client": queries_per_client,
            "total_queries": total_queries,
            "seed_memories": seed_memories,
            "db_path": db_path.display().to_string(),
        },
        "results": {
            "wall_clock_ms": format!("{:.2}", bench_ms),
            "throughput_qps": format!("{:.1}", throughput),
            "errors": total_errors,
            "search_latency_ms": {
                "p50": format!("{:.2}", stats.p50),
                "p95": format!("{:.2}", stats.p95),
                "p99": format!("{:.2}", stats.p99),
                "avg": format!("{:.2}", stats.avg),
            },
        },
    });

    println!("{}", serde_json::to_string_pretty(&report).unwrap());

    std::fs::remove_dir_all(&tmp_dir).ok();
}

struct Percentiles {
    p50: f64,
    p95: f64,
    p99: f64,
    avg: f64,
}

fn percentiles(values: &mut [f64]) -> Percentiles {
    if values.is_empty() {
        return Percentiles { p50: 0.0, p95: 0.0, p99: 0.0, avg: 0.0 };
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |p: f64| -> f64 {
        let idx = ((values.len() as f64 - 1.0) * p).round() as usize;
        values[idx.min(values.len() - 1)]
    };
    let avg = values.iter().sum::<f64>() / values.len() as f64;
    Percentiles { p50: pick(0.50), p95: pick(0.95), p99: pick(0.99), avg }
}

fn print_help() {
    println!(
        "MemoryPilot v{} — MCP memory server with SQLite FTS5",
        VERSION
    );
    println!();
    println!("USAGE:");
    println!("  MemoryPilot                  Start MCP stdio server");
    println!("  MemoryPilot --migrate        Migrate v1 JSON data to SQLite");
    println!("  MemoryPilot --backfill       Compute missing embeddings");
    println!(
        "  MemoryPilot --backfill-force Recompute ALL embeddings (use after switching engine)"
    );
    println!("  MemoryPilot --http [PORT]    Start HTTP server (default: 7437, requires --features http)");
    println!("  MemoryPilot --benchmark-recall [--scenario-limit N]");
    println!("  MemoryPilot --benchmark-search [--scenario-limit N]   Search quality: R@5, R@10, NDCG@10, cluster coherence");
    println!("  MemoryPilot --benchmark-longmemeval [PATH] [--limit N] [--min-r5 PCT]  LongMemEval retrieval benchmark");
    println!("  MemoryPilot --benchmark-fr [--min-r5 PCT]  French retrieval benchmark (30 in-memory scenarios)");
    println!("  MemoryPilot --benchmark-concurrency [--clients N] [--queries-per-client N] [--seed-memories N]  Multi-client concurrent search throughput");
    println!("  MemoryPilot --benchmark-latency [--queries N] [--seed-memories N] [--db PATH]  Cold/warm search latency");
    println!("  MemoryPilot --version        Show version");
    println!("  MemoryPilot --help           Show this help");
    println!();
    println!("MCP TOOLS (41):");
    println!("  recall              Load all context in one shot (start here)");
    println!("  remember_working    Store ephemeral session scratchpad context");
    println!("  recall_working      Recall current in-process working memory");
    println!("  clear_working       Clear scoped working memory");
    println!("  get_project_brain   Instant project summary (<1500 tokens)");
    println!("  search_memory       Hybrid BM25 + fastembed RRF search");
    println!("  get_file_context    Memories related to recently modified files");
    println!("  add_memory          Store with auto-dedup, entities, graph links");
    println!("  add_memories        Bulk add multiple memories in 1 call");
    println!("  add_transcript      Chunk and store long transcripts");
    println!("  ingest_session      Distill local session transcripts without raw index pollution");
    println!("  get_memory          Retrieve by ID");
    println!("  update_memory       Update content/kind/tags/importance/TTL");
    println!("  delete_memory       Delete by ID (cascades links/entities)");
    println!("  list_memories       List with filters & pagination");
    println!("  get_project_context Full context in 1 call + auto-detect");
    println!("  register_project    Register project path for auto-detection");
    println!("  list_projects       List projects with counts");
    println!("  get_stats           Database statistics");
    println!("  benchmark_recall    Measure recall quality with golden scenarios");
    println!("  get_global_prompt   Auto-discover GLOBAL_PROMPT.md");
    println!("  export_memories     Export as JSON or Markdown");
    println!("  set_config          Set config values");
    println!("  run_gc              Garbage collection: merge, clean, vacuum");
    println!("  cleanup_expired     Remove expired memories");
    println!("  migrate_v1          Import from v1 JSON files");
    println!("  toggle_auto_lint    Enable or disable self-healing lint memory");
    println!("  get_file_context    Load memories for recent file changes");
    println!("  kg_add              Add a fact triple to the knowledge graph");
    println!("  kg_invalidate       Mark a fact as ended/expired");
    println!("  kg_query            Query entity relationships with temporal filter");
    println!("  kg_timeline         Chronological story of an entity");
    println!("  kg_stats            Knowledge graph overview");
    println!();
    println!("STORAGE:    ~/.MemoryPilot/memory.db");
    println!("SEARCH:     Hybrid BM25 + vector RRF + KG boost + watcher context");
    println!("EMBEDDINGS: fastembed (multilingual-e5-small, 384-dim, 100+ languages)");
    println!("BUILT BY:   SOFLUTION LTD");
}

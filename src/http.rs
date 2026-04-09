/// MemoryPilot v4.0 — Optional HTTP Server.
/// Multi-threaded worker pool — each worker owns its own DB connection.
/// Feature-gated behind `http` feature flag.

#[cfg(feature = "http")]
use std::sync::Arc;

#[cfg(feature = "http")]
pub fn start_http_server(_db: Arc<crate::db::Database>, port: u16) {
    let addr = format!("0.0.0.0:{}", port);
    eprintln!("[MemoryPilot] HTTP server starting on http://{}...", addr);

    let server = match tiny_http::Server::http(&addr) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("[MemoryPilot] HTTP server failed to start: {}", e);
            return;
        }
    };

    eprintln!("[MemoryPilot] HTTP server listening on port {}", port);

    let num_workers = 4;
    let mut handles = Vec::new();
    for i in 0..num_workers {
        let srv = Arc::clone(&server);
        handles.push(std::thread::spawn(move || {
            let local_db = match crate::db::Database::open() {
                Ok(d) => d,
                Err(e) => { eprintln!("[HTTP worker {}] DB open failed: {}", i, e); return; }
            };
            for mut request in srv.incoming_requests() {
                let response = match (request.method(), request.url()) {
                    (&tiny_http::Method::Get, "/health") => handle_health(),
                    (&tiny_http::Method::Post, "/tools/call") => handle_tool_call(&local_db, &mut request),
                    (&tiny_http::Method::Options, _) => cors_preflight(),
                    _ => {
                        let body = r#"{"error":"Not found. Use POST /tools/call or GET /health"}"#;
                        tiny_http::Response::from_string(body)
                            .with_status_code(404)
                            .with_header(content_type_json())
                    }
                };
                let response = add_cors_headers(response);
                let _ = request.respond(response);
            }
        }));
    }

    for h in handles { let _ = h.join(); }
}

#[cfg(feature = "http")]
fn handle_health() -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let body = serde_json::json!({
        "status": "ok",
        "server": "MemoryPilot",
        "version": env!("CARGO_PKG_VERSION"),
        "embedding_engine": "fastembed (multilingual-e5-small, 384-dim)",
    });
    tiny_http::Response::from_string(serde_json::to_string_pretty(&body).unwrap())
        .with_header(content_type_json())
}

#[cfg(feature = "http")]
fn handle_tool_call(db: &crate::db::Database, request: &mut tiny_http::Request) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let mut body = String::new();
    if let Err(e) = request.as_reader().read_to_string(&mut body) {
        let err = serde_json::json!({"error": format!("Failed to read body: {}", e)});
        return tiny_http::Response::from_string(serde_json::to_string(&err).unwrap())
            .with_status_code(400)
            .with_header(content_type_json());
    }

    let parsed: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            let err = serde_json::json!({"error": format!("Invalid JSON: {}", e)});
            return tiny_http::Response::from_string(serde_json::to_string(&err).unwrap())
                .with_status_code(400)
                .with_header(content_type_json());
        }
    };

    let name = parsed.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let args = parsed.get("arguments").cloned().unwrap_or(serde_json::json!({}));

    if name.is_empty() {
        let err = serde_json::json!({"error": "Missing 'name' field. Expected: {\"name\": \"tool_name\", \"arguments\": {...}}"});
        return tiny_http::Response::from_string(serde_json::to_string(&err).unwrap())
            .with_status_code(400)
            .with_header(content_type_json());
    }

    let result = crate::tools::handle_tool_call(db, name, &args);
    tiny_http::Response::from_string(serde_json::to_string_pretty(&result).unwrap())
        .with_header(content_type_json())
}

#[cfg(feature = "http")]
fn cors_preflight() -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string("")
        .with_status_code(204)
        .with_header(content_type_json())
}

#[cfg(feature = "http")]
fn content_type_json() -> tiny_http::Header {
    "Content-Type: application/json".parse().unwrap()
}

#[cfg(feature = "http")]
fn add_cors_headers(response: tiny_http::Response<std::io::Cursor<Vec<u8>>>) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    response
        .with_header("Access-Control-Allow-Origin: *".parse::<tiny_http::Header>().unwrap())
        .with_header("Access-Control-Allow-Methods: GET, POST, OPTIONS".parse::<tiny_http::Header>().unwrap())
        .with_header("Access-Control-Allow-Headers: Content-Type".parse::<tiny_http::Header>().unwrap())
}

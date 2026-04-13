/// MemoryPilot v4.0 — Optional HTTP Server.
/// Multi-threaded worker pool — each worker owns its own DB connection.
/// Supports both REST API and MCP Streamable HTTP transport (POST/GET /mcp).
/// Feature-gated behind `http` feature flag.

#[cfg(feature = "http")]
use std::sync::Arc;

#[cfg(feature = "http")]
pub fn start_http_server(_db: Arc<crate::db::Database>, port: u16) {
    let mut actual_port = port;
    let mut server_result = None;
    for attempt in 0..10u16 {
        let try_port = port.saturating_add(attempt);
        let addr = format!("127.0.0.1:{}", try_port);
        match tiny_http::Server::http(&addr) {
            Ok(s) => {
                actual_port = try_port;
                server_result = Some(s);
                break;
            }
            Err(e) => {
                if attempt < 9 {
                    eprintln!("[MemoryPilot] Port {} in use, trying {}...", try_port, try_port + 1);
                } else {
                    eprintln!("[MemoryPilot] HTTP server failed to start (ports {}-{} all in use): {}", port, try_port, e);
                    return;
                }
            }
        }
    }
    let server = Arc::new(server_result.unwrap());

    eprintln!("[MemoryPilot] HTTP server listening on http://127.0.0.1:{}", actual_port);
    eprintln!("[MemoryPilot] MCP Streamable HTTP endpoint: http://localhost:{}/mcp", actual_port);

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
                let url = request.url().to_string();
                let origin = request.headers().iter()
                    .find(|h| h.field.equiv("Origin"))
                    .map(|h| h.value.as_str().to_string());
                let response = match (request.method(), url.as_str()) {
                    (&tiny_http::Method::Get, "/health") => handle_health(),
                    (&tiny_http::Method::Post, "/tools/call") => handle_tool_call(&local_db, &mut request),
                    (&tiny_http::Method::Post, "/mcp") => handle_mcp_post(&local_db, &mut request),
                    (&tiny_http::Method::Get, "/mcp") => handle_mcp_sse(),
                    (&tiny_http::Method::Delete, "/mcp") => handle_mcp_session_delete(),
                    (&tiny_http::Method::Options, _) => cors_preflight(),
                    _ => {
                        let body = r#"{"error":"Not found. Endpoints: POST /mcp (MCP Streamable HTTP), POST /tools/call (REST), GET /health"}"#;
                        tiny_http::Response::from_string(body)
                            .with_status_code(404)
                            .with_header(content_type_json())
                    }
                };
                let response = add_cors_headers(response, origin.as_deref());
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

/// MCP Streamable HTTP — POST /mcp
/// Receives JSON-RPC requests, routes them through the same handler as stdio.
/// Returns application/json with the JSON-RPC response.
#[cfg(feature = "http")]
fn handle_mcp_post(db: &crate::db::Database, request: &mut tiny_http::Request) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let mut body = String::new();
    if let Err(e) = request.as_reader().read_to_string(&mut body) {
        let err = crate::protocol::JsonRpcResponse::error(None, -32700, format!("Read error: {}", e));
        return tiny_http::Response::from_string(serde_json::to_string(&err).unwrap())
            .with_status_code(400)
            .with_header(content_type_json());
    }

    let parsed: crate::protocol::JsonRpcRequest = match serde_json::from_str(&body) {
        Ok(r) => r,
        Err(e) => {
            let err = crate::protocol::JsonRpcResponse::error(None, -32700, format!("Parse error: {}", e));
            return tiny_http::Response::from_string(serde_json::to_string(&err).unwrap())
                .with_status_code(400)
                .with_header(content_type_json());
        }
    };

    let is_notification = parsed.id.is_none();
    let response = handle_mcp_request(db, &parsed);

    if is_notification {
        return tiny_http::Response::from_string("")
            .with_status_code(202)
            .with_header(content_type_json());
    }

    tiny_http::Response::from_string(serde_json::to_string(&response).unwrap())
        .with_header(content_type_json())
}

/// Route a JSON-RPC request through the MCP handler (same logic as stdio).
#[cfg(feature = "http")]
fn handle_mcp_request(db: &crate::db::Database, req: &crate::protocol::JsonRpcRequest) -> crate::protocol::JsonRpcResponse {
    use serde_json::json;
    match req.method.as_str() {
        "initialize" => crate::protocol::JsonRpcResponse::success(req.id.clone(), json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": { "name": "MemoryPilot", "version": env!("CARGO_PKG_VERSION") },
            "instructions": "CRITICAL WORKFLOW:\n1. Always call 'recall' tool at the start of a conversation.\n2. DURING the conversation, you MUST proactively call 'add_memory' to store any new architecture decision, convention, or significant bug fix. Do NOT ask the user for permission — act as an autonomous technical secretary.\n3. NEVER store secrets, passwords, API keys, or tokens in memory. Use environment variables or secret managers for credentials."
        })),
        "notifications/initialized" => crate::protocol::JsonRpcResponse::success(req.id.clone(), json!({})),
        "tools/list" => crate::protocol::JsonRpcResponse::success(req.id.clone(), crate::tools::tool_definitions()),
        "tools/call" => {
            let name = req.params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = req.params.get("arguments").cloned().unwrap_or(json!({}));
            crate::protocol::JsonRpcResponse::success(req.id.clone(), crate::tools::handle_tool_call(db, name, &args))
        }
        "ping" => crate::protocol::JsonRpcResponse::success(req.id.clone(), json!({})),
        _ => crate::protocol::JsonRpcResponse::error(req.id.clone(), -32601, format!("Unknown method: {}", req.method)),
    }
}

/// MCP Streamable HTTP — GET /mcp
/// Opens a server-to-client SSE stream. For stateless servers like MemoryPilot,
/// we send a keepalive comment and hold the connection.
#[cfg(feature = "http")]
fn handle_mcp_sse() -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let body = ": MemoryPilot SSE stream\nretry: 5000\n\n";
    tiny_http::Response::from_string(body)
        .with_header("Content-Type: text/event-stream".parse::<tiny_http::Header>().unwrap())
        .with_header("Cache-Control: no-cache".parse::<tiny_http::Header>().unwrap())
        .with_header("X-Accel-Buffering: no".parse::<tiny_http::Header>().unwrap())
}

/// MCP Streamable HTTP — DELETE /mcp
/// Session termination. We're stateless, so just acknowledge.
#[cfg(feature = "http")]
fn handle_mcp_session_delete() -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string("")
        .with_status_code(200)
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
fn is_localhost_origin(origin: &str) -> bool {
    let lower = origin.to_lowercase();
    lower == "http://localhost" || lower.starts_with("http://localhost:")
        || lower == "http://127.0.0.1" || lower.starts_with("http://127.0.0.1:")
}

#[cfg(feature = "http")]
fn add_cors_headers(response: tiny_http::Response<std::io::Cursor<Vec<u8>>>, origin: Option<&str>) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let allowed_origin = match origin {
        Some(o) if is_localhost_origin(o) => o.to_string(),
        _ => "http://localhost".to_string(),
    };
    response
        .with_header(format!("Access-Control-Allow-Origin: {}", allowed_origin).parse::<tiny_http::Header>().unwrap())
        .with_header("Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS".parse::<tiny_http::Header>().unwrap())
        .with_header("Access-Control-Allow-Headers: Content-Type, MCP-Protocol-Version, MCP-Session-Id".parse::<tiny_http::Header>().unwrap())
}

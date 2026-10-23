//! The REST API against a real Postgres (`scripts/test-postgres.sh`), called
//! without the HTTP layer. Each test works in tenants it creates.

use memorypilot_server::api::{Api, Request, Response};
use memorypilot_server::{Permission, PgAdmin, PgConnection, PgPool, TenantLimits};
use serde_json::{json, Value};
use uuid::Uuid;

const NEEDS_POSTGRES: &str = "needs Postgres with pgvector: scripts/test-postgres.sh";

fn plain() -> PgConnection {
    let url = std::env::var("MEMORYPILOT_TEST_PG_URL")
        .unwrap_or_else(|_| panic!("MEMORYPILOT_TEST_PG_URL is not set ({})", NEEDS_POSTGRES));
    PgConnection::plain(&url).expect("url")
}

/// One tenant with one space, a key with every permission and a read-only key.
struct App {
    api: Api,
    tenant: Uuid,
    full: String,
    reader: String,
}

fn app() -> App {
    let mut admin = PgAdmin::connect(&plain()).expect("admin connection");
    admin.migrate().expect("migrations");
    let tenant = admin
        .create_tenant(&format!("api-{}", Uuid::new_v4()))
        .expect("tenant");
    let space = admin.create_space(tenant, "app").expect("space");
    let mut key = |permissions: &[Permission]| {
        admin
            .create_api_key(tenant, space, "test", permissions)
            .expect("key")
            .secret
    };
    App {
        full: key(&Permission::ALL),
        reader: key(&[Permission::Read]),
        api: Api::new(PgPool::new(plain(), 2)),
        tenant,
    }
}

impl App {
    fn call(
        &self,
        method: &str,
        url: &str,
        key: Option<&str>,
        subject: Option<&str>,
        body: &str,
    ) -> Response {
        let authorization = key.map(|key| format!("Bearer {}", key));
        let mut headers = Vec::new();
        if let Some(authorization) = &authorization {
            headers.push(("authorization", authorization.as_str()));
        }
        if let Some(subject) = subject {
            headers.push(("x-memorypilot-subject", subject));
        }
        self.api.handle(&Request {
            method,
            url,
            headers: &headers,
            body: body.as_bytes(),
        })
    }

    fn send(&self, method: &str, url: &str, subject: Option<&str>, body: Value) -> Response {
        let body = if body.is_null() {
            String::new()
        } else {
            body.to_string()
        };
        self.call(method, url, Some(&self.full), subject, &body)
    }

    fn add(&self, subject: Option<&str>, content: &str) -> String {
        let response = self.send(
            "POST",
            "/v1/memories",
            subject,
            json!({ "content": content, "kind": "fact" }),
        );
        assert_eq!(response.status, 201, "{}", response.body);
        id_of(&response)
    }

    fn listed(&self, subject: Option<&str>) -> Vec<String> {
        let response = self.send("GET", "/v1/memories", subject, Value::Null);
        assert_eq!(response.status, 200, "{}", response.body);
        let mut contents: Vec<String> = response.body["memories"]
            .as_array()
            .expect("memories")
            .iter()
            .map(|memory| memory["content"].as_str().expect("content").to_string())
            .collect();
        contents.sort();
        contents
    }
}

fn id_of(response: &Response) -> String {
    response.body["memory"]["id"]
        .as_str()
        .expect("memory id")
        .to_string()
}

fn error_of(response: &Response) -> &str {
    response.body["error"].as_str().unwrap_or("")
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn health_is_open_and_every_route_needs_a_live_key() {
    let app = app();
    assert_eq!(app.call("GET", "/health", None, None, "").status, 200);
    for authorization in [None, Some("garbage"), Some("mpk_short")] {
        let response = app.call("GET", "/v1/memories", authorization, None, "");
        assert_eq!(response.status, 401, "{:?}", authorization);
    }
    let basic = format!("Basic {}", app.full);
    let raw = app.api.handle(&Request {
        method: "GET",
        url: "/v1/memories",
        headers: &[("Authorization", basic.as_str())],
        body: b"",
    });
    assert_eq!(raw.status, 401, "only Bearer");
    assert_eq!(
        app.call("GET", "/v1/nothing", Some(&app.full), None, "")
            .status,
        404
    );
    assert_eq!(
        app.call("PUT", "/v1/memories", Some(&app.full), None, "")
            .status,
        404
    );
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn memories_round_trip_through_the_api() {
    let app = app();
    let content = "Invoices are rounded to the cent with banker's rounding";
    let created = app.send(
        "POST",
        "/v1/memories",
        None,
        json!({ "content": content, "project": "billing", "tags": ["money"] }),
    );
    assert_eq!(created.status, 201, "{}", created.body);
    assert_eq!(created.body["merged"], false);
    let id = id_of(&created);
    assert_eq!(created.body["memory"]["source"], "api");

    let again = app.send(
        "POST",
        "/v1/memories",
        None,
        json!({ "content": content, "project": "billing" }),
    );
    assert_eq!(
        (again.status, id_of(&again)),
        (200, id.clone()),
        "a duplicate merges"
    );
    assert_eq!(again.body["merged"], true);

    let fetched = app.send("GET", &format!("/v1/memories/{}", id), None, Value::Null);
    assert_eq!(fetched.status, 200);
    assert_eq!(fetched.body["memory"]["content"], content);

    let listed = app.send(
        "GET",
        "/v1/memories?project=billing&limit=5",
        None,
        Value::Null,
    );
    assert_eq!(
        (listed.status, listed.body["total"].clone()),
        (200, json!(1))
    );
    let other_project = app.send("GET", "/v1/memories?project=other%20one", None, Value::Null);
    assert_eq!(other_project.body["total"], 0);

    let edited = app.send(
        "PATCH",
        &format!("/v1/memories/{}", id),
        None,
        json!({ "content": "Invoices are rounded half up to the cent", "importance": 5 }),
    );
    assert_eq!(edited.status, 200, "{}", edited.body);
    assert_eq!(edited.body["memory"]["importance"], 5);

    let found = app.send(
        "POST",
        "/v1/search",
        None,
        json!({ "query": "how are invoices rounded", "limit": 3 }),
    );
    assert_eq!(found.status, 200, "{}", found.body);
    assert_eq!(found.body["results"][0]["memory"]["id"], json!(id));
    assert!(found.body["confidence"].is_object());

    let deleted = app.send("DELETE", &format!("/v1/memories/{}", id), None, Value::Null);
    assert_eq!(deleted.status, 200);
    assert_eq!(
        app.send("GET", &format!("/v1/memories/{}", id), None, Value::Null)
            .status,
        404
    );
    assert_eq!(
        app.send("DELETE", &format!("/v1/memories/{}", id), None, Value::Null)
            .status,
        404
    );
    assert_eq!(
        app.send(
            "PATCH",
            &format!("/v1/memories/{}", id),
            None,
            json!({ "importance": 2 })
        )
        .status,
        404
    );
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn key_permissions_are_enforced() {
    let app = app();
    let id = app.add(None, "The staging cluster restarts every Sunday night");
    let reader = Some(app.reader.as_str());
    assert_eq!(
        app.call("GET", &format!("/v1/memories/{}", id), reader, None, "")
            .status,
        200
    );
    assert_eq!(
        app.call(
            "POST",
            "/v1/search",
            reader,
            None,
            r#"{"query":"staging restart"}"#
        )
        .status,
        200
    );
    for (method, url, body) in [
        (
            "POST",
            "/v1/memories".to_string(),
            r#"{"content":"Readers cannot write"}"#,
        ),
        (
            "PATCH",
            format!("/v1/memories/{}", id),
            r#"{"importance":1}"#,
        ),
        ("DELETE", format!("/v1/memories/{}", id), ""),
    ] {
        let response = app.call(method, &url, reader, None, body);
        assert_eq!(response.status, 403, "{} {}", method, url);
        assert!(error_of(&response).contains("'write'"));
    }
    let as_end_user = app.call("GET", "/v1/memories", reader, Some("alice"), "");
    assert_eq!(as_end_user.status, 403);
    assert!(error_of(&as_end_user).contains("'subjects'"));
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn end_users_see_their_memories_and_the_shared_ones_but_change_only_theirs() {
    let app = app();
    let shared = app.add(None, "Support answers within one business day");
    let alices = app.add(Some("alice"), "Alice prefers invoices in English");
    app.add(Some("bob"), "Bob works from Lisbon");

    assert_eq!(
        app.listed(Some("alice")),
        [
            "Alice prefers invoices in English",
            "Support answers within one business day"
        ]
    );
    assert_eq!(
        app.listed(Some("bob")),
        [
            "Bob works from Lisbon",
            "Support answers within one business day"
        ]
    );
    assert_eq!(
        app.listed(None),
        ["Support answers within one business day"]
    );

    let bob = Some("bob");
    assert_eq!(
        app.send("GET", &format!("/v1/memories/{}", alices), bob, Value::Null)
            .status,
        404
    );
    assert_eq!(
        app.send(
            "PATCH",
            &format!("/v1/memories/{}", alices),
            bob,
            json!({ "importance": 1 })
        )
        .status,
        404
    );
    assert_eq!(
        app.send(
            "DELETE",
            &format!("/v1/memories/{}", alices),
            bob,
            Value::Null
        )
        .status,
        404
    );
    let edit_shared = app.send(
        "PATCH",
        &format!("/v1/memories/{}", shared),
        bob,
        json!({ "content": "Bob rewrote this" }),
    );
    assert_eq!(edit_shared.status, 403, "{}", edit_shared.body);
    assert_eq!(
        app.send(
            "DELETE",
            &format!("/v1/memories/{}", shared),
            bob,
            Value::Null
        )
        .status,
        403
    );
    assert!(app
        .listed(None)
        .contains(&"Support answers within one business day".to_string()));

    let found = app.send(
        "POST",
        "/v1/search",
        bob,
        json!({ "query": "invoices in English" }),
    );
    assert!(
        found.body["results"]
            .as_array()
            .expect("results")
            .iter()
            .all(|result| result["memory"]["id"] != json!(alices)),
        "search never returns another end user's memory"
    );
    assert_eq!(
        app.send("GET", "/v1/memories", Some(""), Value::Null)
            .status,
        400
    );
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn another_tenant_sees_nothing_through_the_api() {
    let (first, second) = (app(), app());
    let id = first.add(None, "Deployments freeze on Fridays");
    assert_eq!(
        second
            .send("GET", &format!("/v1/memories/{}", id), None, Value::Null)
            .status,
        404
    );
    assert_eq!(
        second
            .send("DELETE", &format!("/v1/memories/{}", id), None, Value::Null)
            .status,
        404
    );
    assert!(second.listed(None).is_empty());
    let found = second.send(
        "POST",
        "/v1/search",
        None,
        json!({ "query": "deployments freeze" }),
    );
    assert_eq!(found.body["results"], json!([]));
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn invalid_input_is_refused_with_a_reason() {
    let app = app();
    let cases: [(&str, &str, &str, u16); 12] = [
        ("POST", "/v1/memories", "not json", 400),
        ("POST", "/v1/memories", r#"{"content":"   "}"#, 400),
        (
            "POST",
            "/v1/memories",
            r#"{"content":"x","colour":"red"}"#,
            400,
        ),
        (
            "POST",
            "/v1/memories",
            r#"{"content":"x","kind":"poem"}"#,
            400,
        ),
        (
            "POST",
            "/v1/memories",
            r#"{"content":"x","kind":"credential"}"#,
            422,
        ),
        (
            "POST",
            "/v1/memories",
            r#"{"content":"x","importance":9}"#,
            400,
        ),
        (
            "POST",
            "/v1/memories",
            r#"{"content":"x","expires_at":"tomorrow"}"#,
            400,
        ),
        (
            "POST",
            "/v1/memories",
            r#"{"content":"x","metadata":[1]}"#,
            400,
        ),
        (
            "POST",
            "/v1/memories",
            r#"{"content":"Staging password: Zq8!vT3#pLm2"}"#,
            422,
        ),
        ("POST", "/v1/search", r#"{"query":"x","limit":500}"#, 400),
        ("GET", "/v1/memories?limit=1000", "", 400),
        ("GET", "/v1/memories?kind=poem", "", 400),
    ];
    for (method, url, body, status) in cases {
        let response = app.call(method, url, Some(&app.full), None, body);
        assert_eq!(
            response.status, status,
            "{} {} {}: {}",
            method, url, body, response.body
        );
        assert!(!error_of(&response).is_empty(), "a reason is given");
        assert!(
            !error_of(&response).contains("Postgres"),
            "no database detail"
        );
    }
    let too_long = json!({ "content": "a".repeat(64 * 1024 + 1) }).to_string();
    assert_eq!(
        app.call("POST", "/v1/memories", Some(&app.full), None, &too_long)
            .status,
        400
    );
    assert!(app.listed(None).is_empty(), "nothing was stored");
}

const MODERN: &str = "2026-07-28";

impl App {
    fn mcp(&self, key: &str, headers: &[(&str, &str)], body: Value) -> Response {
        let authorization = format!("Bearer {}", key);
        let mut all = vec![("Authorization", authorization.as_str())];
        all.extend_from_slice(headers);
        let body = body.to_string();
        self.api.handle(&Request {
            method: "POST",
            url: "/mcp",
            headers: &all,
            body: body.as_bytes(),
        })
    }

    /// A legacy request (no `_meta` version), as sent after `initialize`.
    fn legacy(&self, key: &str, method: &str, params: Value) -> Response {
        self.mcp(
            key,
            &[],
            json!({ "jsonrpc": "2.0", "id": 1, "method": method, "params": params }),
        )
    }

    fn legacy_tool(&self, key: &str, subject: Option<&str>, name: &str, arguments: Value) -> Value {
        let mut headers = Vec::new();
        if let Some(subject) = subject {
            headers.push(("X-MemoryPilot-Subject", subject));
        }
        let response = self.mcp(
            key,
            &headers,
            json!({ "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                    "params": { "name": name, "arguments": arguments } }),
        );
        assert_eq!(response.status, 200, "{}", response.body);
        response.body["result"].clone()
    }

    /// A modern request: version and capabilities in `_meta`, mirrored in headers.
    fn modern(&self, method: &str, mut params: Value, headers: &[(&str, &str)]) -> Response {
        params["_meta"] = json!({
            "io.modelcontextprotocol/protocolVersion": MODERN,
            "io.modelcontextprotocol/clientInfo": { "name": "test", "version": "1" },
            "io.modelcontextprotocol/clientCapabilities": {},
        });
        self.mcp(
            &self.full,
            headers,
            json!({ "jsonrpc": "2.0", "id": "m-1", "method": method, "params": params }),
        )
    }
}

fn tool_text(result: &Value) -> Value {
    let text = result["content"][0]["text"].as_str().expect("text content");
    serde_json::from_str(text).unwrap_or_else(|_| json!(text))
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn legacy_mcp_clients_initialize_list_and_call_tools() {
    let app = app();
    let init = app.legacy(
        &app.full,
        "initialize",
        json!({ "protocolVersion": "2025-06-18", "capabilities": {} }),
    );
    assert_eq!(init.status, 200, "{}", init.body);
    assert_eq!(init.body["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(init.body["result"]["serverInfo"]["name"], "MemoryPilot");
    let old = app.legacy(
        &app.full,
        "initialize",
        json!({ "protocolVersion": "2024-11-05" }),
    );
    assert_eq!(
        old.body["result"]["protocolVersion"], "2025-11-25",
        "latest legacy offered"
    );

    let notification = app.mcp(
        &app.full,
        &[],
        json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
    );
    assert_eq!(
        (notification.status, notification.body.clone()),
        (202, Value::Null)
    );

    let names = |key: &str| -> Vec<String> {
        let listed = app.legacy(key, "tools/list", json!({}));
        listed.body["result"]["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .map(|tool| tool["name"].as_str().expect("name").to_string())
            .collect()
    };
    assert_eq!(names(&app.full).len(), 6);
    assert_eq!(
        names(&app.reader),
        ["search_memory", "list_memories", "get_memory"],
        "only what the key allows"
    );

    let added = app.legacy_tool(
        &app.full,
        None,
        "add_memory",
        json!({ "content": "Refunds are processed within five days", "kind": "fact" }),
    );
    assert_eq!(added["isError"], false, "{}", added);
    let id = tool_text(&added)["memory"]["id"]
        .as_str()
        .expect("id")
        .to_string();
    let found = app.legacy_tool(
        &app.reader,
        None,
        "search_memory",
        json!({ "query": "how long do refunds take" }),
    );
    assert_eq!(tool_text(&found)["results"][0]["memory"]["id"], json!(id));
    let edited = app.legacy_tool(
        &app.full,
        None,
        "update_memory",
        json!({ "id": id, "importance": 5 }),
    );
    assert_eq!(tool_text(&edited)["memory"]["importance"], 5);
    let listed = app.legacy_tool(&app.full, None, "list_memories", json!({ "limit": 5 }));
    assert_eq!(tool_text(&listed)["total"], 1);

    let refused = app.legacy_tool(&app.reader, None, "delete_memory", json!({ "id": id }));
    assert_eq!(refused["isError"], true);
    assert!(tool_text(&refused)
        .as_str()
        .expect("message")
        .contains("'write'"));
    let invalid = app.legacy_tool(
        &app.full,
        None,
        "add_memory",
        json!({ "content": "x", "importance": 9 }),
    );
    assert_eq!(
        invalid["isError"], true,
        "validation errors reach the model"
    );
    let deleted = app.legacy_tool(&app.full, None, "delete_memory", json!({ "id": id }));
    assert_eq!(deleted["isError"], false);
    let gone = app.legacy_tool(&app.full, None, "get_memory", json!({ "id": id }));
    assert_eq!(gone["isError"], true);

    let unknown_tool = app.legacy(
        &app.full,
        "tools/call",
        json!({ "name": "drop_everything" }),
    );
    assert_eq!(unknown_tool.body["error"]["code"], -32602);
    let unknown_method = app.legacy(&app.full, "resources/list", json!({}));
    assert_eq!(unknown_method.body["error"]["code"], -32601);

    let versioned = app.mcp(
        &app.full,
        &[("MCP-Protocol-Version", "2025-06-18")],
        json!({ "jsonrpc": "2.0", "id": 2, "method": "ping" }),
    );
    assert_eq!(versioned.status, 200);
    let unknown_version = app.mcp(
        &app.full,
        &[("MCP-Protocol-Version", "1999-01-01")],
        json!({ "jsonrpc": "2.0", "id": 3, "method": "ping" }),
    );
    assert_eq!(
        (
            unknown_version.status,
            unknown_version.body["error"]["code"].clone()
        ),
        (400, json!(-32022))
    );
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn modern_mcp_requests_carry_their_version_and_matching_headers() {
    let app = app();
    let version = ("MCP-Protocol-Version", MODERN);
    let discovered = app.modern(
        "server/discover",
        json!({}),
        &[version, ("Mcp-Method", "server/discover")],
    );
    assert_eq!(discovered.status, 200, "{}", discovered.body);
    assert_eq!(discovered.body["result"]["resultType"], "complete");
    assert_eq!(discovered.body["result"]["supportedVersions"][0], MODERN);
    assert_eq!(
        discovered.body["result"]["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
        "MemoryPilot"
    );

    let call = json!({ "name": "add_memory", "arguments": { "content": "The office closes at six", "kind": "fact" } });
    let called = app.modern(
        "tools/call",
        call.clone(),
        &[
            version,
            ("Mcp-Method", "tools/call"),
            ("Mcp-Name", "add_memory"),
        ],
    );
    assert_eq!(called.status, 200, "{}", called.body);
    assert_eq!(called.body["result"]["isError"], false);
    let encoded = app.modern(
        "tools/call",
        json!({ "name": "list_memories", "arguments": {} }),
        &[
            version,
            ("Mcp-Method", "tools/call"),
            ("Mcp-Name", "=?base64?bGlzdF9tZW1vcmllcw==?="),
        ],
    );
    assert_eq!(encoded.body["result"]["isError"], false, "{}", encoded.body);

    let rejected = |response: Response, code: i64, what: &str| {
        assert_eq!(response.status, 400, "{}: {}", what, response.body);
        assert_eq!(
            response.body["error"]["code"], code,
            "{}: {}",
            what, response.body
        );
        assert_eq!(response.body["id"], "m-1", "{}", what);
    };
    rejected(
        app.modern(
            "tools/call",
            call.clone(),
            &[
                version,
                ("Mcp-Method", "tools/call"),
                ("Mcp-Name", "delete_memory"),
            ],
        ),
        -32020,
        "Mcp-Name differs",
    );
    rejected(
        app.modern(
            "tools/call",
            call.clone(),
            &[version, ("Mcp-Method", "tools/call")],
        ),
        -32020,
        "Mcp-Name missing",
    );
    rejected(
        app.modern(
            "tools/list",
            json!({}),
            &[version, ("Mcp-Method", "tools/call")],
        ),
        -32020,
        "Mcp-Method differs",
    );
    rejected(
        app.modern("tools/list", json!({}), &[("Mcp-Method", "tools/list")]),
        -32020,
        "version header missing",
    );

    let mut future = json!({ "_meta": { "io.modelcontextprotocol/protocolVersion": "2030-01-01", "io.modelcontextprotocol/clientCapabilities": {} } });
    let unsupported = app.mcp(
        &app.full,
        &[
            ("MCP-Protocol-Version", "2030-01-01"),
            ("Mcp-Method", "ping"),
        ],
        json!({ "jsonrpc": "2.0", "id": "m-1", "method": "ping", "params": future.take() }),
    );
    assert!(unsupported.body["error"]["data"]["supported"]
        .as_array()
        .expect("supported")
        .contains(&json!(MODERN)));
    rejected(unsupported, -32022, "unsupported version");

    let no_capabilities = app.mcp(
        &app.full,
        &[version, ("Mcp-Method", "ping")],
        json!({ "jsonrpc": "2.0", "id": "m-1", "method": "ping", "params": { "_meta": { "io.modelcontextprotocol/protocolVersion": MODERN } } }),
    );
    rejected(no_capabilities, -32602, "clientCapabilities missing");

    let unknown = app.modern(
        "resources/list",
        json!({}),
        &[version, ("Mcp-Method", "resources/list")],
    );
    assert_eq!(
        (unknown.status, unknown.body["error"]["code"].clone()),
        (404, json!(-32601))
    );
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn the_mcp_endpoint_guards_its_transport() {
    let app = app();
    for method in ["GET", "DELETE"] {
        assert_eq!(
            app.call(method, "/mcp", Some(&app.full), None, "").status,
            405,
            "{}",
            method
        );
    }
    let ping = json!({ "jsonrpc": "2.0", "id": 1, "method": "ping" }).to_string();
    assert_eq!(app.call("POST", "/mcp", None, None, &ping).status, 401);
    let from_site = app.mcp(
        &app.full,
        &[("Origin", "https://evil.example")],
        json!({ "jsonrpc": "2.0", "id": 1, "method": "ping" }),
    );
    assert_eq!(from_site.status, 403);
    let allowed =
        Api::new(PgPool::new(plain(), 1)).with_allowed_origins(vec!["https://claude.ai".into()]);
    let authorization = format!("Bearer {}", app.full);
    let from_claude = allowed.handle(&Request {
        method: "POST",
        url: "/mcp",
        headers: &[
            ("Authorization", authorization.as_str()),
            ("Origin", "https://claude.ai"),
        ],
        body: ping.as_bytes(),
    });
    assert_eq!(from_claude.status, 200, "{}", from_claude.body);

    assert_eq!(
        app.call("POST", "/mcp", Some(&app.full), None, "{not json")
            .body["error"]["code"],
        -32700
    );
    let batch = app.call(
        "POST",
        "/mcp",
        Some(&app.full),
        None,
        &format!("[{}]", ping),
    );
    assert_eq!(
        (batch.status, batch.body["error"]["code"].clone()),
        (400, json!(-32600))
    );
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn mcp_tools_act_for_the_end_user_named_by_the_request() {
    let app = app();
    let added = app.legacy_tool(
        &app.full,
        Some("alice"),
        "add_memory",
        json!({ "content": "Alice is allergic to peanuts", "kind": "fact" }),
    );
    assert_eq!(added["isError"], false, "{}", added);
    let alices = app.legacy_tool(&app.full, Some("alice"), "list_memories", json!({}));
    assert_eq!(tool_text(&alices)["total"], 1);
    let bobs = app.legacy_tool(&app.full, Some("bob"), "list_memories", json!({}));
    assert_eq!(tool_text(&bobs)["total"], 0);
    let reader_as_alice = app.legacy_tool(&app.reader, Some("alice"), "list_memories", json!({}));
    assert_eq!(
        reader_as_alice["isError"], true,
        "acting for an end user needs 'subjects'"
    );
}

fn header<'a>(response: &'a Response, name: &str) -> Option<&'a str> {
    response
        .headers
        .iter()
        .find(|(field, _)| field.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

#[test]
#[ignore = "needs Postgres with pgvector: scripts/test-postgres.sh"]
fn rate_and_memory_limits_apply_per_tenant() {
    let (app, other) = (app(), app());
    let mut admin = PgAdmin::connect(&plain()).expect("admin connection");
    let tight = TenantLimits {
        requests_per_minute: Some(3),
        max_memories: Some(1),
    };
    assert!(admin.set_limits(app.tenant, tight).expect("limits"));
    assert!(
        !admin.set_limits(Uuid::new_v4(), tight).expect("limits"),
        "unknown tenant"
    );

    let listed = app.send("GET", "/v1/memories", None, Value::Null);
    assert_eq!(listed.status, 200);
    assert_eq!(header(&listed, "X-RateLimit-Limit"), Some("3"));
    assert_eq!(header(&listed, "X-RateLimit-Remaining"), Some("2"));
    app.add(None, "The first memory fits the quota");
    let over_quota = app.send(
        "POST",
        "/v1/memories",
        None,
        json!({ "content": "The second does not" }),
    );
    assert_eq!(over_quota.status, 403, "{}", over_quota.body);
    assert!(error_of(&over_quota).contains("quota"));

    let throttled = app.send("GET", "/v1/memories", None, Value::Null);
    assert_eq!(throttled.status, 429);
    assert_eq!(header(&throttled, "Retry-After"), Some("20"));
    let mcp_throttled = app.legacy(&app.full, "ping", json!({}));
    assert_eq!(mcp_throttled.status, 429, "MCP shares the key's rate");

    let elsewhere = other.send("GET", "/v1/memories", None, Value::Null);
    assert_eq!(elsewhere.status, 200, "another tenant keeps its own limits");
    assert_eq!(header(&elsewhere, "X-RateLimit-Limit"), Some("600"));
}

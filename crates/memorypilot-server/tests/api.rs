//! The REST API against a real Postgres (`scripts/test-postgres.sh`), called
//! without the HTTP layer. Each test works in tenants it creates.

use memorypilot_server::api::{Api, Request, Response};
use memorypilot_server::{Permission, PgAdmin, PgConnection, PgPool};
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
        self.api.handle(&Request {
            method,
            url,
            authorization: authorization.as_deref(),
            subject,
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
    let raw = app.api.handle(&Request {
        method: "GET",
        url: "/v1/memories",
        authorization: Some(&format!("Basic {}", app.full)),
        subject: None,
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

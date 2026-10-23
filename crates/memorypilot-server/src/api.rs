//! REST API, independent of the HTTP server that carries it.
//!
//! Every route except `/health` needs `Authorization: Bearer <api key>`.
//! `X-MemoryPilot-Subject: <end user id>` acts for one end user of the app
//! (key permission `subjects`): they see their memories and the shared ones,
//! and write only theirs.

use std::collections::HashMap;

use memorypilot_core::db::{
    add_memory_to, hybrid_search, update_memory_in, Memory, MemoryScope, MemoryStore, MEMORY_KINDS,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::limits::RateLimiter;
use crate::{
    ApiKey, Permission, PgPool, PgStore, DEFAULT_MAX_MEMORIES, DEFAULT_REQUESTS_PER_MINUTE,
};
use std::sync::Arc;
use std::time::Instant;

pub const MAX_BODY_BYTES: usize = 1024 * 1024;
const MAX_CONTENT_BYTES: usize = 64 * 1024;
const MAX_QUERY_BYTES: usize = 2000;
const MAX_TAGS: usize = 32;
const MAX_SHORT_TEXT: usize = 200;
const MAX_METADATA_BYTES: usize = 16 * 1024;
const MAX_LIST_LIMIT: usize = 100;
const MAX_SEARCH_LIMIT: usize = 50;
const SOURCE: &str = "api";
pub const SUBJECT_HEADER: &str = "X-MemoryPilot-Subject";

pub struct Request<'a> {
    pub method: &'a str,
    /// Path and query string.
    pub url: &'a str,
    /// Name and value of each header, names in any case.
    pub headers: &'a [(&'a str, &'a str)],
    pub body: &'a [u8],
}

impl Request<'_> {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(field, _)| field.eq_ignore_ascii_case(name))
            .map(|(_, value)| *value)
    }
}

type Headers = Vec<(&'static str, String)>;

#[derive(Debug)]
pub struct Response {
    pub status: u16,
    /// `Null`: no body.
    pub body: Value,
    /// Extra headers (rate limit state, Retry-After).
    pub headers: Headers,
}

impl Response {
    pub fn new(status: u16, body: Value) -> Self {
        Self {
            status,
            body,
            headers: Vec::new(),
        }
    }

    fn ok(body: Value) -> Self {
        Self::new(200, body)
    }
}

pub(crate) struct Failure {
    pub(crate) status: u16,
    pub(crate) message: String,
}

impl Failure {
    fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(400, message)
    }

    /// Database failures are logged, not shown: they can name tables and
    /// constraints. Anything else from the write path is a refusal the
    /// caller can act on (a secret in the content, a credential edit).
    fn from_store(error: String) -> Self {
        if error.contains("Postgres:") || error.contains("connection lock") {
            eprintln!("[memorypilot-server] {}", error);
            Self::new(500, "Internal error")
        } else {
            Self::new(422, error)
        }
    }
}

impl From<Failure> for Response {
    fn from(failure: Failure) -> Self {
        Response::new(failure.status, json!({ "error": failure.message }))
    }
}

enum Route<'a> {
    AddMemory,
    ListMemories,
    GetMemory(&'a str),
    UpdateMemory(&'a str),
    DeleteMemory(&'a str),
    Search,
}

impl Route<'_> {
    fn permission(&self) -> Permission {
        match self {
            Route::ListMemories | Route::GetMemory(_) | Route::Search => Permission::Read,
            Route::AddMemory | Route::UpdateMemory(_) | Route::DeleteMemory(_) => Permission::Write,
        }
    }
}

fn route<'a>(method: &str, path: &'a str) -> Option<Route<'a>> {
    let id = path
        .strip_prefix("/v1/memories/")
        .filter(|id| !id.is_empty() && !id.contains('/'));
    match (method, path, id) {
        ("POST", "/v1/memories", _) => Some(Route::AddMemory),
        ("GET", "/v1/memories", _) => Some(Route::ListMemories),
        ("POST", "/v1/search", _) => Some(Route::Search),
        ("GET", _, Some(id)) => Some(Route::GetMemory(id)),
        ("PATCH", _, Some(id)) => Some(Route::UpdateMemory(id)),
        ("DELETE", _, Some(id)) => Some(Route::DeleteMemory(id)),
        _ => None,
    }
}

#[derive(Clone)]
pub struct Api {
    pool: PgPool,
    allowed_origins: Vec<String>,
    rate: Arc<RateLimiter>,
    default_requests_per_minute: u32,
    default_max_memories: u64,
}

impl Api {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            allowed_origins: Vec::new(),
            rate: Arc::default(),
            default_requests_per_minute: DEFAULT_REQUESTS_PER_MINUTE,
            default_max_memories: DEFAULT_MAX_MEMORIES,
        }
    }

    /// Limits of tenants that have none of their own.
    pub fn with_default_limits(mut self, requests_per_minute: u32, max_memories: u64) -> Self {
        self.default_requests_per_minute = requests_per_minute.max(1);
        self.default_max_memories = max_memories;
        self
    }

    pub(crate) fn max_memories(&self, key: &ApiKey) -> u64 {
        key.limits.max_memories.unwrap_or(self.default_max_memories)
    }

    /// Authenticates, then charges the key's rate: 429 with Retry-After once
    /// the minute's requests are spent. Returns the key and the headers that
    /// tell the client where it stands.
    fn admit(&self, authorization: Option<&str>) -> Result<(ApiKey, Headers), Response> {
        let key = self.authenticate(authorization).map_err(Response::from)?;
        let limit = key
            .limits
            .requests_per_minute
            .unwrap_or(self.default_requests_per_minute);
        match self.rate.take(key.id, limit, Instant::now()) {
            Ok(remaining) => Ok((
                key,
                vec![
                    ("X-RateLimit-Limit", limit.to_string()),
                    ("X-RateLimit-Remaining", remaining.to_string()),
                ],
            )),
            Err(wait) => {
                let mut response = Response::from(Failure::new(
                    429,
                    format!("Too many requests: {} per minute for this account", limit),
                ));
                response.headers = vec![
                    ("Retry-After", wait.to_string()),
                    ("X-RateLimit-Limit", limit.to_string()),
                    ("X-RateLimit-Remaining", "0".into()),
                ];
                Err(response)
            }
        }
    }

    /// Browser origins allowed to call `/mcp` (DNS rebinding guard: any
    /// other `Origin` is refused). Server-side clients send none.
    pub fn with_allowed_origins(mut self, origins: Vec<String>) -> Self {
        self.allowed_origins = origins;
        self
    }

    pub fn handle(&self, request: &Request) -> Response {
        self.respond(request).unwrap_or_else(Response::from)
    }

    fn respond(&self, request: &Request) -> Result<Response, Failure> {
        let (path, query) = request.url.split_once('?').unwrap_or((request.url, ""));
        if path == "/health" && request.method == "GET" {
            return Ok(Response::ok(
                json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") }),
            ));
        }
        if path == "/mcp" {
            return self.mcp(request);
        }
        let route = route(request.method, path).ok_or_else(|| {
            Failure::new(
                404,
                "Not found. Routes: POST|GET /v1/memories, GET|PATCH|DELETE /v1/memories/{id}, POST /v1/search, POST /mcp",
            )
        })?;
        let (key, headers) = match self.admit(request.header("Authorization")) {
            Ok(admitted) => admitted,
            Err(response) => return Ok(response),
        };
        let mut response = self
            .serve(&key, route, query, request)
            .unwrap_or_else(Response::from);
        response.headers.extend(headers);
        Ok(response)
    }

    fn serve(
        &self,
        key: &ApiKey,
        route: Route,
        query: &str,
        request: &Request,
    ) -> Result<Response, Failure> {
        if !key.allows(route.permission()) {
            return Err(Failure::new(
                403,
                format!(
                    "This key lacks the '{}' permission",
                    route.permission().as_str()
                ),
            ));
        }
        let store = self.store(key, request.header(SUBJECT_HEADER))?;
        match route {
            Route::AddMemory => add(&store, request.body, self.max_memories(key)),
            Route::ListMemories => list(&store, &query_params(query)),
            Route::GetMemory(id) => get(&store, id),
            Route::UpdateMemory(id) => update(&store, id, request.body),
            Route::DeleteMemory(id) => delete(&store, id),
            Route::Search => search(&store, request.body),
        }
    }

    fn mcp(&self, request: &Request) -> Result<Response, Failure> {
        if request.method != "POST" {
            return Err(Failure::new(
                405,
                "The MCP endpoint takes POST only (no session, no stream to open)",
            ));
        }
        if let Some(origin) = request.header("Origin") {
            if !self.allowed_origins.iter().any(|allowed| allowed == origin) {
                return Ok(Response::new(
                    403,
                    json!({ "jsonrpc": "2.0", "error": { "code": -32600, "message": "Origin not allowed" } }),
                ));
            }
        }
        let (key, headers) = match self.admit(request.header("Authorization")) {
            Ok(admitted) => admitted,
            Err(response) => return Ok(response),
        };
        let mut response = crate::mcp::handle(self, &key, request);
        response.headers.extend(headers);
        Ok(response)
    }

    fn authenticate(&self, authorization: Option<&str>) -> Result<ApiKey, Failure> {
        let presented = authorization
            .and_then(|value| value.strip_prefix("Bearer "))
            .map(str::trim)
            .ok_or_else(|| Failure::new(401, "Missing API key (Authorization: Bearer <key>)"))?;
        self.pool
            .authenticate(presented)
            .map_err(Failure::from_store)?
            .ok_or_else(|| Failure::new(401, "Invalid or revoked API key"))
    }

    pub(crate) fn store(&self, key: &ApiKey, subject: Option<&str>) -> Result<PgStore, Failure> {
        let subject = match subject {
            None => None,
            Some(_) if !key.allows(Permission::Subjects) => {
                return Err(Failure::new(
                    403,
                    "This key lacks the 'subjects' permission to act for an end user",
                ))
            }
            Some(external_id) => Some(self.pool.subject(key, external_id).map_err(|error| {
                match error.contains("Postgres:") {
                    true => Failure::from_store(error),
                    false => Failure::invalid(error),
                }
            })?),
        };
        PgStore::from_pool(&self.pool, key.scope(subject)).map_err(Failure::from_store)
    }
}

fn parse<'a, T: Deserialize<'a>>(body: &'a [u8]) -> Result<T, Failure> {
    serde_json::from_slice(body)
        .map_err(|error| Failure::invalid(format!("Invalid JSON body: {}", error)))
}

fn within(field: &str, value: &str, max_bytes: usize) -> Result<(), Failure> {
    if value.len() > max_bytes {
        return Err(Failure::invalid(format!(
            "{} is longer than {} bytes",
            field, max_bytes
        )));
    }
    Ok(())
}

fn non_empty<'a>(field: &str, value: &'a str, max_bytes: usize) -> Result<&'a str, Failure> {
    within(field, value, max_bytes)?;
    if value.trim().is_empty() {
        return Err(Failure::invalid(format!("{} is required", field)));
    }
    Ok(value)
}

fn check_known_kind(kind: &str) -> Result<(), Failure> {
    if !MEMORY_KINDS.contains(&kind) {
        return Err(Failure::invalid(format!(
            "Invalid kind '{}'. Valid: {:?}",
            kind, MEMORY_KINDS
        )));
    }
    Ok(())
}

fn check_kind(kind: &str) -> Result<(), Failure> {
    if kind == "credential" {
        return Err(Failure::new(
            422,
            "Credentials are not accepted by the server yet (they need a per-tenant encryption key)",
        ));
    }
    check_known_kind(kind)
}

fn check_tags(tags: &[String]) -> Result<(), Failure> {
    if tags.len() > MAX_TAGS {
        return Err(Failure::invalid(format!("At most {} tags", MAX_TAGS)));
    }
    tags.iter()
        .try_for_each(|tag| non_empty("tag", tag, MAX_SHORT_TEXT).map(|_| ()))
}

fn check_importance(importance: i32) -> Result<(), Failure> {
    if !(1..=5).contains(&importance) {
        return Err(Failure::invalid("importance is between 1 and 5"));
    }
    Ok(())
}

fn check_expiry(expires_at: &str) -> Result<(), Failure> {
    chrono::DateTime::parse_from_rfc3339(expires_at)
        .map(|_| ())
        .map_err(|_| Failure::invalid("expires_at is an RFC 3339 date (2026-12-31T00:00:00Z)"))
}

fn check_metadata(metadata: &Value) -> Result<(), Failure> {
    if !metadata.is_object() {
        return Err(Failure::invalid("metadata is a JSON object"));
    }
    within("metadata", &metadata.to_string(), MAX_METADATA_BYTES)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NewMemory {
    content: String,
    kind: Option<String>,
    project: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    importance: Option<i32>,
    expires_at: Option<String>,
    metadata: Option<Value>,
}

pub(crate) fn add(store: &PgStore, body: &[u8], max_memories: u64) -> Result<Response, Failure> {
    let input: NewMemory = parse(body)?;
    let content = non_empty("content", &input.content, MAX_CONTENT_BYTES)?;
    let (auto_importance, auto_kind, auto_ttl_days) = memorypilot_core::gc::auto_classify(content);
    let kind = input.kind.as_deref().unwrap_or(auto_kind);
    check_kind(kind)?;
    if let Some(project) = &input.project {
        non_empty("project", project, MAX_SHORT_TEXT)?;
    }
    check_tags(&input.tags)?;
    let importance = input.importance.unwrap_or(auto_importance);
    check_importance(importance)?;
    let expires_at = match input.expires_at {
        Some(expires_at) => {
            check_expiry(&expires_at)?;
            Some(expires_at)
        }
        None => auto_ttl_days
            .map(|days| (chrono::Utc::now() + chrono::Duration::days(days)).to_rfc3339()),
    };
    if let Some(metadata) = &input.metadata {
        check_metadata(metadata)?;
    }
    let stored = store.tenant_memory_count().map_err(Failure::from_store)?;
    if stored >= 0 && stored as u64 >= max_memories {
        return Err(Failure::new(
            403,
            format!(
                "Memory quota reached: {} memories for this account",
                max_memories
            ),
        ));
    }
    let outcome = add_memory_to(
        store,
        None,
        content,
        kind,
        input.project.as_deref(),
        &input.tags,
        SOURCE,
        importance,
        expires_at.as_deref(),
        input.metadata.as_ref(),
        &MemoryScope::default(),
    )
    .map_err(Failure::from_store)?;
    Ok(Response::new(
        if outcome.merged { 200 } else { 201 },
        json!({ "memory": outcome.memory, "merged": outcome.merged }),
    ))
}

fn query_params(query: &str) -> HashMap<String, String> {
    form_urlencoded::parse(query.as_bytes())
        .into_owned()
        .collect()
}

fn bounded(
    params: &HashMap<String, String>,
    name: &str,
    default: usize,
    max: usize,
) -> Result<usize, Failure> {
    match params.get(name) {
        None => Ok(default),
        Some(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&n| n <= max)
            .ok_or_else(|| Failure::invalid(format!("{} is a whole number up to {}", name, max))),
    }
}

pub(crate) fn list(store: &PgStore, params: &HashMap<String, String>) -> Result<Response, Failure> {
    let limit = bounded(params, "limit", 20, MAX_LIST_LIMIT)?;
    let offset = bounded(params, "offset", 0, usize::MAX)?;
    let kind = params.get("kind").map(String::as_str);
    if let Some(kind) = kind {
        check_known_kind(kind)?;
    }
    let (memories, total) = store
        .list_memory_rows(
            params.get("project").map(String::as_str),
            kind,
            None,
            limit,
            offset,
        )
        .map_err(Failure::from_store)?;
    Ok(Response::ok(json!({
        "total": total,
        "count": memories.len(),
        "offset": offset,
        "memories": memories,
    })))
}

/// 404 when the scope does not see the memory, 403 when it sees it but
/// cannot change it (a shared memory, for an end user).
fn changeable(store: &PgStore, id: &str) -> Result<(), Failure> {
    if store.can_change(id).map_err(Failure::from_store)? {
        return Ok(());
    }
    found(store.fetch_memory(id).map_err(Failure::from_store)?, id)?;
    Err(Failure::new(
        403,
        "This memory is shared by the app: an end user cannot change it",
    ))
}

fn found(memory: Option<Memory>, id: &str) -> Result<Memory, Failure> {
    memory.ok_or_else(|| Failure::new(404, format!("No memory '{}'", id)))
}

pub(crate) fn get(store: &PgStore, id: &str) -> Result<Response, Failure> {
    let memory = found(store.fetch_memory(id).map_err(Failure::from_store)?, id)?;
    Ok(Response::ok(json!({ "memory": memory })))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryChanges {
    content: Option<String>,
    kind: Option<String>,
    tags: Option<Vec<String>>,
    importance: Option<i32>,
    expires_at: Option<String>,
    metadata: Option<Value>,
}

pub(crate) fn update(store: &PgStore, id: &str, body: &[u8]) -> Result<Response, Failure> {
    let changes: MemoryChanges = parse(body)?;
    if let Some(content) = &changes.content {
        non_empty("content", content, MAX_CONTENT_BYTES)?;
    }
    if let Some(kind) = &changes.kind {
        check_kind(kind)?;
    }
    if let Some(tags) = &changes.tags {
        check_tags(tags)?;
    }
    if let Some(importance) = changes.importance {
        check_importance(importance)?;
    }
    if let Some(expires_at) = &changes.expires_at {
        check_expiry(expires_at)?;
    }
    if let Some(metadata) = &changes.metadata {
        check_metadata(metadata)?;
    }
    changeable(store, id)?;
    let outcome = update_memory_in(
        store,
        id,
        changes.content.as_deref(),
        changes.kind.as_deref(),
        changes.tags.as_deref(),
        changes.importance,
        changes.expires_at.as_deref(),
        changes.metadata.as_ref(),
    )
    .map_err(Failure::from_store)?;
    let memory = found(outcome.map(|outcome| outcome.memory), id)?;
    Ok(Response::ok(json!({ "memory": memory })))
}

pub(crate) fn delete(store: &PgStore, id: &str) -> Result<Response, Failure> {
    changeable(store, id)?;
    if !store.remove_memory(id).map_err(Failure::from_store)? {
        return Err(Failure::new(404, format!("No memory '{}'", id)));
    }
    Ok(Response::ok(json!({ "deleted": id })))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchQuery {
    query: String,
    limit: Option<usize>,
    project: Option<String>,
    kind: Option<String>,
    tags: Option<Vec<String>>,
}

pub(crate) fn search(store: &PgStore, body: &[u8]) -> Result<Response, Failure> {
    let input: SearchQuery = parse(body)?;
    let query = non_empty("query", &input.query, MAX_QUERY_BYTES)?;
    let limit = input.limit.unwrap_or(10);
    if !(1..=MAX_SEARCH_LIMIT).contains(&limit) {
        return Err(Failure::invalid(format!(
            "limit is between 1 and {}",
            MAX_SEARCH_LIMIT
        )));
    }
    if let Some(kind) = &input.kind {
        check_known_kind(kind)?;
    }
    if let Some(tags) = &input.tags {
        check_tags(tags)?;
    }
    let (results, confidence) = hybrid_search(
        store,
        chrono::Utc::now(),
        query,
        limit,
        input.project.as_deref(),
        input.kind.as_deref(),
        input.tags.as_deref(),
        None,
    )
    .map_err(Failure::from_store)?;
    Ok(Response::ok(
        json!({ "results": results, "confidence": confidence }),
    ))
}

//! MCP over Streamable HTTP (`POST /mcp`), dual era:
//! - modern revisions (2026-07-28): stateless, the version and client
//!   capabilities in each request's `_meta`, mirrored in headers that must
//!   match the body;
//! - legacy revisions (2025-03-26 to 2025-11-25): an `initialize` handshake,
//!   served without a session (each request stands alone anyway).
//!
//! The tools run the REST handlers, so validation and isolation are the same.

use std::collections::HashMap;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use memorypilot_core::db::MEMORY_KINDS;
use serde_json::{json, Map, Value};

use crate::api::{self, Api, Event, Failure, Request, Response, SUBJECT_HEADER};
use crate::{ApiKey, Permission};

const MODERN_VERSIONS: &[&str] = &["2026-07-28"];
const LEGACY_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26"];

const META_VERSION: &str = "io.modelcontextprotocol/protocolVersion";
const META_CAPABILITIES: &str = "io.modelcontextprotocol/clientCapabilities";
const META_SERVER_INFO: &str = "io.modelcontextprotocol/serverInfo";

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const HEADER_MISMATCH: i64 = -32020;
const UNSUPPORTED_VERSION: i64 = -32022;

const INSTRUCTIONS: &str = "MemoryPilot keeps durable knowledge for this app and its users. \
Call search_memory before answering anything that may depend on earlier context. \
Call add_memory for decisions, preferences and facts worth keeping, without asking. \
Never store passwords, API keys or tokens: they are refused.";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Era {
    Modern,
    Legacy,
}

fn server_info() -> Value {
    json!({ "name": "MemoryPilot", "version": env!("CARGO_PKG_VERSION") })
}

fn rpc_error(
    status: u16,
    id: Option<&Value>,
    code: i64,
    message: &str,
    data: Option<Value>,
) -> Response {
    let mut error = json!({ "code": code, "message": message });
    if let Some(data) = data {
        error["data"] = data;
    }
    let mut body = json!({ "jsonrpc": "2.0", "error": error });
    if let Some(id) = id {
        body["id"] = id.clone();
    }
    Response::new(status, body)
}

fn rpc_result(id: &Value, mut result: Value) -> Response {
    result["resultType"] = json!("complete");
    result["_meta"] = json!({ META_SERVER_INFO: server_info() });
    Response::new(200, json!({ "jsonrpc": "2.0", "id": id, "result": result }))
}

/// `Mcp-Name` and `Mcp-Param-*` values may be sent as `=?base64?…?=`.
fn decoded_header(value: &str) -> Option<String> {
    match value
        .strip_prefix("=?base64?")
        .and_then(|rest| rest.strip_suffix("?="))
    {
        Some(encoded) => STANDARD
            .decode(encoded)
            .ok()
            .and_then(|bytes| String::from_utf8(bytes).ok()),
        None => Some(value.to_string()),
    }
}

fn era(
    request: &Request,
    id: &Value,
    method: &str,
    params: &Map<String, Value>,
) -> Result<Era, Response> {
    let mismatch = |message: String| rpc_error(400, Some(id), HEADER_MISMATCH, &message, None);
    let supported: Vec<&str> = MODERN_VERSIONS
        .iter()
        .chain(LEGACY_VERSIONS)
        .copied()
        .collect();
    let unsupported = |requested: &str| {
        rpc_error(
            400,
            Some(id),
            UNSUPPORTED_VERSION,
            "Unsupported protocol version",
            Some(json!({ "supported": supported, "requested": requested })),
        )
    };
    let header_version = request.header("MCP-Protocol-Version");
    let meta = params.get("_meta").and_then(Value::as_object);
    let Some(version) = meta
        .and_then(|meta| meta.get(META_VERSION))
        .and_then(Value::as_str)
    else {
        return match header_version {
            _ if method == "initialize" => Ok(Era::Legacy),
            None => Ok(Era::Legacy),
            Some(version) if LEGACY_VERSIONS.contains(&version) => Ok(Era::Legacy),
            Some(version) if MODERN_VERSIONS.contains(&version) => Err(rpc_error(
                400,
                Some(id),
                INVALID_PARAMS,
                "_meta io.modelcontextprotocol/protocolVersion is required",
                None,
            )),
            Some(version) => Err(unsupported(version)),
        };
    };
    if !MODERN_VERSIONS.contains(&version) {
        return Err(unsupported(version));
    }
    if header_version != Some(version) {
        return Err(mismatch(format!(
            "MCP-Protocol-Version header {:?} does not match the body's {}",
            header_version, version
        )));
    }
    if request.header("Mcp-Method") != Some(method) {
        return Err(mismatch(format!(
            "Mcp-Method header does not match method {}",
            method
        )));
    }
    if method == "tools/call" {
        let name = params.get("name").and_then(Value::as_str);
        let header = request.header("Mcp-Name").and_then(decoded_header);
        if name.is_none() || header.as_deref() != name {
            return Err(mismatch(
                "Mcp-Name header does not match params.name".into(),
            ));
        }
    }
    if !meta
        .and_then(|meta| meta.get(META_CAPABILITIES))
        .is_some_and(Value::is_object)
    {
        return Err(rpc_error(
            400,
            Some(id),
            INVALID_PARAMS,
            "_meta io.modelcontextprotocol/clientCapabilities is required",
            None,
        ));
    }
    Ok(Era::Modern)
}

pub(crate) fn handle(api: &Api, key: &ApiKey, request: &Request) -> Response {
    let Ok(message) = serde_json::from_slice::<Value>(request.body) else {
        return rpc_error(400, None, PARSE_ERROR, "Parse error", None);
    };
    let invalid = |message: &str| rpc_error(400, None, INVALID_REQUEST, message, None);
    let Some(object) = message.as_object() else {
        return invalid("A single JSON-RPC request or notification is expected");
    };
    if object.get("jsonrpc") != Some(&json!("2.0")) {
        return invalid("jsonrpc must be \"2.0\"");
    }
    let Some(method) = object.get("method").and_then(Value::as_str) else {
        return invalid("method is required (clients send no responses)");
    };
    let Some(id) = object.get("id") else {
        return Response::new(202, Value::Null);
    };
    if !(id.is_string() || id.is_number()) {
        return invalid("id is a string or a number");
    }
    let empty = Map::new();
    let params = match object.get("params") {
        None => &empty,
        Some(Value::Object(params)) => params,
        Some(_) => return rpc_error(400, Some(id), INVALID_PARAMS, "params is an object", None),
    };
    let era = match era(request, id, method, params) {
        Ok(era) => era,
        Err(response) => return response,
    };
    match (method, era) {
        ("initialize", Era::Legacy) => {
            let requested = params.get("protocolVersion").and_then(Value::as_str);
            let version = requested
                .filter(|version| LEGACY_VERSIONS.contains(version))
                .unwrap_or(LEGACY_VERSIONS[0]);
            rpc_result(
                id,
                json!({
                    "protocolVersion": version,
                    "capabilities": { "tools": { "listChanged": false } },
                    "serverInfo": server_info(),
                    "instructions": INSTRUCTIONS,
                }),
            )
        }
        ("server/discover", _) => rpc_result(
            id,
            json!({
                "supportedVersions": MODERN_VERSIONS.iter().chain(LEGACY_VERSIONS).collect::<Vec<_>>(),
                "capabilities": { "tools": {} },
                "instructions": INSTRUCTIONS,
            }),
        ),
        ("ping", _) => rpc_result(id, json!({})),
        ("tools/list", _) => {
            let tools: Vec<Value> = TOOLS
                .iter()
                .filter(|tool| key.allows(tool.permission))
                .map(Tool::definition)
                .collect();
            rpc_result(id, json!({ "tools": tools }))
        }
        ("tools/call", _) => call_tool(api, key, request, id, params),
        _ => rpc_error(
            if era == Era::Modern { 404 } else { 200 },
            Some(id),
            METHOD_NOT_FOUND,
            &format!("Method not found: {}", method),
            None,
        ),
    }
}

struct Tool {
    name: &'static str,
    description: &'static str,
    permission: Permission,
    read_only: bool,
    destructive: bool,
    schema: fn() -> Value,
}

impl Tool {
    fn definition(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "inputSchema": (self.schema)(),
            "annotations": {
                "readOnlyHint": self.read_only,
                "destructiveHint": self.destructive,
                "idempotentHint": self.read_only,
                "openWorldHint": false,
            },
        })
    }
}

fn writable_kinds() -> Vec<&'static str> {
    MEMORY_KINDS
        .iter()
        .copied()
        .filter(|kind| *kind != "credential")
        .collect()
}

fn fields(object: Value) -> Map<String, Value> {
    match object {
        Value::Object(fields) => fields,
        _ => unreachable!("a JSON object literal"),
    }
}

fn memory_fields() -> Map<String, Value> {
    fields(json!({
        "content": { "type": "string", "description": "The memory, self-contained" },
        "kind": { "type": "string", "enum": writable_kinds(), "description": "Inferred from the content when omitted" },
        "tags": { "type": "array", "items": { "type": "string" }, "maxItems": 32 },
        "importance": { "type": "integer", "minimum": 1, "maximum": 5 },
        "expires_at": { "type": "string", "format": "date-time" },
        "metadata": { "type": "object" },
    }))
}

fn object_schema(properties: Map<String, Value>, required: &[&str]) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

fn id_field() -> Map<String, Value> {
    let mut fields = Map::new();
    fields.insert("id".into(), json!({ "type": "string" }));
    fields
}

const TOOLS: &[Tool] = &[
    Tool {
        name: "search_memory",
        description: "Search memories by meaning and keywords; best matches first.",
        permission: Permission::Read,
        read_only: true,
        destructive: false,
        schema: || {
            object_schema(
                fields(json!({
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 50 },
                    "project": { "type": "string" },
                    "kind": { "type": "string", "enum": MEMORY_KINDS },
                    "tags": { "type": "array", "items": { "type": "string" } },
                })),
                &["query"],
            )
        },
    },
    Tool {
        name: "add_memory",
        description: "Store a memory. A near duplicate in the same project is merged instead.",
        permission: Permission::Write,
        read_only: false,
        destructive: false,
        schema: || {
            let mut fields = memory_fields();
            fields.insert("project".into(), json!({ "type": "string" }));
            object_schema(fields, &["content"])
        },
    },
    Tool {
        name: "list_memories",
        description: "List memories, most recently updated first.",
        permission: Permission::Read,
        read_only: true,
        destructive: false,
        schema: || {
            object_schema(
                fields(json!({
                    "project": { "type": "string" },
                    "kind": { "type": "string", "enum": MEMORY_KINDS },
                    "limit": { "type": "integer", "minimum": 0, "maximum": 100 },
                    "offset": { "type": "integer", "minimum": 0 },
                })),
                &[],
            )
        },
    },
    Tool {
        name: "get_memory",
        description: "One memory by id.",
        permission: Permission::Read,
        read_only: true,
        destructive: false,
        schema: || object_schema(id_field(), &["id"]),
    },
    Tool {
        name: "update_memory",
        description: "Change a memory: only the fields given are replaced.",
        permission: Permission::Write,
        read_only: false,
        destructive: false,
        schema: || {
            let mut fields = memory_fields();
            fields.extend(id_field());
            object_schema(fields, &["id"])
        },
    },
    Tool {
        name: "delete_memory",
        description: "Delete a memory by id.",
        permission: Permission::Write,
        read_only: false,
        destructive: true,
        schema: || object_schema(id_field(), &["id"]),
    },
];

fn tool_result(body: Value, is_error: bool) -> Value {
    let text = match &body {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    json!({ "content": [{ "type": "text", "text": text }], "isError": is_error })
}

fn take_id(arguments: &mut Map<String, Value>) -> Result<String, Failure> {
    match arguments.remove("id") {
        Some(Value::String(id)) if !id.is_empty() => Ok(id),
        _ => Err(Failure {
            status: 400,
            message: "id is required".into(),
        }),
    }
}

fn list_params(arguments: &Map<String, Value>) -> Result<HashMap<String, String>, Failure> {
    let invalid = |message: String| Failure {
        status: 400,
        message,
    };
    arguments
        .iter()
        .filter(|(_, value)| !value.is_null())
        .map(|(name, value)| {
            if !matches!(name.as_str(), "project" | "kind" | "limit" | "offset") {
                return Err(invalid(format!("Unknown argument {}", name)));
            }
            let text = match value {
                Value::String(text) => text.clone(),
                Value::Number(number) => number.to_string(),
                _ => return Err(invalid(format!("{} is a string or a number", name))),
            };
            Ok((name.clone(), text))
        })
        .collect()
}

fn call_tool(
    api: &Api,
    key: &ApiKey,
    request: &Request,
    id: &Value,
    params: &Map<String, Value>,
) -> Response {
    let Some(name) = params.get("name").and_then(Value::as_str) else {
        return rpc_error(
            400,
            Some(id),
            INVALID_PARAMS,
            "params.name is required",
            None,
        );
    };
    let Some(tool) = TOOLS.iter().find(|tool| tool.name == name) else {
        return rpc_error(
            400,
            Some(id),
            INVALID_PARAMS,
            &format!("Unknown tool: {}", name),
            None,
        );
    };
    let mut arguments = match params.get("arguments") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(arguments)) => arguments.clone(),
        Some(_) => {
            return rpc_error(
                400,
                Some(id),
                INVALID_PARAMS,
                "params.arguments is an object",
                None,
            )
        }
    };
    if !key.allows(tool.permission) {
        let message = format!(
            "This key lacks the '{}' permission",
            tool.permission.as_str()
        );
        return rpc_result(id, tool_result(json!(message), true));
    }
    let memory_id = arguments
        .get("id")
        .and_then(Value::as_str)
        .map(String::from);
    let outcome = api
        .store(key, request.header(SUBJECT_HEADER))
        .and_then(|store| {
            let body = |arguments: &Map<String, Value>| {
                Value::Object(arguments.clone()).to_string().into_bytes()
            };
            let response = match tool.name {
                "search_memory" => api::search(&store, &body(&arguments)),
                "add_memory" => api::add(&store, &body(&arguments), api.max_memories(key)),
                "list_memories" => api::list(&store, &list_params(&arguments)?),
                "get_memory" => api::get(&store, &take_id(&mut arguments)?),
                "update_memory" => {
                    let memory = take_id(&mut arguments)?;
                    api::update(&store, &memory, &body(&arguments))
                }
                "delete_memory" => api::delete(&store, &take_id(&mut arguments)?),
                other => unreachable!("tool {} has no handler", other),
            }?;
            if let Some(event) = Event::of_tool(tool.name, memory_id.as_deref(), &response) {
                api.record(key, &store, event);
            }
            Ok(response)
        });
    let result = match outcome {
        Ok(response) => tool_result(response.body, false),
        Err(failure) => tool_result(json!(failure.message), true),
    };
    rpc_result(id, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_header_values_are_decoded() {
        assert_eq!(decoded_header("add_memory").as_deref(), Some("add_memory"));
        assert_eq!(
            decoded_header("=?base64?SGVsbG8sIOS4lueVjA==?=").as_deref(),
            Some("Hello, 世界")
        );
        assert_eq!(decoded_header("=?base64?not base64?="), None);
    }

    #[test]
    fn tool_schemas_are_closed_objects_and_never_offer_credentials() {
        for tool in TOOLS {
            let schema = (tool.schema)();
            assert_eq!(schema["type"], "object", "{}", tool.name);
            assert_eq!(schema["additionalProperties"], false, "{}", tool.name);
            if let Some(kinds) = schema["properties"]["kind"]["enum"].as_array() {
                let offers_credential = kinds.contains(&json!("credential"));
                assert_eq!(
                    offers_credential, tool.read_only,
                    "{}: credential only to filter reads",
                    tool.name
                );
            }
        }
    }
}

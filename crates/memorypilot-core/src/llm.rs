//! Minimal client for OpenAI-compatible chat completion endpoints.
//!
//! Used by the LongMemEval end-to-end evaluation only: one `user` message,
//! temperature 0, bounded output. The same code reaches OpenAI, Anthropic's
//! OpenAI-compatible endpoint, Ollama or LM Studio through `base_url`.
//! Anthropic rejects `temperature` on its current models, so it is left to
//! the server default there.

use std::time::Duration;

use serde_json::json;

/// Default endpoint: OpenAI, which hosts the official LongMemEval judge.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

const MAX_ATTEMPTS: u32 = 6;
const MAX_BACKOFF_SECS: u64 = 60;

/// One chat completion: the text and the token counts the server reported.
#[derive(Debug, Clone)]
pub struct Completion {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// `finish_reason` was `length`: the output hit `max_tokens`.
    pub truncated: bool,
}

/// Connection to one OpenAI-compatible endpoint.
pub struct ChatClient {
    agent: ureq::Agent,
    endpoint: String,
    api_key: Option<String>,
    sends_temperature: bool,
}

/// Seconds one request may take. Plain HTTP is only accepted on a loopback
/// host, so it is a model running on this machine: reading a long prompt
/// there takes minutes, where a hosted endpoint answers in seconds.
fn request_timeout_secs(base: &str) -> u64 {
    if base.starts_with("http://") {
        1800
    } else {
        180
    }
}

/// Anthropic's endpoint: `temperature` returns a 400 on its current models.
const ANTHROPIC_HOST: &str = "api.anthropic.com";

impl ChatClient {
    /// `base_url` must be HTTPS, or plain HTTP on a loopback host, so an
    /// API key never travels in clear over the network.
    pub fn new(base_url: &str, api_key: Option<String>) -> Result<Self, String> {
        let base = base_url.trim().trim_end_matches('/');
        validate_base_url(base)?;
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(request_timeout_secs(base))))
            .http_status_as_error(false)
            .user_agent(format!("MemoryPilot/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .into();
        Ok(Self {
            agent,
            endpoint: format!("{}/chat/completions", base),
            api_key: api_key.filter(|key| !key.trim().is_empty()),
            sends_temperature: url_host(base) != ANTHROPIC_HOST,
        })
    }

    /// False when requests leave `temperature` to the server default.
    pub fn sends_temperature(&self) -> bool {
        self.sends_temperature
    }

    /// Sends `prompt` as a single user message, retrying on 429, 5xx and
    /// transport errors with exponential backoff. Error text never carries
    /// the API key, even when a server echoes it back.
    pub fn complete(&self, model: &str, prompt: &str, max_tokens: u32) -> Result<Completion, String> {
        self.complete_raw(model, prompt, max_tokens).map_err(|error| self.redact(&error))
    }

    fn redact(&self, text: &str) -> String {
        match &self.api_key {
            Some(key) => text.replace(key.as_str(), "<redacted>"),
            None => text.to_string(),
        }
    }

    fn complete_raw(&self, model: &str, prompt: &str, max_tokens: u32) -> Result<Completion, String> {
        let mut body = json!({
            "model": model,
            "messages": [{ "role": "user", "content": prompt }],
            "n": 1,
            "max_tokens": max_tokens,
        });
        if self.sends_temperature {
            body["temperature"] = json!(0);
        }
        let body = body.to_string();

        let mut last_error = String::new();
        for attempt in 0..MAX_ATTEMPTS {
            if attempt > 0 {
                std::thread::sleep(Duration::from_secs((1u64 << attempt).min(MAX_BACKOFF_SECS)));
            }
            let mut request = self.agent.post(&self.endpoint).header("Content-Type", "application/json");
            if let Some(key) = &self.api_key {
                request = request.header("Authorization", &format!("Bearer {}", key));
            }
            let mut response = match request.send(body.as_str()) {
                Ok(response) => response,
                Err(error) => {
                    last_error = format!("transport error: {}", error);
                    continue;
                }
            };
            let status = response.status().as_u16();
            let text = response
                .body_mut()
                .read_to_string()
                .map_err(|error| format!("cannot read response body (HTTP {}): {}", status, error))?;
            if status == 429 || status >= 500 {
                last_error = format!("HTTP {}: {}", status, excerpt(&text));
                continue;
            }
            if status != 200 {
                return Err(format!("HTTP {} from {}: {}", status, self.endpoint, excerpt(&text)));
            }
            return parse_completion(&text);
        }
        Err(format!("gave up after {} attempts: {}", MAX_ATTEMPTS, last_error))
    }
}

fn validate_base_url(base: &str) -> Result<(), String> {
    let (rest, secure) = if let Some(rest) = base.strip_prefix("https://") {
        (rest, true)
    } else if let Some(rest) = base.strip_prefix("http://") {
        (rest, false)
    } else {
        return Err(format!("LLM base URL must start with https:// (got {})", base));
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    // `user@host` hides the real host behind a loopback-looking prefix and
    // would put credentials in the report: refused outright.
    if authority.contains('@') {
        return Err("LLM base URL must not contain user information (user@host)".to_string());
    }
    if secure {
        return Ok(());
    }
    let host = if authority.starts_with('[') {
        authority.split(']').next().map(|h| format!("{}]", h)).unwrap_or_default()
    } else {
        authority.split(':').next().unwrap_or("").to_string()
    };
    match host.as_str() {
        "localhost" | "127.0.0.1" | "[::1]" => Ok(()),
        _ => Err(format!(
            "plain http:// is only allowed on localhost, 127.0.0.1 or [::1] (got {})",
            base
        )),
    }
}

fn parse_completion(text: &str) -> Result<Completion, String> {
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|error| format!("invalid JSON from LLM: {}", error))?;
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(|content| content.as_str())
        .ok_or_else(|| format!("no choices[0].message.content in LLM response: {}", excerpt(text)))?;
    let usage = |field: &str| value.pointer(&format!("/usage/{}", field)).and_then(|v| v.as_u64()).unwrap_or(0);
    let truncated = value.pointer("/choices/0/finish_reason").and_then(|reason| reason.as_str()) == Some("length");
    Ok(Completion {
        truncated,
        text: content.trim().to_string(),
        prompt_tokens: usage("prompt_tokens"),
        completion_tokens: usage("completion_tokens"),
    })
}

/// Host of an already validated base URL, lowercased, without port.
fn url_host(base: &str) -> String {
    let rest = base.split_once("://").map(|(_, rest)| rest).unwrap_or(base);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    authority.rsplit_once(':').map(|(host, _)| host).unwrap_or(authority).to_lowercase()
}

fn excerpt(text: &str) -> String {
    text.chars().take(300).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_requires_https_except_on_loopback() {
        assert!(validate_base_url("https://api.openai.com/v1").is_ok());
        assert!(validate_base_url("http://localhost:11434/v1").is_ok());
        assert!(validate_base_url("http://127.0.0.1:1234/v1").is_ok());
        assert!(validate_base_url("http://[::1]:8001/v1").is_ok());
        assert!(validate_base_url("http://example.com/v1").is_err());
        assert!(validate_base_url("http://localhost.example.com/v1").is_err());
        assert!(validate_base_url("ftp://localhost/v1").is_err());
        assert!(validate_base_url("http://127.0.0.1:1@evil.com/v1").is_err());
        assert!(validate_base_url("http://localhost:80@evil.com/v1").is_err());
        assert!(validate_base_url("https://user:pass@api.openai.com/v1").is_err());
        assert!(validate_base_url("http://localhost?@evil.com").is_ok());
    }

    #[test]
    fn server_text_echoing_the_key_is_redacted() {
        let client = ChatClient::new("http://localhost:1/v1", Some("sk-secret-123".into())).expect("client");
        assert_eq!(client.redact("bad key sk-secret-123 given"), "bad key <redacted> given");
    }

    #[test]
    fn parses_content_and_usage() {
        let parsed = parse_completion(
            r#"{"choices":[{"message":{"role":"assistant","content":" Yes. "}}],"usage":{"prompt_tokens":120,"completion_tokens":2}}"#,
        )
        .expect("valid completion");
        assert_eq!(parsed.text, "Yes.");
        assert_eq!(parsed.prompt_tokens, 120);
        assert_eq!(parsed.completion_tokens, 2);
        assert!(!parsed.truncated);
        let cut = parse_completion(r#"{"choices":[{"message":{"content":"partial"},"finish_reason":"length"}]}"#)
            .expect("valid completion");
        assert!(cut.truncated);
    }

    #[test]
    fn a_local_model_is_given_more_time_than_a_hosted_one() {
        assert_eq!(request_timeout_secs("http://127.0.0.1:8089/v1"), 1800);
        assert_eq!(request_timeout_secs("https://api.openai.com/v1"), 180);
    }

    #[test]
    fn temperature_is_left_out_only_for_anthropic() {
        let anthropic = ChatClient::new("https://api.anthropic.com/v1", None).expect("client");
        assert!(!anthropic.sends_temperature());
        let openai = ChatClient::new("https://api.openai.com/v1", None).expect("client");
        assert!(openai.sends_temperature());
        let local = ChatClient::new("http://localhost:11434/v1", None).expect("client");
        assert!(local.sends_temperature());
        assert_eq!(url_host("https://API.Anthropic.com:443/v1"), "api.anthropic.com");
    }

    #[test]
    fn missing_content_is_an_error() {
        assert!(parse_completion(r#"{"choices":[]}"#).is_err());
        assert!(parse_completion("not json").is_err());
    }
}

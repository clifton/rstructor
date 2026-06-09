//! Drive the **real** provider client over a local mock HTTP server (`mockito`),
//! exercising request building, response parsing, and the retry/re-ask loop in
//! `utils.rs` — the code paths `MockClient` (which mocks at the `LLMClient` trait
//! level, above HTTP) structurally cannot reach. No API key or network needed.
//!
//! Targets `OpenAIClient` because it exposes `base_url`, but the request/response
//! shaping and the retry loop it drives are shared by the OpenAI-compatible path.
#![cfg(feature = "openai")]

use rstructor::{ApiErrorKind, Instructor, LLMClient, OpenAIClient, RStructorError, ThinkingLevel};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[llm(validate = "validate_movie")]
struct Movie {
    title: String,
    year: u16,
}

fn validate_movie(m: &Movie) -> rstructor::Result<()> {
    if m.year < 1888 {
        return Err(RStructorError::ValidationError(
            "year predates cinema".into(),
        ));
    }
    Ok(())
}

/// An OpenAI chat-completion response whose assistant message content is `content`
/// (which, for structured outputs, is the JSON string the client parses into `T`).
fn chat_completion(content: &str) -> String {
    json!({
        "choices": [{
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop",
        }]
    })
    .to_string()
}

fn client(server: &mockito::Server) -> OpenAIClient {
    OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-4o-mini")
}

#[tokio::test]
async fn materialize_parses_a_real_response() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(chat_completion(r#"{"title":"Inception","year":2010}"#))
        .expect(1)
        .create_async()
        .await;

    let movie: Movie = client(&server)
        .materialize("Describe Inception")
        .await
        .unwrap();
    assert_eq!(
        movie,
        Movie {
            title: "Inception".into(),
            year: 2010
        }
    );
    m.assert_async().await;
}

#[tokio::test]
async fn reask_loop_recovers_from_validation_failure() {
    let mut server = mockito::Server::new_async().await;
    // First response fails the validator; the real re-ask loop must retry with the
    // error fed back into the conversation, then accept the corrected response.
    let bad = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_body(chat_completion(r#"{"title":"Old","year":1700}"#))
        .expect(1)
        .create_async()
        .await;
    let good = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_body(chat_completion(r#"{"title":"Metropolis","year":1927}"#))
        .expect(1)
        .create_async()
        .await;

    let movie: Movie = client(&server).materialize("a film").await.unwrap();
    assert_eq!(movie.year, 1927);
    bad.assert_async().await;
    good.assert_async().await;
}

#[tokio::test]
async fn retryable_status_is_retried() {
    let mut server = mockito::Server::new_async().await;
    // 429 with `Retry-After: 0` → the loop retries immediately, then succeeds.
    let rate_limited = server
        .mock("POST", "/chat/completions")
        .with_status(429)
        .with_header("retry-after", "0")
        .with_body("{}")
        .expect(1)
        .create_async()
        .await;
    let ok = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_body(chat_completion(r#"{"title":"Dune","year":2021}"#))
        .expect(1)
        .create_async()
        .await;

    let movie: Movie = client(&server).materialize("a film").await.unwrap();
    assert_eq!(movie.title, "Dune");
    rate_limited.assert_async().await;
    ok.assert_async().await;
}

#[tokio::test]
async fn auth_error_is_surfaced_and_not_retried() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/chat/completions")
        .with_status(401)
        .with_body(r#"{"error":{"message":"invalid api key"}}"#)
        .expect(1) // must NOT be retried
        .create_async()
        .await;

    let err = client(&server)
        .materialize::<Movie>("a film")
        .await
        .unwrap_err();
    assert!(
        matches!(
            err.api_error_kind(),
            Some(ApiErrorKind::AuthenticationFailed)
        ),
        "expected AuthenticationFailed, got {err:?}"
    );
    m.assert_async().await;
}

// ---------------------------------------------------------------------------
// generate / generate_with_metadata over the real client (offline_mockito)
// ---------------------------------------------------------------------------

/// `generate_with_metadata` parses the assistant text content and the `usage`
/// block (prompt/completion/total) into a `GenerateResult`, and the request body
/// for plain text generation must NOT carry a `response_format`.
#[tokio::test]
async fn generate_with_metadata_parses_content_and_usage() {
    let mut server = mockito::Server::new_async().await;
    let body = json!({
        "choices": [{
            "message": { "role": "assistant", "content": "hello there" },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8 },
        "model": "gpt-4o-mini",
    })
    .to_string();
    let captured: std::sync::Arc<std::sync::Mutex<Vec<Value>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = captured.clone();
    let m = server
        .mock("POST", "/chat/completions")
        .match_request(move |req| {
            if let Ok(b) = req.utf8_lossy_body()
                && let Ok(v) = serde_json::from_str::<Value>(&b)
            {
                sink.lock().unwrap().push(v);
            }
            true
        })
        .with_status(200)
        .with_body(body)
        .expect(1)
        .create_async()
        .await;

    let result = client(&server).generate_with_metadata("hi").await.unwrap();
    assert_eq!(result.text, "hello there");
    let usage = result.usage.expect("usage should be parsed");
    assert_eq!(usage.input_tokens, 3);
    assert_eq!(usage.output_tokens, 5);
    assert_eq!(usage.total_tokens(), 8);
    m.assert_async().await;

    // Plain text generation must not request a structured `response_format`.
    let bodies = captured.lock().unwrap();
    assert_eq!(bodies.len(), 1, "expected exactly one request");
    assert!(
        bodies[0].get("response_format").is_none(),
        "response_format must be absent for plain generation, got {}",
        bodies[0]
    );
}

/// `generate` returns just the text content; usage is dropped.
#[tokio::test]
async fn generate_returns_text_content() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_body(chat_completion("plain answer"))
        .expect(1)
        .create_async()
        .await;

    let text = client(&server).generate("hi").await.unwrap();
    assert_eq!(text, "plain answer");
    m.assert_async().await;
}

/// An empty `choices` array must surface as `UnexpectedResponse`, not a panic.
#[tokio::test]
async fn generate_empty_choices_is_unexpected_response() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_body(json!({ "choices": [] }).to_string())
        .expect(1)
        .create_async()
        .await;

    let err = client(&server).generate("hi").await.unwrap_err();
    assert!(
        matches!(
            err.api_error_kind(),
            Some(ApiErrorKind::UnexpectedResponse { .. })
        ),
        "expected UnexpectedResponse, got {err:?}"
    );
    m.assert_async().await;
}

/// A choice whose message has `content: null` must surface as `UnexpectedResponse`.
#[tokio::test]
async fn generate_null_content_is_unexpected_response() {
    let mut server = mockito::Server::new_async().await;
    let body = json!({
        "choices": [{
            "message": { "role": "assistant", "content": null },
            "finish_reason": "stop",
        }]
    })
    .to_string();
    let m = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_body(body)
        .expect(1)
        .create_async()
        .await;

    let err = client(&server).generate("hi").await.unwrap_err();
    assert!(
        matches!(
            err.api_error_kind(),
            Some(ApiErrorKind::UnexpectedResponse { .. })
        ),
        "expected UnexpectedResponse, got {err:?}"
    );
    m.assert_async().await;
}

// ---------------------------------------------------------------------------
// reasoning_effort + temperature override per model (offline_mockito)
// ---------------------------------------------------------------------------

/// A GPT-5 model with the default thinking level (Medium) sends
/// `reasoning_effort: "medium"` and forces `temperature` to 1.0, even though the
/// configured temperature is 0.0.
#[tokio::test]
async fn gpt5_sends_reasoning_effort_and_forces_temperature_one() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({
            "reasoning_effort": "medium",
            "temperature": 1.0,
        })))
        .with_status(200)
        .with_body(chat_completion("ok"))
        .expect(1)
        .create_async()
        .await;

    // Default temperature is 0.0; reasoning must override it to 1.0 for gpt-5.
    let text = OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-5")
        .generate("hi")
        .await
        .unwrap();
    assert_eq!(text, "ok");
    m.assert_async().await;
}

/// A non-GPT-5 model (gpt-4o-mini) omits `reasoning_effort` entirely and passes
/// the configured temperature through unchanged.
#[tokio::test]
async fn non_gpt5_omits_reasoning_effort_and_passes_temperature_through() {
    let mut server = mockito::Server::new_async().await;
    let captured: std::sync::Arc<std::sync::Mutex<Vec<Value>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = captured.clone();
    let m = server
        .mock("POST", "/chat/completions")
        .match_request(move |req| {
            if let Ok(body) = req.utf8_lossy_body()
                && let Ok(v) = serde_json::from_str::<Value>(&body)
            {
                sink.lock().unwrap().push(v);
            }
            true
        })
        .with_status(200)
        .with_body(chat_completion("ok"))
        .expect(1)
        .create_async()
        .await;

    let text = OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-4o-mini")
        .temperature(0.2)
        .generate("hi")
        .await
        .unwrap();
    assert_eq!(text, "ok");
    m.assert_async().await;

    let bodies = captured.lock().unwrap();
    assert_eq!(bodies.len(), 1, "expected exactly one request");
    let body = &bodies[0];
    assert!(
        body.get("reasoning_effort").is_none(),
        "reasoning_effort must be omitted for non-gpt-5, got {body}"
    );
    assert_eq!(
        body["temperature"],
        json!(0.2),
        "configured temperature must pass through unchanged"
    );
}

// ---------------------------------------------------------------------------
// o-series request parameters (offline_mockito)
// ---------------------------------------------------------------------------

/// Mount a catch-all `/chat/completions` mock that records every request body
/// into the returned buffer and responds with `chat_completion(content)`.
async fn capture_chat_request(
    server: &mut mockito::Server,
    content: &str,
) -> (mockito::Mock, std::sync::Arc<std::sync::Mutex<Vec<Value>>>) {
    let captured: std::sync::Arc<std::sync::Mutex<Vec<Value>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = captured.clone();
    let mock = server
        .mock("POST", "/chat/completions")
        .match_request(move |req| {
            if let Ok(body) = req.utf8_lossy_body()
                && let Ok(v) = serde_json::from_str::<Value>(&body)
            {
                sink.lock().unwrap().push(v);
            }
            true
        })
        .with_status(200)
        .with_body(chat_completion(content))
        .expect(1)
        .create_async()
        .await;
    (mock, captured)
}

/// o-series reasoning models (here `o3`) reject `temperature` (400) and require
/// `max_completion_tokens` instead of `max_tokens`. The structured-output
/// request must omit `temperature` and `max_tokens` entirely, carry the
/// configured limit as `max_completion_tokens`, and still send
/// `reasoning_effort` (the default thinking level is Medium).
#[tokio::test]
async fn o_series_omits_temperature_and_maps_max_tokens() {
    let mut server = mockito::Server::new_async().await;
    let (m, captured) =
        capture_chat_request(&mut server, r#"{"title":"Inception","year":2010}"#).await;

    let movie: Movie = OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("o3")
        .temperature(0.7)
        .max_tokens(1234)
        .materialize("Describe Inception")
        .await
        .unwrap();
    assert_eq!(movie.title, "Inception");
    m.assert_async().await;

    let bodies = captured.lock().unwrap();
    assert_eq!(bodies.len(), 1, "expected exactly one request");
    let body = &bodies[0];
    assert!(
        body.get("temperature").is_none(),
        "temperature must be omitted for o-series models, got {body}"
    );
    assert!(
        body.get("max_tokens").is_none(),
        "max_tokens must be omitted for o-series models, got {body}"
    );
    assert_eq!(
        body["max_completion_tokens"],
        json!(1234),
        "configured max_tokens must be sent as max_completion_tokens"
    );
    assert_eq!(
        body["reasoning_effort"],
        json!("medium"),
        "default thinking level (Medium) must map to reasoning_effort"
    );
}

/// An o-series model with `ThinkingLevel::Off` must omit `reasoning_effort`
/// (o-series models don't accept `"none"`), and with no `max_tokens` configured
/// no token-limit key may appear at all.
#[tokio::test]
async fn o_series_thinking_off_omits_reasoning_effort() {
    let mut server = mockito::Server::new_async().await;
    let (m, captured) = capture_chat_request(&mut server, "ok").await;

    let text = OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("o4-mini")
        .thinking_level(ThinkingLevel::Off)
        .generate("hi")
        .await
        .unwrap();
    assert_eq!(text, "ok");
    m.assert_async().await;

    let bodies = captured.lock().unwrap();
    assert_eq!(bodies.len(), 1, "expected exactly one request");
    let body = &bodies[0];
    assert!(
        body.get("reasoning_effort").is_none(),
        "reasoning_effort 'none' must not be sent to o-series models, got {body}"
    );
    assert!(
        body.get("temperature").is_none(),
        "temperature must be omitted for o-series models, got {body}"
    );
    assert!(
        body.get("max_tokens").is_none(),
        "max_tokens must be omitted for o-series models, got {body}"
    );
    assert!(
        body.get("max_completion_tokens").is_none(),
        "max_completion_tokens must be absent when no max_tokens is configured, got {body}"
    );
}

/// Non-reasoning models keep the existing contract: `temperature` and
/// `max_tokens` are sent as configured and `max_completion_tokens` never
/// appears.
#[tokio::test]
async fn non_reasoning_model_keeps_temperature_and_max_tokens() {
    let mut server = mockito::Server::new_async().await;
    let (m, captured) = capture_chat_request(&mut server, "ok").await;

    let text = OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-4o")
        .temperature(0.5)
        .max_tokens(64)
        .generate("hi")
        .await
        .unwrap();
    assert_eq!(text, "ok");
    m.assert_async().await;

    let bodies = captured.lock().unwrap();
    assert_eq!(bodies.len(), 1, "expected exactly one request");
    let body = &bodies[0];
    assert_eq!(body["temperature"], json!(0.5));
    assert_eq!(body["max_tokens"], json!(64));
    assert!(
        body.get("max_completion_tokens").is_none(),
        "max_completion_tokens must never be sent for non-o-series models, got {body}"
    );
}

/// Grok shares the OpenAI-compatible request struct but has no o-series
/// models: its requests must still always carry `temperature` and `max_tokens`
/// and never `max_completion_tokens` or `reasoning_effort`.
#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_request_still_sends_temperature_and_max_tokens() {
    use rstructor::GrokClient;

    let mut server = mockito::Server::new_async().await;
    let (m, captured) = capture_chat_request(&mut server, "ok").await;

    let text = GrokClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("grok-3")
        .temperature(0.5)
        .max_tokens(64)
        .generate("hi")
        .await
        .unwrap();
    assert_eq!(text, "ok");
    m.assert_async().await;

    let bodies = captured.lock().unwrap();
    assert_eq!(bodies.len(), 1, "expected exactly one request");
    let body = &bodies[0];
    assert_eq!(
        body["temperature"],
        json!(0.5),
        "Grok must always send temperature"
    );
    assert_eq!(
        body["max_tokens"],
        json!(64),
        "Grok must always send max_tokens"
    );
    assert!(
        body.get("max_completion_tokens").is_none(),
        "Grok must never send max_completion_tokens, got {body}"
    );
    assert!(
        body.get("reasoning_effort").is_none(),
        "Grok must never send reasoning_effort, got {body}"
    );
}

// ---------------------------------------------------------------------------
// list_models prefix filter (offline_mockito)
// ---------------------------------------------------------------------------

/// `list_models` keeps only chat-completion model ids (those prefixed
/// `gpt-`/`o1`/`o3`/`o4`) and drops embeddings, whisper, dall-e, etc.
#[tokio::test]
async fn list_models_keeps_only_chat_models() {
    let mut server = mockito::Server::new_async().await;
    let body = json!({
        "data": [
            { "id": "gpt-4o" },
            { "id": "o3" },
            { "id": "o4-mini" },
            { "id": "o1-pro" },
            { "id": "whisper-1" },
            { "id": "text-embedding-3-small" },
            { "id": "dall-e-3" },
        ]
    })
    .to_string();
    let m = server
        .mock("GET", "/models")
        .with_status(200)
        .with_body(body)
        .expect(1)
        .create_async()
        .await;

    let models = client(&server).list_models().await.unwrap();
    let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, vec!["gpt-4o", "o3", "o4-mini", "o1-pro"]);
    m.assert_async().await;
}

/// A models response with no `data` key yields an empty list (not an error).
#[tokio::test]
async fn list_models_no_data_returns_empty() {
    let mut server = mockito::Server::new_async().await;
    let m = server
        .mock("GET", "/models")
        .with_status(200)
        .with_body("{}")
        .expect(1)
        .create_async()
        .await;

    let models = client(&server).list_models().await.unwrap();
    assert!(models.is_empty(), "expected empty list, got {models:?}");
    m.assert_async().await;
}

// ---------------------------------------------------------------------------
// usage model-name fallback (offline_mockito)
// ---------------------------------------------------------------------------

/// When the completion response omits the `model` field, the parsed usage's
/// model name falls back to the client's configured model.
#[tokio::test]
async fn usage_model_name_falls_back_to_client_model() {
    let mut server = mockito::Server::new_async().await;
    // No "model" field in the response body.
    let body = json!({
        "choices": [{
            "message": { "role": "assistant", "content": "hi" },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 },
    })
    .to_string();
    let m = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_body(body)
        .expect(1)
        .create_async()
        .await;

    let result = OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-4o-mini")
        .generate_with_metadata("hi")
        .await
        .unwrap();
    let usage = result.usage.expect("usage should be parsed");
    assert_eq!(usage.model, "gpt-4o-mini");
    m.assert_async().await;
}

// ---------------------------------------------------------------------------
// OpenAI tool-calling loop over the real client (offline_mockito, tools feature)
// ---------------------------------------------------------------------------

/// A chat-completion response in which the assistant requests a single tool call
/// `name(args)` with id `call_id`.
#[cfg(feature = "tools")]
fn tool_call_response(call_id: &str, name: &str, args: &str) -> String {
    json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": { "name": name, "arguments": args },
                }],
            },
            "finish_reason": "tool_calls",
        }]
    })
    .to_string()
}

/// Build an `add` tool whose closure flips the shared flag when invoked and
/// returns `{sum: a + b}`.
#[cfg(feature = "tools")]
fn recording_add_tool(
    flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> rstructor::FnTool<
    AddArgs,
    impl Fn(AddArgs) -> std::future::Ready<rstructor::Result<Value>> + Clone,
> {
    rstructor::FnTool::new("add", "Add two integers", move |args: AddArgs| {
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
        std::future::ready(Ok(json!({ "sum": args.a + args.b })))
    })
}

#[cfg(feature = "tools")]
#[derive(Instructor, Serialize, Deserialize)]
struct AddArgs {
    #[llm(description = "First addend")]
    a: i64,
    #[llm(description = "Second addend")]
    b: i64,
}

/// Full OpenAI tool round-trip: the first response asks for a tool call, the loop
/// executes the (real) tool and feeds the result back as a `role: tool` message
/// carrying the original `tool_call_id`, and the second response returns the final
/// text answer.
#[cfg(feature = "tools")]
#[tokio::test]
async fn tool_loop_full_round_trip() {
    use rstructor::{RequestExt, Toolbox};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let mut server = mockito::Server::new_async().await;

    // Capture every request body so we can assert the fed-back tool message shape.
    let captured: Arc<std::sync::Mutex<Vec<Value>>> = Arc::new(std::sync::Mutex::new(Vec::new()));

    // First request: no `tool` messages yet -> respond with a tool_call.
    let sink1 = captured.clone();
    let first = server
        .mock("POST", "/chat/completions")
        .match_request(move |req| {
            let v: Value = serde_json::from_str(&req.utf8_lossy_body().unwrap()).unwrap();
            sink1.lock().unwrap().push(v.clone());
            // Match only when no tool result has been fed back yet.
            !messages_contain_tool_role(&v)
        })
        .with_status(200)
        .with_body(tool_call_response("c1", "add", r#"{"a":2,"b":3}"#))
        .expect(1)
        .create_async()
        .await;

    // Second request: a `tool` message is present -> respond with final answer.
    let sink2 = captured.clone();
    let second = server
        .mock("POST", "/chat/completions")
        .match_request(move |req| {
            let v: Value = serde_json::from_str(&req.utf8_lossy_body().unwrap()).unwrap();
            sink2.lock().unwrap().push(v.clone());
            messages_contain_tool_role(&v)
        })
        .with_status(200)
        .with_body(chat_completion("the sum is 5"))
        .expect(1)
        .create_async()
        .await;

    let invoked = Arc::new(AtomicBool::new(false));
    let toolbox = Toolbox::new().with(recording_add_tool(invoked.clone()));

    let answer = client(&server)
        .with_tools(&toolbox)
        .run("add 2 and 3")
        .await
        .unwrap();

    assert_eq!(answer, "the sum is 5");
    assert!(
        invoked.load(Ordering::SeqCst),
        "the real tool closure must have run"
    );
    first.assert_async().await;
    second.assert_async().await;

    // The second request's last message must be the tool result, tagged with the
    // original tool_call_id and containing the computed sum.
    let bodies = captured.lock().unwrap();
    let second_body = bodies
        .iter()
        .find(|v| messages_contain_tool_role(v))
        .expect("a request carrying the tool result must exist");
    let messages = second_body["messages"].as_array().unwrap();
    let tool_msg = messages
        .iter()
        .find(|m| m["role"] == json!("tool"))
        .expect("a role:tool message must be present");
    assert_eq!(tool_msg["tool_call_id"], json!("c1"));
    let content = tool_msg["content"].as_str().unwrap();
    assert!(
        content.contains("\"sum\":5"),
        "tool result content should carry the sum, got {content}"
    );
}

/// Helper: does the request body contain a message with `role: "tool"`?
#[cfg(feature = "tools")]
fn messages_contain_tool_role(body: &Value) -> bool {
    body.get("messages")
        .and_then(Value::as_array)
        .map(|msgs| msgs.iter().any(|m| m.get("role") == Some(&json!("tool"))))
        .unwrap_or(false)
}

/// When the model calls a tool that does not exist in the toolbox, the loop feeds
/// back a `role: tool` message whose content is `{"error":"unknown tool: …"}` and
/// continues; the model then produces a final answer.
#[cfg(feature = "tools")]
#[tokio::test]
async fn tool_loop_unknown_tool_continues() {
    use rstructor::{RequestExt, Toolbox};
    use std::sync::Arc;

    let mut server = mockito::Server::new_async().await;
    let captured: Arc<std::sync::Mutex<Vec<Value>>> = Arc::new(std::sync::Mutex::new(Vec::new()));

    let sink1 = captured.clone();
    let first = server
        .mock("POST", "/chat/completions")
        .match_request(move |req| {
            let v: Value = serde_json::from_str(&req.utf8_lossy_body().unwrap()).unwrap();
            sink1.lock().unwrap().push(v.clone());
            !messages_contain_tool_role(&v)
        })
        .with_status(200)
        // Model calls a tool that is NOT in the toolbox.
        .with_body(tool_call_response("c1", "does_not_exist", "{}"))
        .expect(1)
        .create_async()
        .await;

    let sink2 = captured.clone();
    let second = server
        .mock("POST", "/chat/completions")
        .match_request(move |req| {
            let v: Value = serde_json::from_str(&req.utf8_lossy_body().unwrap()).unwrap();
            sink2.lock().unwrap().push(v.clone());
            messages_contain_tool_role(&v)
        })
        .with_status(200)
        .with_body(chat_completion("recovered"))
        .expect(1)
        .create_async()
        .await;

    let invoked = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let toolbox = Toolbox::new().with(recording_add_tool(invoked.clone()));

    let answer = client(&server)
        .with_tools(&toolbox)
        .run("call a missing tool")
        .await
        .unwrap();

    assert_eq!(answer, "recovered");
    assert!(
        !invoked.load(std::sync::atomic::Ordering::SeqCst),
        "the real add tool must NOT have run for an unknown tool"
    );
    first.assert_async().await;
    second.assert_async().await;

    let bodies = captured.lock().unwrap();
    let second_body = bodies
        .iter()
        .find(|v| messages_contain_tool_role(v))
        .expect("a request carrying the error result must exist");
    let messages = second_body["messages"].as_array().unwrap();
    let tool_msg = messages
        .iter()
        .find(|m| m["role"] == json!("tool"))
        .expect("a role:tool message must be present");
    let content = tool_msg["content"].as_str().unwrap();
    assert!(
        content.contains("unknown tool: does_not_exist"),
        "error content should name the unknown tool, got {content}"
    );
}

/// When a tool's closure returns `Err`, the loop swallows it into a `role: tool`
/// message containing `{"error":…}` and continues to a final answer.
#[cfg(feature = "tools")]
#[tokio::test]
async fn tool_loop_tool_error_is_swallowed() {
    use rstructor::{FnTool, RequestExt, Toolbox};
    use std::sync::Arc;

    let mut server = mockito::Server::new_async().await;
    let captured: Arc<std::sync::Mutex<Vec<Value>>> = Arc::new(std::sync::Mutex::new(Vec::new()));

    let sink1 = captured.clone();
    let first = server
        .mock("POST", "/chat/completions")
        .match_request(move |req| {
            let v: Value = serde_json::from_str(&req.utf8_lossy_body().unwrap()).unwrap();
            sink1.lock().unwrap().push(v.clone());
            !messages_contain_tool_role(&v)
        })
        .with_status(200)
        .with_body(tool_call_response("c1", "boom", r#"{"a":1,"b":1}"#))
        .expect(1)
        .create_async()
        .await;

    let sink2 = captured.clone();
    let second = server
        .mock("POST", "/chat/completions")
        .match_request(move |req| {
            let v: Value = serde_json::from_str(&req.utf8_lossy_body().unwrap()).unwrap();
            sink2.lock().unwrap().push(v.clone());
            messages_contain_tool_role(&v)
        })
        .with_status(200)
        .with_body(chat_completion("handled"))
        .expect(1)
        .create_async()
        .await;

    // A tool that always errors.
    let boom = FnTool::new("boom", "always fails", |_args: AddArgs| {
        std::future::ready(Err(RStructorError::ValidationError(
            "tool blew up".to_string(),
        )))
    });
    let toolbox = Toolbox::new().with(boom);

    let answer = client(&server)
        .with_tools(&toolbox)
        .run("trigger the failing tool")
        .await
        .unwrap();

    assert_eq!(answer, "handled");
    first.assert_async().await;
    second.assert_async().await;

    let bodies = captured.lock().unwrap();
    let second_body = bodies
        .iter()
        .find(|v| messages_contain_tool_role(v))
        .expect("a request carrying the error result must exist");
    let messages = second_body["messages"].as_array().unwrap();
    let tool_msg = messages
        .iter()
        .find(|m| m["role"] == json!("tool"))
        .expect("a role:tool message must be present");
    let content = tool_msg["content"].as_str().unwrap();
    assert!(
        content.contains("error"),
        "swallowed tool error should appear in the content, got {content}"
    );
    assert!(
        content.contains("tool blew up"),
        "the tool's error message should be preserved, got {content}"
    );
}

/// When the model never stops calling tools, the loop gives up after
/// `max_iterations` round-trips and returns a `ValidationError` whose message says
/// it "did not converge" and names the iteration budget.
#[cfg(feature = "tools")]
#[tokio::test]
async fn tool_loop_exhaustion_errors() {
    use rstructor::{RequestExt, Toolbox};
    use std::sync::Arc;

    let mut server = mockito::Server::new_async().await;
    // Every response asks for another tool call -> the loop never converges.
    let always_tool = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_body(tool_call_response("c1", "add", r#"{"a":1,"b":1}"#))
        // max_iterations(2) -> exactly two model round-trips before giving up.
        .expect(2)
        .create_async()
        .await;

    let invoked = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let toolbox = Toolbox::new().with(recording_add_tool(invoked.clone()));

    let err = client(&server)
        .with_tools(&toolbox)
        .max_iterations(2)
        .run("loop forever")
        .await
        .unwrap_err();

    let msg = err.to_string();
    assert!(
        matches!(err, RStructorError::ValidationError(_)),
        "expected ValidationError, got {err:?}"
    );
    assert!(
        msg.contains("did not converge"),
        "error should say it did not converge, got: {msg}"
    );
    assert!(
        msg.contains('2'),
        "error should mention the iteration budget (2), got: {msg}"
    );
    always_tool.assert_async().await;
}

//! Regression coverage for first-class system prompt request shapes.
//!
//! These tests drive each real provider client over a local HTTP server. The
//! stable system instructions must remain separate from the dynamic user turn
//! so providers can apply the correct instruction semantics and reuse a common
//! prompt-cache prefix across requests.

#![cfg(feature = "_client")]

#[cfg(feature = "derive")]
use rstructor::Instructor;
use rstructor::RequestExt;
#[cfg(all(feature = "derive", feature = "openai"))]
use rstructor::{AnyClient, MediaFile};
#[cfg(feature = "derive")]
use serde::{Deserialize, Serialize};
use serde_json::json;

#[cfg(feature = "streaming")]
use futures_util::StreamExt;

#[cfg(feature = "derive")]
#[derive(Debug, Deserialize, Instructor, PartialEq, Serialize)]
struct RiskSummary {
    status: String,
    gross_exposure: f64,
}

#[cfg(any(feature = "openai", feature = "grok"))]
fn openai_compatible_completion(text: &str) -> String {
    json!({
        "choices": [{
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop",
        }]
    })
    .to_string()
}

#[cfg(feature = "streaming")]
const OPENAI_COMPATIBLE_TEXT_STREAM: &str = concat!(
    "data: {\"id\":\"chatcmpl-risk\",\"object\":\"chat.completion.chunk\",",
    "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",",
    "\"content\":\"Risk is within limits.\"},\"finish_reason\":\"stop\"}]}\n\n",
    "data: [DONE]\n\n",
);

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_generate_sends_a_system_message_before_the_user_message() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({
            "messages": [
                { "role": "system", "content": "Use the fund's reporting policy." },
                { "role": "user", "content": "Summarize today's risk." },
            ],
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_compatible_completion("Risk is within limits."))
        .expect(1)
        .create_async()
        .await;

    let response = rstructor::OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-4.1-mini")
        .with_system("Use the fund's reporting policy.")
        .generate("Summarize today's risk.")
        .await
        .unwrap();

    assert_eq!(response, "Risk is within limits.");
    request.assert_async().await;
}

#[cfg(all(feature = "derive", feature = "openai"))]
#[tokio::test]
async fn openai_materialize_keeps_system_separate_from_a_media_user_turn() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({
            "messages": [
                { "role": "system", "content": "Use the fund's reporting policy." },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Read the exposure chart." },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgo=",
                                "detail": "auto",
                            },
                        },
                    ],
                },
            ],
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_compatible_completion(
            r#"{"status":"within_limits","gross_exposure":1.42}"#,
        ))
        .expect(1)
        .create_async()
        .await;

    let media = [MediaFile::from_bytes(b"\x89PNG\r\n\x1a\n", "image/png")];
    let summary: RiskSummary = rstructor::OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-4.1-mini")
        .no_retries()
        .with_system("Use the fund's reporting policy.")
        .media(media.to_vec())
        .materialize("Read the exposure chart.")
        .await
        .unwrap();

    assert_eq!(
        summary,
        RiskSummary {
            status: "within_limits".to_string(),
            gross_exposure: 1.42,
        }
    );
    request.assert_async().await;
}

#[cfg(all(feature = "derive", feature = "openai"))]
#[tokio::test]
async fn openai_attempt_ledger_terminal_keeps_the_system_message_separate() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({
            "messages": [
                { "role": "system", "content": "Use the fund's reporting policy." },
                { "role": "user", "content": "Summarize today's risk." },
            ],
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_compatible_completion(
            r#"{"status":"within_limits","gross_exposure":1.42}"#,
        ))
        .expect(1)
        .create_async()
        .await;

    let report = rstructor::OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-4.1-mini")
        .no_retries()
        .with_system("Use the fund's reporting policy.")
        .materialize_with_attempts::<RiskSummary>("Summarize today's risk.")
        .await
        .unwrap();

    assert_eq!(report.data.gross_exposure, 1.42);
    assert_eq!(report.attempts.len(), 1);
    request.assert_async().await;
}

#[cfg(all(feature = "derive", feature = "openai"))]
#[tokio::test]
async fn any_client_dispatch_preserves_the_system_message() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({
            "messages": [
                { "role": "system", "content": "Use the fund's reporting policy." },
                { "role": "user", "content": "Summarize today's risk." },
            ],
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_compatible_completion(
            r#"{"status":"within_limits","gross_exposure":1.42}"#,
        ))
        .expect(1)
        .create_async()
        .await;

    let client: AnyClient = rstructor::OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-4.1-mini")
        .no_retries()
        .into();
    let summary: RiskSummary = client
        .with_system("Use the fund's reporting policy.")
        .materialize("Summarize today's risk.")
        .await
        .unwrap();

    assert_eq!(summary.status, "within_limits");
    request.assert_async().await;
}

#[cfg(all(feature = "openai", feature = "tools"))]
#[tokio::test]
async fn run_without_tools_still_sends_a_first_class_system_message() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({
            "messages": [
                { "role": "system", "content": "Use the fund's reporting policy." },
                { "role": "user", "content": "Summarize today's risk." },
            ],
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_compatible_completion("Risk is within limits."))
        .expect(1)
        .create_async()
        .await;

    let response = rstructor::OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-4.1-mini")
        .with_system("Use the fund's reporting policy.")
        .run("Summarize today's risk.")
        .await
        .unwrap();

    assert_eq!(response, "Risk is within limits.");
    request.assert_async().await;
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_generate_uses_the_top_level_system_field() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/messages")
        .match_body(mockito::Matcher::PartialJson(json!({
            "system": "Use the fund's reporting policy.",
            "messages": [
                { "role": "user", "content": "Summarize today's risk." },
            ],
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "content": [{ "type": "text", "text": "Risk is within limits." }],
                "usage": { "input_tokens": 12, "output_tokens": 5 },
                "model": "claude-sonnet-4-5-20250929",
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let response = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("claude-sonnet-4-5-20250929")
        .with_system("Use the fund's reporting policy.")
        .generate("Summarize today's risk.")
        .await
        .unwrap();

    assert_eq!(response, "Risk is within limits.");
    request.assert_async().await;
}

#[cfg(all(feature = "anthropic", feature = "derive"))]
#[tokio::test]
async fn anthropic_materialize_uses_the_top_level_system_field() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/messages")
        .match_body(mockito::Matcher::PartialJson(json!({
            "system": "Use the fund's reporting policy.",
            "messages": [
                { "role": "user", "content": "Summarize today's risk." },
            ],
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "content": [{
                    "type": "text",
                    "text": r#"{"status":"within_limits","gross_exposure":1.42}"#,
                }],
                "usage": { "input_tokens": 12, "output_tokens": 5 },
                "model": "claude-sonnet-4-5-20250929",
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let summary = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("claude-sonnet-4-5-20250929")
        .no_retries()
        .with_system("Use the fund's reporting policy.")
        .materialize::<RiskSummary>("Summarize today's risk.")
        .await
        .unwrap();

    assert_eq!(summary.gross_exposure, 1.42);
    request.assert_async().await;
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_generate_uses_system_instruction() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/models/test-model:generateContent")
        .match_query(mockito::Matcher::UrlEncoded(
            "key".to_string(),
            "test-key".to_string(),
        ))
        .match_body(mockito::Matcher::PartialJson(json!({
            "systemInstruction": {
                "parts": [{ "text": "Use the fund's reporting policy." }],
            },
            "contents": [{
                "role": "user",
                "parts": [{ "text": "Summarize today's risk." }],
            }],
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "Risk is within limits." }],
                    },
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 5,
                },
                "modelVersion": "test-model",
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let response = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-model")
        .with_system("Use the fund's reporting policy.")
        .generate("Summarize today's risk.")
        .await
        .unwrap();

    assert_eq!(response, "Risk is within limits.");
    request.assert_async().await;
}

#[cfg(all(feature = "gemini", feature = "derive"))]
#[tokio::test]
async fn gemini_materialize_uses_system_instruction() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/models/test-model:generateContent")
        .match_query(mockito::Matcher::UrlEncoded(
            "key".to_string(),
            "test-key".to_string(),
        ))
        .match_body(mockito::Matcher::PartialJson(json!({
            "systemInstruction": {
                "parts": [{ "text": "Use the fund's reporting policy." }],
            },
            "contents": [{
                "role": "user",
                "parts": [{ "text": "Summarize today's risk." }],
            }],
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": r#"{"status":"within_limits","gross_exposure":1.42}"#,
                        }],
                    },
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 5,
                },
                "modelVersion": "test-model",
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let summary = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-model")
        .no_retries()
        .with_system("Use the fund's reporting policy.")
        .materialize::<RiskSummary>("Summarize today's risk.")
        .await
        .unwrap();

    assert_eq!(summary.gross_exposure, 1.42);
    request.assert_async().await;
}

#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_generate_sends_a_system_message_before_the_user_message() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({
            "messages": [
                { "role": "system", "content": "Use the fund's reporting policy." },
                { "role": "user", "content": "Summarize today's risk." },
            ],
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_compatible_completion("Risk is within limits."))
        .expect(1)
        .create_async()
        .await;

    let response = rstructor::GrokClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("grok-4.1-fast")
        .with_system("Use the fund's reporting policy.")
        .generate("Summarize today's risk.")
        .await
        .unwrap();

    assert_eq!(response, "Risk is within limits.");
    request.assert_async().await;
}

#[cfg(all(feature = "grok", feature = "derive"))]
#[tokio::test]
async fn grok_materialize_sends_a_system_message_before_the_user_message() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({
            "messages": [
                { "role": "system", "content": "Use the fund's reporting policy." },
                { "role": "user", "content": "Summarize today's risk." },
            ],
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_compatible_completion(
            r#"{"status":"within_limits","gross_exposure":1.42}"#,
        ))
        .expect(1)
        .create_async()
        .await;

    let summary = rstructor::GrokClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("grok-4.1-fast")
        .no_retries()
        .with_system("Use the fund's reporting policy.")
        .materialize::<RiskSummary>("Summarize today's risk.")
        .await
        .unwrap();

    assert_eq!(summary.gross_exposure, 1.42);
    request.assert_async().await;
}

#[cfg(all(feature = "openai", feature = "streaming"))]
#[tokio::test]
async fn openai_stream_keeps_the_system_message_separate() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({
            "stream": true,
            "messages": [
                { "role": "system", "content": "Use the fund's reporting policy." },
                { "role": "user", "content": "Summarize today's risk." },
            ],
        })))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(OPENAI_COMPATIBLE_TEXT_STREAM)
        .expect(1)
        .create_async()
        .await;

    let chunks = rstructor::OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-4.1-mini")
        .with_system("Use the fund's reporting policy.")
        .generate_stream("Summarize today's risk.")
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        chunks
            .into_iter()
            .collect::<rstructor::Result<Vec<_>>>()
            .unwrap(),
        vec!["Risk is within limits."]
    );
    request.assert_async().await;
}

#[cfg(all(feature = "anthropic", feature = "streaming"))]
#[tokio::test]
async fn anthropic_stream_keeps_the_top_level_system_field() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/messages")
        .match_body(mockito::Matcher::PartialJson(json!({
            "stream": true,
            "system": "Use the fund's reporting policy.",
            "messages": [
                { "role": "user", "content": "Summarize today's risk." },
            ],
        })))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(include_str!(
            "fixtures/streaming/anthropic_text_complete.sse"
        ))
        .expect(1)
        .create_async()
        .await;

    let chunks = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("claude-sonnet-4-5-20250929")
        .with_system("Use the fund's reporting policy.")
        .generate_stream("Summarize today's risk.")
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        chunks
            .into_iter()
            .collect::<rstructor::Result<Vec<_>>>()
            .unwrap(),
        vec!["Net delta: $18.4mm"]
    );
    request.assert_async().await;
}

#[cfg(all(feature = "gemini", feature = "streaming"))]
#[tokio::test]
async fn gemini_stream_keeps_system_instruction_separate() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/models/test-model:streamGenerateContent")
        .match_query(mockito::Matcher::AllOf(vec![
            mockito::Matcher::UrlEncoded("alt".to_string(), "sse".to_string()),
            mockito::Matcher::UrlEncoded("key".to_string(), "test-key".to_string()),
        ]))
        .match_body(mockito::Matcher::PartialJson(json!({
            "systemInstruction": {
                "parts": [{ "text": "Use the fund's reporting policy." }],
            },
            "contents": [{
                "role": "user",
                "parts": [{ "text": "Summarize today's risk." }],
            }],
        })))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(include_str!("fixtures/streaming/gemini_text_complete.sse"))
        .expect(1)
        .create_async()
        .await;

    let chunks = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-model")
        .with_system("Use the fund's reporting policy.")
        .generate_stream("Summarize today's risk.")
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        chunks
            .into_iter()
            .collect::<rstructor::Result<Vec<_>>>()
            .unwrap(),
        vec!["VaR: $2.1mm"]
    );
    request.assert_async().await;
}

#[cfg(all(feature = "grok", feature = "streaming"))]
#[tokio::test]
async fn grok_stream_keeps_the_system_message_separate() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({
            "stream": true,
            "messages": [
                { "role": "system", "content": "Use the fund's reporting policy." },
                { "role": "user", "content": "Summarize today's risk." },
            ],
        })))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(OPENAI_COMPATIBLE_TEXT_STREAM)
        .expect(1)
        .create_async()
        .await;

    let chunks = rstructor::GrokClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("grok-4.1-fast")
        .with_system("Use the fund's reporting policy.")
        .generate_stream("Summarize today's risk.")
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        chunks
            .into_iter()
            .collect::<rstructor::Result<Vec<_>>>()
            .unwrap(),
        vec!["Risk is within limits."]
    );
    request.assert_async().await;
}

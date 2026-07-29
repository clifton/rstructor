//! Provider-envelope coverage for the additive materialization attempt ledger.
//!
//! These tests drive each concrete client through a local HTTP server. The
//! fixtures use a realistic fund position so provider-specific response and
//! usage parsing are tested without paid network calls.
#![cfg(any(feature = "anthropic", feature = "gemini", feature = "grok"))]

use rstructor::{
    AttemptKind, AttemptOutcome, Instructor, LLMClient, MaterializeFailure, MaterializeReport,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Instructor, Serialize, Deserialize, PartialEq)]
struct Position {
    fund_id: String,
    symbol: String,
    quantity: i64,
}

const POSITION_JSON: &str = r#"{"fund_id":"HF-ALPHA-001","symbol":"ESU6","quantity":-240}"#;

fn assert_single_success(
    report: MaterializeReport<Position>,
    expected_model: &str,
    expected_tokens: u64,
) {
    assert_eq!(report.data.fund_id, "HF-ALPHA-001");
    assert_eq!(report.data.symbol, "ESU6");
    assert_eq!(report.data.quantity, -240);
    assert_eq!(report.attempts.len(), 1);
    assert_eq!(report.attempts[0].kind, AttemptKind::Semantic);
    assert_eq!(report.attempts[0].outcome, AttemptOutcome::Succeeded);
    assert_eq!(report.final_usage.as_ref().unwrap().model, expected_model);
    assert_eq!(
        report.cumulative_usage.as_ref().unwrap().total_tokens(),
        expected_tokens
    );
}

fn assert_usage_bearing_protocol_failure(failure: MaterializeFailure, expected_tokens: u64) {
    assert_eq!(failure.attempts.len(), 1);
    assert_eq!(failure.attempts[0].kind, AttemptKind::Transport);
    assert!(matches!(
        failure.attempts[0].outcome,
        AttemptOutcome::Failed { retried: false, .. }
    ));
    assert_eq!(
        failure.attempts[0].usage.as_ref().unwrap().total_tokens(),
        expected_tokens
    );
    assert_eq!(
        failure.cumulative_usage.as_ref().unwrap().total_tokens(),
        expected_tokens
    );
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_attempt_report_parses_usage() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "content": [{ "type": "text", "text": POSITION_JSON }],
                "model": "claude-risk-router-2026-07-15",
                "usage": { "input_tokens": 70, "output_tokens": 14 },
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let report = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-model")
        .materialize_with_attempts::<Position>("reconcile the futures position")
        .await
        .unwrap();

    assert_single_success(report, "claude-risk-router-2026-07-15", 84);
    request.assert_async().await;
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_protocol_failure_keeps_reported_usage() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "content": [],
                "model": "claude-risk-router",
                "usage": { "input_tokens": 50, "output_tokens": 2 },
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let failure = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-model")
        .materialize_with_attempts::<Position>("reconcile")
        .await
        .unwrap_err();

    assert_usage_bearing_protocol_failure(failure, 52);
    request.assert_async().await;
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_attempt_report_parses_usage() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/models/test-model:generateContent")
        .match_query(mockito::Matcher::UrlEncoded(
            "key".to_string(),
            "test-key".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "candidates": [{
                    "content": { "parts": [{ "text": POSITION_JSON }] },
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 60,
                    "candidatesTokenCount": 12,
                },
                "modelVersion": "gemini-risk-router-2026-07-15",
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let report = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-model")
        .materialize_with_attempts::<Position>("reconcile the futures position")
        .await
        .unwrap();

    assert_single_success(report, "gemini-risk-router-2026-07-15", 72);
    request.assert_async().await;
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_protocol_failure_keeps_reported_usage() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/models/test-model:generateContent")
        .match_query(mockito::Matcher::UrlEncoded(
            "key".to_string(),
            "test-key".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "candidates": [],
                "usageMetadata": {
                    "promptTokenCount": 45,
                    "candidatesTokenCount": 3,
                },
                "modelVersion": "gemini-risk-router",
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let failure = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-model")
        .materialize_with_attempts::<Position>("reconcile")
        .await
        .unwrap_err();

    assert_usage_bearing_protocol_failure(failure, 48);
    request.assert_async().await;
}

#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_attempt_report_parses_usage() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "choices": [{
                    "message": { "role": "assistant", "content": POSITION_JSON },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 65,
                    "completion_tokens": 13,
                    "total_tokens": 78,
                },
                "model": "grok-risk-router-2026-07-15",
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let report = rstructor::GrokClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-model")
        .materialize_with_attempts::<Position>("reconcile the futures position")
        .await
        .unwrap();

    assert_single_success(report, "grok-risk-router-2026-07-15", 78);
    request.assert_async().await;
}

#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_protocol_failure_keeps_reported_usage() {
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "choices": [],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 4,
                    "total_tokens": 44,
                },
                "model": "grok-risk-router",
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let failure = rstructor::GrokClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-model")
        .materialize_with_attempts::<Position>("reconcile")
        .await
        .unwrap_err();

    assert_usage_bearing_protocol_failure(failure, 44);
    request.assert_async().await;
}

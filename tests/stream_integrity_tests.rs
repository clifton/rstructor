//! Offline HTTP tests for strict streaming integrity.
//!
//! The fixtures use sanitized OpenAI Chat Completions event shapes captured from
//! realistic portfolio workflows. They drive the real HTTP/SSE/provider pipeline,
//! which `MockClient` intentionally bypasses.

#![cfg(all(feature = "openai", feature = "streaming"))]

use futures_util::StreamExt;
use rstructor::{
    Instructor, LLMClient, OpenAIClient, RStructorError, StreamErrorKind, StreamedObject,
};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
struct Position {
    symbol: String,
    quantity: i64,
}

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
struct PortfolioSummary {
    portfolio_id: String,
    gross_exposure: f64,
}

fn client(server: &mockito::Server) -> OpenAIClient {
    OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-5-mini")
}

async fn serve_fixture(body: &str) -> (mockito::ServerGuard, mockito::Mock) {
    serve_fixture_at("/chat/completions", body).await
}

async fn serve_fixture_at(path: &str, body: &str) -> (mockito::ServerGuard, mockito::Mock) {
    let mut server = mockito::Server::new_async().await;
    let response = server
        .mock("POST", path)
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(body)
        .expect(1)
        .create_async()
        .await;
    (server, response)
}

async fn serve_gemini_fixture(body: &str) -> (mockito::ServerGuard, mockito::Mock) {
    let mut server = mockito::Server::new_async().await;
    let response = server
        .mock("POST", "/models/test-model:streamGenerateContent")
        .match_query(mockito::Matcher::AllOf(vec![
            mockito::Matcher::UrlEncoded("alt".to_string(), "sse".to_string()),
            mockito::Matcher::UrlEncoded("key".to_string(), "test-key".to_string()),
        ]))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(body)
        .expect(1)
        .create_async()
        .await;
    (server, response)
}

fn assert_stream_error(error: RStructorError, expected: StreamErrorKind) {
    match error {
        RStructorError::StreamingError { kind, message } => {
            assert_eq!(kind, expected);
            assert!(!message.is_empty());
        }
        other => panic!("expected StreamingError, got {other:?}"),
    }
}

#[tokio::test]
async fn malformed_sse_json_is_terminal_after_prior_text_deltas() {
    let fixture = include_str!("fixtures/streaming/openai_text_malformed_event.sse");
    let (server, response) = serve_fixture(fixture).await;
    let client = client(&server);
    let mut stream = client.generate_stream("summarize gross exposure");

    assert_eq!(stream.next().await.unwrap().unwrap(), "Gross exposure: ");
    assert_eq!(stream.next().await.unwrap().unwrap(), "1.42x");
    assert_stream_error(
        stream.next().await.unwrap().unwrap_err(),
        StreamErrorKind::InvalidEventJson,
    );
    assert!(stream.next().await.is_none(), "nothing may follow an error");
    response.assert_async().await;
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_requires_message_stop_after_content() {
    let fixture = include_str!("fixtures/streaming/anthropic_text_missing_stop.sse");
    let (server, response) = serve_fixture_at("/messages", fixture).await;
    let client = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("claude-sonnet-4-5-20250929");
    let mut stream = client.generate_stream("summarize net delta");

    assert_eq!(stream.next().await.unwrap().unwrap(), "Net delta: $18.4mm");
    assert_stream_error(
        stream.next().await.unwrap().unwrap_err(),
        StreamErrorKind::IncompleteEventStream,
    );
    assert!(stream.next().await.is_none());
    response.assert_async().await;
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_message_stop_completes_cleanly() {
    let fixture = include_str!("fixtures/streaming/anthropic_text_complete.sse");
    let (server, response) = serve_fixture_at("/messages", fixture).await;
    let client = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("claude-sonnet-4-5-20250929");
    let chunks = client
        .generate_stream("summarize net delta")
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        chunks
            .into_iter()
            .collect::<rstructor::Result<Vec<_>>>()
            .unwrap(),
        vec!["Net delta: $18.4mm"]
    );
    response.assert_async().await;
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_in_band_error_is_terminal_and_typed() {
    let fixture = include_str!("fixtures/streaming/anthropic_in_band_error.sse");
    let (server, response) = serve_fixture_at("/messages", fixture).await;
    let client = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("claude-sonnet-4-5-20250929");
    let mut stream = client.generate_stream("summarize net delta");

    assert_stream_error(
        stream.next().await.unwrap().unwrap_err(),
        StreamErrorKind::ProviderStreamError,
    );
    assert!(stream.next().await.is_none());
    response.assert_async().await;
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_final_text_and_finish_reason_complete_together() {
    let fixture = include_str!("fixtures/streaming/gemini_text_complete.sse");
    let (server, response) = serve_gemini_fixture(fixture).await;
    let client = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-model");
    let chunks = client
        .generate_stream("summarize value at risk")
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        chunks
            .into_iter()
            .collect::<rstructor::Result<Vec<_>>>()
            .unwrap(),
        vec!["VaR: $2.1mm"]
    );
    response.assert_async().await;
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_finish_only_content_may_omit_parts() {
    let fixture = include_str!("fixtures/streaming/gemini_finish_only_content.sse");
    let (server, response) = serve_gemini_fixture(fixture).await;
    let client = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-model");
    let chunks = client
        .generate_stream("summarize value at risk")
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        chunks
            .into_iter()
            .collect::<rstructor::Result<Vec<_>>>()
            .unwrap(),
        vec!["VaR: $2.1mm"]
    );
    response.assert_async().await;
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_requires_a_nonempty_finish_reason() {
    let fixture = include_str!("fixtures/streaming/gemini_text_missing_finish.sse");
    let (server, response) = serve_gemini_fixture(fixture).await;
    let client = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-model");
    let mut stream = client.generate_stream("summarize value at risk");

    assert_eq!(stream.next().await.unwrap().unwrap(), "VaR: $2.1mm");
    assert_stream_error(
        stream.next().await.unwrap().unwrap_err(),
        StreamErrorKind::IncompleteEventStream,
    );
    assert!(stream.next().await.is_none());
    response.assert_async().await;
}

#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_requires_done_after_its_finish_reason_chunk() {
    let fixture = include_str!("fixtures/streaming/grok_text_missing_done.sse");
    let (server, response) = serve_fixture(fixture).await;
    let client = rstructor::GrokClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("grok-4-1-fast-reasoning");
    let mut stream = client.generate_stream("estimate market beta");

    assert_eq!(stream.next().await.unwrap().unwrap(), "Beta: 0.84");
    assert_stream_error(
        stream.next().await.unwrap().unwrap_err(),
        StreamErrorKind::IncompleteEventStream,
    );
    assert!(stream.next().await.is_none());
    response.assert_async().await;
}

#[tokio::test]
async fn object_stream_never_completes_after_a_malformed_event() {
    let fixture = include_str!("fixtures/streaming/openai_object_valid_then_malformed.sse");
    let (server, response) = serve_fixture(fixture).await;
    let client = client(&server);
    let mut stream = client.materialize_stream::<PortfolioSummary>("summarize the portfolio");

    let first = stream.next().await.unwrap().unwrap();
    assert!(matches!(first, StreamedObject::Partial(_)));
    assert_stream_error(
        stream.next().await.unwrap().unwrap_err(),
        StreamErrorKind::InvalidEventJson,
    );
    assert!(
        stream.next().await.is_none(),
        "Complete must not follow an integrity error"
    );
    response.assert_async().await;
}

#[tokio::test]
async fn invalid_array_element_is_not_dropped_or_renumbered() {
    let fixture = include_str!("fixtures/streaming/openai_items_invalid_element.sse");
    let (server, response) = serve_fixture(fixture).await;
    let client = client(&server);
    let mut stream = client.materialize_iter::<Position>("reconcile positions");

    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        Position {
            symbol: "AAPL".to_string(),
            quantity: 125_000,
        }
    );
    assert_stream_error(
        stream.next().await.unwrap().unwrap_err(),
        StreamErrorKind::InvalidArrayElement { index: 1 },
    );
    assert!(stream.next().await.is_none(), "nothing may follow an error");
    response.assert_async().await;
}

#[tokio::test]
async fn done_sentinel_does_not_hide_a_truncated_array_tail() {
    let fixture = include_str!("fixtures/streaming/openai_items_truncated.sse");
    let (server, response) = serve_fixture(fixture).await;
    let client = client(&server);
    let mut stream = client.materialize_iter::<Position>("reconcile positions");

    assert_eq!(stream.next().await.unwrap().unwrap().symbol, "AAPL");
    assert_stream_error(
        stream.next().await.unwrap().unwrap_err(),
        StreamErrorKind::IncompleteArray { next_index: 1 },
    );
    assert!(stream.next().await.is_none(), "nothing may follow an error");
    response.assert_async().await;
}

#[tokio::test]
async fn valid_json_at_eof_is_not_authoritative_without_done() {
    let fixture = include_str!("fixtures/streaming/openai_items_complete_without_done.sse");
    let (server, response) = serve_fixture(fixture).await;
    let client = client(&server);
    let mut stream = client.materialize_iter::<Position>("reconcile positions");

    assert_eq!(stream.next().await.unwrap().unwrap().symbol, "AAPL");
    assert_eq!(stream.next().await.unwrap().unwrap().symbol, "ESU6");
    assert_stream_error(
        stream.next().await.unwrap().unwrap_err(),
        StreamErrorKind::IncompleteEventStream,
    );
    assert!(stream.next().await.is_none(), "nothing may follow an error");
    response.assert_async().await;
}

#[tokio::test]
async fn complete_array_and_terminal_marker_end_cleanly() {
    let fixture = include_str!("fixtures/streaming/openai_items_complete.sse");
    let (server, response) = serve_fixture(fixture).await;
    let client = client(&server);
    let results = client
        .materialize_iter::<Position>("reconcile positions")
        .collect::<Vec<_>>()
        .await;

    let positions = results
        .into_iter()
        .collect::<rstructor::Result<Vec<_>>>()
        .unwrap();
    assert_eq!(
        positions,
        vec![
            Position {
                symbol: "AAPL".to_string(),
                quantity: 125_000,
            },
            Position {
                symbol: "ESU6".to_string(),
                quantity: -240,
            },
        ]
    );
    response.assert_async().await;
}

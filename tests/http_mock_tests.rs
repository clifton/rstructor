//! Drive the **real** provider client over a local mock HTTP server (`mockito`),
//! exercising request building, response parsing, and the retry/re-ask loop in
//! `utils.rs` — the code paths `MockClient` (which mocks at the `LLMClient` trait
//! level, above HTTP) structurally cannot reach. No API key or network needed.
//!
//! Targets `OpenAIClient` because it exposes `base_url`, but the request/response
//! shaping and the retry loop it drives are shared by the OpenAI-compatible path.
#![cfg(feature = "openai")]

use rstructor::{ApiErrorKind, Instructor, LLMClient, OpenAIClient, RStructorError};
use serde::{Deserialize, Serialize};
use serde_json::json;

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

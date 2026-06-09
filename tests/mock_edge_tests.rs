//! Edge-case offline tests for [`MockClient`] (companion to `mock_client_tests.rs`).
//!
//! These exercise the response-resolution precedence (responder → queue →
//! default), the retry/scripted-error semantics, error cloning/serialization
//! branches, and the streaming/tools/builder routing — all without touching the
//! network or needing an API key. Every test gates on exactly the features it
//! needs; the bare-`mock` tests run even in a schema-only build.

#![cfg(feature = "mock")]

use rstructor::{Instructor, LLMClient, MockClient, MockResponse, RStructorError, RequestKind};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[llm(validate = "validate_movie")]
struct Movie {
    title: String,
    year: u16,
}

fn validate_movie(m: &Movie) -> rstructor::Result<()> {
    if m.year < 1888 {
        return Err(RStructorError::ValidationError(format!(
            "year {} predates cinema",
            m.year
        )));
    }
    Ok(())
}

const GOOD: &str = r#"{"title":"Good","year":2000}"#;
const BAD: &str = r#"{"title":"Bad","year":1000}"#; // fails validate_movie

// ---------------------------------------------------------------------------
// Response-resolution precedence: responder → queue → default
// ---------------------------------------------------------------------------

/// A responder that returns `None` falls through to the queue; once the queue is
/// drained, the default response is used (here, the built-in `Unsupported` error).
#[tokio::test]
async fn responder_none_falls_through_to_queue_then_default() {
    let client = MockClient::new()
        .with_responder(|_view| None) // always defers
        .with_response("Q");

    // First call: responder declines, queue serves "Q".
    assert_eq!(client.generate("p").await.unwrap(), "Q");
    assert!(client.responses_exhausted());

    // Second call: queue empty, responder still declines → default error.
    let err = client.generate("p").await.unwrap_err();
    assert!(matches!(err, RStructorError::Unsupported(_)));
}

/// A responder returning an `Error` is surfaced verbatim and is NOT retried, so a
/// good payload queued behind it is never consumed.
#[tokio::test]
async fn responder_error_is_surfaced_and_not_retried() {
    let client = MockClient::new()
        .with_responder(|_view| Some(MockResponse::error(RStructorError::Timeout)))
        .with_response(GOOD)
        .with_retries(5); // retries must NOT rescue a scripted responder error

    let err = client.materialize::<Movie>("p").await.unwrap_err();
    assert_eq!(err, RStructorError::Timeout);
    // The good response queued behind the responder was never popped.
    assert!(
        !client.responses_exhausted(),
        "the queued good response must remain unconsumed"
    );
}

// ---------------------------------------------------------------------------
// Retry semantics on materialize
// ---------------------------------------------------------------------------

/// `with_retries(0)` means exactly one attempt: the bad payload fails and the
/// good payload queued behind it is left untouched.
#[tokio::test]
async fn with_retries_zero_is_single_attempt() {
    let client = MockClient::new()
        .with_response(BAD)
        .with_response(GOOD)
        .with_retries(0);

    let err = client.materialize::<Movie>("p").await.unwrap_err();
    assert!(matches!(err, RStructorError::ValidationError(_)));
    // Only one response (the bad one) was consumed; the good one is still queued.
    assert!(
        !client.responses_exhausted(),
        "the good response must not be consumed on a single attempt"
    );
    // Confirm the survivor is the good one.
    let movie: Movie = client.materialize("p").await.unwrap();
    assert_eq!(movie.title, "Good");
}

/// With retries enabled, a scripted `Error` sitting in the queue stops the retry
/// loop immediately (returned verbatim), leaving anything behind it untouched.
#[tokio::test]
async fn with_retries_stops_at_scripted_error_in_queue() {
    let client = MockClient::new()
        .with_response(BAD) // attempt 1: validation failure → retry
        .with_error(RStructorError::Timeout) // attempt 2: scripted error → stop
        .with_response(GOOD) // never reached
        .with_retries(5);

    let err = client.materialize::<Movie>("p").await.unwrap_err();
    assert_eq!(err, RStructorError::Timeout);
    // The good response after the scripted error is still queued.
    assert!(
        !client.responses_exhausted(),
        "the good response after the scripted error must remain queued"
    );
    let movie: Movie = client.materialize("p").await.unwrap();
    assert_eq!(movie.title, "Good");
}

// ---------------------------------------------------------------------------
// Recording / clear_requests preserves queue + responder
// ---------------------------------------------------------------------------

/// `clear_requests` only wipes the recording log — the responder (and queue)
/// remain active for subsequent calls.
#[tokio::test]
async fn clear_requests_preserves_responder() {
    let client = MockClient::new().with_responder(|view| {
        view.prompt
            .contains("haiku")
            .then(|| MockResponse::text("a haiku"))
    });
    assert_eq!(client.generate("write a haiku").await.unwrap(), "a haiku");
    assert_eq!(client.request_count(), 1);

    client.clear_requests();
    assert_eq!(client.request_count(), 0);

    // The responder still fires after the log was cleared.
    assert_eq!(client.generate("another haiku").await.unwrap(), "a haiku");
    assert_eq!(client.request_count(), 1);
}

// ---------------------------------------------------------------------------
// Shared-clone queue scripting is FIFO across push_response / push_error
// ---------------------------------------------------------------------------

/// `push_response` / `push_error` on a shared clone enqueue FIFO into the same
/// state the original draws from.
#[tokio::test]
async fn push_response_and_push_error_are_fifo_on_shared_clone() {
    let client = MockClient::new().with_response(GOOD);
    let clone = client.clone();
    // Enqueue an error behind the good response via the clone.
    clone.push_error(RStructorError::Timeout);

    // First out: the good response queued at construction.
    let movie: Movie = client.materialize("p").await.unwrap();
    assert_eq!(movie.title, "Good");
    // Second out: the error pushed on the clone (FIFO order preserved).
    let err = client.materialize::<Movie>("p").await.unwrap_err();
    assert_eq!(err, RStructorError::Timeout);
    assert!(client.responses_exhausted());
}

// ---------------------------------------------------------------------------
// Error cloning: non-Clone variants are stringified (downgraded) on reuse
// ---------------------------------------------------------------------------

/// `RStructorError::JsonError` is not `Clone`; when a default response holding one
/// is reused, `clone_error` downgrades it to a `SerializationError` carrying the
/// stringified message.
#[tokio::test]
async fn default_json_error_is_downgraded_to_serialization_error_on_reuse() {
    // Build a real serde_json::Error to wrap in a non-Clone JsonError variant.
    let json_err: serde_json::Error = serde_json::from_str::<i32>("not a number").unwrap_err();
    let original = RStructorError::JsonError(json_err);
    let original_message = original.to_string();

    let client = MockClient::new().with_default_response(MockResponse::error(original));

    // The default is cloned on each use; the clone is the downgraded variant.
    let err = client.generate("p").await.unwrap_err();
    match err {
        RStructorError::SerializationError(msg) => assert_eq!(msg, original_message),
        other => panic!("expected SerializationError downgrade, got {other:?}"),
    }
}

/// `MockResponse::json` returns a `SerializationError` when the value cannot be
/// serialized to JSON (a map with non-string keys is invalid JSON).
#[test]
fn mock_response_json_serialization_error_branch() {
    use std::collections::HashMap;
    let mut bad: HashMap<Vec<u8>, u8> = HashMap::new();
    bad.insert(vec![1, 2, 3], 9);
    let err = MockResponse::json(&bad).unwrap_err();
    assert!(matches!(err, RStructorError::SerializationError(_)));
}

// ---------------------------------------------------------------------------
// generate_with_metadata: usage attachment + error arm
// ---------------------------------------------------------------------------

/// `generate_with_metadata` carries any configured usage on success.
#[tokio::test]
async fn generate_with_metadata_carries_usage() {
    use rstructor::TokenUsage;
    let client = MockClient::new()
        .with_response("hello")
        .with_usage(TokenUsage::new("mock-model", 3, 5));
    let result = client.generate_with_metadata("p").await.unwrap();
    assert_eq!(result.text, "hello");
    let usage = result.usage.unwrap();
    assert_eq!(usage.input_tokens, 3);
    assert_eq!(usage.total_tokens(), 8);
    assert_eq!(
        client.last_request().unwrap().kind,
        RequestKind::GenerateWithMetadata
    );
}

/// `generate_with_metadata` surfaces a scripted error from its error arm.
#[tokio::test]
async fn generate_with_metadata_surfaces_error_arm() {
    let client = MockClient::new().with_error(RStructorError::Timeout);
    let err = client.generate_with_metadata("p").await.unwrap_err();
    assert_eq!(err, RStructorError::Timeout);
}

// ---------------------------------------------------------------------------
// Streaming (requires `streaming`, which implies `_client`)
// ---------------------------------------------------------------------------

#[cfg(feature = "streaming")]
mod streaming {
    use super::*;
    use futures_util::StreamExt;

    /// `materialize_iter` validates each element: a good first item is yielded,
    /// then a per-item validation failure surfaces as a `ValidationError`.
    #[tokio::test]
    async fn materialize_iter_per_item_validation_failure_mid_stream() {
        let client = MockClient::new()
            .with_response(r#"[{"title":"Good","year":2000},{"title":"Bad","year":1000}]"#);
        let mut stream = client.materialize_iter::<Movie>("p");

        let first = stream.next().await.expect("first item present");
        assert_eq!(first.unwrap().title, "Good");

        let second = stream.next().await.expect("second item present");
        assert!(matches!(second, Err(RStructorError::ValidationError(_))));
    }

    /// `materialize_iter` on a non-array, non-`items` shape yields a
    /// `ValidationError` explaining it expects a JSON array.
    #[tokio::test]
    async fn materialize_iter_bad_shape_is_validation_error() {
        let client = MockClient::new().with_response(r#"{"foo":1}"#);
        let mut stream = client.materialize_iter::<Movie>("p");
        let item = stream.next().await.expect("an error item is yielded");
        match item {
            Err(RStructorError::ValidationError(msg)) => {
                assert!(
                    msg.contains("JSON array"),
                    "expected a 'JSON array' shape message, got: {msg}"
                );
            }
            other => panic!("expected ValidationError, got {other:?}"),
        }
    }

    /// `materialize_iter` on malformed JSON yields a `ValidationError` reporting a
    /// parse failure.
    #[tokio::test]
    async fn materialize_iter_malformed_json_is_validation_error() {
        let client = MockClient::new().with_response("not json");
        let mut stream = client.materialize_iter::<Movie>("p");
        let item = stream.next().await.expect("an error item is yielded");
        match item {
            Err(RStructorError::ValidationError(msg)) => {
                assert!(
                    msg.contains("Failed to parse response as JSON"),
                    "expected a parse-failure message, got: {msg}"
                );
            }
            other => panic!("expected ValidationError, got {other:?}"),
        }
    }

    /// A scripted error makes the first streamed item an `Err` for both the text
    /// and object streams.
    #[tokio::test]
    async fn scripted_error_yields_err_as_first_stream_item() {
        let client = MockClient::new().with_error(RStructorError::Timeout);
        let mut text_stream = client.generate_stream("p");
        let first_text = text_stream.next().await.expect("first item present");
        assert_eq!(first_text.unwrap_err(), RStructorError::Timeout);

        // Re-queue for the object stream (the previous error was consumed).
        let client = MockClient::new().with_error(RStructorError::Timeout);
        let mut obj_stream = client.materialize_stream::<Movie>("p");
        let first_obj = obj_stream.next().await.expect("first item present");
        assert_eq!(first_obj.unwrap_err(), RStructorError::Timeout);
    }
}

// ---------------------------------------------------------------------------
// Fluent builder routing (requires `_client`; `system`/`media` live there)
// ---------------------------------------------------------------------------

#[cfg(feature = "_client")]
mod builder {
    use super::*;
    use rstructor::{MediaFile, RequestExt};

    /// `with_system(..).media(..).materialize(..)` routes through
    /// `materialize_with_media`, prepending the system context to the prompt and
    /// carrying the attached media.
    #[tokio::test]
    async fn with_system_and_media_routes_to_materialize_with_media() {
        let client = MockClient::new().with_response(GOOD);
        let _: Movie = client
            .with_system("CTX")
            .media(vec![MediaFile::new("u", "image/png")])
            .materialize("describe")
            .await
            .unwrap();
        let req = client.last_request().unwrap();
        assert_eq!(req.kind, RequestKind::MaterializeWithMedia);
        assert_eq!(req.prompt, "CTX\n\ndescribe");
        assert_eq!(req.media.len(), 1);
        assert_eq!(req.media[0].mime_type, "image/png");
    }

    /// Calling `system(..)` twice keeps the last value when building the combined
    /// prompt.
    #[tokio::test]
    async fn system_twice_is_last_wins() {
        let client = MockClient::new().with_response(GOOD);
        let _: Movie = client
            .with_system("A")
            .system("B")
            .materialize("hi")
            .await
            .unwrap();
        let req = client.last_request().unwrap();
        assert_eq!(req.kind, RequestKind::Materialize);
        assert_eq!(req.prompt, "B\n\nhi");
    }

    /// `with_media(..).generate(..)` routes through `generate_with_media`,
    /// carrying the attached media instead of silently dropping it.
    #[tokio::test]
    async fn with_media_generate_routes_to_generate_with_media() {
        let client = MockClient::new().with_response("a caption");
        let media = [MediaFile::new("u", "image/png")];
        let out = client
            .with_media(&media)
            .generate("describe")
            .await
            .unwrap();
        assert_eq!(out, "a caption");
        let req = client.last_request().unwrap();
        assert_eq!(req.kind, RequestKind::GenerateWithMedia);
        assert_eq!(req.prompt, "describe");
        assert_eq!(req.media.len(), 1);
        assert_eq!(req.media[0].mime_type, "image/png");
    }
}

// ---------------------------------------------------------------------------
// Tool loop fallback (requires `tools`; `run` lives there)
// ---------------------------------------------------------------------------

#[cfg(feature = "tools")]
mod tools {
    use super::*;
    use rstructor::RequestExt;

    /// `run` with NO tools attached falls back to `generate`, recording a
    /// `Generate` request with the system-prepended prompt.
    #[tokio::test]
    async fn run_with_no_tools_falls_back_to_generate() {
        let client = MockClient::new().with_response("answer");
        let out = client.with_system("CTX").run("hi").await.unwrap();
        assert_eq!(out, "answer");
        let req = client.last_request().unwrap();
        assert_eq!(req.kind, RequestKind::Generate);
        assert_eq!(req.prompt, "CTX\n\nhi");
        assert!(
            req.tool_names.is_empty(),
            "no tools were attached, so no tool loop should have run"
        );
    }

    /// `run` with NO tools but WITH media falls back to `generate_with_media`,
    /// carrying the attached media instead of silently dropping it.
    #[tokio::test]
    async fn run_with_no_tools_and_media_falls_back_to_generate_with_media() {
        let client = MockClient::new().with_response("answer");
        let media = [rstructor::MediaFile::new("u", "image/png")];
        let out = client.with_media(&media).run("hi").await.unwrap();
        assert_eq!(out, "answer");
        let req = client.last_request().unwrap();
        assert_eq!(req.kind, RequestKind::GenerateWithMedia);
        assert_eq!(req.media.len(), 1);
    }

    /// `run` WITH tools forwards attached media into the tool loop's request.
    #[tokio::test]
    async fn run_with_tools_carries_media_into_tool_loop() {
        use rstructor::{FnTool, Instructor, Toolbox};
        use serde::{Deserialize, Serialize};

        #[derive(Instructor, Serialize, Deserialize)]
        struct EchoArgs {
            value: String,
        }

        let toolbox = Toolbox::new().with(FnTool::new("echo", "Echo", |args: EchoArgs| {
            std::future::ready(Ok(serde_json::json!(args.value)))
        }));
        let client = MockClient::new().with_response("done");
        let media = [rstructor::MediaFile::new("u", "image/png")];
        let out = client
            .with_tools(&toolbox)
            .media(media.to_vec())
            .run("hi")
            .await
            .unwrap();
        assert_eq!(out, "done");
        let req = client.last_request().unwrap();
        assert_eq!(req.kind, RequestKind::RunToolLoop);
        assert_eq!(req.media.len(), 1, "media must reach the tool loop");
        assert_eq!(req.tool_names, vec!["echo"]);
    }
}

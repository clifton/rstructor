//! Key-free smoke tests for the canonical end-to-end flows, driven by `MockClient`.
//! These mirror the live integration tests (which require API keys) and double as a
//! template for how downstream code unit-tests its own LLM pipelines offline.
#![cfg(feature = "mock")]

use rstructor::{ApiErrorKind, Instructor, LLMClient, MockClient, RStructorError};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[llm(validate = "validate_ticket")]
struct Ticket {
    title: String,
    priority: u8,
}

fn validate_ticket(t: &Ticket) -> rstructor::Result<()> {
    if !(1..=5).contains(&t.priority) {
        return Err(RStructorError::ValidationError(format!(
            "priority must be 1-5, got {}",
            t.priority
        )));
    }
    Ok(())
}

/// A representative "pipeline" function generic over the client — exactly the kind
/// of code a downstream app writes and wants to test without a network.
async fn triage<C: LLMClient + Sync>(client: &C, email: &str) -> rstructor::Result<Ticket> {
    client.materialize(email).await
}

#[tokio::test]
async fn pipeline_happy_path() {
    let client = MockClient::new().with_response(r#"{"title":"Login down","priority":4}"#);
    let ticket = triage(&client, "the login page is broken").await.unwrap();
    assert_eq!(
        ticket,
        Ticket {
            title: "Login down".into(),
            priority: 4
        }
    );
}

#[tokio::test]
async fn pipeline_rejects_invalid_then_recovers_on_reask() {
    let client = MockClient::new()
        .with_response(r#"{"title":"x","priority":99}"#) // fails validation
        .with_response(r#"{"title":"x","priority":2}"#) // corrected on re-ask
        .with_retries(1);
    let ticket = triage(&client, "anything").await.unwrap();
    assert_eq!(ticket.priority, 2);
}

#[tokio::test]
async fn pipeline_surfaces_api_errors() {
    let client = MockClient::new().with_error(RStructorError::api_error(
        "OpenAI",
        ApiErrorKind::AuthenticationFailed,
    ));
    let err = triage(&client, "anything").await.unwrap_err();
    assert!(matches!(
        err.api_error_kind(),
        Some(ApiErrorKind::AuthenticationFailed)
    ));
}

#[cfg(feature = "streaming")]
#[tokio::test]
async fn pipeline_streams_a_list() {
    use futures_util::StreamExt;
    let client = MockClient::new()
        .with_response(r#"{"items":[{"title":"a","priority":1},{"title":"b","priority":2}]}"#);
    let mut stream = client.materialize_iter::<Ticket>("list the open tickets");
    let mut count = 0;
    while let Some(item) = stream.next().await {
        item.unwrap();
        count += 1;
    }
    assert_eq!(count, 2);
}

#[cfg(feature = "tools")]
#[tokio::test]
async fn pipeline_runs_a_tool_loop() {
    use rstructor::{FnTool, RequestExt, Toolbox};
    use serde_json::json;

    #[derive(Instructor, Serialize, Deserialize)]
    struct LookupArgs {
        #[allow(dead_code)]
        id: String,
    }

    let toolbox = Toolbox::new().with(FnTool::new(
        "lookup_order",
        "Look up an order by id",
        |_a: LookupArgs| async move { Ok(json!({ "status": "shipped" })) },
    ));
    let client = MockClient::new().with_response("resolved");
    let answer = client
        .with_tools(&toolbox)
        .run("triage this")
        .await
        .unwrap();
    assert_eq!(answer, "resolved");
}

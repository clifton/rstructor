//! Offline testing with [`MockClient`] — no API key, no network, fully deterministic.
//!
//! Run with: `cargo run --example mock_testing_example --features mock`
//!
//! The point of `MockClient` is that scripted responses flow through the *real*
//! deserialize + `validate()` path, so your tests exercise schema/validation
//! behavior exactly as a live provider would — just instantly and offline.

use rstructor::{Instructor, LLMClient, MockClient, RStructorError};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(validate = "validate_ticket")]
struct Ticket {
    #[llm(description = "Short, imperative summary of the issue")]
    title: String,
    #[llm(description = "Priority from 1 (low) to 5 (urgent)")]
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

/// The code under test: generic over any `LLMClient`, so the mock drops straight in
/// wherever a real `OpenAIClient`/`AnthropicClient`/… would go.
async fn triage<C: LLMClient + Sync>(client: &C, message: &str) -> rstructor::Result<Ticket> {
    client.materialize(message).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Happy path — a scripted JSON payload is deserialized and validated.
    let client = MockClient::new().with_response(r#"{"title": "Login page down", "priority": 4}"#);
    let ticket = triage(&client, "the login page is throwing 500s").await?;
    println!("extracted: {ticket:?}");
    assert_eq!(ticket.priority, 4);
    assert_eq!(client.request_count(), 1);
    assert_eq!(
        client.last_request().unwrap().schema_name.as_deref(),
        Some("Ticket")
    );

    // Failure path — invalid data is rejected by the *real* validator, offline.
    let client = MockClient::new().with_response(r#"{"title": "x", "priority": 99}"#);
    match triage(&client, "anything").await {
        Err(e) => println!("correctly rejected invalid data: {e}"),
        Ok(_) => unreachable!("priority 99 should fail validation"),
    }

    // Simulate the provider re-ask loop: first attempt bad, second good.
    let client = MockClient::new()
        .with_response(r#"{"title": "Retry me", "priority": 9}"#) // fails validation
        .with_response(r#"{"title": "Retry me", "priority": 2}"#) // succeeds
        .with_retries(1);
    let ticket = triage(&client, "flaky model").await?;
    println!("recovered after re-ask: {ticket:?}");

    println!("\nAll offline assertions passed — no network, no API key.");
    Ok(())
}

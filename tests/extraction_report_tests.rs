//! End-to-end coverage for the uniform extraction report using sanitized
//! real-world portfolio fixtures.

#![cfg(feature = "mock")]

use rstructor::{
    AttemptOutcome, Instructor, LLMClient, MockClient, RStructorError, RetryDisposition, TokenUsage,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Instructor, PartialEq, Serialize)]
struct Portfolio {
    portfolio_id: String,
    positions: Vec<Position>,
}

#[derive(Debug, Deserialize, Instructor, PartialEq, Serialize)]
struct Position {
    symbol: String,
    quantity: i64,
}

#[tokio::test]
async fn successful_reask_returns_value_and_shared_report() {
    let client = MockClient::new()
        .with_response_and_usage(
            include_str!("fixtures/structured/portfolio_invalid_quantity.json"),
            TokenUsage::new("mock-risk-router-v1", 90, 15),
        )
        .with_response_and_usage(
            include_str!("fixtures/structured/portfolio_valid.json"),
            TokenUsage::new("mock-risk-router-v2", 120, 18),
        )
        .with_retries(1);

    let extraction = client
        .extract_with_report::<Portfolio>("reconcile positions")
        .await
        .unwrap();

    assert_eq!(extraction.data.portfolio_id, "HF-ALPHA-001");
    assert_eq!(extraction.data.positions[1].quantity, -240);
    assert_eq!(extraction.report.attempts.len(), 2);
    assert!(matches!(
        extraction.report.attempts[0].outcome,
        AttemptOutcome::Failed {
            disposition: RetryDisposition::Retried,
            ..
        }
    ));
    assert_eq!(
        extraction.report.attempts[1].outcome,
        AttemptOutcome::Succeeded
    );
    assert_eq!(
        extraction
            .report
            .cumulative_usage
            .as_ref()
            .unwrap()
            .total_tokens(),
        243
    );
    assert_eq!(
        extraction.report.final_usage.as_ref().unwrap().model,
        "mock-risk-router-v2"
    );
    assert!(extraction.report.attempts_complete);
}

#[tokio::test]
async fn exhausted_reask_returns_the_same_report_shape_with_the_error() {
    let invalid = include_str!("fixtures/structured/portfolio_invalid_quantity.json");
    let client = MockClient::new()
        .with_response_and_usage(invalid, TokenUsage::new("mock-risk-router-v1", 80, 10))
        .with_response_and_usage(invalid, TokenUsage::new("mock-risk-router-v2", 100, 12))
        .with_retries(1);

    let failure = client
        .extract_with_report::<Portfolio>("reconcile positions")
        .await
        .unwrap_err();

    assert!(matches!(
        failure.error(),
        RStructorError::OutputDecodeError { path, .. }
            if path == "$.positions[1].quantity"
    ));
    assert_eq!(failure.report.attempts.len(), 2);
    assert!(matches!(
        failure.report.attempts[1].outcome,
        AttemptOutcome::Failed {
            disposition: RetryDisposition::BudgetExhausted,
            ..
        }
    ));
    assert_eq!(
        failure
            .report
            .cumulative_usage
            .as_ref()
            .unwrap()
            .total_tokens(),
        202
    );
    assert_eq!(
        failure.report.final_usage.as_ref().unwrap().model,
        "mock-risk-router-v2"
    );
    assert!(failure.report.attempts_complete);
}

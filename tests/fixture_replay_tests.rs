#![cfg(all(feature = "derive", feature = "mock"))]

use rstructor::{Fixture, Instructor, LLMClient};
use serde::{Deserialize, Serialize};

const APPLE_2023_PROMPT: &str = "From Apple Inc.'s 2023 Form 10-K: Total net sales were $383,285 million for the fiscal year ended September 30, 2023. Extract the issuer, fiscal year, and net sales in USD millions.";
const APPLE_2023_FIXTURE: &str = include_str!("fixtures/record_replay/apple_2023_10k.fixture.json");

#[derive(Debug, PartialEq, Instructor, Serialize, Deserialize)]
struct FilingMetric {
    issuer: String,
    fiscal_year: u16,
    net_sales_usd_millions: u64,
}

#[tokio::test]
async fn replays_real_world_filing_metric_with_usage_and_attempts() {
    let fixture = Fixture::from_json(APPLE_2023_FIXTURE).unwrap();
    assert_eq!(fixture.to_json().unwrap(), APPLE_2023_FIXTURE);

    let replay = fixture.replay();
    let extraction = replay
        .extract_with_report::<FilingMetric>(APPLE_2023_PROMPT)
        .await
        .unwrap();

    assert_eq!(
        extraction.data,
        FilingMetric {
            issuer: "Apple Inc.".to_string(),
            fiscal_year: 2023,
            net_sales_usd_millions: 383_285,
        }
    );
    assert_eq!(extraction.report.final_usage.unwrap().total_tokens(), 91);
    assert_eq!(extraction.report.attempts.len(), 1);
    assert!(extraction.report.attempts_complete);
    replay.assert_finished().unwrap();
}

#[tokio::test]
async fn fixture_replay_rejects_schema_drift_without_consuming_the_exchange() {
    #[derive(Debug, Instructor, Serialize, Deserialize)]
    struct DriftedMetric {
        issuer: String,
        fiscal_year: u16,
        net_sales_usd_millions: u64,
        operating_income_usd_millions: u64,
    }

    let replay = Fixture::from_json(APPLE_2023_FIXTURE).unwrap().replay();
    let error = replay
        .extract_with_report::<DriftedMetric>(APPLE_2023_PROMPT)
        .await
        .unwrap_err();

    assert!(error.to_string().contains("schema name differs"));
    assert_eq!(replay.remaining(), 1);
}

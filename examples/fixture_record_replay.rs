//! Record, sanitize, save, and strictly replay an offline fixture.
//!
//! Run with:
//! `cargo run --example fixture_record_replay --features mock`

use rstructor::{
    Fixture, FixtureRecorder, FixtureSanitizer, Instructor, LLMClient, MockClient, TokenUsage,
};
use serde::{Deserialize, Serialize};

const APPLE_2023_PROMPT: &str = "From Apple Inc.'s 2023 Form 10-K: Total net sales were $383,285 million for the fiscal year ended September 30, 2023. Extract the issuer, fiscal year, and net sales in USD millions.";

#[derive(Debug, PartialEq, Instructor, Serialize, Deserialize)]
struct FilingMetric {
    issuer: String,
    fiscal_year: u16,
    net_sales_usd_millions: u64,
}

fn fixture_sanitizer() -> FixtureSanitizer {
    // Replace account IDs, customer names, or other private strings here.
    // Credential-shaped JSON fields and inline media bytes are redacted separately.
    FixtureSanitizer::new(str::to_owned)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A live recording run can wrap OpenAIClient, AnthropicClient, GeminiClient,
    // or GrokClient in exactly the same way. MockClient keeps this example key-free.
    let source = MockClient::new().with_response_and_usage(
        r#"{"issuer":"Apple Inc.","fiscal_year":2023,"net_sales_usd_millions":383285}"#,
        TokenUsage::new("recorded-provider-model", 67, 24),
    );
    let recorder = FixtureRecorder::new(source, fixture_sanitizer());
    let recorded = recorder
        .extract_with_report::<FilingMetric>(APPLE_2023_PROMPT)
        .await?;
    assert_eq!(recorded.data.net_sales_usd_millions, 383_285);

    std::fs::create_dir_all("tmp")?;
    recorder.save("tmp/apple-2023-10k.fixture.json")?;

    let fixture = Fixture::load("tmp/apple-2023-10k.fixture.json")?;
    let replay = fixture.replay_with_sanitizer(fixture_sanitizer());
    let replayed = replay
        .extract_with_report::<FilingMetric>(APPLE_2023_PROMPT)
        .await?;
    assert_eq!(replayed.data, recorded.data);
    assert_eq!(replayed.report, recorded.report);
    replay.assert_finished()?;

    println!("recorded and replayed {} interaction", fixture.len());
    Ok(())
}

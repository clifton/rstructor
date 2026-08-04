//! Inspect every retry plus cumulative known token usage for an extraction run.

use rstructor::{AttemptOutcome, AttemptRecord, Instructor, LLMClient, OpenAIClient, RunUsage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Instructor, Serialize, Deserialize)]
#[llm(description = "A reconciled investment portfolio")]
struct Portfolio {
    portfolio_id: String,
    as_of: String,
    positions: Vec<Position>,
}

#[derive(Debug, Instructor, Serialize, Deserialize)]
struct Position {
    symbol: String,
    quantity: i64,
    market_value_usd: f64,
}

fn print_attempts(attempts: &[AttemptRecord], cumulative_usage: Option<&RunUsage>) {
    for attempt in attempts {
        let outcome = match &attempt.outcome {
            AttemptOutcome::Succeeded => "succeeded".to_string(),
            AttemptOutcome::Failed {
                message,
                disposition,
            } => {
                format!("failed ({disposition:?}): {message}")
            }
            _ => "unknown outcome".to_string(),
        };
        let tokens = attempt.usage.as_ref().map_or_else(
            || "unknown".to_string(),
            |usage| usage.total_tokens().to_string(),
        );

        println!(
            "attempt {}: {:?}, {outcome}, tokens={tokens}",
            attempt.number, attempt.kind
        );
    }

    if let Some(usage) = cumulative_usage {
        println!(
            "known run usage: {} input + {} output = {} tokens across {} reported attempts",
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens(),
            usage.reported_attempts
        );
        for (model, model_usage) in &usage.by_model {
            println!("  {model}: {} tokens", model_usage.total_tokens());
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAIClient::from_env()?.max_retries(2);
    let prompt = "
        Reconcile portfolio HF-ALPHA-001 as of 2026-07-29:
        ESU6: short 240 contracts, market value -$38,400,000
        AAPL: long 125,000 shares, market value $29,750,000
    ";

    match client.extract_with_report::<Portfolio>(prompt).await {
        Ok(extraction) => {
            println!("{:#?}", extraction.data);
            println!(
                "complete attempt history: {}",
                extraction.report.attempts_complete
            );
            print_attempts(
                &extraction.report.attempts,
                extraction.report.cumulative_usage.as_ref(),
            );
        }
        Err(failure) => {
            eprintln!("extraction failed: {}", failure.error());
            eprintln!(
                "complete attempt history: {}",
                failure.report.attempts_complete
            );
            print_attempts(
                &failure.report.attempts,
                failure.report.cumulative_usage.as_ref(),
            );
        }
    }

    Ok(())
}

//! Choose a provider and model at runtime with `rstructor::client`.
//!
//! The first slash separates the route from the provider-native model ID, so
//! model IDs may contain additional slashes:
//!
//! ```text
//! RSTRUCTOR_CLIENT=openrouter/moonshotai/kimi-k3 \
//!   cargo run --example runtime_provider_example
//! ```
//!
//! The example exits without a request when neither a CLI spec nor
//! `RSTRUCTOR_CLIENT` is supplied, keeping CI deterministic.

use rstructor::{Instructor, LLMClient};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Instructor, Serialize)]
struct Position {
    symbol: String,
    quantity: i64,
    market_value_usd: f64,
}

fn client_spec() -> Option<String> {
    std::env::args()
        .nth(1)
        .or_else(|| std::env::var("RSTRUCTOR_CLIENT").ok())
        .filter(|spec| !spec.trim().is_empty())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(spec) = client_spec() else {
        eprintln!(
            "Skipping provider request. Pass a provider/model argument or set \
             RSTRUCTOR_CLIENT (for example, openai/gpt-5.6-sol)."
        );
        return Ok(());
    };

    let client = rstructor::client(&spec)?;
    let position: Position = client
        .materialize("AAPL: long 125,000 shares, market value $29,750,000")
        .await?;

    println!("{position:#?}");
    Ok(())
}

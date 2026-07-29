//! Reuse a Serde + schemars domain model without deriving `Instructor`.
//!
//! This runnable recipe uses `MockClient`, so it needs no API key or network:
//!
//! ```text
//! cargo run --example schemars_bridge_example --features "mock,schemars"
//! ```

use rstructor::{LLMClient, MockClient, Schemars};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A position reported by a prime broker.
#[derive(Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
struct Position {
    /// Fund or separately managed account identifier.
    portfolio_id: String,
    /// Exchange ticker or contract symbol.
    symbol: String,
    /// Signed quantity: positive is long and negative is short.
    quantity: i64,
    /// Position market value in US dollars.
    market_value_usd: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = MockClient::new().with_response(
        r#"{
            "portfolio_id": "HF-ALPHA-001",
            "symbol": "AAPL",
            "quantity": 125000,
            "market_value_usd": 29750000.0
        }"#,
    );

    let position = client
        .materialize::<Schemars<Position>>("HF-ALPHA-001 owns 125,000 AAPL worth $29.75 million.")
        .await?
        .into_inner();

    assert_eq!(position.quantity, 125_000);
    assert_eq!(client.request_count(), 1);
    println!("{position:#?}");
    Ok(())
}

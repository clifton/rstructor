#![cfg(all(feature = "schemars", feature = "mock"))]

use rstructor::{LLMClient, MockClient, RStructorError, Schemars};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// A position held by an institutional portfolio.
#[derive(Debug, PartialEq, JsonSchema, Serialize, Deserialize)]
struct Position {
    /// Exchange ticker symbol.
    symbol: String,
    /// Signed number of shares held.
    quantity: i64,
}

/// A security listed on a public exchange.
#[derive(Debug, PartialEq, JsonSchema, Serialize, Deserialize)]
struct Instrument {
    /// Exchange ticker symbol.
    symbol: String,
    /// ISO 10383 market identifier code.
    venue: String,
}

/// A top-of-book quote for a listed instrument.
#[derive(Debug, PartialEq, JsonSchema, Serialize, Deserialize)]
struct Quote {
    /// Instrument whose order book was observed.
    instrument: Instrument,
    /// Best displayed bid in USD.
    bid: f64,
    /// Best displayed offer in USD.
    ask: f64,
}

/// Current exchange trading state.
#[derive(Debug, PartialEq, JsonSchema, Serialize, Deserialize)]
enum MarketState {
    Open,
    Halted,
    Closed,
}

#[test]
fn wrapper_serializes_transparently() {
    let wrapped = Schemars(Position {
        symbol: "AAPL".to_string(),
        quantity: 125_000,
    });

    assert_eq!(
        serde_json::to_value(&wrapped).expect("serialize transparent wrapper"),
        json!({"symbol": "AAPL", "quantity": 125_000})
    );
}

#[tokio::test]
async fn materializes_a_schemars_only_struct() {
    let client = MockClient::new().with_response(r#"{"symbol":"AAPL","quantity":125000}"#);

    let position = client
        .materialize::<Schemars<Position>>("AAPL: long 125,000 shares")
        .await
        .expect("valid position fixture")
        .into_inner();

    assert_eq!(
        position,
        Position {
            symbol: "AAPL".to_string(),
            quantity: 125_000,
        }
    );
    let request = client.last_request().expect("recorded materialization");
    assert_eq!(request.schema_name.as_deref(), Some("Position"));
    assert_eq!(
        request.schema.as_ref().unwrap()["properties"]["quantity"]["type"],
        "integer"
    );
}

#[tokio::test]
async fn materializes_a_schemars_only_nested_struct() {
    let client = MockClient::new().with_response(
        r#"{"instrument":{"symbol":"MSFT","venue":"XNAS"},"bid":512.31,"ask":512.34}"#,
    );

    let quote = client
        .materialize::<Schemars<Quote>>("Microsoft on Nasdaq is 512.31 bid and offered at 512.34")
        .await
        .expect("valid quote fixture")
        .into_inner();

    assert_eq!(
        quote,
        Quote {
            instrument: Instrument {
                symbol: "MSFT".to_string(),
                venue: "XNAS".to_string(),
            },
            bid: 512.31,
            ask: 512.34,
        }
    );

    let schema = client
        .last_request()
        .and_then(|request| request.schema)
        .expect("recorded nested schema");
    assert_eq!(
        schema["properties"]["instrument"]["properties"]["venue"]["description"],
        "ISO 10383 market identifier code."
    );
    assert!(!schema.to_string().contains("\"$ref\""));
    assert!(schema.get("definitions").is_none());
}

#[tokio::test]
async fn materializes_a_schemars_only_enum() {
    let client = MockClient::new().with_response(r#""Halted""#);

    let state = client
        .materialize::<Schemars<MarketState>>("The exchange paused trading after a limit move")
        .await
        .expect("valid market-state fixture")
        .into_inner();

    assert_eq!(state, MarketState::Halted);
    let schema = client
        .last_request()
        .and_then(|request| request.schema)
        .expect("recorded enum schema");
    assert_eq!(schema["enum"], json!(["Open", "Halted", "Closed"]));
}

#[derive(Debug, JsonSchema, Serialize, Deserialize)]
struct RecursiveNode {
    name: String,
    children: Vec<RecursiveNode>,
}

#[tokio::test]
async fn recursive_schemars_type_returns_a_clear_preflight_error() {
    let client = MockClient::new().with_response(r#"{"name":"root","children":[]}"#);

    let error = client
        .materialize::<Schemars<RecursiveNode>>("Build a recursive tree")
        .await
        .expect_err("recursive schemars models must be rejected");

    assert!(matches!(
        error,
        RStructorError::SchemaError(message)
            if message.contains("reference-free schemars schema")
                && message.contains("RecursiveNode")
                && message.contains("$ref")
                && message.contains("recursive types are not supported")
    ));
    assert_eq!(
        client.request_count(),
        0,
        "schema errors must occur before the mock records a provider request"
    );
}

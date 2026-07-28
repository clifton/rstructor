use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};

fn validate_position(position: &Position) -> rstructor::Result<()> {
    if position.symbol.is_empty() {
        return Err(rstructor::RStructorError::ValidationError(
            "symbol must not be empty".into(),
        ));
    }
    Ok(())
}

#[derive(Instructor, Serialize, Deserialize, Default)]
struct Metadata {
    venue: String,
}

#[derive(Instructor, Serialize, Deserialize)]
struct Leg {
    symbol: String,
    weight: f64,
}

#[derive(Instructor, Serialize, Deserialize, Default)]
#[llm(
    description = "A position from a risk system",
    title = "RiskPosition",
    examples = [::serde_json::json!({"symbol": "AAPL", "quantity": 100})],
    validate = "validate_position"
)]
#[serde(default, rename_all = "camelCase")]
struct Position {
    #[llm(
        description = "Exchange-listed symbol",
        example = "AAPL",
        examples = ["AAPL", "MSFT"]
    )]
    #[serde(alias = "ticker")]
    symbol: String,

    #[llm(description = "Signed share count", example = -100)]
    quantity: i64,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    note: Option<String>,

    // Instructor deliberately ignores Serde options it does not own. Their
    // schema semantics are covered by a separate compatibility project.
    #[serde(skip)]
    ignored: String,

    #[serde(flatten)]
    metadata: Metadata,

    #[llm(
        description = "Constituent positions",
        example = [
            ::serde_json::json!({"symbol": "AAPL", "weight": 0.5}),
            ::serde_json::json!({"symbol": "MSFT", "weight": 0.5})
        ]
    )]
    legs: Vec<Leg>,
}

#[derive(Instructor, Serialize, Deserialize)]
#[serde(tag = "kind", content = "payload")]
enum Event {
    #[llm(description = "A fill event")]
    #[serde(rename = "fill")]
    Fill {
        #[llm(description = "Execution price", example = 187.25)]
        price: f64,
    },
    #[serde(rename(serialize = "haltOut", deserialize = "haltIn"))]
    Halt,
}

#[derive(Instructor, Serialize, Deserialize)]
#[llm(examples(::serde_json::json!({"tickers": ["SPY", "QQQ"]})))]
struct ParenthesizedExamples {
    #[llm(examples("SPY", "QQQ"))]
    tickers: Vec<String>,
}

fn main() {
    let _ = Position::schema();
    let _ = Event::schema();
    let _ = ParenthesizedExamples::schema();
}

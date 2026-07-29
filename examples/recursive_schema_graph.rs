//! Context-aware schema generation for mutually recursive domain types.
//!
//! Derived schemas share one build context, so cycles terminate in root-level
//! `$defs` and each `$ref` resolves from the JSON Schema document root.

use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Instructor, Serialize, Deserialize)]
struct Fund {
    lei: String,
    prime_broker: Option<Box<PrimeBroker>>,
}

#[derive(Debug, Instructor, Serialize, Deserialize)]
struct PrimeBroker {
    lei: String,
    sponsored_funds: Vec<Fund>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Fund::schema().to_json();
    println!("{}", serde_json::to_string_pretty(&schema)?);

    let fund: Fund = serde_json::from_value(serde_json::json!({
        "lei": "5493001KJTIIGC8Y1R12",
        "prime_broker": {
            "lei": "7H6GLXDRUGQFU57RNE97",
            "sponsored_funds": []
        }
    }))?;
    println!("{fund:#?}");

    Ok(())
}

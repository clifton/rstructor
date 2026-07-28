use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Instructor, Serialize, Deserialize)]
#[llm(description = "A compact risk snapshot", title = "RiskSnapshot")]
#[serde(rename_all = "camelCase")]
struct RiskSnapshot {
    #[llm(description = "Gross market value", example = 1250000.0)]
    gross_value: f64,
    #[llm(description = "Optional desk note")]
    desk_note: Option<String>,
}

#[derive(Instructor, Serialize, Deserialize)]
struct AllocationLeg {
    symbol: String,
    weight: f64,
}

#[derive(Instructor, Serialize, Deserialize)]
struct AllocationExamples {
    #[llm(example = ::serde_json::json!({"symbol": "AAPL", "weight": 0.6}))]
    primary: AllocationLeg,

    #[llm(example = [
        ::serde_json::json!({"symbol": "AAPL", "weight": 0.6}),
        ::serde_json::json!({"symbol": "MSFT", "weight": 0.4})
    ])]
    legs: Vec<AllocationLeg>,
}

#[test]
fn valid_attributes_keep_the_existing_schema_shape() {
    assert_eq!(
        RiskSnapshot::schema().to_json(),
        json!({
            "type": "object",
            "title": "RiskSnapshot",
            "description": "A compact risk snapshot",
            "properties": {
                "grossValue": {
                    "type": "number",
                    "description": "Gross market value",
                    "example": 1250000.0
                },
                "deskNote": {
                    "type": "string",
                    "description": "Optional desk note"
                }
            },
            "required": ["grossValue"]
        })
    );
}

#[test]
fn native_object_examples_remain_json_values() {
    let schema = AllocationExamples::schema().to_json();

    assert_eq!(
        schema["properties"]["primary"]["example"],
        json!({"symbol": "AAPL", "weight": 0.6})
    );
    assert_eq!(
        schema["properties"]["legs"]["example"],
        json!([
            {"symbol": "AAPL", "weight": 0.6},
            {"symbol": "MSFT", "weight": 0.4}
        ])
    );
}

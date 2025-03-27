use super::{Schema, SchemaBuilder};
use serde_json::json;

#[test]
fn test_schema_creation() {
    let schema_json = json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name field"
            },
            "age": {
                "type": "integer",
                "description": "The age field"
            }
        },
        "required": ["name"]
    });

    let schema = Schema::new(schema_json.clone());

    // Compare the schema using string representations
    // Enhanced schema might add array properties, but core values should be the same
    let schema_str1 = serde_json::to_string(&schema_json).unwrap();
    let schema_str2 = serde_json::to_string(&schema.schema).unwrap();
    assert_eq!(schema_str1, schema_str2);
}

#[test]
fn test_schema_builder() {
    let schema = SchemaBuilder::object()
        .title("Person")
        .description("A person object")
        .property(
            "name",
            json!({
                "type": "string",
                "description": "The person's name"
            }),
            true,
        )
        .property(
            "age",
            json!({
                "type": "integer",
                "description": "The person's age"
            }),
            true,
        )
        .property(
            "address",
            json!({
                "type": "object",
                "properties": {
                    "street": { "type": "string" },
                    "city": { "type": "string" }
                }
            }),
            false,
        )
        .build();

    // Get the schema JSON to check properties
    let schema_json = schema.to_json();

    // Verify correct properties
    assert_eq!(schema_json["type"], "object");
    assert_eq!(schema_json["title"], "Person");
    assert_eq!(schema_json["description"], "A person object");

    // Check that properties exist and are correct
    assert!(schema_json["properties"]["name"]["type"] == "string");
    assert!(schema_json["properties"]["name"]["description"] == "The person's name");
    assert!(schema_json["properties"]["age"]["type"] == "integer");
    assert!(schema_json["properties"]["age"]["description"] == "The person's age");
    assert!(schema_json["properties"]["address"]["type"] == "object");

    // Check that required fields exist (but don't enforce order)
    let required = schema_json["required"].as_array().unwrap();
    assert_eq!(required.len(), 2);
    assert!(required.iter().any(|v| v == "name"));
    assert!(required.iter().any(|v| v == "age"));
}

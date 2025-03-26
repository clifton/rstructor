use serde_json::json;
use super::{Schema, SchemaBuilder};

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
    
    assert_eq!(schema.to_json(), &schema_json);
    assert_eq!(schema.to_string(), schema_json.to_string());
}

#[test]
fn test_schema_builder() {
    let expected = json!({
        "type": "object",
        "title": "Person",
        "description": "A person object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The person's name"
            },
            "age": {
                "type": "integer",
                "description": "The person's age"
            },
            "address": {
                "type": "object",
                "properties": {
                    "street": { "type": "string" },
                    "city": { "type": "string" }
                }
            }
        },
        "required": ["age", "name"]
    });
    
    let schema = SchemaBuilder::object()
        .title("Person")
        .description("A person object")
        .property(
            "name",
            json!({
                "type": "string",
                "description": "The person's name"
            }),
            true
        )
        .property(
            "age",
            json!({
                "type": "integer",
                "description": "The person's age"
            }),
            true
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
            false
        )
        .build();
    
    assert_eq!(schema.to_json(), &expected);
}
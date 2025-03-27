use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// Simple enum with primitive associated data
#[derive(Instructor, Serialize, Deserialize, Debug)]
enum UserStatus {
    #[llm(description = "The user is online")]
    Online,

    #[llm(description = "The user is offline")]
    Offline,

    #[llm(description = "The user is away with an optional message")]
    Away(String),

    #[llm(description = "The user is busy until a specific time")]
    Busy(u32),
}

#[test]
fn test_enum_with_data_schema() {
    let schema_obj = UserStatus::schema();
    let schema = schema_obj.to_json();

    // Check that we're using oneOf for complex enums
    assert!(
        schema.get("oneOf").is_some(),
        "Schema should use oneOf for enums with associated data"
    );

    if let Some(Value::Array(variants)) = schema.get("oneOf") {
        // Should have 4 variants
        assert_eq!(variants.len(), 4, "Should have 4 variants");
    }
}

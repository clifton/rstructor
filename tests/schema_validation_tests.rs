#[cfg(test)]
mod schema_validation_tests {
    use rstructor::{Instructor, SchemaType};
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    // Test model for validation
    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct User {
        #[llm(description = "User's full name", example = "John Doe")]
        name: String,

        #[llm(description = "User's email address", example = "john@example.com")]
        email: String,

        #[llm(description = "User's age in years", example = 30)]
        age: u32,

        #[llm(description = "User's subscription status", example = true)]
        is_subscribed: bool,

        #[llm(description = "User's tags", example = ["premium", "early-adopter"])]
        tags: Vec<String>,

        #[llm(description = "User's profile picture URL")]
        profile_picture: Option<String>,
    }

    // Manual validation check since we don't have a validate method directly
    fn validate_against_schema(
        schema_json: &serde_json::Value,
        instance: &serde_json::Value,
    ) -> bool {
        // Check if required fields are present
        if let Some(required) = schema_json["required"].as_array() {
            for field in required {
                let field_name = field.as_str().unwrap();
                if !instance.as_object().unwrap().contains_key(field_name) {
                    return false;
                }
            }
        }

        // Check if field types match
        if let Some(properties) = schema_json["properties"].as_object() {
            for (field_name, field_schema) in properties {
                if let Some(field_value) = instance.get(field_name) {
                    // Skip null values for optional fields
                    if field_value.is_null() {
                        continue;
                    }

                    // Check type
                    let schema_type = field_schema["type"].as_str().unwrap_or("object");
                    match schema_type {
                        "string" => {
                            if !field_value.is_string() {
                                return false;
                            }
                        }
                        "integer" | "number" => {
                            if !field_value.is_number() {
                                return false;
                            }
                        }
                        "boolean" => {
                            if !field_value.is_boolean() {
                                return false;
                            }
                        }
                        "array" => {
                            if !field_value.is_array() {
                                return false;
                            }
                        }
                        "object" => {
                            if !field_value.is_object() {
                                return false;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        true
    }

    #[test]
    fn test_schema_validation_valid_instance() {
        let schema = User::schema();
        let schema_json = schema.to_json();

        // Create a valid instance
        let valid_instance = json!({
            "name": "Jane Smith",
            "email": "jane@example.com",
            "age": 28,
            "is_subscribed": false,
            "tags": ["new-user", "trial"]
        });

        // Validation should pass for valid instance
        assert!(validate_against_schema(&schema_json, &valid_instance));
    }

    #[test]
    fn test_schema_validation_missing_required() {
        let schema = User::schema();
        let schema_json = schema.to_json();

        // Missing required fields
        let missing_required = json!({
            "name": "Jane Smith",
            "email": "jane@example.com",
            // Missing age
            "is_subscribed": false,
        });

        // Validation should fail
        assert!(!validate_against_schema(&schema_json, &missing_required));
    }

    #[test]
    fn test_schema_validation_type_mismatch() {
        let schema = User::schema();
        let schema_json = schema.to_json();

        // Type mismatch (age as string)
        let type_mismatch = json!({
            "name": "Jane Smith",
            "email": "jane@example.com",
            "age": "twenty-eight", // Should be a number
            "is_subscribed": false,
            "tags": ["new-user", "trial"]
        });

        // Validation should fail
        assert!(!validate_against_schema(&schema_json, &type_mismatch));
    }

    #[test]
    fn test_schema_validation_with_optional() {
        let schema = User::schema();
        let schema_json = schema.to_json();

        // Including optional field
        let with_optional = json!({
            "name": "Jane Smith",
            "email": "jane@example.com",
            "age": 28,
            "is_subscribed": false,
            "tags": ["new-user", "trial"],
            "profile_picture": "https://example.com/jane.jpg"
        });

        // Validation should pass with optional field
        assert!(validate_against_schema(&schema_json, &with_optional));
    }
}

#[cfg(test)]
mod schema_builder_tests {
    use rstructor::schema::SchemaBuilder;
    use serde_json::json;

    #[test]
    fn test_object_schema_builder() {
        let schema = SchemaBuilder::object()
            .title("TestObject")
            .description("A test object schema")
            .property(
                "name",
                json!({"type": "string", "description": "The name"}),
                true,
            )
            .property(
                "age",
                json!({"type": "integer", "description": "The age"}),
                true,
            )
            .property(
                "email",
                json!({"type": "string", "description": "Email address"}),
                false,
            )
            .build();

        let json = schema.to_json();

        // Validate schema
        assert_eq!(json["type"], "object");
        assert_eq!(json["title"], "TestObject");
        assert_eq!(json["description"], "A test object schema");

        // Check properties
        let props = json["properties"].as_object().unwrap();
        assert!(props.contains_key("name"));
        assert!(props.contains_key("age"));
        assert!(props.contains_key("email"));

        assert_eq!(props["name"]["type"], "string");
        assert_eq!(props["name"]["description"], "The name");

        // Check required fields
        let required = json["required"].as_array().unwrap();
        assert_eq!(required.len(), 2);
        assert!(required.iter().any(|v| v == "name"));
        assert!(required.iter().any(|v| v == "age"));
        assert!(!required.iter().any(|v| v == "email"));
    }

    #[test]
    fn test_array_schema_builder() {
        let schema = SchemaBuilder::array(json!({"type": "string"}))
            .title("StringArray")
            .description("An array of strings")
            .build();

        let json = schema.to_json();

        // Validate schema
        assert_eq!(json["type"], "array");
        assert_eq!(json["title"], "StringArray");
        assert_eq!(json["description"], "An array of strings");
        assert_eq!(json["items"]["type"], "string");
    }

    #[test]
    fn test_nested_schema_builder() {
        let address_schema = SchemaBuilder::object()
            .property("street", json!({"type": "string"}), true)
            .property("city", json!({"type": "string"}), true)
            .property("zip", json!({"type": "string"}), true)
            .build();

        let schema = SchemaBuilder::object()
            .title("Person")
            .property("name", json!({"type": "string"}), true)
            .property("age", json!({"type": "integer"}), true)
            .property("address", address_schema.to_json().clone(), true)
            .build();

        let json = schema.to_json();

        // Check root schema
        assert_eq!(json["type"], "object");
        assert_eq!(json["title"], "Person");

        // Check nested schema
        let address_prop = &json["properties"]["address"];
        assert_eq!(address_prop["type"], "object");

        let address_props = address_prop["properties"].as_object().unwrap();
        assert!(address_props.contains_key("street"));
        assert!(address_props.contains_key("city"));
        assert!(address_props.contains_key("zip"));
    }

    #[test]
    fn test_property_with_example() {
        let schema = SchemaBuilder::object()
            .property(
                "name",
                json!({"type": "string", "example": "John Doe"}),
                true,
            )
            .property("score", json!({"type": "number", "example": 42.5}), false)
            .build();

        let json = schema.to_json();
        let props = json["properties"].as_object().unwrap();

        assert_eq!(props["name"]["example"], "John Doe");
        assert_eq!(props["score"]["example"], 42.5);
    }

    #[test]
    fn test_custom_schema_builder() {
        // We'll create a string enum schema manually since there's no string_enum helper
        let schema = SchemaBuilder::new()
            .title("Color")
            .description("A color enum")
            .property("type", json!("string"), false)
            .property("enum", json!(["Red", "Green", "Blue"]), false)
            .build();

        let json = schema.to_json();

        // Validate basic properties
        assert_eq!(json["title"], "Color");
        assert_eq!(json["description"], "A color enum");

        // The structure is a bit different since we're building it manually,
        // but it still contains the enum values
        assert!(json["properties"]["enum"].is_array());
        let enum_values = json["properties"]["enum"].as_array().unwrap();
        assert_eq!(enum_values.len(), 3);
        assert!(enum_values.iter().any(|v| v == "Red"));
        assert!(enum_values.iter().any(|v| v == "Green"));
        assert!(enum_values.iter().any(|v| v == "Blue"));
    }
}

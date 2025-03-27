#[cfg(test)]
mod llm_model_tests {
    use rstructor::{Instructor, RStructorError, Schema, SchemaType};
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    // Simple struct for testing the Instructor trait
    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestModel {
        name: String,
        age: u32,
        active: bool,
    }

    // Manually implement SchemaType for TestModel
    impl SchemaType for TestModel {
        fn schema() -> Schema {
            Schema::new(json!({
                "type": "object",
                "title": "TestModel",
                "properties": {
                    "name": {
                        "type": "string"
                    },
                    "age": {
                        "type": "integer"
                    },
                    "active": {
                        "type": "boolean"
                    }
                },
                "required": ["name", "age", "active"]
            }))
        }

        fn schema_name() -> Option<String> {
            Some("TestModel".to_string())
        }
    }

    // Custom validation implementation
    impl TestModel {
        fn validate(&self) -> rstructor::Result<()> {
            if self.name.is_empty() {
                return Err(RStructorError::ValidationError(
                    "Name cannot be empty".to_string(),
                ));
            }

            if self.age < 18 {
                return Err(RStructorError::ValidationError(
                    "Age must be at least 18".to_string(),
                ));
            }

            Ok(())
        }
    }

    #[test]
    fn test_llm_model_default_validate() {
        // The default implementation of Instructor::validate should return Ok(())
        // This test confirms the default behavior using a type that doesn't override validate

        // Create a simple struct that just inherits the default validate implementation
        #[derive(Serialize, Deserialize, Debug)]
        struct SimpleModel {
            value: String,
        }

        impl SchemaType for SimpleModel {
            fn schema() -> Schema {
                Schema::new(json!({"type": "object"}))
            }
        }

        // SimpleModel should have the default implementation of validate
        let model = SimpleModel {
            value: "test".to_string(),
        };
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_llm_model_custom_validate_success() {
        // Test that a custom implementation of validate works correctly in the success case
        let model = TestModel {
            name: "Alice".to_string(),
            age: 30,
            active: true,
        };

        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_llm_model_custom_validate_empty_name() {
        // Test that validation fails when the name is empty
        let model = TestModel {
            name: "".to_string(), // empty name
            age: 30,
            active: true,
        };

        let result = model.validate();
        assert!(result.is_err());

        if let Err(RStructorError::ValidationError(msg)) = result {
            assert_eq!(msg, "Name cannot be empty");
        } else {
            panic!("Expected ValidationError, got {:?}", result);
        }
    }

    #[test]
    fn test_llm_model_custom_validate_underage() {
        // Test that validation fails when the age is less than 18
        let model = TestModel {
            name: "Bob".to_string(),
            age: 17, // underage
            active: true,
        };

        let result = model.validate();
        assert!(result.is_err());

        if let Err(RStructorError::ValidationError(msg)) = result {
            assert_eq!(msg, "Age must be at least 18");
        } else {
            panic!("Expected ValidationError, got {:?}", result);
        }
    }

    #[test]
    fn test_schema_generation() {
        // Test that the schema generation works correctly
        let schema = TestModel::schema();
        let json = schema.to_json();

        assert_eq!(json["type"], "object");
        assert_eq!(json["title"], "TestModel");

        let properties = json["properties"].as_object().unwrap();
        assert!(properties.contains_key("name"));
        assert!(properties.contains_key("age"));
        assert!(properties.contains_key("active"));

        assert_eq!(properties["name"]["type"], "string");
        assert_eq!(properties["age"]["type"], "integer");
        assert_eq!(properties["active"]["type"], "boolean");
    }

    #[test]
    fn test_schema_name() {
        // Test that schema_name returns the expected value
        assert_eq!(TestModel::schema_name(), Some("TestModel".to_string()));
    }
}

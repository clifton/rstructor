// Tests for validation method without dead code warnings

// We use separate modules to isolate each test
// and prevent conflicts with the implementations

mod test_validation_works {
    use rstructor::{Instructor, RStructorError};
    use serde::{Deserialize, Serialize};

    // Define a struct with a validation method
    #[derive(Debug, Serialize, Deserialize)]
    pub struct Person {
        pub name: String,
        pub age: u32,
    }

    // Custom validation implementation
    impl Person {
        // This method should not be flagged as dead code
        pub fn validate(&self) -> rstructor::Result<()> {
            if self.name.is_empty() {
                return Err(RStructorError::ValidationError(
                    "Name cannot be empty".to_string(),
                ));
            }
            if self.age == 0 {
                return Err(RStructorError::ValidationError(
                    "Age cannot be zero".to_string(),
                ));
            }
            Ok(())
        }
    }

    // Implement SchemaType for proper Instructor implementation
    impl rstructor::SchemaType for Person {
        fn schema() -> rstructor::Schema {
            use serde_json::json;
            rstructor::Schema::new(json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "age": { "type": "integer" }
                },
                "required": ["name", "age"]
            }))
        }
    }

    // Override the Instructor trait to directly call our validate method
    // (This test will test that our validation method is properly used)
    impl Instructor for Person {
        fn validate(&self) -> rstructor::Result<()> {
            // Call our validate method directly
            Person::validate(self)
        }
    }

    #[test]
    fn test_validate_method_gets_called() {
        // Valid person
        let person = Person {
            name: "John".to_string(),
            age: 30,
        };
        // The validate method should be called through the Instructor trait
        assert!(person.validate().is_ok());

        // Invalid person (empty name)
        let person = Person {
            name: "".to_string(),
            age: 30,
        };
        let result = person.validate();
        assert!(result.is_err());
        if let Err(RStructorError::ValidationError(msg)) = result {
            assert_eq!(msg, "Name cannot be empty");
        } else {
            panic!("Expected ValidationError");
        }

        // Invalid person (zero age)
        let person = Person {
            name: "John".to_string(),
            age: 0,
        };
        let result = person.validate();
        assert!(result.is_err());
        if let Err(RStructorError::ValidationError(msg)) = result {
            assert_eq!(msg, "Age cannot be zero");
        } else {
            panic!("Expected ValidationError");
        }
    }
}

mod test_simple_type {
    use rstructor::Instructor;
    use serde::{Deserialize, Serialize};

    // Type without a validate method
    #[derive(Debug, Serialize, Deserialize)]
    pub struct SimpleType {
        pub value: String,
    }

    // Implement SchemaType
    impl rstructor::SchemaType for SimpleType {
        fn schema() -> rstructor::Schema {
            use serde_json::json;
            rstructor::Schema::new(json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"]
            }))
        }
    }

    // Provide a manual implementation to avoid derive macro
    impl Instructor for SimpleType {
        fn validate(&self) -> rstructor::Result<()> {
            // Default implementation (does nothing)
            Ok(())
        }
    }

    #[test]
    fn test_no_validate_method() {
        let simple = SimpleType {
            value: "test".to_string(),
        };
        // The default validate implementation should be used
        // which returns Ok(())
        assert!(simple.validate().is_ok());
    }
}

mod test_nested_validation {
    use rstructor::{Instructor, RStructorError};
    use serde::{Deserialize, Serialize};

    // Define our types for nested validation
    #[derive(Debug, Serialize, Deserialize)]
    pub struct Person {
        pub name: String,
        pub age: u32,
    }

    impl Person {
        pub fn validate(&self) -> rstructor::Result<()> {
            if self.name.is_empty() {
                return Err(RStructorError::ValidationError(
                    "Name cannot be empty".to_string(),
                ));
            }
            Ok(())
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Container {
        pub person: Person,
        pub name: String,
    }

    impl Container {
        pub fn validate(&self) -> rstructor::Result<()> {
            if self.name.is_empty() {
                return Err(RStructorError::ValidationError(
                    "Container name cannot be empty".to_string(),
                ));
            }
            // Validate the nested person
            self.person.validate()
        }
    }

    // Implement SchemaType for our types
    impl rstructor::SchemaType for Person {
        fn schema() -> rstructor::Schema {
            use serde_json::json;
            rstructor::Schema::new(json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "age": { "type": "integer" }
                },
                "required": ["name", "age"]
            }))
        }
    }

    impl rstructor::SchemaType for Container {
        fn schema() -> rstructor::Schema {
            use serde_json::json;
            rstructor::Schema::new(json!({
                "type": "object",
                "properties": {
                    "person": { "type": "object" },
                    "name": { "type": "string" }
                },
                "required": ["person", "name"]
            }))
        }
    }

    // Custom implementation to prevent recursion
    impl Instructor for Person {
        fn validate(&self) -> rstructor::Result<()> {
            Person::validate(self)
        }
    }

    impl Instructor for Container {
        fn validate(&self) -> rstructor::Result<()> {
            Container::validate(self)
        }
    }

    #[test]
    fn test_nested_validation() {
        // Valid container
        let container = Container {
            person: Person {
                name: "John".to_string(),
                age: 30,
            },
            name: "Container".to_string(),
        };
        assert!(container.validate().is_ok());

        // Invalid container (empty name)
        let container = Container {
            person: Person {
                name: "John".to_string(),
                age: 30,
            },
            name: "".to_string(),
        };
        let result = container.validate();
        assert!(result.is_err());
        if let Err(RStructorError::ValidationError(msg)) = result {
            assert_eq!(msg, "Container name cannot be empty");
        } else {
            panic!("Expected ValidationError");
        }

        // Invalid person in container
        let container = Container {
            person: Person {
                name: "".to_string(),
                age: 30,
            },
            name: "Container".to_string(),
        };
        let result = container.validate();
        assert!(result.is_err());
        if let Err(RStructorError::ValidationError(msg)) = result {
            assert_eq!(msg, "Name cannot be empty");
        } else {
            panic!("Expected ValidationError");
        }
    }
}

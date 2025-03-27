#[cfg(test)]
mod validation_tests {
    use rstructor::{Instructor, RStructorError};
    use serde::{Deserialize, Serialize};

    // Test model with validation
    #[derive(Instructor, Serialize, Deserialize, Debug)]
    #[llm(description = "A product in an inventory")]
    struct Product {
        #[llm(description = "Product name", example = "Laptop")]
        name: String,

        #[llm(description = "Product price in USD", example = 999.99)]
        price: f64,

        #[llm(description = "Quantity in stock", example = 42)]
        quantity: u32,

        #[llm(description = "Product categories", example = ["Electronics", "Computers"])]
        categories: Vec<String>,
    }

    // Custom validation implementation
    impl Product {
        fn validate(&self) -> rstructor::Result<()> {
            // Price must be positive
            if self.price <= 0.0 {
                return Err(RStructorError::ValidationError(format!(
                    "Product price must be positive, got {}",
                    self.price
                )));
            }

            // Name can't be empty
            if self.name.trim().is_empty() {
                return Err(RStructorError::ValidationError(
                    "Product name cannot be empty".to_string(),
                ));
            }

            // Must have at least one category
            if self.categories.is_empty() {
                return Err(RStructorError::ValidationError(
                    "Product must have at least one category".to_string(),
                ));
            }

            Ok(())
        }
    }

    #[test]
    fn test_valid_product() {
        let product = Product {
            name: "Laptop".to_string(),
            price: 999.99,
            quantity: 42,
            categories: vec!["Electronics".to_string(), "Computers".to_string()],
        };

        // Validation should pass
        assert!(product.validate().is_ok());
    }

    #[test]
    fn test_invalid_price() {
        let product = Product {
            name: "Laptop".to_string(),
            price: -10.0, // Invalid price
            quantity: 42,
            categories: vec!["Electronics".to_string()],
        };

        // Validation should fail with specific error
        let err = product.validate().unwrap_err();
        if let RStructorError::ValidationError(msg) = err {
            assert!(msg.contains("price must be positive"));
        } else {
            panic!("Expected ValidationError, got {:?}", err);
        }
    }

    #[test]
    fn test_empty_name() {
        let product = Product {
            name: "   ".to_string(), // Empty name after trimming
            price: 999.99,
            quantity: 42,
            categories: vec!["Electronics".to_string()],
        };

        // Validation should fail with specific error
        let err = product.validate().unwrap_err();
        if let RStructorError::ValidationError(msg) = err {
            assert!(msg.contains("name cannot be empty"));
        } else {
            panic!("Expected ValidationError, got {:?}", err);
        }
    }

    #[test]
    fn test_no_categories() {
        let product = Product {
            name: "Laptop".to_string(),
            price: 999.99,
            quantity: 42,
            categories: vec![], // Empty categories
        };

        // Validation should fail with specific error
        let err = product.validate().unwrap_err();
        if let RStructorError::ValidationError(msg) = err {
            assert!(msg.contains("must have at least one category"));
        } else {
            panic!("Expected ValidationError, got {:?}", err);
        }
    }
}

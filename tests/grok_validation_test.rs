//! This test file demonstrates validation with the Grok client.
//!
//! These tests require a valid xAI API key in the environment:
//!
//! ```bash
//! export XAI_API_KEY=your_key_here
//! cargo test --test grok_validation_test
//! ```

#[cfg(test)]
mod grok_validation_tests {
    use rstructor::{GrokClient, GrokModel, Instructor, LLMClient, RStructorError};
    use serde::{Deserialize, Serialize};

    // Define a data model with validation rules
    #[derive(Instructor, Serialize, Deserialize, Debug)]
    #[llm(description = "Information about a product")]
    struct ProductInfo {
        #[llm(description = "Name of the product")]
        name: String,

        #[llm(description = "Price in USD", example = 29.99)]
        price: f32,

        #[llm(description = "Stock quantity", example = 100)]
        stock: u32,

        #[llm(description = "Product category", example = "Electronics")]
        category: String,
    }

    impl ProductInfo {
        fn validate(&self) -> rstructor::Result<()> {
            // Product name can't be empty
            if self.name.trim().is_empty() {
                return Err(RStructorError::ValidationError(
                    "Product name cannot be empty".to_string(),
                ));
            }

            // Price must be positive
            if self.price < 0.0 {
                return Err(RStructorError::ValidationError(format!(
                    "Price must be positive, got ${:.2}",
                    self.price
                )));
            }

            // Price should be reasonable (less than $1 million)
            if self.price > 1_000_000.0 {
                return Err(RStructorError::ValidationError(format!(
                    "Price seems unreasonably high: ${:.2}",
                    self.price
                )));
            }

            // Category can't be empty
            if self.category.trim().is_empty() {
                return Err(RStructorError::ValidationError(
                    "Product category cannot be empty".to_string(),
                ));
            }

            Ok(())
        }
    }

    // Test validation catching an invalid price
    #[cfg(feature = "grok")]
    #[tokio::test]
    async fn test_grok_validation_fails_with_negative_price() {
        // For this test, we really just need to validate manually, but we'll
        // skip full API validation if no API key is available
        // Test with empty string to use XAI_API_KEY env var
        let client = match GrokClient::from_env() {
            Ok(client) => client.model(GrokModel::Grok4).temperature(0.0).build(),
            Err(_) => {
                println!("Skipping Grok API test (no API key found)");
                // Just test the validation logic directly
                let invalid_product = ProductInfo {
                    name: "Test Product".to_string(),
                    price: -10.0, // Negative price - invalid
                    stock: 100,
                    category: "Electronics".to_string(),
                };

                let validation_result = invalid_product.validate();
                assert!(
                    validation_result.is_err(),
                    "Validation should fail with negative price"
                );

                if let Err(RStructorError::ValidationError(msg)) = validation_result {
                    assert!(
                        msg.contains("Price") || msg.contains("price"),
                        "Error should mention price: {}",
                        msg
                    );
                }
                return;
            }
        };

        // First get a valid product response
        let prompt = "Describe a smartphone product with realistic details";
        let valid_result = client.generate_struct::<ProductInfo>(prompt).await;

        // Skip the test if we have API issues
        if let Err(RStructorError::ApiError(_)) = &valid_result {
            println!("Skipping due to API error: {:?}", valid_result);

            // Still test the validation directly
            let invalid_product = ProductInfo {
                name: "Test Product".to_string(),
                price: -10.0, // Negative price - invalid
                stock: 100,
                category: "Electronics".to_string(),
            };

            let validation_result = invalid_product.validate();
            assert!(
                validation_result.is_err(),
                "Validation should fail with negative price"
            );

            if let Err(RStructorError::ValidationError(msg)) = validation_result {
                assert!(
                    msg.contains("Price") || msg.contains("price"),
                    "Error should mention price: {}",
                    msg
                );
            }
            return;
        }

        // Make sure we can get a valid response
        assert!(
            valid_result.is_ok(),
            "Should be able to get valid product data"
        );

        // Now create a new ProductInfo with an invalid price
        let invalid_product = ProductInfo {
            name: "Price Test Product".to_string(),
            price: -10.0, // Negative price - invalid
            stock: 100,
            category: "Electronics".to_string(),
        };

        // Validate it - should fail with price error
        let validation_result = invalid_product.validate();

        // Check that validation failed
        assert!(
            validation_result.is_err(),
            "Validation should fail with negative price"
        );

        // Check that the error is about price
        if let Err(RStructorError::ValidationError(msg)) = validation_result {
            assert!(
                msg.contains("Price") || msg.contains("price"),
                "Error should mention price: {}",
                msg
            );
        } else if let Err(e) = validation_result {
            panic!("Expected ValidationError about price, got: {:?}", e);
        }
    }

    // Test validation catching an unreasonably high price
    #[cfg(feature = "grok")]
    #[tokio::test]
    async fn test_grok_validation_fails_with_extreme_price() {
        // For this test, we'll manually create a ProductInfo with extreme price
        // We don't even need to make an API call for this test

        // Create a product info with an unreasonably high price
        let invalid_product = ProductInfo {
            name: "Extreme Price Product".to_string(),
            price: 2_000_000.0, // Way too high
            stock: 100,
            category: "Electronics".to_string(),
        };

        // Validate it - should fail with price error
        let validation_result = invalid_product.validate();

        // Check that validation failed
        assert!(
            validation_result.is_err(),
            "Validation should fail with extreme price"
        );

        // Check that the error is about price
        if let Err(RStructorError::ValidationError(msg)) = validation_result {
            assert!(
                msg.contains("Price") || msg.contains("price") || msg.contains("high"),
                "Error should mention price: {}",
                msg
            );
        } else if let Err(e) = validation_result {
            panic!("Expected ValidationError about price, got: {:?}", e);
        }
    }

    // Valid data test
    #[cfg(feature = "grok")]
    #[tokio::test]
    async fn test_grok_validation_succeeds_with_valid_data() {
        // This test demonstrates successful validation with reasonable data
        // Test with empty string to use XAI_API_KEY env var
        let client = match GrokClient::from_env() {
            Ok(client) => client
                .model(GrokModel::Grok4)
                .temperature(0.0) // Use deterministic temperature for consistent results
                .build(),
            Err(_) => {
                println!("Skipping Grok API test (no API key found)");
                // Create a valid product object to test validation directly
                let valid_product = ProductInfo {
                    name: "Laptop".to_string(),
                    price: 999.99,
                    stock: 50,
                    category: "Electronics".to_string(),
                };

                assert!(
                    valid_product.validate().is_ok(),
                    "Valid product data should pass validation"
                );
                return;
            }
        };

        // Normal prompt asking for product information
        let prompt = "Describe a laptop product with realistic details";

        // Should succeed validation
        let result = client.generate_struct::<ProductInfo>(prompt).await;

        // If we get API errors, skip the test but still test validation directly
        if let Err(RStructorError::ApiError(_)) = &result {
            println!("Skipping due to API error: {:?}", result);

            // Create a valid product object to test validation directly
            let valid_product = ProductInfo {
                name: "Laptop".to_string(),
                price: 999.99,
                stock: 50,
                category: "Electronics".to_string(),
            };

            assert!(
                valid_product.validate().is_ok(),
                "Valid product data should pass validation"
            );
            return;
        }

        // Validation should pass
        assert!(result.is_ok(), "Validation failed: {:?}", result.err());

        // Check the data looks reasonable
        let product = result.unwrap();
        assert!(!product.name.is_empty());
        assert!(product.price > 0.0 && product.price < 1_000_000.0);
        assert!(!product.category.is_empty());
    }
}

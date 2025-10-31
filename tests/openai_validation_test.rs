//! This test file demonstrates validation with the OpenAI client.
//!
//! These tests require a valid OpenAI API key in the environment:
//!
//! ```bash
//! export OPENAI_API_KEY=your_key_here
//! cargo test --test openai_validation_test
//! ```

#[cfg(test)]
mod openai_validation_tests {
    use rstructor::{Instructor, LLMClient, OpenAIClient, OpenAIModel, RStructorError};
    use serde::{Deserialize, Serialize};
    use std::env;

    // Define a data model with validation rules
    #[derive(Instructor, Serialize, Deserialize, Debug)]
    #[llm(description = "Information about a recipe")]
    struct RecipeInfo {
        #[llm(description = "Name of the recipe")]
        name: String,

        #[llm(description = "Cooking time in minutes", example = 30)]
        cooking_time: u32,

        #[llm(description = "Preparation time in minutes", example = 15)]
        prep_time: u32,

        #[llm(description = "Difficulty level (1-5)", example = 3)]
        difficulty: u8,

        #[llm(description = "Ingredients list", example = ["flour", "sugar", "eggs"])]
        ingredients: Vec<String>,
    }

    impl RecipeInfo {
        fn validate(&self) -> rstructor::Result<()> {
            // Recipe name can't be empty
            if self.name.trim().is_empty() {
                return Err(RStructorError::ValidationError(
                    "Recipe name cannot be empty".to_string(),
                ));
            }

            // Cooking time should be reasonable
            if self.cooking_time > 24 * 60 {
                // More than 24 hours
                return Err(RStructorError::ValidationError(format!(
                    "Cooking time should be less than 24 hours, got {} minutes",
                    self.cooking_time
                )));
            }

            // Difficulty must be 1-5
            if self.difficulty < 1 || self.difficulty > 5 {
                return Err(RStructorError::ValidationError(format!(
                    "Difficulty must be between 1 and 5, got {}",
                    self.difficulty
                )));
            }

            // Must have at least one ingredient
            if self.ingredients.is_empty() {
                return Err(RStructorError::ValidationError(
                    "Recipe must have at least one ingredient".to_string(),
                ));
            }

            Ok(())
        }
    }

    // Test validation failure with impossible cooking time
    #[tokio::test]
    async fn test_openai_validation_fails_with_long_cooking_time() {
        // For this test, we'll manually create a RecipeInfo with an invalid cooking time

        // Create recipe with excessive cooking time
        let invalid_recipe = RecipeInfo {
            name: "Slow Cooked Stew".to_string(),
            cooking_time: 10000, // 10,000 minutes = ~7 days
            prep_time: 30,
            difficulty: 3,
            ingredients: vec![
                "beef".to_string(),
                "vegetables".to_string(),
                "broth".to_string(),
            ],
        };

        // Validate it - should fail with cooking time error
        let validation_result = invalid_recipe.validate();

        // Check that validation failed
        assert!(
            validation_result.is_err(),
            "Validation should fail with extreme cooking time"
        );

        // Check that the error is about cooking time
        if let Err(RStructorError::ValidationError(msg)) = validation_result {
            assert!(
                msg.contains("Cooking time") || msg.contains("cooking time"),
                "Error should mention cooking time: {}",
                msg
            );
        } else if let Err(e) = validation_result {
            panic!("Expected ValidationError about cooking time, got: {:?}", e);
        }
    }

    // Test validation failure with invalid difficulty
    #[tokio::test]
    async fn test_openai_validation_fails_with_invalid_difficulty() {
        // For this test, we'll manually create a RecipeInfo with an invalid difficulty

        // Create recipe with difficulty out of range
        let invalid_recipe = RecipeInfo {
            name: "Extremely Difficult Soufflé".to_string(),
            cooking_time: 45,
            prep_time: 60,
            difficulty: 10, // Difficulty should be 1-5
            ingredients: vec![
                "eggs".to_string(),
                "cheese".to_string(),
                "flour".to_string(),
            ],
        };

        // Validate it - should fail with difficulty error
        let validation_result = invalid_recipe.validate();

        // Check that validation failed
        assert!(
            validation_result.is_err(),
            "Validation should fail with extreme difficulty"
        );

        // Check that the error is about difficulty
        if let Err(RStructorError::ValidationError(msg)) = validation_result {
            assert!(
                msg.contains("Difficulty") || msg.contains("difficulty"),
                "Error should mention difficulty: {}",
                msg
            );
        } else if let Err(e) = validation_result {
            panic!("Expected ValidationError about difficulty, got: {:?}", e);
        }
    }

    // Test successful validation
    #[tokio::test]
    async fn test_openai_validation_succeeds_with_valid_data() {
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var not set");

        let client = OpenAIClient::new(api_key)
            .expect("Failed to create OpenAI client")
            .model(OpenAIModel::Gpt4O)
            .temperature(0.0); // Deterministic for consistent results

        // Normal prompt for a typical recipe
        let prompt = "Give me a recipe for chocolate chip cookies.";

        // Should pass validation
        let result = client.generate_struct::<RecipeInfo>(prompt).await;

        // Validation should pass
        assert!(result.is_ok(), "Validation failed: {:?}", result.err());

        // Check that the data looks reasonable
        let recipe = result.unwrap();
        assert!(!recipe.name.is_empty());
        assert!(recipe.cooking_time > 0 && recipe.cooking_time < 24 * 60);
        assert!(recipe.difficulty >= 1 && recipe.difficulty <= 5);
        assert!(!recipe.ingredients.is_empty());

        // Print the recipe for inspection
        println!("Recipe: {:?}", recipe);
    }
}

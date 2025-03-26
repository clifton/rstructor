//! This test file demonstrates validation with the OpenAI client.
//! 
//! It's marked with #[ignore] by default as it requires an API key.
//! To run this test specifically:
//! 
//! ```bash
//! export OPENAI_API_KEY=your_key_here
//! cargo test --test openai_validation_test -- --ignored
//! ```

#[cfg(test)]
mod openai_validation_tests {
    use rstructor::{LLMModel, OpenAIClient, OpenAIModel, RStructorError};
    use serde::{Serialize, Deserialize};
    use std::env;
    
    // Define a data model with validation rules
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
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
                    "Recipe name cannot be empty".to_string()
                ));
            }
            
            // Cooking time should be reasonable
            if self.cooking_time > 24 * 60 {  // More than 24 hours
                return Err(RStructorError::ValidationError(
                    format!("Cooking time should be less than 24 hours, got {} minutes", self.cooking_time)
                ));
            }
            
            // Difficulty must be 1-5
            if self.difficulty < 1 || self.difficulty > 5 {
                return Err(RStructorError::ValidationError(
                    format!("Difficulty must be between 1 and 5, got {}", self.difficulty)
                ));
            }
            
            // Must have at least one ingredient
            if self.ingredients.is_empty() {
                return Err(RStructorError::ValidationError(
                    "Recipe must have at least one ingredient".to_string()
                ));
            }
            
            Ok(())
        }
    }
    
    // Test validation failure with impossible cooking time
    #[tokio::test]
    #[ignore] // Requires API key
    async fn test_openai_validation_fails_with_long_cooking_time() {
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var not set");
        
        let client = OpenAIClient::new(api_key)
            .expect("Failed to create OpenAI client")
            .model(OpenAIModel::Gpt35Turbo)
            .temperature(0.7) // Higher temperature for more creative responses
            .build();
        
        // This prompt is designed to encourage an extremely long cooking time
        let prompt = "Give me a recipe that requires extremely long cooking time, something that cooks for multiple days.";
        
        // Should fail validation
        let result = client.generate_struct::<RecipeInfo>(prompt).await;
        
        // Check that we got a validation error
        assert!(result.is_err());
        
        // Make sure it's specifically a validation error about cooking time
        if let Err(RStructorError::ValidationError(msg)) = result {
            assert!(msg.contains("Cooking time"), "Error should mention cooking time: {}", msg);
        } else if let Err(e) = result {
            panic!("Expected ValidationError about cooking time, got: {:?}", e);
        }
    }
    
    // Test validation failure with invalid difficulty
    #[tokio::test]
    #[ignore] // Requires API key
    async fn test_openai_validation_fails_with_invalid_difficulty() {
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var not set");
        
        let client = OpenAIClient::new(api_key)
            .expect("Failed to create OpenAI client")
            .model(OpenAIModel::Gpt35Turbo)
            .temperature(0.7) // Higher temperature for more creative responses
            .build();
        
        // This prompt is designed to encourage an out-of-range difficulty
        let prompt = "Give me a recipe with a difficulty level of 10 out of 10, the most difficult recipe possible.";
        
        // Should fail validation
        let result = client.generate_struct::<RecipeInfo>(prompt).await;
        
        // Check that we got a validation error
        assert!(result.is_err());
        
        // Make sure it's specifically a validation error about difficulty
        if let Err(RStructorError::ValidationError(msg)) = result {
            assert!(msg.contains("Difficulty"), "Error should mention difficulty: {}", msg);
        } else if let Err(e) = result {
            panic!("Expected ValidationError about difficulty, got: {:?}", e);
        }
    }
    
    // Test successful validation
    #[tokio::test]
    #[ignore] // Requires API key
    async fn test_openai_validation_succeeds_with_valid_data() {
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var not set");
        
        let client = OpenAIClient::new(api_key)
            .expect("Failed to create OpenAI client")
            .model(OpenAIModel::Gpt35Turbo)
            .temperature(0.0) // Deterministic for consistent results
            .build();
        
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
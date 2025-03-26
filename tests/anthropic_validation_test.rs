//! This test file demonstrates validation with the Anthropic client.
//! 
//! It's marked with #[ignore] by default as it requires an API key.
//! To run this test specifically:
//! 
//! ```bash
//! export ANTHROPIC_API_KEY=your_key_here
//! cargo test --test anthropic_validation_test -- --ignored
//! ```

#[cfg(test)]
mod anthropic_validation_tests {
    use rstructor::{LLMModel, AnthropicClient, AnthropicModel, RStructorError};
    use serde::{Serialize, Deserialize};
    use std::env;
    
    // Define a data model with validation rules
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    #[llm(description = "Information about a city's weather")]
    struct WeatherInfo {
        #[llm(description = "Name of the city")]
        city: String,
        
        #[llm(description = "Current temperature in Celsius", example = 22.5)]
        temperature: f32,
        
        #[llm(description = "Weather condition", example = "Sunny")]
        condition: String,
        
        #[llm(description = "Humidity percentage", example = 65)]
        humidity: u8,
    }
    
    impl WeatherInfo {
        fn validate(&self) -> rstructor::Result<()> {
            // City name can't be empty
            if self.city.trim().is_empty() {
                return Err(RStructorError::ValidationError(
                    "City name cannot be empty".to_string()
                ));
            }
            
            // Temperature must be in a reasonable range
            if self.temperature < -100.0 || self.temperature > 60.0 {
                return Err(RStructorError::ValidationError(
                    format!("Temperature must be between -100°C and 60°C, got {}°C", self.temperature)
                ));
            }
            
            // Humidity must be a percentage
            if self.humidity > 100 {
                return Err(RStructorError::ValidationError(
                    format!("Humidity must be between 0 and 100%, got {}%", self.humidity)
                ));
            }
            
            Ok(())
        }
    }
    
    // Invalid temperature test
    #[tokio::test]
    #[ignore] // Requires API key
    async fn test_anthropic_validation_fails_with_extreme_temperature() {
        // This test demonstrates validation catching an invalid temperature
        let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY env var not set");
        
        let client = AnthropicClient::new(api_key)
            .expect("Failed to create Anthropic client")
            .model(AnthropicModel::Claude3Haiku)
            .temperature(0.7) // Using higher temperature to encourage creative responses
            .build();
        
        // This prompt is designed to encourage an unrealistic temperature
        let prompt = "Describe the weather on a very extreme day on Venus with temperatures in Celsius. Make it truly extreme.";
        
        // Should fail validation
        let result = client.generate_struct::<WeatherInfo>(prompt).await;
        
        // Check that we got a validation error
        assert!(result.is_err());
        
        // Make sure it's specifically a validation error about temperature
        if let Err(RStructorError::ValidationError(msg)) = result {
            assert!(msg.contains("Temperature"), "Error should mention temperature: {}", msg);
        } else if let Err(e) = result {
            panic!("Expected ValidationError about temperature, got: {:?}", e);
        }
    }
    
    // Invalid humidity test
    #[tokio::test]
    #[ignore] // Requires API key
    async fn test_anthropic_validation_fails_with_invalid_humidity() {
        // This test demonstrates validation catching an invalid humidity
        let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY env var not set");
        
        let client = AnthropicClient::new(api_key)
            .expect("Failed to create Anthropic client")
            .model(AnthropicModel::Claude3Haiku)
            .temperature(0.7) // Using higher temperature to encourage creative responses
            .build();
        
        // This prompt is designed to encourage an impossible humidity value
        let prompt = "Describe the weather in a sci-fi world with impossible extreme humidity values well over 100%.";
        
        // Should fail validation
        let result = client.generate_struct::<WeatherInfo>(prompt).await;
        
        // Check that we got a validation error
        assert!(result.is_err());
        
        // Make sure it's specifically a validation error about humidity
        if let Err(RStructorError::ValidationError(msg)) = result {
            assert!(msg.contains("Humidity"), "Error should mention humidity: {}", msg);
        } else if let Err(e) = result {
            panic!("Expected ValidationError about humidity, got: {:?}", e);
        }
    }
    
    // Valid data test
    #[tokio::test]
    #[ignore] // Requires API key
    async fn test_anthropic_validation_succeeds_with_valid_data() {
        // This test demonstrates successful validation with reasonable data
        let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY env var not set");
        
        let client = AnthropicClient::new(api_key)
            .expect("Failed to create Anthropic client")
            .model(AnthropicModel::Claude3Haiku)
            .temperature(0.0) // Use deterministic temperature for consistent results
            .build();
        
        // Normal prompt asking for weather in a real city
        let prompt = "What's the weather like in Paris today? Use realistic values.";
        
        // Should succeed validation
        let result = client.generate_struct::<WeatherInfo>(prompt).await;
        
        // Validation should pass
        assert!(result.is_ok(), "Validation failed: {:?}", result.err());
        
        // Check the data looks reasonable
        let weather = result.unwrap();
        assert_eq!(weather.city, "Paris");
        assert!(weather.temperature >= -30.0 && weather.temperature <= 45.0);
        assert!(weather.humidity <= 100);
        assert!(!weather.condition.is_empty());
    }
}
//! This test file demonstrates validation with the Anthropic client.
//!
//! These tests require a valid Anthropic API key in the environment:
//!
//! ```bash
//! export ANTHROPIC_API_KEY=your_key_here
//! cargo test --test anthropic_validation_test
//! ```

#[cfg(test)]
mod anthropic_validation_tests {
    use rstructor::{AnthropicClient, AnthropicModel, Instructor, LLMClient, RStructorError};
    use serde::{Deserialize, Serialize};
    use std::env;

    // Define a data model with validation rules
    #[derive(Instructor, Serialize, Deserialize, Debug)]
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
                    "City name cannot be empty".to_string(),
                ));
            }

            // Temperature must be in a reasonable range
            if self.temperature < -100.0 || self.temperature > 60.0 {
                return Err(RStructorError::ValidationError(format!(
                    "Temperature must be between -100°C and 60°C, got {}°C",
                    self.temperature
                )));
            }

            // Humidity must be a percentage
            if self.humidity > 100 {
                return Err(RStructorError::ValidationError(format!(
                    "Humidity must be between 0 and 100%, got {}%",
                    self.humidity
                )));
            }

            Ok(())
        }
    }

    // Test validation catching an invalid temperature
    #[tokio::test]
    async fn test_anthropic_validation_fails_with_extreme_temperature() {
        // For this test, we'll get a valid weather response but then manually create one with invalid temperature
        let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY env var not set");

        let client = AnthropicClient::new(api_key)
            .expect("Failed to create Anthropic client")
            .model(AnthropicModel::Claude35Sonnet)
            .temperature(0.0) // Use deterministic output
            .build();

        // First get a valid weather response
        let prompt = "What is the current weather in New York City?";
        let valid_result = client.generate_struct::<WeatherInfo>(prompt).await;

        // Make sure we can get a valid response
        assert!(
            valid_result.is_ok(),
            "Should be able to get valid weather data"
        );

        // Now create a new WeatherInfo with an invalid temperature
        let invalid_weather = WeatherInfo {
            city: "Temperature Test City".to_string(),
            temperature: 999.0, // Way outside the valid range
            condition: "Extreme Heat".to_string(),
            humidity: 50, // Valid humidity
        };

        // Validate it - should fail with temperature error
        let validation_result = invalid_weather.validate();

        // Check that validation failed
        assert!(
            validation_result.is_err(),
            "Validation should fail with extreme temperature"
        );

        // Check that the error is about temperature
        if let Err(RStructorError::ValidationError(msg)) = validation_result {
            assert!(
                msg.contains("Temperature") || msg.contains("temperature"),
                "Error should mention temperature: {}",
                msg
            );
        } else if let Err(e) = validation_result {
            panic!("Expected ValidationError about temperature, got: {:?}", e);
        }
    }

    // Test validation catching an invalid humidity
    #[tokio::test]
    async fn test_anthropic_validation_fails_with_invalid_humidity() {
        // For this test, we'll manually create a WeatherInfo with invalid humidity
        // We don't even need to make an API call for this test

        // Create a weather info with an invalid humidity (over 100%)
        let invalid_weather = WeatherInfo {
            city: "Humidity Test City".to_string(),
            temperature: 25.0, // Valid temperature
            condition: "Rainy".to_string(),
            humidity: 150, // Invalid humidity (over 100%)
        };

        // Validate it - should fail with humidity error
        let validation_result = invalid_weather.validate();

        // Check that validation failed
        assert!(
            validation_result.is_err(),
            "Validation should fail with extreme humidity"
        );

        // Check that the error is about humidity
        if let Err(RStructorError::ValidationError(msg)) = validation_result {
            assert!(
                msg.contains("Humidity") || msg.contains("humidity"),
                "Error should mention humidity: {}",
                msg
            );
        } else if let Err(e) = validation_result {
            panic!("Expected ValidationError about humidity, got: {:?}", e);
        }
    }

    // Valid data test
    #[tokio::test]
    async fn test_anthropic_validation_succeeds_with_valid_data() {
        // This test demonstrates successful validation with reasonable data
        let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY env var not set");

        let client = AnthropicClient::new(api_key)
            .expect("Failed to create Anthropic client")
            .model(AnthropicModel::Claude35Sonnet)
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

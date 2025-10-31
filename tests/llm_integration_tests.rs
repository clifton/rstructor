//! Integration tests for LLM functionality.
//!
//! These tests require valid API keys in the environment:
//!
//! ```bash
//! export OPENAI_API_KEY=your_key_here
//! export ANTHROPIC_API_KEY=your_key_here
//! export XAI_API_KEY=your_key_here
//! export GEMINI_API_KEY=your_key_here
//! cargo test --test llm_integration_tests
//! ```

#[cfg(test)]
mod llm_integration_tests {
    #[cfg(feature = "anthropic")]
    use rstructor::{AnthropicClient, AnthropicModel};
    #[cfg(feature = "gemini")]
    use rstructor::{GeminiClient, GeminiModel};
    #[cfg(feature = "grok")]
    use rstructor::{GrokClient, GrokModel};
    use rstructor::{Instructor, LLMClient, SchemaType};
    #[cfg(feature = "openai")]
    use rstructor::{OpenAIClient, OpenAIModel};
    use serde::{Deserialize, Serialize};
    use std::env;

    // Simple model for testing
    #[derive(Instructor, Serialize, Deserialize, Debug)]
    #[llm(description = "Information about a movie")]
    struct Movie {
        #[llm(description = "Title of the movie")]
        title: String,

        #[llm(description = "Year the movie was released", example = 2010)]
        year: u16,

        #[llm(description = "Director of the movie", example = "Christopher Nolan")]
        director: String,

        #[llm(description = "Main actors in the movie",
              example = ["Leonardo DiCaprio", "Ellen Page"])]
        actors: Vec<String>,

        #[llm(description = "Brief plot summary")]
        plot: String,
    }

    // Custom validation implementation
    impl Movie {
        #[allow(dead_code)]
        fn validate(&self) -> rstructor::Result<()> {
            // Validate year is reasonable
            if self.year < 1888 || self.year > 2030 {
                return Err(rstructor::RStructorError::ValidationError(format!(
                    "Movie year must be between 1888 and 2030, got {}",
                    self.year
                )));
            }

            // Validate we have at least one actor
            if self.actors.is_empty() {
                return Err(rstructor::RStructorError::ValidationError(
                    "Movie must have at least one actor".to_string(),
                ));
            }

            Ok(())
        }
    }

    // Test using OpenAI
    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn test_openai_generate_struct() {
        // Skip test if API key is not available
        let api_key = match env::var("OPENAI_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("Skipping test: OPENAI_API_KEY not set");
                return;
            }
        };

        let client = match OpenAIClient::new(api_key) {
            Ok(client) => client.model(OpenAIModel::Gpt4O).temperature(0.0),
            Err(e) => {
                println!("Skipping test: Failed to create OpenAI client: {:?}", e);
                return;
            }
        };

        let prompt = "Provide information about the movie Inception";
        let movie_result = client.generate_struct::<Movie>(prompt).await;

        // Handle API errors gracefully
        if let Err(e) = &movie_result {
            println!("Skipping test due to API error: {:?}", e);
            return;
        }

        // Only validate when we have a successful response
        let movie = movie_result.expect("Failed to get movie info");

        // Validate response
        assert_eq!(movie.title, "Inception");
        assert_eq!(movie.year, 2010);
        assert_eq!(movie.director, "Christopher Nolan");
        assert!(!movie.actors.is_empty());
        assert!(movie.plot.len() > 10);

        println!("OpenAI response: {:#?}", movie);
    }

    // Test using Anthropic
    #[cfg(feature = "anthropic")]
    #[tokio::test]
    async fn test_anthropic_generate_struct() {
        // Skip test if API key is not available
        let api_key = match env::var("ANTHROPIC_API_KEY") {
            Ok(key) => key,
            Err(_) => {
                println!("Skipping test: ANTHROPIC_API_KEY not set");
                return;
            }
        };

        let client = match AnthropicClient::new(api_key) {
            Ok(client) => client
                .model(AnthropicModel::ClaudeSonnet45)
                .temperature(0.0),
            Err(e) => {
                println!("Skipping test: Failed to create Anthropic client: {:?}", e);
                return;
            }
        };

        let prompt = "Provide information about the movie Inception";
        let movie_result = client.generate_struct::<Movie>(prompt).await;

        // Handle API errors gracefully
        if let Err(e) = &movie_result {
            println!("Skipping test due to API error: {:?}", e);
            return;
        }

        // Only validate when we have a successful response
        let movie = movie_result.expect("Failed to get movie info");

        // Validate response
        assert_eq!(movie.title, "Inception");
        assert_eq!(movie.year, 2010);
        assert_eq!(movie.director, "Christopher Nolan");
        assert!(!movie.actors.is_empty());
        assert!(movie.plot.len() > 10);

        println!("Anthropic response: {:#?}", movie);
    }

    // Test using Grok
    #[cfg(feature = "grok")]
    #[tokio::test]
    async fn test_grok_generate_struct() {
        // Skip test if API key is not available
        // Read from XAI_API_KEY env var
        let client = match GrokClient::from_env() {
            Ok(client) => client.model(GrokModel::Grok4).temperature(0.0),
            Err(e) => {
                println!(
                    "Skipping test: Failed to create Grok client (XAI_API_KEY not set): {:?}",
                    e
                );
                return;
            }
        };

        let prompt = "Provide information about the movie Inception";
        let movie_result = client.generate_struct::<Movie>(prompt).await;

        // Handle API errors gracefully
        if let Err(e) = &movie_result {
            println!("Skipping test due to API error: {:?}", e);
            return;
        }

        // Only validate when we have a successful response
        let movie = movie_result.expect("Failed to get movie info");

        // Validate response
        assert_eq!(movie.title, "Inception");
        assert_eq!(movie.year, 2010);
        assert_eq!(movie.director, "Christopher Nolan");
        assert!(!movie.actors.is_empty());
        assert!(movie.plot.len() > 10);

        println!("Grok response: {:#?}", movie);
    }

    // Test using Gemini
    #[cfg(feature = "gemini")]
    #[tokio::test]
    async fn test_gemini_generate_struct() {
        // Skip test if API key is not available
        // Read from GEMINI_API_KEY env var
        let client = match GeminiClient::from_env() {
            Ok(client) => client.model(GeminiModel::Gemini25Flash).temperature(0.0),
            Err(e) => {
                println!(
                    "Skipping test: Failed to create Gemini client (GEMINI_API_KEY not set): {:?}",
                    e
                );
                return;
            }
        };

        let prompt = "Provide information about the movie Inception";
        let movie_result = client.generate_struct::<Movie>(prompt).await;

        // Handle API errors gracefully
        if let Err(e) = &movie_result {
            println!("Skipping test due to API error: {:?}", e);
            return;
        }

        // Only validate when we have a successful response
        let movie = movie_result.expect("Failed to get movie info");

        // Validate response
        assert_eq!(movie.title, "Inception");
        assert_eq!(movie.year, 2010);
        assert_eq!(movie.director, "Christopher Nolan");
        assert!(!movie.actors.is_empty());
        assert!(movie.plot.len() > 10);

        println!("Gemini response: {:#?}", movie);
    }

    // Test to ensure schema is generated correctly
    #[test]
    fn test_movie_schema_generation() {
        let schema = Movie::schema();
        let schema_json = schema.to_json();

        // Check basic schema properties
        assert_eq!(schema_json["type"], "object");
        assert_eq!(schema_json["title"], "Movie");
        assert_eq!(schema_json["description"], "Information about a movie");

        // Check that all fields are in the schema
        let properties = schema_json["properties"].as_object().unwrap();
        assert!(properties.contains_key("title"));
        assert!(properties.contains_key("year"));
        assert!(properties.contains_key("director"));
        assert!(properties.contains_key("actors"));
        assert!(properties.contains_key("plot"));

        // Check field descriptions
        assert_eq!(properties["title"]["description"], "Title of the movie");
        assert_eq!(
            properties["year"]["description"],
            "Year the movie was released"
        );

        // Check examples
        assert_eq!(properties["year"]["example"], 2010);

        // Check required fields
        let required = schema_json["required"].as_array().unwrap();
        assert_eq!(required.len(), 5); // All fields are required
    }
}

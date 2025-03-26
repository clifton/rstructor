use async_trait::async_trait;
use serde::de::DeserializeOwned;

use crate::error::Result;
use crate::model::LLMModel;

/// LLMClient trait defines the interface for all LLM API clients.
///
/// This trait is the core abstraction for interacting with different LLM providers
/// like OpenAI or Anthropic. It provides methods for generating structured data
/// and raw text completions.
///
/// The library includes implementations for popular LLM providers:
/// - `OpenAIClient` for OpenAI's GPT models (gpt-3.5-turbo, gpt-4, etc.)
/// - `AnthropicClient` for Anthropic's Claude models
///
/// # Examples
///
/// Using OpenAI client:
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use rstructor::{LLMClient, LLMModel, OpenAIClient, OpenAIModel};
/// use serde::{Serialize, Deserialize};
///
/// // Define your data model
/// #[derive(LLMModel, Serialize, Deserialize, Debug)]
/// struct Movie {
///     title: String,
///     director: String,
///     year: u16,
/// }
///
/// // Create a client
/// let client = OpenAIClient::new("your-openai-api-key")?
///     .model(OpenAIModel::Gpt35Turbo)
///     .temperature(0.0)
///     .build();
///
/// // Generate a structured response
/// let prompt = "Describe the movie Inception";
/// let movie: Movie = client.generate_struct(prompt).await?;
///
/// println!("Title: {}", movie.title);
/// println!("Director: {}", movie.director);
/// println!("Year: {}", movie.year);
/// # Ok(())
/// # }
/// ```
///
/// Using Anthropic client:
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use rstructor::{LLMClient, LLMModel, AnthropicClient, AnthropicModel};
/// use serde::{Serialize, Deserialize};
///
/// // Define your data model
/// #[derive(LLMModel, Serialize, Deserialize, Debug)]
/// struct MovieReview {
///     movie_title: String,
///     rating: f32,
///     review: String,
/// }
///
/// // Create a client
/// let client = AnthropicClient::new("your-anthropic-api-key")?
///     .model(AnthropicModel::Claude3Haiku)
///     .temperature(0.0)
///     .build();
///
/// // Generate a structured response
/// let prompt = "Write a short review of the movie The Matrix";
/// let review: MovieReview = client.generate_struct(prompt).await?;
///
/// println!("Movie: {}", review.movie_title);
/// println!("Rating: {}/10", review.rating);
/// println!("Review: {}", review.review);
/// # Ok(())
/// # }
/// ```
#[async_trait]
pub trait LLMClient {
    /// Generate a structured object of type T from a prompt.
    ///
    /// This method takes a text prompt and returns a structured object
    /// of type T, where T implements the `LLMModel` trait. The LLM is guided
    /// to produce output that conforms to the JSON schema defined by T.
    ///
    /// If the returned data doesn't match the expected schema or fails validation,
    /// an error is returned.
    async fn generate_struct<T>(&self, prompt: &str) -> Result<T>
    where
        T: LLMModel + DeserializeOwned + Send + 'static;

    /// Raw completion without structure (returns plain text).
    ///
    /// This method provides a simpler interface for getting raw text completions
    /// from the LLM without enforcing any structure.
    async fn generate(&self, prompt: &str) -> Result<String>;
}

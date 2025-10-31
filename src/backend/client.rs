use async_trait::async_trait;
use serde::de::DeserializeOwned;

use crate::error::Result;
use crate::model::Instructor;

/// LLMClient trait defines the interface for all LLM API clients.
///
/// This trait is the core abstraction for interacting with different LLM providers
/// like OpenAI or Anthropic. It provides methods for generating structured data
/// and raw text completions.
///
/// The library includes implementations for popular LLM providers:
/// - `OpenAIClient` for OpenAI's GPT models (gpt-3.5-turbo, gpt-4, etc.)
/// - `AnthropicClient` for Anthropic's Claude models
/// - `GrokClient` for xAI's Grok models
/// - `GeminiClient` for Google's Gemini models
///
/// All clients implement a consistent interface:
/// - `new(api_key)` - Create client with explicit API key (rejects empty strings)
/// - `from_env()` - Create client from environment variable (required by this trait):
///   - OpenAI: `OPENAI_API_KEY`
///   - Anthropic: `ANTHROPIC_API_KEY`
///   - Grok: `XAI_API_KEY`
///   - Gemini: `GEMINI_API_KEY`
/// - Builder methods: `model()`, `temperature()`, `max_tokens()`, `timeout()`
/// - All clients validate `max_tokens >= 1` to avoid API errors
/// - Timeout is applied immediately when `timeout()` is called - no need to call `build()`
///
/// # Examples
///
/// Using OpenAI client:
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use rstructor::{LLMClient, Instructor, OpenAIClient, OpenAIModel};
/// use serde::{Serialize, Deserialize};
/// use std::time::Duration;
///
/// // Define your data model
/// #[derive(Instructor, Serialize, Deserialize, Debug)]
/// struct Movie {
///     title: String,
///     director: String,
///     year: u16,
/// }
///
/// // Create a client
/// let client = OpenAIClient::new("your-openai-api-key")?
///     .model(OpenAIModel::Gpt4OMini)
///     .temperature(0.0)
///     .timeout(Duration::from_secs(30));  // Optional: set 30 second timeout
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
/// use rstructor::{LLMClient, Instructor, AnthropicClient, AnthropicModel};
/// use serde::{Serialize, Deserialize};
/// use std::time::Duration;
///
/// // Define your data model
/// #[derive(Instructor, Serialize, Deserialize, Debug)]
/// struct MovieReview {
///     movie_title: String,
///     rating: f32,
///     review: String,
/// }
///
/// // Create a client
/// let client = AnthropicClient::new("your-anthropic-api-key")?
///     .model(AnthropicModel::ClaudeSonnet4)
///     .temperature(0.0)
///     .timeout(Duration::from_secs(30));  // Optional: set 30 second timeout
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
    /// of type T, where T implements the `Instructor` trait. The LLM is guided
    /// to produce output that conforms to the JSON schema defined by T.
    ///
    /// If the returned data doesn't match the expected schema or fails validation,
    /// an error is returned.
    async fn generate_struct<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static;

    /// Generate a structured object with automatic retry for validation errors.
    ///
    /// **Deprecated**: Use `generate_struct()` with client retry configuration instead.
    /// Set retry options using `.max_retries()` and `.include_error_feedback()` builder methods.
    ///
    /// This method is kept for backward compatibility but delegates to `generate_struct()`
    /// with temporary retry configuration.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// use rstructor::{LLMClient, OpenAIClient, OpenAIModel, Instructor};
    /// use serde::{Serialize, Deserialize};
    ///
    /// #[derive(Instructor, Serialize, Deserialize, Debug)]
    /// struct Recipe {
    ///     name: String,
    ///     ingredients: Vec<String>,
    ///     steps: Vec<String>,
    /// }
    ///
    /// // Recommended: Use client configuration
    /// let client = OpenAIClient::new("your-api-key")?
    ///     .model(OpenAIModel::Gpt4O)
    ///     .temperature(0.0)
    ///     .max_retries(3)
    ///     .include_error_feedback(true);
    ///
    /// let recipe = client.generate_struct::<Recipe>("Give me a chocolate cake recipe").await?;
    ///
    /// println!("Recipe: {}", recipe.name);
    /// # Ok(())
    /// # }
    /// ```
    #[deprecated(
        since = "0.2.0",
        note = "Use generate_struct() with client retry configuration instead. Set retry options using .max_retries() and .include_error_feedback() builder methods."
    )]
    async fn generate_struct_with_retry<T>(
        &self,
        prompt: &str,
        max_retries: Option<usize>,
        include_error_feedback: Option<bool>,
    ) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static;

    /// Raw completion without structure (returns plain text).
    ///
    /// This method provides a simpler interface for getting raw text completions
    /// from the LLM without enforcing any structure.
    async fn generate(&self, prompt: &str) -> Result<String>;

    /// Create a new client by reading the API key from an environment variable.
    ///
    /// This is a required associated function that all `LLMClient` implementations must provide.
    /// The specific environment variable name depends on the provider:
    /// - OpenAI: `OPENAI_API_KEY`
    /// - Anthropic: `ANTHROPIC_API_KEY`
    /// - Grok: `XAI_API_KEY`
    /// - Gemini: `GEMINI_API_KEY`
    ///
    /// # Errors
    ///
    /// Returns an error if the required environment variable is not set.
    fn from_env() -> Result<Self>
    where
        Self: Sized;
}

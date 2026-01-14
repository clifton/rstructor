use async_trait::async_trait;
use serde::de::DeserializeOwned;

use crate::backend::ModelInfo;
use crate::backend::usage::{GenerateResult, MaterializeResult};
use crate::error::Result;
use crate::model::Instructor;

/// File reference for media-aware prompts (e.g., Gemini file URI + MIME type).
#[derive(Debug, Clone)]
pub struct MediaFile {
    pub uri: String,
    pub mime_type: String,
}

impl MediaFile {
    #[must_use]
    pub fn new(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            mime_type: mime_type.into(),
        }
    }
}

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
/// // Materialize a structured response
/// let prompt = "Describe the movie Inception";
/// let movie: Movie = client.materialize(prompt).await?;
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
/// // Materialize a structured response
/// let prompt = "Write a short review of the movie The Matrix";
/// let review: MovieReview = client.materialize(prompt).await?;
///
/// println!("Movie: {}", review.movie_title);
/// println!("Rating: {}/10", review.rating);
/// println!("Review: {}", review.review);
/// # Ok(())
/// # }
/// ```
#[async_trait]
pub trait LLMClient {
    /// Materialize a structured object of type T from a prompt.
    ///
    /// This method takes a text prompt and returns the structured object.
    /// The LLM is guided to produce output that conforms to the JSON schema defined by T.
    /// If the returned data doesn't match the expected schema or fails validation,
    /// the client will automatically retry up to 3 times (configurable via `.max_retries()`
    /// or disabled via `.no_retries()`).
    ///
    /// For token usage information, use [`materialize_with_metadata`](Self::materialize_with_metadata).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rstructor::{LLMClient, OpenAIClient, Instructor};
    /// # use serde::{Serialize, Deserialize};
    /// # #[derive(Instructor, Serialize, Deserialize)]
    /// # struct Movie { title: String }
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenAIClient::from_env()?;
    /// let movie: Movie = client.materialize("Describe Inception").await?;
    /// println!("Title: {}", movie.title);
    /// # Ok(())
    /// # }
    /// ```
    async fn materialize<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static;

    /// Materialize a structured object with media references (if supported).
    ///
    /// Providers that do not support media inputs ignore the `media` parameter.
    async fn materialize_with_media<T>(&self, prompt: &str, _media: &[MediaFile]) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        self.materialize(prompt).await
    }

    /// Materialize a structured object with metadata (token usage).
    ///
    /// Like [`materialize`](Self::materialize), but returns a [`MaterializeResult<T>`]
    /// that includes token usage information for monitoring and cost tracking.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rstructor::{LLMClient, OpenAIClient, Instructor};
    /// # use serde::{Serialize, Deserialize};
    /// # #[derive(Instructor, Serialize, Deserialize)]
    /// # struct Movie { title: String }
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenAIClient::from_env()?;
    /// let result = client.materialize_with_metadata::<Movie>("Describe Inception").await?;
    ///
    /// println!("Title: {}", result.data.title);
    /// if let Some(usage) = result.usage {
    ///     println!("Model: {}", usage.model);
    ///     println!("Tokens: {} in, {} out", usage.input_tokens, usage.output_tokens);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    async fn materialize_with_metadata<T>(&self, prompt: &str) -> Result<MaterializeResult<T>>
    where
        T: Instructor + DeserializeOwned + Send + 'static;

    /// Raw completion without structure (returns plain text).
    ///
    /// This method provides a simpler interface for getting raw text completions
    /// from the LLM without enforcing any structure.
    ///
    /// For token usage information, use [`generate_with_metadata`](Self::generate_with_metadata).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rstructor::{LLMClient, OpenAIClient};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenAIClient::from_env()?;
    /// let text = client.generate("Write a haiku").await?;
    /// println!("{}", text);
    /// # Ok(())
    /// # }
    /// ```
    async fn generate(&self, prompt: &str) -> Result<String>;

    /// Raw completion with metadata (token usage).
    ///
    /// Like [`generate`](Self::generate), but returns a [`GenerateResult`]
    /// that includes token usage information.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rstructor::{LLMClient, OpenAIClient};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenAIClient::from_env()?;
    /// let result = client.generate_with_metadata("Write a haiku").await?;
    ///
    /// println!("{}", result.text);
    /// if let Some(usage) = result.usage {
    ///     println!("Used {} total tokens", usage.total_tokens());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    async fn generate_with_metadata(&self, prompt: &str) -> Result<GenerateResult>;

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

    /// Fetch available models from the provider's API.
    ///
    /// This method queries the provider's models endpoint to return a list of
    /// models available for use. The results are filtered to include only
    /// chat/completion models relevant to this library.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rstructor::{LLMClient, OpenAIClient};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenAIClient::from_env()?;
    /// let models = client.list_models().await?;
    ///
    /// println!("Available models:");
    /// for model in models {
    ///     println!("  - {}", model.id);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
}

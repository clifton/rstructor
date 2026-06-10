use async_trait::async_trait;
use serde::de::DeserializeOwned;

use crate::backend::ModelInfo;
use crate::backend::usage::{GenerateResult, MaterializeResult};
use crate::error::Result;
use crate::model::Instructor;

/// File reference for media-aware prompts (e.g., Gemini file URI or inline data).
///
/// `MediaFile` supports two modes:
///
/// - **URI-based**: Reference a file by URI (e.g., Google Cloud Storage or Gemini Files API).
///   Created with [`MediaFile::new`].
/// - **Inline data**: Embed base64-encoded file content directly in the request.
///   Created with [`MediaFile::from_bytes`]. This is useful for public images
///   downloaded over HTTPS.
///
/// The `mime_type` decides how each provider encodes the attachment: `image/*`
/// is sent in the provider's image format, and `application/pdf` is routed to
/// the provider's document/file format (OpenAI `file` part for inline data,
/// Anthropic `document` block, Gemini `inlineData`/`fileData`). Combinations a
/// provider does not document — e.g. any PDF on Grok, or a URL-based PDF on
/// OpenAI — produce a clear error instead of a silently broken request.
///
/// # Examples
///
/// ```no_run
/// use rstructor::MediaFile;
///
/// // URI-based (Gemini Files API or GCS)
/// let media = MediaFile::new(
///     "https://generativelanguage.googleapis.com/v1beta/files/abc123",
///     "image/png",
/// );
///
/// // Inline data from bytes
/// let image_bytes = std::fs::read("photo.png").unwrap();
/// let media = MediaFile::from_bytes(&image_bytes, "image/png");
///
/// // Inline PDF (OpenAI, Anthropic, and Gemini)
/// let pdf_bytes = std::fs::read("report.pdf").unwrap();
/// let media = MediaFile::from_bytes(&pdf_bytes, "application/pdf");
/// ```
#[derive(Debug, Clone)]
pub struct MediaFile {
    pub uri: String,
    pub mime_type: String,
    /// Base64-encoded inline data. When set, backends that support inline data
    /// will use this instead of the URI.
    pub data: Option<String>,
}

impl MediaFile {
    /// Create a URI-based media file reference.
    ///
    /// Use this for Gemini Files API URIs or Google Cloud Storage URIs.
    ///
    /// # Examples
    ///
    /// ```
    /// use rstructor::MediaFile;
    ///
    /// let media = MediaFile::new(
    ///     "https://generativelanguage.googleapis.com/v1beta/files/abc123",
    ///     "image/png",
    /// );
    /// assert!(media.data.is_none());
    /// ```
    #[must_use]
    pub fn new(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            mime_type: mime_type.into(),
            data: None,
        }
    }

    /// Create a media file from raw bytes, base64-encoding them for inline use.
    ///
    /// This is useful when you have image data in memory (e.g., downloaded from
    /// a public URL) and want to send it directly without uploading to the
    /// Gemini Files API first.
    ///
    /// # Examples
    ///
    /// ```
    /// use rstructor::MediaFile;
    ///
    /// let bytes = b"fake image data";
    /// let media = MediaFile::from_bytes(bytes, "image/png");
    /// assert!(media.data.is_some());
    /// assert!(media.uri.is_empty());
    /// ```
    #[cfg(feature = "_client")]
    #[must_use]
    pub fn from_bytes(data: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        use base64::Engine;
        let encoded = base64::engine::general_purpose::STANDARD.encode(data.as_ref());
        Self {
            uri: String::new(),
            mime_type: mime_type.into(),
            data: Some(encoded),
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
///     .model(OpenAIModel::Gpt55)
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
    /// The default implementation forwards to [`materialize`](Self::materialize)
    /// when no media is provided, and otherwise returns
    /// [`RStructorError::Unsupported`](crate::RStructorError::Unsupported) so that
    /// media is never silently dropped. Providers with media support override this
    /// method. All four built-in clients (OpenAI, Anthropic, Grok, Gemini) support media.
    async fn materialize_with_media<T>(&self, prompt: &str, media: &[MediaFile]) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        if media.is_empty() {
            self.materialize(prompt).await
        } else {
            Err(crate::error::RStructorError::Unsupported(
                "this client does not support media inputs".to_string(),
            ))
        }
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

    /// Raw text completion with media attachments (if supported).
    ///
    /// Like [`generate`](Self::generate), but the prompt is sent together with
    /// `media` (images, or PDFs where the provider supports them), encoded in the
    /// provider's documented multimodal format.
    ///
    /// The default implementation forwards to [`generate`](Self::generate) when
    /// no media is provided, and otherwise returns
    /// [`RStructorError::Unsupported`](crate::RStructorError::Unsupported) so that
    /// media is never silently dropped. Providers with media support override this
    /// method. All four built-in clients (OpenAI, Anthropic, Grok, Gemini) support
    /// media here; PDF support varies by provider (Grok, for example, accepts only
    /// images and returns a clear error for PDFs).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rstructor::{LLMClient, OpenAIClient, MediaFile};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenAIClient::from_env()?;
    /// let pdf_bytes = std::fs::read("report.pdf")?;
    /// let media = [MediaFile::from_bytes(&pdf_bytes, "application/pdf")];
    /// let summary = client
    ///     .generate_with_media("Summarize this report", &media)
    ///     .await?;
    /// println!("{summary}");
    /// # Ok(())
    /// # }
    /// ```
    async fn generate_with_media(&self, prompt: &str, media: &[MediaFile]) -> Result<String> {
        if media.is_empty() {
            self.generate(prompt).await
        } else {
            Err(crate::error::RStructorError::Unsupported(
                "this client does not support media inputs".to_string(),
            ))
        }
    }

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

    /// Stream a raw text completion as a sequence of token deltas.
    ///
    /// Returns a [`Stream`](futures_util::Stream) of text chunks; concatenating
    /// every `Ok` item yields the full response. The default implementation falls
    /// back to a single chunk from [`generate`](Self::generate); the built-in
    /// providers override it with true server-sent-events streaming.
    ///
    /// Requires the `streaming` feature.
    #[cfg(feature = "streaming")]
    fn generate_stream<'a>(&'a self, prompt: &'a str) -> crate::backend::streaming::TextStream<'a>
    where
        Self: Sync,
    {
        Box::pin(async_stream::try_stream! {
            let text = self.generate(prompt).await?;
            yield text;
        })
    }

    /// Stream a structured object as it is generated.
    ///
    /// Yields [`StreamedObject::Partial`](crate::StreamedObject::Partial) snapshots
    /// (the object's JSON filling in) as the response streams, followed by a single
    /// [`StreamedObject::Complete`](crate::StreamedObject::Complete) carrying the
    /// fully parsed and validated `T`. The default implementation falls back to a
    /// single `Complete` from [`materialize`](Self::materialize); the built-in
    /// providers override it with true streaming.
    ///
    /// Note: unlike [`materialize`](Self::materialize), streaming is single-shot —
    /// a validation failure ends the stream with an error rather than re-asking.
    ///
    /// Requires the `streaming` feature.
    #[cfg(feature = "streaming")]
    fn materialize_stream<'a, T>(
        &'a self,
        prompt: &'a str,
    ) -> crate::backend::streaming::ObjectStream<'a, T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
        Self: Sync,
    {
        Box::pin(async_stream::try_stream! {
            let value: T = self.materialize(prompt).await?;
            yield crate::backend::streaming::StreamedObject::Complete(value);
        })
    }

    /// Stream a **list** of structured objects, yielding each `T` as soon as that
    /// element of the response array is fully generated and validated.
    ///
    /// This is the primary streaming use case: extracting a long list of items
    /// without buffering the whole response. The model is asked for a JSON object
    /// with an `items` array of `T`; elements are parsed and validated one at a
    /// time.
    ///
    /// The default implementation has no streaming fallback (it errors); the
    /// built-in providers override it. Requires the `streaming` feature.
    #[cfg(feature = "streaming")]
    fn materialize_iter<'a, T>(
        &'a self,
        _prompt: &'a str,
    ) -> crate::backend::streaming::ItemStream<'a, T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
        Self: Sync,
    {
        Box::pin(futures_util::stream::once(async move {
            Err::<T, crate::error::RStructorError>(crate::error::RStructorError::Unsupported(
                "materialize_iter is not implemented for this client".to_string(),
            ))
        }))
    }

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

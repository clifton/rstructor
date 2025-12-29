/// Token usage information from an LLM API call.
///
/// This struct contains the token counts returned by LLM providers,
/// which can be used for monitoring usage and debugging.
///
/// # Example
///
/// ```no_run
/// use rstructor::{LLMClient, OpenAIClient, Instructor};
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Instructor, Serialize, Deserialize)]
/// struct Movie { title: String }
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = OpenAIClient::from_env()?;
/// let result = client.materialize_with_metadata::<Movie>("Describe Inception").await?;
///
/// println!("Movie: {}", result.data.title);
/// if let Some(usage) = &result.usage {
///     println!("Model: {}", usage.model);
///     println!("Input tokens: {}", usage.input_tokens);
///     println!("Output tokens: {}", usage.output_tokens);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenUsage {
    /// The model used for this request
    pub model: String,
    /// Number of tokens in the input/prompt
    pub input_tokens: u64,
    /// Number of tokens in the output/completion
    pub output_tokens: u64,
}

impl TokenUsage {
    /// Create a new TokenUsage instance
    pub fn new(model: impl Into<String>, input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            model: model.into(),
            input_tokens,
            output_tokens,
        }
    }

    /// Total tokens used (input + output)
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// Result of a materialize call, containing both the data and optional usage information.
///
/// This struct wraps the deserialized data along with token usage metadata
/// from the LLM API call.
///
/// # Example
///
/// ```no_run
/// use rstructor::{LLMClient, OpenAIClient, Instructor};
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Instructor, Serialize, Deserialize)]
/// struct Person { name: String, age: u8 }
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = OpenAIClient::from_env()?;
/// let result = client.materialize_with_metadata::<Person>("Describe a person").await?;
///
/// // Access the data directly
/// println!("Name: {}", result.data.name);
///
/// // Check token usage
/// if let Some(usage) = result.usage {
///     println!("Used {} total tokens", usage.total_tokens());
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MaterializeResult<T> {
    /// The deserialized data
    pub data: T,
    /// Token usage information (if available from the provider)
    pub usage: Option<TokenUsage>,
}

impl<T> MaterializeResult<T> {
    /// Create a new MaterializeResult with data and usage
    pub fn new(data: T, usage: Option<TokenUsage>) -> Self {
        Self { data, usage }
    }

    /// Create a MaterializeResult with just data (no usage info)
    pub fn from_data(data: T) -> Self {
        Self { data, usage: None }
    }

    /// Map the data to a new type
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> MaterializeResult<U> {
        MaterializeResult {
            data: f(self.data),
            usage: self.usage,
        }
    }
}

/// Result of a generate call, containing the text and optional usage information.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// The generated text
    pub text: String,
    /// Token usage information (if available from the provider)
    pub usage: Option<TokenUsage>,
}

impl GenerateResult {
    /// Create a new GenerateResult with text and usage
    pub fn new(text: String, usage: Option<TokenUsage>) -> Self {
        Self { text, usage }
    }
}

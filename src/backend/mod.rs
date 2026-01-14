pub mod client;
mod messages;
pub mod usage;
mod utils;

#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "gemini")]
pub mod gemini;
#[cfg(feature = "grok")]
pub mod grok;
#[cfg(feature = "openai")]
pub mod openai;

pub use client::{LLMClient, MediaFile};
pub use messages::{ChatMessage, ChatRole, MaterializeInternalOutput, ValidationFailureContext};
pub use usage::{GenerateResult, MaterializeResult, TokenUsage};

/// Information about an available model from an LLM provider.
///
/// This struct is returned by [`LLMClient::list_models()`] to provide
/// information about models available through the provider's API.
///
/// # Example
///
/// ```no_run
/// # use rstructor::{OpenAIClient, LLMClient};
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = OpenAIClient::from_env()?;
/// let models = client.list_models().await?;
///
/// for model in models {
///     println!("{}: {}", model.id, model.description.unwrap_or_default());
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// The model identifier used in API requests
    pub id: String,
    /// Human-readable display name (if different from id)
    pub name: Option<String>,
    /// Description of the model's capabilities
    pub description: Option<String>,
}
pub(crate) use utils::{
    ResponseFormat, check_response_status, generate_with_retry_with_history, handle_http_error,
    parse_validate_and_create_output, prepare_strict_schema,
};

/// Thinking level configuration for models that support extended reasoning.
///
/// This controls the depth of reasoning the model applies to prompts,
/// balancing between response speed and complexity.
///
/// # Provider Support
///
/// - **OpenAI (GPT-5.x)**: Uses `reasoning_effort` parameter ("none", "low", "medium", "high")
/// - **Gemini 3**: Supports `Minimal`, `Low`, `Medium`, `High` (Flash) or `Low`, `High` (Pro)
/// - **Anthropic (Claude 4.x)**: Thinking is enabled via budget tokens when level is not `Off`
///
/// # Examples
///
/// ```rust
/// use rstructor::{OpenAIClient, GeminiClient, ThinkingLevel};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // OpenAI with low thinking (default)
/// let client = OpenAIClient::new("key")?;
///
/// // Enable high thinking for complex tasks
/// # let client = OpenAIClient::new("key")?;
/// let client = client.thinking_level(ThinkingLevel::High);
///
/// // Gemini with custom thinking level
/// # let client = GeminiClient::new("key")?;
/// let client = client.thinking_level(ThinkingLevel::Medium);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingLevel {
    /// Disable extended thinking (fastest, no reasoning overhead)
    Off,
    /// Minimal reasoning - ideal for high-throughput applications (Gemini Flash only)
    Minimal,
    /// Low reasoning - reduces latency and cost, suitable for straightforward tasks
    #[default]
    Low,
    /// Medium reasoning - balanced for most tasks (Gemini Flash only)
    Medium,
    /// High reasoning - deep reasoning for complex problem-solving
    High,
}

impl ThinkingLevel {
    /// Returns the Gemini API string for this thinking level
    pub fn gemini_level(&self) -> Option<&'static str> {
        match self {
            ThinkingLevel::Off => None,
            ThinkingLevel::Minimal => Some("minimal"),
            ThinkingLevel::Low => Some("low"),
            ThinkingLevel::Medium => Some("medium"),
            ThinkingLevel::High => Some("high"),
        }
    }

    /// Returns whether Claude thinking should be enabled
    pub fn claude_thinking_enabled(&self) -> bool {
        !matches!(self, ThinkingLevel::Off)
    }

    /// Returns the budget tokens for Claude thinking
    /// Higher thinking levels get more budget
    pub fn claude_budget_tokens(&self) -> u32 {
        match self {
            ThinkingLevel::Off => 0,
            ThinkingLevel::Minimal => 1024,
            ThinkingLevel::Low => 2048,
            ThinkingLevel::Medium => 4096,
            ThinkingLevel::High => 8192,
        }
    }

    /// Returns the OpenAI reasoning_effort string for GPT-5.x models
    /// Maps: Off -> "none", Minimal -> "low", Low -> "low", Medium -> "medium", High -> "high"
    pub fn openai_reasoning_effort(&self) -> Option<&'static str> {
        match self {
            ThinkingLevel::Off => Some("none"),
            ThinkingLevel::Minimal => Some("low"),
            ThinkingLevel::Low => Some("low"),
            ThinkingLevel::Medium => Some("medium"),
            ThinkingLevel::High => Some("high"),
        }
    }
}

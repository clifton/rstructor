pub mod client;
mod utils;

#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "gemini")]
pub mod gemini;
#[cfg(feature = "grok")]
pub mod grok;
#[cfg(feature = "openai")]
pub mod openai;

pub use client::LLMClient;
pub(crate) use utils::{
    check_response_status, extract_json_from_markdown, generate_with_retry, handle_http_error,
};

/// Thinking level configuration for models that support extended reasoning.
///
/// This controls the depth of reasoning the model applies to prompts,
/// balancing between response speed and complexity.
///
/// # Provider Support
///
/// - **Gemini 3**: Supports `Minimal`, `Low`, `Medium`, `High` (Flash) or `Low`, `High` (Pro)
/// - **Anthropic (Claude 4.x)**: Thinking is enabled via budget tokens when level is not `Off`
/// - **OpenAI (O-series)**: Has built-in reasoning, but not configurable via this parameter
///
/// # Examples
///
/// ```rust
/// use rstructor::{GeminiClient, ThinkingLevel};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = GeminiClient::new("key")?;
/// // Use low thinking for fast responses
/// let client = client.thinking_level(ThinkingLevel::Low);
///
/// // Use high thinking for complex reasoning tasks
/// # let client = GeminiClient::new("key")?;
/// let client = client.thinking_level(ThinkingLevel::High);
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
}

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::str::FromStr;
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::{
    AnthropicMessageContent, ChatMessage, GenerateResult, LLMClient, MaterializeInternalOutput,
    MaterializeResult, ModelInfo, ThinkingLevel, TokenUsage, ValidationFailureContext,
    build_anthropic_message_content, check_response_status, generate_with_retry_with_history,
    handle_http_error, materialize_with_media_with_retry, parse_validate_and_create_output,
    prepare_strict_schema,
};
use crate::error::{ApiErrorKind, RStructorError, Result};
use crate::model::Instructor;

/// Anthropic models available for completion
///
/// For the latest available models and their identifiers, check the
/// [Anthropic Models Documentation](https://docs.anthropic.com/en/docs/about-claude/models/all-models).
///
/// # Using Custom Models
///
/// You can specify any model name as a string using `Custom` variant or `FromStr`:
///
/// ```rust
/// use rstructor::AnthropicModel;
/// use std::str::FromStr;
///
/// // Using Custom variant
/// let model = AnthropicModel::Custom("claude-custom".to_string());
///
/// // Using FromStr (useful for config files)
/// let model = AnthropicModel::from_str("claude-custom").unwrap();
///
/// // Or use the convenience method
/// let model = AnthropicModel::from_string("claude-custom");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnthropicModel {
    /// Claude Opus 4.6 (latest most capable model)
    ClaudeOpus46,
    /// Claude Opus 4.5 (enhanced Opus 4.5)
    ClaudeOpus45,
    /// Claude Haiku 4.5 (latest fastest model)
    ClaudeHaiku45,
    /// Claude Sonnet 4.5 (latest balanced model)
    ClaudeSonnet45,
    /// Claude Opus 4.1 (enhanced reasoning capabilities)
    ClaudeOpus41,
    /// Claude Opus 4 (high-intelligence model)
    ClaudeOpus4,
    /// Claude Sonnet 4 (balanced performance model)
    ClaudeSonnet4,
    /// Claude Sonnet 3.7 (enhanced reasoning)
    Claude37Sonnet,
    /// Claude Haiku 3.5 (fast, cost-effective model)
    Claude35Haiku,
    /// Claude Haiku 3 (fast, cost-effective model)
    Claude3Haiku,
    /// Claude Opus 3 (most capable model for complex tasks)
    Claude3Opus,
    /// Custom model name (for new models or Anthropic-compatible endpoints)
    Custom(String),
}

impl AnthropicModel {
    pub fn as_str(&self) -> &str {
        match self {
            AnthropicModel::ClaudeOpus46 => "claude-opus-4-6",
            AnthropicModel::ClaudeOpus45 => "claude-opus-4-5-20251101",
            AnthropicModel::ClaudeHaiku45 => "claude-haiku-4-5-20251001",
            AnthropicModel::ClaudeSonnet45 => "claude-sonnet-4-5-20250929",
            AnthropicModel::ClaudeOpus41 => "claude-opus-4-1-20250805",
            AnthropicModel::ClaudeOpus4 => "claude-opus-4-20250514",
            AnthropicModel::ClaudeSonnet4 => "claude-sonnet-4-20250514",
            AnthropicModel::Claude37Sonnet => "claude-3-7-sonnet-20250219",
            AnthropicModel::Claude35Haiku => "claude-3-5-haiku-20241022",
            AnthropicModel::Claude3Haiku => "claude-3-haiku-20240307",
            AnthropicModel::Claude3Opus => "claude-3-opus-20240229",
            AnthropicModel::Custom(name) => name,
        }
    }

    /// Create a model from a string. This is a convenience method that always succeeds.
    ///
    /// If the string matches a known model variant, it returns that variant.
    /// Otherwise, it returns `Custom(name)`.
    pub fn from_string(name: impl Into<String>) -> Self {
        let name = name.into();
        match name.as_str() {
            "claude-opus-4-6" => AnthropicModel::ClaudeOpus46,
            "claude-opus-4-5-20251101" => AnthropicModel::ClaudeOpus45,
            "claude-haiku-4-5-20251001" => AnthropicModel::ClaudeHaiku45,
            "claude-sonnet-4-5-20250929" => AnthropicModel::ClaudeSonnet45,
            "claude-opus-4-1-20250805" => AnthropicModel::ClaudeOpus41,
            "claude-opus-4-20250514" => AnthropicModel::ClaudeOpus4,
            "claude-sonnet-4-20250514" => AnthropicModel::ClaudeSonnet4,
            "claude-3-7-sonnet-20250219" => AnthropicModel::Claude37Sonnet,
            "claude-3-5-haiku-20241022" => AnthropicModel::Claude35Haiku,
            "claude-3-haiku-20240307" => AnthropicModel::Claude3Haiku,
            "claude-3-opus-20240229" => AnthropicModel::Claude3Opus,
            _ => AnthropicModel::Custom(name),
        }
    }
}

impl FromStr for AnthropicModel {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(AnthropicModel::from_string(s))
    }
}

impl From<&str> for AnthropicModel {
    fn from(s: &str) -> Self {
        AnthropicModel::from_string(s)
    }
}

impl From<String> for AnthropicModel {
    fn from(s: String) -> Self {
        AnthropicModel::from_string(s)
    }
}

/// Configuration for the Anthropic client
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub model: AnthropicModel,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub timeout: Option<Duration>,
    pub max_retries: Option<usize>,
    /// Custom base URL for Anthropic-compatible APIs
    /// Defaults to "https://api.anthropic.com/v1" if not set
    pub base_url: Option<String>,
    /// Thinking level for Claude 4.x models (Sonnet 4, Opus 4, etc.)
    /// When enabled, temperature is automatically set to 1.0 as required by the API
    pub thinking_level: Option<ThinkingLevel>,
}

/// Anthropic client for generating completions
pub struct AnthropicClient {
    config: AnthropicConfig,
    client: reqwest::Client,
}

// Anthropic API request and response structures
#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicMessageContent,
}

/// Output format for structured outputs (native Anthropic structured outputs)
#[derive(Debug, Serialize)]
struct OutputFormat {
    #[serde(rename = "type")]
    format_type: String,
    schema: Value,
}

#[derive(Debug, Serialize)]
struct CompletionRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    temperature: f32,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ClaudeThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_format: Option<OutputFormat>,
}

#[derive(Debug, Serialize)]
struct ClaudeThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: String,
    budget_tokens: u32,
}

const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 1024;

fn effective_max_tokens(
    configured_max_tokens: Option<u32>,
    thinking_config: Option<&ClaudeThinkingConfig>,
) -> u32 {
    let configured = configured_max_tokens.unwrap_or(DEFAULT_ANTHROPIC_MAX_TOKENS);

    match thinking_config {
        Some(thinking) if configured <= thinking.budget_tokens => {
            let required_min = thinking.budget_tokens.saturating_add(1);
            warn!(
                configured_max_tokens = configured,
                thinking_budget_tokens = thinking.budget_tokens,
                adjusted_max_tokens = required_min,
                "Adjusted max_tokens to satisfy Anthropic requirement: max_tokens must be greater than thinking.budget_tokens"
            );
            required_min
        }
        _ => configured,
    }
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct UsageInfo {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct CompletionResponse {
    content: Vec<ContentBlock>,
    model: Option<String>,
    #[serde(default)]
    usage: Option<UsageInfo>,
}

impl AnthropicClient {
    /// Create a new Anthropic client with the provided API key.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Your Anthropic API key
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rstructor::AnthropicClient;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = AnthropicClient::new("your-anthropic-api-key")?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(name = "anthropic_client_new", skip(api_key), fields(model = ?AnthropicModel::ClaudeSonnet45))]
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(RStructorError::api_error(
                "Anthropic",
                ApiErrorKind::AuthenticationFailed,
            ));
        }
        info!("Creating new Anthropic client");
        trace!("API key length: {}", api_key.len());

        let config = AnthropicConfig {
            api_key,
            model: AnthropicModel::ClaudeSonnet45, // Default to Claude Sonnet 4.5 (latest flagship)
            temperature: 0.0,
            max_tokens: None,
            timeout: None,        // Default: no timeout (uses reqwest's default)
            max_retries: Some(3), // Default: 3 retries with error feedback
            base_url: None,       // Default: use official Anthropic API
            thinking_level: None, // Default: no extended thinking (faster responses)
        };

        debug!("Anthropic client created with default configuration");
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    /// Create a new Anthropic client by reading the API key from the `ANTHROPIC_API_KEY` environment variable.
    ///
    /// # Errors
    ///
    /// Returns an error if `ANTHROPIC_API_KEY` is not set.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rstructor::AnthropicClient;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = AnthropicClient::from_env()?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(name = "anthropic_client_from_env", fields(model = ?AnthropicModel::ClaudeSonnet45))]
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            RStructorError::api_error("Anthropic", ApiErrorKind::AuthenticationFailed)
        })?;

        info!("Creating new Anthropic client from environment variable");
        trace!("API key length: {}", api_key.len());

        let config = AnthropicConfig {
            api_key,
            model: AnthropicModel::ClaudeSonnet45, // Default to Claude Sonnet 4.5 (latest flagship)
            temperature: 0.0,
            max_tokens: None,
            timeout: None,        // Default: no timeout (uses reqwest's default)
            max_retries: Some(3), // Default: 3 retries with error feedback
            base_url: None,       // Default: use official Anthropic API
            thinking_level: None, // Default: no extended thinking (faster responses)
        };

        debug!("Anthropic client created with default configuration");
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    // Builder methods are generated by the macro below
}

impl AnthropicClient {
    /// Internal implementation of materialize (without retry logic)
    /// Accepts conversation history for multi-turn interactions.
    /// Returns the data, raw response, and optional usage info.
    ///
    /// Uses Anthropic's native Structured Outputs with `output_format: json_schema`
    /// for guaranteed schema compliance.
    ///
    /// The raw response is included to enable conversation history tracking for retries,
    /// which improves prompt caching efficiency.
    async fn materialize_internal<T>(
        &self,
        messages: &[ChatMessage],
    ) -> std::result::Result<
        MaterializeInternalOutput<T>,
        (RStructorError, Option<ValidationFailureContext>),
    >
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        info!("Generating structured response with Anthropic (native structured outputs)");

        // Get the schema for type T
        let schema = T::schema();
        trace!("Retrieved JSON schema for type");

        // Prepare schema with additionalProperties: false recursively for all nested objects
        let schema_json = prepare_strict_schema(&schema);

        // Build API messages from conversation history
        // With native structured outputs, we don't need to include schema instructions in the prompt
        let api_messages: Vec<AnthropicMessage> = messages
            .iter()
            .map(|msg| {
                Ok(AnthropicMessage {
                    role: msg.role.as_str().to_string(),
                    content: build_anthropic_message_content(msg)?,
                })
            })
            .collect::<Result<Vec<_>>>()
            .map_err(|e| (e, None))?;

        // Build thinking config for Claude 4.x models
        let is_thinking_model = self.config.model.as_str().contains("sonnet-4")
            || self.config.model.as_str().contains("opus-4");
        let thinking_config = self.config.thinking_level.and_then(|level| {
            if is_thinking_model && level.claude_thinking_enabled() {
                Some(ClaudeThinkingConfig {
                    thinking_type: "enabled".to_string(),
                    budget_tokens: level.claude_budget_tokens(),
                })
            } else {
                None
            }
        });

        // Claude requires temperature=1 when thinking is enabled
        let effective_temp = if thinking_config.is_some() {
            1.0
        } else {
            self.config.temperature
        };

        // Create output format for native structured outputs
        let output_format = OutputFormat {
            format_type: "json_schema".to_string(),
            schema: schema_json,
        };

        // Build the request with native structured outputs
        debug!(
            "Building Anthropic API request with structured outputs (history_len={})",
            api_messages.len()
        );
        let request = CompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: api_messages,
            temperature: effective_temp,
            max_tokens: effective_max_tokens(self.config.max_tokens, thinking_config.as_ref()),
            thinking: thinking_config,
            output_format: Some(output_format),
        };

        // Send the request to Anthropic with structured outputs beta header
        debug!(
            model = %self.config.model.as_str(),
            max_tokens = request.max_tokens,
            "Sending request to Anthropic API with structured outputs"
        );
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com/v1");
        let url = format!("{}/messages", base_url);
        debug!(url = %url, "Using Anthropic API endpoint");
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "structured-outputs-2025-11-13")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| (handle_http_error(e, "Anthropic"), None))?;

        // Parse the response
        let response = check_response_status(response, "Anthropic")
            .await
            .map_err(|e| (e, None))?;

        debug!("Successfully received response from Anthropic");
        let completion: CompletionResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse JSON response from Anthropic");
            (RStructorError::from(e), None)
        })?;

        // Extract usage info
        let model_name = completion
            .model
            .clone()
            .unwrap_or_else(|| self.config.model.as_str().to_string());
        let usage = completion
            .usage
            .as_ref()
            .map(|u| TokenUsage::new(model_name.clone(), u.input_tokens, u.output_tokens));

        // Extract the content, assuming the first block is text containing JSON
        let raw_response = match completion
            .content
            .iter()
            .find(|block| block.block_type == "text")
            .map(|block| block.text.clone())
        {
            Some(text) => {
                debug!(
                    content_len = text.len(),
                    "Successfully extracted text content from response"
                );
                text
            }
            None => {
                error!("No text content in Anthropic response");
                return Err((
                    RStructorError::api_error(
                        "Anthropic",
                        ApiErrorKind::UnexpectedResponse {
                            details: "No text content in response".to_string(),
                        },
                    ),
                    None,
                ));
            }
        };

        // Parse the JSON content directly using shared utility
        // With native structured outputs, the response is guaranteed to be valid JSON
        trace!(json = %raw_response, "Parsing structured output response");
        parse_validate_and_create_output(raw_response, usage)
    }
}

// Generate builder methods using macro
crate::impl_client_builder_methods! {
    client_type: AnthropicClient,
    config_type: AnthropicConfig,
    model_type: AnthropicModel,
    provider_name: "Anthropic"
}

impl AnthropicClient {
    /// Set a custom base URL for Anthropic-compatible APIs.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL without trailing slash (e.g., "http://localhost:1234/v1" or "https://api.example.com/v1")
    #[tracing::instrument(skip(self, base_url))]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url_str = base_url.into();
        tracing::debug!(
            previous_base_url = ?self.config.base_url,
            new_base_url = %base_url_str,
            "Setting custom base URL"
        );
        self.config.base_url = Some(base_url_str);
        self
    }

    /// Set the thinking level for Claude 4.x models (Sonnet 4, Opus 4, etc.).
    ///
    /// When thinking is enabled, the model will engage in extended reasoning before responding.
    /// Note: Temperature is automatically set to 1.0 when thinking is enabled, as required by the API.
    /// If `max_tokens` is not set (or is too low), it is automatically adjusted so it remains greater
    /// than the configured thinking budget.
    ///
    /// # Thinking Levels
    ///
    /// - `Off`: Disable extended thinking (default, fastest)
    /// - `Minimal`: 1024 budget tokens - minimal reasoning
    /// - `Low`: 2048 budget tokens - light reasoning
    /// - `Medium`: 4096 budget tokens - balanced reasoning
    /// - `High`: 8192 budget tokens - deep reasoning
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use rstructor::{AnthropicClient, ThinkingLevel};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = AnthropicClient::from_env()?
    ///     .thinking_level(ThinkingLevel::Low);
    /// # Ok(())
    /// # }
    /// ```
    #[tracing::instrument(skip(self))]
    pub fn thinking_level(mut self, level: ThinkingLevel) -> Self {
        tracing::debug!(
            previous_level = ?self.config.thinking_level,
            new_level = ?level,
            "Setting thinking level"
        );
        self.config.thinking_level = Some(level);
        self
    }
}

#[async_trait]
impl LLMClient for AnthropicClient {
    fn from_env() -> Result<Self> {
        Self::from_env()
    }

    #[instrument(
        name = "anthropic_materialize",
        skip(self, prompt),
        fields(
            type_name = std::any::type_name::<T>(),
            model = %self.config.model.as_str(),
            prompt_len = prompt.len()
        )
    )]
    async fn materialize<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        let output = generate_with_retry_with_history(
            |messages: Vec<ChatMessage>| {
                let this = self;
                async move { this.materialize_internal::<T>(&messages).await }
            },
            prompt,
            self.config.max_retries,
        )
        .await?;
        Ok(output.data)
    }

    #[instrument(
        name = "anthropic_materialize_with_media",
        skip(self, prompt, media),
        fields(
            type_name = std::any::type_name::<T>(),
            model = %self.config.model.as_str(),
            prompt_len = prompt.len(),
            media_len = media.len()
        )
    )]
    async fn materialize_with_media<T>(&self, prompt: &str, media: &[super::MediaFile]) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        materialize_with_media_with_retry(
            |messages: Vec<ChatMessage>| {
                let this = self;
                async move { this.materialize_internal::<T>(&messages).await }
            },
            prompt,
            media,
            self.config.max_retries,
        )
        .await
    }

    #[instrument(
        name = "anthropic_materialize_with_metadata",
        skip(self, prompt),
        fields(
            type_name = std::any::type_name::<T>(),
            model = %self.config.model.as_str(),
            prompt_len = prompt.len()
        )
    )]
    async fn materialize_with_metadata<T>(&self, prompt: &str) -> Result<MaterializeResult<T>>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        let output = generate_with_retry_with_history(
            |messages: Vec<ChatMessage>| {
                let this = self;
                async move { this.materialize_internal::<T>(&messages).await }
            },
            prompt,
            self.config.max_retries,
        )
        .await?;
        Ok(MaterializeResult::new(output.data, output.usage))
    }

    #[instrument(
        name = "anthropic_generate",
        skip(self, prompt),
        fields(
            model = %self.config.model.as_str(),
            prompt_len = prompt.len()
        )
    )]
    async fn generate(&self, prompt: &str) -> Result<String> {
        let result = self.generate_with_metadata(prompt).await?;
        Ok(result.text)
    }

    #[instrument(
        name = "anthropic_generate_with_metadata",
        skip(self, prompt),
        fields(
            model = %self.config.model.as_str(),
            prompt_len = prompt.len()
        )
    )]
    async fn generate_with_metadata(&self, prompt: &str) -> Result<GenerateResult> {
        info!("Generating raw text response with Anthropic");

        // Build thinking config for Claude 4.x models
        let is_thinking_model = self.config.model.as_str().contains("sonnet-4")
            || self.config.model.as_str().contains("opus-4");
        let thinking_config = self.config.thinking_level.and_then(|level| {
            if is_thinking_model && level.claude_thinking_enabled() {
                Some(ClaudeThinkingConfig {
                    thinking_type: "enabled".to_string(),
                    budget_tokens: level.claude_budget_tokens(),
                })
            } else {
                None
            }
        });

        // Claude requires temperature=1 when thinking is enabled
        let effective_temp = if thinking_config.is_some() {
            1.0
        } else {
            self.config.temperature
        };

        // Build the request (no output_format for raw text generation)
        debug!("Building Anthropic API request for text generation");
        let request = CompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::Text(prompt.to_string()),
            }],
            temperature: effective_temp,
            max_tokens: effective_max_tokens(self.config.max_tokens, thinking_config.as_ref()),
            thinking: thinking_config,
            output_format: None, // Raw text generation doesn't use structured outputs
        };

        // Send the request to Anthropic
        debug!(
            model = %self.config.model.as_str(),
            max_tokens = request.max_tokens,
            "Sending request to Anthropic API"
        );
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com/v1");
        let url = format!("{}/messages", base_url);
        debug!(url = %url, "Using Anthropic API endpoint");
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| handle_http_error(e, "Anthropic"))?;

        // Parse the response
        let response = check_response_status(response, "Anthropic").await?;

        debug!("Successfully received response from Anthropic");
        let completion: CompletionResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse JSON response from Anthropic");
            e
        })?;

        // Extract usage info
        let model_name = completion
            .model
            .clone()
            .unwrap_or_else(|| self.config.model.as_str().to_string());
        let usage = completion
            .usage
            .as_ref()
            .map(|u| TokenUsage::new(model_name, u.input_tokens, u.output_tokens));

        // Extract the content
        debug!("Extracting text content from response blocks");
        let content: String = completion
            .content
            .iter()
            .filter(|block| block.block_type == "text")
            .map(|block| block.text.clone())
            .collect::<Vec<String>>()
            .join("");

        if content.is_empty() {
            error!("No text content in Anthropic response");
            return Err(RStructorError::api_error(
                "Anthropic",
                ApiErrorKind::UnexpectedResponse {
                    details: "No text content in response".to_string(),
                },
            ));
        }

        debug!(
            content_len = content.len(),
            "Successfully extracted text content"
        );
        Ok(GenerateResult::new(content, usage))
    }

    /// Fetch available models from Anthropic's API.
    ///
    /// Returns a list of Claude models available for chat completions.
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.anthropic.com/v1");
        let url = format!("{}/models", base_url);

        debug!(url = %url, "Fetching available models from Anthropic");

        let response = self
            .client
            .get(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| handle_http_error(e, "Anthropic"))?;

        let response = check_response_status(response, "Anthropic").await?;

        let json: serde_json::Value = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse models response from Anthropic");
            e
        })?;

        let models = json
            .get("data")
            .and_then(|data| data.as_array())
            .map(|models_array| {
                models_array
                    .iter()
                    .filter_map(|model| {
                        let id = model.get("id").and_then(|id| id.as_str())?;
                        // Filter to only Claude models
                        if id.starts_with("claude-") {
                            let display_name = model
                                .get("display_name")
                                .and_then(|n| n.as_str())
                                .map(|s| s.to_string());
                            Some(ModelInfo {
                                id: id.to_string(),
                                name: display_name,
                                description: None,
                            })
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        debug!(count = models.len(), "Fetched Anthropic models");
        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::{ClaudeThinkingConfig, DEFAULT_ANTHROPIC_MAX_TOKENS, effective_max_tokens};

    fn thinking_config_with_budget(budget_tokens: u32) -> ClaudeThinkingConfig {
        ClaudeThinkingConfig {
            thinking_type: "enabled".to_string(),
            budget_tokens,
        }
    }

    #[test]
    fn effective_max_tokens_uses_default_without_thinking() {
        let result = effective_max_tokens(None, None);
        assert_eq!(result, DEFAULT_ANTHROPIC_MAX_TOKENS);
    }

    #[test]
    fn effective_max_tokens_uses_configured_without_thinking() {
        let result = effective_max_tokens(Some(2048), None);
        assert_eq!(result, 2048);
    }

    #[test]
    fn effective_max_tokens_adjusts_default_when_thinking_budget_is_higher() {
        let thinking = thinking_config_with_budget(2048);
        let result = effective_max_tokens(None, Some(&thinking));
        assert_eq!(result, 2049);
    }

    #[test]
    fn effective_max_tokens_adjusts_when_configured_equals_budget() {
        let thinking = thinking_config_with_budget(4096);
        let result = effective_max_tokens(Some(4096), Some(&thinking));
        assert_eq!(result, 4097);
    }

    #[test]
    fn effective_max_tokens_keeps_configured_when_already_valid() {
        let thinking = thinking_config_with_budget(2048);
        let result = effective_max_tokens(Some(8192), Some(&thinking));
        assert_eq!(result, 8192);
    }

    #[test]
    fn effective_max_tokens_saturates_on_extreme_budget() {
        let thinking = thinking_config_with_budget(u32::MAX);
        let result = effective_max_tokens(None, Some(&thinking));
        assert_eq!(result, u32::MAX);
    }
}

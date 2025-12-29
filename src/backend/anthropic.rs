use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::{
    LLMClient, check_response_status, extract_json_from_markdown, generate_with_retry,
    handle_http_error,
};
use crate::error::{RStructorError, Result};
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

use crate::backend::ThinkingLevel;

/// Configuration for the Anthropic client
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub model: AnthropicModel,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub timeout: Option<Duration>,
    pub max_retries: Option<usize>,
    pub include_error_feedback: Option<bool>,
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
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct CompletionRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ClaudeThinkingConfig>,
}

#[derive(Debug, Serialize)]
struct ClaudeThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: String,
    budget_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ResponseMessage {
    role: String,
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct CompletionResponse {
    content: Vec<ContentBlock>,
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
            return Err(RStructorError::ApiError(
                "API key cannot be empty. Use AnthropicClient::from_env() to read from ANTHROPIC_API_KEY environment variable.".to_string(),
            ));
        }
        info!("Creating new Anthropic client");
        trace!("API key length: {}", api_key.len());

        let config = AnthropicConfig {
            api_key,
            model: AnthropicModel::ClaudeSonnet45, // Default to Claude Sonnet 4.5 (latest flagship)
            temperature: 0.0,
            max_tokens: None,
            timeout: None,     // Default: no timeout (uses reqwest's default)
            max_retries: None, // Default: no retries (configure via .max_retries())
            include_error_feedback: None, // Default: include error feedback in retry prompts
            base_url: None,    // Default: use official Anthropic API
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
            RStructorError::ApiError(
                "ANTHROPIC_API_KEY environment variable is not set".to_string(),
            )
        })?;

        info!("Creating new Anthropic client from environment variable");
        trace!("API key length: {}", api_key.len());

        let config = AnthropicConfig {
            api_key,
            model: AnthropicModel::ClaudeSonnet45, // Default to Claude Sonnet 4.5 (latest flagship)
            temperature: 0.0,
            max_tokens: None,
            timeout: None,     // Default: no timeout (uses reqwest's default)
            max_retries: None, // Default: no retries (configure via .max_retries())
            include_error_feedback: None, // Default: include error feedback in retry prompts
            base_url: None,    // Default: use official Anthropic API
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
    async fn materialize_internal<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        info!("Generating structured response with Anthropic");

        // Get the schema for type T
        let schema = T::schema();
        // Avoid calling to_string() to prevent potential stack overflow with complex schemas
        trace!("Retrieved JSON schema for type");
        // Get schema as JSON string - avoid Display impl which might cause recursion
        let schema_str =
            serde_json::to_string(&schema.to_json()).unwrap_or_else(|_| "{}".to_string());
        debug!("Building structured prompt with schema");
        let structured_prompt = format!(
            "You are a helpful assistant that outputs JSON. The user wants data in the following JSON schema format:\n\n{}\n\nYou MUST provide your answer in valid JSON format according to the schema above.\n1. Include ALL required fields\n2. Format as a complete, valid JSON object\n3. DO NOT include explanations, just return the JSON\n4. Make sure to use double quotes for all strings and property names\n5. For enum fields, use EXACTLY one of the values listed in the descriptions\n6. Include ALL nested objects with all their required fields\n7. For array fields:\n   - MOST IMPORTANT: When an array items.type is \"object\", provide an array of complete objects with ALL required fields\n   - DO NOT provide arrays of strings when arrays of objects are required\n   - Include multiple items (at least 2-3) in each array\n   - Every object in an array must match the schema for that object type\n8. Follow type specifications EXACTLY (string, number, boolean, array, object)\n\nUser query: {}",
            schema_str, prompt
        );

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

        // Build the request
        debug!("Building Anthropic API request");
        let request = CompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: structured_prompt,
            }],
            temperature: effective_temp,
            max_tokens: self.config.max_tokens.unwrap_or(1024), // Default to 1024 if not specified
            thinking: thinking_config,
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

        // Extract the content, assuming the first block is text containing JSON
        let content = match completion
            .content
            .iter()
            .find(|block| block.block_type == "text")
            .map(|block| &block.text)
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
                return Err(RStructorError::ApiError(
                    "No text content in response".to_string(),
                ));
            }
        };

        // Try to parse the content as JSON
        // First, try to extract JSON from markdown code blocks if present
        let json_content = extract_json_from_markdown(content);
        trace!(json = %json_content, "Attempting to parse response as JSON");
        let result: T = match serde_json::from_str(&json_content) {
            Ok(parsed) => parsed,
            Err(e) => {
                let error_msg = format!(
                    "Failed to parse response as JSON: {}\nPartial JSON: {}",
                    e, json_content
                );
                error!(
                    error = %e,
                    content = %json_content,
                    "JSON parsing error"
                );
                return Err(RStructorError::ValidationError(error_msg));
            }
        };

        // Apply any custom validation
        debug!("Applying custom validation");
        if let Err(e) = result.validate() {
            error!(error = ?e, "Custom validation failed");
            return Err(e);
        }

        info!("Successfully generated and validated structured data");
        Ok(result)
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
        generate_with_retry(
            |prompt_owned: String| {
                let this = self;
                async move { this.materialize_internal::<T>(&prompt_owned).await }
            },
            prompt,
            self.config.max_retries,
            self.config.include_error_feedback,
        )
        .await
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

        // Build the request
        debug!("Building Anthropic API request for text generation");
        let request = CompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: effective_temp,
            max_tokens: self.config.max_tokens.unwrap_or(1024), // Default to 1024 if not specified
            thinking: thinking_config,
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
            return Err(RStructorError::ApiError(
                "No text content in response".to_string(),
            ));
        }

        debug!(
            content_len = content.len(),
            "Successfully extracted text content"
        );
        Ok(content)
    }
}

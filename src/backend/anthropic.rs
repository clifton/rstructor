use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::LLMClient;
use crate::error::{RStructorError, Result};
use crate::model::Instructor;

/// Extract JSON from markdown code blocks if present, otherwise return the content as-is
fn extract_json_from_markdown(content: &str) -> String {
    // Check if content is wrapped in markdown code blocks
    let trimmed = content.trim();

    // Match ```json ... ``` or ``` ... ```
    if trimmed.starts_with("```") {
        // Find the first newline after ```
        if let Some(start_idx) = trimmed.find('\n') {
            let after_start = &trimmed[start_idx + 1..];
            // Find the closing ```
            if let Some(end_idx) = after_start.rfind("```") {
                return after_start[..end_idx].trim().to_string();
            }
        }
    }

    // If no markdown code blocks found, return as-is
    trimmed.to_string()
}

/// Anthropic models available for completion
///
/// For the latest available models and their identifiers, check the
/// [Anthropic Models Documentation](https://docs.anthropic.com/en/docs/about-claude/models/all-models).
#[derive(Debug, Clone)]
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
}

impl AnthropicModel {
    pub fn as_str(&self) -> &'static str {
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
        }
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
            timeout: None, // Default: no timeout (uses reqwest's default)
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
            timeout: None, // Default: no timeout (uses reqwest's default)
        };

        debug!("Anthropic client created with default configuration");
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    /// Set the model to use
    #[instrument(skip(self))]
    pub fn model(mut self, model: AnthropicModel) -> Self {
        debug!(
            previous_model = ?self.config.model,
            new_model = ?model,
            "Setting Anthropic model"
        );
        self.config.model = model;
        self
    }

    /// Set the temperature (0.0 to 1.0, lower = more deterministic)
    #[instrument(skip(self))]
    pub fn temperature(mut self, temp: f32) -> Self {
        debug!(
            previous_temp = self.config.temperature,
            new_temp = temp,
            "Setting temperature"
        );
        self.config.temperature = temp;
        self
    }

    /// Set the maximum tokens to generate
    #[instrument(skip(self))]
    pub fn max_tokens(mut self, max: u32) -> Self {
        debug!(
            previous_max = ?self.config.max_tokens,
            new_max = max,
            "Setting max_tokens"
        );
        // Ensure max_tokens is at least 1 to avoid API errors
        self.config.max_tokens = Some(max.max(1));
        self
    }

    /// Set the timeout for HTTP requests.
    ///
    /// This sets the timeout for both the connection and the entire request.
    /// The timeout applies to each HTTP request made by the client.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Timeout duration (e.g., `Duration::from_secs(30)` for 30 seconds)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rstructor::AnthropicClient;
    /// # use std::time::Duration;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = AnthropicClient::new("api-key")?
    ///     .timeout(Duration::from_secs(30));  // 30 second timeout
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self))]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        debug!(
            previous_timeout = ?self.config.timeout,
            new_timeout = ?timeout,
            "Setting timeout"
        );
        self.config.timeout = Some(timeout);

        // Rebuild reqwest client with timeout immediately
        let mut client_builder = reqwest::Client::builder();
        client_builder = client_builder.timeout(timeout);
        self.client = client_builder.build().unwrap_or_else(|e| {
            warn!(error = %e, "Failed to build reqwest client with timeout, using default");
            reqwest::Client::new()
        });

        self
    }
}

#[async_trait]
impl LLMClient for AnthropicClient {
    fn from_env() -> Result<Self> {
        Self::from_env()
    }
    #[instrument(
        name = "anthropic_generate_struct",
        skip(self, prompt),
        fields(
            type_name = std::any::type_name::<T>(),
            model = %self.config.model.as_str(),
            prompt_len = prompt.len()
        )
    )]
    async fn generate_struct<T>(&self, prompt: &str) -> Result<T>
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

        // Build the request
        debug!("Building Anthropic API request");
        let request = CompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: structured_prompt,
            }],
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens.unwrap_or(1024), // Default to 1024 if not specified
        };

        // Send the request to Anthropic
        debug!(
            model = %self.config.model.as_str(),
            max_tokens = request.max_tokens,
            "Sending request to Anthropic API"
        );
        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "HTTP request to Anthropic failed");
                // Check if it's a timeout error from reqwest
                if e.is_timeout() {
                    RStructorError::Timeout
                } else {
                    RStructorError::HttpError(e)
                }
            })?;

        // Parse the response
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            error!(
                status = %status,
                error = %error_text,
                "Anthropic API returned error response"
            );
            return Err(RStructorError::ApiError(format!(
                "Anthropic API error: {}",
                error_text
            )));
        }

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

        // Build the request
        debug!("Building Anthropic API request for text generation");
        let request = CompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens.unwrap_or(1024), // Default to 1024 if not specified
        };

        // Send the request to Anthropic
        debug!(
            model = %self.config.model.as_str(),
            max_tokens = request.max_tokens,
            "Sending request to Anthropic API"
        );
        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "HTTP request to Anthropic failed");
                // Check if it's a timeout error from reqwest
                if e.is_timeout() {
                    RStructorError::Timeout
                } else {
                    RStructorError::HttpError(e)
                }
            })?;

        // Parse the response
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            error!(
                status = %status,
                error = %error_text,
                "Anthropic API returned error response"
            );
            return Err(RStructorError::ApiError(format!(
                "Anthropic API error: {}",
                error_text
            )));
        }

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

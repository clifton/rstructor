use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::{
    ChatMessage, GenerateResult, LLMClient, MaterializeInternalOutput, MaterializeResult,
    ModelInfo, ResponseFormat, TokenUsage, ValidationFailureContext, check_response_status,
    generate_with_retry_with_history, generate_with_retry_with_initial_messages, handle_http_error,
    media_to_url, parse_validate_and_create_output, prepare_strict_schema,
};
use crate::error::{ApiErrorKind, RStructorError, Result};
use crate::model::Instructor;

/// Grok models available for completion
///
/// These are convenience variants for common Grok models.
/// For the latest available models and their identifiers, check the
/// [xAI Models Documentation](https://docs.x.ai/docs/models).
///
/// # Using Custom Models
///
/// You can specify any model name as a string using `Custom` variant or `FromStr`:
///
/// ```rust
/// use rstructor::GrokModel;
/// use std::str::FromStr;
///
/// // Using Custom variant
/// let model = GrokModel::Custom("grok-custom".to_string());
///
/// // Using FromStr (useful for config files)
/// let model = GrokModel::from_str("grok-custom").unwrap();
///
/// // Or use the convenience method
/// let model = GrokModel::from_string("grok-custom");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Model {
    /// Grok-4 (latest flagship model with 256k context window)
    Grok4,
    /// Grok-4 Fast Reasoning (faster variant optimized for reasoning tasks)
    Grok4FastReasoning,
    /// Grok-4 Fast Non-Reasoning (faster variant optimized for non-reasoning tasks)
    Grok4FastNonReasoning,
    /// Grok-4.1 Fast Reasoning (latest frontier model with 2M context window)
    Grok41FastReasoning,
    /// Grok-4.1 Fast Non-Reasoning (latest frontier model with 2M context window)
    Grok41FastNonReasoning,
    /// Grok-3 (previous generation model with 131k context window)
    Grok3,
    /// Grok-3 Mini (efficient variant with 131k context window)
    Grok3Mini,
    /// Grok Code Fast 1 (optimized for coding tasks)
    GrokCodeFast1,
    /// Grok-2 Vision (multimodal vision model)
    Grok2Vision,
    /// Custom model name (for new models or Grok-compatible endpoints)
    Custom(String),
}

impl Model {
    pub fn as_str(&self) -> &str {
        match self {
            Model::Grok4 => "grok-4-0709",
            Model::Grok4FastReasoning => "grok-4-fast-reasoning",
            Model::Grok4FastNonReasoning => "grok-4-fast-non-reasoning",
            Model::Grok41FastReasoning => "grok-4-1-fast-reasoning",
            Model::Grok41FastNonReasoning => "grok-4-1-fast-non-reasoning",
            Model::Grok3 => "grok-3",
            Model::Grok3Mini => "grok-3-mini",
            Model::GrokCodeFast1 => "grok-code-fast-1",
            Model::Grok2Vision => "grok-2-vision-1212",
            Model::Custom(name) => name,
        }
    }

    /// Create a model from a string. This is a convenience method that always succeeds.
    ///
    /// If the string matches a known model variant, it returns that variant.
    /// Otherwise, it returns `Custom(name)`.
    pub fn from_string(name: impl Into<String>) -> Self {
        let name = name.into();
        match name.as_str() {
            "grok-4-0709" => Model::Grok4,
            "grok-4-fast-reasoning" => Model::Grok4FastReasoning,
            "grok-4-fast-non-reasoning" => Model::Grok4FastNonReasoning,
            "grok-4-1-fast-reasoning" => Model::Grok41FastReasoning,
            "grok-4-1-fast-non-reasoning" => Model::Grok41FastNonReasoning,
            "grok-3" => Model::Grok3,
            "grok-3-mini" => Model::Grok3Mini,
            "grok-code-fast-1" => Model::GrokCodeFast1,
            "grok-2-vision-1212" => Model::Grok2Vision,
            _ => Model::Custom(name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MediaFile;

    #[test]
    fn test_grok_message_content_text_only_serializes_as_string() {
        let msg = ChatMessage::user("hello");
        let content = GrokClient::build_message_content(&msg).expect("content should build");
        let json = serde_json::to_value(&content).expect("content should serialize");
        assert_eq!(json, serde_json::json!("hello"));
    }

    #[test]
    fn test_grok_message_content_with_media_serializes_as_parts() {
        let msg = ChatMessage::user_with_media(
            "describe image",
            vec![MediaFile::from_bytes(b"abc", "image/png")],
        );
        let content = GrokClient::build_message_content(&msg).expect("content should build");
        let json = serde_json::to_value(&content).expect("content should serialize");

        assert_eq!(json[0]["type"], "text");
        assert_eq!(json[0]["text"], "describe image");
        assert_eq!(json[1]["type"], "image_url");
        assert_eq!(json[1]["image_url"]["url"], "data:image/png;base64,YWJj");
    }
}

impl FromStr for Model {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Model::from_string(s))
    }
}

impl From<&str> for Model {
    fn from(s: &str) -> Self {
        Model::from_string(s)
    }
}

impl From<String> for Model {
    fn from(s: String) -> Self {
        Model::from_string(s)
    }
}

/// Configuration for the Grok client
#[derive(Debug, Clone)]
pub struct GrokConfig {
    pub api_key: String,
    pub model: Model,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub timeout: Option<Duration>,
    pub max_retries: Option<usize>,
    /// Custom base URL for Grok-compatible APIs (e.g., local LLMs, proxy endpoints)
    /// Defaults to "https://api.x.ai/v1" if not set
    pub base_url: Option<String>,
}

/// Grok client for generating completions
pub struct GrokClient {
    config: GrokConfig,
    client: reqwest::Client,
}

// Grok API request and response structures (OpenAI-compatible)
#[derive(Debug, Serialize)]
struct GrokChatMessage {
    role: String,
    content: GrokMessageContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum GrokMessageContent {
    Text(String),
    Parts(Vec<GrokMessagePart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GrokMessagePart {
    Text { text: String },
    ImageUrl { image_url: GrokImageUrl },
}

#[derive(Debug, Serialize)]
struct GrokImageUrl {
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

// ResponseFormat and JsonSchemaFormat are now imported from utils

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<GrokChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ResponseMessage {
    role: String,
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatCompletionChoice {
    message: ResponseMessage,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct UsageInfo {
    prompt_tokens: u64,
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
    #[serde(default)]
    usage: Option<UsageInfo>,
    model: Option<String>,
}

impl GrokClient {
    /// Create a new Grok client with the provided API key.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Your xAI API key
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rstructor::GrokClient;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = GrokClient::new("your-xai-api-key")?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(name = "grok_client_new", skip(api_key), fields(model = ?Model::Grok4))]
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(RStructorError::api_error(
                "Grok",
                ApiErrorKind::AuthenticationFailed,
            ));
        }

        info!("Creating new Grok client");
        trace!("API key length: {}", api_key.len());

        let config = GrokConfig {
            api_key,
            model: Model::Grok41FastNonReasoning, // Default to Grok-4.1 Fast Non-Reasoning
            temperature: 0.0,
            max_tokens: None,
            timeout: None,        // Default: no timeout (uses reqwest's default)
            max_retries: Some(3), // Default: 3 retries with error feedback
            base_url: None,       // Default: use official Grok API
        };

        debug!("Grok client created with default configuration");
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    /// Create a new Grok client by reading the API key from the `XAI_API_KEY` environment variable.
    ///
    /// # Errors
    ///
    /// Returns an error if `XAI_API_KEY` is not set.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rstructor::GrokClient;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = GrokClient::from_env()?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(name = "grok_client_from_env", fields(model = ?Model::Grok4))]
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("XAI_API_KEY")
            .map_err(|_| RStructorError::api_error("Grok", ApiErrorKind::AuthenticationFailed))?;

        info!("Creating new Grok client from environment variable");
        trace!("API key length: {}", api_key.len());

        let config = GrokConfig {
            api_key,
            model: Model::Grok41FastNonReasoning, // Default to Grok-4.1 Fast Non-Reasoning
            temperature: 0.0,
            max_tokens: None,
            timeout: None,        // Default: no timeout (uses reqwest's default)
            max_retries: Some(3), // Default: 3 retries with error feedback
            base_url: None,       // Default: use official Grok API
        };

        debug!("Grok client created with default configuration");
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    // Builder methods are generated by the macro below
}

impl GrokClient {
    fn build_message_content(msg: &ChatMessage) -> Result<GrokMessageContent> {
        if msg.media.is_empty() {
            return Ok(GrokMessageContent::Text(msg.content.clone()));
        }

        let mut parts = Vec::new();
        if !msg.content.is_empty() {
            parts.push(GrokMessagePart::Text {
                text: msg.content.clone(),
            });
        }

        for media in &msg.media {
            let url = media_to_url(media, "Grok")?;
            parts.push(GrokMessagePart::ImageUrl {
                image_url: GrokImageUrl {
                    url,
                    detail: Some("auto".to_string()),
                },
            });
        }

        Ok(GrokMessageContent::Parts(parts))
    }

    /// Internal implementation of materialize (without retry logic)
    /// Accepts conversation history for multi-turn interactions.
    /// Returns the data, raw response, and optional usage info.
    ///
    /// Uses Grok's native Structured Outputs with `response_format: json_schema`
    /// for guaranteed schema compliance via constrained decoding.
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
        info!("Generating structured response with Grok (native structured outputs)");

        // Get the schema for type T
        let schema = T::schema();
        let schema_name = T::schema_name().unwrap_or_else(|| "output".to_string());
        trace!(schema_name = schema_name, "Retrieved JSON schema for type");

        // Prepare schema with additionalProperties: false recursively for all nested objects
        let schema_json = prepare_strict_schema(&schema);

        // Build API messages from conversation history
        // With native structured outputs, we don't need to include schema instructions in the prompt
        let api_messages: Vec<GrokChatMessage> = messages
            .iter()
            .map(|msg| {
                Ok(GrokChatMessage {
                    role: msg.role.as_str().to_string(),
                    content: Self::build_message_content(msg)?,
                })
            })
            .collect::<Result<Vec<_>>>()
            .map_err(|e| (e, None))?;

        // Create response format for native structured outputs
        let response_format = ResponseFormat::json_schema(schema_name.clone(), schema_json, None);

        debug!(
            "Building Grok API request with structured outputs (history_len={})",
            api_messages.len()
        );
        let request = ChatCompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: api_messages,
            response_format: Some(response_format),
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.x.ai/v1");
        let url = format!("{}/chat/completions", base_url);
        debug!(url = %url, "Sending request to Grok API with structured outputs");
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| (handle_http_error(e, "Grok"), None))?;

        let response = check_response_status(response, "Grok")
            .await
            .map_err(|e| (e, None))?;

        debug!("Successfully received response from Grok API");
        let completion: ChatCompletionResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse JSON response from Grok API");
            (RStructorError::from(e), None)
        })?;

        if completion.choices.is_empty() {
            error!("Grok API returned empty choices array");
            return Err((
                RStructorError::api_error(
                    "Grok",
                    ApiErrorKind::UnexpectedResponse {
                        details: "No completion choices returned".to_string(),
                    },
                ),
                None,
            ));
        }

        // Extract usage info
        let model_name = completion
            .model
            .clone()
            .unwrap_or_else(|| self.config.model.as_str().to_string());
        let usage = completion
            .usage
            .as_ref()
            .map(|u| TokenUsage::new(model_name.clone(), u.prompt_tokens, u.completion_tokens));

        let message = &completion.choices[0].message;
        trace!(finish_reason = %completion.choices[0].finish_reason, "Completion finish reason");

        // With native structured outputs, the response is in message.content as guaranteed JSON
        if let Some(content) = &message.content {
            let raw_response = content.clone();
            debug!(
                content_len = raw_response.len(),
                "Received structured output response"
            );

            // Parse and validate the response using shared utility
            trace!(json = %raw_response, "Parsing structured output response");
            parse_validate_and_create_output(raw_response, usage)
        } else {
            error!("No content in Grok API response");
            Err((
                RStructorError::api_error(
                    "Grok",
                    ApiErrorKind::UnexpectedResponse {
                        details: "No content in response".to_string(),
                    },
                ),
                None,
            ))
        }
    }
}

// Generate builder methods using macro
crate::impl_client_builder_methods! {
    client_type: GrokClient,
    config_type: GrokConfig,
    model_type: Model,
    provider_name: "Grok"
}

impl GrokClient {
    /// Set a custom base URL for Grok-compatible APIs (e.g., local LLMs, proxy endpoints).
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
}

#[async_trait]
impl LLMClient for GrokClient {
    fn from_env() -> Result<Self> {
        Self::from_env()
    }

    #[instrument(
        name = "grok_materialize",
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
        name = "grok_materialize_with_media",
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
        let initial_messages = vec![ChatMessage::user_with_media(prompt, media.to_vec())];
        let output = generate_with_retry_with_initial_messages(
            |messages: Vec<ChatMessage>| {
                let this = self;
                async move { this.materialize_internal::<T>(&messages).await }
            },
            initial_messages,
            self.config.max_retries,
        )
        .await?;
        Ok(output.data)
    }

    #[instrument(
        name = "grok_materialize_with_metadata",
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
        name = "grok_generate",
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
        name = "grok_generate_with_metadata",
        skip(self, prompt),
        fields(
            model = %self.config.model.as_str(),
            prompt_len = prompt.len()
        )
    )]
    async fn generate_with_metadata(&self, prompt: &str) -> Result<GenerateResult> {
        info!("Generating raw text response with Grok");

        // Build the request without structured outputs
        debug!("Building Grok API request for text generation");
        let request = ChatCompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![GrokChatMessage {
                role: "user".to_string(),
                content: GrokMessageContent::Text(prompt.to_string()),
            }],
            response_format: None,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        // Send the request to Grok/xAI API
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.x.ai/v1");
        let url = format!("{}/chat/completions", base_url);
        debug!(url = %url, "Sending request to Grok API");
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| handle_http_error(e, "Grok"))?;

        // Parse the response
        let response = check_response_status(response, "Grok").await?;

        debug!("Successfully received response from Grok API");
        let completion: ChatCompletionResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse JSON response from Grok API");
            e
        })?;

        if completion.choices.is_empty() {
            error!("Grok API returned empty choices array");
            return Err(RStructorError::api_error(
                "Grok",
                ApiErrorKind::UnexpectedResponse {
                    details: "No completion choices returned".to_string(),
                },
            ));
        }

        // Extract usage info
        let model_name = completion
            .model
            .clone()
            .unwrap_or_else(|| self.config.model.as_str().to_string());
        let usage = completion
            .usage
            .as_ref()
            .map(|u| TokenUsage::new(model_name, u.prompt_tokens, u.completion_tokens));

        let message = &completion.choices[0].message;
        trace!(finish_reason = %completion.choices[0].finish_reason, "Completion finish reason");

        if let Some(content) = &message.content {
            debug!(
                content_len = content.len(),
                "Successfully extracted content from response"
            );
            Ok(GenerateResult::new(content.clone(), usage))
        } else {
            error!("No content in Grok API response");
            Err(RStructorError::api_error(
                "Grok",
                ApiErrorKind::UnexpectedResponse {
                    details: "No content in response".to_string(),
                },
            ))
        }
    }

    /// Fetch available models from Grok's API.
    ///
    /// Returns a list of Grok models available for chat completions.
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.x.ai/v1");
        let url = format!("{}/models", base_url);

        debug!(url = %url, "Fetching available models from Grok");

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| handle_http_error(e, "Grok"))?;

        let response = check_response_status(response, "Grok").await?;

        let json: serde_json::Value = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse models response from Grok");
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
                        // Filter to only Grok models
                        if id.starts_with("grok-") {
                            Some(ModelInfo {
                                id: id.to_string(),
                                name: None,
                                description: None,
                            })
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        debug!(count = models.len(), "Fetched Grok models");
        Ok(models)
    }
}

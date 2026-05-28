use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::model_macro::define_model_enum;
use crate::backend::{
    ChatMessage, GenerateResult, LLMClient, MaterializeInternalOutput, MaterializeResult,
    ModelInfo, ThinkingLevel, TokenUsage, ValidationFailureContext, check_response_status,
    generate_with_retry_with_history, handle_http_error, materialize_with_media_with_retry,
    parse_validate_and_create_output,
};
use crate::error::{ApiErrorKind, RStructorError, Result};
use crate::model::Instructor;

define_model_enum! {
    /// Gemini models available for completion
    ///
    /// For the latest available models and their identifiers, check the
    /// [Google AI Models Documentation](https://ai.google.dev/models).
    /// Use the API endpoint `GET https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY`
    /// to get the current list of available models.
    ///
    /// # Using Custom Models
    ///
    /// You can specify any model name as a string using `Custom` variant or `FromStr`:
    ///
    /// ```rust
    /// use rstructor::GeminiModel;
    /// use std::str::FromStr;
    ///
    /// // Using Custom variant
    /// let model = GeminiModel::Custom("gemini-custom".to_string());
    ///
    /// // Using FromStr (useful for config files)
    /// let model = GeminiModel::from_str("gemini-custom").unwrap();
    ///
    /// // Or use the convenience method
    /// let model = GeminiModel::from_string("gemini-custom");
    /// ```
    pub enum Model {
        /// Gemini 3.5 Flash (latest Flash model, best price/performance, default)
        Gemini35Flash => "gemini-3.5-flash",
        /// Gemini 3.1 Pro Preview (latest Pro model)
        Gemini31ProPreview => "gemini-3.1-pro-preview",
        /// Gemini 3.1 Pro Preview Custom Tools (agentic custom-tool variant)
        Gemini31ProPreviewCustomTools => "gemini-3.1-pro-preview-customtools",
        /// Gemini 3 Flash Preview (preview Flash model)
        Gemini3FlashPreview => "gemini-3-flash-preview",
        /// Gemini 3.1 Flash-Lite (cost-efficient multimodal model)
        Gemini31FlashLite => "gemini-3.1-flash-lite",
        /// Gemini 3.1 Flash-Lite Preview (preview cost-efficient multimodal model)
        Gemini31FlashLitePreview => "gemini-3.1-flash-lite-preview",
        /// Gemini 3 Pro Preview (previous preview Pro model)
        Gemini3ProPreview => "gemini-3-pro-preview",
        /// Gemini 2.5 Pro (latest production Pro model)
        Gemini25Pro => "gemini-2.5-pro",
        /// Gemini 2.5 Flash (latest production Flash model, best price/performance)
        Gemini25Flash => "gemini-2.5-flash",
        /// Gemini 2.5 Flash Lite (smaller, faster variant)
        Gemini25FlashLite => "gemini-2.5-flash-lite",
        /// Gemini 2.5 Flash Image (image generation/analysis tuned variant)
        Gemini25FlashImage => "gemini-2.5-flash-image",
        /// Gemini 2.0 Flash (deprecated 2.0 Flash model)
        Gemini20Flash => "gemini-2.0-flash",
        /// Gemini 2.0 Flash 001 (deprecated specific version of 2.0 Flash)
        Gemini20Flash001 => "gemini-2.0-flash-001",
        /// Gemini 2.0 Flash Lite (deprecated smaller 2.0 Flash variant)
        Gemini20FlashLite => "gemini-2.0-flash-lite",
        /// Gemini 2.0 Flash Lite 001 (deprecated specific version of 2.0 Flash Lite)
        Gemini20FlashLite001 => "gemini-2.0-flash-lite-001",
        /// Gemini Pro Latest (alias for latest Pro model)
        GeminiProLatest => "gemini-pro-latest",
        /// Gemini Flash Latest (alias for latest Flash model)
        GeminiFlashLatest => "gemini-flash-latest",
        /// Gemini Flash Lite Latest (alias for latest Flash Lite model)
        GeminiFlashLiteLatest => "gemini-flash-lite-latest",
    }
}

/// Configuration for the Gemini client
#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub model: Model,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub timeout: Option<Duration>,
    pub max_retries: Option<usize>,
    /// Custom base URL for Gemini-compatible APIs
    /// Defaults to "https://generativelanguage.googleapis.com/v1beta" if not set
    pub base_url: Option<String>,
    /// Thinking level for Gemini 3.x models
    /// Controls the depth of reasoning applied to prompts
    pub thinking_level: Option<ThinkingLevel>,
}

/// Gemini client for generating completions
#[derive(Clone)]
pub struct GeminiClient {
    config: GeminiConfig,
    client: reqwest::Client,
}

// Gemini API request and response structures
#[derive(Debug, Serialize)]
struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Part {
    Text {
        text: String,
    },
    FileData {
        #[serde(rename = "fileData")]
        file_data: FileData,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FileData {
    mime_type: String,
    file_uri: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct InlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct GenerateContentRequest {
    contents: Vec<Content>,
    generation_config: GenerationConfig,
}

#[derive(Debug, Serialize)]
struct GenerationConfig {
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinkingConfig")]
    thinking_config: Option<ThinkingConfig>,
}

#[derive(Debug, Serialize)]
struct ThinkingConfig {
    #[serde(rename = "thinkingLevel")]
    thinking_level: String,
}

#[derive(Debug, Deserialize)]
struct UsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    prompt_token_count: u64,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: u64,
}

#[derive(Debug, Deserialize)]
struct GenerateContentResponse {
    candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata", default)]
    usage_metadata: Option<UsageMetadata>,
    #[serde(rename = "modelVersion", default)]
    model_version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: CandidateContent,
    #[serde(default)]
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct CandidateContent {
    parts: Vec<CandidatePart>,
}

#[derive(Debug, Deserialize)]
struct CandidatePart {
    text: Option<String>,
}

impl GeminiClient {
    /// Create a new Gemini client with the provided API key.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Your Google Gemini API key
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rstructor::GeminiClient;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = GeminiClient::new("your-gemini-api-key")?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(name = "gemini_client_new", skip(api_key), fields(model = ?Model::Gemini35Flash))]
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(RStructorError::api_error(
                "Gemini",
                ApiErrorKind::AuthenticationFailed,
            ));
        }

        let config = GeminiConfig {
            api_key,
            model: Model::Gemini35Flash, // Default to Gemini 3.5 Flash (latest Flash, best price/performance)
            temperature: 0.0,
            max_tokens: None,
            timeout: None,        // Default: no timeout (uses reqwest's default)
            max_retries: Some(3), // Default: 3 retries with error feedback
            base_url: None,       // Default: use official Gemini API
            thinking_level: Some(ThinkingLevel::Low), // Default to Low thinking for Gemini 3.x
        };

        let client = reqwest::Client::new();

        info!(
            model = %config.model.as_str(),
            thinking_level = ?config.thinking_level,
            "Created Gemini client"
        );

        Ok(Self { config, client })
    }

    /// Create a new Gemini client by reading the API key from the `GEMINI_API_KEY` environment variable.
    ///
    /// # Errors
    ///
    /// Returns an error if `GEMINI_API_KEY` is not set.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rstructor::GeminiClient;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = GeminiClient::from_env()?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(name = "gemini_client_from_env", fields(model = ?Model::Gemini35Flash))]
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .map_err(|_| RStructorError::api_error("Gemini", ApiErrorKind::AuthenticationFailed))?;

        let config = GeminiConfig {
            api_key,
            model: Model::Gemini35Flash, // Default to Gemini 3.5 Flash (latest Flash, best price/performance)
            temperature: 0.0,
            max_tokens: None,
            timeout: None,        // Default: no timeout (uses reqwest's default)
            max_retries: Some(3), // Default: 3 retries with error feedback
            base_url: None,       // Default: use official Gemini API
            thinking_level: Some(ThinkingLevel::Low), // Default to Low thinking for Gemini 3.x
        };

        let client = reqwest::Client::new();

        info!(
            model = %config.model.as_str(),
            "Created Gemini client from environment variable"
        );

        Ok(Self { config, client })
    }

    // Builder methods are generated by the macro below
}

impl GeminiClient {
    /// Internal implementation of materialize (without retry logic)
    /// Accepts conversation history for multi-turn interactions.
    /// Returns the data, raw response, and optional usage info.
    ///
    /// Uses Gemini's native Structured Outputs with `response_schema`
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
        info!("Generating structured response with Gemini");

        let schema = T::schema();
        let schema_name = T::schema_name().unwrap_or_else(|| "output".to_string());
        trace!(schema_name = schema_name, "Retrieved JSON schema for type");

        // Build API contents from conversation history
        // With native response_schema, we don't need to include schema instructions in the prompt
        let contents: Vec<Content> = messages
            .iter()
            .map(|msg| {
                // Gemini uses "user" and "model" (not "assistant")
                let role = if msg.role.as_str() == "assistant" {
                    "model"
                } else {
                    msg.role.as_str()
                };
                let mut parts = Vec::new();
                if !msg.content.is_empty() {
                    parts.push(Part::Text {
                        text: msg.content.clone(),
                    });
                }
                for media in &msg.media {
                    if let Some(ref base64_data) = media.data {
                        parts.push(Part::InlineData {
                            inline_data: InlineData {
                                mime_type: media.mime_type.clone(),
                                data: base64_data.clone(),
                            },
                        });
                    } else {
                        parts.push(Part::FileData {
                            file_data: FileData {
                                mime_type: media.mime_type.clone(),
                                file_uri: media.uri.clone(),
                            },
                        });
                    }
                }
                Content {
                    role: Some(role.to_string()),
                    parts,
                }
            })
            .collect();

        // Build thinking config only for Gemini 3.x models
        let is_gemini3 = self.config.model.as_str().starts_with("gemini-3");
        let thinking_config = if is_gemini3 {
            self.config.thinking_level.and_then(|level| {
                level.gemini_level().map(|l| ThinkingConfig {
                    thinking_level: l.to_string(),
                })
            })
        } else {
            None
        };

        // Extract adjacently tagged enum info before transformation (for response conversion)
        let adjacently_tagged_info =
            crate::backend::utils::extract_adjacently_tagged_info(&schema.to_json());

        // Prepare schema for Gemini by stripping unsupported keywords (examples, additionalProperties, etc.)
        let gemini_schema = crate::backend::utils::prepare_gemini_schema(&schema);
        let generation_config = GenerationConfig {
            temperature: self.config.temperature,
            max_output_tokens: self.config.max_tokens,
            response_mime_type: Some("application/json".to_string()),
            response_schema: Some(gemini_schema),
            thinking_config,
        };

        let request = GenerateContentRequest {
            contents,
            generation_config,
        };

        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://generativelanguage.googleapis.com/v1beta");
        let url = format!(
            "{}/models/{}:generateContent",
            base_url,
            self.config.model.as_str()
        );
        debug!(
            url = %url,
            model = %self.config.model.as_str(),
            history_len = messages.len(),
            "Sending request to Gemini API"
        );
        let response = self
            .client
            .post(&url)
            .query(&[("key", &self.config.api_key)])
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| (handle_http_error(e, "Gemini"), None))?;

        let response = check_response_status(response, "Gemini")
            .await
            .map_err(|e| (e, None))?;

        debug!("Successfully received response from Gemini API");
        let completion: GenerateContentResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse JSON response from Gemini API");
            (RStructorError::from(e), None)
        })?;

        if completion.candidates.is_empty() {
            error!("Gemini API returned empty candidates array");
            return Err((
                RStructorError::api_error(
                    "Gemini",
                    ApiErrorKind::UnexpectedResponse {
                        details: "No completion candidates returned".to_string(),
                    },
                ),
                None,
            ));
        }

        // Extract usage info
        let model_name = completion
            .model_version
            .clone()
            .unwrap_or_else(|| self.config.model.as_str().to_string());
        let usage = completion.usage_metadata.as_ref().map(|u| {
            TokenUsage::new(
                model_name.clone(),
                u.prompt_token_count,
                u.candidates_token_count,
            )
        });

        let candidate = &completion.candidates[0];
        trace!(finish_reason = ?candidate.finish_reason, "Completion finish reason");

        let parts = &candidate.content.parts;
        debug!(parts = parts.len(), "Processing candidate content parts");
        for part in parts {
            if let Some(text) = &part.text {
                let mut raw_response = text.clone();
                debug!(content_len = raw_response.len(), "Processing text part");
                // With native response_schema, the response is guaranteed to be valid JSON
                trace!(json = %raw_response, "Parsing structured output response");

                // Transform internally tagged enums back to adjacently tagged format if needed
                if let Some(ref enum_info) = adjacently_tagged_info
                    && let Ok(mut json_value) =
                        serde_json::from_str::<serde_json::Value>(&raw_response)
                {
                    crate::backend::utils::transform_internally_to_adjacently_tagged(
                        &mut json_value,
                        enum_info,
                    );
                    raw_response = serde_json::to_string(&json_value).unwrap_or(raw_response);
                }

                // Parse and validate the response using shared utility
                return parse_validate_and_create_output(raw_response, usage);
            }
        }

        error!("No text content in Gemini response");
        Err((
            RStructorError::api_error(
                "Gemini",
                ApiErrorKind::UnexpectedResponse {
                    details: "No text content in response".to_string(),
                },
            ),
            None,
        ))
    }
}

// Generate builder methods using macro
crate::impl_client_builder_methods! {
    client_type: GeminiClient,
    config_type: GeminiConfig,
    model_type: Model,
    provider_name: "Gemini"
}

impl GeminiClient {
    /// Set a custom base URL for Gemini-compatible APIs.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL without trailing slash (e.g., "http://localhost:1234/v1beta" or "https://api.example.com/v1beta")
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

    /// Set the thinking level for Gemini 3.x models.
    ///
    /// Controls the depth of reasoning the model applies to prompts.
    /// Higher thinking levels provide deeper reasoning but increase latency.
    ///
    /// # Thinking Levels for Gemini 3 Flash
    ///
    /// - `Minimal`: Engages in minimal reasoning, ideal for high-throughput applications
    /// - `Low`: Reduces latency and cost, appropriate for straightforward tasks (default)
    /// - `Medium`: Provides balanced reasoning for most tasks
    /// - `High`: Offers deep reasoning, suitable for complex problem-solving
    ///
    /// # Thinking Levels for Gemini 3.1 Pro
    ///
    /// - `Low`: Minimizes latency and cost, suitable for simple tasks
    /// - `High`: Maximizes reasoning depth for complex tasks
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use rstructor::{GeminiClient, ThinkingLevel};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = GeminiClient::from_env()?
    ///     .thinking_level(ThinkingLevel::High);
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

#[cfg(feature = "streaming")]
impl GeminiClient {
    /// Build the JSON request body for a streaming call, optionally with a
    /// structured-output `response_schema`. (Gemini streams via the
    /// `:streamGenerateContent?alt=sse` endpoint; the body itself is unchanged.)
    fn stream_body(&self, prompt: &str, response_schema: Option<Value>) -> Value {
        let is_gemini3 = self.config.model.as_str().starts_with("gemini-3");
        let thinking_config = if is_gemini3 {
            self.config.thinking_level.and_then(|level| {
                level.gemini_level().map(|l| ThinkingConfig {
                    thinking_level: l.to_string(),
                })
            })
        } else {
            None
        };
        let request = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part::Text {
                    text: prompt.to_string(),
                }],
            }],
            generation_config: GenerationConfig {
                temperature: self.config.temperature,
                max_output_tokens: self.config.max_tokens,
                response_mime_type: response_schema
                    .as_ref()
                    .map(|_| "application/json".to_string()),
                response_schema,
                thinking_config,
            },
        };
        serde_json::to_value(&request).unwrap_or_else(|_| serde_json::json!({}))
    }

    /// Send a streaming request and return the raw SSE response.
    fn send_stream(
        &self,
        body: Value,
    ) -> impl std::future::Future<Output = Result<reqwest::Response>> + Send + 'static {
        let client = self.client.clone();
        let api_key = self.config.api_key.clone();
        let base_url = self
            .config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://generativelanguage.googleapis.com/v1beta".to_string());
        let model = self.config.model.as_str().to_string();
        async move {
            let url = format!("{}/models/{}:streamGenerateContent", base_url, model);
            let resp = client
                .post(&url)
                .query(&[("alt", "sse"), ("key", api_key.as_str())])
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| handle_http_error(e, "Gemini"))?;
            check_response_status(resp, "Gemini").await
        }
    }
}

#[cfg(feature = "tools")]
impl GeminiClient {
    /// Begin a tool-calling request: `client.with_tools(&toolbox).run("...").await?`.
    ///
    /// Requires the `tools` feature.
    pub fn with_tools<'a>(
        &'a self,
        toolbox: &'a crate::backend::tools::Toolbox,
    ) -> crate::backend::tools::ToolRequest<'a, Self> {
        crate::backend::tools::ToolRequest::new(self, toolbox)
    }
}

#[cfg(feature = "tools")]
#[async_trait]
impl crate::backend::tools::ToolRunner for GeminiClient {
    async fn run_tool_loop(
        &self,
        system: Option<&str>,
        prompt: &str,
        toolbox: &crate::backend::tools::Toolbox,
        max_iterations: usize,
    ) -> Result<String> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://generativelanguage.googleapis.com/v1beta");
        crate::backend::tools::run_gemini_tools(
            &self.client,
            base_url,
            &self.config.api_key,
            self.config.model.as_str(),
            self.config.temperature,
            self.config.max_tokens,
            system,
            prompt,
            toolbox,
            max_iterations,
        )
        .await
    }
}

#[async_trait]
impl LLMClient for GeminiClient {
    fn from_env() -> Result<Self> {
        Self::from_env()
    }

    #[instrument(
        name = "gemini_materialize",
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
        name = "gemini_materialize_with_media",
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
        name = "gemini_materialize_with_metadata",
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
        name = "gemini_generate",
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
        name = "gemini_generate_with_metadata",
        skip(self, prompt),
        fields(
            model = %self.config.model.as_str(),
            prompt_len = prompt.len()
        )
    )]
    async fn generate_with_metadata(&self, prompt: &str) -> Result<GenerateResult> {
        info!("Generating raw text response with Gemini");

        // Build thinking config only for Gemini 3.x models
        let is_gemini3 = self.config.model.as_str().starts_with("gemini-3");
        let thinking_config = if is_gemini3 {
            self.config.thinking_level.and_then(|level| {
                level.gemini_level().map(|l| ThinkingConfig {
                    thinking_level: l.to_string(),
                })
            })
        } else {
            None
        };

        // Build the request
        debug!("Building Gemini API request");
        let request = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part::Text {
                    text: prompt.to_string(),
                }],
            }],
            generation_config: GenerationConfig {
                temperature: self.config.temperature,
                max_output_tokens: self.config.max_tokens,
                response_mime_type: None,
                response_schema: None,
                thinking_config,
            },
        };

        // Send the request to Gemini API
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://generativelanguage.googleapis.com/v1beta");
        let url = format!(
            "{}/models/{}:generateContent",
            base_url,
            self.config.model.as_str()
        );
        debug!(
            url = %url,
            model = %self.config.model.as_str(),
            "Sending request to Gemini API"
        );
        let response = self
            .client
            .post(&url)
            .query(&[("key", &self.config.api_key)])
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| handle_http_error(e, "Gemini"))?;

        // Parse the response
        let response = check_response_status(response, "Gemini").await?;

        debug!("Successfully received response from Gemini API");
        let completion: GenerateContentResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse JSON response from Gemini API");
            e
        })?;

        if completion.candidates.is_empty() {
            error!("Gemini API returned empty candidates array");
            return Err(RStructorError::api_error(
                "Gemini",
                ApiErrorKind::UnexpectedResponse {
                    details: "No completion candidates returned".to_string(),
                },
            ));
        }

        // Extract usage info
        let model_name = completion
            .model_version
            .clone()
            .unwrap_or_else(|| self.config.model.as_str().to_string());
        let usage = completion
            .usage_metadata
            .as_ref()
            .map(|u| TokenUsage::new(model_name, u.prompt_token_count, u.candidates_token_count));

        let candidate = &completion.candidates[0];
        trace!(finish_reason = %candidate.finish_reason, "Completion finish reason");

        // Extract the text content
        match candidate
            .content
            .parts
            .first()
            .and_then(|p| p.text.as_ref())
        {
            Some(text) => {
                debug!(
                    content_len = text.len(),
                    "Successfully extracted text content from response"
                );
                Ok(GenerateResult::new(text.clone(), usage))
            }
            None => {
                error!("No text content in Gemini response");
                Err(RStructorError::api_error(
                    "Gemini",
                    ApiErrorKind::UnexpectedResponse {
                        details: "No text content in response".to_string(),
                    },
                ))
            }
        }
    }

    #[cfg(feature = "streaming")]
    fn generate_stream<'a>(&'a self, prompt: &'a str) -> crate::backend::streaming::TextStream<'a>
    where
        Self: Sync,
    {
        let body = self.stream_body(prompt, None);
        crate::backend::streaming::sse_text_stream(
            self.send_stream(body),
            crate::backend::streaming::gemini_delta,
        )
    }

    #[cfg(feature = "streaming")]
    fn materialize_stream<'a, T>(
        &'a self,
        prompt: &'a str,
    ) -> crate::backend::streaming::ObjectStream<'a, T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
        Self: Sync,
    {
        let schema = T::schema();
        // Gemini may return internally-tagged enums; capture the mapping so the
        // final buffer can be transformed back before deserializing into `T`.
        let adjacently_tagged_info =
            crate::backend::utils::extract_adjacently_tagged_info(&schema.to_json());
        let gemini_schema = crate::backend::utils::prepare_gemini_schema(&schema);
        let body = self.stream_body(prompt, Some(gemini_schema));

        let finalize = move |raw: &str| -> Result<T> {
            let mut json = raw.to_string();
            if let Some(ref info) = adjacently_tagged_info
                && let Ok(mut value) = serde_json::from_str::<Value>(raw)
            {
                crate::backend::utils::transform_internally_to_adjacently_tagged(&mut value, info);
                json = serde_json::to_string(&value).unwrap_or(json);
            }
            crate::backend::utils::parse_and_validate_response::<T>(&json).map_err(|(e, _)| e)
        };

        crate::backend::streaming::object_stream_with(
            self.send_stream(body),
            crate::backend::streaming::gemini_delta,
            finalize,
        )
    }

    #[cfg(feature = "streaming")]
    fn materialize_iter<'a, T>(
        &'a self,
        prompt: &'a str,
    ) -> crate::backend::streaming::ItemStream<'a, T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
        Self: Sync,
    {
        let schema = T::schema();
        let adjacently_tagged_info =
            crate::backend::utils::extract_adjacently_tagged_info(&schema.to_json());
        let item_schema = crate::backend::utils::prepare_gemini_schema(&schema);
        let wrapper = crate::backend::streaming::array_wrapper_schema(item_schema, false);
        let body = self.stream_body(prompt, Some(wrapper));

        // Each streamed array element is a `T`; transform internally-tagged enums
        // back before deserializing.
        let finalize = move |mut value: Value| -> Result<T> {
            if let Some(ref info) = adjacently_tagged_info {
                crate::backend::utils::transform_internally_to_adjacently_tagged(&mut value, info);
            }
            let item: T = serde_json::from_value(value)
                .map_err(|e| RStructorError::SerializationError(e.to_string()))?;
            item.validate()?;
            Ok(item)
        };

        crate::backend::streaming::iter_stream(
            self.send_stream(body),
            crate::backend::streaming::gemini_delta,
            finalize,
        )
    }

    /// Fetch available models from Gemini's API.
    ///
    /// Returns a list of Gemini models that support content generation.
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://generativelanguage.googleapis.com/v1beta");
        let url = format!("{}/models?key={}", base_url, self.config.api_key);

        debug!("Fetching available models from Gemini");

        let response = self
            .client
            .get(&url)
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| handle_http_error(e, "Gemini"))?;

        let response = check_response_status(response, "Gemini").await?;

        let json: serde_json::Value = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse models response from Gemini");
            e
        })?;

        let models = json
            .get("models")
            .and_then(|data| data.as_array())
            .map(|models_array| {
                models_array
                    .iter()
                    .filter_map(|model| {
                        let name = model.get("name").and_then(|n| n.as_str())?;
                        // Strip "models/" prefix to get just the model ID
                        let id = name.strip_prefix("models/").unwrap_or(name);

                        // Filter to only Gemini models that support generateContent
                        let supports_generate = model
                            .get("supportedGenerationMethods")
                            .and_then(|m| m.as_array())
                            .map(|methods| {
                                methods
                                    .iter()
                                    .any(|m| m.as_str().is_some_and(|s| s == "generateContent"))
                            })
                            .unwrap_or(false);

                        if id.starts_with("gemini") && supports_generate {
                            let display_name = model
                                .get("displayName")
                                .and_then(|n| n.as_str())
                                .map(|s| s.to_string());
                            let description = model
                                .get("description")
                                .and_then(|n| n.as_str())
                                .map(|s| s.to_string());
                            Some(ModelInfo {
                                id: id.to_string(),
                                name: display_name,
                                description,
                            })
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        debug!(count = models.len(), "Fetched Gemini models");
        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    /// Helper to construct the URL that list_models would use
    fn build_list_models_url(base_url: &str, api_key: &str) -> String {
        format!("{}/models?key={}", base_url, api_key)
    }

    /// Helper to construct the URL that materialize/generate would use
    fn build_generate_url(base_url: &str, model: &str) -> String {
        format!("{}/models/{}:generateContent", base_url, model)
    }

    #[test]
    fn url_construction_consistent_with_default_base_url() {
        let base_url = "https://generativelanguage.googleapis.com/v1beta";
        let api_key = "test-key";
        let model = "gemini-2.5-flash";

        let list_url = build_list_models_url(base_url, api_key);
        let generate_url = build_generate_url(base_url, model);

        assert_eq!(
            list_url,
            "https://generativelanguage.googleapis.com/v1beta/models?key=test-key"
        );
        assert_eq!(
            generate_url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
    }

    #[test]
    fn url_construction_consistent_with_custom_base_url_no_trailing_slash() {
        // User provides base_url without trailing slash (consistent with docs)
        let base_url = "http://localhost:8080/v1beta";
        let api_key = "test-key";
        let model = "gemini-2.5-flash";

        let list_url = build_list_models_url(base_url, api_key);
        let generate_url = build_generate_url(base_url, model);

        // Both should produce valid URLs with /models path
        assert_eq!(list_url, "http://localhost:8080/v1beta/models?key=test-key");
        assert_eq!(
            generate_url,
            "http://localhost:8080/v1beta/models/gemini-2.5-flash:generateContent"
        );

        // Verify that neither URL has the malformed pattern "v1betamodels"
        assert!(!list_url.contains("v1betamodels"));
        assert!(!generate_url.contains("v1betamodels"));
    }

    #[test]
    fn url_construction_with_trailing_slash_base_url() {
        // If user provides trailing slash, we get double slash (but that's their choice)
        // This test documents current behavior
        let base_url = "http://localhost:8080/v1beta/";
        let api_key = "test-key";

        let list_url = build_list_models_url(base_url, api_key);

        // Double slash is expected when user provides trailing slash
        assert_eq!(
            list_url,
            "http://localhost:8080/v1beta//models?key=test-key"
        );
    }
}

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::model_macro::define_model_enum;
use crate::backend::{
    ChatMessage, GenerateResult, LLMClient, MaterializeInternalOutput, MaterializeResult,
    ModelInfo, OpenAICompatibleChatCompletionRequest, OpenAICompatibleChatCompletionResponse,
    OpenAICompatibleChatMessage, OpenAICompatibleMessageContent, ResponseFormat, TokenUsage,
    ValidationFailureContext, check_response_status, convert_openai_compatible_chat_messages,
    generate_with_retry_with_history, handle_http_error, materialize_with_media_with_retry,
    parse_validate_and_create_output, prepare_strict_schema,
};
use crate::error::{ApiErrorKind, RStructorError, Result};
use crate::model::Instructor;

define_model_enum! {
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
    pub enum Model {
        /// Grok 4.3 (latest recommended chat model, multimodal text + image input)
        Grok43 => "grok-4.3",
        /// Grok 4.20 Reasoning (reasoning-optimized variant)
        Grok420Reasoning => "grok-4.20-0309-reasoning",
        /// Grok 4.20 Non-Reasoning (lower-latency non-reasoning variant)
        Grok420NonReasoning => "grok-4.20-0309-non-reasoning",
        /// Grok 4.20 Multi-Agent (agentic multi-agent variant)
        Grok420MultiAgent => "grok-4.20-multi-agent-0309",
        /// Grok Build 0.1 (coding-optimized model; supersedes grok-code-fast-1)
        GrokBuild01 => "grok-build-0.1",
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
#[derive(Clone)]
pub struct GrokClient {
    config: GrokConfig,
    client: reqwest::Client,
}

// Grok uses shared OpenAI-compatible chat completion request/response types.

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
    #[instrument(name = "grok_client_new", skip(api_key), fields(model = ?Model::Grok43))]
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
            model: Model::Grok43, // Default to Grok 4.3 (latest recommended chat model)
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
    #[instrument(name = "grok_client_from_env", fields(model = ?Model::Grok43))]
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("XAI_API_KEY")
            .map_err(|_| RStructorError::api_error("Grok", ApiErrorKind::AuthenticationFailed))?;

        info!("Creating new Grok client from environment variable");
        trace!("API key length: {}", api_key.len());

        let config = GrokConfig {
            api_key,
            model: Model::Grok43, // Default to Grok 4.3 (latest recommended chat model)
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
        let api_messages =
            convert_openai_compatible_chat_messages(messages, "Grok").map_err(|e| (e, None))?;

        // Create response format for native structured outputs
        let response_format = ResponseFormat::json_schema(schema_name.clone(), schema_json, None);

        debug!(
            "Building Grok API request with structured outputs (history_len={})",
            api_messages.len()
        );
        let request = OpenAICompatibleChatCompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: api_messages,
            response_format: Some(response_format),
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            reasoning_effort: None,
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
        let completion: OpenAICompatibleChatCompletionResponse =
            response.json().await.map_err(|e| {
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

#[cfg(feature = "streaming")]
impl GrokClient {
    /// Build the JSON request body for a streaming call (`stream: true`),
    /// optionally with a structured-output `response_format`.
    fn stream_body(
        &self,
        prompt: &str,
        response_format: Option<ResponseFormat>,
    ) -> serde_json::Value {
        let request = OpenAICompatibleChatCompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![OpenAICompatibleChatMessage {
                role: "user".to_string(),
                content: OpenAICompatibleMessageContent::Text(prompt.to_string()),
            }],
            response_format,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            reasoning_effort: None,
        };
        let mut body = serde_json::to_value(&request).unwrap_or_else(|_| serde_json::json!({}));
        body["stream"] = serde_json::Value::Bool(true);
        body
    }

    /// Send a streaming request and return the raw SSE response.
    fn send_stream(
        &self,
        body: serde_json::Value,
    ) -> impl std::future::Future<Output = Result<reqwest::Response>> + Send + 'static {
        let client = self.client.clone();
        let api_key = self.config.api_key.clone();
        let base_url = self
            .config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.x.ai/v1".to_string());
        async move {
            let url = format!("{}/chat/completions", base_url);
            let resp = client
                .post(&url)
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| handle_http_error(e, "Grok"))?;
            check_response_status(resp, "Grok").await
        }
    }
}

#[cfg(feature = "tools")]
impl GrokClient {
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
impl crate::backend::tools::ToolRunner for GrokClient {
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
            .unwrap_or("https://api.x.ai/v1");
        let url = format!("{}/chat/completions", base_url);

        crate::backend::tools::run_openai_compatible_tools(
            &self.client,
            &url,
            &self.config.api_key,
            "Grok",
            self.config.model.as_str(),
            self.config.temperature,
            self.config.max_tokens,
            None,
            system,
            prompt,
            toolbox,
            max_iterations,
        )
        .await
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
        let request = OpenAICompatibleChatCompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![OpenAICompatibleChatMessage {
                role: "user".to_string(),
                content: OpenAICompatibleMessageContent::Text(prompt.to_string()),
            }],
            response_format: None,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            reasoning_effort: None,
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
        let completion: OpenAICompatibleChatCompletionResponse =
            response.json().await.map_err(|e| {
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

    #[cfg(feature = "streaming")]
    fn generate_stream<'a>(&'a self, prompt: &'a str) -> crate::backend::streaming::TextStream<'a>
    where
        Self: Sync,
    {
        let body = self.stream_body(prompt, None);
        crate::backend::streaming::sse_text_stream(
            self.send_stream(body),
            crate::backend::streaming::openai_delta,
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
        let schema_name = T::schema_name().unwrap_or_else(|| "output".to_string());
        let schema_json = prepare_strict_schema(&schema);
        let response_format = ResponseFormat::json_schema(schema_name, schema_json, None);
        let body = self.stream_body(prompt, Some(response_format));
        crate::backend::streaming::object_stream(
            self.send_stream(body),
            crate::backend::streaming::openai_delta,
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
        let item_schema = prepare_strict_schema(&T::schema());
        let wrapper = crate::backend::streaming::array_wrapper_schema(item_schema, true);
        let response_format = ResponseFormat::json_schema("items".to_string(), wrapper, None);
        let body = self.stream_body(prompt, Some(response_format));
        crate::backend::streaming::iter_stream(
            self.send_stream(body),
            crate::backend::streaming::openai_delta,
            crate::backend::streaming::finalize_item::<T>,
        )
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

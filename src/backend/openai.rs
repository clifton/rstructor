use async_trait::async_trait;
use serde::de::DeserializeOwned;
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::model_macro::define_model_enum;
use crate::backend::{
    ChatMessage, GenerateResult, LLMClient, MaterializeInternalOutput, MaterializeResult,
    ModelInfo, OpenAICompatibleChatCompletionRequest, OpenAICompatibleChatCompletionResponse,
    OpenAICompatibleChatMessage, OpenAICompatibleMessageContent, ResponseFormat, ThinkingLevel,
    TokenUsage, ValidationFailureContext, check_response_status,
    convert_openai_compatible_chat_messages, generate_with_retry_with_history, handle_http_error,
    materialize_with_media_with_retry, parse_validate_and_create_output, prepare_strict_schema,
};
use crate::error::{ApiErrorKind, RStructorError, Result};
use crate::model::Instructor;

define_model_enum! {
    /// OpenAI models available for completion
    ///
    /// For the latest available models and their identifiers, check the
    /// [OpenAI Models Documentation](https://platform.openai.com/docs/models).
    ///
    /// # Using Custom Models
    ///
    /// You can specify any model name as a string using `Custom` variant or `FromStr`:
    ///
    /// ```rust
    /// use rstructor::OpenAIModel;
    /// use std::str::FromStr;
    ///
    /// // Using Custom variant
    /// let model = OpenAIModel::Custom("gpt-4-custom".to_string());
    ///
    /// // Using FromStr (useful for config files)
    /// let model = OpenAIModel::from_str("gpt-4-custom").unwrap();
    ///
    /// // Or use the convenience method
    /// let model = OpenAIModel::from_string("gpt-4-custom");
    /// ```
    pub enum Model {
        /// GPT-5.5 Pro (most capable GPT-5.5 model)
        Gpt55Pro => "gpt-5.5-pro",
        /// GPT-5.5 (latest frontier model for complex professional work)
        Gpt55 => "gpt-5.5",
        /// GPT-5.4 Pro (most capable GPT-5.4-class model)
        Gpt54Pro => "gpt-5.4-pro",
        /// GPT-5.4 (more affordable frontier model for complex professional work)
        Gpt54 => "gpt-5.4",
        /// GPT-5.4 Mini (lower-latency, lower-cost GPT-5.4-class model)
        Gpt54Mini => "gpt-5.4-mini",
        /// GPT-5.4 Nano (cheapest GPT-5.4-class model for high-volume tasks)
        Gpt54Nano => "gpt-5.4-nano",
        /// GPT-5.3 Chat Latest (ChatGPT GPT-5.3 model)
        Gpt53ChatLatest => "gpt-5.3-chat-latest",
        /// GPT-5.3 Codex (coding-focused GPT-5.3 model)
        Gpt53Codex => "gpt-5.3-codex",
        /// GPT-5.2 Pro (previous GPT-5.2 pro model)
        Gpt52Pro => "gpt-5.2-pro",
        /// GPT-5.2 (previous GPT-5.2 model)
        Gpt52 => "gpt-5.2",
        /// GPT-5.2 Chat Latest (ChatGPT GPT-5.2 model)
        Gpt52ChatLatest => "gpt-5.2-chat-latest",
        /// GPT-5.2 Codex (coding-focused GPT-5.2 model)
        Gpt52Codex => "gpt-5.2-codex",
        /// GPT-5.1 (GPT-5.1 model)
        Gpt51 => "gpt-5.1",
        /// GPT-5 Chat Latest (ChatGPT GPT-5 model)
        Gpt5ChatLatest => "gpt-5-chat-latest",
        /// GPT-5 Pro (most capable GPT-5 model)
        Gpt5Pro => "gpt-5-pro",
        /// GPT-5 (standard GPT-5 model)
        Gpt5 => "gpt-5",
        /// GPT-5 Nano (smallest GPT-5 model)
        Gpt5Nano => "gpt-5-nano",
        /// GPT-5 Mini (smaller, faster GPT-5 model)
        Gpt5Mini => "gpt-5-mini",
        /// GPT-4.1 (GPT-4.1 model)
        Gpt41 => "gpt-4.1",
        /// GPT-4.1 Mini (smaller GPT-4.1)
        Gpt41Mini => "gpt-4.1-mini",
        /// GPT-4.1 Nano (smallest GPT-4.1)
        Gpt41Nano => "gpt-4.1-nano",
        /// GPT-4o (previous GPT-4o model, optimized for chat)
        Gpt4O => "gpt-4o",
        /// GPT-4o Mini (smaller, faster, more cost-effective version)
        Gpt4OMini => "gpt-4o-mini",
        /// GPT-4 Turbo (high-intelligence model)
        Gpt4Turbo => "gpt-4-turbo",
        /// GPT-4 (standard GPT-4 model)
        Gpt4 => "gpt-4",
        /// GPT-3.5 Turbo (efficient model for simple tasks)
        Gpt35Turbo => "gpt-3.5-turbo",
        /// O4 Mini (previous small reasoning model)
        O4Mini => "o4-mini",
        /// O3 (reasoning model)
        O3 => "o3",
        /// O3 Mini (smaller reasoning model)
        O3Mini => "o3-mini",
        /// O1 (reasoning model optimized for complex problem-solving)
        O1 => "o1",
        /// O1 Pro (most capable reasoning model)
        O1Pro => "o1-pro",
    }
}

/// Configuration for the OpenAI client
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub model: Model,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub timeout: Option<Duration>,
    pub max_retries: Option<usize>,
    /// Custom base URL for OpenAI-compatible APIs (e.g., local LLMs, proxy endpoints)
    /// Defaults to "https://api.openai.com/v1" if not set
    pub base_url: Option<String>,
    /// Thinking level for GPT-5.x models (reasoning effort)
    /// Controls the depth of reasoning applied to prompts
    pub thinking_level: Option<ThinkingLevel>,
}

/// OpenAI client for generating completions
#[derive(Clone)]
pub struct OpenAIClient {
    config: OpenAIConfig,
    client: reqwest::Client,
}

// ResponseFormat and JsonSchemaFormat are imported from utils and shared
// OpenAI-compatible chat completion request/response types are in openai_compatible.rs.

impl OpenAIClient {
    /// Create a new OpenAI client with the provided API key.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Your OpenAI API key
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rstructor::OpenAIClient;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenAIClient::new("your-openai-api-key")?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(name = "openai_client_new", skip(api_key), fields(model = ?Model::Gpt55))]
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(RStructorError::api_error(
                "OpenAI",
                ApiErrorKind::AuthenticationFailed,
            ));
        }
        info!("Creating new OpenAI client");
        trace!("API key length: {}", api_key.len());

        let config = OpenAIConfig {
            api_key,
            model: Model::Gpt55, // Default to GPT-5.5 (latest frontier model)
            temperature: 0.0,
            max_tokens: None,
            timeout: None,        // Default: no timeout (uses reqwest's default)
            max_retries: Some(3), // Default: 3 retries with error feedback
            base_url: None,       // Default: use official OpenAI API
            thinking_level: Some(ThinkingLevel::Medium), // GPT-5.5 defaults to medium reasoning
        };

        debug!("OpenAI client created with default configuration");
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    /// Create a new OpenAI client by reading the API key from the `OPENAI_API_KEY` environment variable.
    ///
    /// # Errors
    ///
    /// Returns an error if `OPENAI_API_KEY` is not set.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rstructor::OpenAIClient;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenAIClient::from_env()?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(name = "openai_client_from_env", fields(model = ?Model::Gpt55))]
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| RStructorError::api_error("OpenAI", ApiErrorKind::AuthenticationFailed))?;

        info!("Creating new OpenAI client from environment variable");
        trace!("API key length: {}", api_key.len());

        let config = OpenAIConfig {
            api_key,
            model: Model::Gpt55, // Default to GPT-5.5 (latest frontier model)
            temperature: 0.0,
            max_tokens: None,
            timeout: None,        // Default: no timeout (uses reqwest's default)
            max_retries: Some(3), // Default: 3 retries with error feedback
            base_url: None,       // Default: use official OpenAI API
            thinking_level: Some(ThinkingLevel::Medium), // GPT-5.5 defaults to medium reasoning
        };

        debug!("OpenAI client created with default configuration");
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    // Builder methods are generated by the macro below
}

// Generate builder methods using macro
crate::impl_client_builder_methods! {
    client_type: OpenAIClient,
    config_type: OpenAIConfig,
    model_type: Model,
    provider_name: "OpenAI"
}

impl OpenAIClient {
    /// Set a custom base URL for OpenAI-compatible APIs (e.g., local LLMs, proxy endpoints).
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL without trailing slash (e.g., "http://localhost:1234/v1" or "https://api.example.com/v1")
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use rstructor::OpenAIClient;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenAIClient::new("api-key")?
    ///     .base_url("http://localhost:1234/v1")
    ///     .model("llama-3.1-70b");
    /// # Ok(())
    /// # }
    /// ```
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

    /// Set the thinking level for GPT-5.x models (reasoning effort).
    ///
    /// Controls the depth of reasoning the model applies to prompts.
    /// Higher thinking levels provide deeper reasoning but increase latency and cost.
    ///
    /// Note: When reasoning is enabled (any level except `Off`), temperature is
    /// automatically set to 1.0 as required by the API.
    ///
    /// # Reasoning Effort Levels
    ///
    /// - `Off`: No extended reasoning (maps to "none")
    /// - `Minimal`: Light reasoning (maps to "low")
    /// - `Low`: Standard reasoning (maps to "low", default)
    /// - `Medium`: Balanced reasoning (maps to "medium")
    /// - `High`: Deep reasoning (maps to "high")
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use rstructor::{OpenAIClient, ThinkingLevel};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenAIClient::from_env()?
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

    /// Internal implementation of materialize (without retry logic)
    /// Accepts conversation history for multi-turn interactions.
    /// Returns the data, raw response, and optional usage info.
    ///
    /// Uses OpenAI's native Structured Outputs with `response_format: json_schema`
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
        info!("Generating structured response with OpenAI (native structured outputs)");

        // Get the schema for type T
        let schema = T::schema();
        let schema_name = T::schema_name().unwrap_or_else(|| "output".to_string());
        // Avoid calling to_string() in trace to prevent potential stack overflow with complex schemas
        trace!(schema_name = schema_name, "Retrieved JSON schema for type");

        // Prepare schema with additionalProperties: false recursively for all nested objects
        let schema_json = prepare_strict_schema(&schema);

        // Create response format with JSON schema (strict mode)
        let response_format = ResponseFormat::json_schema(
            schema_name.clone(),
            schema_json,
            Some("Output in the specified format. Include ALL required fields and follow the schema exactly.".to_string()),
        );

        // Build reasoning_effort for GPT-5.x models
        let is_gpt5 = self.config.model.as_str().starts_with("gpt-5");
        let reasoning_effort = if is_gpt5 {
            self.config
                .thinking_level
                .and_then(|level| level.openai_reasoning_effort().map(|s| s.to_string()))
        } else {
            None
        };

        // GPT-5.x with reasoning requires temperature=1.0
        let effective_temp = if reasoning_effort.is_some() {
            1.0
        } else {
            self.config.temperature
        };

        // Convert ChatMessage to OpenAI's format
        let api_messages =
            convert_openai_compatible_chat_messages(messages, "OpenAI").map_err(|e| (e, None))?;

        // Build the request with native structured outputs
        debug!(
            "Building OpenAI API request with structured outputs (history_len={})",
            api_messages.len()
        );
        let request = OpenAICompatibleChatCompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: api_messages,
            response_format: Some(response_format),
            temperature: effective_temp,
            max_tokens: self.config.max_tokens,
            reasoning_effort,
        };

        // Send the request to OpenAI
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1");
        let url = format!("{}/chat/completions", base_url);
        debug!(url = %url, "Sending request to OpenAI API");
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| (handle_http_error(e, "OpenAI"), None))?;

        // Parse the response
        let response = check_response_status(response, "OpenAI")
            .await
            .map_err(|e| (e, None))?;

        debug!("Successfully received response from OpenAI");
        let completion: OpenAICompatibleChatCompletionResponse =
            response.json().await.map_err(|e| {
                error!(error = %e, "Failed to parse JSON response from OpenAI");
                (RStructorError::from(e), None)
            })?;

        if completion.choices.is_empty() {
            error!("OpenAI returned empty choices array");
            return Err((
                RStructorError::api_error(
                    "OpenAI",
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

        // With structured outputs, the response is in message.content as guaranteed-valid JSON
        if let Some(content) = &message.content {
            let raw_response = content.clone();
            debug!(
                content_len = raw_response.len(),
                "Structured output received from OpenAI"
            );

            // Parse and validate the response using shared utility
            parse_validate_and_create_output(raw_response, usage)
        } else {
            error!("No content in OpenAI response");
            Err((
                RStructorError::api_error(
                    "OpenAI",
                    ApiErrorKind::UnexpectedResponse {
                        details: "No content in response".to_string(),
                    },
                ),
                None,
            ))
        }
    }
}

#[cfg(feature = "tools")]
impl OpenAIClient {
    /// Begin a tool-calling request. The model may call the [`Toolbox`](crate::Toolbox)'s
    /// tools (whose results are fed back) until it produces a final text answer.
    ///
    /// Requires the `tools` feature.
    ///
    /// ```no_run
    /// # use rstructor::{OpenAIClient, Toolbox, FnTool, Instructor};
    /// # use serde::{Serialize, Deserialize};
    /// # use serde_json::json;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// #[derive(Instructor, Serialize, Deserialize)]
    /// struct WeatherArgs { city: String }
    ///
    /// let toolbox = Toolbox::new().with(FnTool::new(
    ///     "get_weather",
    ///     "Get the current weather for a city",
    ///     |args: WeatherArgs| async move { Ok(json!({ "city": args.city, "temp_f": 72 })) },
    /// ));
    ///
    /// let client = OpenAIClient::from_env()?;
    /// let answer = client.with_tools(&toolbox).run("What's the weather in Paris?").await?;
    /// println!("{answer}");
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tools<'a>(
        &'a self,
        toolbox: &'a crate::backend::tools::Toolbox,
    ) -> crate::backend::tools::ToolRequest<'a, Self> {
        crate::backend::tools::ToolRequest::new(self, toolbox)
    }
}

#[cfg(feature = "tools")]
#[async_trait]
impl crate::backend::tools::ToolRunner for OpenAIClient {
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
            .unwrap_or("https://api.openai.com/v1");
        let url = format!("{}/chat/completions", base_url);

        // GPT-5.x models require temperature=1.0. `reasoning_effort` combined with
        // function tools is rejected on /v1/chat/completions, so it is omitted for
        // the tool loop.
        let is_gpt5 = self.config.model.as_str().starts_with("gpt-5");
        let effective_temp = if is_gpt5 {
            1.0
        } else {
            self.config.temperature
        };

        crate::backend::tools::run_openai_compatible_tools(
            &self.client,
            &url,
            &self.config.api_key,
            "OpenAI",
            self.config.model.as_str(),
            effective_temp,
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
impl LLMClient for OpenAIClient {
    fn from_env() -> Result<Self> {
        Self::from_env()
    }

    #[instrument(
        name = "openai_materialize",
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
        name = "openai_materialize_with_media",
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
        name = "openai_materialize_with_metadata",
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
        name = "openai_generate",
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
        name = "openai_generate_with_metadata",
        skip(self, prompt),
        fields(
            model = %self.config.model.as_str(),
            prompt_len = prompt.len()
        )
    )]
    async fn generate_with_metadata(&self, prompt: &str) -> Result<GenerateResult> {
        info!("Generating raw text response with OpenAI");

        // Build reasoning_effort for GPT-5.x models
        let is_gpt5 = self.config.model.as_str().starts_with("gpt-5");
        let reasoning_effort = if is_gpt5 {
            self.config
                .thinking_level
                .and_then(|level| level.openai_reasoning_effort().map(|s| s.to_string()))
        } else {
            None
        };

        // GPT-5.x with reasoning requires temperature=1.0
        let effective_temp = if reasoning_effort.is_some() {
            1.0
        } else {
            self.config.temperature
        };

        // Build the request for text generation (no structured output)
        debug!("Building OpenAI API request for text generation");
        let request = OpenAICompatibleChatCompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![OpenAICompatibleChatMessage {
                role: "user".to_string(),
                content: OpenAICompatibleMessageContent::Text(prompt.to_string()),
            }],
            response_format: None,
            temperature: effective_temp,
            max_tokens: self.config.max_tokens,
            reasoning_effort,
        };

        // Send the request to OpenAI
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1");
        let url = format!("{}/chat/completions", base_url);
        debug!(url = %url, "Sending request to OpenAI API");
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| handle_http_error(e, "OpenAI"))?;

        // Parse the response
        let response = check_response_status(response, "OpenAI").await?;

        debug!("Successfully received response from OpenAI");
        let completion: OpenAICompatibleChatCompletionResponse =
            response.json().await.map_err(|e| {
                error!(error = %e, "Failed to parse JSON response from OpenAI");
                e
            })?;

        if completion.choices.is_empty() {
            error!("OpenAI returned empty choices array");
            return Err(RStructorError::api_error(
                "OpenAI",
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
            error!("No content in OpenAI response");
            Err(RStructorError::api_error(
                "OpenAI",
                ApiErrorKind::UnexpectedResponse {
                    details: "No content in response".to_string(),
                },
            ))
        }
    }

    /// Fetch available models from OpenAI's API.
    ///
    /// Returns a list of GPT models available for chat completions.
    /// Filters out embedding, whisper, and other non-chat models.
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let base_url = self
            .config
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1");
        let url = format!("{}/models", base_url);

        debug!(url = %url, "Fetching available models from OpenAI");

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| handle_http_error(e, "OpenAI"))?;

        let response = check_response_status(response, "OpenAI").await?;

        let json: serde_json::Value = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse models response from OpenAI");
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
                        // Filter to only GPT models (chat completion models)
                        if id.starts_with("gpt-")
                            || id.starts_with("o1")
                            || id.starts_with("o3")
                            || id.starts_with("o4")
                        {
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

        debug!(count = models.len(), "Fetched OpenAI models");
        Ok(models)
    }
}

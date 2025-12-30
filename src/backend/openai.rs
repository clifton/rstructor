use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::str::FromStr;
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::{
    GenerateResult, LLMClient, MaterializeResult, ModelInfo, ThinkingLevel, TokenUsage,
    check_response_status, generate_with_retry, handle_http_error,
};
use crate::error::{ApiErrorKind, RStructorError, Result};
use crate::model::Instructor;

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Model {
    /// GPT-5.2 (latest GPT-5 model)
    Gpt52,
    /// GPT-5 Chat Latest (latest GPT-5 model for chat)
    Gpt5ChatLatest,
    /// GPT-5 Pro (most capable GPT-5 model)
    Gpt5Pro,
    /// GPT-5 (standard GPT-5 model)
    Gpt5,
    /// GPT-5 Mini (smaller, faster GPT-5 model)
    Gpt5Mini,
    /// GPT-4o (latest GPT-4 model, optimized for chat)
    Gpt4O,
    /// GPT-4o Mini (smaller, faster, more cost-effective version)
    Gpt4OMini,
    /// GPT-4 Turbo (high-intelligence model)
    Gpt4Turbo,
    /// GPT-4 (standard GPT-4 model)
    Gpt4,
    /// GPT-3.5 Turbo (efficient model for simple tasks)
    Gpt35Turbo,
    /// O1 (reasoning model optimized for complex problem-solving)
    O1,
    /// O1 Mini (smaller reasoning model)
    O1Mini,
    /// O1 Pro (most capable reasoning model)
    O1Pro,
    /// Custom model name (for new models, local LLMs, or OpenAI-compatible endpoints)
    Custom(String),
}

impl Model {
    pub fn as_str(&self) -> &str {
        match self {
            Model::Gpt52 => "gpt-5.2",
            Model::Gpt5ChatLatest => "gpt-5-chat-latest",
            Model::Gpt5Pro => "gpt-5-pro",
            Model::Gpt5 => "gpt-5",
            Model::Gpt5Mini => "gpt-5-mini",
            Model::Gpt4O => "gpt-4o",
            Model::Gpt4OMini => "gpt-4o-mini",
            Model::Gpt4Turbo => "gpt-4-turbo",
            Model::Gpt4 => "gpt-4",
            Model::Gpt35Turbo => "gpt-3.5-turbo",
            Model::O1 => "o1",
            Model::O1Mini => "o1-mini",
            Model::O1Pro => "o1-pro",
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
            "gpt-5.2" => Model::Gpt52,
            "gpt-5-chat-latest" => Model::Gpt5ChatLatest,
            "gpt-5-pro" => Model::Gpt5Pro,
            "gpt-5" => Model::Gpt5,
            "gpt-5-mini" => Model::Gpt5Mini,
            "gpt-4o" => Model::Gpt4O,
            "gpt-4o-mini" => Model::Gpt4OMini,
            "gpt-4-turbo" => Model::Gpt4Turbo,
            "gpt-4" => Model::Gpt4,
            "gpt-3.5-turbo" => Model::Gpt35Turbo,
            "o1" => Model::O1,
            "o1-mini" => Model::O1Mini,
            "o1-pro" => Model::O1Pro,
            _ => Model::Custom(name),
        }
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

/// Configuration for the OpenAI client
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub model: Model,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub timeout: Option<Duration>,
    pub max_retries: Option<usize>,
    pub include_error_feedback: Option<bool>,
    /// Custom base URL for OpenAI-compatible APIs (e.g., local LLMs, proxy endpoints)
    /// Defaults to "https://api.openai.com/v1" if not set
    pub base_url: Option<String>,
    /// Thinking level for GPT-5.x models (reasoning effort)
    /// Controls the depth of reasoning applied to prompts
    pub thinking_level: Option<ThinkingLevel>,
}

/// OpenAI client for generating completions
pub struct OpenAIClient {
    config: OpenAIConfig,
    client: reqwest::Client,
}

// OpenAI API request and response structures
#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct FunctionDef {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    functions: Option<Vec<FunctionDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<Value>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// Reasoning effort for GPT-5.x models
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct FunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ResponseMessage {
    role: String,
    content: Option<String>,
    function_call: Option<FunctionCall>,
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
    #[instrument(name = "openai_client_new", skip(api_key), fields(model = ?Model::Gpt52))]
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
            model: Model::Gpt52, // Default to GPT-5.2 (latest GPT-5)
            temperature: 0.0,
            max_tokens: None,
            timeout: None,     // Default: no timeout (uses reqwest's default)
            max_retries: None, // Default: no retries (configure via .max_retries())
            include_error_feedback: None, // Default: include error feedback in retry prompts
            base_url: None,    // Default: use official OpenAI API
            thinking_level: Some(ThinkingLevel::Low), // Default to Low thinking for GPT-5.x
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
    #[instrument(name = "openai_client_from_env", fields(model = ?Model::Gpt52))]
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| RStructorError::api_error("OpenAI", ApiErrorKind::AuthenticationFailed))?;

        info!("Creating new OpenAI client from environment variable");
        trace!("API key length: {}", api_key.len());

        let config = OpenAIConfig {
            api_key,
            model: Model::Gpt52, // Default to GPT-5.2 (latest GPT-5)
            temperature: 0.0,
            max_tokens: None,
            timeout: None,     // Default: no timeout (uses reqwest's default)
            max_retries: None, // Default: no retries (configure via .max_retries())
            include_error_feedback: None, // Default: include error feedback in retry prompts
            base_url: None,    // Default: use official OpenAI API
            thinking_level: Some(ThinkingLevel::Low), // Default to Low thinking for GPT-5.x
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
    /// Returns both the data and optional usage info
    async fn materialize_internal<T>(&self, prompt: &str) -> Result<(T, Option<TokenUsage>)>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        info!("Generating structured response with OpenAI");

        // Get the schema for type T
        let schema = T::schema();
        let schema_name = T::schema_name().unwrap_or_else(|| "output".to_string());
        // Avoid calling to_string() in trace to prevent potential stack overflow with complex schemas
        trace!(schema_name = schema_name, "Retrieved JSON schema for type");

        // Create function definition with the schema
        let function = FunctionDef {
            name: schema_name.clone(),
            description: "Output in the specified format. IMPORTANT: 1) Include ALL required fields. 2) For enum fields, use EXACTLY one of the values allowed in the description. 3) Include all nested objects with ALL their required fields. 4) For arrays of objects, always provide complete objects with all required fields - never arrays of strings. 5) Include multiple items (2-3) in each array.".to_string(),
            parameters: schema.to_json(),
        };

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

        // Build the request
        debug!("Building OpenAI API request with function calling");
        let request = ChatCompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            functions: Some(vec![function]),
            function_call: Some(json!({ "name": schema_name })),
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
        let completion: ChatCompletionResponse = response.json().await.map_err(|e| {
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
            .map(|u| TokenUsage::new(model_name.clone(), u.prompt_tokens, u.completion_tokens));

        let message = &completion.choices[0].message;
        trace!(finish_reason = %completion.choices[0].finish_reason, "Completion finish reason");

        // Extract the function arguments JSON
        if let Some(function_call) = &message.function_call {
            debug!(
                function_name = %function_call.name,
                args_len = function_call.arguments.len(),
                "Function call received from OpenAI"
            );

            // Parse the arguments JSON string into our target type
            let result: T = match serde_json::from_str(&function_call.arguments) {
                Ok(parsed) => parsed,
                Err(e) => {
                    let error_msg = format!(
                        "Failed to parse response: {}\nPartial JSON: {}",
                        e, &function_call.arguments
                    );
                    error!(
                        error = %e,
                        partial_json = %function_call.arguments,
                        "JSON parsing error"
                    );
                    return Err(RStructorError::ValidationError(error_msg));
                }
            };

            // Apply any custom validation
            if let Err(e) = result.validate() {
                error!(error = ?e, "Custom validation failed");
                return Err(e);
            }

            info!("Successfully generated and validated structured data");
            Ok((result, usage))
        } else {
            // If no function call, try to extract from content if available
            if let Some(content) = &message.content {
                warn!(
                    content_len = content.len(),
                    "No function call in response, attempting to parse content as JSON"
                );

                // Try to extract JSON from the content (assuming the model might have returned JSON directly)
                use crate::backend::extract_json_from_markdown;
                let json_content = extract_json_from_markdown(content);
                let result: T = match serde_json::from_str(&json_content) {
                    Ok(parsed) => parsed,
                    Err(e) => {
                        let error_msg = format!(
                            "Failed to parse response content: {}\nPartial JSON: {}",
                            e, &json_content
                        );
                        error!(
                            error = %e,
                            content = %json_content,
                            "Failed to parse content as JSON"
                        );
                        return Err(RStructorError::ValidationError(error_msg));
                    }
                };

                // Apply any custom validation
                if let Err(e) = result.validate() {
                    error!(error = ?e, "Custom validation failed");
                    return Err(e);
                }

                info!("Successfully generated and validated structured data from content");
                Ok((result, usage))
            } else {
                error!("No function call or content in OpenAI response");
                Err(RStructorError::api_error(
                    "OpenAI",
                    ApiErrorKind::UnexpectedResponse {
                        details: "No function call or content in response".to_string(),
                    },
                ))
            }
        }
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
        let (result, _usage) = generate_with_retry(
            |prompt_owned: String| {
                let this = self;
                async move { this.materialize_internal::<T>(&prompt_owned).await }
            },
            prompt,
            self.config.max_retries,
            self.config.include_error_feedback,
        )
        .await?;
        Ok(result)
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
        let (result, usage) = generate_with_retry(
            |prompt_owned: String| {
                let this = self;
                async move { this.materialize_internal::<T>(&prompt_owned).await }
            },
            prompt,
            self.config.max_retries,
            self.config.include_error_feedback,
        )
        .await?;
        Ok(MaterializeResult::new(result, usage))
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

        // Build the request without functions
        debug!("Building OpenAI API request for text generation");
        let request = ChatCompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            functions: None,
            function_call: None,
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
        let completion: ChatCompletionResponse = response.json().await.map_err(|e| {
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
                        if id.starts_with("gpt-") || id.starts_with("o1") || id.starts_with("o3") {
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

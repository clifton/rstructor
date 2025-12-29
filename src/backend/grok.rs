use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::str::FromStr;
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::{
    GenerateResult, LLMClient, MaterializeResult, TokenUsage, check_response_status,
    extract_json_from_markdown, generate_with_retry, handle_http_error,
};
use crate::error::{RStructorError, Result};
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
    /// Grok-2-1212 (enhanced accuracy and instruction adherence)
    Grok21212,
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
            Model::Grok21212 => "grok-2-1212",
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
            "grok-2-1212" => Model::Grok21212,
            "grok-2-vision-1212" => Model::Grok2Vision,
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

/// Configuration for the Grok client
#[derive(Debug, Clone)]
pub struct GrokConfig {
    pub api_key: String,
    pub model: Model,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub timeout: Option<Duration>,
    pub max_retries: Option<usize>,
    pub include_error_feedback: Option<bool>,
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
            return Err(RStructorError::ApiError(
                "API key cannot be empty. Use GrokClient::from_env() to read from XAI_API_KEY environment variable.".to_string(),
            ));
        }

        info!("Creating new Grok client");
        trace!("API key length: {}", api_key.len());

        let config = GrokConfig {
            api_key,
            model: Model::Grok41FastNonReasoning, // Default to Grok-4.1 Fast Non-Reasoning
            temperature: 0.0,
            max_tokens: None,
            timeout: None,     // Default: no timeout (uses reqwest's default)
            max_retries: None, // Default: no retries (configure via .max_retries())
            include_error_feedback: None, // Default: include error feedback in retry prompts
            base_url: None,    // Default: use official Grok API
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
        let api_key = std::env::var("XAI_API_KEY").map_err(|_| {
            RStructorError::ApiError("XAI_API_KEY environment variable is not set".to_string())
        })?;

        info!("Creating new Grok client from environment variable");
        trace!("API key length: {}", api_key.len());

        let config = GrokConfig {
            api_key,
            model: Model::Grok41FastNonReasoning, // Default to Grok-4.1 Fast Non-Reasoning
            temperature: 0.0,
            max_tokens: None,
            timeout: None,     // Default: no timeout (uses reqwest's default)
            max_retries: None, // Default: no retries (configure via .max_retries())
            include_error_feedback: None, // Default: include error feedback in retry prompts
            base_url: None,    // Default: use official Grok API
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
    /// Returns both the data and optional usage info
    async fn materialize_internal<T>(&self, prompt: &str) -> Result<(T, Option<TokenUsage>)>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        info!("Generating structured response with Grok");

        // Get the schema for type T
        let schema = T::schema();
        let schema_name = T::schema_name().unwrap_or_else(|| "output".to_string());
        trace!(schema_name = schema_name, "Retrieved JSON schema for type");

        let schema_str =
            serde_json::to_string(&schema.to_json()).unwrap_or_else(|_| "{}".to_string());
        debug!("Building structured prompt with schema");

        let structured_prompt = format!(
            "You are a helpful assistant that outputs JSON. The user wants data in the following JSON schema format:\n\n{}\n\nYou MUST provide your answer in valid JSON format according to the schema above.\n1. Include ALL required fields\n2. Format as a complete, valid JSON object\n3. DO NOT include explanations, just return the JSON\n4. Make sure to use double quotes for all strings and property names\n5. For enum fields, use EXACTLY one of the values listed in the descriptions\n6. Include ALL nested objects with all their required fields\n7. For array fields:\n   - MOST IMPORTANT: When an array items.type is \"object\", provide an array of complete objects with ALL required fields\n   - DO NOT provide arrays of strings when arrays of objects are required\n   - Include multiple items (at least 2-3) in each array\n   - Every object in an array must match the schema for that object type\n8. Follow type specifications EXACTLY (string, number, boolean, array, object)\n\nUser query: {}",
            schema_str, prompt
        );

        let function = FunctionDef {
            name: schema_name.clone(),
            description: "Output in the specified format. IMPORTANT: 1) Include ALL required fields. 2) For enum fields, use EXACTLY one of the values allowed in the description. 3) Include all nested objects with ALL their required fields. 4) For arrays of objects, always provide complete objects with all required fields - never arrays of strings. 5) Include multiple items (2-3) in each array.".to_string(),
            parameters: schema.to_json(),
        };

        debug!("Building Grok API request with function calling");
        let request = ChatCompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: structured_prompt,
            }],
            functions: Some(vec![function]),
            function_call: Some(json!({ "name": schema_name })),
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

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

        let response = check_response_status(response, "Grok").await?;

        debug!("Successfully received response from Grok API");
        let completion: ChatCompletionResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse JSON response from Grok API");
            e
        })?;

        if completion.choices.is_empty() {
            error!("Grok API returned empty choices array");
            return Err(RStructorError::ApiError(
                "No completion choices returned".to_string(),
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

        if let Some(function_call) = &message.function_call {
            debug!(
                function_name = %function_call.name,
                args_len = function_call.arguments.len(),
                "Function call received from Grok API"
            );

            let json_content = extract_json_from_markdown(&function_call.arguments);
            trace!(json = %json_content, "Attempting to parse function call arguments as JSON");
            let result: T = match serde_json::from_str(&json_content) {
                Ok(parsed) => parsed,
                Err(e) => {
                    let error_msg = format!(
                        "Failed to parse response: {}\nPartial JSON: {}",
                        e, json_content
                    );
                    error!(
                        error = %e,
                        partial_json = %json_content,
                        "JSON parsing error"
                    );
                    return Err(RStructorError::ValidationError(error_msg));
                }
            };

            if let Err(e) = result.validate() {
                error!(error = ?e, "Custom validation failed");
                return Err(e);
            }

            info!("Successfully generated and validated structured data");
            Ok((result, usage))
        } else if let Some(content) = &message.content {
            warn!(
                content_len = content.len(),
                "No function call in response, attempting to parse content as JSON"
            );

            let json_content = extract_json_from_markdown(content);
            trace!(json = %json_content, "Attempting to parse response as JSON");
            let result: T = match serde_json::from_str(&json_content) {
                Ok(parsed) => parsed,
                Err(e) => {
                    let error_msg = format!(
                        "Failed to parse response content: {}\nPartial JSON: {}",
                        e, json_content
                    );
                    error!(
                        error = %e,
                        content = %json_content,
                        "Failed to parse content as JSON"
                    );
                    return Err(RStructorError::ValidationError(error_msg));
                }
            };

            if let Err(e) = result.validate() {
                error!(error = ?e, "Custom validation failed");
                return Err(e);
            }

            info!("Successfully generated and validated structured data from content");
            Ok((result, usage))
        } else {
            error!("No function call or content in Grok API response");
            Err(RStructorError::ApiError(
                "No function call or content in response".to_string(),
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

        // Build the request without functions
        debug!("Building Grok API request for text generation");
        let request = ChatCompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            functions: None,
            function_call: None,
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
            return Err(RStructorError::ApiError(
                "No completion choices returned".to_string(),
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
            Err(RStructorError::ApiError(
                "No content in response".to_string(),
            ))
        }
    }
}

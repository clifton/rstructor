use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::str::FromStr;
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::{
    LLMClient, check_response_status, extract_json_from_markdown, generate_with_retry,
    handle_http_error,
};
use crate::error::{RStructorError, Result};
use crate::model::Instructor;

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Model {
    /// Gemini 3 Pro Preview (latest preview Pro model)
    Gemini3ProPreview,
    /// Gemini 3 Flash Preview (latest preview Flash model)
    Gemini3FlashPreview,
    /// Gemini 2.5 Pro (latest production Pro model)
    Gemini25Pro,
    /// Gemini 2.5 Flash (latest production Flash model, best price/performance)
    Gemini25Flash,
    /// Gemini 2.5 Flash Lite (smaller, faster variant)
    Gemini25FlashLite,
    /// Gemini 2.0 Flash (stable 2.0 Flash model)
    Gemini20Flash,
    /// Gemini 2.0 Flash 001 (specific version of 2.0 Flash)
    Gemini20Flash001,
    /// Gemini 2.0 Flash Experimental (experimental 2.0 Flash variant)
    Gemini20FlashExp,
    /// Gemini 2.0 Flash Lite (smaller 2.0 Flash variant)
    Gemini20FlashLite,
    /// Gemini 2.0 Pro Experimental (experimental 2.0 Pro model)
    Gemini20ProExp,
    /// Gemini Pro Latest (alias for latest Pro model)
    GeminiProLatest,
    /// Gemini Flash Latest (alias for latest Flash model)
    GeminiFlashLatest,
    /// Custom model name (for new models or Gemini-compatible endpoints)
    Custom(String),
}

impl Model {
    pub fn as_str(&self) -> &str {
        match self {
            Model::Gemini3ProPreview => "gemini-3-pro-preview",
            Model::Gemini3FlashPreview => "gemini-3-flash-preview",
            Model::Gemini25Pro => "gemini-2.5-pro",
            Model::Gemini25Flash => "gemini-2.5-flash",
            Model::Gemini25FlashLite => "gemini-2.5-flash-lite",
            Model::Gemini20Flash => "gemini-2.0-flash",
            Model::Gemini20Flash001 => "gemini-2.0-flash-001",
            Model::Gemini20FlashExp => "gemini-2.0-flash-exp",
            Model::Gemini20FlashLite => "gemini-2.0-flash-lite",
            Model::Gemini20ProExp => "gemini-2.0-pro-exp",
            Model::GeminiProLatest => "gemini-pro-latest",
            Model::GeminiFlashLatest => "gemini-flash-latest",
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
            "gemini-3-pro-preview" => Model::Gemini3ProPreview,
            "gemini-3-flash-preview" => Model::Gemini3FlashPreview,
            "gemini-2.5-pro" => Model::Gemini25Pro,
            "gemini-2.5-flash" => Model::Gemini25Flash,
            "gemini-2.5-flash-lite" => Model::Gemini25FlashLite,
            "gemini-2.0-flash" => Model::Gemini20Flash,
            "gemini-2.0-flash-001" => Model::Gemini20Flash001,
            "gemini-2.0-flash-exp" => Model::Gemini20FlashExp,
            "gemini-2.0-flash-lite" => Model::Gemini20FlashLite,
            "gemini-2.0-pro-exp" => Model::Gemini20ProExp,
            "gemini-pro-latest" => Model::GeminiProLatest,
            "gemini-flash-latest" => Model::GeminiFlashLatest,
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

use crate::backend::ThinkingLevel;

/// Configuration for the Gemini client
#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub model: Model,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub timeout: Option<Duration>,
    pub max_retries: Option<usize>,
    pub include_error_feedback: Option<bool>,
    /// Custom base URL for Gemini-compatible APIs
    /// Defaults to "https://generativelanguage.googleapis.com/v1beta" if not set
    pub base_url: Option<String>,
    /// Thinking level for Gemini 3 models
    /// Controls the depth of reasoning applied to prompts
    pub thinking_level: Option<ThinkingLevel>,
}

/// Gemini client for generating completions
pub struct GeminiClient {
    config: GeminiConfig,
    client: reqwest::Client,
}

// Gemini API request and response structures
#[derive(Debug, Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
struct Part {
    text: String,
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
struct GenerateContentResponse {
    candidates: Vec<Candidate>,
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
    #[instrument(name = "gemini_client_new", skip(api_key), fields(model = ?Model::Gemini25Flash))]
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(RStructorError::ApiError(
                "API key cannot be empty. Use GeminiClient::from_env() to read from GEMINI_API_KEY environment variable.".to_string(),
            ));
        }

        let config = GeminiConfig {
            api_key,
            model: Model::Gemini3FlashPreview, // Default to Gemini 3 Flash Preview (latest)
            temperature: 0.0,
            max_tokens: None,
            timeout: None,     // Default: no timeout (uses reqwest's default)
            max_retries: None, // Default: no retries (configure via .max_retries())
            include_error_feedback: None, // Default: include error feedback in retry prompts
            base_url: None,    // Default: use official Gemini API
            thinking_level: Some(ThinkingLevel::Low), // Default to Low thinking for Gemini 3
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
    #[instrument(name = "gemini_client_from_env", fields(model = ?Model::Gemini3FlashPreview))]
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY").map_err(|_| {
            RStructorError::ApiError("GEMINI_API_KEY environment variable is not set".to_string())
        })?;

        let config = GeminiConfig {
            api_key,
            model: Model::Gemini3FlashPreview, // Default to Gemini 3 Flash Preview (latest)
            temperature: 0.0,
            max_tokens: None,
            timeout: None,     // Default: no timeout (uses reqwest's default)
            max_retries: None, // Default: no retries (configure via .max_retries())
            include_error_feedback: None, // Default: include error feedback in retry prompts
            base_url: None,    // Default: use official Gemini API
            thinking_level: Some(ThinkingLevel::Low), // Default to Low thinking for Gemini 3
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
    async fn materialize_internal<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        info!("Generating structured response with Gemini");

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

        // Build thinking config only for Gemini 3 models
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

        let generation_config = GenerationConfig {
            temperature: self.config.temperature,
            max_output_tokens: self.config.max_tokens,
            response_mime_type: Some("application/json".to_string()),
            response_schema: Some(schema.to_json()),
            thinking_config,
        };

        let request = GenerateContentRequest {
            contents: vec![Content {
                parts: vec![Part {
                    text: structured_prompt,
                }],
            }],
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

        let response = check_response_status(response, "Gemini").await?;

        debug!("Successfully received response from Gemini API");
        let completion: GenerateContentResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse JSON response from Gemini API");
            e
        })?;

        if completion.candidates.is_empty() {
            error!("Gemini API returned empty candidates array");
            return Err(RStructorError::ApiError(
                "No completion candidates returned".to_string(),
            ));
        }

        let candidate = &completion.candidates[0];
        trace!(finish_reason = ?candidate.finish_reason, "Completion finish reason");

        let parts = &candidate.content.parts;
        debug!(parts = parts.len(), "Processing candidate content parts");
        for part in parts {
            if let Some(text) = &part.text {
                debug!(content_len = text.len(), "Processing text part");
                let json_content = extract_json_from_markdown(text);
                trace!(json = %json_content, "Attempting to parse response as JSON");
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
                return Ok(result);
            }
        }

        error!("No text content in Gemini response");
        Err(RStructorError::ApiError(
            "No text content in response".to_string(),
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

    /// Set the thinking level for Gemini 3 models.
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
    /// # Thinking Levels for Gemini 3 Pro
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
        name = "gemini_generate",
        skip(self, prompt),
        fields(
            model = %self.config.model.as_str(),
            prompt_len = prompt.len()
        )
    )]
    async fn generate(&self, prompt: &str) -> Result<String> {
        info!("Generating raw text response with Gemini");

        // Build thinking config only for Gemini 3 models
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
                parts: vec![Part {
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
            return Err(RStructorError::ApiError(
                "No completion candidates returned".to_string(),
            ));
        }

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
                Ok(text.clone())
            }
            None => {
                error!("No text content in Gemini response");
                Err(RStructorError::ApiError(
                    "No text content in response".to_string(),
                ))
            }
        }
    }
}

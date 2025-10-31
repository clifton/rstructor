use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::LLMClient;
use crate::error::{RStructorError, Result};
use crate::model::Instructor;

/// Gemini models available for completion
///
/// For the latest available models and their identifiers, check the
/// [Google AI Models Documentation](https://ai.google.dev/models).
/// Use the API endpoint `GET https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY`
/// to get the current list of available models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
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
}

impl Model {
    pub fn as_str(&self) -> &'static str {
        match self {
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
        }
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
            model: Model::Gemini25Flash, // Default to Gemini 2.5 Flash (best price/performance for structured outputs)
            temperature: 0.0,
            max_tokens: None,
            timeout: None, // Default: no timeout (uses reqwest's default)
        };

        let client = reqwest::Client::new();

        info!(
            model = %config.model.as_str(),
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
    #[instrument(name = "gemini_client_from_env", fields(model = ?Model::Gemini25Flash))]
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY").map_err(|_| {
            RStructorError::ApiError("GEMINI_API_KEY environment variable is not set".to_string())
        })?;

        let config = GeminiConfig {
            api_key,
            model: Model::Gemini25Flash, // Default to Gemini 2.5 Flash (best price/performance for structured outputs)
            temperature: 0.0,
            max_tokens: None,
            timeout: None, // Default: no timeout (uses reqwest's default)
        };

        let client = reqwest::Client::new();

        info!(
            model = %config.model.as_str(),
            "Created Gemini client from environment variable"
        );

        Ok(Self { config, client })
    }

    /// Set the model to use
    #[instrument(skip(self))]
    pub fn model(mut self, model: Model) -> Self {
        debug!(
            previous_model = ?self.config.model,
            new_model = ?model,
            "Setting model"
        );
        self.config.model = model;
        self
    }

    /// Set the temperature (0.0 to 1.0)
    #[instrument(skip(self))]
    pub fn temperature(mut self, temperature: f32) -> Self {
        debug!(
            previous_temperature = self.config.temperature,
            new_temperature = temperature,
            "Setting temperature"
        );
        self.config.temperature = temperature;
        self
    }

    /// Set the maximum number of tokens to generate
    #[instrument(skip(self))]
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        debug!(
            previous_max_tokens = ?self.config.max_tokens,
            new_max_tokens = max_tokens,
            "Setting max_tokens"
        );
        // Ensure max_tokens is at least 1 to avoid API errors
        self.config.max_tokens = Some(max_tokens.max(1));
        self
    }

    /// Set a timeout for requests
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rstructor::{GeminiClient, GeminiModel};
    /// # use std::time::Duration;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = GeminiClient::new("your-api-key")?
    ///     .model(GeminiModel::Gemini25Flash)
    ///     .temperature(0.0)
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
impl LLMClient for GeminiClient {
    fn from_env() -> Result<Self> {
        Self::from_env()
    }
    #[instrument(
        name = "gemini_generate_struct",
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
        info!("Generating structured response with Gemini");

        // Get the schema for type T
        let schema = T::schema();
        let schema_name = T::schema_name().unwrap_or_else(|| "output".to_string());
        trace!(schema_name = schema_name, "Retrieved JSON schema for type");

        // Build the structured prompt with schema instructions
        let schema_str =
            serde_json::to_string(&schema.to_json()).unwrap_or_else(|_| "{}".to_string());
        debug!("Building structured prompt with schema");
        let structured_prompt = format!(
            "You are a helpful assistant that outputs JSON. The user wants data in the following JSON schema format:\n\n{}\n\nYou MUST provide your answer in valid JSON format according to the schema above.\n1. Include ALL required fields\n2. Format as a complete, valid JSON object\n3. DO NOT include explanations, just return the JSON\n4. Make sure to use double quotes for all strings and property names\n5. For enum fields, use EXACTLY one of the values listed in the descriptions\n6. Include ALL nested objects with all their required fields\n7. For array fields:\n   - MOST IMPORTANT: When an array items.type is \"object\", provide an array of complete objects with ALL required fields\n   - DO NOT provide arrays of strings when arrays of objects are required\n   - Include multiple items (at least 2-3) in each array\n   - Every object in an array must match the schema for that object type\n8. Follow type specifications EXACTLY (string, number, boolean, array, object)\n\nUser query: {}",
            schema_str, prompt
        );

        // Build the request with JSON schema for structured output
        debug!("Building Gemini API request");
        let generation_config = GenerationConfig {
            temperature: self.config.temperature,
            max_output_tokens: self.config.max_tokens,
            response_mime_type: Some("application/json".to_string()),
            response_schema: Some(schema.to_json()),
        };

        let request = GenerateContentRequest {
            contents: vec![Content {
                parts: vec![Part {
                    text: structured_prompt,
                }],
            }],
            generation_config,
        };

        // Send the request to Gemini API
        debug!(
            model = %self.config.model.as_str(),
            max_output_tokens = ?self.config.max_tokens,
            "Sending request to Gemini API"
        );
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            self.config.model.as_str()
        );
        let response = self
            .client
            .post(&url)
            .query(&[("key", &self.config.api_key)])
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "HTTP request to Gemini API failed");
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
                "Gemini API returned error response"
            );
            return Err(RStructorError::ApiError(format!(
                "Gemini API error: {}",
                error_text
            )));
        }

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
        let text = match candidate
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
                text
            }
            None => {
                error!("No text content in Gemini response");
                return Err(RStructorError::ApiError(
                    "No text content in response".to_string(),
                ));
            }
        };

        // Try to parse the content as JSON
        // First, try to extract JSON from markdown code blocks if present
        let json_content = extract_json_from_markdown(text);
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

        info!("Successfully generated and validated structured response");
        Ok(result)
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
        info!("Generating text response with Gemini");

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
            },
        };

        // Send the request to Gemini API
        debug!(
            model = %self.config.model.as_str(),
            "Sending request to Gemini API"
        );
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            self.config.model.as_str()
        );
        let response = self
            .client
            .post(&url)
            .query(&[("key", &self.config.api_key)])
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "HTTP request to Gemini API failed");
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
                "Gemini API returned error response"
            );
            return Err(RStructorError::ApiError(format!(
                "Gemini API error: {}",
                error_text
            )));
        }

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

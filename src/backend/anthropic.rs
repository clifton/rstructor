use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::LLMClient;
use crate::error::{RStructorError, Result};
use crate::model::Instructor;

/// Anthropic models available for completion
#[derive(Debug, Clone)]
pub enum AnthropicModel {
    Claude3Haiku,
    Claude3Sonnet,
    Claude3Opus,
    Claude35Sonnet, // Added Claude 3.5 Sonnet
}

impl AnthropicModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            AnthropicModel::Claude3Haiku => "claude-3-haiku-20240307",
            AnthropicModel::Claude3Sonnet => "claude-3-sonnet-20240229",
            AnthropicModel::Claude3Opus => "claude-3-opus-20240229",
            AnthropicModel::Claude35Sonnet => "claude-3-5-sonnet-20240620", // Claude 3.5 Sonnet model ID
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
    /// Create a new Anthropic client with default configuration
    #[instrument(name = "anthropic_client_new", skip(api_key), fields(model = ?AnthropicModel::Claude35Sonnet))]
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        info!("Creating new Anthropic client");
        trace!("API key length: {}", api_key.len());

        let config = AnthropicConfig {
            api_key,
            model: AnthropicModel::Claude35Sonnet, // Default to Claude 3.5 Sonnet
            temperature: 0.0,
            max_tokens: None,
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

    /// Build the client (chainable after configuration)
    #[instrument(skip(self))]
    pub fn build(self) -> Self {
        info!(
            model = ?self.config.model,
            temperature = self.config.temperature,
            max_tokens = ?self.config.max_tokens,
            "Anthropic client configuration complete"
        );
        self
    }
}

#[async_trait]
impl LLMClient for AnthropicClient {
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
        trace!(
            "Retrieved JSON schema for type"
        );
        // Get schema as JSON string - avoid Display impl which might cause recursion
        let schema_str = serde_json::to_string(&schema.to_json()).unwrap_or_else(|_| "{}".to_string());
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
                e
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
        trace!(json = %content, "Attempting to parse response as JSON");
        let result: T = match serde_json::from_str(content) {
            Ok(parsed) => parsed,
            Err(e) => {
                let error_msg = format!(
                    "Failed to parse response as JSON: {}\nPartial JSON: {}",
                    e, content
                );
                error!(
                    error = %e,
                    content = %content,
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
                e
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

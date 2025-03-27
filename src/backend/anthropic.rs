use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

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
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let config = AnthropicConfig {
            api_key: api_key.into(),
            model: AnthropicModel::Claude35Sonnet, // Default to Claude 3.5 Sonnet
            temperature: 0.0,
            max_tokens: None,
        };

        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    /// Set the model to use
    pub fn model(mut self, model: AnthropicModel) -> Self {
        self.config.model = model;
        self
    }

    /// Set the temperature (0.0 to 1.0, lower = more deterministic)
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp;
        self
    }

    /// Set the maximum tokens to generate
    pub fn max_tokens(mut self, max: u32) -> Self {
        // Ensure max_tokens is at least 1 to avoid API errors
        self.config.max_tokens = Some(max.max(1));
        self
    }

    /// Build the client (chainable after configuration)
    pub fn build(self) -> Self {
        self
    }
}

#[async_trait]
impl LLMClient for AnthropicClient {
    async fn generate_struct<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        // Get the schema for type T
        let schema = T::schema();

        // Create a prompt that includes the schema
        let schema_str = schema.to_string();
        let structured_prompt = format!(
            "You are a helpful assistant that outputs JSON. The user wants data in the following JSON schema format:\n\n{}\n\nYou MUST provide your answer in valid JSON format according to the schema above.\n1. Include ALL required fields\n2. Format as a complete, valid JSON object\n3. DO NOT include explanations, just return the JSON\n4. Make sure to use double quotes for all strings and property names\n\nUser query: {}",
            schema_str, prompt
        );

        // Build the request
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
        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        // Parse the response
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(RStructorError::ApiError(format!(
                "Anthropic API error: {}",
                error_text
            )));
        }

        let completion: CompletionResponse = response.json().await?;

        // Extract the content, assuming the first block is text containing JSON
        let content = completion
            .content
            .iter()
            .find(|block| block.block_type == "text")
            .map(|block| &block.text)
            .ok_or_else(|| RStructorError::ApiError("No text content in response".to_string()))?;

        // Try to parse the content as JSON
        let result: T = serde_json::from_str(content).map_err(|e| {
            RStructorError::ValidationError(format!("Failed to parse response as JSON: {}", e))
        })?;

        // Apply any custom validation
        result.validate()?;

        Ok(result)
    }

    async fn generate(&self, prompt: &str) -> Result<String> {
        // Build the request
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
        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        // Parse the response
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(RStructorError::ApiError(format!(
                "Anthropic API error: {}",
                error_text
            )));
        }

        let completion: CompletionResponse = response.json().await?;

        // Extract the content
        let content: String = completion
            .content
            .iter()
            .filter(|block| block.block_type == "text")
            .map(|block| block.text.clone())
            .collect::<Vec<String>>()
            .join("");

        if content.is_empty() {
            return Err(RStructorError::ApiError(
                "No text content in response".to_string(),
            ));
        }

        Ok(content)
    }
}

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::backend::LLMClient;
use crate::error::{RStructorError, Result};
use crate::model::Instructor;

/// OpenAI models available for completion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    Gpt35Turbo,
    Gpt4,
    Gpt4Turbo,
    Gpt4O,
}

impl Model {
    pub fn as_str(&self) -> &'static str {
        match self {
            Model::Gpt35Turbo => "gpt-3.5-turbo",
            Model::Gpt4 => "gpt-4",
            Model::Gpt4Turbo => "gpt-4-turbo-preview",
            Model::Gpt4O => "gpt-4o",
        }
    }
}

/// Configuration for the OpenAI client
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub model: Model,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
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
    functions: Option<Vec<FunctionDef>>,
    function_call: Option<Value>,
    temperature: f32,
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
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
}

impl OpenAIClient {
    /// Create a new OpenAI client with default configuration
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let config = OpenAIConfig {
            api_key: api_key.into(),
            model: Model::Gpt4O, // Default to GPT-4o
            temperature: 0.0,
            max_tokens: None,
        };

        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    /// Set the model to use
    pub fn model(mut self, model: Model) -> Self {
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
        self.config.max_tokens = Some(max);
        self
    }

    /// Build the client (chainable after configuration)
    pub fn build(self) -> Self {
        self
    }
}

#[async_trait]
impl LLMClient for OpenAIClient {
    async fn generate_struct<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        // Get the schema for type T
        let schema = T::schema();
        let schema_name = T::schema_name().unwrap_or_else(|| "output".to_string());

        // Create function definition with the schema
        let function = FunctionDef {
            name: schema_name.clone(),
            description: "Output in the specified format".to_string(),
            parameters: schema.to_json().clone(),
        };

        // Build the request
        let request = ChatCompletionRequest {
            model: self.config.model.as_str().to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            functions: Some(vec![function]),
            function_call: Some(json!({ "name": schema_name })),
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        // Send the request to OpenAI
        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        // Parse the response
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(RStructorError::ApiError(format!(
                "OpenAI API error: {}",
                error_text
            )));
        }

        let completion: ChatCompletionResponse = response.json().await?;
        if completion.choices.is_empty() {
            return Err(RStructorError::ApiError(
                "No completion choices returned".to_string(),
            ));
        }

        let message = &completion.choices[0].message;

        // Extract the function arguments JSON
        if let Some(function_call) = &message.function_call {
            // Parse the arguments JSON string into our target type
            let result: T = serde_json::from_str(&function_call.arguments).map_err(|e| {
                RStructorError::ValidationError(format!(
                    "Failed to parse response: {}\nPartial JSON: {}",
                    e, &function_call.arguments
                ))
            })?;

            // Apply any custom validation
            result.validate()?;

            Ok(result)
        } else {
            // If no function call, try to extract from content if available
            if let Some(content) = &message.content {
                // Try to extract JSON from the content (assuming the model might have returned JSON directly)
                let result: T = serde_json::from_str(content).map_err(|e| {
                    RStructorError::ValidationError(format!(
                        "Failed to parse response content: {}\nPartial JSON: {}",
                        e, content
                    ))
                })?;

                // Apply any custom validation
                result.validate()?;

                Ok(result)
            } else {
                Err(RStructorError::ApiError(
                    "No function call or content in response".to_string(),
                ))
            }
        }
    }

    async fn generate(&self, prompt: &str) -> Result<String> {
        // Build the request without functions
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

        // Send the request to OpenAI
        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        // Parse the response
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(RStructorError::ApiError(format!(
                "OpenAI API error: {}",
                error_text
            )));
        }

        let completion: ChatCompletionResponse = response.json().await?;
        if completion.choices.is_empty() {
            return Err(RStructorError::ApiError(
                "No completion choices returned".to_string(),
            ));
        }

        let message = &completion.choices[0].message;

        if let Some(content) = &message.content {
            Ok(content.clone())
        } else {
            Err(RStructorError::ApiError(
                "No content in response".to_string(),
            ))
        }
    }
}

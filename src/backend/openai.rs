use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

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
    pub timeout: Option<Duration>,
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
    #[instrument(name = "openai_client_new", skip(api_key), fields(model = ?Model::Gpt4O))]
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        info!("Creating new OpenAI client");
        trace!("API key length: {}", api_key.len());

        let config = OpenAIConfig {
            api_key,
            model: Model::Gpt4O, // Default to GPT-4o
            temperature: 0.0,
            max_tokens: None,
            timeout: None, // Default: no timeout (uses reqwest's default)
        };

        debug!("OpenAI client created with default configuration");
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    /// Set the model to use
    #[instrument(skip(self))]
    pub fn model(mut self, model: Model) -> Self {
        debug!(previous_model = ?self.config.model, new_model = ?model, "Setting OpenAI model");
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
        debug!(previous_max = ?self.config.max_tokens, new_max = max, "Setting max_tokens");
        self.config.max_tokens = Some(max);
        self
    }

    /// Set the timeout for HTTP requests.
    ///
    /// This sets the timeout for both the connection and the entire request.
    /// The timeout applies to each HTTP request made by the client.
    ///
    /// # Arguments
    ///
    /// * `timeout_secs` - Timeout in seconds (e.g., 2.5 for 2.5 seconds)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rstructor::OpenAIClient;
    /// # use std::time::Duration;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = OpenAIClient::new("api-key")?
    ///     .with_timeout(30.0)  // 30 second timeout
    ///     .build();
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self))]
    pub fn with_timeout(mut self, timeout_secs: f64) -> Self {
        let timeout = Duration::from_secs_f64(timeout_secs);
        debug!(
            previous_timeout = ?self.config.timeout,
            new_timeout = ?timeout,
            "Setting timeout"
        );
        self.config.timeout = Some(timeout);
        self
    }

    /// Build the client (chainable after configuration)
    #[instrument(skip(self))]
    pub fn build(mut self) -> Self {
        info!(
            model = ?self.config.model,
            temperature = self.config.temperature,
            max_tokens = ?self.config.max_tokens,
            timeout = ?self.config.timeout,
            "OpenAI client configuration complete"
        );

        // Configure reqwest client with timeout if specified
        let mut client_builder = reqwest::Client::builder();
        if let Some(timeout) = self.config.timeout {
            client_builder = client_builder.timeout(timeout);
        }
        self.client = client_builder.build().unwrap_or_else(|e| {
            warn!(error = %e, "Failed to build reqwest client with timeout, using default");
            reqwest::Client::new()
        });

        self
    }
}

#[async_trait]
impl LLMClient for OpenAIClient {
    #[instrument(
        name = "openai_generate_struct",
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
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        // Send the request to OpenAI
        debug!("Sending request to OpenAI API");
        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "HTTP request to OpenAI failed");
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
                "OpenAI API returned error response"
            );
            return Err(RStructorError::ApiError(format!(
                "OpenAI API error: {}",
                error_text
            )));
        }

        debug!("Successfully received response from OpenAI");
        let completion: ChatCompletionResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse JSON response from OpenAI");
            e
        })?;

        if completion.choices.is_empty() {
            error!("OpenAI returned empty choices array");
            return Err(RStructorError::ApiError(
                "No completion choices returned".to_string(),
            ));
        }

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
            Ok(result)
        } else {
            // If no function call, try to extract from content if available
            if let Some(content) = &message.content {
                warn!(
                    content_len = content.len(),
                    "No function call in response, attempting to parse content as JSON"
                );

                // Try to extract JSON from the content (assuming the model might have returned JSON directly)
                let result: T = match serde_json::from_str(content) {
                    Ok(parsed) => parsed,
                    Err(e) => {
                        let error_msg = format!(
                            "Failed to parse response content: {}\nPartial JSON: {}",
                            e, content
                        );
                        error!(
                            error = %e,
                            content = %content,
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
                Ok(result)
            } else {
                error!("No function call or content in OpenAI response");
                Err(RStructorError::ApiError(
                    "No function call or content in response".to_string(),
                ))
            }
        }
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
        info!("Generating raw text response with OpenAI");

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
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        // Send the request to OpenAI
        debug!("Sending request to OpenAI API");
        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "HTTP request to OpenAI failed");
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
                "OpenAI API returned error response"
            );
            return Err(RStructorError::ApiError(format!(
                "OpenAI API error: {}",
                error_text
            )));
        }

        debug!("Successfully received response from OpenAI");
        let completion: ChatCompletionResponse = response.json().await.map_err(|e| {
            error!(error = %e, "Failed to parse JSON response from OpenAI");
            e
        })?;

        if completion.choices.is_empty() {
            error!("OpenAI returned empty choices array");
            return Err(RStructorError::ApiError(
                "No completion choices returned".to_string(),
            ));
        }

        let message = &completion.choices[0].message;
        trace!(finish_reason = %completion.choices[0].finish_reason, "Completion finish reason");

        if let Some(content) = &message.content {
            debug!(
                content_len = content.len(),
                "Successfully extracted content from response"
            );
            Ok(content.clone())
        } else {
            error!("No content in OpenAI response");
            Err(RStructorError::ApiError(
                "No content in response".to_string(),
            ))
        }
    }
}

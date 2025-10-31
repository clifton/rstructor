use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::backend::LLMClient;
use crate::error::{RStructorError, Result};
use crate::model::Instructor;

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

/// Grok models available for completion
///
/// These are convenience variants for common Grok models.
/// For the latest available models and their identifiers, check the
/// [xAI Models Documentation](https://docs.x.ai/docs/models).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// Grok-4 (latest flagship model with 256k context window)
    Grok4,
    /// Grok-4 Fast Reasoning (faster variant optimized for reasoning tasks)
    Grok4FastReasoning,
    /// Grok-4 Fast Non-Reasoning (faster variant optimized for non-reasoning tasks)
    Grok4FastNonReasoning,
    /// Grok-3 (advanced model with enhanced reasoning)
    Grok3,
    /// Grok-3 Mini (efficient variant with 131k context window)
    Grok3Mini,
    /// Grok Code Fast 1 (optimized for coding tasks)
    GrokCodeFast1,
    /// Grok-2-1212 (enhanced accuracy and instruction adherence)
    Grok21212,
    /// Grok-2 Vision (multimodal vision model)
    Grok2Vision,
}

impl Model {
    pub fn as_str(&self) -> &'static str {
        match self {
            Model::Grok4 => "grok-4-0709",
            Model::Grok4FastReasoning => "grok-4-fast-reasoning",
            Model::Grok4FastNonReasoning => "grok-4-fast-non-reasoning",
            Model::Grok3 => "grok-3",
            Model::Grok3Mini => "grok-3-mini",
            Model::GrokCodeFast1 => "grok-code-fast-1",
            Model::Grok21212 => "grok-2-1212",
            Model::Grok2Vision => "grok-2-vision-1212",
        }
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
            model: Model::Grok4, // Default to Grok-4 (latest flagship model)
            temperature: 0.0,
            max_tokens: None,
            timeout: None, // Default: no timeout (uses reqwest's default)
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
            model: Model::Grok4, // Default to Grok-4 (latest flagship model)
            temperature: 0.0,
            max_tokens: None,
            timeout: None, // Default: no timeout (uses reqwest's default)
        };

        debug!("Grok client created with default configuration");
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    /// Set the model to use
    #[instrument(skip(self))]
    pub fn model(mut self, model: Model) -> Self {
        debug!(previous_model = ?self.config.model, new_model = ?model, "Setting Grok model");
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
        // Ensure max_tokens is at least 1 to avoid API errors
        self.config.max_tokens = Some(max.max(1));
        self
    }

    /// Set the timeout for HTTP requests.
    ///
    /// This sets the timeout for both the connection and the entire request.
    /// The timeout applies to each HTTP request made by the client.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Timeout duration (e.g., `Duration::from_secs(30)` for 30 seconds)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use rstructor::GrokClient;
    /// # use std::time::Duration;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = GrokClient::from_env()?  // Reads from XAI_API_KEY env var
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
impl LLMClient for GrokClient {
    fn from_env() -> Result<Self> {
        Self::from_env()
    }
    #[instrument(
        name = "grok_generate_struct",
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
        info!("Generating structured response with Grok");

        // Get the schema for type T
        let schema = T::schema();
        let schema_name = T::schema_name().unwrap_or_else(|| "output".to_string());
        // Avoid calling to_string() in trace to prevent potential stack overflow with complex schemas
        trace!(schema_name = schema_name, "Retrieved JSON schema for type");

        // Get schema as JSON string - avoid Display impl which might cause recursion
        let schema_str =
            serde_json::to_string(&schema.to_json()).unwrap_or_else(|_| "{}".to_string());
        debug!("Building structured prompt with schema");

        // Enhance the prompt to explicitly request JSON output
        // This helps when Grok doesn't respect function calling
        let structured_prompt = format!(
            "You are a helpful assistant that outputs JSON. The user wants data in the following JSON schema format:\n\n{}\n\nYou MUST provide your answer in valid JSON format according to the schema above.\n1. Include ALL required fields\n2. Format as a complete, valid JSON object\n3. DO NOT include explanations, just return the JSON\n4. Make sure to use double quotes for all strings and property names\n5. For enum fields, use EXACTLY one of the values listed in the descriptions\n6. Include ALL nested objects with all their required fields\n7. For array fields:\n   - MOST IMPORTANT: When an array items.type is \"object\", provide an array of complete objects with ALL required fields\n   - DO NOT provide arrays of strings when arrays of objects are required\n   - Include multiple items (at least 2-3) in each array\n   - Every object in an array must match the schema for that object type\n8. Follow type specifications EXACTLY (string, number, boolean, array, object)\n\nUser query: {}",
            schema_str, prompt
        );

        // Create function definition with the schema
        let function = FunctionDef {
            name: schema_name.clone(),
            description: "Output in the specified format. IMPORTANT: 1) Include ALL required fields. 2) For enum fields, use EXACTLY one of the values allowed in the description. 3) Include all nested objects with ALL their required fields. 4) For arrays of objects, always provide complete objects with all required fields - never arrays of strings. 5) Include multiple items (2-3) in each array.".to_string(),
            parameters: schema.to_json(),
        };

        // Build the request
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

        // Send the request to Grok/xAI API
        debug!("Sending request to Grok API");
        let response = self
            .client
            .post("https://api.x.ai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "HTTP request to Grok API failed");
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
                "Grok API returned error response"
            );
            return Err(RStructorError::ApiError(format!(
                "Grok API error: {}",
                error_text
            )));
        }

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

        let message = &completion.choices[0].message;
        trace!(finish_reason = %completion.choices[0].finish_reason, "Completion finish reason");

        // Extract the function arguments JSON
        if let Some(function_call) = &message.function_call {
            debug!(
                function_name = %function_call.name,
                args_len = function_call.arguments.len(),
                "Function call received from Grok API"
            );

            // Parse the arguments JSON string into our target type
            // First, try to extract JSON from markdown code blocks if present
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

                // First, try to extract JSON from markdown code blocks if present
                let json_content = extract_json_from_markdown(content);
                trace!(json = %json_content, "Attempting to parse response as JSON");

                // Try to extract JSON from the content (assuming the model might have returned JSON directly)
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

                // Apply any custom validation
                if let Err(e) = result.validate() {
                    error!(error = ?e, "Custom validation failed");
                    return Err(e);
                }

                info!("Successfully generated and validated structured data from content");
                Ok(result)
            } else {
                error!("No function call or content in Grok API response");
                Err(RStructorError::ApiError(
                    "No function call or content in response".to_string(),
                ))
            }
        }
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
        debug!("Sending request to Grok API");
        let response = self
            .client
            .post("https://api.x.ai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "HTTP request to Grok API failed");
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
                "Grok API returned error response"
            );
            return Err(RStructorError::ApiError(format!(
                "Grok API error: {}",
                error_text
            )));
        }

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

        let message = &completion.choices[0].message;
        trace!(finish_reason = %completion.choices[0].finish_reason, "Completion finish reason");

        if let Some(content) = &message.content {
            debug!(
                content_len = content.len(),
                "Successfully extracted content from response"
            );
            Ok(content.clone())
        } else {
            error!("No content in Grok API response");
            Err(RStructorError::ApiError(
                "No content in response".to_string(),
            ))
        }
    }
}

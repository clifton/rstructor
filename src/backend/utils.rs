use crate::error::{RStructorError, Result};
use reqwest::Response;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, error, info, trace, warn};

/// Extract JSON from markdown code blocks if present, otherwise return the content as-is.
///
/// This function handles cases where LLM providers wrap JSON responses in markdown code blocks
/// like ```json ... ``` or ``` ... ```.
pub fn extract_json_from_markdown(content: &str) -> String {
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

/// Convert a reqwest error to a RStructorError, handling timeout errors specially.
pub fn handle_http_error(e: reqwest::Error, provider_name: &str) -> RStructorError {
    error!(error = %e, "HTTP request to {} failed", provider_name);
    if e.is_timeout() {
        RStructorError::Timeout
    } else {
        RStructorError::HttpError(e)
    }
}

/// Check HTTP response status and extract error message if unsuccessful.
pub async fn check_response_status(response: Response, provider_name: &str) -> Result<Response> {
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await?;
        error!(
            status = %status,
            error = %error_text,
            "{} API returned error response", provider_name
        );
        return Err(RStructorError::ApiError(format!(
            "{} API error: {}",
            provider_name, error_text
        )));
    }
    Ok(response)
}

/// Helper function to execute generation with retry logic.
///
/// This function handles automatic retries for validation errors when retry configuration
/// is enabled. It will retry up to `max_retries` times, optionally including error feedback
/// in retry prompts.
pub async fn generate_with_retry<F, Fut, T>(
    mut generate_fn: F,
    prompt: &str,
    max_retries: Option<usize>,
    include_error_feedback: Option<bool>,
) -> Result<T>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let Some(max_retries) = max_retries.filter(|&n| n > 0) else {
        return generate_fn(prompt.to_string()).await;
    };

    let max_attempts = max_retries + 1; // +1 for initial attempt
    let include_error_feedback = include_error_feedback.unwrap_or(true);
    let base_prompt = prompt.to_string();
    let mut current_prompt = base_prompt.clone();
    let mut validation_errors: Option<String> = None;

    trace!(
        "Starting structured generation with retry: max_attempts={}, include_error_feedback={}",
        max_attempts, include_error_feedback
    );

    for attempt in 0..max_attempts {
        if attempt > 0 {
            current_prompt = base_prompt.clone();

            if include_error_feedback && let Some(error_msg) = validation_errors.as_ref() {
                debug!(
                    attempt,
                    error = error_msg,
                    "Retrying with validation error feedback"
                );

                current_prompt = format!(
                    "{}\n\nYour previous response contained validation errors. Please provide a complete, valid JSON response that includes ALL required fields and follows the schema exactly.\n\nError details:\n{}\n\nPlease fix the issues in your response. Make sure to:\n1. Include ALL required fields exactly as specified in the schema\n2. For enum fields, use EXACTLY one of the allowed values from the description\n3. CRITICAL: For arrays where items.type = 'object':\n   - You MUST provide an array of OBJECTS, not strings or primitive values\n   - Each object must be a complete JSON object with all its required fields\n   - Include multiple items (at least 2-3) in arrays of objects\n4. Verify all nested objects have their complete structure\n5. Follow ALL type specifications (string, number, boolean, array, object)",
                    base_prompt, error_msg
                );
            }
        }

        // Log attempt information
        info!(
            attempt = attempt + 1,
            total_attempts = max_attempts,
            "Generation attempt"
        );

        // Attempt to generate structured data
        match generate_fn(current_prompt.clone()).await {
            Ok(result) => {
                if attempt > 0 {
                    // If we succeeded after retries
                    info!(
                        attempts_used = attempt + 1,
                        "Successfully generated after {} retries", attempt
                    );
                } else {
                    debug!("Successfully generated on first attempt");
                }
                return Ok(result);
            }
            Err(err) => {
                // Only retry for validation errors
                if let RStructorError::ValidationError(msg) = &err {
                    if attempt < max_attempts - 1 {
                        warn!(
                            attempt = attempt + 1,
                            error = msg,
                            "Validation error in generation attempt"
                        );
                        // Store error for next attempt
                        validation_errors = Some(msg.clone());
                        // Wait briefly before retrying
                        sleep(Duration::from_millis(500)).await;
                        continue;
                    } else {
                        // Last attempt failed
                        error!(
                            attempts = max_attempts,
                            error = msg,
                            "Failed after maximum retry attempts with validation errors"
                        );
                    }
                } else {
                    // For non-validation errors
                    error!(
                        error = ?err,
                        "Non-validation error occurred during generation"
                    );
                }

                // For non-validation errors or last attempt, return the error
                return Err(err);
            }
        }
    }

    // This should never be reached due to the returns in the loop
    unreachable!()
}

/// Macro to generate standard builder methods for LLM clients.
///
/// This macro generates `model()`, `temperature()`, `max_tokens()`, and `timeout()` methods
/// that are identical across all LLM client implementations.
#[macro_export]
macro_rules! impl_client_builder_methods {
    (
        client_type: $client:ty,
        config_type: $config:ty,
        model_type: $model:ty,
        provider_name: $provider:expr
    ) => {
        impl $client {
            /// Set the model to use
            #[tracing::instrument(skip(self))]
            pub fn model(mut self, model: $model) -> Self {
                tracing::debug!(
                    previous_model = ?self.config.model,
                    new_model = ?model,
                    "Setting {} model", $provider
                );
                self.config.model = model;
                self
            }

            /// Set the temperature (0.0 to 1.0, lower = more deterministic)
            #[tracing::instrument(skip(self))]
            pub fn temperature(mut self, temp: f32) -> Self {
                tracing::debug!(
                    previous_temp = self.config.temperature,
                    new_temp = temp,
                    "Setting temperature"
                );
                self.config.temperature = temp;
                self
            }

            /// Set the maximum tokens to generate
            #[tracing::instrument(skip(self))]
            pub fn max_tokens(mut self, max: u32) -> Self {
                tracing::debug!(
                    previous_max = ?self.config.max_tokens,
                    new_max = max,
                    "Setting max_tokens"
                );
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
            #[tracing::instrument(skip(self))]
            pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
                tracing::debug!(
                    previous_timeout = ?self.config.timeout,
                    new_timeout = ?timeout,
                    "Setting timeout"
                );
                self.config.timeout = Some(timeout);

                // Rebuild reqwest client with timeout immediately
                self.client = reqwest::Client::builder()
                    .timeout(timeout)
                    .build()
                    .unwrap_or_else(|e| {
                        tracing::warn!(
                            error = %e,
                            "Failed to build reqwest client with timeout, using default"
                        );
                        reqwest::Client::new()
                    });

                self
            }

            /// Set the maximum number of retry attempts for validation errors.
            ///
            /// When `generate_struct` encounters a validation error, it will automatically
            /// retry up to this many times, including the validation error message in subsequent attempts.
            ///
            /// # Arguments
            ///
            /// * `max_retries` - Maximum number of retry attempts (0 = no retries, only single attempt)
            ///
            /// # Examples
            ///
            /// ```no_run
            /// # use rstructor::OpenAIClient;
            /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
            /// let client = OpenAIClient::new("api-key")?
            ///     .max_retries(3);  // Retry up to 3 times on validation errors
            /// # Ok(())
            /// # }
            /// ```
            #[tracing::instrument(skip(self))]
            pub fn max_retries(mut self, max_retries: usize) -> Self {
                tracing::debug!(
                    previous_max_retries = ?self.config.max_retries,
                    new_max_retries = max_retries,
                    "Setting max_retries"
                );
                self.config.max_retries = Some(max_retries);
                self
            }

            /// Set whether to include validation error feedback in retry prompts.
            ///
            /// When enabled (default: true), validation error messages are included in retry prompts
            /// to help the LLM understand what went wrong and fix issues. When disabled, retries
            /// happen without error feedback, relying only on the original prompt.
            ///
            /// # Arguments
            ///
            /// * `include_error_feedback` - Whether to include validation error messages in retry prompts
            ///
            /// # Examples
            ///
            /// ```no_run
            /// # use rstructor::OpenAIClient;
            /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
            /// let client = OpenAIClient::new("api-key")?
            ///     .max_retries(3)
            ///     .include_error_feedback(true);  // Include error messages in retries
            /// # Ok(())
            /// # }
            /// ```
            #[tracing::instrument(skip(self))]
            pub fn include_error_feedback(mut self, include_error_feedback: bool) -> Self {
                tracing::debug!(
                    previous_include_error_feedback = ?self.config.include_error_feedback,
                    new_include_error_feedback = include_error_feedback,
                    "Setting include_error_feedback"
                );
                self.config.include_error_feedback = Some(include_error_feedback);
                self
            }
        }
    };
}

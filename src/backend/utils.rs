use crate::backend::{ChatMessage, MaterializeInternalOutput, ValidationFailureContext};
use crate::error::{ApiErrorKind, RStructorError, Result};
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

/// Parse retry-after header value to Duration.
fn parse_retry_after(value: &str) -> Option<Duration> {
    // Try parsing as seconds (most common)
    if let Ok(secs) = value.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    // Could also parse HTTP-date format, but seconds is most common
    None
}

/// Classify an API error based on HTTP status code and response body.
fn classify_api_error(
    status: reqwest::StatusCode,
    error_text: &str,
    retry_after: Option<Duration>,
    model_hint: Option<&str>,
) -> ApiErrorKind {
    let code = status.as_u16();
    let error_lower = error_text.to_lowercase();

    match code {
        // Authentication errors
        401 => ApiErrorKind::AuthenticationFailed,

        // Permission errors
        403 => ApiErrorKind::PermissionDenied,

        // Not found - check if it's a model error
        404 => {
            // Check if the error message mentions "model"
            if error_lower.contains("model") {
                let model = model_hint
                    .map(|s| s.to_string())
                    .or_else(|| extract_model_from_error(&error_lower))
                    .unwrap_or_else(|| "unknown".to_string());
                ApiErrorKind::InvalidModel {
                    model,
                    suggestion: suggest_model(&error_lower),
                }
            } else {
                ApiErrorKind::Other {
                    code,
                    message: error_text.to_string(),
                }
            }
        }

        // Bad request
        400 => ApiErrorKind::BadRequest {
            details: truncate_message(error_text, 200),
        },

        // Payload too large
        413 => ApiErrorKind::RequestTooLarge,

        // Rate limited
        429 => ApiErrorKind::RateLimited { retry_after },

        // Server errors
        500 | 502 => ApiErrorKind::ServerError { code },

        // Service unavailable
        503 => ApiErrorKind::ServiceUnavailable,

        // Gateway/Cloudflare errors
        520..=524 => ApiErrorKind::GatewayError { code },

        // Other errors
        _ => ApiErrorKind::Other {
            code,
            message: truncate_message(error_text, 500),
        },
    }
}

/// Extract model name from error message if present.
fn extract_model_from_error(error_text: &str) -> Option<String> {
    // Look for quoted model names like 'gpt-4' or "gpt-4"
    for quote in ['\'', '"'] {
        if let Some(start) = error_text.find(quote) {
            let rest = &error_text[start + 1..];
            if let Some(end) = rest.find(quote) {
                let candidate = &rest[..end];
                // Model names typically have alphanumeric chars, dots, or dashes
                if candidate.len() > 2
                    && candidate
                        .chars()
                        .all(|c| c.is_alphanumeric() || c == '-' || c == '.' || c == '_')
                {
                    return Some(candidate.to_string());
                }
            }
        }
    }
    None
}

/// Suggest an alternative model based on error context.
fn suggest_model(error_text: &str) -> Option<String> {
    // Common model name patterns and their suggestions
    if error_text.contains("gpt") {
        Some("gpt-5.2".to_string())
    } else if error_text.contains("claude") || error_text.contains("sonnet") {
        Some("claude-sonnet-4-5-20250929".to_string())
    } else if error_text.contains("gemini") {
        Some("gemini-3-flash-preview".to_string())
    } else {
        None
    }
}

/// Truncate a message to a maximum length.
///
/// Uses `floor_char_boundary` to ensure we don't slice in the middle of a
/// multi-byte UTF-8 character, which would cause a panic.
fn truncate_message(msg: &str, max_len: usize) -> String {
    if msg.len() <= max_len {
        msg.to_string()
    } else {
        // Find a valid UTF-8 character boundary at or before max_len
        let boundary = msg.floor_char_boundary(max_len);
        format!("{}...", &msg[..boundary])
    }
}

/// Check HTTP response status and extract error message if unsuccessful.
///
/// This function classifies errors into actionable types (rate limit, auth failure, etc.)
/// and provides user-friendly error messages with suggested actions.
pub async fn check_response_status(response: Response, provider_name: &str) -> Result<Response> {
    if !response.status().is_success() {
        let status = response.status();

        // Extract retry-after header if present
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_retry_after);

        let error_text = response.text().await?;

        let kind = classify_api_error(status, &error_text, retry_after, None);

        error!(
            status = %status,
            error = %error_text,
            kind = %kind,
            "{} API returned error response", provider_name
        );

        return Err(RStructorError::api_error(provider_name, kind));
    }
    Ok(response)
}

/// Helper function to execute generation with retry logic using conversation history.
///
/// This function maintains a conversation history across retry attempts, which enables:
/// - **Prompt caching**: Providers like Anthropic and OpenAI can cache the prefix of the
///   conversation, reducing token costs and latency on retries.
/// - **Better error correction**: The model sees its previous (failed) response and the
///   specific error, making it more likely to produce a correct response.
///
/// # How it works
///
/// 1. On first attempt: Sends `[User(prompt)]`
/// 2. On validation failure: Appends `[Assistant(failed_response), User(error_feedback)]`
/// 3. On retry: Sends the full conversation history
///
/// This approach preserves the original prompt exactly, maximizing cache hit rates.
///
/// # Arguments
///
/// * `generate_fn` - Function that takes a conversation history and returns the result plus raw response
/// * `prompt` - The initial user prompt
/// * `max_retries` - Maximum number of retry attempts (None or 0 means no retries)
/// * `include_error_feedback` - Whether to include validation errors in retry prompts (default: true)
pub async fn generate_with_retry_with_history<F, Fut, T>(
    mut generate_fn: F,
    prompt: &str,
    max_retries: Option<usize>,
    include_error_feedback: Option<bool>,
) -> Result<MaterializeInternalOutput<T>>
where
    F: FnMut(Vec<ChatMessage>) -> Fut,
    Fut: std::future::Future<
            Output = std::result::Result<
                MaterializeInternalOutput<T>,
                (RStructorError, Option<ValidationFailureContext>),
            >,
        >,
{
    let Some(max_retries) = max_retries.filter(|&n| n > 0) else {
        // No retries configured - just run once with a single user message
        let messages = vec![ChatMessage::user(prompt)];
        return generate_fn(messages).await.map_err(|(err, _)| err);
    };

    let max_attempts = max_retries + 1; // +1 for initial attempt
    let include_error_feedback = include_error_feedback.unwrap_or(true);

    // Initialize conversation history with the original user prompt
    let mut messages = vec![ChatMessage::user(prompt)];

    trace!(
        "Starting structured generation with conversation history: max_attempts={}, include_error_feedback={}",
        max_attempts, include_error_feedback
    );

    for attempt in 0..max_attempts {
        // Log attempt information
        info!(
            attempt = attempt + 1,
            total_attempts = max_attempts,
            history_len = messages.len(),
            "Generation attempt with conversation history"
        );

        // Attempt to generate structured data
        match generate_fn(messages.clone()).await {
            Ok(result) => {
                if attempt > 0 {
                    info!(
                        attempts_used = attempt + 1,
                        "Successfully generated after {} retries (with conversation history)",
                        attempt
                    );
                } else {
                    debug!("Successfully generated on first attempt");
                }
                return Ok(result);
            }
            Err((err, validation_ctx)) => {
                let is_last_attempt = attempt >= max_attempts - 1;

                // Handle validation errors with conversation history
                if let RStructorError::ValidationError(ref msg) = err {
                    if !is_last_attempt {
                        warn!(
                            attempt = attempt + 1,
                            error = msg,
                            "Validation error in generation attempt"
                        );

                        // Build conversation history for retry
                        if include_error_feedback {
                            if let Some(ctx) = validation_ctx {
                                // Add the failed assistant response to history
                                messages.push(ChatMessage::assistant(&ctx.raw_response));

                                // Add user message with error feedback
                                let error_feedback = format!(
                                    "Your previous response contained validation errors. Please provide a complete, valid JSON response that includes ALL required fields and follows the schema exactly.\n\nError details:\n{}\n\nPlease fix the issues in your response. Make sure to:\n1. Include ALL required fields exactly as specified in the schema\n2. For enum fields, use EXACTLY one of the allowed values from the description\n3. CRITICAL: For arrays where items.type = 'object':\n   - You MUST provide an array of OBJECTS, not strings or primitive values\n   - Each object must be a complete JSON object with all its required fields\n   - Include multiple items (at least 2-3) in arrays of objects\n4. Verify all nested objects have their complete structure\n5. Follow ALL type specifications (string, number, boolean, array, object)",
                                    ctx.error_message
                                );
                                messages.push(ChatMessage::user(error_feedback));

                                debug!(
                                    history_len = messages.len(),
                                    "Updated conversation history for retry"
                                );
                            } else {
                                // Fallback: no raw response context available.
                                // We cannot add error feedback without the raw response because:
                                // 1. Adding only a user message would create consecutive user messages,
                                //    violating the alternating user/assistant pattern expected by LLM APIs
                                // 2. The error message references "your previous response" but we can't show it
                                // Instead, we retry with the original conversation (no history modification)
                                warn!(
                                    "Validation error occurred but no raw response context available. \
                                     Retrying without error feedback in conversation history."
                                );
                            }
                        }

                        // Wait briefly before retrying
                        sleep(Duration::from_millis(500)).await;
                        continue;
                    } else {
                        error!(
                            attempts = max_attempts,
                            error = msg,
                            "Failed after maximum retry attempts with validation errors"
                        );
                    }
                }
                // Handle retryable API errors (rate limits, transient failures)
                else if err.is_retryable() && !is_last_attempt {
                    let delay = err.retry_delay().unwrap_or(Duration::from_secs(1));
                    warn!(
                        attempt = attempt + 1,
                        error = ?err,
                        delay_ms = delay.as_millis(),
                        "Retryable API error, waiting before retry"
                    );
                    // For API errors, we don't modify the conversation history
                    // Just retry with the same messages
                    sleep(delay).await;
                    continue;
                }
                // Non-retryable errors or last attempt
                else if is_last_attempt {
                    error!(
                        attempts = max_attempts,
                        error = ?err,
                        "Failed after maximum retry attempts"
                    );
                } else {
                    error!(
                        error = ?err,
                        "Non-retryable error occurred during generation"
                    );
                }

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
            /// Set the model to use. Accepts either a Model enum variant or a string.
            ///
            /// When a string is provided, it will be converted to a Model enum. If the string
            /// matches a known model variant, that variant is used; otherwise, it becomes `Custom(name)`.
            /// This allows using any model name, including new models or local LLMs, without needing
            /// to update the enum.
            #[tracing::instrument(skip(self, model))]
            pub fn model<M: Into<$model>>(mut self, model: M) -> Self {
                let model = model.into();
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
            /// When `materialize` encounters a validation error, it will automatically
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_message_ascii_within_limit() {
        let msg = "Hello, world!";
        assert_eq!(truncate_message(msg, 20), "Hello, world!");
    }

    #[test]
    fn truncate_message_ascii_exact_limit() {
        let msg = "Hello";
        assert_eq!(truncate_message(msg, 5), "Hello");
    }

    #[test]
    fn truncate_message_ascii_exceeds_limit() {
        let msg = "Hello, world!";
        assert_eq!(truncate_message(msg, 5), "Hello...");
    }

    #[test]
    fn truncate_message_utf8_within_limit() {
        let msg = "你好世界"; // 12 bytes (3 bytes per character)
        assert_eq!(truncate_message(msg, 20), "你好世界");
    }

    #[test]
    fn truncate_message_utf8_boundary_safe() {
        // "你好世界" is 12 bytes total (3 bytes per character)
        // Truncating at 5 bytes would fall in the middle of the second character
        // floor_char_boundary(5) should return 3 (end of first character)
        let msg = "你好世界";
        let result = truncate_message(msg, 5);
        assert_eq!(result, "你...");
    }

    #[test]
    fn truncate_message_utf8_exact_boundary() {
        // Truncating at exactly 6 bytes should include first two characters
        let msg = "你好世界";
        let result = truncate_message(msg, 6);
        assert_eq!(result, "你好...");
    }

    #[test]
    fn truncate_message_emoji() {
        // Emojis are typically 4 bytes each
        let msg = "🎉🎊🎈";
        // Truncating at 5 bytes falls in the middle of second emoji
        // floor_char_boundary(5) should return 4 (end of first emoji)
        let result = truncate_message(msg, 5);
        assert_eq!(result, "🎉...");
    }

    #[test]
    fn truncate_message_mixed_utf8() {
        let msg = "Error: 无效的请求";
        // "Error: " is 7 bytes, then Chinese characters are 3 bytes each
        // Truncating at 10 bytes falls at the boundary after the first Chinese char
        // floor_char_boundary(10) should return 10 (end of first Chinese char after "Error: ")
        let result = truncate_message(msg, 10);
        assert_eq!(result, "Error: 无...");
    }

    #[test]
    fn truncate_message_empty_string() {
        let msg = "";
        assert_eq!(truncate_message(msg, 10), "");
    }

    #[test]
    fn truncate_message_zero_limit() {
        let msg = "Hello";
        // floor_char_boundary(0) returns 0, so we get just "..."
        assert_eq!(truncate_message(msg, 0), "...");
    }
}

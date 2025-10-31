use crate::error::{RStructorError, Result};
use reqwest::Response;
use tracing::error;

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
        }
    };
}

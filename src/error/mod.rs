use std::time::Duration;
use thiserror::Error;

/// Classification of API errors for better handling and retry logic.
///
/// This enum categorizes HTTP errors from LLM providers into actionable types,
/// making it easier to determine appropriate responses (retry, fix config, wait, etc.).
///
/// # Example
///
/// ```
/// use rstructor::{RStructorError, ApiErrorKind};
///
/// fn handle_api_error(err: &RStructorError) {
///     if let Some(kind) = err.api_error_kind() {
///         match kind {
///             ApiErrorKind::RateLimited { retry_after } => {
///                 println!("Rate limited! Wait {:?} and retry", retry_after);
///             }
///             ApiErrorKind::AuthenticationFailed => {
///                 println!("Check your API key");
///             }
///             ApiErrorKind::InvalidModel { model, suggestion } => {
///                 println!("Model '{}' not found", model);
///                 if let Some(s) = suggestion {
///                     println!("Try: {}", s);
///                 }
///             }
///             _ if err.is_retryable() => {
///                 println!("Transient error, will retry");
///             }
///             _ => {
///                 println!("Unrecoverable error");
///             }
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum ApiErrorKind {
    /// Rate limit exceeded (HTTP 429)
    ///
    /// The API is rate limiting requests. Wait for the specified duration before retrying.
    RateLimited {
        /// How long to wait before retrying (if provided by the API)
        retry_after: Option<Duration>,
    },

    /// Invalid or unknown model (HTTP 404 with model context)
    ///
    /// The requested model does not exist or is not accessible.
    InvalidModel {
        /// The model name that was not found
        model: String,
        /// A suggested alternative model, if available
        suggestion: Option<String>,
    },

    /// Service temporarily unavailable (HTTP 503)
    ///
    /// The API service is temporarily down. This is usually transient.
    ServiceUnavailable,

    /// Gateway/proxy error (HTTP 520-524, Cloudflare errors)
    ///
    /// An error occurred at the gateway level. Usually transient.
    GatewayError {
        /// The specific HTTP status code
        code: u16,
    },

    /// Authentication failed (HTTP 401)
    ///
    /// The API key is invalid, expired, or missing.
    AuthenticationFailed,

    /// Permission denied (HTTP 403)
    ///
    /// The API key doesn't have permission for this operation or model.
    PermissionDenied,

    /// Request too large (HTTP 413)
    ///
    /// The request payload (usually the prompt) is too large.
    RequestTooLarge,

    /// Invalid request (HTTP 400)
    ///
    /// The request was malformed or contained invalid parameters.
    BadRequest {
        /// Details about what was invalid
        details: String,
    },

    /// Server error (HTTP 500, 502)
    ///
    /// An internal server error occurred. May be transient.
    ServerError {
        /// The specific HTTP status code
        code: u16,
    },

    /// Generic/unclassified API error
    Other {
        /// The HTTP status code
        code: u16,
        /// The error message from the API
        message: String,
    },

    /// Unexpected response format from API
    ///
    /// The API returned a successful HTTP status but the response content
    /// was missing expected fields (e.g., empty choices array, no content).
    UnexpectedResponse {
        /// Description of what was expected vs received
        details: String,
    },
}

impl ApiErrorKind {
    /// Returns whether this error is potentially retryable.
    ///
    /// Retryable errors are transient issues that may succeed on a subsequent attempt.
    ///
    /// # Example
    ///
    /// ```
    /// use rstructor::ApiErrorKind;
    /// use std::time::Duration;
    ///
    /// let rate_limited = ApiErrorKind::RateLimited { retry_after: Some(Duration::from_secs(5)) };
    /// assert!(rate_limited.is_retryable());
    ///
    /// let auth_failed = ApiErrorKind::AuthenticationFailed;
    /// assert!(!auth_failed.is_retryable());
    /// ```
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ApiErrorKind::RateLimited { .. }
                | ApiErrorKind::ServiceUnavailable
                | ApiErrorKind::GatewayError { .. }
                | ApiErrorKind::ServerError { .. }
        )
    }

    /// Returns the suggested wait duration for retryable errors.
    ///
    /// For rate-limited errors, returns the `retry_after` duration if available.
    /// For other retryable errors, returns a sensible default.
    pub fn retry_delay(&self) -> Option<Duration> {
        match self {
            ApiErrorKind::RateLimited { retry_after } => {
                Some(retry_after.unwrap_or(Duration::from_secs(5)))
            }
            ApiErrorKind::ServiceUnavailable => Some(Duration::from_secs(2)),
            ApiErrorKind::GatewayError { .. } => Some(Duration::from_secs(1)),
            ApiErrorKind::ServerError { .. } => Some(Duration::from_secs(2)),
            _ => None,
        }
    }

    /// Returns a user-friendly message describing the error and suggested action.
    pub fn user_message(&self, provider_name: &str) -> String {
        match self {
            ApiErrorKind::RateLimited { retry_after } => {
                if let Some(duration) = retry_after {
                    format!(
                        "Rate limit exceeded. Please wait {} seconds and try again.",
                        duration.as_secs()
                    )
                } else {
                    "Rate limit exceeded. Please wait a moment and try again.".to_string()
                }
            }
            ApiErrorKind::InvalidModel { model, suggestion } => {
                let mut msg = format!("Model '{}' not found.", model);
                if let Some(s) = suggestion {
                    msg.push_str(&format!(" Try using '{}'.", s));
                }
                msg
            }
            ApiErrorKind::ServiceUnavailable => {
                format!(
                    "{} service is temporarily unavailable. Please try again.",
                    provider_name
                )
            }
            ApiErrorKind::GatewayError { code } => {
                format!(
                    "Gateway error ({}). This is usually transient - please retry.",
                    code
                )
            }
            ApiErrorKind::AuthenticationFailed => {
                format!(
                    "Authentication failed. Check your {}_API_KEY environment variable.",
                    provider_name.to_uppercase()
                )
            }
            ApiErrorKind::PermissionDenied => {
                "Permission denied. Your API key may not have access to this model or feature."
                    .to_string()
            }
            ApiErrorKind::RequestTooLarge => {
                "Request too large. Try reducing the prompt length or max_tokens.".to_string()
            }
            ApiErrorKind::BadRequest { details } => {
                format!("Invalid request: {}", details)
            }
            ApiErrorKind::ServerError { code } => {
                format!(
                    "{} server error ({}). This may be transient - please retry.",
                    provider_name, code
                )
            }
            ApiErrorKind::Other { code, message } => {
                format!("{} API error ({}): {}", provider_name, code, message)
            }
            ApiErrorKind::UnexpectedResponse { details } => {
                format!(
                    "{} returned an unexpected response: {}",
                    provider_name, details
                )
            }
        }
    }
}

impl std::fmt::Display for ApiErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiErrorKind::RateLimited { retry_after } => {
                write!(f, "Rate limited")?;
                if let Some(d) = retry_after {
                    write!(f, " (retry after {}s)", d.as_secs())?;
                }
                Ok(())
            }
            ApiErrorKind::InvalidModel { model, .. } => write!(f, "Invalid model: {}", model),
            ApiErrorKind::ServiceUnavailable => write!(f, "Service unavailable"),
            ApiErrorKind::GatewayError { code } => write!(f, "Gateway error ({})", code),
            ApiErrorKind::AuthenticationFailed => write!(f, "Authentication failed"),
            ApiErrorKind::PermissionDenied => write!(f, "Permission denied"),
            ApiErrorKind::RequestTooLarge => write!(f, "Request too large"),
            ApiErrorKind::BadRequest { details } => write!(f, "Bad request: {}", details),
            ApiErrorKind::ServerError { code } => write!(f, "Server error ({})", code),
            ApiErrorKind::Other { code, message } => write!(f, "API error ({}): {}", code, message),
            ApiErrorKind::UnexpectedResponse { details } => {
                write!(f, "Unexpected response: {}", details)
            }
        }
    }
}

/// Error types for the rstructor library.
///
/// This enum defines the various error types that can occur within the rstructor library.
/// Each variant represents a different category of error and includes context about what went wrong.
///
/// # Examples
///
/// Creating and handling errors:
///
/// ```
/// use rstructor::{RStructorError, Result};
///
/// // Function that might return an error
/// fn validate_age(age: i32) -> Result<()> {
///     if age < 0 {
///         return Err(RStructorError::ValidationError("Age cannot be negative".into()));
///     }
///     if age > 150 {
///         return Err(RStructorError::ValidationError("Age is unrealistically high".into()));
///     }
///     Ok(())
/// }
///
/// // Using the function and handling errors
/// let result = validate_age(200);
/// match result {
///     Ok(()) => println!("Age is valid"),
///     Err(RStructorError::ValidationError(msg)) => println!("Invalid age: {}", msg),
///     Err(e) => println!("Unexpected error: {}", e),
/// }
/// ```
#[derive(Error, Debug)]
pub enum RStructorError {
    /// Error interacting with the LLM API (with rich error classification)
    #[error("{}", .kind.user_message(.provider))]
    ApiError {
        /// The provider that returned the error (e.g., "OpenAI", "Anthropic")
        provider: String,
        /// The classified error kind
        kind: ApiErrorKind,
    },

    /// Error validating data against schema or business rules
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Error related to JSON Schema generation or processing
    #[error("Schema error: {0}")]
    SchemaError(String),

    /// Error serializing or deserializing data
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Operation timed out
    #[error("Timeout error")]
    Timeout,

    /// HTTP client error (from reqwest)
    #[error("HTTP client error: {0}")]
    HttpError(#[from] reqwest::Error),

    /// JSON parsing error (from serde_json)
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

impl RStructorError {
    /// Create a new API error with rich classification.
    ///
    /// # Arguments
    ///
    /// * `provider` - The LLM provider name (e.g., "OpenAI", "Anthropic")
    /// * `kind` - The classified error kind
    pub fn api_error(provider: impl Into<String>, kind: ApiErrorKind) -> Self {
        RStructorError::ApiError {
            provider: provider.into(),
            kind,
        }
    }

    /// Returns the API error kind if this is an API error.
    ///
    /// # Example
    ///
    /// ```
    /// use rstructor::{RStructorError, ApiErrorKind};
    ///
    /// let err = RStructorError::api_error("OpenAI", ApiErrorKind::AuthenticationFailed);
    /// assert!(matches!(err.api_error_kind(), Some(ApiErrorKind::AuthenticationFailed)));
    /// ```
    pub fn api_error_kind(&self) -> Option<&ApiErrorKind> {
        match self {
            RStructorError::ApiError { kind, .. } => Some(kind),
            _ => None,
        }
    }

    /// Returns whether this error is potentially retryable.
    ///
    /// Retryable errors include:
    /// - Rate limiting (429)
    /// - Service unavailable (503)
    /// - Gateway errors (520-524)
    /// - Server errors (500, 502)
    /// - Timeout errors
    ///
    /// # Example
    ///
    /// ```
    /// use rstructor::{RStructorError, ApiErrorKind};
    /// use std::time::Duration;
    ///
    /// let rate_limited = RStructorError::api_error(
    ///     "OpenAI",
    ///     ApiErrorKind::RateLimited { retry_after: Some(Duration::from_secs(5)) }
    /// );
    /// assert!(rate_limited.is_retryable());
    ///
    /// let auth_error = RStructorError::api_error("OpenAI", ApiErrorKind::AuthenticationFailed);
    /// assert!(!auth_error.is_retryable());
    /// ```
    pub fn is_retryable(&self) -> bool {
        match self {
            RStructorError::ApiError { kind, .. } => kind.is_retryable(),
            RStructorError::Timeout => true,
            _ => false,
        }
    }

    /// Returns the suggested retry delay for retryable errors.
    ///
    /// Returns `None` for non-retryable errors.
    pub fn retry_delay(&self) -> Option<Duration> {
        match self {
            RStructorError::ApiError { kind, .. } => kind.retry_delay(),
            RStructorError::Timeout => Some(Duration::from_secs(1)),
            _ => None,
        }
    }
}

// Manual implementation of PartialEq for RStructorError
// Note: HttpError and JsonError variants are considered unequal
// because reqwest::Error and serde_json::Error don't implement PartialEq
impl PartialEq for RStructorError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::ApiError {
                    provider: p1,
                    kind: k1,
                },
                Self::ApiError {
                    provider: p2,
                    kind: k2,
                },
            ) => p1 == p2 && k1 == k2,
            (Self::ValidationError(a), Self::ValidationError(b)) => a == b,
            (Self::SchemaError(a), Self::SchemaError(b)) => a == b,
            (Self::SerializationError(a), Self::SerializationError(b)) => a == b,
            (Self::Timeout, Self::Timeout) => true,
            // HttpError and JsonError don't implement PartialEq, so we always return false
            (Self::HttpError(_), Self::HttpError(_)) => false,
            (Self::JsonError(_), Self::JsonError(_)) => false,
            _ => false,
        }
    }
}

/// A specialized Result type for rstructor operations.
///
/// This type is used throughout the rstructor library to return either
/// a success value of type T or an RStructorError.
///
/// # Examples
///
/// Using Result type in functions:
///
/// ```
/// use rstructor::{RStructorError, Result};
///
/// fn parse_json_data(data: &str) -> Result<serde_json::Value> {
///     match serde_json::from_str(data) {
///         Ok(value) => Ok(value),
///         Err(e) => Err(RStructorError::JsonError(e)),
///     }
/// }
///
/// // Using the ? operator with Result
/// fn process_data(input: &str) -> Result<String> {
///     let json = parse_json_data(input)?;
///     // Process the JSON...
///     Ok("Processed successfully".to_string())
/// }
/// ```
pub type Result<T> = std::result::Result<T, RStructorError>;

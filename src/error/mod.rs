use thiserror::Error;

/// Error types for the RStructor library.
///
/// This enum defines the various error types that can occur within the RStructor library.
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
    /// Error interacting with the LLM API
    #[error("API error: {0}")]
    ApiError(String),

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

// Manual implementation of PartialEq for RStructorError
// Note: HttpError and JsonError variants are considered unequal
// because reqwest::Error and serde_json::Error don't implement PartialEq
impl PartialEq for RStructorError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ApiError(a), Self::ApiError(b)) => a == b,
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

/// A specialized Result type for RStructor operations.
///
/// This type is used throughout the RStructor library to return either
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

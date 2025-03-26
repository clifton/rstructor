use thiserror::Error;

#[derive(Error, Debug)]
pub enum RStructorError {
    #[error("API error: {0}")]
    ApiError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Schema error: {0}")]
    SchemaError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Timeout error")]
    Timeout,

    #[error("HTTP client error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, RStructorError>;

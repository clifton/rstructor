pub mod backend;
pub mod error;
pub mod model;
pub mod schema;

// Re-export primary types that users will need
pub use backend::LLMClient;
pub use error::{RStructorError, Result};
pub use model::{LLMModel, Validatable};
pub use schema::{Schema, SchemaType};

// Re-export backend implementations when features are enabled
#[cfg(feature = "openai")]
pub use backend::openai::{Model as OpenAIModel, OpenAIClient};

#[cfg(feature = "anthropic")]
pub use backend::anthropic::{AnthropicClient, AnthropicModel};

// Re-export derive macro when the "derive" feature is enabled
#[cfg(feature = "derive")]
pub use rstructor_derive::LLMModel;

// Version 0.1.0: This library provides structured outputs from LLMs
// with automatic JSON Schema generation, validation, and pluggable backends.

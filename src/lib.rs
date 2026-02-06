/// rstructor: A Rust library for structured outputs from LLMs
///
/// # Overview
///
/// rstructor simplifies getting validated, strongly-typed outputs from Large Language Models
/// (LLMs) like GPT-4 and Claude. It automatically generates JSON Schema from your Rust types,
/// sends the schema to LLMs, parses responses, and validates against the schema.
///
/// Key features:
/// - Derive macro for automatic JSON Schema generation
/// - Built-in OpenAI and Anthropic API clients
/// - Validation of responses against schemas
/// - Type-safe conversion from LLM outputs to Rust structs and enums
/// - Customizable client configurations
///
/// # Quick Start
///
/// ```no_run
/// use rstructor::{LLMClient, OpenAIClient, Instructor};
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Instructor, Serialize, Deserialize, Debug)]
/// struct Person {
///     name: String,
///     age: u8,
///     bio: String,
/// }
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create a client
///     let client = OpenAIClient::new("your-openai-api-key")?;
///
///     // Generate a structured response
///     let person: Person = client.materialize("Describe a fictional person").await?;
///
///     println!("Name: {}", person.name);
///     println!("Age: {}", person.age);
///     println!("Bio: {}", person.bio);
///
///     Ok(())
/// }
/// ```
mod backend;
pub mod error;
#[cfg(feature = "logging")]
pub mod logging;
pub mod model;
pub mod schema;

// Re-exports for convenience
pub use error::{ApiErrorKind, RStructorError, Result};
pub use model::Instructor;
pub use schema::{CustomTypeSchema, Schema, SchemaBuilder, SchemaType};

#[cfg(feature = "openai")]
pub use backend::openai::{Model as OpenAIModel, OpenAIClient};

#[cfg(feature = "anthropic")]
pub use backend::anthropic::{AnthropicClient, AnthropicModel};

#[cfg(feature = "gemini")]
pub use backend::gemini::{GeminiClient, Model as GeminiModel};

#[cfg(feature = "grok")]
pub use backend::grok::{GrokClient, Model as GrokModel};

#[cfg(feature = "derive")]
pub use rstructor_derive::Instructor;

pub use backend::LLMClient;
pub use backend::ModelInfo;
pub use backend::ThinkingLevel;
pub use backend::{
    ChatMessage, ChatRole, GenerateResult, MaterializeResult, MediaFile, TokenUsage,
};

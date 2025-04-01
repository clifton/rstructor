/// RStructor: A Rust library for structured outputs from LLMs
///
/// # Overview
///
/// RStructor simplifies getting validated, strongly-typed outputs from Large Language Models
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
///     let client = OpenAIClient::new("your-openai-api-key")?
///         .build();
///
///     // Generate a structured response
///     let person: Person = client.generate_struct("Describe a fictional person").await?;
///     
///     println!("Name: {}", person.name);
///     println!("Age: {}", person.age);
///     println!("Bio: {}", person.bio);
///     
///     Ok(())
/// }
/// ```
mod backend;
mod error;
pub mod schema;
pub mod model;
#[cfg(feature = "logging")]
pub mod logging;

// Re-exports for convenience
pub use error::{RStructorError, Result};
pub use schema::{SchemaBuilder, CustomTypeSchema, SchemaType, Schema};
pub use model::Instructor;

#[cfg(feature = "openai")]
pub use backend::openai::{OpenAIClient, Model as OpenAIModel};

#[cfg(feature = "anthropic")]
pub use backend::anthropic::{AnthropicClient, AnthropicModel};

#[cfg(feature = "derive")]
pub use rstructor_derive::Instructor;

pub use backend::LLMClient;
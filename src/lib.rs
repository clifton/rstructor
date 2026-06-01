//! rstructor: get structured, validated data out of LLMs as native Rust structs and enums
//!
//! # Overview
//!
//! rstructor gets structured, validated data out of large language models (LLMs) as native
//! Rust structs and enums. It generates JSON Schema from your types, prompts the model, parses
//! the response, and validates the result — retrying automatically when validation fails.
//!
//! Key features:
//! - Derive macro for automatic JSON Schema generation
//! - Built-in clients for OpenAI, Anthropic, Google Gemini, and xAI Grok
//! - Validation of responses against schemas
//! - Type-safe conversion from LLM outputs to Rust structs and enums
//! - Customizable client configurations
//!
//! # Quick Start
//!
//! ```no_run
//! use rstructor::{LLMClient, OpenAIClient, Instructor};
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Instructor, Serialize, Deserialize, Debug)]
//! struct Person {
//!     name: String,
//!     age: u8,
//!     bio: String,
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a client
//!     let client = OpenAIClient::new("your-openai-api-key")?;
//!
//!     // Generate a structured response
//!     let person: Person = client.materialize("Describe a fictional person").await?;
//!
//!     println!("Name: {}", person.name);
//!     println!("Age: {}", person.age);
//!     println!("Bio: {}", person.bio);
//!
//!     Ok(())
//! }
//! ```
// Let the crate refer to itself as `rstructor` so `#[derive(Instructor)]` — which
// emits absolute `::rstructor::…` paths — works in the crate's own unit tests.
extern crate self as rstructor;

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
#[cfg(feature = "_client")]
pub use backend::{AnyClient, Provider, Request, RequestExt};
pub use backend::{
    ChatMessage, ChatRole, GenerateResult, MaterializeResult, MediaFile, TokenUsage,
};
#[cfg(feature = "tools")]
pub use backend::{DynTool, FnTool, Tool, ToolRunner, Toolbox};
#[cfg(feature = "streaming")]
pub use backend::{ItemStream, ObjectStream, StreamedObject, TextStream};
#[cfg(feature = "mock")]
pub use backend::{MockClient, MockRequestView, MockResponse, RecordedRequest, RequestKind};

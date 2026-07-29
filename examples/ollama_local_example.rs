//! Materialize a typed value with a local Ollama model.
//!
//! Pull a model, start Ollama, and opt in by setting `OLLAMA_MODEL`:
//!
//! ```text
//! ollama pull llama3.3
//! OLLAMA_MODEL=llama3.3 cargo run --example ollama_local_example --features openai
//! ```
//!
//! The example exits successfully without making a request when `OLLAMA_MODEL`
//! is unset, which keeps example validation safe on machines without Ollama.

use rstructor::{Instructor, LLMClient, OpenAIClient};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Instructor)]
struct LocalSummary {
    #[llm(description = "A concise summary of the input")]
    summary: String,
    #[llm(description = "The two most important topics in the input")]
    topics: Vec<String>,
}

#[tokio::main]
async fn main() -> rstructor::Result<()> {
    let model = match std::env::var("OLLAMA_MODEL") {
        Ok(model) if !model.is_empty() => model,
        _ => {
            eprintln!(
                "Skipping Ollama request. Set OLLAMA_MODEL to a pulled model, \
                 for example: OLLAMA_MODEL=llama3.3"
            );
            return Ok(());
        }
    };

    let client = OpenAIClient::ollama()?.model(model);
    let result: LocalSummary = client
        .materialize(
            "Rust combines memory safety without garbage collection, zero-cost \
             abstractions, and fearless concurrency.",
        )
        .await?;

    println!("{result:#?}");
    Ok(())
}

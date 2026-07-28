//! Streaming a list of structured objects.
//!
//! Run with:
//!   cargo run --example streaming_example --features streaming
//!
//! `materialize_iter` streams a list, yielding each item as soon as it is fully
//! generated and validated — ideal for long lists where you want to start
//! processing (or rendering) items without waiting for the whole response.

use futures_util::StreamExt;
use rstructor::{Instructor, LLMClient, OpenAIClient};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "A notable invention")]
struct Invention {
    #[llm(description = "Name of the invention")]
    name: String,
    #[llm(description = "Year it was invented")]
    year: i32,
    #[llm(description = "Who invented it")]
    inventor: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAIClient::from_env()?;

    println!("--- Streaming a list of inventions (one at a time) ---\n");
    let mut stream =
        client.materialize_iter::<Invention>("List 8 important inventions of the 20th century.");

    let mut inventions = Vec::new();
    while let Some(item) = stream.next().await {
        // An integrity error can follow already-validated items if the provider
        // truncates the response. Only commit irreversible side effects after
        // this loop drains cleanly.
        let invention = item?; // each item is fully parsed and validated
        println!(
            "  tentative {}. {} ({}) — {}",
            inventions.len() + 1,
            invention.name,
            invention.year,
            invention.inventor
        );
        inventions.push(invention);
    }

    println!(
        "\nStream completed cleanly; {} items are authoritative.",
        inventions.len()
    );
    Ok(())
}

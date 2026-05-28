//! Streaming structured output.
//!
//! Run with:
//!   cargo run --example streaming_example --features streaming
//!
//! `materialize_stream` yields progressively-completed JSON snapshots of the object
//! as the model generates it, then a final, validated, typed value.

use futures_util::StreamExt;
use rstructor::{Instructor, LLMClient, OpenAIClient, StreamedObject};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "A simple cooking recipe")]
struct Recipe {
    #[llm(description = "Name of the dish")]
    name: String,
    #[llm(description = "Ingredients with quantities")]
    ingredients: Vec<String>,
    #[llm(description = "Ordered preparation steps")]
    steps: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAIClient::from_env()?;

    println!("--- Streaming a structured Recipe ---\n");
    let mut stream = client.materialize_stream::<Recipe>("Give me a simple recipe for pancakes.");

    while let Some(item) = stream.next().await {
        match item? {
            // Progressive snapshots: the object filling in, field by field.
            StreamedObject::Partial(json) => {
                let name = json.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let ingredients = json
                    .get("ingredients")
                    .and_then(|v| v.as_array())
                    .map_or(0, |a| a.len());
                println!("  partial: name={name:?}, ingredients so far={ingredients}");
            }
            // The final, fully parsed and validated value.
            StreamedObject::Complete(recipe) => {
                println!("\nDone! {recipe:#?}");
            }
        }
    }

    Ok(())
}

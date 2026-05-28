//! Tool (function) calling.
//!
//! Run with:
//!   cargo run --example tool_calling_example --features tools
//!
//! The model is given two tools and decides which to call (with typed,
//! schema-validated arguments); rstructor runs them and feeds the results back
//! until the model produces a final answer.

use rstructor::{FnTool, Instructor, OpenAIClient, Toolbox};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Instructor, Serialize, Deserialize)]
struct WeatherArgs {
    #[llm(description = "City name, e.g. 'Paris'")]
    city: String,
}

#[derive(Instructor, Serialize, Deserialize)]
struct AddArgs {
    #[llm(description = "First addend")]
    a: f64,
    #[llm(description = "Second addend")]
    b: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let toolbox = Toolbox::new()
        .with(FnTool::new(
            "get_weather",
            "Get the current weather (Fahrenheit) for a city",
            |args: WeatherArgs| async move {
                // A real tool would call a weather API here.
                Ok(json!({ "city": args.city, "temp_f": 68, "conditions": "sunny" }))
            },
        ))
        .with(FnTool::new(
            "add",
            "Add two numbers",
            |args: AddArgs| async move { Ok(json!({ "sum": args.a + args.b })) },
        ));

    let client = OpenAIClient::from_env()?;

    let answer = client
        .with_tools(&toolbox)
        .system("You are a helpful assistant. Use the provided tools when relevant.")
        .run("What's the weather in Paris, and what is 21 + 21?")
        .await?;

    println!("{answer}");
    Ok(())
}

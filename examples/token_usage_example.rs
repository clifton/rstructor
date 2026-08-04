//! Example demonstrating token usage tracking with a uniform extraction report.
//!
//! This is useful for monitoring API costs and understanding token consumption.

use rstructor::{Instructor, LLMClient, OpenAIClient};
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "A brief book summary")]
struct BookSummary {
    #[llm(description = "Title of the book")]
    title: String,

    #[llm(description = "Author of the book")]
    author: String,

    #[llm(description = "One-sentence summary of the book")]
    summary: String,

    #[llm(description = "Main themes of the book")]
    themes: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key =
        env::var("OPENAI_API_KEY").expect("Please set OPENAI_API_KEY environment variable");

    let client = OpenAIClient::new(api_key)?;

    let prompt = "Summarize the book '1984' by George Orwell";

    let extraction = client.extract_with_report::<BookSummary>(prompt).await?;

    // Access the extracted data
    println!(
        "Book: {} by {}",
        extraction.data.title, extraction.data.author
    );
    println!("Summary: {}", extraction.data.summary);
    println!("Themes: {:?}", extraction.data.themes);

    // Access cumulative usage across retries for cost tracking.
    if let Some(usage) = extraction.report.cumulative_usage {
        println!("\n--- Token Usage ---");
        println!("Input tokens: {}", usage.input_tokens);
        println!("Output tokens: {}", usage.output_tokens);
        println!("Total tokens: {}", usage.total_tokens());
    }

    Ok(())
}

//! Anthropic Multimodal Structured Extraction Example
//!
//! Run with:
//! ```bash
//! export ANTHROPIC_API_KEY=your_key_here
//! cargo run --example anthropic_multimodal_example --features anthropic
//! ```

use rstructor::{AnthropicClient, AnthropicModel, Instructor, LLMClient, MediaFile};
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct ImageAnalysis {
    subject: String,
    summary: String,
    colors: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env::var("ANTHROPIC_API_KEY").expect("Please set ANTHROPIC_API_KEY environment variable");

    let image_url = "https://www.rust-lang.org/logos/rust-logo-512x512.png";
    let image_bytes = reqwest::get(image_url).await?.bytes().await?;
    let media = MediaFile::from_bytes(&image_bytes, "image/png");

    let client = AnthropicClient::from_env()?
        .model(AnthropicModel::ClaudeOpus46)
        .temperature(0.0);

    let analysis: ImageAnalysis = client
        .materialize_with_media("Describe this image and list dominant colors.", &[media])
        .await?;

    println!("{:#?}", analysis);
    Ok(())
}

//! OpenAI Multimodal Structured Extraction Example
//!
//! Run with:
//! ```bash
//! export OPENAI_API_KEY=your_key_here
//! cargo run --example openai_multimodal_example --features openai
//! ```

use rstructor::{Instructor, LLMClient, MediaFile, OpenAIClient, OpenAIModel};
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
    env::var("OPENAI_API_KEY").expect("Please set OPENAI_API_KEY environment variable");

    let image_url = "https://www.rust-lang.org/logos/rust-logo-512x512.png";
    let image_bytes = reqwest::get(image_url).await?.bytes().await?;
    let media = MediaFile::from_bytes(&image_bytes, "image/png");

    let client = OpenAIClient::from_env()?
        .model(OpenAIModel::Gpt52)
        .temperature(0.0);

    let analysis: ImageAnalysis = client
        .materialize_with_media("Describe this image and list dominant colors.", &[media])
        .await?;

    println!("{:#?}", analysis);
    Ok(())
}

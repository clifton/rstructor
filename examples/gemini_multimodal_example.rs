//! Gemini Multimodal Structured Extraction Example
//!
//! Demonstrates how to use `MediaFile::from_bytes` with Gemini to analyze
//! an image and extract structured data. The example downloads a public
//! image (the Rust programming language logo), sends it to Gemini along
//! with a text prompt, and receives a structured `ImageAnalysis` response.
//!
//! # Usage
//!
//! ```bash
//! export GEMINI_API_KEY=your_key_here
//! cargo run --example gemini_multimodal_example --features gemini
//! ```

use rstructor::{GeminiClient, Instructor, LLMClient, MediaFile};
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "Structured analysis of an image")]
struct ImageAnalysis {
    #[llm(
        description = "The main subject depicted in the image",
        example = "Rust logo"
    )]
    subject: String,

    #[llm(description = "The dominant color in the image", example = "orange")]
    dominant_color: String,

    #[llm(description = "Colors visible in the image", example = ["orange", "black", "white"])]
    colors: Vec<String>,

    #[llm(description = "Whether the image is a logo or icon")]
    is_logo: bool,

    #[llm(description = "A detailed description of what the image depicts")]
    description: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key =
        env::var("GEMINI_API_KEY").expect("Please set GEMINI_API_KEY environment variable");

    // Download a public image (the Rust logo)
    println!("Downloading Rust logo...");
    let image_url = "https://www.rust-lang.org/logos/rust-logo-512x512.png";
    let image_bytes = reqwest::get(image_url).await?.bytes().await?;
    println!("Downloaded {} bytes", image_bytes.len());

    // Create a MediaFile with inline base64 data
    let media = MediaFile::from_bytes(&image_bytes, "image/png");

    // Create the Gemini client and analyze the image
    println!("Analyzing image with Gemini...\n");
    let client = GeminiClient::new(api_key)?.temperature(0.0);

    let analysis: ImageAnalysis = client
        .materialize_with_media(
            "Analyze this image and describe what you see in detail.",
            &[media],
        )
        .await?;

    // Display the results
    println!("===== Image Analysis =====");
    println!("Subject: {}", analysis.subject);
    println!("Dominant color: {}", analysis.dominant_color);
    println!("Colors: {}", analysis.colors.join(", "));
    println!("Is logo: {}", analysis.is_logo);
    println!("\nDescription:");
    println!("{}", analysis.description);

    Ok(())
}

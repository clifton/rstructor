//! Integration tests for Gemini multimodal (inline image) functionality.
//!
//! These tests require a valid Gemini API key:
//!
//! ```bash
//! export GEMINI_API_KEY=your_key_here
//! cargo test --test gemini_multimodal_tests --features gemini
//! ```

#[cfg(test)]
mod gemini_multimodal_tests {
    #[cfg(feature = "gemini")]
    use rstructor::{GeminiClient, GeminiModel};
    use rstructor::{Instructor, LLMClient, MediaFile};
    use serde::{Deserialize, Serialize};

    #[derive(Instructor, Serialize, Deserialize, Debug)]
    #[llm(description = "Description of an image")]
    struct ImageDescription {
        #[llm(description = "The main subject of the image")]
        subject: String,

        #[llm(description = "Colors visible in the image", example = ["red", "blue"])]
        colors: Vec<String>,

        #[llm(description = "A brief description of what the image depicts")]
        description: String,
    }

    #[cfg(feature = "gemini")]
    #[tokio::test]
    async fn test_gemini_multimodal_image_analysis() {
        // Download a small, stable public image (Rust logo)
        let image_url = "https://www.rust-lang.org/logos/rust-logo-512x512.png";
        let image_bytes = reqwest::get(image_url)
            .await
            .expect("Failed to download test image")
            .bytes()
            .await
            .expect("Failed to read image bytes");

        let media = MediaFile::from_bytes(&image_bytes, "image/png");

        let client = GeminiClient::from_env()
            .expect("GEMINI_API_KEY must be set for this test")
            .model(GeminiModel::Gemini3FlashPreview)
            .temperature(0.0);

        let result: ImageDescription = client
            .materialize_with_media("Describe this image in detail", &[media])
            .await
            .expect("Failed to materialize image description");

        // Verify we got meaningful results
        assert!(!result.subject.is_empty(), "Subject should not be empty");
        assert!(!result.colors.is_empty(), "Colors should not be empty");
        assert!(
            !result.description.is_empty(),
            "Description should not be empty"
        );

        // The Rust logo should be recognized as related to Rust or as a logo/gear
        let subject_lower = result.subject.to_lowercase();
        let desc_lower = result.description.to_lowercase();
        let mentions_rust_or_logo = subject_lower.contains("rust")
            || subject_lower.contains("logo")
            || subject_lower.contains("gear")
            || subject_lower.contains("cog")
            || desc_lower.contains("rust")
            || desc_lower.contains("logo")
            || desc_lower.contains("gear")
            || desc_lower.contains("cog");
        assert!(
            mentions_rust_or_logo,
            "Expected the image to be recognized as the Rust logo/gear, got subject='{}', description='{}'",
            result.subject, result.description
        );
    }
}

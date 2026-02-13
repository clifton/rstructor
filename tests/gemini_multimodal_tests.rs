//! Integration tests for Gemini multimodal (inline image) functionality.
//!
//! These tests require a valid Gemini API key:
//!
//! ```bash
//! export GEMINI_API_KEY=your_key_here
//! cargo test --test gemini_multimodal_tests --features gemini
//! ```

#[path = "common/mod.rs"]
mod common;

#[cfg(test)]
mod gemini_multimodal_tests {
    #[cfg(feature = "gemini")]
    use rstructor::GeminiClient;
    use rstructor::{GeminiModel, Instructor, LLMClient};
    use serde::{Deserialize, Serialize};

    use crate::common::{
        RUST_LOGO_MIME, RUST_LOGO_URL, RUST_SOCIAL_MIME, RUST_SOCIAL_URL, download_media,
    };

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

    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct MultiImageSummary {
        image_count: u8,
        summary: String,
    }

    #[cfg(feature = "gemini")]
    #[tokio::test]
    async fn test_gemini_multimodal_image_analysis() {
        let media = download_media(RUST_LOGO_URL, RUST_LOGO_MIME).await;

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
    }

    #[cfg(feature = "gemini")]
    #[tokio::test]
    async fn test_gemini_multimodal_second_real_world_image() {
        let media = download_media(RUST_SOCIAL_URL, RUST_SOCIAL_MIME).await;

        let client = GeminiClient::from_env()
            .expect("GEMINI_API_KEY must be set for this test")
            .model(GeminiModel::Gemini3FlashPreview)
            .temperature(0.0);

        let result: ImageDescription = client
            .materialize_with_media(
                "Describe the environment in this image and list the dominant colors.",
                &[media],
            )
            .await
            .expect("Failed to materialize secondary real-world image description");

        assert!(!result.subject.is_empty(), "Subject should not be empty");
        assert!(!result.colors.is_empty(), "Colors should not be empty");
        assert!(
            !result.description.is_empty(),
            "Description should not be empty"
        );
    }

    #[cfg(feature = "gemini")]
    #[tokio::test]
    async fn test_gemini_multimodal_multiple_images() {
        let rust_media = download_media(RUST_LOGO_URL, RUST_LOGO_MIME).await;
        let social_media = download_media(RUST_SOCIAL_URL, RUST_SOCIAL_MIME).await;

        let client = GeminiClient::from_env()
            .expect("GEMINI_API_KEY must be set for this test")
            .model(GeminiModel::Gemini3FlashPreview)
            .temperature(0.0);

        let result: MultiImageSummary = client
            .materialize_with_media(
                "You are given two images. Return the exact count in image_count and summarize both images.",
                &[rust_media, social_media],
            )
            .await
            .expect("Failed to materialize multi-image summary");

        assert!(
            result.image_count >= 2,
            "expected at least 2 images, got {}",
            result.image_count
        );
        assert!(!result.summary.is_empty(), "summary should not be empty");
    }
}

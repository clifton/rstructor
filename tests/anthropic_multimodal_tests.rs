//! Integration tests for Anthropic multimodal structured extraction.
//!
//! Requires:
//! - `ANTHROPIC_API_KEY`
//! - `--features anthropic`

#[path = "common/mod.rs"]
mod common;

#[cfg(test)]
mod anthropic_multimodal_tests {
    #[cfg(feature = "anthropic")]
    use rstructor::AnthropicClient;
    use rstructor::{AnthropicModel, Instructor, LLMClient};
    use serde::{Deserialize, Serialize};

    use crate::common::{RUST_LOGO_MIME, RUST_LOGO_URL, download_media, media_url};

    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct ImageSummary {
        subject: String,
        summary: String,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct MultiImageSummary {
        image_count: u8,
        summary: String,
    }

    #[cfg(feature = "anthropic")]
    #[tokio::test]
    async fn test_anthropic_multimodal_inline_image() {
        let media = download_media(RUST_LOGO_URL, RUST_LOGO_MIME).await;
        let client = AnthropicClient::from_env()
            .expect("ANTHROPIC_API_KEY must be set for this test")
            .model(AnthropicModel::ClaudeOpus46)
            .temperature(0.0);

        let result: ImageSummary = client
            .materialize_with_media(
                "Identify the main subject in this image and summarize it briefly.",
                &[media],
            )
            .await
            .expect("Anthropic multimodal inline request failed");

        assert!(!result.subject.is_empty(), "subject should not be empty");
        assert!(!result.summary.is_empty(), "summary should not be empty");
    }

    #[cfg(feature = "anthropic")]
    #[tokio::test]
    async fn test_anthropic_multimodal_url_image() {
        let media = media_url(RUST_LOGO_URL, RUST_LOGO_MIME);
        let client = AnthropicClient::from_env()
            .expect("ANTHROPIC_API_KEY must be set for this test")
            .model(AnthropicModel::ClaudeOpus46)
            .temperature(0.0);

        let result: ImageSummary = client
            .materialize_with_media(
                "Describe this image in one concise sentence, focusing on scene type.",
                &[media],
            )
            .await
            .expect("Anthropic multimodal URL request failed");

        assert!(!result.subject.is_empty(), "subject should not be empty");
        assert!(!result.summary.is_empty(), "summary should not be empty");
    }

    #[cfg(feature = "anthropic")]
    #[tokio::test]
    async fn test_anthropic_multimodal_multiple_images() {
        let rust_media = download_media(RUST_LOGO_URL, RUST_LOGO_MIME).await;
        let lake_media = media_url(RUST_LOGO_URL, RUST_LOGO_MIME);
        let client = AnthropicClient::from_env()
            .expect("ANTHROPIC_API_KEY must be set for this test")
            .model(AnthropicModel::ClaudeOpus46)
            .temperature(0.0);

        let result: MultiImageSummary = client
            .materialize_with_media(
                "You are given two images. Return the exact count in image_count and summarize both images.",
                &[rust_media, lake_media],
            )
            .await
            .expect("Anthropic multimodal multi-image request failed");

        assert!(
            result.image_count >= 2,
            "expected at least 2 images, got {}",
            result.image_count
        );
        assert!(!result.summary.is_empty(), "summary should not be empty");
    }
}

//! Integration tests for OpenAI multimodal structured extraction.
//!
//! Requires:
//! - `OPENAI_API_KEY`
//! - `--features openai`

#[path = "common/mod.rs"]
mod common;

#[cfg(test)]
mod openai_multimodal_tests {
    #[cfg(feature = "openai")]
    use rstructor::OpenAIClient;
    use rstructor::{Instructor, LLMClient, OpenAIModel};
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

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn test_openai_multimodal_inline_image() {
        let media = download_media(RUST_LOGO_URL, RUST_LOGO_MIME).await;
        let client = OpenAIClient::from_env()
            .expect("OPENAI_API_KEY must be set for this test")
            .model(OpenAIModel::Gpt52)
            .temperature(0.0);

        let result: ImageSummary = client
            .materialize_with_media(
                "Identify the main subject in this image and summarize it briefly.",
                &[media],
            )
            .await
            .expect("OpenAI multimodal inline request failed");

        assert!(!result.subject.is_empty(), "subject should not be empty");
        assert!(!result.summary.is_empty(), "summary should not be empty");
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn test_openai_multimodal_url_image() {
        let media = media_url(RUST_LOGO_URL, RUST_LOGO_MIME);
        let client = OpenAIClient::from_env()
            .expect("OPENAI_API_KEY must be set for this test")
            .model(OpenAIModel::Gpt52)
            .temperature(0.0);

        let result: ImageSummary = client
            .materialize_with_media(
                "Describe this image in one concise sentence, focusing on scene type.",
                &[media],
            )
            .await
            .expect("OpenAI multimodal URL request failed");

        assert!(!result.subject.is_empty(), "subject should not be empty");
        assert!(!result.summary.is_empty(), "summary should not be empty");
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn test_openai_multimodal_multiple_images() {
        let rust_media = download_media(RUST_LOGO_URL, RUST_LOGO_MIME).await;
        let lake_media = media_url(RUST_LOGO_URL, RUST_LOGO_MIME);
        let client = OpenAIClient::from_env()
            .expect("OPENAI_API_KEY must be set for this test")
            .model(OpenAIModel::Gpt52)
            .temperature(0.0);

        let result: MultiImageSummary = client
            .materialize_with_media(
                "You are given two images. Return the exact count in image_count and summarize both images.",
                &[rust_media, lake_media],
            )
            .await
            .expect("OpenAI multimodal multi-image request failed");

        assert!(
            result.image_count >= 2,
            "expected at least 2 images, got {}",
            result.image_count
        );
        assert!(!result.summary.is_empty(), "summary should not be empty");
    }
}

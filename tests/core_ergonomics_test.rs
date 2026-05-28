//! Tests for the core ergonomics improvements: the media-drop default,
//! and the runtime-selectable `AnyClient`.
#![cfg(feature = "_client")]

use async_trait::async_trait;
use rstructor::{
    GenerateResult, Instructor, LLMClient, MaterializeResult, MediaFile, ModelInfo, RStructorError,
    Result,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Dummy {
    value: String,
}

/// A custom client with no media support: it relies entirely on the default
/// `materialize_with_media` implementation provided by the `LLMClient` trait.
struct NoMediaClient;

#[async_trait]
impl LLMClient for NoMediaClient {
    async fn materialize<T>(&self, _prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        // Sentinel error so callers can confirm this path was taken.
        Err(RStructorError::ValidationError("materialize-called".into()))
    }

    async fn materialize_with_metadata<T>(&self, _prompt: &str) -> Result<MaterializeResult<T>>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        Err(RStructorError::ValidationError("materialize-called".into()))
    }

    async fn generate(&self, _prompt: &str) -> Result<String> {
        Err(RStructorError::ValidationError("generate-called".into()))
    }

    async fn generate_with_metadata(&self, _prompt: &str) -> Result<GenerateResult> {
        Err(RStructorError::ValidationError("generate-called".into()))
    }

    fn from_env() -> Result<Self>
    where
        Self: Sized,
    {
        Ok(NoMediaClient)
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![])
    }
}

#[tokio::test]
async fn media_default_passes_through_when_empty() {
    // With no media, the default delegates to `materialize`.
    let client = NoMediaClient;
    let result = client.materialize_with_media::<Dummy>("hi", &[]).await;
    assert!(
        matches!(result, Err(RStructorError::ValidationError(m)) if m == "materialize-called"),
        "empty media should delegate to materialize()"
    );
}

#[tokio::test]
async fn media_default_errors_instead_of_silently_dropping() {
    // With media, a client lacking media support must error loudly rather than
    // silently discard the media.
    let client = NoMediaClient;
    let media = [MediaFile::new("https://example.com/cat.png", "image/png")];
    let result = client
        .materialize_with_media::<Dummy>("describe", &media)
        .await;
    assert!(
        matches!(result, Err(RStructorError::Unsupported(_))),
        "non-empty media on an unsupported client should return Unsupported, got {result:?}"
    );
}

#[cfg(feature = "openai")]
#[test]
fn any_client_wraps_and_reports_openai() {
    use rstructor::{AnyClient, OpenAIClient, Provider};

    let client: AnyClient = OpenAIClient::new("test-key").unwrap().into();
    assert_eq!(client.provider(), Provider::OpenAI);
}

#[cfg(feature = "anthropic")]
#[test]
fn any_client_wraps_and_reports_anthropic() {
    use rstructor::{AnthropicClient, AnyClient, Provider};

    let client: AnyClient = AnthropicClient::new("test-key").unwrap().into();
    assert_eq!(client.provider(), Provider::Anthropic);
}

#[cfg(feature = "gemini")]
#[test]
fn any_client_wraps_and_reports_gemini() {
    use rstructor::{AnyClient, GeminiClient, Provider};

    let client: AnyClient = GeminiClient::new("test-key").unwrap().into();
    assert_eq!(client.provider(), Provider::Gemini);
}

#[cfg(feature = "grok")]
#[test]
fn any_client_wraps_and_reports_grok() {
    use rstructor::{AnyClient, GrokClient, Provider};

    let client: AnyClient = GrokClient::new("test-key").unwrap().into();
    assert_eq!(client.provider(), Provider::Grok);
}

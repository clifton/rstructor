//! Tests for the core ergonomics improvements: the media-drop default,
//! the runtime-selectable `AnyClient`, and the fluent `Request` builder.
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
///
/// This must be a hand-rolled client, not `MockClient`: we're testing the trait
/// *default*, which `MockClient` deliberately overrides (so media flows offline).
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

/// Fluent `Request` builder routing, exercised with the first-party `MockClient`.
///
/// These assert *how the builder composes and dispatches* by reading back the
/// recorded request (method kind, combined prompt, attached media), rather than
/// hand-rolling an echo client. Requires the `mock` feature (CI runs all features).
#[cfg(feature = "mock")]
mod builder {
    use super::{Dummy, MediaFile};
    use rstructor::{MockClient, RequestExt, RequestKind};

    #[tokio::test]
    async fn generate_has_no_system_by_default() {
        let client = MockClient::new().with_response("ok");
        let out = client.request().generate("hello").await.unwrap();
        assert_eq!(out, "ok");
        // The underlying client received the prompt unchanged.
        assert_eq!(client.last_request().unwrap().prompt, "hello");
    }

    #[tokio::test]
    async fn prepends_system_context() {
        let client = MockClient::new().with_response("ok");
        client.with_system("CTX").generate("hello").await.unwrap();
        assert_eq!(client.last_request().unwrap().prompt, "CTX\n\nhello");
    }

    #[tokio::test]
    async fn materialize_routes_through_materialize_with_combined_prompt() {
        let client = MockClient::new().with_response(r#"{"value":"x"}"#);
        let _: Dummy = client
            .with_system("CTX")
            .materialize("hello")
            .await
            .unwrap();
        let req = client.last_request().unwrap();
        assert_eq!(req.kind, RequestKind::Materialize);
        assert_eq!(req.prompt, "CTX\n\nhello");
    }

    #[tokio::test]
    async fn media_routes_through_materialize_with_media() {
        let client = MockClient::new().with_response(r#"{"value":"x"}"#);
        let media = [MediaFile::new("https://example.com/cat.png", "image/png")];
        let _: Dummy = client
            .with_media(&media)
            .materialize("describe")
            .await
            .unwrap();
        let req = client.last_request().unwrap();
        assert_eq!(req.kind, RequestKind::MaterializeWithMedia);
        assert_eq!(req.prompt, "describe");
        assert_eq!(req.media.len(), 1);
    }

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn generate_stream_prepends_system() {
        use futures_util::StreamExt;
        let client = MockClient::new().with_response("anything");
        let _: Vec<String> = client
            .with_system("CTX")
            .generate_stream("hi")
            .map(|c| c.unwrap())
            .collect()
            .await;
        // The streaming terminal also prepends the system context before dispatch.
        assert_eq!(client.last_request().unwrap().prompt, "CTX\n\nhi");
    }
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

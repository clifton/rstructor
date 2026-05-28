//! Tests for the core ergonomics improvements: the media-drop default,
//! the runtime-selectable `AnyClient`, and the fluent `Request` builder.
#![cfg(feature = "_client")]

use async_trait::async_trait;
use rstructor::{
    GenerateResult, Instructor, LLMClient, MaterializeResult, MediaFile, ModelInfo, RStructorError,
    RequestExt, Result,
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

/// A client that echoes the prompt back, so tests can observe exactly what the
/// `Request` builder sends to the underlying client (e.g. system prepending).
struct EchoClient;

#[async_trait]
impl LLMClient for EchoClient {
    async fn materialize<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        // Surface the received prompt through the error so the test can inspect it.
        Err(RStructorError::ValidationError(prompt.to_string()))
    }

    async fn materialize_with_metadata<T>(&self, prompt: &str) -> Result<MaterializeResult<T>>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        Err(RStructorError::ValidationError(prompt.to_string()))
    }

    async fn generate(&self, prompt: &str) -> Result<String> {
        Ok(prompt.to_string())
    }

    async fn generate_with_metadata(&self, prompt: &str) -> Result<GenerateResult> {
        Ok(GenerateResult {
            text: prompt.to_string(),
            usage: None,
        })
    }

    fn from_env() -> Result<Self>
    where
        Self: Sized,
    {
        Ok(EchoClient)
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![])
    }
}

#[tokio::test]
async fn builder_generate_has_no_system_by_default() {
    let client = EchoClient;
    let out = client.request().generate("hello").await.unwrap();
    assert_eq!(
        out, "hello",
        "no system context should pass the prompt as-is"
    );
}

#[tokio::test]
async fn builder_prepends_system_context() {
    let client = EchoClient;
    let out = client.with_system("CTX").generate("hello").await.unwrap();
    assert_eq!(
        out, "CTX\n\nhello",
        "system context should be prepended to the prompt"
    );
}

#[tokio::test]
async fn builder_materialize_routes_through_materialize() {
    let client = EchoClient;
    // With no media, `materialize` is used and receives the combined prompt.
    let result = client
        .with_system("CTX")
        .materialize::<Dummy>("hello")
        .await;
    assert!(
        matches!(&result, Err(RStructorError::ValidationError(m)) if m == "CTX\n\nhello"),
        "builder materialize should route through materialize with the combined prompt, got {result:?}"
    );
}

#[tokio::test]
async fn builder_with_media_routes_through_materialize_with_media() {
    // `EchoClient` has no media support, so non-empty media must error loudly
    // rather than silently dropping the attachment.
    let client = EchoClient;
    let media = [MediaFile::new("https://example.com/cat.png", "image/png")];
    let result = client
        .with_media(&media)
        .materialize::<Dummy>("describe")
        .await;
    assert!(
        matches!(result, Err(RStructorError::Unsupported(_))),
        "builder media should route through materialize_with_media, got {result:?}"
    );
}

#[cfg(feature = "streaming")]
#[tokio::test]
async fn builder_generate_stream_prepends_system() {
    use futures_util::StreamExt;
    // `EchoClient` uses the default `generate_stream`, which falls back to a single
    // chunk from `generate` — enough to confirm the builder's stream terminal works
    // and prepends the system context.
    let client = EchoClient;
    let chunks: Vec<String> = client
        .with_system("CTX")
        .generate_stream("hi")
        .map(|c| c.unwrap())
        .collect()
        .await;
    assert_eq!(
        chunks.concat(),
        "CTX\n\nhi",
        "streaming terminal should prepend system context"
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

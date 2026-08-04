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

/// A minimal custom client only implements the four core operations. Metadata,
/// media, and reporting behavior comes from compatibility defaults.
struct MinimalClient;

#[async_trait]
impl LLMClient for MinimalClient {
    async fn materialize<T>(&self, _prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        Err(RStructorError::ValidationError("minimal-extract".into()))
    }

    async fn generate(&self, prompt: &str) -> Result<String> {
        Ok(format!("minimal:{prompt}"))
    }

    fn from_env() -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self)
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn minimal_custom_client_inherits_metadata_and_report_defaults() {
    let client = MinimalClient;

    let generated = client.generate_with_metadata("status").await.unwrap();
    assert_eq!(generated.text, "minimal:status");
    assert!(generated.usage.is_none());

    let failure = client
        .extract_with_report::<Dummy>("position")
        .await
        .unwrap_err();
    assert!(matches!(
        failure.error(),
        RStructorError::ValidationError(message) if message == "minimal-extract"
    ));
    assert!(failure.report.attempts.is_empty());
    assert!(!failure.report.attempts_complete);
}

#[tokio::test]
async fn extract_alias_delegates_to_materialize_for_custom_clients() {
    let client = NoMediaClient;
    let result = client.extract::<Dummy>("hi").await;
    assert!(
        matches!(result, Err(RStructorError::ValidationError(message)) if message == "materialize-called"),
        "extract() must preserve compatibility with existing custom clients"
    );
}

#[tokio::test]
async fn extract_report_fallback_is_honest_for_custom_clients() {
    let client = NoMediaClient;
    let failure = client.extract_with_report::<Dummy>("hi").await.unwrap_err();

    assert!(matches!(
        failure.error(),
        RStructorError::ValidationError(message) if message == "materialize-called"
    ));
    assert!(failure.report.attempts.is_empty());
    assert!(failure.report.final_usage.is_none());
    assert!(failure.report.cumulative_usage.is_none());
    assert!(!failure.report.attempts_complete);
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

#[tokio::test]
async fn custom_client_attempt_defaults_add_no_required_trait_methods_or_fake_attempts() {
    let client = NoMediaClient;
    let failure = client
        .materialize_with_attempts::<Dummy>("hi")
        .await
        .unwrap_err();

    assert!(matches!(
        failure.error(),
        RStructorError::ValidationError(message) if message == "materialize-called"
    ));
    assert!(failure.attempts.is_empty());
    assert!(failure.cumulative_usage.is_none());
    assert!(!failure.attempts_complete);
}

/// Fluent `Request` builder routing, exercised with the first-party `MockClient`.
///
/// These assert *how the builder composes and dispatches* by reading back the
/// recorded request (method kind, combined prompt, attached media), rather than
/// hand-rolling an echo client. Requires the `mock` feature (CI runs all features).
#[cfg(feature = "mock")]
mod builder {
    use super::{Dummy, MediaFile, RStructorError};
    #[cfg(feature = "streaming")]
    use rstructor::StreamedObject;
    use rstructor::{MockClient, RequestExt, RequestKind};

    #[cfg(feature = "streaming")]
    fn assert_streaming_media_error(error: RStructorError) {
        let RStructorError::Unsupported(message) = error else {
            panic!("expected Unsupported, got {error:?}");
        };
        assert_eq!(
            message,
            "streaming requests with media are not supported; remove the media attachment or use a non-streaming request terminal"
        );
        assert!(!message.contains("confidential"));
        assert!(!message.contains("iVBOR"));
    }

    #[cfg(feature = "streaming")]
    fn real_world_media() -> Vec<MediaFile> {
        vec![
            MediaFile::new(
                "https://files.example.com/confidential-risk-chart.png",
                "image/png",
            ),
            // Sanitized bytes from the header of a real PNG. The error must not
            // expose their base64 representation.
            MediaFile::from_bytes(b"\x89PNG\r\n\x1a\n", "image/png"),
        ]
    }

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
    async fn extract_routes_through_structured_request_with_combined_prompt() {
        let client = MockClient::new().with_response(r#"{"value":"x"}"#);
        let value: Dummy = client.with_system("CTX").extract("hello").await.unwrap();

        assert_eq!(value.value, "x");
        let request = client.last_request().unwrap();
        assert_eq!(request.kind, RequestKind::Materialize);
        assert_eq!(request.prompt, "CTX\n\nhello");
    }

    #[tokio::test]
    async fn extract_report_preserves_builder_system_and_media_routing() {
        let client = MockClient::new().with_response(r#"{"value":"risk chart"}"#);
        let media = [MediaFile::new(
            "https://example.com/risk-chart.png",
            "image/png",
        )];

        let extraction = client
            .with_system("Use the fund's base currency.")
            .media(media.to_vec())
            .extract_with_report::<Dummy>("read the chart")
            .await
            .unwrap();

        assert_eq!(extraction.data.value, "risk chart");
        assert_eq!(extraction.report.attempts.len(), 1);
        let request = client.last_request().unwrap();
        assert_eq!(request.kind, RequestKind::MaterializeWithMediaAndAttempts);
        assert_eq!(
            request.prompt,
            "Use the fund's base currency.\n\nread the chart"
        );
        assert_eq!(request.media.len(), 1);
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

    #[tokio::test]
    async fn attempt_terminal_preserves_system_and_media_routing() {
        let client = MockClient::new().with_response(r#"{"value":"risk chart"}"#);
        let media = [MediaFile::new(
            "https://example.com/risk-chart.png",
            "image/png",
        )];

        let report = client
            .with_system("Use the fund's base currency.")
            .media(media.to_vec())
            .materialize_with_attempts::<Dummy>("read the chart")
            .await
            .unwrap();

        assert_eq!(report.data.value, "risk chart");
        assert_eq!(report.attempts.len(), 1);
        let request = client.last_request().unwrap();
        assert_eq!(request.kind, RequestKind::MaterializeWithMediaAndAttempts);
        assert_eq!(
            request.prompt,
            "Use the fund's base currency.\n\nread the chart"
        );
        assert_eq!(request.media.len(), 1);
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

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn generate_stream_rejects_media_without_calling_client() {
        use futures_util::StreamExt;

        let client = MockClient::new().with_response("must not be consumed");
        let media = real_world_media();
        let mut stream = client.with_media(&media[..1]).generate_stream("analyze");

        assert_streaming_media_error(
            stream
                .next()
                .await
                .expect("one error must be yielded")
                .unwrap_err(),
        );
        assert!(stream.next().await.is_none());
        assert_eq!(client.request_count(), 0);
        assert!(!client.responses_exhausted());
    }

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn materialize_stream_rejects_media_without_calling_client() {
        use futures_util::StreamExt;

        let client = MockClient::new().with_response(r#"{"value":"must not be consumed"}"#);
        let media = real_world_media();
        let mut stream = client
            .with_media(&media)
            .materialize_stream::<Dummy>("analyze");

        assert_streaming_media_error(
            stream
                .next()
                .await
                .expect("one error must be yielded")
                .unwrap_err(),
        );
        assert!(stream.next().await.is_none());
        assert_eq!(client.request_count(), 0);
        assert!(!client.responses_exhausted());
    }

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn materialize_iter_rejects_media_without_calling_client() {
        use futures_util::StreamExt;

        let client =
            MockClient::new().with_response(r#"{"items":[{"value":"must not be consumed"}]}"#);
        let media = real_world_media();
        let mut stream = client
            .with_media(&media)
            .materialize_iter::<Dummy>("analyze");

        assert_streaming_media_error(
            stream
                .next()
                .await
                .expect("one error must be yielded")
                .unwrap_err(),
        );
        assert!(stream.next().await.is_none());
        assert_eq!(client.request_count(), 0);
        assert!(!client.responses_exhausted());
    }

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn empty_media_streams_delegate_with_system_context() {
        use futures_util::StreamExt;

        let text_client = MockClient::new().with_response("VaR is $2.1mm");
        let text: Vec<_> = text_client
            .with_system("Use the risk policy.")
            .media(Vec::new())
            .generate_stream("Summarize VaR")
            .collect()
            .await;
        assert_eq!(
            text.into_iter()
                .collect::<rstructor::Result<Vec<_>>>()
                .unwrap(),
            vec!["VaR is $2.1mm"]
        );
        let request = text_client.last_request().unwrap();
        assert_eq!(request.kind, RequestKind::GenerateStream);
        assert_eq!(request.prompt, "Use the risk policy.\n\nSummarize VaR");

        let object_client = MockClient::new().with_response(r#"{"value":"$2.1mm"}"#);
        let objects: Vec<_> = object_client
            .with_system("Use the risk policy.")
            .media(Vec::new())
            .materialize_stream::<Dummy>("Summarize VaR")
            .collect()
            .await;
        assert!(matches!(
            objects.last(),
            Some(Ok(StreamedObject::Complete(Dummy { value }))) if value == "$2.1mm"
        ));
        let request = object_client.last_request().unwrap();
        assert_eq!(request.kind, RequestKind::MaterializeStream);
        assert_eq!(request.prompt, "Use the risk policy.\n\nSummarize VaR");

        let item_client = MockClient::new().with_response(r#"{"items":[{"value":"$2.1mm"}]}"#);
        let items: Vec<_> = item_client
            .with_system("Use the risk policy.")
            .media(Vec::new())
            .materialize_iter::<Dummy>("Summarize VaR")
            .collect()
            .await;
        assert_eq!(items.len(), 1);
        assert!(matches!(
            items.first(),
            Some(Ok(Dummy { value })) if value == "$2.1mm"
        ));
        let request = item_client.last_request().unwrap();
        assert_eq!(request.kind, RequestKind::MaterializeIter);
        assert_eq!(request.prompt, "Use the risk policy.\n\nSummarize VaR");
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

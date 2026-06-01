//! Offline tests for [`MockClient`]. None of these touch the network or need an
//! API key. The core suite runs in any build with the `mock` feature (including
//! schema-only); the streaming/tools/builder sections gate on their features.

#![cfg(feature = "mock")]

use rstructor::{Instructor, LLMClient, MockClient, MockResponse, RStructorError, RequestKind};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[llm(validate = "validate_movie")]
struct Movie {
    title: String,
    year: u16,
}

fn validate_movie(m: &Movie) -> rstructor::Result<()> {
    if m.year < 1888 {
        return Err(RStructorError::ValidationError(format!(
            "year {} predates cinema",
            m.year
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Core: materialize / generate / metadata / list_models / recording
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_returns_scripted_text() {
    let client = MockClient::new().with_response("a haiku");
    assert_eq!(client.generate("write a haiku").await.unwrap(), "a haiku");
    assert_eq!(client.last_request().unwrap().kind, RequestKind::Generate);
}

#[tokio::test]
async fn generate_with_metadata_usage_is_none_by_default() {
    let client = MockClient::new().with_response("hello");
    let result = client.generate_with_metadata("p").await.unwrap();
    assert_eq!(result.text, "hello");
    assert!(result.usage.is_none());
}

#[tokio::test]
async fn metadata_usage_can_be_configured() {
    use rstructor::TokenUsage;
    let client = MockClient::new()
        .with_response(r#"{"title":"A","year":2000}"#)
        .with_usage(TokenUsage::new("mock-model", 11, 22));
    let result = client
        .materialize_with_metadata::<Movie>("p")
        .await
        .unwrap();
    assert_eq!(result.data.year, 2000);
    let usage = result.usage.unwrap();
    assert_eq!(usage.input_tokens, 11);
    assert_eq!(usage.total_tokens(), 33);
}

#[tokio::test]
async fn queue_is_fifo() {
    let client = MockClient::new()
        .with_response(r#"{"title":"First","year":2001}"#)
        .with_response(r#"{"title":"Second","year":2002}"#);
    let a: Movie = client.materialize("p").await.unwrap();
    let b: Movie = client.materialize("p").await.unwrap();
    assert_eq!(a.title, "First");
    assert_eq!(b.title, "Second");
    assert!(client.responses_exhausted());
}

#[tokio::test]
async fn with_responses_bulk_and_json_helper() {
    let client = MockClient::new()
        .with_json(&Movie {
            title: "Dune".into(),
            year: 2021,
        })
        .unwrap()
        .with_responses(vec![MockResponse::text(
            r#"{"title":"Arrival","year":2016}"#,
        )]);
    let a: Movie = client.materialize("p").await.unwrap();
    let b: Movie = client.materialize("p").await.unwrap();
    assert_eq!(a.title, "Dune");
    assert_eq!(b.title, "Arrival");
}

#[tokio::test]
async fn list_models_default_and_custom() {
    use rstructor::ModelInfo;
    let client = MockClient::new();
    assert_eq!(client.list_models().await.unwrap().len(), 1);
    assert_eq!(client.last_request().unwrap().kind, RequestKind::ListModels);

    let client = MockClient::new().with_models(vec![ModelInfo {
        id: "gpt-test".into(),
        name: None,
        description: None,
    }]);
    let models = client.list_models().await.unwrap();
    assert_eq!(models[0].id, "gpt-test");
}

#[tokio::test]
async fn materialize_with_media_records_media() {
    use rstructor::MediaFile;
    let client = MockClient::new().with_response(r#"{"title":"Poster","year":1999}"#);
    let media = [MediaFile::new(
        "https://example.com/poster.png",
        "image/png",
    )];
    let movie: Movie = client
        .materialize_with_media("describe", &media)
        .await
        .unwrap();
    assert_eq!(movie.title, "Poster");
    let req = client.last_request().unwrap();
    assert_eq!(req.kind, RequestKind::MaterializeWithMedia);
    assert_eq!(req.media.len(), 1);
    assert_eq!(req.media[0].mime_type, "image/png");
}

#[tokio::test]
async fn custom_default_response_used_after_exhaustion() {
    let client =
        MockClient::new().with_default_response(MockResponse::error(RStructorError::Timeout));
    assert_eq!(
        client.materialize::<Movie>("p").await.unwrap_err(),
        RStructorError::Timeout
    );
}

#[tokio::test]
async fn clear_requests_resets_log_only() {
    let client = MockClient::new()
        .with_response(r#"{"title":"A","year":2000}"#)
        .with_response(r#"{"title":"B","year":2001}"#);
    let _: Movie = client.materialize("p").await.unwrap();
    assert_eq!(client.request_count(), 1);
    client.clear_requests();
    assert_eq!(client.request_count(), 0);
    // Queue is untouched: the second response is still available.
    let m: Movie = client.materialize("p").await.unwrap();
    assert_eq!(m.title, "B");
}

#[tokio::test]
async fn nested_validation_recurses_through_the_mock() {
    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct Festival {
        name: String,
        films: Vec<Movie>, // each Movie is validated recursively
    }
    // Inner film has an invalid year → the whole materialize must fail.
    let client = MockClient::new().with_response(
        r#"{"name":"Cannes","films":[{"title":"OK","year":2000},{"title":"Bad","year":1000}]}"#,
    );
    let err = client.materialize::<Festival>("p").await.unwrap_err();
    assert!(matches!(err, RStructorError::ValidationError(_)));
}

// ---------------------------------------------------------------------------
// Streaming (requires `streaming`, which implies `_client`)
// ---------------------------------------------------------------------------

#[cfg(feature = "streaming")]
mod streaming {
    use super::*;
    use futures_util::StreamExt;
    use rstructor::StreamedObject;

    #[tokio::test]
    async fn generate_stream_emits_text() {
        let client = MockClient::new().with_response("streamed text");
        let mut stream = client.generate_stream("p");
        let mut out = String::new();
        while let Some(chunk) = stream.next().await {
            out.push_str(&chunk.unwrap());
        }
        assert_eq!(out, "streamed text");
    }

    #[tokio::test]
    async fn materialize_stream_yields_partial_then_complete() {
        let client = MockClient::new().with_response(r#"{"title":"Heat","year":1995}"#);
        let mut stream = client.materialize_stream::<Movie>("p");
        let mut saw_partial = false;
        let mut complete: Option<Movie> = None;
        while let Some(item) = stream.next().await {
            match item.unwrap() {
                StreamedObject::Partial(_) => saw_partial = true,
                StreamedObject::Complete(m) => complete = Some(m),
            }
        }
        assert!(saw_partial, "expected at least one Partial snapshot");
        assert_eq!(complete.unwrap().title, "Heat");
    }

    #[tokio::test]
    async fn materialize_stream_validation_failure_ends_with_err() {
        let client = MockClient::new().with_response(r#"{"title":"Old","year":1000}"#);
        let mut stream = client.materialize_stream::<Movie>("p");
        let mut last_err = None;
        while let Some(item) = stream.next().await {
            if let Err(e) = item {
                last_err = Some(e);
            }
        }
        assert!(matches!(last_err, Some(RStructorError::ValidationError(_))));
    }

    #[tokio::test]
    async fn materialize_iter_items_wrapper() {
        let client = MockClient::new()
            .with_response(r#"{"items":[{"title":"A","year":2001},{"title":"B","year":2002}]}"#);
        let mut stream = client.materialize_iter::<Movie>("p");
        let mut titles = Vec::new();
        while let Some(item) = stream.next().await {
            titles.push(item.unwrap().title);
        }
        assert_eq!(titles, vec!["A", "B"]);
    }

    #[tokio::test]
    async fn materialize_iter_bare_array() {
        let client = MockClient::new()
            .with_response(r#"[{"title":"X","year":2003},{"title":"Y","year":2004}]"#);
        let stream = client.materialize_iter::<Movie>("p");
        let count = stream.count().await;
        assert_eq!(count, 2);
    }
}

// ---------------------------------------------------------------------------
// Tools (requires `tools`, which implies `_client`)
// ---------------------------------------------------------------------------

#[cfg(feature = "tools")]
mod tools {
    use super::*;
    use rstructor::{FnTool, RequestExt, Toolbox};
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

    #[derive(Instructor, Serialize, Deserialize)]
    struct PingArgs {
        #[allow(dead_code)]
        value: u32,
    }

    #[tokio::test]
    async fn run_tool_loop_records_names_and_returns_final_text() {
        let tb = Toolbox::new().with(FnTool::new(
            "ping",
            "a ping tool",
            |_a: PingArgs| async move { Ok(json!({ "pong": true })) },
        ));
        let client = MockClient::new().with_response("final answer");
        let out = client.with_tools(&tb).run("do it").await.unwrap();
        assert_eq!(out, "final answer");
        let req = client.last_request().unwrap();
        assert_eq!(req.kind, RequestKind::RunToolLoop);
        assert_eq!(req.tool_names, vec!["ping".to_string()]);
    }

    #[tokio::test]
    async fn scripted_tool_loop_actually_invokes_the_tool() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_in_tool = calls.clone();
        let tb = Toolbox::new().with(FnTool::new("ping", "a ping tool", move |a: PingArgs| {
            let c = calls_in_tool.clone();
            async move {
                c.fetch_add(1, SeqCst);
                Ok(json!({ "echo": a.value }))
            }
        }));
        let client = MockClient::new()
            .with_tool_script(vec![("ping".to_string(), json!({ "value": 7 }))])
            .with_response("done");
        let out = client.with_tools(&tb).run("call the tool").await.unwrap();
        assert_eq!(out, "done");
        assert_eq!(calls.load(SeqCst), 1, "the tool's invoke should have run");
    }

    #[tokio::test]
    async fn scripted_tool_loop_unknown_tool_errors() {
        let tb = Toolbox::new();
        let client = MockClient::new().with_tool_script(vec![("nope".to_string(), json!({}))]);
        let err = client.with_tools(&tb).run("p").await.unwrap_err();
        assert!(matches!(err, RStructorError::Unsupported(_)));
    }
}

// ---------------------------------------------------------------------------
// Fluent builder integration (requires `_client`, brought in by any provider)
// ---------------------------------------------------------------------------

#[cfg(feature = "_client")]
mod builder {
    use super::*;
    use rstructor::RequestExt;

    #[tokio::test]
    async fn with_system_is_prepended_into_recorded_prompt() {
        let client = MockClient::new().with_response(r#"{"title":"Sys","year":2005}"#);
        let _: Movie = client
            .with_system("Always answer in USD.")
            .materialize("Describe a film")
            .await
            .unwrap();
        let req = client.last_request().unwrap();
        // The builder prepends the system text before dispatch.
        assert!(req.prompt.contains("Always answer in USD."));
        assert!(req.prompt.contains("Describe a film"));
    }
}

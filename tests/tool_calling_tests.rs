//! Tool-calling tests. Only compiled with `--features tools`.
//!
//! The unit-style tests (schema, invocation) run offline; the loop tests hit the
//! live provider APIs and are gated on each provider's feature.
#![cfg(feature = "tools")]

use rstructor::{DynTool, FnTool, Instructor, RequestExt, Toolbox};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Instructor, Serialize, Deserialize)]
struct AddArgs {
    #[llm(description = "First addend")]
    a: i64,
    #[llm(description = "Second addend")]
    b: i64,
}

#[test]
fn fn_tool_exposes_schema_and_metadata() {
    let tool = FnTool::new("add", "Add two integers", |args: AddArgs| async move {
        Ok(json!({ "sum": args.a + args.b }))
    });
    assert_eq!(tool.name(), "add");
    assert_eq!(tool.description(), "Add two integers");
    let schema = tool.parameters_schema();
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"].get("a").is_some());
    assert!(schema["properties"].get("b").is_some());
}

#[tokio::test]
async fn fn_tool_invokes_with_deserialized_args() {
    let tool = FnTool::new("add", "Add two integers", |args: AddArgs| async move {
        Ok(json!({ "sum": args.a + args.b }))
    });
    let result = tool.invoke_json(json!({ "a": 2, "b": 3 })).await.unwrap();
    assert_eq!(result, json!({ "sum": 5 }));
}

#[tokio::test]
async fn unknown_args_error_is_surfaced() {
    let tool = FnTool::new("add", "Add", |args: AddArgs| async move {
        Ok(json!(args.a + args.b))
    });
    // Missing required field `b`.
    let err = tool.invoke_json(json!({ "a": 1 })).await;
    assert!(err.is_err());
}

// ---- Live agentic-loop tests (one per provider) ----

#[derive(Instructor, Serialize, Deserialize)]
struct WeatherArgs {
    #[llm(description = "City name")]
    city: String,
}

#[cfg(any(
    feature = "openai",
    feature = "grok",
    feature = "anthropic",
    feature = "gemini"
))]
use std::sync::Arc;
#[cfg(any(
    feature = "openai",
    feature = "grok",
    feature = "anthropic",
    feature = "gemini"
))]
use std::sync::atomic::{AtomicBool, Ordering};

/// A toolbox with a single `get_weather` tool that records whether it was called.
#[cfg(any(
    feature = "openai",
    feature = "grok",
    feature = "anthropic",
    feature = "gemini"
))]
fn weather_toolbox(called: Arc<AtomicBool>) -> Toolbox {
    Toolbox::new().with(FnTool::new(
        "get_weather",
        "Get the current temperature in Fahrenheit for a city",
        move |args: WeatherArgs| {
            let called = called.clone();
            async move {
                called.store(true, Ordering::SeqCst);
                Ok(json!({ "city": args.city, "temp_f": 71 }))
            }
        },
    ))
}

const TOOL_PROMPT: &str =
    "Use the get_weather tool to find the temperature in Paris, then state it.";

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_tool_loop() {
    use rstructor::OpenAIClient;
    let called = Arc::new(AtomicBool::new(false));
    let toolbox = weather_toolbox(called.clone());
    let client = OpenAIClient::from_env().unwrap().model("gpt-4.1-mini");
    let answer = client
        .with_tools(&toolbox)
        .run(TOOL_PROMPT)
        .await
        .expect("tool loop should succeed");
    assert!(
        called.load(Ordering::SeqCst),
        "tool should have been called"
    );
    assert!(answer.contains("71"), "answer should cite 71: {answer}");
}

#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_tool_loop() {
    use rstructor::GrokClient;
    let called = Arc::new(AtomicBool::new(false));
    let toolbox = weather_toolbox(called.clone());
    let client = GrokClient::from_env().unwrap();
    let answer = client
        .with_tools(&toolbox)
        .run(TOOL_PROMPT)
        .await
        .expect("tool loop should succeed");
    assert!(
        called.load(Ordering::SeqCst),
        "tool should have been called"
    );
    assert!(answer.contains("71"), "answer should cite 71: {answer}");
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_tool_loop() {
    use rstructor::AnthropicClient;
    let called = Arc::new(AtomicBool::new(false));
    let toolbox = weather_toolbox(called.clone());
    let client = AnthropicClient::from_env()
        .unwrap()
        .model("claude-haiku-4-5-20251001")
        .max_tokens(1024);
    let answer = client
        .with_tools(&toolbox)
        .run(TOOL_PROMPT)
        .await
        .expect("tool loop should succeed");
    assert!(
        called.load(Ordering::SeqCst),
        "tool should have been called"
    );
    assert!(answer.contains("71"), "answer should cite 71: {answer}");
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_tool_loop() {
    use rstructor::GeminiClient;
    let called = Arc::new(AtomicBool::new(false));
    let toolbox = weather_toolbox(called.clone());
    let client = GeminiClient::from_env().unwrap().model("gemini-2.5-flash");
    let answer = client
        .with_tools(&toolbox)
        .run(TOOL_PROMPT)
        .await
        .expect("tool loop should succeed");
    assert!(
        called.load(Ordering::SeqCst),
        "tool should have been called"
    );
    assert!(answer.contains("71"), "answer should cite 71: {answer}");
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn tool_request_supports_system_prompt() {
    use rstructor::OpenAIClient;
    let called = Arc::new(AtomicBool::new(false));
    let toolbox = weather_toolbox(called.clone());
    let client = OpenAIClient::from_env().unwrap().model("gpt-4.1-mini");
    let answer = client
        .with_tools(&toolbox)
        .system("Always answer in French.")
        .run(TOOL_PROMPT)
        .await
        .expect("tool loop with system prompt should succeed");
    assert!(
        called.load(Ordering::SeqCst),
        "tool should have been called"
    );
    assert!(answer.contains("71"), "answer should cite 71: {answer}");
}

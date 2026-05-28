//! Tool-calling tests. Only compiled with `--features tools`.
//!
//! The unit-style tests (schema, invocation) run offline; the loop test hits the
//! live OpenAI API and is gated on the `openai` feature.
#![cfg(feature = "tools")]

use rstructor::{DynTool, FnTool, Instructor, Toolbox};
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

#[derive(Instructor, Serialize, Deserialize)]
struct WeatherArgs {
    #[llm(description = "City name")]
    city: String,
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_tool_loop_calls_tool_and_answers() {
    use rstructor::OpenAIClient;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let called = Arc::new(AtomicBool::new(false));
    let flag = called.clone();

    let toolbox = Toolbox::new().with(FnTool::new(
        "get_weather",
        "Get the current temperature in Fahrenheit for a city",
        move |args: WeatherArgs| {
            let flag = flag.clone();
            async move {
                flag.store(true, Ordering::SeqCst);
                Ok(json!({ "city": args.city, "temp_f": 71 }))
            }
        },
    ));

    let client = OpenAIClient::from_env().unwrap().model("gpt-4.1-mini");
    let answer = client
        .run_with_tools(
            "Use the get_weather tool to find the temperature in Paris, then state it.",
            &toolbox,
        )
        .await
        .expect("tool loop should succeed");

    assert!(
        called.load(Ordering::SeqCst),
        "the get_weather tool should have been invoked"
    );
    assert!(
        answer.contains("71"),
        "final answer should cite 71: {answer}"
    );
}

#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_tool_loop_calls_tool_and_answers() {
    use rstructor::GrokClient;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let called = Arc::new(AtomicBool::new(false));
    let flag = called.clone();

    let toolbox = Toolbox::new().with(FnTool::new(
        "get_weather",
        "Get the current temperature in Fahrenheit for a city",
        move |args: WeatherArgs| {
            let flag = flag.clone();
            async move {
                flag.store(true, Ordering::SeqCst);
                Ok(json!({ "city": args.city, "temp_f": 71 }))
            }
        },
    ));

    let client = GrokClient::from_env().unwrap();
    let answer = client
        .run_with_tools(
            "Use the get_weather tool to find the temperature in Paris, then state it.",
            &toolbox,
        )
        .await
        .expect("tool loop should succeed");

    assert!(
        called.load(Ordering::SeqCst),
        "the get_weather tool should have been invoked"
    );
    assert!(
        answer.contains("71"),
        "final answer should cite 71: {answer}"
    );
}

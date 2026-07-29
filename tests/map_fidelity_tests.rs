//! Provider-boundary tests for dynamic map schemas.
//!
//! `HashMap<String, V>` is the canonical Rust representation for runtime keys.
//! Providers must either preserve those keys and the value schema or reject the
//! request locally before a paid HTTP call.

use std::collections::HashMap;

use rstructor::{Instructor, LLMClient, RStructorError};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
struct MappedPortfolio {
    portfolio_id: String,
    as_of: String,
    positions: HashMap<String, Position>,
}

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
struct Position {
    asset_class: String,
    quantity: i64,
    mark_price: f64,
}

#[cfg(feature = "tools")]
#[derive(Instructor, Serialize, Deserialize)]
struct ScenarioArgs {
    scenario_prices: HashMap<String, f64>,
}

#[cfg(feature = "tools")]
fn scenario_price_toolbox() -> rstructor::Toolbox {
    rstructor::Toolbox::new().with(rstructor::FnTool::new(
        "scenario_price",
        "Apply per-symbol scenario prices",
        |args: ScenarioArgs| async move {
            Ok(serde_json::json!({
                "count": args.scenario_prices.len()
            }))
        },
    ))
}

fn assert_map_compatibility_error(
    error: RStructorError,
    expected_provider: &str,
    expected_context: &str,
) {
    assert!(matches!(
        error,
        RStructorError::SchemaCompatibilityError {
            provider,
            context,
            path,
            message,
        } if provider.as_ref() == expected_provider
            && context.as_ref() == expected_context
            && path.as_ref() == "$.properties.positions"
            && message.contains("additionalProperties: false")
    ));
}

#[cfg(feature = "mock")]
#[tokio::test]
async fn mock_client_decodes_the_public_hashmap_representation() {
    let response = include_str!("fixtures/structured/portfolio_map_valid.json");
    let portfolio: MappedPortfolio = rstructor::MockClient::new()
        .with_response(response)
        .materialize("reconcile the portfolio")
        .await
        .unwrap();

    assert_eq!(portfolio.positions.len(), 2);
    assert_eq!(portfolio.positions["AAPL"].quantity, 125_000);
    assert_eq!(portfolio.positions["ESU6"].quantity, -240);
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_rejects_dynamic_maps_before_http() {
    let server = mockito::Server::new_async().await;
    let error = rstructor::OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .no_retries()
        .materialize::<MappedPortfolio>("reconcile the portfolio")
        .await
        .unwrap_err();

    assert_map_compatibility_error(error, "OpenAI", "structured output");
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_attempt_report_keeps_preflight_failure_out_of_provider_ledger() {
    let server = mockito::Server::new_async().await;
    let failure = rstructor::OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .materialize_with_attempts::<MappedPortfolio>("reconcile the portfolio")
        .await
        .unwrap_err();

    assert!(failure.attempts.is_empty());
    assert!(failure.cumulative_usage.is_none());
    assert_map_compatibility_error(failure.into_error(), "OpenAI", "structured output");
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_rejects_dynamic_maps_before_http() {
    let server = mockito::Server::new_async().await;
    let error = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .no_retries()
        .materialize::<MappedPortfolio>("reconcile the portfolio")
        .await
        .unwrap_err();

    assert_map_compatibility_error(error, "Anthropic", "structured output");
}

#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_rejects_dynamic_maps_before_http() {
    let server = mockito::Server::new_async().await;
    let error = rstructor::GrokClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .no_retries()
        .materialize::<MappedPortfolio>("reconcile the portfolio")
        .await
        .unwrap_err();

    assert_map_compatibility_error(error, "Grok", "structured output");
}

#[cfg(all(feature = "openai", feature = "streaming"))]
#[tokio::test]
async fn openai_object_stream_yields_local_map_error() {
    use futures_util::StreamExt;

    let server = mockito::Server::new_async().await;
    let client = rstructor::OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url());
    let mut stream = client.materialize_stream::<MappedPortfolio>("reconcile the portfolio");
    let error = stream.next().await.expect("one local error").unwrap_err();

    assert_map_compatibility_error(error, "OpenAI", "streamed structured output");
    assert!(stream.next().await.is_none());
}

#[cfg(all(feature = "openai", feature = "streaming"))]
#[tokio::test]
async fn openai_item_stream_yields_local_map_error() {
    use futures_util::StreamExt;

    let server = mockito::Server::new_async().await;
    let client = rstructor::OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url());
    let mut stream = client.materialize_iter::<MappedPortfolio>("reconcile portfolios");
    let error = stream.next().await.expect("one local error").unwrap_err();

    assert_map_compatibility_error(error, "OpenAI", "streamed item");
    assert!(stream.next().await.is_none());
}

#[cfg(all(feature = "anthropic", feature = "streaming"))]
#[tokio::test]
async fn anthropic_object_stream_yields_local_map_error() {
    use futures_util::StreamExt;

    let server = mockito::Server::new_async().await;
    let client = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url());
    let mut stream = client.materialize_stream::<MappedPortfolio>("reconcile the portfolio");
    let error = stream.next().await.expect("one local error").unwrap_err();

    assert_map_compatibility_error(error, "Anthropic", "streamed structured output");
    assert!(stream.next().await.is_none());
}

#[cfg(all(feature = "anthropic", feature = "streaming"))]
#[tokio::test]
async fn anthropic_item_stream_yields_local_map_error() {
    use futures_util::StreamExt;

    let server = mockito::Server::new_async().await;
    let client = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url());
    let mut stream = client.materialize_iter::<MappedPortfolio>("reconcile portfolios");
    let error = stream.next().await.expect("one local error").unwrap_err();

    assert_map_compatibility_error(error, "Anthropic", "streamed item");
    assert!(stream.next().await.is_none());
}

#[cfg(all(feature = "grok", feature = "streaming"))]
#[tokio::test]
async fn grok_object_stream_yields_local_map_error() {
    use futures_util::StreamExt;

    let server = mockito::Server::new_async().await;
    let client = rstructor::GrokClient::new("test-key")
        .unwrap()
        .base_url(server.url());
    let mut stream = client.materialize_stream::<MappedPortfolio>("reconcile the portfolio");
    let error = stream.next().await.expect("one local error").unwrap_err();

    assert_map_compatibility_error(error, "Grok", "streamed structured output");
    assert!(stream.next().await.is_none());
}

#[cfg(all(feature = "grok", feature = "streaming"))]
#[tokio::test]
async fn grok_item_stream_yields_local_map_error() {
    use futures_util::StreamExt;

    let server = mockito::Server::new_async().await;
    let client = rstructor::GrokClient::new("test-key")
        .unwrap()
        .base_url(server.url());
    let mut stream = client.materialize_iter::<MappedPortfolio>("reconcile portfolios");
    let error = stream.next().await.expect("one local error").unwrap_err();

    assert_map_compatibility_error(error, "Grok", "streamed item");
    assert!(stream.next().await.is_none());
}

#[cfg(all(feature = "openai", feature = "tools"))]
#[tokio::test]
async fn openai_tool_map_is_rejected_before_http() {
    use rstructor::RequestExt;

    let server = mockito::Server::new_async().await;
    let toolbox = scenario_price_toolbox();
    let error = rstructor::OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .with_tools(&toolbox)
        .run("price the scenario")
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        RStructorError::SchemaCompatibilityError {
            provider,
            context,
            path,
            ..
        } if provider.as_ref() == "OpenAI"
            && context.as_ref() == "tool `scenario_price` arguments"
            && path.as_ref() == "$.properties.scenario_prices"
    ));
}

#[cfg(all(feature = "anthropic", feature = "tools"))]
#[tokio::test]
async fn anthropic_tool_map_is_rejected_before_http() {
    use rstructor::RequestExt;

    let server = mockito::Server::new_async().await;
    let toolbox = scenario_price_toolbox();
    let error = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .with_tools(&toolbox)
        .run("price the scenario")
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        RStructorError::SchemaCompatibilityError {
            provider,
            context,
            path,
            ..
        } if provider.as_ref() == "Anthropic"
            && context.as_ref() == "tool `scenario_price` arguments"
            && path.as_ref() == "$.properties.scenario_prices"
    ));
}

#[cfg(all(feature = "grok", feature = "tools"))]
#[tokio::test]
async fn grok_tool_map_is_rejected_before_http() {
    use rstructor::RequestExt;

    let server = mockito::Server::new_async().await;
    let toolbox = scenario_price_toolbox();
    let error = rstructor::GrokClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .with_tools(&toolbox)
        .run("price the scenario")
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        RStructorError::SchemaCompatibilityError {
            provider,
            context,
            path,
            ..
        } if provider.as_ref() == "Grok"
            && context.as_ref() == "tool `scenario_price` arguments"
            && path.as_ref() == "$.properties.scenario_prices"
    ));
}

#[cfg(all(feature = "gemini", feature = "tools"))]
#[tokio::test]
async fn gemini_tool_sends_and_invokes_a_native_typed_map() {
    use rstructor::RequestExt;
    use serde_json::{Value, json};

    let mut server = mockito::Server::new_async().await;
    let first_request = server
        .mock("POST", "/models/test-map-model:generateContent")
        .match_query(mockito::Matcher::UrlEncoded(
            "key".to_string(),
            "test-key".to_string(),
        ))
        .match_request(|request| {
            let body: Value =
                serde_json::from_str(&request.utf8_lossy_body().expect("UTF-8 request body"))
                    .expect("JSON request body");
            let declaration = &body["tools"][0]["functionDeclarations"][0];
            let map = &declaration["parametersJsonSchema"]["properties"]["scenario_prices"];

            declaration["name"] == "scenario_price"
                && declaration.get("parameters").is_none()
                && map["type"] == "object"
                && map["additionalProperties"]["type"] == "number"
                && map.get("properties").is_none()
                && body["contents"]
                    .as_array()
                    .is_some_and(|items| items.len() == 1)
        })
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "functionCall": {
                                "name": "scenario_price",
                                "args": {
                                    "scenario_prices": {
                                        "AAPL": 190.25,
                                        "ESU6": 6200.0
                                    }
                                }
                            }
                        }]
                    }
                }]
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;
    let second_request = server
        .mock("POST", "/models/test-map-model:generateContent")
        .match_query(mockito::Matcher::UrlEncoded(
            "key".to_string(),
            "test-key".to_string(),
        ))
        .match_request(|request| {
            let body: Value =
                serde_json::from_str(&request.utf8_lossy_body().expect("UTF-8 request body"))
                    .expect("JSON request body");
            body.pointer("/contents/2/parts/0/functionResponse/response/count") == Some(&json!(2))
        })
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "priced 2 symbols" }]
                    }
                }]
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let toolbox = scenario_price_toolbox();
    let answer = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-map-model")
        .with_tools(&toolbox)
        .run("Apply AAPL at 190.25 and ESU6 at 6200")
        .await
        .unwrap();

    first_request.assert_async().await;
    second_request.assert_async().await;
    assert_eq!(answer, "priced 2 symbols");
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_sends_and_decodes_a_native_typed_map() {
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};

    let response = include_str!("fixtures/structured/portfolio_map_valid.json");
    let captured: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = captured.clone();
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/models/test-map-model:generateContent")
        .match_query(mockito::Matcher::UrlEncoded(
            "key".to_string(),
            "test-key".to_string(),
        ))
        .match_request(move |request| {
            let body: Value =
                serde_json::from_str(&request.utf8_lossy_body().expect("UTF-8 request body"))
                    .expect("JSON request body");
            *sink.lock().expect("capture lock") = Some(body);
            true
        })
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": response }]
                    },
                    "finishReason": "STOP"
                }]
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let portfolio: MappedPortfolio = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-map-model")
        .no_retries()
        .materialize("reconcile the portfolio")
        .await
        .unwrap();

    request.assert_async().await;
    assert_eq!(portfolio.positions["AAPL"].quantity, 125_000);
    assert_eq!(portfolio.positions["ESU6"].quantity, -240);

    let body = captured.lock().expect("capture lock");
    let generation_config = &body.as_ref().expect("captured request")["generation_config"];
    let positions_schema = &generation_config["responseJsonSchema"]["properties"]["positions"];
    assert_eq!(positions_schema["type"], "object");
    assert_eq!(
        positions_schema["additionalProperties"]["properties"]["quantity"]["type"],
        "integer"
    );
    assert!(positions_schema.get("properties").is_none());
    assert!(generation_config.get("response_schema").is_none());
}

#[cfg(all(feature = "gemini", feature = "streaming"))]
#[tokio::test]
async fn gemini_object_stream_sends_and_decodes_a_native_typed_map() {
    use futures_util::StreamExt;
    use serde_json::{Value, json};

    let response = include_str!("fixtures/structured/portfolio_map_valid.json");
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/models/test-map-model:streamGenerateContent")
        .match_query(mockito::Matcher::AllOf(vec![
            mockito::Matcher::UrlEncoded("alt".to_string(), "sse".to_string()),
            mockito::Matcher::UrlEncoded("key".to_string(), "test-key".to_string()),
        ]))
        .match_request(|request| {
            let body: Value =
                serde_json::from_str(&request.utf8_lossy_body().expect("UTF-8 request body"))
                    .expect("JSON request body");
            let config = &body["generation_config"];
            let positions = &config["responseJsonSchema"]["properties"]["positions"];

            config["response_mime_type"] == "application/json"
                && config.get("response_schema").is_none()
                && positions["type"] == "object"
                && positions["additionalProperties"]["properties"]["quantity"]["type"] == "integer"
                && positions.get("properties").is_none()
        })
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(format!(
            "data: {}\n\n",
            json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": response }]
                    },
                    "finishReason": "STOP"
                }]
            })
        ))
        .expect(1)
        .create_async()
        .await;

    let client = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-map-model");
    let mut stream = client.materialize_stream::<MappedPortfolio>("reconcile the portfolio");
    let mut complete = None;
    while let Some(update) = stream.next().await {
        match update.unwrap() {
            rstructor::StreamedObject::Partial(value) => {
                assert!(value.is_object());
            }
            rstructor::StreamedObject::Complete(portfolio) => complete = Some(portfolio),
        }
    }

    request.assert_async().await;
    let portfolio = complete.expect("complete streamed portfolio");
    assert_eq!(portfolio.positions["AAPL"].quantity, 125_000);
    assert_eq!(portfolio.positions["ESU6"].quantity, -240);
}

#[cfg(all(feature = "gemini", feature = "streaming"))]
#[tokio::test]
async fn gemini_item_stream_sends_and_decodes_native_typed_maps() {
    use futures_util::StreamExt;
    use serde_json::{Value, json};

    let portfolio: Value =
        serde_json::from_str(include_str!("fixtures/structured/portfolio_map_valid.json"))
            .expect("portfolio fixture");
    let stream_payload = json!({ "items": [portfolio.clone(), portfolio] }).to_string();
    let mut server = mockito::Server::new_async().await;
    let request = server
        .mock("POST", "/models/test-map-model:streamGenerateContent")
        .match_query(mockito::Matcher::AllOf(vec![
            mockito::Matcher::UrlEncoded("alt".to_string(), "sse".to_string()),
            mockito::Matcher::UrlEncoded("key".to_string(), "test-key".to_string()),
        ]))
        .match_request(|request| {
            let body: Value =
                serde_json::from_str(&request.utf8_lossy_body().expect("UTF-8 request body"))
                    .expect("JSON request body");
            let config = &body["generation_config"];
            let positions =
                &config["responseJsonSchema"]["properties"]["items"]["items"]["properties"]
                    ["positions"];

            config["response_mime_type"] == "application/json"
                && config.get("response_schema").is_none()
                && positions["type"] == "object"
                && positions["additionalProperties"]["properties"]["quantity"]["type"] == "integer"
                && positions.get("properties").is_none()
        })
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(format!(
            "data: {}\n\n",
            json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": stream_payload }]
                    },
                    "finishReason": "STOP"
                }]
            })
        ))
        .expect(1)
        .create_async()
        .await;

    let client = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("test-map-model");
    let portfolios = client
        .materialize_iter::<MappedPortfolio>("reconcile portfolios")
        .map(|item| item.unwrap())
        .collect::<Vec<_>>()
        .await;

    request.assert_async().await;
    assert_eq!(portfolios.len(), 2);
    assert!(
        portfolios
            .iter()
            .all(|portfolio| portfolio.positions["AAPL"].quantity == 125_000)
    );
}

//! Tool (function) calling: let the model invoke typed Rust functions and feed
//! the results back, looping until it produces a final answer.
//!
//! Define tools whose argument types derive [`Instructor`](crate::Instructor) (so
//! their JSON Schema is generated for you), collect them in a [`Toolbox`], and run
//! the agentic loop with a client's `with_tools(...).run(prompt)`.
//!
//! This module is only compiled with the `tools` feature.

use std::future::Future;
use std::marker::PhantomData;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{RStructorError, Result};
use crate::model::Instructor;
use crate::schema::SchemaType;

/// A typed tool the model can call.
///
/// Implement this directly, or use [`FnTool`] to wrap a closure. The argument
/// type `Args` must derive [`Instructor`](crate::Instructor); its JSON Schema is
/// sent to the model so it knows how to call the tool.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool's argument type. Its schema is derived via `Instructor`.
    type Args: Instructor + DeserializeOwned + Send;

    /// The tool name the model uses to call it (must be unique within a toolbox).
    fn name(&self) -> String;

    /// A description telling the model what the tool does and when to use it.
    fn description(&self) -> String;

    /// Execute the tool with deserialized arguments, returning a JSON result that
    /// is fed back to the model.
    async fn invoke(&self, args: Self::Args) -> Result<Value>;
}

/// Object-safe, type-erased view of a [`Tool`], used to store heterogeneous tools
/// in a [`Toolbox`]. Implemented automatically for every `Tool`.
#[async_trait]
pub trait DynTool: Send + Sync {
    /// The tool name.
    fn name(&self) -> String;
    /// The tool description.
    fn description(&self) -> String;
    /// The JSON Schema for the tool's arguments (strict form: `additionalProperties:
    /// false`), as used by OpenAI/Grok/Anthropic.
    fn parameters_schema(&self) -> Value;
    /// The argument schema with Gemini-unsupported keywords stripped.
    fn parameters_schema_gemini(&self) -> Value;
    /// Invoke the tool with raw JSON arguments (deserialized into `Args`).
    async fn invoke_json(&self, args: Value) -> Result<Value>;
}

#[async_trait]
impl<T: Tool> DynTool for T {
    fn name(&self) -> String {
        Tool::name(self)
    }

    fn description(&self) -> String {
        Tool::description(self)
    }

    fn parameters_schema(&self) -> Value {
        crate::backend::utils::prepare_strict_schema(&<T::Args as SchemaType>::schema())
    }

    fn parameters_schema_gemini(&self) -> Value {
        crate::backend::utils::prepare_gemini_schema(&<T::Args as SchemaType>::schema())
    }

    async fn invoke_json(&self, args: Value) -> Result<Value> {
        let typed: T::Args = serde_json::from_value(args)
            .map_err(|e| RStructorError::SerializationError(e.to_string()))?;
        self.invoke(typed).await
    }
}

/// A [`Tool`] built from a closure.
///
/// ```no_run
/// # use rstructor::{FnTool, Instructor, Toolbox};
/// # use serde::{Serialize, Deserialize};
/// # use serde_json::json;
/// #[derive(Instructor, Serialize, Deserialize)]
/// struct WeatherArgs {
///     #[llm(description = "City name")]
///     city: String,
/// }
///
/// let tool = FnTool::new("get_weather", "Get the current weather for a city", |args: WeatherArgs| async move {
///     Ok(json!({ "city": args.city, "temp_f": 72 }))
/// });
/// let toolbox = Toolbox::new().with(tool);
/// ```
pub struct FnTool<A, F> {
    name: String,
    description: String,
    func: F,
    _marker: PhantomData<fn() -> A>,
}

impl<A, F> FnTool<A, F> {
    /// Create a tool from a name, description, and an async closure over the
    /// (derived-schema) argument type.
    pub fn new(name: impl Into<String>, description: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            func,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<A, F, Fut> Tool for FnTool<A, F>
where
    A: Instructor + DeserializeOwned + Send + 'static,
    F: Fn(A) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Value>> + Send,
{
    type Args = A;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    async fn invoke(&self, args: A) -> Result<Value> {
        (self.func)(args).await
    }
}

/// A collection of tools made available to the model.
#[derive(Default)]
pub struct Toolbox {
    tools: Vec<Box<dyn DynTool>>,
}

impl Toolbox {
    /// Create an empty toolbox.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tool, returning the toolbox (builder style).
    #[must_use]
    pub fn with(mut self, tool: impl DynTool + 'static) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// Add a tool in place.
    pub fn add(&mut self, tool: impl DynTool + 'static) -> &mut Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// Whether the toolbox has no tools.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Number of tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// The names of the tools in this toolbox, in insertion order.
    #[must_use]
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Find a tool by name.
    pub(crate) fn get(&self, name: &str) -> Option<&dyn DynTool> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(AsRef::as_ref)
    }

    /// Render the tools as OpenAI-compatible `tools` JSON.
    #[cfg(any(feature = "openai", feature = "grok"))]
    fn openai_tools_json(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name(),
                        "description": t.description(),
                        "parameters": t.parameters_schema(),
                    }
                })
            })
            .collect()
    }

    /// Render the tools as Anthropic `tools` JSON.
    #[cfg(feature = "anthropic")]
    fn anthropic_tools_json(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name(),
                    "description": t.description(),
                    "input_schema": t.parameters_schema(),
                })
            })
            .collect()
    }

    /// Render the tools as Gemini `tools` JSON (a single `functionDeclarations`).
    #[cfg(feature = "gemini")]
    fn gemini_tools_json(&self) -> Vec<Value> {
        if self.tools.is_empty() {
            return Vec::new();
        }
        let declarations: Vec<Value> = self
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name(),
                    "description": t.description(),
                    "parameters": t.parameters_schema_gemini(),
                })
            })
            .collect();
        vec![serde_json::json!({ "functionDeclarations": declarations })]
    }
}

/// The default maximum number of model round-trips before the tool loop gives up.
pub(crate) const DEFAULT_MAX_TOOL_ITERATIONS: usize = 10;

/// Run the agentic tool-calling loop against an OpenAI-compatible chat endpoint
/// (OpenAI and Grok). Returns the model's final text answer.
#[cfg(any(feature = "openai", feature = "grok"))]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_openai_compatible_tools(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    provider: &str,
    model: &str,
    temperature: f32,
    max_tokens: Option<u32>,
    reasoning_effort: Option<String>,
    system: Option<&str>,
    prompt: &str,
    media: &[crate::backend::MediaFile],
    toolbox: &Toolbox,
    max_iterations: usize,
) -> Result<String> {
    use crate::backend::{check_response_status, handle_http_error};
    use serde_json::json;
    use tracing::{debug, warn};

    let tools_json = toolbox.openai_tools_json();
    let mut messages: Vec<Value> = Vec::new();
    if let Some(system) = system {
        messages.push(json!({ "role": "system", "content": system }));
    }
    // Encode any attached media with the same content builder as materialize, so
    // images/PDFs are carried (or rejected with a clear error) per provider rules.
    let user_msg = crate::backend::ChatMessage::user_with_media(prompt, media.to_vec());
    let user_content =
        crate::backend::build_openai_compatible_message_content(&user_msg, provider)?;
    messages.push(json!({ "role": "user", "content": user_content }));

    for iteration in 0..max_iterations {
        let mut body = json!({
            "model": model,
            "messages": messages,
            "temperature": temperature,
        });
        if !tools_json.is_empty() {
            body["tools"] = json!(tools_json);
            body["tool_choice"] = json!("auto");
        }
        if let Some(mt) = max_tokens {
            body["max_tokens"] = json!(mt);
        }
        if let Some(ref effort) = reasoning_effort {
            body["reasoning_effort"] = json!(effort);
        }

        let response = client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| handle_http_error(e, provider))?;
        let response = check_response_status(response, provider).await?;
        let payload: Value = response.json().await.map_err(RStructorError::from)?;

        let message = payload
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .ok_or_else(|| {
                RStructorError::api_error(
                    provider,
                    crate::ApiErrorKind::UnexpectedResponse {
                        details: "No message in tool-calling response".to_string(),
                    },
                )
            })?
            .clone();

        let tool_calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .filter(|calls| !calls.is_empty());

        let Some(tool_calls) = tool_calls else {
            // No tool calls: the model produced its final answer.
            let content = message
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            debug!(iteration, "Tool loop finished with final answer");
            return Ok(content);
        };

        // Record the assistant's tool-call message, then execute each call.
        messages.push(message.clone());
        for call in tool_calls {
            let call_id = call.get("id").and_then(Value::as_str).unwrap_or_default();
            let function = call.get("function");
            let name = function
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let args_str = function
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str)
                .unwrap_or("{}");
            let args: Value = serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));

            debug!(tool = name, "Model requested tool call");
            let result = match toolbox.get(name) {
                Some(tool) => tool.invoke_json(args).await.unwrap_or_else(|e| {
                    warn!(tool = name, error = %e, "Tool returned an error");
                    json!({ "error": e.to_string() })
                }),
                None => {
                    warn!(tool = name, "Model called an unknown tool");
                    json!({ "error": format!("unknown tool: {name}") })
                }
            };

            messages.push(json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": serde_json::to_string(&result).unwrap_or_default(),
            }));
        }
    }

    Err(RStructorError::ValidationError(format!(
        "tool-calling loop did not converge within {max_iterations} iterations"
    )))
}

/// Error returned when a tool loop exhausts its iteration budget.
#[cfg(any(feature = "anthropic", feature = "gemini"))]
fn loop_exhausted(max_iterations: usize) -> RStructorError {
    RStructorError::ValidationError(format!(
        "tool-calling loop did not converge within {max_iterations} iterations"
    ))
}

/// Run the agentic tool-calling loop against Anthropic's Messages API.
#[cfg(feature = "anthropic")]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_anthropic_tools(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    temperature: f32,
    max_tokens: u32,
    system: Option<&str>,
    prompt: &str,
    media: &[crate::backend::MediaFile],
    toolbox: &Toolbox,
    max_iterations: usize,
) -> Result<String> {
    use crate::backend::{check_response_status, handle_http_error};
    use serde_json::json;
    use tracing::debug;

    let tools_json = toolbox.anthropic_tools_json();
    let url = format!("{base_url}/messages");
    // Encode any attached media with the same content builder as materialize, so
    // images/PDFs are carried (or rejected with a clear error) per provider rules.
    let user_msg = crate::backend::ChatMessage::user_with_media(prompt, media.to_vec());
    let user_content = crate::backend::build_anthropic_message_content(&user_msg)?;
    let mut messages: Vec<Value> = vec![json!({ "role": "user", "content": user_content })];

    for _ in 0..max_iterations {
        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        });
        if let Some(system) = system {
            body["system"] = json!(system);
        }
        if !tools_json.is_empty() {
            body["tools"] = json!(tools_json);
        }

        let response = client
            .post(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| handle_http_error(e, "Anthropic"))?;
        let response = check_response_status(response, "Anthropic").await?;
        let payload: Value = response.json().await.map_err(RStructorError::from)?;

        let content = payload
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let stop_reason = payload.get("stop_reason").and_then(Value::as_str);

        let tool_uses: Vec<&Value> = content
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
            .collect();

        if stop_reason != Some("tool_use") || tool_uses.is_empty() {
            let text: String = content
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect();
            return Ok(text);
        }

        // Echo the assistant's content (incl. tool_use blocks), then return results.
        let mut results = Vec::with_capacity(tool_uses.len());
        for tu in &tool_uses {
            let id = tu.get("id").and_then(Value::as_str).unwrap_or_default();
            let name = tu.get("name").and_then(Value::as_str).unwrap_or_default();
            let input = tu.get("input").cloned().unwrap_or_else(|| json!({}));

            debug!(tool = name, "Model requested tool call");
            let result = match toolbox.get(name) {
                Some(tool) => tool
                    .invoke_json(input)
                    .await
                    .unwrap_or_else(|e| json!({ "error": e.to_string() })),
                None => json!({ "error": format!("unknown tool: {name}") }),
            };
            results.push(json!({
                "type": "tool_result",
                "tool_use_id": id,
                "content": serde_json::to_string(&result).unwrap_or_default(),
            }));
        }

        messages.push(json!({ "role": "assistant", "content": content }));
        messages.push(json!({ "role": "user", "content": results }));
    }

    Err(loop_exhausted(max_iterations))
}

/// Run the agentic tool-calling loop against Gemini's `generateContent` API.
#[cfg(feature = "gemini")]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_gemini_tools(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    temperature: f32,
    max_tokens: Option<u32>,
    system: Option<&str>,
    prompt: &str,
    media: &[crate::backend::MediaFile],
    toolbox: &Toolbox,
    max_iterations: usize,
) -> Result<String> {
    use crate::backend::{check_response_status, handle_http_error};
    use serde_json::json;
    use tracing::debug;

    let tools_json = toolbox.gemini_tools_json();
    let url = format!("{base_url}/models/{model}:generateContent");
    // Attach any media to the initial user turn, mirroring the materialize path:
    // inline base64 data becomes `inlineData`, URI references become `fileData`.
    let mut user_parts: Vec<Value> = vec![json!({ "text": prompt })];
    for m in media {
        if let Some(data) = m.data.as_ref() {
            user_parts.push(json!({
                "inlineData": { "mimeType": m.mime_type, "data": data }
            }));
        } else {
            user_parts.push(json!({
                "fileData": { "mimeType": m.mime_type, "fileUri": m.uri }
            }));
        }
    }
    let mut contents: Vec<Value> = vec![json!({ "role": "user", "parts": user_parts })];

    for _ in 0..max_iterations {
        let mut generation_config = json!({ "temperature": temperature });
        if let Some(mt) = max_tokens {
            generation_config["maxOutputTokens"] = json!(mt);
        }
        let mut body = json!({ "contents": contents, "generationConfig": generation_config });
        if let Some(system) = system {
            body["systemInstruction"] = json!({ "parts": [{ "text": system }] });
        }
        if !tools_json.is_empty() {
            body["tools"] = json!(tools_json);
        }

        let response = client
            .post(&url)
            .query(&[("key", api_key)])
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| handle_http_error(e, "Gemini"))?;
        let response = check_response_status(response, "Gemini").await?;
        let payload: Value = response.json().await.map_err(RStructorError::from)?;

        let parts = payload
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let function_calls: Vec<&Value> = parts
            .iter()
            .filter(|p| p.get("functionCall").is_some())
            .collect();

        if function_calls.is_empty() {
            let text: String = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect();
            return Ok(text);
        }

        let mut response_parts = Vec::with_capacity(function_calls.len());
        for fc in &function_calls {
            let call = fc.get("functionCall").unwrap();
            let name = call.get("name").and_then(Value::as_str).unwrap_or_default();
            let args = call.get("args").cloned().unwrap_or_else(|| json!({}));

            debug!(tool = name, "Model requested tool call");
            let result = match toolbox.get(name) {
                Some(tool) => tool
                    .invoke_json(args)
                    .await
                    .unwrap_or_else(|e| json!({ "error": e.to_string() })),
                None => json!({ "error": format!("unknown tool: {name}") }),
            };
            // Gemini requires `functionResponse.response` to be a JSON object.
            let response_obj = if result.is_object() {
                result
            } else {
                json!({ "result": result })
            };
            response_parts.push(json!({
                "functionResponse": { "name": name, "response": response_obj }
            }));
        }

        contents.push(json!({ "role": "model", "parts": parts }));
        contents.push(json!({ "role": "user", "parts": response_parts }));
    }

    Err(loop_exhausted(max_iterations))
}

/// A client capable of running the tool-calling loop.
///
/// Implemented for each provider client and driven by the fluent
/// [`Request`](crate::Request) builder (`client.with_tools(..).run(..)`); not
/// called directly. `media` carries any attachments from
/// [`Request::media`](crate::Request::media), included in the initial user turn.
#[doc(hidden)]
#[async_trait]
pub trait ToolRunner {
    async fn run_tool_loop(
        &self,
        system: Option<&str>,
        prompt: &str,
        media: &[crate::backend::MediaFile],
        toolbox: &Toolbox,
        max_iterations: usize,
    ) -> Result<String>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    /// Argument type for the test tool; its schema is derived via `Instructor`.
    #[derive(crate::Instructor, Serialize, Deserialize)]
    struct AddArgs {
        #[llm(description = "First addend")]
        a: i64,
        #[llm(description = "Second addend")]
        b: i64,
    }

    /// Build a simple `add` tool whose arguments derive `Instructor`.
    fn add_tool() -> FnTool<AddArgs, impl Fn(AddArgs) -> std::future::Ready<Result<Value>>> {
        FnTool::new("add", "Add two integers", |args: AddArgs| {
            std::future::ready(Ok(json!({ "sum": args.a + args.b })))
        })
    }

    /// A second, distinct tool so multi-tool ordering/lookup can be exercised.
    fn echo_tool() -> FnTool<AddArgs, impl Fn(AddArgs) -> std::future::Ready<Result<Value>>> {
        FnTool::new("echo", "Echo the first addend", |args: AddArgs| {
            std::future::ready(Ok(json!({ "value": args.a })))
        })
    }

    // ---- Toolbox add()/len()/is_empty() ----

    #[test]
    fn empty_toolbox_is_empty_and_len_zero() {
        let toolbox = Toolbox::new();
        assert!(toolbox.is_empty());
        assert_eq!(toolbox.len(), 0);
        assert!(toolbox.tool_names().is_empty());
    }

    #[test]
    fn add_makes_toolbox_non_empty_and_increments_len() {
        let mut toolbox = Toolbox::new();
        toolbox.add(add_tool());
        assert!(!toolbox.is_empty());
        assert_eq!(toolbox.len(), 1);

        toolbox.add(echo_tool());
        assert_eq!(toolbox.len(), 2);
    }

    // ---- Toolbox::get() hit / miss / duplicate-first-wins ----

    #[tokio::test]
    async fn get_returns_matching_tool() {
        let toolbox = Toolbox::new().with(add_tool());
        let tool = toolbox.get("add").expect("add tool should be found");
        assert_eq!(tool.name(), "add");
        let result = tool.invoke_json(json!({ "a": 2, "b": 3 })).await.unwrap();
        assert_eq!(result, json!({ "sum": 5 }));
    }

    #[test]
    fn get_returns_none_for_missing_tool() {
        let toolbox = Toolbox::new().with(add_tool());
        assert!(toolbox.get("nonexistent").is_none());
    }

    #[tokio::test]
    async fn get_with_duplicate_names_dispatches_to_first() {
        // Two tools share the name "add"; the first inserted must win.
        let first = FnTool::new("add", "first add", |args: AddArgs| {
            std::future::ready(Ok(json!({ "which": 1, "sum": args.a + args.b })))
        });
        let second = FnTool::new("add", "second add", |args: AddArgs| {
            std::future::ready(Ok(json!({ "which": 2, "sum": args.a + args.b })))
        });
        let toolbox = Toolbox::new().with(first).with(second);

        assert_eq!(toolbox.len(), 2);
        assert_eq!(toolbox.tool_names(), vec!["add", "add"]);

        let tool = toolbox.get("add").expect("a tool named add should exist");
        let result = tool.invoke_json(json!({ "a": 1, "b": 1 })).await.unwrap();
        assert_eq!(result["which"], json!(1), "first-inserted tool must win");
    }

    // ---- tool_names() insertion order + mixed with()/add() ----

    #[test]
    fn tool_names_preserve_insertion_order_mixed_with_and_add() {
        let mut toolbox = Toolbox::new()
            .with(FnTool::new("first", "f", |args: AddArgs| {
                std::future::ready(Ok(json!(args.a)))
            }))
            .with(FnTool::new("second", "s", |args: AddArgs| {
                std::future::ready(Ok(json!(args.a)))
            }));
        toolbox.add(FnTool::new("third", "t", |args: AddArgs| {
            std::future::ready(Ok(json!(args.a)))
        }));

        assert_eq!(toolbox.tool_names(), vec!["first", "second", "third"]);
    }

    // ---- openai_tools_json render shape ----

    #[cfg(any(feature = "openai", feature = "grok"))]
    #[test]
    fn openai_tools_json_render_shape() {
        let toolbox = Toolbox::new().with(add_tool());
        let rendered = toolbox.openai_tools_json();
        assert_eq!(rendered.len(), 1);

        let entry = &rendered[0];
        assert_eq!(entry["type"], "function");

        let function = &entry["function"];
        assert_eq!(function["name"], "add");
        assert_eq!(function["description"], "Add two integers");

        let params = &function["parameters"];
        assert_eq!(params["type"], "object");
        // Strict schema flips additionalProperties to false.
        assert_eq!(params["additionalProperties"], json!(false));

        let required = params["required"]
            .as_array()
            .expect("required should be an array");
        assert!(required.contains(&json!("a")));
        assert!(required.contains(&json!("b")));

        // The argument properties are present under parameters.
        assert!(params["properties"].get("a").is_some());
        assert!(params["properties"].get("b").is_some());
    }

    // ---- anthropic_tools_json uses input_schema not parameters ----

    #[cfg(feature = "anthropic")]
    #[test]
    fn anthropic_tools_json_uses_input_schema() {
        let toolbox = Toolbox::new().with(add_tool());
        let rendered = toolbox.anthropic_tools_json();
        assert_eq!(rendered.len(), 1);

        let entry = &rendered[0];
        assert_eq!(entry["name"], "add");
        assert_eq!(entry["description"], "Add two integers");

        // Anthropic uses `input_schema`, never `parameters`.
        assert!(
            entry.get("input_schema").is_some(),
            "input_schema must be present"
        );
        assert!(
            entry.get("parameters").is_none(),
            "parameters must be absent for Anthropic"
        );

        let input_schema = &entry["input_schema"];
        assert_eq!(input_schema["type"], "object");
        assert_eq!(input_schema["additionalProperties"], json!(false));
    }

    // ---- gemini_tools_json functionDeclarations wrapper + empty early-return ----

    #[cfg(feature = "gemini")]
    #[test]
    fn gemini_tools_json_empty_returns_empty_vec() {
        let toolbox = Toolbox::new();
        assert!(toolbox.gemini_tools_json().is_empty());
    }

    #[cfg(feature = "gemini")]
    #[test]
    fn gemini_tools_json_wraps_declarations() {
        let toolbox = Toolbox::new().with(add_tool());
        let rendered = toolbox.gemini_tools_json();
        // Populated toolbox yields a single wrapper object.
        assert_eq!(rendered.len(), 1);

        let declarations = rendered[0]["functionDeclarations"]
            .as_array()
            .expect("functionDeclarations should be an array");
        assert_eq!(declarations.len(), 1);

        let decl = &declarations[0];
        assert_eq!(decl["name"], "add");
        assert_eq!(decl["description"], "Add two integers");
        assert_eq!(decl["parameters"]["type"], "object");

        // Gemini schema strips examples/title.
        assert!(
            decl["parameters"].get("examples").is_none(),
            "examples should be stripped from Gemini schema"
        );
        assert!(
            decl["parameters"].get("title").is_none(),
            "title should be stripped from Gemini schema"
        );
    }
}

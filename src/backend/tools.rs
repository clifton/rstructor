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

    /// Find a tool by name.
    fn get(&self, name: &str) -> Option<&dyn DynTool> {
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
    messages.push(json!({ "role": "user", "content": prompt }));

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
    toolbox: &Toolbox,
    max_iterations: usize,
) -> Result<String> {
    use crate::backend::{check_response_status, handle_http_error};
    use serde_json::json;
    use tracing::debug;

    let tools_json = toolbox.anthropic_tools_json();
    let url = format!("{base_url}/messages");
    let mut messages: Vec<Value> = vec![json!({ "role": "user", "content": prompt })];

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
    toolbox: &Toolbox,
    max_iterations: usize,
) -> Result<String> {
    use crate::backend::{check_response_status, handle_http_error};
    use serde_json::json;
    use tracing::debug;

    let tools_json = toolbox.gemini_tools_json();
    let url = format!("{base_url}/models/{model}:generateContent");
    let mut contents: Vec<Value> = vec![json!({ "role": "user", "parts": [{ "text": prompt }] })];

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
/// Implemented for each provider client; you don't call this directly — use the
/// client's `with_tools` to get a [`ToolRequest`].
#[doc(hidden)]
#[async_trait]
pub trait ToolRunner {
    async fn run_tool_loop(
        &self,
        system: Option<&str>,
        prompt: &str,
        toolbox: &Toolbox,
        max_iterations: usize,
    ) -> Result<String>;
}

/// A fluent tool-calling request, created by a client's `with_tools`.
///
/// ```no_run
/// # use rstructor::{OpenAIClient, Toolbox};
/// # async fn example(toolbox: Toolbox) -> Result<(), Box<dyn std::error::Error>> {
/// let client = OpenAIClient::from_env()?;
/// let answer = client
///     .with_tools(&toolbox)
///     .system("You are a concise assistant.")
///     .run("What's the weather in Paris?")
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct ToolRequest<'a, C: ToolRunner + ?Sized> {
    client: &'a C,
    toolbox: &'a Toolbox,
    system: Option<String>,
    max_iterations: usize,
}

impl<'a, C: ToolRunner + ?Sized> ToolRequest<'a, C> {
    /// Create a tool request (used by clients' `with_tools`).
    pub(crate) fn new(client: &'a C, toolbox: &'a Toolbox) -> Self {
        Self {
            client,
            toolbox,
            system: None,
            max_iterations: DEFAULT_MAX_TOOL_ITERATIONS,
        }
    }

    /// Attach a system prompt.
    #[must_use]
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set the maximum number of model round-trips before giving up (default 10).
    #[must_use]
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Run the agentic loop with `prompt` as the user message, returning the
    /// model's final text answer once it stops calling tools.
    pub async fn run(self, prompt: &str) -> Result<String> {
        self.client
            .run_tool_loop(
                self.system.as_deref(),
                prompt,
                self.toolbox,
                self.max_iterations,
            )
            .await
    }
}

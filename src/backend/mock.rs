//! An in-memory [`MockClient`] for offline unit testing.
//!
//! `MockClient` implements [`LLMClient`](crate::LLMClient) without any network or
//! API key, so you can unit-test code that extracts structured data — fast and
//! deterministically. Responses are scripted as **raw payloads** (JSON for
//! `materialize*`, text for `generate*`), so the mock runs the *real*
//! deserialize + [`Instructor::validate`](crate::Instructor::validate) round-trip:
//! you can exercise schema/validation failures, not just happy paths.
//!
//! This module is only compiled with the `mock` feature. It pulls in **no extra
//! dependencies** and works even in a schema-only build
//! (`default-features = false, features = ["derive", "mock"]`); the streaming and
//! tool-calling overrides additionally require the `streaming` / `tools` features.
//!
//! # Example
//!
//! ```
//! # use rstructor::{MockClient, LLMClient, Instructor, RStructorError};
//! # use serde::{Serialize, Deserialize};
//! #[derive(Instructor, Serialize, Deserialize, Debug)]
//! #[llm(validate = "validate_movie")]
//! struct Movie {
//!     title: String,
//!     year: u16,
//! }
//!
//! fn validate_movie(m: &Movie) -> rstructor::Result<()> {
//!     if m.year < 1888 {
//!         return Err(RStructorError::ValidationError("year too early".into()));
//!     }
//!     Ok(())
//! }
//!
//! /// Code under test: generic over any LLM client, so the mock drops right in.
//! async fn extract_movie<C: LLMClient + Sync>(client: &C, blurb: &str) -> rstructor::Result<Movie> {
//!     client.materialize(blurb).await
//! }
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let client = MockClient::new().with_response(r#"{"title": "Inception", "year": 2010}"#);
//! let movie = extract_movie(&client, "Describe Inception").await?;
//! assert_eq!(movie.title, "Inception");
//!
//! // The mock ran the real validator — bad data fails just like a live call.
//! let client = MockClient::new().with_response(r#"{"title": "X", "year": 1700}"#);
//! assert!(extract_movie(&client, "..").await.is_err());
//! # Ok(())
//! # }
//! ```

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::backend::ModelInfo;
use crate::backend::client::{LLMClient, MediaFile};
use crate::backend::usage::{GenerateResult, MaterializeResult, TokenUsage};
use crate::error::{RStructorError, Result};
use crate::model::Instructor;
use crate::schema::SchemaType;

/// One scripted reply the mock will hand back for a call.
///
/// Stored as a raw payload so that `materialize*` runs the real
/// deserialize + [`Instructor::validate`](crate::Instructor::validate) round-trip,
/// letting you reproduce malformed output and validation failures.
#[derive(Debug)]
pub enum MockResponse {
    /// A successful raw payload: JSON for `materialize*`, plain text for `generate*`.
    Text(String),
    /// A pre-built error returned verbatim (e.g. a simulated rate limit).
    Error(RStructorError),
}

impl MockResponse {
    /// Serialize any [`Serialize`](serde::Serialize) value into a JSON [`Text`](MockResponse::Text)
    /// response — convenient for `materialize` happy-paths without writing JSON by hand.
    ///
    /// # Errors
    /// Returns [`RStructorError::SerializationError`] if `value` cannot be serialized.
    pub fn json<T: serde::Serialize>(value: &T) -> Result<Self> {
        serde_json::to_string(value)
            .map(MockResponse::Text)
            .map_err(|e| RStructorError::SerializationError(e.to_string()))
    }

    /// A raw text/JSON response.
    pub fn text(s: impl Into<String>) -> Self {
        MockResponse::Text(s.into())
    }

    /// An error response (e.g. from [`RStructorError::api_error`]).
    pub fn error(err: RStructorError) -> Self {
        MockResponse::Error(err)
    }
}

impl From<&str> for MockResponse {
    fn from(s: &str) -> Self {
        MockResponse::Text(s.to_string())
    }
}

impl From<String> for MockResponse {
    fn from(s: String) -> Self {
        MockResponse::Text(s)
    }
}

/// `RStructorError` is intentionally not `Clone` (its `HttpError`/`JsonError`
/// sources aren't), but a queued/default response may be handed out more than
/// once, so we best-effort clone the clonable variants and stringify the rest.
fn clone_error(e: &RStructorError) -> RStructorError {
    match e {
        RStructorError::ApiError { provider, kind } => RStructorError::ApiError {
            provider: provider.clone(),
            kind: kind.clone(),
        },
        RStructorError::ValidationError(s) => RStructorError::ValidationError(s.clone()),
        RStructorError::SchemaError(s) => RStructorError::SchemaError(s.clone()),
        RStructorError::SerializationError(s) => RStructorError::SerializationError(s.clone()),
        RStructorError::Timeout => RStructorError::Timeout,
        RStructorError::Unsupported(s) => RStructorError::Unsupported(s.clone()),
        // Sources below don't implement Clone; preserve the message instead.
        #[cfg(feature = "_client")]
        RStructorError::HttpError(_) => RStructorError::Unsupported(e.to_string()),
        RStructorError::JsonError(_) => RStructorError::SerializationError(e.to_string()),
    }
}

impl Clone for MockResponse {
    fn clone(&self) -> Self {
        match self {
            MockResponse::Text(s) => MockResponse::Text(s.clone()),
            MockResponse::Error(e) => MockResponse::Error(clone_error(e)),
        }
    }
}

/// Which [`LLMClient`](crate::LLMClient) (or extension) method produced a
/// [`RecordedRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    /// [`LLMClient::materialize`](crate::LLMClient::materialize)
    Materialize,
    /// [`LLMClient::materialize_with_metadata`](crate::LLMClient::materialize_with_metadata)
    MaterializeWithMetadata,
    /// [`LLMClient::materialize_with_media`](crate::LLMClient::materialize_with_media)
    MaterializeWithMedia,
    /// [`LLMClient::generate`](crate::LLMClient::generate)
    Generate,
    /// [`LLMClient::generate_with_media`](crate::LLMClient::generate_with_media)
    GenerateWithMedia,
    /// [`LLMClient::generate_with_metadata`](crate::LLMClient::generate_with_metadata)
    GenerateWithMetadata,
    /// [`LLMClient::list_models`](crate::LLMClient::list_models)
    ListModels,
    /// [`LLMClient::generate_stream`](crate::LLMClient::generate_stream)
    #[cfg(feature = "streaming")]
    GenerateStream,
    /// [`LLMClient::materialize_stream`](crate::LLMClient::materialize_stream)
    #[cfg(feature = "streaming")]
    MaterializeStream,
    /// [`LLMClient::materialize_iter`](crate::LLMClient::materialize_iter)
    #[cfg(feature = "streaming")]
    MaterializeIter,
    /// The tool-calling loop (`with_tools(..).run(..)`).
    #[cfg(feature = "tools")]
    RunToolLoop,
}

/// A single request the mock received, captured for assertions.
#[derive(Debug, Clone)]
pub struct RecordedRequest {
    /// Which trait method was invoked.
    pub kind: RequestKind,
    /// The prompt the method was called with. Note: when called through the fluent
    /// [`Request`](crate::Request) builder with `.with_system(..)`, the system text
    /// is prepended before dispatch, so this is the combined `"system\n\nprompt"`.
    pub prompt: String,
    /// JSON Schema of the target type (set for `materialize*` and object/iter
    /// streaming; `None` for `generate*`).
    pub schema: Option<Value>,
    /// The schema name of the target type, when known.
    pub schema_name: Option<String>,
    /// Media attached to the call (for `materialize_with_media`,
    /// `generate_with_media`, and the tool loop).
    pub media: Vec<MediaFile>,
    /// Tool names offered to the call (for the tool loop; empty otherwise).
    #[cfg(feature = "tools")]
    pub tool_names: Vec<String>,
}

/// A borrowed view of an incoming request, passed to a responder closure so it can
/// branch on the prompt/target type without cloning. Mirrors [`RecordedRequest`].
pub struct MockRequestView<'a> {
    /// Which trait method was invoked.
    pub kind: RequestKind,
    /// The prompt the method was called with.
    pub prompt: &'a str,
    /// JSON Schema of the target type, when known.
    pub schema: Option<&'a Value>,
    /// The schema name of the target type, when known.
    pub schema_name: Option<&'a str>,
    /// Media attached to the call.
    pub media: &'a [MediaFile],
    /// Tool names offered to the call.
    #[cfg(feature = "tools")]
    pub tool_names: &'a [String],
}

impl<'a> MockRequestView<'a> {
    fn bare(kind: RequestKind, prompt: &'a str) -> Self {
        Self {
            kind,
            prompt,
            schema: None,
            schema_name: None,
            media: &[],
            #[cfg(feature = "tools")]
            tool_names: &[],
        }
    }

    fn to_recorded(&self) -> RecordedRequest {
        RecordedRequest {
            kind: self.kind,
            prompt: self.prompt.to_string(),
            schema: self.schema.cloned(),
            schema_name: self.schema_name.map(str::to_string),
            media: self.media.to_vec(),
            #[cfg(feature = "tools")]
            tool_names: self.tool_names.to_vec(),
        }
    }
}

type Responder = Box<dyn Fn(&MockRequestView) -> Option<MockResponse> + Send + Sync>;

struct MockState {
    queue: Mutex<VecDeque<MockResponse>>,
    responder: Mutex<Option<Responder>>,
    log: Mutex<Vec<RecordedRequest>>,
    models: Mutex<Vec<ModelInfo>>,
    default_response: Mutex<MockResponse>,
    default_usage: Mutex<Option<TokenUsage>>,
    /// Extra parse+validate attempts on failure (simulates the provider re-ask
    /// loop): on a failed `materialize`, consume the next queued response.
    retries: Mutex<usize>,
    /// Optional scripted tool invocations performed during the tool loop.
    #[cfg(feature = "tools")]
    tool_script: Mutex<VecDeque<(String, Value)>>,
}

impl Default for MockState {
    fn default() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            responder: Mutex::new(None),
            log: Mutex::new(Vec::new()),
            models: Mutex::new(vec![ModelInfo {
                id: "mock-model".to_string(),
                name: Some("Mock Model".to_string()),
                description: Some("In-memory mock model".to_string()),
            }]),
            default_response: Mutex::new(MockResponse::Error(RStructorError::Unsupported(
                "MockClient: no scripted response configured (use .with_response/.with_responder/.with_default_response)"
                    .to_string(),
            ))),
            default_usage: Mutex::new(None),
            retries: Mutex::new(0),
            #[cfg(feature = "tools")]
            tool_script: Mutex::new(VecDeque::new()),
        }
    }
}

/// An in-memory [`LLMClient`](crate::LLMClient) for offline tests.
///
/// Script responses with [`with_response`](MockClient::with_response) /
/// [`push_response`](MockClient::push_response) (a FIFO queue) and/or
/// [`with_responder`](MockClient::with_responder) (a closure over the request),
/// then call any `LLMClient` method. Every call is recorded; read it back with
/// [`requests`](MockClient::requests), [`request_count`](MockClient::request_count),
/// or [`last_request`](MockClient::last_request).
///
/// `MockClient` is `Clone` (clones share state via `Arc`), `Send`, and `Sync`, so
/// it slots into the same `C: LLMClient + Sync` generic positions the real clients
/// fill, and can be reconfigured after being shared behind an `Arc`.
#[derive(Clone)]
pub struct MockClient {
    inner: Arc<MockState>,
}

impl Default for MockClient {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MockClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockClient")
            .field("queued_responses", &self.inner.queue.lock().unwrap().len())
            .field("recorded_requests", &self.inner.log.lock().unwrap().len())
            .finish()
    }
}

impl MockClient {
    /// Create an empty mock. Until you script responses, every call returns the
    /// configured default error.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MockState::default()),
        }
    }

    // ---- queue scripting (builder, chainable) ----

    /// Queue a response (FIFO). Chainable.
    #[must_use]
    pub fn with_response(self, resp: impl Into<MockResponse>) -> Self {
        self.inner.queue.lock().unwrap().push_back(resp.into());
        self
    }

    /// Queue several responses (FIFO).
    #[must_use]
    pub fn with_responses<I>(self, resps: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<MockResponse>,
    {
        let mut q = self.inner.queue.lock().unwrap();
        for r in resps {
            q.push_back(r.into());
        }
        drop(q);
        self
    }

    /// Queue a JSON-serialized value as a success response.
    ///
    /// # Errors
    /// Returns [`RStructorError::SerializationError`] if `value` cannot be serialized.
    pub fn with_json<T: serde::Serialize>(self, value: &T) -> Result<Self> {
        let resp = MockResponse::json(value)?;
        Ok(self.with_response(resp))
    }

    /// Queue an error response.
    #[must_use]
    pub fn with_error(self, err: RStructorError) -> Self {
        self.with_response(MockResponse::Error(err))
    }

    // ---- queue scripting (post-construction, &self) ----

    /// Queue a response after construction (e.g. on a shared `Arc<MockClient>` clone).
    pub fn push_response(&self, resp: impl Into<MockResponse>) {
        self.inner.queue.lock().unwrap().push_back(resp.into());
    }

    /// Queue an error response after construction.
    pub fn push_error(&self, err: RStructorError) {
        self.push_response(MockResponse::Error(err));
    }

    // ---- closure responder ----

    /// Map a request to a response. Returning `None` falls through to the queue,
    /// then to the default response.
    ///
    /// ```
    /// # use rstructor::{MockClient, LLMClient, MockResponse};
    /// # #[tokio::main]
    /// # async fn main() {
    /// let client = MockClient::new().with_responder(|req| {
    ///     req.prompt.contains("haiku").then(|| MockResponse::text("a haiku"))
    /// });
    /// assert_eq!(client.generate("write a haiku").await.unwrap(), "a haiku");
    /// # }
    /// ```
    #[must_use]
    pub fn with_responder<F>(self, f: F) -> Self
    where
        F: Fn(&MockRequestView) -> Option<MockResponse> + Send + Sync + 'static,
    {
        *self.inner.responder.lock().unwrap() = Some(Box::new(f));
        self
    }

    // ---- list_models / default / usage / retries ----

    /// Set the models returned by [`list_models`](LLMClient::list_models).
    #[must_use]
    pub fn with_models(self, models: Vec<ModelInfo>) -> Self {
        *self.inner.models.lock().unwrap() = models;
        self
    }

    /// Set the response used when the queue and responder are both empty.
    #[must_use]
    pub fn with_default_response(self, resp: impl Into<MockResponse>) -> Self {
        *self.inner.default_response.lock().unwrap() = resp.into();
        self
    }

    /// Attach token usage returned by the `*_with_metadata` methods.
    #[must_use]
    pub fn with_usage(self, usage: TokenUsage) -> Self {
        *self.inner.default_usage.lock().unwrap() = Some(usage);
        self
    }

    /// Allow up to `n` extra `materialize` attempts on parse/validation failure,
    /// consuming the next queued response each time. This simulates the provider
    /// re-ask loop: queue a bad payload followed by a good one and set
    /// `with_retries(1)` to test recovery. (Has no effect with a responder closure,
    /// which is a pure function of the request.)
    ///
    /// ```
    /// # use rstructor::{MockClient, LLMClient, Instructor, RStructorError};
    /// # use serde::{Serialize, Deserialize};
    /// #[derive(Instructor, Serialize, Deserialize)]
    /// #[llm(validate = "non_negative")]
    /// struct Count { n: i32 }
    ///
    /// fn non_negative(c: &Count) -> rstructor::Result<()> {
    ///     if c.n < 0 {
    ///         return Err(RStructorError::ValidationError("must be >= 0".into()));
    ///     }
    ///     Ok(())
    /// }
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = MockClient::new()
    ///     .with_response(r#"{"n": -1}"#) // first attempt fails validation
    ///     .with_response(r#"{"n": 7}"#)  // re-ask succeeds
    ///     .with_retries(1);
    /// let count: Count = client.materialize("p").await?;
    /// assert_eq!(count.n, 7);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn with_retries(self, n: usize) -> Self {
        *self.inner.retries.lock().unwrap() = n;
        self
    }

    /// Script tool invocations the mock performs (in order) during the tool loop,
    /// before returning the final answer. Each `(name, args)` calls the matching
    /// tool in the toolbox, so the tool's `invoke` is exercised offline.
    #[cfg(feature = "tools")]
    #[must_use]
    pub fn with_tool_script<I>(self, calls: I) -> Self
    where
        I: IntoIterator<Item = (String, Value)>,
    {
        let mut s = self.inner.tool_script.lock().unwrap();
        for c in calls {
            s.push_back(c);
        }
        drop(s);
        self
    }

    // ---- assertions / recording ----

    /// All requests received, in order — for asserting *what the client was asked*.
    ///
    /// ```
    /// # use rstructor::{MockClient, LLMClient, RequestKind};
    /// # #[tokio::main]
    /// # async fn main() {
    /// let client = MockClient::new().with_response("ok");
    /// let _ = client.generate("hello").await.unwrap();
    /// assert_eq!(client.request_count(), 1);
    /// let req = client.last_request().unwrap();
    /// assert_eq!(req.kind, RequestKind::Generate);
    /// assert_eq!(req.prompt, "hello");
    /// # }
    /// ```
    #[must_use]
    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.inner.log.lock().unwrap().clone()
    }

    /// Number of requests received.
    #[must_use]
    pub fn request_count(&self) -> usize {
        self.inner.log.lock().unwrap().len()
    }

    /// The most recent request, if any.
    #[must_use]
    pub fn last_request(&self) -> Option<RecordedRequest> {
        self.inner.log.lock().unwrap().last().cloned()
    }

    /// Whether all queued responses have been consumed.
    #[must_use]
    pub fn responses_exhausted(&self) -> bool {
        self.inner.queue.lock().unwrap().is_empty()
    }

    /// Clear the recording log (queue and responder are untouched).
    pub fn clear_requests(&self) {
        self.inner.log.lock().unwrap().clear();
    }

    // ---- internals ----

    fn record(&self, view: &MockRequestView) {
        self.inner.log.lock().unwrap().push(view.to_recorded());
    }

    /// Pick a response without recording: responder closure, then queue, then default.
    fn pick_response(&self, view: &MockRequestView) -> MockResponse {
        {
            let guard = self.inner.responder.lock().unwrap();
            if let Some(f) = guard.as_ref()
                && let Some(r) = f(view)
            {
                return r;
            }
        }
        if let Some(r) = self.inner.queue.lock().unwrap().pop_front() {
            return r;
        }
        self.inner.default_response.lock().unwrap().clone()
    }

    fn resolve_materialize<T>(&self, view: &MockRequestView) -> Result<T>
    where
        T: Instructor + DeserializeOwned,
    {
        let attempts = 1 + *self.inner.retries.lock().unwrap();
        let mut last_err: Option<RStructorError> = None;
        for _ in 0..attempts {
            match self.pick_response(view) {
                MockResponse::Text(s) => match parse_and_validate::<T>(&s) {
                    Ok(v) => return Ok(v),
                    Err(e) => last_err = Some(e),
                },
                // An explicitly scripted error is returned verbatim (not retried).
                MockResponse::Error(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or_else(|| {
            RStructorError::Unsupported("MockClient: no scripted response configured".to_string())
        }))
    }
}

/// Mirror of the real `parse_and_validate_response`: deserialize then validate,
/// mapping a JSON parse failure to a [`ValidationError`](RStructorError::ValidationError)
/// (matching live providers so tests behave identically against either).
fn parse_and_validate<T>(raw: &str) -> Result<T>
where
    T: Instructor + DeserializeOwned,
{
    let value: T = serde_json::from_str(raw).map_err(|e| {
        RStructorError::ValidationError(format!(
            "Failed to parse response as JSON: {e}\nPartial JSON: {raw}"
        ))
    })?;
    value.validate()?;
    Ok(value)
}

#[async_trait]
impl LLMClient for MockClient {
    async fn materialize<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        let schema = <T as SchemaType>::schema().to_json();
        let schema_name = <T as SchemaType>::schema_name();
        let mut view = MockRequestView::bare(RequestKind::Materialize, prompt);
        view.schema = Some(&schema);
        view.schema_name = schema_name.as_deref();
        self.record(&view);
        self.resolve_materialize::<T>(&view)
    }

    async fn materialize_with_media<T>(&self, prompt: &str, media: &[MediaFile]) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        let schema = <T as SchemaType>::schema().to_json();
        let schema_name = <T as SchemaType>::schema_name();
        let mut view = MockRequestView::bare(RequestKind::MaterializeWithMedia, prompt);
        view.schema = Some(&schema);
        view.schema_name = schema_name.as_deref();
        view.media = media;
        self.record(&view);
        self.resolve_materialize::<T>(&view)
    }

    async fn materialize_with_metadata<T>(&self, prompt: &str) -> Result<MaterializeResult<T>>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        let schema = <T as SchemaType>::schema().to_json();
        let schema_name = <T as SchemaType>::schema_name();
        let mut view = MockRequestView::bare(RequestKind::MaterializeWithMetadata, prompt);
        view.schema = Some(&schema);
        view.schema_name = schema_name.as_deref();
        self.record(&view);
        let data = self.resolve_materialize::<T>(&view)?;
        let usage = self.inner.default_usage.lock().unwrap().clone();
        Ok(MaterializeResult { data, usage })
    }

    async fn generate(&self, prompt: &str) -> Result<String> {
        let view = MockRequestView::bare(RequestKind::Generate, prompt);
        self.record(&view);
        match self.pick_response(&view) {
            MockResponse::Text(s) => Ok(s),
            MockResponse::Error(e) => Err(e),
        }
    }

    async fn generate_with_media(&self, prompt: &str, media: &[MediaFile]) -> Result<String> {
        let mut view = MockRequestView::bare(RequestKind::GenerateWithMedia, prompt);
        view.media = media;
        self.record(&view);
        match self.pick_response(&view) {
            MockResponse::Text(s) => Ok(s),
            MockResponse::Error(e) => Err(e),
        }
    }

    async fn generate_with_metadata(&self, prompt: &str) -> Result<GenerateResult> {
        let view = MockRequestView::bare(RequestKind::GenerateWithMetadata, prompt);
        self.record(&view);
        let text = match self.pick_response(&view) {
            MockResponse::Text(s) => s,
            MockResponse::Error(e) => return Err(e),
        };
        let usage = self.inner.default_usage.lock().unwrap().clone();
        Ok(GenerateResult { text, usage })
    }

    #[cfg(feature = "streaming")]
    fn generate_stream<'a>(&'a self, prompt: &'a str) -> crate::backend::streaming::TextStream<'a>
    where
        Self: Sync,
    {
        let view = MockRequestView::bare(RequestKind::GenerateStream, prompt);
        self.record(&view);
        let resp = self.pick_response(&view);
        Box::pin(async_stream::try_stream! {
            let s = match resp {
                MockResponse::Text(s) => s,
                MockResponse::Error(e) => Err(e)?,
            };
            yield s;
        })
    }

    #[cfg(feature = "streaming")]
    fn materialize_stream<'a, T>(
        &'a self,
        prompt: &'a str,
    ) -> crate::backend::streaming::ObjectStream<'a, T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
        Self: Sync,
    {
        use crate::backend::streaming::StreamedObject;
        let schema = <T as SchemaType>::schema().to_json();
        let schema_name = <T as SchemaType>::schema_name();
        let mut view = MockRequestView::bare(RequestKind::MaterializeStream, prompt);
        view.schema = Some(&schema);
        view.schema_name = schema_name.as_deref();
        self.record(&view);
        let resp = self.pick_response(&view);
        Box::pin(async_stream::try_stream! {
            let s = match resp {
                MockResponse::Text(s) => s,
                MockResponse::Error(e) => Err(e)?,
            };
            // Emit one Partial snapshot, then the validated Complete value.
            let snapshot: Value = serde_json::from_str(&s).map_err(|e| {
                RStructorError::ValidationError(format!(
                    "Failed to parse response as JSON: {e}\nPartial JSON: {s}"
                ))
            })?;
            yield StreamedObject::Partial(snapshot);
            let value: T = parse_and_validate::<T>(&s)?;
            yield StreamedObject::Complete(value);
        })
    }

    #[cfg(feature = "streaming")]
    fn materialize_iter<'a, T>(
        &'a self,
        prompt: &'a str,
    ) -> crate::backend::streaming::ItemStream<'a, T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
        Self: Sync,
    {
        let schema = <T as SchemaType>::schema().to_json();
        let schema_name = <T as SchemaType>::schema_name();
        let mut view = MockRequestView::bare(RequestKind::MaterializeIter, prompt);
        view.schema = Some(&schema);
        view.schema_name = schema_name.as_deref();
        self.record(&view);
        let resp = self.pick_response(&view);
        Box::pin(async_stream::try_stream! {
            let s = match resp {
                MockResponse::Text(s) => s,
                MockResponse::Error(e) => Err(e)?,
            };
            let root: Value = serde_json::from_str(&s).map_err(|e| {
                RStructorError::ValidationError(format!(
                    "Failed to parse response as JSON: {e}\nPartial JSON: {s}"
                ))
            })?;
            // Accept either a bare top-level array or an `{ "items": [...] }` wrapper.
            let items: Vec<Value> = if let Some(arr) = root.as_array() {
                arr.clone()
            } else if let Some(arr) = root.get("items").and_then(Value::as_array) {
                arr.clone()
            } else {
                Err(RStructorError::ValidationError(
                    "MockClient::materialize_iter expects a JSON array or {\"items\": [...]}"
                        .to_string(),
                ))?
            };
            for item in items {
                let value: T = crate::backend::streaming::finalize_item::<T>(item)?;
                yield value;
            }
        })
    }

    fn from_env() -> Result<Self>
    where
        Self: Sized,
    {
        // The mock needs no environment; this exists to satisfy the trait.
        Ok(Self::new())
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let view = MockRequestView::bare(RequestKind::ListModels, "");
        self.record(&view);
        Ok(self.inner.models.lock().unwrap().clone())
    }
}

#[cfg(feature = "tools")]
#[async_trait]
impl crate::backend::tools::ToolRunner for MockClient {
    async fn run_tool_loop(
        &self,
        _system: Option<&str>,
        prompt: &str,
        media: &[MediaFile],
        toolbox: &crate::backend::tools::Toolbox,
        _max_iterations: usize,
    ) -> Result<String> {
        let tool_names = toolbox.tool_names();
        let mut view = MockRequestView::bare(RequestKind::RunToolLoop, prompt);
        view.media = media;
        view.tool_names = &tool_names;
        self.record(&view);

        // Run any scripted tool calls so the tools' `invoke` is exercised offline.
        let script: Vec<(String, Value)> =
            self.inner.tool_script.lock().unwrap().drain(..).collect();
        for (name, args) in script {
            match toolbox.get(&name) {
                Some(tool) => {
                    tool.invoke_json(args).await?;
                }
                None => {
                    return Err(RStructorError::Unsupported(format!(
                        "MockClient tool script referenced unknown tool: {name}"
                    )));
                }
            }
        }

        match self.pick_response(&view) {
            MockResponse::Text(s) => Ok(s),
            MockResponse::Error(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Bring the derive macro (and its `#[llm(...)]` helper attribute) into scope;
    // `super::*` only re-exports the `Instructor` *trait* used by the impl bounds.
    use crate::Instructor;
    use serde::{Deserialize, Serialize};

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    #[llm(validate = "validate_movie")]
    struct Movie {
        title: String,
        year: u16,
    }

    fn validate_movie(m: &Movie) -> Result<()> {
        if m.year < 1888 {
            return Err(RStructorError::ValidationError("year too early".into()));
        }
        Ok(())
    }

    #[tokio::test]
    async fn materialize_returns_scripted_json() {
        let client = MockClient::new().with_response(r#"{"title":"Inception","year":2010}"#);
        let movie: Movie = client.materialize("p").await.unwrap();
        assert_eq!(
            movie,
            Movie {
                title: "Inception".into(),
                year: 2010
            }
        );
    }

    #[tokio::test]
    async fn materialize_runs_validate_and_fails() {
        let client = MockClient::new().with_response(r#"{"title":"X","year":1700}"#);
        let err = client.materialize::<Movie>("p").await.unwrap_err();
        assert!(matches!(err, RStructorError::ValidationError(_)));
    }

    #[tokio::test]
    async fn bad_json_is_validation_error() {
        let client = MockClient::new().with_response("not json");
        let err = client.materialize::<Movie>("p").await.unwrap_err();
        assert!(matches!(err, RStructorError::ValidationError(_)));
    }

    #[tokio::test]
    async fn retries_consume_next_response() {
        let client = MockClient::new()
            .with_response(r#"{"title":"X","year":1700}"#) // fails validation
            .with_response(r#"{"title":"Dune","year":2021}"#) // good
            .with_retries(1);
        let movie: Movie = client.materialize("p").await.unwrap();
        assert_eq!(movie.year, 2021);
    }

    #[tokio::test]
    async fn records_prompt_and_schema() {
        let client = MockClient::new().with_response(r#"{"title":"A","year":2000}"#);
        let _: Movie = client.materialize("the prompt").await.unwrap();
        let req = client.last_request().unwrap();
        assert_eq!(req.kind, RequestKind::Materialize);
        assert_eq!(req.prompt, "the prompt");
        assert_eq!(req.schema_name.as_deref(), Some("Movie"));
        assert!(req.schema.is_some());
        assert_eq!(client.request_count(), 1);
    }

    #[tokio::test]
    async fn responder_closure_branches_on_prompt() {
        let client = MockClient::new().with_responder(|req| {
            if req.prompt.contains("movie") {
                Some(MockResponse::text(r#"{"title":"Matrix","year":1999}"#))
            } else {
                None
            }
        });
        let movie: Movie = client.materialize("a movie please").await.unwrap();
        assert_eq!(movie.title, "Matrix");
    }

    #[tokio::test]
    async fn error_response_returned_verbatim() {
        let err = RStructorError::api_error("OpenAI", crate::ApiErrorKind::AuthenticationFailed);
        let client = MockClient::new().with_error(err);
        let got = client.generate("p").await.unwrap_err();
        assert_eq!(
            got,
            RStructorError::api_error("OpenAI", crate::ApiErrorKind::AuthenticationFailed)
        );
    }

    #[tokio::test]
    async fn clone_shares_state() {
        let client = MockClient::new();
        let clone = client.clone();
        clone.push_response(r#"{"title":"Shared","year":2020}"#);
        let movie: Movie = client.materialize("p").await.unwrap();
        assert_eq!(movie.title, "Shared");
    }

    #[tokio::test]
    async fn default_after_exhaustion() {
        let client = MockClient::new().with_response(r#"{"title":"A","year":2000}"#);
        let _: Movie = client.materialize("p").await.unwrap();
        assert!(client.responses_exhausted());
        let err = client.materialize::<Movie>("p").await.unwrap_err();
        assert!(matches!(err, RStructorError::Unsupported(_)));
    }

    #[tokio::test]
    async fn from_env_needs_no_key() {
        assert!(MockClient::from_env().is_ok());
    }

    #[tokio::test]
    async fn metadata_carries_usage() {
        let client = MockClient::new()
            .with_response(r#"{"title":"A","year":2000}"#)
            .with_usage(TokenUsage::new("mock-model", 10, 20));
        let result = client
            .materialize_with_metadata::<Movie>("p")
            .await
            .unwrap();
        assert_eq!(result.usage.unwrap().total_tokens(), 30);
    }
}

//! Versioned, privacy-aware fixtures for recording and replaying LLM calls.
//!
//! Wrap a live or mock client in [`FixtureRecorder`], make normal non-streaming
//! calls through the wrapper, and persist the resulting [`Fixture`]. Tests can
//! load that file and replay it through [`ReplayClient`] without a network or
//! API key. Replay is ordered and strict: the operation, prompt, schema, and
//! media metadata must match before a recorded response is consumed.

use std::collections::{BTreeSet, VecDeque};
use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::ResponseMetadata;
use crate::backend::ModelInfo;
use crate::backend::client::{LLMClient, MediaFile};
use crate::backend::usage::{
    AttemptOutcome, ExtractionReport, GenerateResult, MaterializeFailure, MaterializeReport,
    MaterializeResult, RunUsage, TokenUsage,
};
use crate::error::{ApiErrorKind, RStructorError, Result, StreamErrorKind};
use crate::model::Instructor;
use crate::schema::SchemaType;

/// Current on-disk fixture schema version.
pub const FIXTURE_SCHEMA_VERSION: u32 = 1;

type TextSanitizer = dyn Fn(&str) -> String + Send + Sync + 'static;

/// Mandatory privacy boundary used before requests and responses enter a fixture.
///
/// The callback runs on every retained string. Common credential-shaped JSON
/// fields are also replaced structurally, and inline media bytes are never
/// stored. Build the same sanitizer in replay tests when it changes request
/// text that must be matched.
#[derive(Clone)]
pub struct FixtureSanitizer {
    sanitize_text: Arc<TextSanitizer>,
    redacted_json_keys: BTreeSet<String>,
}

impl FixtureSanitizer {
    /// Create a sanitizer with a required string callback and safe JSON-key defaults.
    pub fn new<F>(sanitize_text: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        let redacted_json_keys = [
            "api_key",
            "authorization",
            "cookie",
            "password",
            "secret",
            "set_cookie",
            "token",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();

        Self {
            sanitize_text: Arc::new(sanitize_text),
            redacted_json_keys,
        }
    }

    /// Add case-insensitive JSON keys whose values must be replaced entirely.
    #[must_use]
    pub fn redact_json_keys<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.redacted_json_keys
            .extend(keys.into_iter().map(|key| normalize_json_key(key.as_ref())));
        self
    }

    fn text(&self, text: &str) -> String {
        (self.sanitize_text)(text)
    }

    fn json(&self, value: &Value) -> Value {
        match value {
            Value::Object(object) => Value::Object(
                object
                    .iter()
                    .map(|(key, value)| {
                        let value = if self.redacted_json_keys.contains(&normalize_json_key(key)) {
                            Value::String("[REDACTED]".to_string())
                        } else {
                            self.json(value)
                        };
                        (key.clone(), value)
                    })
                    .collect(),
            ),
            Value::Array(items) => Value::Array(items.iter().map(|item| self.json(item)).collect()),
            Value::String(text) => Value::String(self.text(text)),
            primitive => primitive.clone(),
        }
    }

    fn request(&self, mut request: StoredRequest) -> StoredRequest {
        request.prompt = self.text(&request.prompt);
        request.schema_name = request.schema_name.map(|name| self.text(&name));
        request.schema = request.schema.map(|schema| self.json(&schema));
        for media in &mut request.media {
            media.uri = self.text(&media.uri);
            media.mime_type = self.text(&media.mime_type);
        }
        request
    }

    fn report(&self, mut report: ExtractionReport) -> ExtractionReport {
        sanitize_usage(self, report.final_usage.as_mut());
        if let Some(cumulative) = report.cumulative_usage.as_mut() {
            sanitize_run_usage(self, cumulative);
        }
        for attempt in &mut report.attempts {
            sanitize_usage(self, attempt.usage.as_mut());
            if let AttemptOutcome::Failed { message, .. } = &mut attempt.outcome {
                *message = self.text(message);
            }
            if let Some(response) = attempt.response.as_mut() {
                sanitize_response_metadata(self, response);
            }
        }
        report
    }
}

impl fmt::Debug for FixtureSanitizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FixtureSanitizer")
            .field("sanitize_text", &"<caller-provided>")
            .field("redacted_json_keys", &self.redacted_json_keys)
            .finish()
    }
}

fn normalize_json_key(key: &str) -> String {
    key.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn sanitize_usage(sanitizer: &FixtureSanitizer, usage: Option<&mut TokenUsage>) {
    if let Some(usage) = usage {
        usage.model = sanitizer.text(&usage.model);
    }
}

fn sanitize_run_usage(sanitizer: &FixtureSanitizer, usage: &mut RunUsage) {
    let old = std::mem::take(&mut usage.by_model);
    usage.by_model = old
        .into_iter()
        .map(|(model, mut token_usage)| {
            token_usage.model = sanitizer.text(&token_usage.model);
            (sanitizer.text(&model), token_usage)
        })
        .collect();
}

fn sanitize_response_metadata(sanitizer: &FixtureSanitizer, response: &mut ResponseMetadata) {
    for request_id in response.request_ids.values_mut() {
        *request_id = sanitizer.text(request_id);
    }
    if let Some(body) = response.sanitized_body.as_mut() {
        body.text = sanitize_body(sanitizer, &body.text);
    }
}

fn sanitize_body(sanitizer: &FixtureSanitizer, body: &str) -> String {
    serde_json::from_str(body).map_or_else(
        |_| sanitizer.text(body),
        |value| sanitizer.json(&value).to_string(),
    )
}

/// File-format and replay-completion errors for fixtures.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FixtureError {
    /// The fixture could not be read or written.
    #[error("fixture I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The fixture JSON was invalid.
    #[error("invalid fixture JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The fixture was written with an unsupported schema version.
    #[error("unsupported fixture schema version {found}; this build supports {supported}")]
    UnsupportedVersion {
        /// Version found in the fixture.
        found: u32,
        /// Version supported by this library build.
        supported: u32,
    },
    /// A replay finished while recorded interactions remained unused.
    #[error("fixture replay left {remaining} interaction(s) unused")]
    ReplayIncomplete {
        /// Number of unused interactions.
        remaining: usize,
    },
}

/// A versioned collection of sanitized, ordered request/response interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Fixture {
    schema_version: u32,
    interactions: Vec<StoredInteraction>,
}

impl Default for Fixture {
    fn default() -> Self {
        Self::new()
    }
}

impl Fixture {
    /// Create an empty fixture using the current schema version.
    #[must_use]
    pub fn new() -> Self {
        Self {
            schema_version: FIXTURE_SCHEMA_VERSION,
            interactions: Vec::new(),
        }
    }

    /// Return the on-disk schema version.
    #[must_use]
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Number of recorded interactions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.interactions.len()
    }

    /// Whether no interactions have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.interactions.is_empty()
    }

    /// Parse and version-check a fixture from JSON.
    pub fn from_json(json: &str) -> std::result::Result<Self, FixtureError> {
        let fixture: Self = serde_json::from_str(json)?;
        fixture.validate_version()?;
        Ok(fixture)
    }

    /// Serialize the fixture as stable, pretty JSON with a trailing newline.
    pub fn to_json(&self) -> std::result::Result<String, FixtureError> {
        self.validate_version()?;
        let mut json = serde_json::to_string_pretty(self)?;
        json.push('\n');
        Ok(json)
    }

    /// Load and version-check a fixture file.
    pub fn load(path: impl AsRef<Path>) -> std::result::Result<Self, FixtureError> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Write a fixture as stable, pretty JSON.
    pub fn save(&self, path: impl AsRef<Path>) -> std::result::Result<(), FixtureError> {
        std::fs::write(path, self.to_json()?)?;
        Ok(())
    }

    /// Create a strict offline replay client using identity string sanitization.
    #[must_use]
    pub fn replay(&self) -> ReplayClient {
        self.replay_with_sanitizer(FixtureSanitizer::new(str::to_owned))
    }

    /// Create a replay client that sanitizes incoming requests before matching.
    ///
    /// Use the same sanitizer construction as the recording run when request
    /// prompts or schema string values were transformed.
    #[must_use]
    pub fn replay_with_sanitizer(&self, sanitizer: FixtureSanitizer) -> ReplayClient {
        ReplayClient::new(self.clone(), sanitizer)
    }

    fn validate_version(&self) -> std::result::Result<(), FixtureError> {
        if self.schema_version == FIXTURE_SCHEMA_VERSION {
            Ok(())
        } else {
            Err(FixtureError::UnsupportedVersion {
                found: self.schema_version,
                supported: FIXTURE_SCHEMA_VERSION,
            })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StoredOperation {
    Materialize,
    MaterializeWithMedia,
    MaterializeWithMetadata,
    MaterializeWithAttempts,
    MaterializeWithMediaAndAttempts,
    Generate,
    GenerateWithMedia,
    GenerateWithMetadata,
    ListModels,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StoredRequest {
    operation: StoredOperation,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    media: Vec<StoredMedia>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StoredMedia {
    uri: String,
    mime_type: String,
    inline_data: bool,
}

impl StoredMedia {
    fn from_media(media: &MediaFile) -> Self {
        Self {
            uri: media.uri.clone(),
            mime_type: media.mime_type.clone(),
            inline_data: media.data.is_some(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredInteraction {
    request: StoredRequest,
    response: StoredResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StoredResponse {
    Success {
        body: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<TokenUsage>,
        #[serde(skip_serializing_if = "Option::is_none")]
        report: Option<ExtractionReport>,
    },
    Error {
        error: StoredError,
        #[serde(skip_serializing_if = "Option::is_none")]
        report: Option<ExtractionReport>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum StoredError {
    Api {
        provider: String,
        api_kind: ApiErrorKind,
        response: Option<Box<ResponseMetadata>>,
    },
    Validation {
        message: String,
    },
    Schema {
        message: String,
    },
    SchemaCompatibility {
        provider: String,
        context: String,
        path: String,
        message: String,
    },
    Serialization {
        message: String,
    },
    OutputDecode {
        path: String,
        message: String,
    },
    ToolArgumentDecode {
        path: String,
        message: String,
    },
    Streaming {
        stream_kind: StreamErrorKind,
        message: String,
    },
    Timeout,
    Unsupported {
        message: String,
    },
}

impl StoredError {
    fn from_error(error: &RStructorError, sanitizer: &FixtureSanitizer) -> Self {
        match error {
            RStructorError::ApiError {
                provider,
                kind,
                response,
            } => {
                let mut kind = kind.clone();
                sanitize_api_error_kind(sanitizer, &mut kind);
                let mut response = response.clone();
                if let Some(response) = response.as_mut() {
                    sanitize_response_metadata(sanitizer, response);
                }
                Self::Api {
                    provider: sanitizer.text(provider),
                    api_kind: kind,
                    response,
                }
            }
            RStructorError::ValidationError(message) => Self::Validation {
                message: sanitizer.text(message),
            },
            RStructorError::SchemaError(message) => Self::Schema {
                message: sanitizer.text(message),
            },
            RStructorError::SchemaCompatibilityError {
                provider,
                context,
                path,
                message,
            } => Self::SchemaCompatibility {
                provider: sanitizer.text(provider),
                context: sanitizer.text(context),
                path: sanitizer.text(path),
                message: sanitizer.text(message),
            },
            RStructorError::SerializationError(message) => Self::Serialization {
                message: sanitizer.text(message),
            },
            RStructorError::OutputDecodeError { path, message } => Self::OutputDecode {
                path: sanitizer.text(path),
                message: sanitizer.text(message),
            },
            RStructorError::ToolArgumentDecodeError { path, message } => Self::ToolArgumentDecode {
                path: sanitizer.text(path),
                message: sanitizer.text(message),
            },
            RStructorError::StreamingError { kind, message } => Self::Streaming {
                stream_kind: *kind,
                message: sanitizer.text(message),
            },
            RStructorError::Timeout => Self::Timeout,
            RStructorError::Unsupported(message) => Self::Unsupported {
                message: sanitizer.text(message),
            },
            #[cfg(feature = "_client")]
            RStructorError::HttpError(error) => Self::Unsupported {
                message: sanitizer.text(&format!("recorded HTTP transport error: {error}")),
            },
            RStructorError::JsonError(error) => Self::Serialization {
                message: sanitizer.text(&format!("recorded JSON error: {error}")),
            },
        }
    }

    fn into_error(self) -> RStructorError {
        match self {
            Self::Api {
                provider,
                api_kind,
                response,
            } => RStructorError::ApiError {
                provider,
                kind: api_kind,
                response,
            },
            Self::Validation { message } => RStructorError::ValidationError(message),
            Self::Schema { message } => RStructorError::SchemaError(message),
            Self::SchemaCompatibility {
                provider,
                context,
                path,
                message,
            } => RStructorError::SchemaCompatibilityError {
                provider: provider.into(),
                context: context.into(),
                path: path.into(),
                message: message.into(),
            },
            Self::Serialization { message } => RStructorError::SerializationError(message),
            Self::OutputDecode { path, message } => {
                RStructorError::OutputDecodeError { path, message }
            }
            Self::ToolArgumentDecode { path, message } => {
                RStructorError::ToolArgumentDecodeError { path, message }
            }
            Self::Streaming {
                stream_kind,
                message,
            } => RStructorError::StreamingError {
                kind: stream_kind,
                message: message.into(),
            },
            Self::Timeout => RStructorError::Timeout,
            Self::Unsupported { message } => RStructorError::Unsupported(message),
        }
    }
}

fn sanitize_api_error_kind(sanitizer: &FixtureSanitizer, kind: &mut ApiErrorKind) {
    match kind {
        ApiErrorKind::InvalidModel { model, suggestion } => {
            *model = sanitizer.text(model);
            *suggestion = suggestion.take().map(|value| sanitizer.text(&value));
        }
        ApiErrorKind::BadRequest { details } | ApiErrorKind::UnexpectedResponse { details } => {
            *details = sanitizer.text(details);
        }
        ApiErrorKind::Other { message, .. } => {
            *message = sanitizer.text(message);
        }
        _ => {}
    }
}

fn structured_request<T>(
    operation: StoredOperation,
    prompt: &str,
    media: &[MediaFile],
) -> Result<StoredRequest>
where
    T: Instructor,
{
    Ok(StoredRequest {
        operation,
        prompt: prompt.to_string(),
        schema_name: <T as SchemaType>::schema_name(),
        schema: Some(<T as SchemaType>::try_schema()?.to_json()),
        media: media.iter().map(StoredMedia::from_media).collect(),
    })
}

fn text_request(operation: StoredOperation, prompt: &str, media: &[MediaFile]) -> StoredRequest {
    StoredRequest {
        operation,
        prompt: prompt.to_string(),
        schema_name: None,
        schema: None,
        media: media.iter().map(StoredMedia::from_media).collect(),
    }
}

fn sanitized_structured_body<T>(value: &T, sanitizer: &FixtureSanitizer) -> Result<String>
where
    T: Serialize,
{
    let value = serde_json::to_value(value)
        .map_err(|error| RStructorError::SerializationError(error.to_string()))?;
    serde_json::to_string(&sanitizer.json(&value))
        .map_err(|error| RStructorError::SerializationError(error.to_string()))
}

fn success_report<T>(report: &MaterializeReport<T>) -> ExtractionReport {
    ExtractionReport {
        final_usage: report.final_usage.clone(),
        cumulative_usage: report.cumulative_usage.clone(),
        attempts: report.attempts.clone(),
        attempts_complete: report.attempts_complete,
    }
}

fn failure_report(failure: &MaterializeFailure) -> ExtractionReport {
    ExtractionReport {
        final_usage: failure
            .attempts
            .last()
            .and_then(|attempt| attempt.usage.clone()),
        cumulative_usage: failure.cumulative_usage.clone(),
        attempts: failure.attempts.clone(),
        attempts_complete: failure.attempts_complete,
    }
}

/// An [`LLMClient`] wrapper that records sanitized non-streaming interactions.
///
/// Recording is in memory until [`fixture`](Self::fixture) or [`save`](Self::save)
/// is called. The sanitizer is applied synchronously before an interaction is
/// retained; the recorder never keeps the original prompt or output.
pub struct FixtureRecorder<C> {
    inner: C,
    sanitizer: FixtureSanitizer,
    fixture: Arc<Mutex<Fixture>>,
}

impl<C> FixtureRecorder<C> {
    /// Wrap a client with a mandatory fixture sanitizer.
    #[must_use]
    pub fn new(inner: C, sanitizer: FixtureSanitizer) -> Self {
        Self {
            inner,
            sanitizer,
            fixture: Arc::new(Mutex::new(Fixture::new())),
        }
    }

    /// Borrow the wrapped client.
    #[must_use]
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// Snapshot the interactions recorded so far.
    #[must_use]
    pub fn fixture(&self) -> Fixture {
        self.fixture.lock().unwrap().clone()
    }

    /// Number of interactions recorded so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fixture.lock().unwrap().len()
    }

    /// Whether no interactions have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fixture.lock().unwrap().is_empty()
    }

    /// Persist a snapshot of the interactions recorded so far.
    pub fn save(&self, path: impl AsRef<Path>) -> std::result::Result<(), FixtureError> {
        self.fixture().save(path)
    }

    fn record(&self, request: StoredRequest, response: StoredResponse) {
        self.fixture
            .lock()
            .unwrap()
            .interactions
            .push(StoredInteraction {
                request: self.sanitizer.request(request),
                response,
            });
    }

    fn success(
        &self,
        request: StoredRequest,
        body: String,
        mut usage: Option<TokenUsage>,
        report: Option<ExtractionReport>,
    ) {
        sanitize_usage(&self.sanitizer, usage.as_mut());
        let report = report.map(|report| self.sanitizer.report(report));
        self.record(
            request,
            StoredResponse::Success {
                body,
                usage,
                report,
            },
        );
    }

    fn error(
        &self,
        request: StoredRequest,
        error: &RStructorError,
        report: Option<ExtractionReport>,
    ) {
        let report = report.map(|report| self.sanitizer.report(report));
        self.record(
            request,
            StoredResponse::Error {
                error: StoredError::from_error(error, &self.sanitizer),
                report,
            },
        );
    }
}

impl<C> fmt::Debug for FixtureRecorder<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FixtureRecorder")
            .field("inner", &std::any::type_name::<C>())
            .field("recorded_interactions", &self.len())
            .field("sanitizer", &self.sanitizer)
            .finish()
    }
}

#[async_trait]
impl<C> LLMClient for FixtureRecorder<C>
where
    C: LLMClient + Sync,
{
    async fn materialize<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + serde::de::DeserializeOwned + Send + 'static,
    {
        let request = structured_request::<T>(StoredOperation::Materialize, prompt, &[])?;
        match self.inner.materialize(prompt).await {
            Ok(data) => {
                let body = sanitized_structured_body(&data, &self.sanitizer)?;
                self.success(request, body, None, None);
                Ok(data)
            }
            Err(error) => {
                self.error(request, &error, None);
                Err(error)
            }
        }
    }

    async fn materialize_with_media<T>(&self, prompt: &str, media: &[MediaFile]) -> Result<T>
    where
        T: Instructor + serde::de::DeserializeOwned + Send + 'static,
    {
        let request =
            structured_request::<T>(StoredOperation::MaterializeWithMedia, prompt, media)?;
        match self.inner.materialize_with_media(prompt, media).await {
            Ok(data) => {
                let body = sanitized_structured_body(&data, &self.sanitizer)?;
                self.success(request, body, None, None);
                Ok(data)
            }
            Err(error) => {
                self.error(request, &error, None);
                Err(error)
            }
        }
    }

    async fn materialize_with_metadata<T>(&self, prompt: &str) -> Result<MaterializeResult<T>>
    where
        T: Instructor + serde::de::DeserializeOwned + Send + 'static,
    {
        let request =
            structured_request::<T>(StoredOperation::MaterializeWithMetadata, prompt, &[])?;
        match self.inner.materialize_with_metadata(prompt).await {
            Ok(result) => {
                let body = sanitized_structured_body(&result.data, &self.sanitizer)?;
                self.success(request, body, result.usage.clone(), None);
                Ok(result)
            }
            Err(error) => {
                self.error(request, &error, None);
                Err(error)
            }
        }
    }

    async fn materialize_with_attempts<T>(
        &self,
        prompt: &str,
    ) -> std::result::Result<MaterializeReport<T>, MaterializeFailure>
    where
        T: Instructor + serde::de::DeserializeOwned + Send + 'static,
    {
        let request =
            structured_request::<T>(StoredOperation::MaterializeWithAttempts, prompt, &[])
                .map_err(MaterializeFailure::from_error)?;
        match self.inner.materialize_with_attempts(prompt).await {
            Ok(report) => {
                let body = sanitized_structured_body(&report.data, &self.sanitizer)
                    .map_err(MaterializeFailure::from_error)?;
                self.success(
                    request,
                    body,
                    report.final_usage.clone(),
                    Some(success_report(&report)),
                );
                Ok(report)
            }
            Err(failure) => {
                self.error(request, failure.error(), Some(failure_report(&failure)));
                Err(failure)
            }
        }
    }

    async fn materialize_with_media_and_attempts<T>(
        &self,
        prompt: &str,
        media: &[MediaFile],
    ) -> std::result::Result<MaterializeReport<T>, MaterializeFailure>
    where
        T: Instructor + serde::de::DeserializeOwned + Send + 'static,
    {
        let request = structured_request::<T>(
            StoredOperation::MaterializeWithMediaAndAttempts,
            prompt,
            media,
        )
        .map_err(MaterializeFailure::from_error)?;
        match self
            .inner
            .materialize_with_media_and_attempts(prompt, media)
            .await
        {
            Ok(report) => {
                let body = sanitized_structured_body(&report.data, &self.sanitizer)
                    .map_err(MaterializeFailure::from_error)?;
                self.success(
                    request,
                    body,
                    report.final_usage.clone(),
                    Some(success_report(&report)),
                );
                Ok(report)
            }
            Err(failure) => {
                self.error(request, failure.error(), Some(failure_report(&failure)));
                Err(failure)
            }
        }
    }

    async fn generate(&self, prompt: &str) -> Result<String> {
        let request = text_request(StoredOperation::Generate, prompt, &[]);
        match self.inner.generate(prompt).await {
            Ok(text) => {
                self.success(request, self.sanitizer.text(&text), None, None);
                Ok(text)
            }
            Err(error) => {
                self.error(request, &error, None);
                Err(error)
            }
        }
    }

    async fn generate_with_media(&self, prompt: &str, media: &[MediaFile]) -> Result<String> {
        let request = text_request(StoredOperation::GenerateWithMedia, prompt, media);
        match self.inner.generate_with_media(prompt, media).await {
            Ok(text) => {
                self.success(request, self.sanitizer.text(&text), None, None);
                Ok(text)
            }
            Err(error) => {
                self.error(request, &error, None);
                Err(error)
            }
        }
    }

    async fn generate_with_metadata(&self, prompt: &str) -> Result<GenerateResult> {
        let request = text_request(StoredOperation::GenerateWithMetadata, prompt, &[]);
        match self.inner.generate_with_metadata(prompt).await {
            Ok(result) => {
                self.success(
                    request,
                    self.sanitizer.text(&result.text),
                    result.usage.clone(),
                    None,
                );
                Ok(result)
            }
            Err(error) => {
                self.error(request, &error, None);
                Err(error)
            }
        }
    }

    fn from_env() -> Result<Self>
    where
        Self: Sized,
    {
        Err(RStructorError::Unsupported(
            "FixtureRecorder requires an explicit inner client and sanitizer".to_string(),
        ))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let request = text_request(StoredOperation::ListModels, "", &[]);
        match self.inner.list_models().await {
            Ok(models) => {
                let body = sanitized_structured_body(&models, &self.sanitizer)?;
                self.success(request, body, None, None);
                Ok(models)
            }
            Err(error) => {
                self.error(request, &error, None);
                Err(error)
            }
        }
    }
}

struct ReplayState {
    remaining: Mutex<VecDeque<StoredInteraction>>,
    total: usize,
}

/// Strict, ordered, offline replay of a [`Fixture`].
///
/// Clones share replay position. Call [`assert_finished`](Self::assert_finished)
/// at the end of a test to catch interactions that were silently skipped.
#[derive(Clone)]
pub struct ReplayClient {
    state: Arc<ReplayState>,
    sanitizer: FixtureSanitizer,
}

impl ReplayClient {
    fn new(fixture: Fixture, sanitizer: FixtureSanitizer) -> Self {
        let total = fixture.interactions.len();
        Self {
            state: Arc::new(ReplayState {
                remaining: Mutex::new(fixture.interactions.into()),
                total,
            }),
            sanitizer,
        }
    }

    /// Number of recorded interactions not yet consumed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.state.remaining.lock().unwrap().len()
    }

    /// Fail if replay did not consume every recorded interaction.
    pub fn assert_finished(&self) -> std::result::Result<(), FixtureError> {
        let remaining = self.remaining();
        if remaining == 0 {
            Ok(())
        } else {
            Err(FixtureError::ReplayIncomplete { remaining })
        }
    }

    fn take(&self, request: StoredRequest) -> Result<StoredResponse> {
        let request = self.sanitizer.request(request);
        let mut remaining = self.state.remaining.lock().unwrap();
        let interaction_number = self.state.total - remaining.len() + 1;
        let expected = remaining.front().ok_or_else(|| {
            RStructorError::Unsupported(format!(
                "fixture replay received unexpected interaction {interaction_number}: fixture exhausted"
            ))
        })?;
        if let Some(field) = request_mismatch(&expected.request, &request) {
            return Err(RStructorError::Unsupported(format!(
                "fixture request mismatch at interaction {interaction_number}: {field} differs"
            )));
        }
        Ok(remaining.pop_front().unwrap().response)
    }

    fn structured<T>(&self, request: StoredRequest) -> Result<T>
    where
        T: Instructor + serde::de::DeserializeOwned,
    {
        match self.take(request)? {
            StoredResponse::Success { body, .. } => parse_and_validate(&body),
            StoredResponse::Error { error, .. } => Err(error.into_error()),
        }
    }

    fn structured_with_attempts<T>(
        &self,
        request: StoredRequest,
    ) -> std::result::Result<MaterializeReport<T>, MaterializeFailure>
    where
        T: Instructor + serde::de::DeserializeOwned,
    {
        let response = self.take(request).map_err(MaterializeFailure::from_error)?;
        match response {
            StoredResponse::Success {
                body,
                usage,
                report,
            } => {
                let data = parse_and_validate(&body).map_err(MaterializeFailure::from_error)?;
                Ok(match report {
                    Some(report) => MaterializeReport::from_fixture_parts(
                        data,
                        report.final_usage,
                        report.cumulative_usage,
                        report.attempts,
                        report.attempts_complete,
                    ),
                    None => MaterializeReport::from_result(MaterializeResult::new(data, usage)),
                })
            }
            StoredResponse::Error { error, report } => {
                let error = error.into_error();
                Err(match report {
                    Some(report) => MaterializeFailure::from_fixture_parts(
                        error,
                        report.cumulative_usage,
                        report.attempts,
                        report.attempts_complete,
                    ),
                    None => MaterializeFailure::from_error(error),
                })
            }
        }
    }
}

impl fmt::Debug for ReplayClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReplayClient")
            .field("remaining", &self.remaining())
            .field("total", &self.state.total)
            .finish()
    }
}

fn request_mismatch(expected: &StoredRequest, actual: &StoredRequest) -> Option<&'static str> {
    if expected.operation != actual.operation {
        Some("operation")
    } else if expected.prompt != actual.prompt {
        Some("prompt")
    } else if expected.schema_name != actual.schema_name {
        Some("schema name")
    } else if expected.schema != actual.schema {
        Some("schema")
    } else if expected.media != actual.media {
        Some("media metadata")
    } else {
        None
    }
}

fn parse_and_validate<T>(body: &str) -> Result<T>
where
    T: Instructor + serde::de::DeserializeOwned,
{
    let value: T = crate::decode::output_from_str(body)?;
    value.validate()?;
    Ok(value)
}

#[async_trait]
impl LLMClient for ReplayClient {
    async fn materialize<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + serde::de::DeserializeOwned + Send + 'static,
    {
        self.structured(structured_request::<T>(
            StoredOperation::Materialize,
            prompt,
            &[],
        )?)
    }

    async fn materialize_with_media<T>(&self, prompt: &str, media: &[MediaFile]) -> Result<T>
    where
        T: Instructor + serde::de::DeserializeOwned + Send + 'static,
    {
        self.structured(structured_request::<T>(
            StoredOperation::MaterializeWithMedia,
            prompt,
            media,
        )?)
    }

    async fn materialize_with_metadata<T>(&self, prompt: &str) -> Result<MaterializeResult<T>>
    where
        T: Instructor + serde::de::DeserializeOwned + Send + 'static,
    {
        let request =
            structured_request::<T>(StoredOperation::MaterializeWithMetadata, prompt, &[])?;
        match self.take(request)? {
            StoredResponse::Success { body, usage, .. } => {
                Ok(MaterializeResult::new(parse_and_validate(&body)?, usage))
            }
            StoredResponse::Error { error, .. } => Err(error.into_error()),
        }
    }

    async fn materialize_with_attempts<T>(
        &self,
        prompt: &str,
    ) -> std::result::Result<MaterializeReport<T>, MaterializeFailure>
    where
        T: Instructor + serde::de::DeserializeOwned + Send + 'static,
    {
        let request =
            structured_request::<T>(StoredOperation::MaterializeWithAttempts, prompt, &[])
                .map_err(MaterializeFailure::from_error)?;
        self.structured_with_attempts(request)
    }

    async fn materialize_with_media_and_attempts<T>(
        &self,
        prompt: &str,
        media: &[MediaFile],
    ) -> std::result::Result<MaterializeReport<T>, MaterializeFailure>
    where
        T: Instructor + serde::de::DeserializeOwned + Send + 'static,
    {
        let request = structured_request::<T>(
            StoredOperation::MaterializeWithMediaAndAttempts,
            prompt,
            media,
        )
        .map_err(MaterializeFailure::from_error)?;
        self.structured_with_attempts(request)
    }

    async fn generate(&self, prompt: &str) -> Result<String> {
        match self.take(text_request(StoredOperation::Generate, prompt, &[]))? {
            StoredResponse::Success { body, .. } => Ok(body),
            StoredResponse::Error { error, .. } => Err(error.into_error()),
        }
    }

    async fn generate_with_media(&self, prompt: &str, media: &[MediaFile]) -> Result<String> {
        match self.take(text_request(
            StoredOperation::GenerateWithMedia,
            prompt,
            media,
        ))? {
            StoredResponse::Success { body, .. } => Ok(body),
            StoredResponse::Error { error, .. } => Err(error.into_error()),
        }
    }

    async fn generate_with_metadata(&self, prompt: &str) -> Result<GenerateResult> {
        match self.take(text_request(
            StoredOperation::GenerateWithMetadata,
            prompt,
            &[],
        ))? {
            StoredResponse::Success { body, usage, .. } => Ok(GenerateResult::new(body, usage)),
            StoredResponse::Error { error, .. } => Err(error.into_error()),
        }
    }

    fn from_env() -> Result<Self>
    where
        Self: Sized,
    {
        Err(RStructorError::Unsupported(
            "ReplayClient must be created from a Fixture".to_string(),
        ))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        match self.take(text_request(StoredOperation::ListModels, "", &[]))? {
            StoredResponse::Success { body, .. } => serde_json::from_str(&body)
                .map_err(|error| RStructorError::SerializationError(error.to_string())),
            StoredResponse::Error { error, .. } => Err(error.into_error()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Instructor, LLMClient, MockClient};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Instructor, Serialize, Deserialize)]
    struct Position {
        account_id: String,
        symbol: String,
        quantity: i64,
    }

    fn sanitizer() -> FixtureSanitizer {
        FixtureSanitizer::new(|text| text.replace("HF-ALPHA-001", "[ACCOUNT]"))
    }

    #[tokio::test]
    async fn records_sanitized_report_and_replays_strictly() {
        let usage = TokenUsage::new("gpt-5.6-2026-07-15", 82, 19);
        let inner = MockClient::new()
            .with_response_and_usage(
                r#"{"account_id":"HF-ALPHA-001","symbol":"AAPL","quantity":125000}"#,
                usage.clone(),
            )
            .with_retries(0);
        let recorder = FixtureRecorder::new(inner, sanitizer());

        let extraction = recorder
            .extract_with_report::<Position>("HF-ALPHA-001 owns 125,000 AAPL shares")
            .await
            .unwrap();
        assert_eq!(extraction.report.final_usage, Some(usage));

        let fixture = recorder.fixture();
        let json = fixture.to_json().unwrap();
        assert!(!json.contains("HF-ALPHA-001"));
        assert!(json.contains("[ACCOUNT]"));

        let replay = fixture.replay_with_sanitizer(sanitizer());
        let replayed = replay
            .extract_with_report::<Position>("HF-ALPHA-001 owns 125,000 AAPL shares")
            .await
            .unwrap();
        assert_eq!(replayed.data.account_id, "[ACCOUNT]");
        assert_eq!(replayed.report, extraction.report);
        replay.assert_finished().unwrap();
    }

    #[tokio::test]
    async fn request_mismatch_does_not_consume_or_echo_prompts() {
        let inner = MockClient::new().with_response("approved");
        let recorder = FixtureRecorder::new(inner, sanitizer());
        recorder.generate("approve HF-ALPHA-001").await.unwrap();
        let replay = recorder.fixture().replay_with_sanitizer(sanitizer());

        let error = replay.generate("reject HF-ALPHA-001").await.unwrap_err();
        let message = error.to_string();
        assert!(message.contains("prompt differs"));
        assert!(!message.contains("approve"));
        assert!(!message.contains("reject"));
        assert_eq!(replay.remaining(), 1);
    }

    #[test]
    fn common_sensitive_json_keys_and_inline_media_are_not_retained() {
        let sanitizer = FixtureSanitizer::new(str::to_owned);
        let value = serde_json::json!({
            "authorization": "Bearer live-secret",
            "nested": {"api-key": "live-key", "safe": "kept"}
        });
        let sanitized = sanitizer.json(&value);
        assert_eq!(sanitized["authorization"], "[REDACTED]");
        assert_eq!(sanitized["nested"]["api-key"], "[REDACTED]");
        assert_eq!(sanitized["nested"]["safe"], "kept");

        let media = MediaFile {
            uri: String::new(),
            mime_type: "image/png".to_string(),
            data: Some("base64-secret".to_string()),
        };
        let stored = StoredMedia::from_media(&media);
        let json = serde_json::to_string(&stored).unwrap();
        assert!(stored.inline_data);
        assert!(!json.contains("base64-secret"));
    }

    #[test]
    fn rejects_unknown_schema_version_and_malformed_json() {
        let unsupported = r#"{"schema_version":99,"interactions":[]}"#;
        assert!(matches!(
            Fixture::from_json(unsupported),
            Err(FixtureError::UnsupportedVersion {
                found: 99,
                supported: FIXTURE_SCHEMA_VERSION
            })
        ));
        assert!(matches!(
            Fixture::from_json("{"),
            Err(FixtureError::Json(_))
        ));
    }

    #[tokio::test]
    async fn typed_api_error_round_trips_with_sanitized_diagnostics() {
        let mut metadata = ResponseMetadata::new(429);
        metadata.request_ids.insert(
            "x-request-id".to_string(),
            "HF-ALPHA-001-request".to_string(),
        );
        let error = RStructorError::api_error_with_response(
            "OpenAI",
            ApiErrorKind::RateLimited { retry_after: None },
            metadata,
        );
        let inner = MockClient::new().with_error(error);
        let recorder = FixtureRecorder::new(inner, sanitizer());
        let original = recorder.generate("HF-ALPHA-001").await.unwrap_err();
        assert_eq!(original.status_code(), Some(429));

        let replay = recorder.fixture().replay_with_sanitizer(sanitizer());
        let replayed = replay.generate("HF-ALPHA-001").await.unwrap_err();
        assert_eq!(replayed.status_code(), Some(429));
        assert_eq!(replayed.request_id(), Some("[ACCOUNT]-request"));
        replay.assert_finished().unwrap();
    }

    #[test]
    fn incomplete_replay_reports_remaining_interactions() {
        let fixture = Fixture {
            schema_version: FIXTURE_SCHEMA_VERSION,
            interactions: vec![StoredInteraction {
                request: text_request(StoredOperation::Generate, "hello", &[]),
                response: StoredResponse::Success {
                    body: "world".to_string(),
                    usage: None,
                    report: None,
                },
            }],
        };
        assert!(matches!(
            fixture.replay().assert_finished(),
            Err(FixtureError::ReplayIncomplete { remaining: 1 })
        ));
    }
}

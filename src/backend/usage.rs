use std::collections::BTreeMap;
use std::fmt;

use crate::error::RStructorError;

/// Token usage information from an LLM API call.
///
/// This struct contains the token counts returned by LLM providers,
/// which can be used for monitoring usage and debugging.
///
/// # Example
///
/// ```no_run
/// use rstructor::{LLMClient, OpenAIClient};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = OpenAIClient::from_env()?;
/// let result = client.generate_with_metadata("Describe Inception").await?;
///
/// if let Some(usage) = &result.usage {
///     println!("Model: {}", usage.model);
///     println!("Input tokens: {}", usage.input_tokens);
///     println!("Cached input tokens: {}", usage.cached_input_tokens);
///     println!("Output tokens: {}", usage.output_tokens);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenUsage {
    /// The model used for this request
    pub model: String,
    /// Number of tokens in the input/prompt
    pub input_tokens: u64,
    /// Input tokens served from a provider prompt cache.
    ///
    /// This is a subset of `input_tokens`, not an additional token count.
    pub cached_input_tokens: u64,
    /// Input tokens written to a provider prompt cache.
    ///
    /// This is a subset of `input_tokens`, not an additional token count.
    /// Providers that perform implicit cache writes may not report this value.
    pub cache_write_input_tokens: u64,
    /// Number of tokens in the output/completion
    pub output_tokens: u64,
}

impl TokenUsage {
    /// Create a new TokenUsage instance
    pub fn new(model: impl Into<String>, input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            model: model.into(),
            input_tokens,
            cached_input_tokens: 0,
            cache_write_input_tokens: 0,
            output_tokens,
        }
    }

    /// Attach provider-reported prompt-cache token counts.
    #[must_use]
    pub fn with_cache_tokens(
        mut self,
        cached_input_tokens: u64,
        cache_write_input_tokens: u64,
    ) -> Self {
        self.cached_input_tokens = cached_input_tokens;
        self.cache_write_input_tokens = cache_write_input_tokens;
        self
    }

    /// Total tokens used (input + output)
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// Cumulative token usage across every provider response in one materialization run.
///
/// Providers normally use one model for the whole run, but `by_model` preserves
/// accounting if a provider reports different concrete model versions across
/// retries. Keys use the response's model identifier when present and the
/// configured model as a fallback.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct RunUsage {
    /// Number of attempts whose provider response included token usage.
    ///
    /// This can be lower than the number of attempts when a transport error
    /// occurs or a provider omits usage metadata.
    pub reported_attempts: usize,
    /// Cumulative input tokens across all reported responses.
    pub input_tokens: u64,
    /// Cumulative input tokens served from provider prompt caches.
    ///
    /// This is a subset of `input_tokens`.
    pub cached_input_tokens: u64,
    /// Cumulative input tokens written to provider prompt caches.
    ///
    /// This is a subset of `input_tokens`.
    pub cache_write_input_tokens: u64,
    /// Cumulative output tokens across all reported responses.
    pub output_tokens: u64,
    /// Cumulative usage grouped by reported model, or configured-model fallback.
    pub by_model: BTreeMap<String, TokenUsage>,
    /// Whether any cumulative counter exceeded its representable range.
    ///
    /// When this is `true`, affected counters and `total_tokens()` saturate at
    /// their maximum value rather than panicking or wrapping.
    pub overflowed: bool,
}

impl RunUsage {
    /// Create an empty cumulative usage record.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create cumulative usage from one provider response.
    #[must_use]
    pub fn from_response(usage: TokenUsage) -> Self {
        let mut total = Self::new();
        total.record(usage);
        total
    }

    /// Add one provider response to the cumulative totals.
    pub fn record(&mut self, usage: TokenUsage) {
        self.reported_attempts = match self.reported_attempts.checked_add(1) {
            Some(attempts) => attempts,
            None => {
                self.overflowed = true;
                usize::MAX
            }
        };
        self.input_tokens =
            saturating_add(&mut self.overflowed, self.input_tokens, usage.input_tokens);
        self.cached_input_tokens = saturating_add(
            &mut self.overflowed,
            self.cached_input_tokens,
            usage.cached_input_tokens,
        );
        self.cache_write_input_tokens = saturating_add(
            &mut self.overflowed,
            self.cache_write_input_tokens,
            usage.cache_write_input_tokens,
        );
        self.output_tokens = saturating_add(
            &mut self.overflowed,
            self.output_tokens,
            usage.output_tokens,
        );

        let model_usage = self
            .by_model
            .entry(usage.model.clone())
            .or_insert_with(|| TokenUsage::new(usage.model, 0, 0));
        model_usage.input_tokens = saturating_add(
            &mut self.overflowed,
            model_usage.input_tokens,
            usage.input_tokens,
        );
        model_usage.cached_input_tokens = saturating_add(
            &mut self.overflowed,
            model_usage.cached_input_tokens,
            usage.cached_input_tokens,
        );
        model_usage.cache_write_input_tokens = saturating_add(
            &mut self.overflowed,
            model_usage.cache_write_input_tokens,
            usage.cache_write_input_tokens,
        );
        model_usage.output_tokens = saturating_add(
            &mut self.overflowed,
            model_usage.output_tokens,
            usage.output_tokens,
        );

        if self.input_tokens.checked_add(self.output_tokens).is_none()
            || model_usage
                .input_tokens
                .checked_add(model_usage.output_tokens)
                .is_none()
        {
            self.overflowed = true;
        }
    }

    /// Total known tokens used across the run.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

fn saturating_add(overflowed: &mut bool, left: u64, right: u64) -> u64 {
    match left.checked_add(right) {
        Some(total) => total,
        None => {
            *overflowed = true;
            u64::MAX
        }
    }
}

/// Whether an attempt reached structured-output validation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptKind {
    /// A structured response reached decoding and custom validation.
    Semantic,
    /// A provider request did not produce a usable structured response.
    Transport,
}

/// Why execution did or did not continue after a failed attempt.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDisposition {
    /// Another provider attempt was made.
    Retried,
    /// The error was retryable, but the configured attempt budget was exhausted.
    BudgetExhausted,
    /// The active retry policy did not permit another attempt for this error.
    NonRetryable,
}

/// Outcome of one materialization attempt.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// Structured output decoded and validated successfully.
    Succeeded,
    /// The attempt failed.
    Failed {
        /// Human-readable error message.
        message: String,
        /// Whether execution continued, exhausted its budget, or stopped early.
        disposition: RetryDisposition,
    },
}

/// Immutable record of one materialization attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AttemptRecord {
    /// One-indexed attempt number.
    pub number: usize,
    /// Whether this was a semantic or transport attempt.
    pub kind: AttemptKind,
    /// Success or categorized failure information.
    pub outcome: AttemptOutcome,
    /// Per-response token usage, when reported by the provider.
    pub usage: Option<TokenUsage>,
    /// HTTP response diagnostics, when the attempt reached a provider.
    pub response: Option<crate::ResponseMetadata>,
}

impl AttemptRecord {
    #[cfg(any(test, feature = "mock"))]
    pub(crate) fn succeeded(number: usize, usage: Option<TokenUsage>) -> Self {
        Self::succeeded_with_response(number, usage, None)
    }

    #[cfg(any(feature = "_client", feature = "mock"))]
    pub(crate) fn succeeded_with_response(
        number: usize,
        usage: Option<TokenUsage>,
        response: Option<crate::ResponseMetadata>,
    ) -> Self {
        Self {
            number,
            kind: AttemptKind::Semantic,
            outcome: AttemptOutcome::Succeeded,
            usage,
            response,
        }
    }

    #[cfg(any(test, feature = "mock"))]
    pub(crate) fn failed(
        number: usize,
        kind: AttemptKind,
        error: &RStructorError,
        disposition: RetryDisposition,
        usage: Option<TokenUsage>,
    ) -> Self {
        Self::failed_with_response(number, kind, error, disposition, usage, None)
    }

    #[cfg(any(feature = "_client", feature = "mock"))]
    pub(crate) fn failed_with_response(
        number: usize,
        kind: AttemptKind,
        error: &RStructorError,
        disposition: RetryDisposition,
        usage: Option<TokenUsage>,
        response: Option<crate::ResponseMetadata>,
    ) -> Self {
        Self {
            number,
            kind,
            outcome: AttemptOutcome::Failed {
                message: error.to_string(),
                disposition,
            },
            usage,
            response,
        }
    }
}

/// Run metadata shared by successful and failed structured extractions.
///
/// The same report shape is available on [`Extraction<T>`] and
/// [`ExtractionError`], so callers can inspect attempts and usage without
/// maintaining separate success and failure accounting code.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ExtractionReport {
    /// Usage from the final provider response, when it was reported.
    ///
    /// On failure this is taken from the final recorded attempt. A transport
    /// failure or provider response that omitted usage leaves it as `None`.
    pub final_usage: Option<TokenUsage>,
    /// Cumulative known usage across every provider response in this run.
    pub cumulative_usage: Option<RunUsage>,
    /// Ordered, one-indexed attempt ledger.
    pub attempts: Vec<AttemptRecord>,
    /// Whether `attempts` and `cumulative_usage` cover the complete run.
    ///
    /// Built-in providers and the optional `MockClient` set this to `true`.
    /// Compatibility fallbacks for custom clients set it to `false` rather
    /// than inventing attempts they cannot observe.
    pub attempts_complete: bool,
}

impl ExtractionReport {
    fn from_success<T>(report: MaterializeReport<T>) -> (T, Self) {
        let MaterializeReport {
            data,
            final_usage,
            cumulative_usage,
            attempts,
            attempts_complete,
        } = report;
        (
            data,
            Self {
                final_usage,
                cumulative_usage,
                attempts,
                attempts_complete,
            },
        )
    }

    fn from_failure(failure: MaterializeFailure) -> (Box<RStructorError>, Self) {
        let MaterializeFailure {
            error,
            cumulative_usage,
            attempts,
            attempts_complete,
        } = failure;
        let final_usage = attempts.last().and_then(|attempt| attempt.usage.clone());
        (
            error,
            Self {
                final_usage,
                cumulative_usage,
                attempts,
                attempts_complete,
            },
        )
    }
}

/// A validated value and its complete available extraction report.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Extraction<T> {
    /// Deserialized and validated data.
    pub data: T,
    /// Usage and attempt metadata for the run.
    pub report: ExtractionReport,
}

impl<T> Extraction<T> {
    /// Map the extracted value while preserving the run report.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Extraction<U> {
        Extraction {
            data: f(self.data),
            report: self.report,
        }
    }

    /// Consume the extraction and return its validated value.
    #[must_use]
    pub fn into_data(self) -> T {
        self.data
    }
}

impl<T> From<MaterializeReport<T>> for Extraction<T> {
    fn from(report: MaterializeReport<T>) -> Self {
        let (data, report) = ExtractionReport::from_success(report);
        Self { data, report }
    }
}

/// A failed extraction and the same report shape returned on success.
#[derive(Debug)]
pub struct ExtractionError {
    error: Box<RStructorError>,
    /// Usage and attempt metadata collected before the failure.
    pub report: ExtractionReport,
}

impl ExtractionError {
    /// Return the original final error.
    #[must_use]
    pub fn error(&self) -> &RStructorError {
        &self.error
    }

    /// Consume the extraction error and return the original final error.
    #[must_use]
    pub fn into_error(self) -> RStructorError {
        *self.error
    }
}

impl From<MaterializeFailure> for ExtractionError {
    fn from(failure: MaterializeFailure) -> Self {
        let (error, report) = ExtractionReport::from_failure(failure);
        Self { error, report }
    }
}

impl fmt::Display for ExtractionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error().fmt(formatter)
    }
}

impl std::error::Error for ExtractionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error())
    }
}

/// The advanced structured-extraction result type.
///
/// Both variants carry an [`ExtractionReport`]: success through
/// [`Extraction::report`] and failure through [`ExtractionError::report`].
pub type ExtractionResult<T> = std::result::Result<Extraction<T>, ExtractionError>;

/// Successful structured-output run with usage and available attempt metadata.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MaterializeReport<T> {
    /// Deserialized and validated data.
    pub data: T,
    /// Usage from the final successful provider response.
    pub final_usage: Option<TokenUsage>,
    /// Cumulative known usage across every provider response in this run.
    pub cumulative_usage: Option<RunUsage>,
    /// Ordered, one-indexed attempt ledger.
    pub attempts: Vec<AttemptRecord>,
    /// Whether `attempts` and `cumulative_usage` cover the complete run.
    ///
    /// This is `true` for built-in providers and `MockClient`. It is `false`
    /// for the default implementation used by custom clients, whose existing
    /// materialization methods do not expose their internal attempts.
    pub attempts_complete: bool,
}

impl<T> MaterializeReport<T> {
    #[cfg(any(feature = "_client", feature = "mock"))]
    pub(crate) fn new(
        data: T,
        final_usage: Option<TokenUsage>,
        cumulative_usage: Option<RunUsage>,
        attempts: Vec<AttemptRecord>,
    ) -> Self {
        Self {
            data,
            final_usage,
            cumulative_usage,
            attempts,
            attempts_complete: true,
        }
    }

    /// Build a report from final-only metadata with unavailable attempt history.
    ///
    /// This is used by the default [`LLMClient`](crate::LLMClient)
    /// implementation for custom clients that do not expose per-attempt
    /// responses. `final_usage` remains available, but `cumulative_usage` and
    /// `attempts` are empty and `attempts_complete` is `false`.
    #[must_use]
    pub fn from_result(result: MaterializeResult<T>) -> Self {
        Self {
            data: result.data,
            final_usage: result.usage,
            cumulative_usage: None,
            attempts: Vec::new(),
            attempts_complete: false,
        }
    }

    /// Map the successful data while preserving usage and attempt metadata.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> MaterializeReport<U> {
        MaterializeReport {
            data: f(self.data),
            final_usage: self.final_usage,
            cumulative_usage: self.cumulative_usage,
            attempts: self.attempts,
            attempts_complete: self.attempts_complete,
        }
    }

    /// Discard the attempt ledger and keep the successful data and final usage.
    #[must_use]
    pub fn into_result(self) -> MaterializeResult<T> {
        MaterializeResult::new(self.data, self.final_usage)
    }
}

/// Failed structured-output run with available usage and attempt metadata.
#[non_exhaustive]
#[derive(Debug)]
pub struct MaterializeFailure {
    /// Final error returned by the last or non-retryable attempt.
    error: Box<RStructorError>,
    /// Cumulative known token usage across every provider response.
    pub cumulative_usage: Option<RunUsage>,
    /// Ordered, one-indexed attempt ledger.
    pub attempts: Vec<AttemptRecord>,
    /// Whether `attempts` and `cumulative_usage` cover the complete run.
    pub attempts_complete: bool,
}

impl MaterializeFailure {
    #[cfg(any(feature = "_client", feature = "mock"))]
    pub(crate) fn new(
        error: RStructorError,
        cumulative_usage: Option<RunUsage>,
        attempts: Vec<AttemptRecord>,
    ) -> Self {
        Self {
            error: Box::new(error),
            cumulative_usage,
            attempts,
            attempts_complete: true,
        }
    }

    /// Create an empty-ledger failure when a client cannot expose attempt metadata.
    #[must_use]
    pub fn from_error(error: RStructorError) -> Self {
        Self {
            error: Box::new(error),
            cumulative_usage: None,
            attempts: Vec::new(),
            attempts_complete: false,
        }
    }

    /// Return the original final error.
    #[must_use]
    pub fn error(&self) -> &RStructorError {
        &self.error
    }

    /// Consume the report and return the original final error.
    #[must_use]
    pub fn into_error(self) -> RStructorError {
        *self.error
    }
}

impl fmt::Display for MaterializeFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error().fmt(formatter)
    }
}

impl std::error::Error for MaterializeFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error())
    }
}

/// Result of a materialize call, containing both the data and optional usage information.
///
/// This struct wraps the deserialized data along with token usage metadata
/// from the final successful LLM API call.
///
/// # Example
///
/// ```no_run
/// use rstructor::{LLMClient, OpenAIClient, Instructor};
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Instructor, Serialize, Deserialize)]
/// struct Person { name: String, age: u8 }
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = OpenAIClient::from_env()?;
/// let result = client.materialize_with_metadata::<Person>("Describe a person").await?;
///
/// // Access the data directly
/// println!("Name: {}", result.data.name);
///
/// // Check token usage
/// if let Some(usage) = result.usage {
///     println!("Used {} total tokens", usage.total_tokens());
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MaterializeResult<T> {
    /// The deserialized data
    pub data: T,
    /// Token usage information (if available from the provider)
    pub usage: Option<TokenUsage>,
}

impl<T> MaterializeResult<T> {
    /// Create a new MaterializeResult with data and usage
    pub fn new(data: T, usage: Option<TokenUsage>) -> Self {
        Self { data, usage }
    }

    /// Create a MaterializeResult with just data (no usage info)
    pub fn from_data(data: T) -> Self {
        Self { data, usage: None }
    }

    /// Map the data to a new type
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> MaterializeResult<U> {
        MaterializeResult {
            data: f(self.data),
            usage: self.usage,
        }
    }
}

/// Result of a generate call, containing the text and optional usage information.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// The generated text
    pub text: String,
    /// Token usage information (if available from the provider)
    pub usage: Option<TokenUsage>,
}

impl GenerateResult {
    /// Create a new GenerateResult with text and usage
    pub fn new(text: String, usage: Option<TokenUsage>) -> Self {
        Self { text, usage }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_usage_groups_exact_provider_model_versions() {
        let mut usage = RunUsage::new();
        usage.record(TokenUsage::new("gpt-5.6-2026-07-01", 120, 30).with_cache_tokens(80, 0));
        usage.record(TokenUsage::new("gpt-5.6-2026-07-01", 180, 45).with_cache_tokens(100, 20));
        usage.record(TokenUsage::new("gpt-5.6-2026-07-15", 200, 50).with_cache_tokens(0, 150));

        assert_eq!(usage.reported_attempts, 3);
        assert_eq!(usage.input_tokens, 500);
        assert_eq!(usage.cached_input_tokens, 180);
        assert_eq!(usage.cache_write_input_tokens, 170);
        assert_eq!(usage.output_tokens, 125);
        assert_eq!(usage.total_tokens(), 625);
        assert!(!usage.overflowed);
        assert_eq!(
            usage.by_model["gpt-5.6-2026-07-01"],
            TokenUsage::new("gpt-5.6-2026-07-01", 300, 75).with_cache_tokens(180, 20)
        );
        assert_eq!(
            usage.by_model["gpt-5.6-2026-07-15"],
            TokenUsage::new("gpt-5.6-2026-07-15", 200, 50).with_cache_tokens(0, 150)
        );
    }

    #[test]
    fn run_usage_saturates_and_flags_untrusted_counter_overflow() {
        let mut usage = RunUsage::new();
        usage.record(
            TokenUsage::new("hostile-compatible-endpoint", u64::MAX, 1)
                .with_cache_tokens(u64::MAX, u64::MAX),
        );
        usage.record(
            TokenUsage::new("hostile-compatible-endpoint", 1, u64::MAX).with_cache_tokens(1, 1),
        );

        assert!(usage.overflowed);
        assert_eq!(usage.reported_attempts, 2);
        assert_eq!(usage.input_tokens, u64::MAX);
        assert_eq!(usage.cached_input_tokens, u64::MAX);
        assert_eq!(usage.cache_write_input_tokens, u64::MAX);
        assert_eq!(usage.output_tokens, u64::MAX);
        assert_eq!(usage.total_tokens(), u64::MAX);
        assert_eq!(
            usage.by_model["hostile-compatible-endpoint"].input_tokens,
            u64::MAX
        );
        assert_eq!(
            usage.by_model["hostile-compatible-endpoint"].output_tokens,
            u64::MAX
        );
        assert_eq!(
            usage.by_model["hostile-compatible-endpoint"].cached_input_tokens,
            u64::MAX
        );
        assert_eq!(
            usage.by_model["hostile-compatible-endpoint"].cache_write_input_tokens,
            u64::MAX
        );
    }

    #[test]
    fn custom_client_result_preserves_final_usage_without_inventing_attempts() {
        let final_usage = TokenUsage::new("mock-risk-model", 42, 11);
        let report =
            MaterializeReport::from_result(MaterializeResult::new("portfolio", Some(final_usage)));

        assert_eq!(report.data, "portfolio");
        assert_eq!(report.final_usage.as_ref().unwrap().total_tokens(), 53);
        assert!(report.cumulative_usage.is_none());
        assert!(report.attempts.is_empty());
        assert!(!report.attempts_complete);
    }

    #[test]
    fn unknown_custom_client_failure_does_not_invent_a_provider_attempt() {
        let failure =
            MaterializeFailure::from_error(RStructorError::SchemaError("bad schema".into()));

        assert!(failure.attempts.is_empty());
        assert!(failure.cumulative_usage.is_none());
        assert!(!failure.attempts_complete);
        assert!(matches!(failure.error(), RStructorError::SchemaError(_)));
    }

    #[test]
    fn extraction_success_uses_the_shared_report_and_maps_data() {
        let final_usage = TokenUsage::new("risk-model-v2", 120, 18);
        let cumulative_usage = RunUsage::from_response(final_usage.clone());
        let legacy = MaterializeReport::new(
            "HF-ALPHA-001",
            Some(final_usage.clone()),
            Some(cumulative_usage.clone()),
            vec![AttemptRecord::succeeded(1, Some(final_usage.clone()))],
        );

        let extraction = Extraction::from(legacy).map(str::len);

        assert_eq!(extraction.data, 12);
        assert_eq!(extraction.report.final_usage, Some(final_usage));
        assert_eq!(extraction.report.cumulative_usage, Some(cumulative_usage));
        assert_eq!(extraction.report.attempts.len(), 1);
        assert!(extraction.report.attempts_complete);
    }

    #[test]
    fn extraction_failure_uses_the_same_report_shape_and_final_attempt_usage() {
        let first_usage = TokenUsage::new("risk-model-v1", 90, 15);
        let final_usage = TokenUsage::new("risk-model-v2", 100, 12);
        let mut cumulative_usage = RunUsage::new();
        cumulative_usage.record(first_usage.clone());
        cumulative_usage.record(final_usage.clone());
        let final_error = RStructorError::OutputDecodeError {
            path: "$.positions[1].quantity".into(),
            message: "invalid type: string, expected i64".into(),
        };
        let failure = MaterializeFailure::new(
            final_error,
            Some(cumulative_usage.clone()),
            vec![
                AttemptRecord::failed(
                    1,
                    AttemptKind::Semantic,
                    &RStructorError::ValidationError("invalid quantity".into()),
                    RetryDisposition::Retried,
                    Some(first_usage),
                ),
                AttemptRecord::failed(
                    2,
                    AttemptKind::Semantic,
                    &RStructorError::ValidationError("invalid quantity".into()),
                    RetryDisposition::BudgetExhausted,
                    Some(final_usage.clone()),
                ),
            ],
        );

        let error = ExtractionError::from(failure);

        assert!(matches!(
            error.error(),
            RStructorError::OutputDecodeError { path, .. }
                if path == "$.positions[1].quantity"
        ));
        assert_eq!(error.report.final_usage, Some(final_usage));
        assert_eq!(error.report.cumulative_usage, Some(cumulative_usage));
        assert_eq!(error.report.attempts.len(), 2);
        assert!(error.report.attempts_complete);
    }
}

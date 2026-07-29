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
    /// Number of tokens in the output/completion
    pub output_tokens: u64,
}

impl TokenUsage {
    /// Create a new TokenUsage instance
    pub fn new(model: impl Into<String>, input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            model: model.into(),
            input_tokens,
            output_tokens,
        }
    }

    /// Total tokens used (input + output)
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// Cumulative token usage across every provider response in one materialization run.
///
/// Providers normally use one model for the whole run, but `by_model` preserves
/// exact accounting if a provider reports different concrete model versions
/// across retries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RunUsage {
    /// Number of attempts whose provider response included token usage.
    ///
    /// This can be lower than the number of attempts when a transport error
    /// occurs or a provider omits usage metadata.
    pub reported_attempts: usize,
    /// Cumulative input tokens across all reported responses.
    pub input_tokens: u64,
    /// Cumulative output tokens across all reported responses.
    pub output_tokens: u64,
    /// Cumulative usage grouped by the exact model identifier reported.
    pub by_model: BTreeMap<String, TokenUsage>,
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
        self.reported_attempts += 1;
        self.input_tokens += usage.input_tokens;
        self.output_tokens += usage.output_tokens;

        let model_usage = self
            .by_model
            .entry(usage.model.clone())
            .or_insert_with(|| TokenUsage::new(usage.model, 0, 0));
        model_usage.input_tokens += usage.input_tokens;
        model_usage.output_tokens += usage.output_tokens;
    }

    /// Total known tokens used across the run.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// Whether an attempt reached structured-output validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptKind {
    /// A structured response reached decoding and custom validation.
    Semantic,
    /// A provider request did not produce a usable structured response.
    Transport,
}

/// Outcome of one materialization attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// Structured output decoded and validated successfully.
    Succeeded,
    /// The attempt failed.
    Failed {
        /// Human-readable error message.
        message: String,
        /// Whether another attempt was actually made after this failure.
        retried: bool,
    },
}

/// Immutable record of one materialization attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttemptRecord {
    /// One-indexed attempt number.
    pub number: usize,
    /// Whether this was a semantic or transport attempt.
    pub kind: AttemptKind,
    /// Success or categorized failure information.
    pub outcome: AttemptOutcome,
    /// Per-response token usage, when reported by the provider.
    pub usage: Option<TokenUsage>,
}

impl AttemptRecord {
    pub(crate) fn succeeded(number: usize, usage: Option<TokenUsage>) -> Self {
        Self {
            number,
            kind: AttemptKind::Semantic,
            outcome: AttemptOutcome::Succeeded,
            usage,
        }
    }

    pub(crate) fn failed(
        number: usize,
        kind: AttemptKind,
        error: &RStructorError,
        retried: bool,
        usage: Option<TokenUsage>,
    ) -> Self {
        Self {
            number,
            kind,
            outcome: AttemptOutcome::Failed {
                message: error.to_string(),
                retried,
            },
            usage,
        }
    }
}

/// Successful structured-output run with its complete attempt ledger.
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
}

impl<T> MaterializeReport<T> {
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
        }
    }

    /// Build a one-attempt report from a client's existing metadata result.
    ///
    /// This is used by the default [`LLMClient`](crate::LLMClient)
    /// implementation for custom clients that do not expose per-attempt
    /// responses. Built-in clients override that method with the full ledger.
    #[must_use]
    pub fn from_result(result: MaterializeResult<T>) -> Self {
        let final_usage = result.usage;
        let cumulative_usage = final_usage.clone().map(RunUsage::from_response);
        Self::new(
            result.data,
            final_usage.clone(),
            cumulative_usage,
            vec![AttemptRecord::succeeded(1, final_usage)],
        )
    }

    /// Map the successful data while preserving usage and attempt metadata.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> MaterializeReport<U> {
        MaterializeReport {
            data: f(self.data),
            final_usage: self.final_usage,
            cumulative_usage: self.cumulative_usage,
            attempts: self.attempts,
        }
    }

    /// Discard the attempt ledger and keep the successful data and final usage.
    #[must_use]
    pub fn into_result(self) -> MaterializeResult<T> {
        MaterializeResult::new(self.data, self.final_usage)
    }
}

/// Failed structured-output run with cumulative usage and every attempt.
#[derive(Debug)]
pub struct MaterializeFailure {
    /// Final error returned by the last or non-retryable attempt.
    error: Box<RStructorError>,
    /// Cumulative known token usage across every provider response.
    pub cumulative_usage: Option<RunUsage>,
    /// Ordered, one-indexed attempt ledger.
    pub attempts: Vec<AttemptRecord>,
}

impl MaterializeFailure {
    pub(crate) fn new(
        error: RStructorError,
        cumulative_usage: Option<RunUsage>,
        attempts: Vec<AttemptRecord>,
    ) -> Self {
        Self {
            error: Box::new(error),
            cumulative_usage,
            attempts,
        }
    }

    /// Create an empty-ledger failure when a client cannot expose attempt metadata.
    #[must_use]
    pub fn from_error(error: RStructorError) -> Self {
        Self::new(error, None, Vec::new())
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
        usage.record(TokenUsage::new("gpt-5.6-2026-07-01", 120, 30));
        usage.record(TokenUsage::new("gpt-5.6-2026-07-01", 180, 45));
        usage.record(TokenUsage::new("gpt-5.6-2026-07-15", 200, 50));

        assert_eq!(usage.reported_attempts, 3);
        assert_eq!(usage.input_tokens, 500);
        assert_eq!(usage.output_tokens, 125);
        assert_eq!(usage.total_tokens(), 625);
        assert_eq!(
            usage.by_model["gpt-5.6-2026-07-01"],
            TokenUsage::new("gpt-5.6-2026-07-01", 300, 75)
        );
        assert_eq!(
            usage.by_model["gpt-5.6-2026-07-15"],
            TokenUsage::new("gpt-5.6-2026-07-15", 200, 50)
        );
    }

    #[test]
    fn custom_client_result_becomes_one_successful_semantic_attempt() {
        let final_usage = TokenUsage::new("mock-risk-model", 42, 11);
        let report =
            MaterializeReport::from_result(MaterializeResult::new("portfolio", Some(final_usage)));

        assert_eq!(report.data, "portfolio");
        assert_eq!(report.final_usage.as_ref().unwrap().total_tokens(), 53);
        assert_eq!(report.cumulative_usage.as_ref().unwrap().total_tokens(), 53);
        assert_eq!(report.attempts.len(), 1);
        assert_eq!(report.attempts[0].kind, AttemptKind::Semantic);
        assert_eq!(report.attempts[0].outcome, AttemptOutcome::Succeeded);
    }

    #[test]
    fn unknown_custom_client_failure_does_not_invent_a_provider_attempt() {
        let failure =
            MaterializeFailure::from_error(RStructorError::SchemaError("bad schema".into()));

        assert!(failure.attempts.is_empty());
        assert!(failure.cumulative_usage.is_none());
        assert!(matches!(failure.error(), RStructorError::SchemaError(_)));
    }
}

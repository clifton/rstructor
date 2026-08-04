//! Privacy-aware diagnostics for provider responses.

use std::collections::BTreeMap;
#[cfg(feature = "_client")]
use std::{fmt, sync::Arc};

/// A response body that has passed through a caller-provided sanitizer.
///
/// rstructor never stores provider response bodies by default. When capture is
/// explicitly enabled, the sanitizer runs before the value is retained and the
/// sanitized value is bounded by the configured byte limit.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SanitizedResponseBody {
    /// Sanitized response text retained for diagnostics.
    pub text: String,
    /// Whether the sanitized text exceeded the capture limit and was truncated.
    pub truncated: bool,
}

/// HTTP metadata retained for one provider response.
///
/// Status and recognized request-ID headers are captured for both successful
/// and failed structured-output attempts. Response bodies remain absent unless
/// the client was configured with opt-in response body capture.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResponseMetadata {
    /// Exact HTTP status code returned by the provider.
    pub status: u16,
    /// Recognized request-ID headers, keyed by their lowercase header name.
    pub request_ids: BTreeMap<String, String>,
    /// Optional caller-sanitized response body.
    pub sanitized_body: Option<SanitizedResponseBody>,
}

impl ResponseMetadata {
    /// Create response metadata with no request IDs or body capture.
    #[must_use]
    pub fn new(status: u16) -> Self {
        Self {
            status,
            request_ids: BTreeMap::new(),
            sanitized_body: None,
        }
    }

    /// Return a preferred request ID when the provider supplied one.
    ///
    /// All recognized IDs remain available in [`request_ids`](Self::request_ids).
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        const PREFERRED_HEADERS: &[&str] = &[
            "x-request-id",
            "request-id",
            "anthropic-request-id",
            "openai-request-id",
            "x-goog-request-id",
            "x-amzn-requestid",
            "x-amz-request-id",
        ];

        PREFERRED_HEADERS
            .iter()
            .find_map(|header| self.request_ids.get(*header).map(String::as_str))
            .or_else(|| self.request_ids.values().next().map(String::as_str))
    }
}

#[cfg(feature = "_client")]
type Sanitizer = dyn Fn(&str) -> String + Send + Sync + 'static;

/// Opt-in, caller-controlled response-body capture.
///
/// The callback is the privacy boundary: it receives the body transiently and
/// must return only content safe to retain in errors, reports, logs, or test
/// fixtures. rstructor stores only the callback's bounded output.
#[derive(Clone)]
#[cfg(feature = "_client")]
pub struct ResponseBodyCapture {
    max_bytes: usize,
    sanitizer: Arc<Sanitizer>,
}

#[cfg(feature = "_client")]
impl ResponseBodyCapture {
    /// Default maximum number of sanitized bytes retained per response.
    pub const DEFAULT_MAX_BYTES: usize = 16 * 1024;

    /// Enable response-body capture with a caller-provided sanitizer.
    ///
    /// The sanitizer is mandatory so enabling diagnostics cannot accidentally
    /// become a boolean switch that stores raw model output.
    pub fn new<F>(sanitizer: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        Self {
            max_bytes: Self::DEFAULT_MAX_BYTES,
            sanitizer: Arc::new(sanitizer),
        }
    }

    /// Set the maximum number of sanitized bytes retained per response.
    #[must_use]
    pub fn max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    pub(crate) fn capture(&self, raw_body: &str) -> SanitizedResponseBody {
        let sanitized = (self.sanitizer)(raw_body);
        let boundary = sanitized.floor_char_boundary(self.max_bytes.min(sanitized.len()));
        let truncated = boundary < sanitized.len();
        SanitizedResponseBody {
            text: sanitized[..boundary].to_string(),
            truncated,
        }
    }
}

#[cfg(feature = "_client")]
impl fmt::Debug for ResponseBodyCapture {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResponseBodyCapture")
            .field("max_bytes", &self.max_bytes)
            .field("sanitizer", &"<caller-provided>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "_client")]
    #[test]
    fn capture_sanitizes_before_retaining_and_truncates_on_utf8_boundary() {
        let capture =
            ResponseBodyCapture::new(|body| body.replace("secret", "[REDACTED]")).max_bytes(13);

        let body = capture.capture("secret тест");

        assert_eq!(body.text, "[REDACTED] т");
        assert!(body.truncated);
        assert!(!body.text.contains("secret"));
    }

    #[test]
    fn preferred_request_id_is_deterministic() {
        let mut metadata = ResponseMetadata::new(200);
        metadata
            .request_ids
            .insert("request-id".to_string(), "fallback".to_string());
        metadata
            .request_ids
            .insert("x-request-id".to_string(), "preferred".to_string());

        assert_eq!(metadata.request_id(), Some("preferred"));
    }
}

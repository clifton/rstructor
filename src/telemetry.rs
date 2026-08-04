//! OpenTelemetry-compatible GenAI tracing fields.
//!
//! Built-in non-streaming extraction and generation calls emit `tracing` spans
//! that follow the OpenTelemetry GenAI inference-client semantic conventions.
//! Applications can export them with their existing `tracing-opentelemetry`
//! layer; rstructor does not install a global subscriber or add an
//! OpenTelemetry SDK dependency.
//!
//! The GenAI conventions are currently development-stability. Prompt, system,
//! response, and tool content attributes are deliberately never emitted.

use tracing::{Span, field, info_span};

use crate::{ApiErrorKind, RStructorError, TokenUsage};

/// Stability of the implemented OpenTelemetry GenAI semantic conventions.
pub const GEN_AI_SEMCONV_STABILITY: &str = "development";

/// Create one GenAI client span for one remote inference attempt.
pub(crate) fn inference_span(
    provider: &'static str,
    operation: &'static str,
    model: &str,
    endpoint: &str,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
) -> Span {
    let otel_name = format!("{operation} {model}");
    let span = info_span!(
        "gen_ai.client.inference",
        otel.name = %otel_name,
        otel.kind = "client",
        otel.status_code = field::Empty,
        gen_ai.provider.name = provider,
        gen_ai.operation.name = operation,
        gen_ai.request.model = model,
        gen_ai.request.temperature = field::Empty,
        gen_ai.request.max_tokens = field::Empty,
        gen_ai.output.type = "json",
        gen_ai.response.id = field::Empty,
        gen_ai.response.model = field::Empty,
        gen_ai.usage.input_tokens = field::Empty,
        gen_ai.usage.output_tokens = field::Empty,
        gen_ai.usage.cache_read.input_tokens = field::Empty,
        gen_ai.usage.cache_creation.input_tokens = field::Empty,
        server.address = field::Empty,
        server.port = field::Empty,
        error.type = field::Empty,
    );

    if let Some(temperature) = temperature {
        span.record("gen_ai.request.temperature", temperature);
    }
    if let Some(max_tokens) = max_tokens {
        span.record("gen_ai.request.max_tokens", max_tokens);
    }
    if let Ok(url) = reqwest::Url::parse(endpoint) {
        if let Some(host) = url.host_str() {
            span.record("server.address", host);
        }
        if let Some(port) = url.port_or_known_default() {
            span.record("server.port", u64::from(port));
        }
    }

    span
}

pub(crate) fn record_success(
    span: &Span,
    response_id: Option<&str>,
    response_model: Option<&str>,
    usage: Option<&TokenUsage>,
) {
    if let Some(response_id) = response_id {
        span.record("gen_ai.response.id", response_id);
    }
    if let Some(response_model) = response_model {
        span.record("gen_ai.response.model", response_model);
    }
    if let Some(usage) = usage {
        span.record("gen_ai.usage.input_tokens", usage.input_tokens);
        span.record("gen_ai.usage.output_tokens", usage.output_tokens);
        span.record(
            "gen_ai.usage.cache_read.input_tokens",
            usage.cached_input_tokens,
        );
        span.record(
            "gen_ai.usage.cache_creation.input_tokens",
            usage.cache_write_input_tokens,
        );
    }
}

pub(crate) fn record_error(span: &Span, error: &RStructorError) {
    let error_type = error_type(error);
    span.record("error.type", error_type.as_str());
    span.record("otel.status_code", "ERROR");
}

pub(crate) fn record_http_client_error(span: &Span, error: &reqwest::Error) {
    let error_type = if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connection_error"
    } else if error.is_builder() {
        "request_build_error"
    } else {
        "http_error"
    };
    span.record("error.type", error_type);
    span.record("otel.status_code", "ERROR");
}

fn error_type(error: &RStructorError) -> String {
    if let Some(status) = error.status_code() {
        return status.to_string();
    }

    match error {
        RStructorError::ApiError { kind, .. } => match kind {
            ApiErrorKind::RateLimited { .. } => "rate_limit",
            ApiErrorKind::InvalidModel { .. } => "invalid_model",
            ApiErrorKind::ServiceUnavailable => "service_unavailable",
            ApiErrorKind::GatewayError { .. } => "gateway_error",
            ApiErrorKind::AuthenticationFailed => "authentication_error",
            ApiErrorKind::PermissionDenied => "permission_denied",
            ApiErrorKind::RequestTooLarge => "request_too_large",
            ApiErrorKind::BadRequest { .. } => "bad_request",
            ApiErrorKind::ServerError { .. } => "server_error",
            ApiErrorKind::Other { .. } => "api_error",
            ApiErrorKind::UnexpectedResponse { .. } => "unexpected_response",
        },
        RStructorError::ValidationError(_) => "validation_error",
        RStructorError::SchemaError(_) | RStructorError::SchemaCompatibilityError { .. } => {
            "schema_error"
        }
        RStructorError::SerializationError(_)
        | RStructorError::OutputDecodeError { .. }
        | RStructorError::ToolArgumentDecodeError { .. }
        | RStructorError::JsonError(_) => "decode_error",
        RStructorError::StreamingError { .. } => "stream_error",
        RStructorError::Timeout => "timeout",
        RStructorError::Unsupported(_) => "unsupported",
        RStructorError::HttpError(error) if error.is_connect() => "connection_error",
        RStructorError::HttpError(_) => "http_error",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "logging")]
    use std::{collections::BTreeMap, sync::Arc};
    #[cfg(feature = "logging")]
    use tracing::{Subscriber, field::Visit, span};
    #[cfg(feature = "logging")]
    use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

    #[test]
    fn http_status_is_the_low_cardinality_error_type_when_available() {
        let error = RStructorError::api_error_with_response(
            "OpenAI",
            ApiErrorKind::RateLimited { retry_after: None },
            crate::ResponseMetadata::new(429),
        );

        assert_eq!(error_type(&error), "429");
    }

    #[test]
    fn internal_errors_have_stable_categories() {
        assert_eq!(error_type(&RStructorError::Timeout), "timeout");
        assert_eq!(
            error_type(&RStructorError::OutputDecodeError {
                path: "$.quantity".to_string(),
                message: "invalid type".to_string(),
            }),
            "decode_error"
        );
    }

    #[cfg(feature = "logging")]
    struct FieldVisitor<'a> {
        fields: &'a mut BTreeMap<String, String>,
    }

    #[cfg(feature = "logging")]
    impl Visit for FieldVisitor<'_> {
        fn record_debug(&mut self, field: &field::Field, value: &dyn std::fmt::Debug) {
            self.fields
                .insert(field.name().to_string(), format!("{value:?}"));
        }

        fn record_str(&mut self, field: &field::Field, value: &str) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_u64(&mut self, field: &field::Field, value: u64) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_f64(&mut self, field: &field::Field, value: f64) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }
    }

    #[cfg(feature = "logging")]
    #[derive(Clone)]
    struct CaptureLayer(Arc<std::sync::Mutex<BTreeMap<String, String>>>);

    #[cfg(feature = "logging")]
    impl<S> Layer<S> for CaptureLayer
    where
        S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    {
        fn on_new_span(
            &self,
            attributes: &span::Attributes<'_>,
            _id: &span::Id,
            _context: Context<'_, S>,
        ) {
            attributes.record(&mut FieldVisitor {
                fields: &mut self.0.lock().unwrap(),
            });
        }

        fn on_record(&self, _id: &span::Id, values: &span::Record<'_>, _context: Context<'_, S>) {
            values.record(&mut FieldVisitor {
                fields: &mut self.0.lock().unwrap(),
            });
        }
    }

    #[cfg(feature = "logging")]
    #[test]
    fn span_uses_gen_ai_semantic_fields_without_sensitive_content() {
        let fields = Arc::new(std::sync::Mutex::new(BTreeMap::new()));
        let subscriber = tracing_subscriber::registry().with(CaptureLayer(fields.clone()));

        tracing::subscriber::with_default(subscriber, || {
            let span = inference_span(
                "openai",
                "chat",
                "gpt-5",
                "https://api.openai.com/v1/chat/completions",
                Some(0.2),
                Some(512),
            );
            record_success(
                &span,
                Some("resp_123"),
                Some("gpt-5-2026-07-01"),
                Some(&TokenUsage::new("gpt-5", 21, 8).with_cache_tokens(5, 3)),
            );
        });

        let fields = fields.lock().unwrap();
        assert_eq!(fields.get("otel.kind").map(String::as_str), Some("client"));
        assert_eq!(
            fields.get("gen_ai.provider.name").map(String::as_str),
            Some("openai")
        );
        assert_eq!(
            fields.get("gen_ai.operation.name").map(String::as_str),
            Some("chat")
        );
        assert_eq!(
            fields.get("gen_ai.response.id").map(String::as_str),
            Some("resp_123")
        );
        assert_eq!(
            fields.get("gen_ai.usage.input_tokens").map(String::as_str),
            Some("21")
        );
        assert_eq!(
            fields
                .get("gen_ai.usage.cache_read.input_tokens")
                .map(String::as_str),
            Some("5")
        );
        assert!(!fields.keys().any(|key| {
            key.contains("input.messages")
                || key.contains("output.messages")
                || key.contains("system_instructions")
        }));
    }
}

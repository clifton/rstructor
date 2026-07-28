//! Shared infrastructure for streaming responses over server-sent events (SSE).
//!
//! All four providers stream chat/text responses as an SSE body: a sequence of
//! `data: <json>` lines separated by blank lines, optionally terminated by a
//! `data: [DONE]` sentinel (OpenAI/Grok). The JSON shape differs per provider, so
//! each backend supplies a small `extract` closure that pulls the incremental text
//! out of one parsed event; the SSE framing and chunk-boundary buffering are shared
//! here.
//!
//! Two kinds of stream are built on this:
//!
//! - **Text streaming** ([`sse_text_stream`]) yields raw text deltas.
//! - **Object streaming** ([`object_stream`]) accumulates the streamed text (which
//!   for a structured request is partial JSON), and after each delta tries to
//!   repair the buffer into valid JSON and yield a [`StreamedObject::Partial`]
//!   snapshot; when the stream ends it parses and validates the full buffer into
//!   the target type and yields [`StreamedObject::Complete`].
//!
//! This module is only compiled with the `streaming` feature.

use std::future::Future;
use std::pin::Pin;

use async_stream::try_stream;
use futures_util::{Stream, StreamExt};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{RStructorError, Result, StreamErrorKind};
use crate::model::Instructor;

/// A boxed stream of text deltas. Each item is either an incremental piece of the
/// model's text output or a transport/decode error.
pub type TextStream<'a> = Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>>;

/// A boxed stream of [`StreamedObject`] items for a streaming structured request.
pub type ObjectStream<'a, T> = Pin<Box<dyn Stream<Item = Result<StreamedObject<T>>> + Send + 'a>>;

/// Build a stream that yields one locally detected error and performs no I/O.
///
/// Provider methods use this when synchronous request construction discovers an
/// incompatible schema but their public streaming API cannot return `Result`
/// directly.
pub(crate) fn error_stream<'a, T>(
    error: RStructorError,
) -> Pin<Box<dyn Stream<Item = Result<T>> + Send + 'a>>
where
    T: Send + 'a,
{
    Box::pin(futures_util::stream::once(async move { Err(error) }))
}

/// An item yielded by a streaming structured ("object") request.
#[derive(Debug, Clone)]
pub enum StreamedObject<T> {
    /// A progressively-completed snapshot of the object as raw JSON, emitted as
    /// more of the response arrives. Fields not yet generated are simply absent.
    Partial(Value),
    /// The final, fully parsed and validated value. Always the last item on a
    /// successful stream.
    Complete(T),
}

impl<T> StreamedObject<T> {
    /// The final value, if this is the [`Complete`](StreamedObject::Complete) item.
    pub fn complete(self) -> Option<T> {
        match self {
            StreamedObject::Complete(value) => Some(value),
            StreamedObject::Partial(_) => None,
        }
    }
}

/// One decoded SSE event of interest.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SseEvent {
    /// The payload of a `data:` line (raw, usually JSON).
    Data(String),
    /// The `[DONE]` sentinel that ends an OpenAI-style stream.
    Done,
}

/// Incremental SSE line decoder.
///
/// Bytes arrive in arbitrary HTTP chunks that do not respect line boundaries, so
/// the decoder buffers a partial trailing line between [`push`](Self::push) calls
/// and only emits events for lines it has seen in full.
pub(crate) struct SseDecoder {
    buf: Vec<u8>,
    data_lines: Vec<String>,
    first_line: bool,
}

impl Default for SseDecoder {
    fn default() -> Self {
        Self {
            buf: Vec::new(),
            data_lines: Vec::new(),
            first_line: true,
        }
    }
}

impl SseDecoder {
    /// Feed a chunk of bytes, returning any complete `data:` events it completed.
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>> {
        self.buf.extend_from_slice(chunk);
        let mut events = Vec::new();

        while let Some(end) = self
            .buf
            .iter()
            .position(|&byte| matches!(byte, b'\r' | b'\n'))
        {
            if self.buf[end] == b'\r' && end + 1 == self.buf.len() {
                // Preserve a trailing CR until the next chunk tells us whether
                // it is a CRLF pair or a standalone SSE line ending.
                break;
            }
            let delimiter_len =
                usize::from(self.buf[end] == b'\r' && self.buf.get(end + 1) == Some(&b'\n')) + 1;
            let line_bytes: Vec<u8> = self.buf.drain(..end).collect();
            self.buf.drain(..delimiter_len);
            self.process_line(&line_bytes, &mut events)?;
        }

        Ok(events)
    }

    /// Finish decoding at HTTP EOF.
    ///
    /// SSE permits the final line/event to end at EOF without a trailing blank
    /// line, so any buffered line is processed before the pending event is
    /// dispatched.
    pub(crate) fn finish(&mut self) -> Result<Vec<SseEvent>> {
        let mut events = Vec::new();
        if !self.buf.is_empty() {
            let line_bytes = std::mem::take(&mut self.buf);
            let line_bytes = line_bytes.strip_suffix(b"\r").unwrap_or(&line_bytes);
            self.process_line(line_bytes, &mut events)?;
        }
        self.dispatch(&mut events);
        Ok(events)
    }

    fn process_line(&mut self, line_bytes: &[u8], events: &mut Vec<SseEvent>) -> Result<()> {
        let line = std::str::from_utf8(line_bytes).map_err(|error| {
            streaming_error(
                StreamErrorKind::InvalidEventEncoding,
                format!("SSE data was not valid UTF-8: {error}"),
            )
        })?;
        let line = if self.first_line {
            self.first_line = false;
            line.strip_prefix('\u{feff}').unwrap_or(line)
        } else {
            line
        };

        if line.is_empty() {
            self.dispatch(events);
            return Ok(());
        }

        // SSE: only `data` fields carry content. Ignore `event`, `id`, `retry`,
        // comment lines, and unknown fields. Multiple `data` lines in one event
        // are joined with newlines when the blank separator dispatches it.
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        if field == "data" {
            self.data_lines
                .push(value.strip_prefix(' ').unwrap_or(value).to_owned());
        }
        Ok(())
    }

    fn dispatch(&mut self, events: &mut Vec<SseEvent>) {
        if self.data_lines.is_empty() {
            return;
        }
        let data = self.data_lines.join("\n");
        self.data_lines.clear();
        if data == "[DONE]" {
            events.push(SseEvent::Done);
        } else {
            events.push(SseEvent::Data(data));
        }
    }
}

fn streaming_error(kind: StreamErrorKind, message: impl Into<Box<str>>) -> RStructorError {
    RStructorError::StreamingError {
        kind,
        message: message.into(),
    }
}

fn parse_event_json(data: &str) -> Result<Value> {
    serde_json::from_str(data).map_err(|error| {
        streaming_error(
            StreamErrorKind::InvalidEventJson,
            format!(
                "SSE data payload was not valid JSON at line {}, column {}: {}",
                error.line(),
                error.column(),
                error
            ),
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminalMarker {
    DoneSentinel,
    AnthropicMessageStop,
    GeminiFinishReason,
}

impl TerminalMarker {
    fn description(self) -> &'static str {
        match self {
            Self::DoneSentinel => "`[DONE]` sentinel",
            Self::AnthropicMessageStop => "Anthropic `message_stop` event",
            Self::GeminiFinishReason => "Gemini `finishReason`",
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ProviderStreamEvent {
    text: Option<String>,
    terminal: bool,
}

fn invalid_provider_event(message: impl Into<Box<str>>) -> RStructorError {
    streaming_error(StreamErrorKind::InvalidProviderEvent, message)
}

fn require_terminal(terminal: bool, marker: TerminalMarker) -> Result<()> {
    if terminal {
        Ok(())
    } else {
        Err(streaming_error(
            StreamErrorKind::IncompleteEventStream,
            format!(
                "HTTP response ended before the required {}",
                marker.description()
            ),
        ))
    }
}

/// Build a raw-text stream from an SSE HTTP response.
///
/// `send` is the (async) request that yields the streaming response; deferring it
/// lets this function return a `Stream` synchronously. `extract` pulls the
/// incremental text out of each parsed `data:` JSON event.
pub(crate) fn sse_text_stream<'a, Fut, F>(
    send: Fut,
    extract: F,
    marker: TerminalMarker,
) -> TextStream<'a>
where
    Fut: Future<Output = Result<reqwest::Response>> + Send + 'a,
    F: Fn(&Value) -> Result<ProviderStreamEvent> + Send + 'a,
{
    Box::pin(try_stream! {
        let response = send.await?;
        let mut bytes = response.bytes_stream();
        let mut decoder = SseDecoder::default();

        let mut terminal = false;
        loop {
            let (events, eof) = match bytes.next().await {
                Some(chunk) => {
                    let chunk = chunk.map_err(RStructorError::from)?;
                    (decoder.push(chunk.as_ref())?, false)
                }
                None => (decoder.finish()?, true),
            };
            for event in events {
                match event {
                    SseEvent::Done => {
                        if marker != TerminalMarker::DoneSentinel {
                            Err(invalid_provider_event(format!(
                                "received an unexpected `[DONE]` sentinel while waiting for {}",
                                marker.description()
                            )))?;
                        }
                        terminal = true;
                        break;
                    }
                    SseEvent::Data(data) => {
                        let json = parse_event_json(&data)?;
                        let event = extract(&json)?;
                        if let Some(text) = event.text
                            && !text.is_empty()
                        {
                            yield text;
                        }
                        if event.terminal {
                            terminal = true;
                            break;
                        }
                    }
                }
            }
            if terminal || eof {
                break;
            }
        }
        require_terminal(terminal, marker)?;
    })
}

/// Build a structured "object" stream from an SSE HTTP response, parsing and
/// validating the final buffer into `T`.
///
/// The streamed text is the model's (partial) JSON. After each delta the buffer is
/// repaired into valid JSON (best effort) and, when that succeeds and the snapshot
/// changed, a [`StreamedObject::Partial`] is yielded. When the stream ends the full
/// buffer is parsed and validated into `T` and yielded as
/// [`StreamedObject::Complete`].
pub(crate) fn object_stream<'a, T, Fut, F>(
    send: Fut,
    extract: F,
    marker: TerminalMarker,
) -> ObjectStream<'a, T>
where
    T: Instructor + DeserializeOwned + Send + 'a,
    Fut: Future<Output = Result<reqwest::Response>> + Send + 'a,
    F: Fn(&Value) -> Result<ProviderStreamEvent> + Send + 'a,
{
    object_stream_with(send, extract, marker, |raw: &str| {
        super::utils::parse_and_validate_response::<T>(raw).map_err(|(err, _ctx)| err)
    })
}

/// Like [`object_stream`], but with a caller-supplied `finalize` that turns the
/// complete raw buffer into the validated `T`. Used by providers (e.g. Gemini)
/// that must transform the response before deserializing.
pub(crate) fn object_stream_with<'a, T, Fut, F, Fin>(
    send: Fut,
    extract: F,
    marker: TerminalMarker,
    finalize: Fin,
) -> ObjectStream<'a, T>
where
    T: Send + 'a,
    Fut: Future<Output = Result<reqwest::Response>> + Send + 'a,
    F: Fn(&Value) -> Result<ProviderStreamEvent> + Send + 'a,
    Fin: FnOnce(&str) -> Result<T> + Send + 'a,
{
    Box::pin(try_stream! {
        let mut buf = String::new();
        let mut last_partial: Option<Value> = None;
        let mut deltas = sse_text_stream(send, extract, marker);

        while let Some(delta) = deltas.next().await {
            buf.push_str(&delta?);
            if let Some(partial) = complete_json(&buf)
                && last_partial.as_ref() != Some(&partial)
            {
                last_partial = Some(partial.clone());
                yield StreamedObject::Partial(partial);
            }
        }

        let value: T = finalize(buf.trim())?;
        yield StreamedObject::Complete(value);
    })
}

/// Extract the text delta from an OpenAI/Grok streaming chunk
/// (`{"choices":[{"delta":{"content":"..."}}]}`).
pub(crate) fn openai_stream_event(event: &Value) -> Result<ProviderStreamEvent> {
    if let Some(error) = event.get("error") {
        let error_type = error
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown_error");
        return Err(streaming_error(
            StreamErrorKind::ProviderStreamError,
            format!("OpenAI-compatible provider emitted an in-stream `{error_type}` event"),
        ));
    }
    let choices = event
        .get("choices")
        .ok_or_else(|| invalid_provider_event("OpenAI stream event was missing `choices`"))?
        .as_array()
        .ok_or_else(|| invalid_provider_event("OpenAI stream event `choices` was not an array"))?;
    let Some(choice) = choices.first() else {
        // OpenAI may emit a final usage-only chunk with an empty choices array.
        return Ok(ProviderStreamEvent::default());
    };
    let delta = choice
        .get("delta")
        .ok_or_else(|| invalid_provider_event("OpenAI stream choice was missing `delta`"))?
        .as_object()
        .ok_or_else(|| invalid_provider_event("OpenAI stream choice `delta` was not an object"))?;
    let text = match delta.get("content") {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(_) => {
            return Err(invalid_provider_event(
                "OpenAI stream delta `content` was not a string",
            ));
        }
    };
    Ok(ProviderStreamEvent {
        text,
        terminal: false,
    })
}

/// Extract the text delta from an Anthropic streaming event
/// (`{"type":"content_block_delta","delta":{"text":"..."}}`). Also accepts
/// `input_json_delta.partial_json`, used when streaming structured output.
pub(crate) fn anthropic_stream_event(event: &Value) -> Result<ProviderStreamEvent> {
    let event_type = event.get("type").and_then(Value::as_str).ok_or_else(|| {
        invalid_provider_event("Anthropic stream event was missing string `type`")
    })?;

    match event_type {
        "message_stop" => Ok(ProviderStreamEvent {
            text: None,
            terminal: true,
        }),
        "content_block_delta" => {
            let delta = event
                .get("delta")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    invalid_provider_event("Anthropic content delta was missing object `delta`")
                })?;
            let delta_type = delta.get("type").and_then(Value::as_str).ok_or_else(|| {
                invalid_provider_event("Anthropic content delta was missing string `type`")
            })?;
            let field = match delta_type {
                "text_delta" => "text",
                "input_json_delta" => "partial_json",
                // Anthropic asks clients to ignore unknown future event types.
                _ => return Ok(ProviderStreamEvent::default()),
            };
            let text = delta.get(field).and_then(Value::as_str).ok_or_else(|| {
                invalid_provider_event(format!(
                    "Anthropic `{delta_type}` was missing string `{field}`"
                ))
            })?;
            Ok(ProviderStreamEvent {
                text: Some(text.to_owned()),
                terminal: false,
            })
        }
        "error" => {
            let error_type = event
                .get("error")
                .and_then(|error| error.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("unknown_error");
            Err(streaming_error(
                StreamErrorKind::ProviderStreamError,
                format!("Anthropic emitted an in-stream `{error_type}` event"),
            ))
        }
        _ => Ok(ProviderStreamEvent::default()),
    }
}

/// Extract the text delta from a Gemini streaming chunk
/// (`{"candidates":[{"content":{"parts":[{"text":"..."}]}}]}`). Concatenates the
/// text of every part in the chunk.
pub(crate) fn gemini_stream_event(event: &Value) -> Result<ProviderStreamEvent> {
    if let Some(error) = event.get("error") {
        let status = error
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("UNKNOWN");
        return Err(streaming_error(
            StreamErrorKind::ProviderStreamError,
            format!("Gemini emitted an in-stream `{status}` error"),
        ));
    }
    let candidates = event
        .get("candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            invalid_provider_event("Gemini stream event was missing array `candidates`")
        })?;
    let candidate = candidates
        .first()
        .ok_or_else(|| invalid_provider_event("Gemini stream event had no candidates"))?;

    let terminal = match candidate.get("finishReason") {
        None | Some(Value::Null) => false,
        Some(Value::String(reason)) => !reason.is_empty(),
        Some(_) => {
            return Err(invalid_provider_event(
                "Gemini candidate `finishReason` was not a string",
            ));
        }
    };

    let mut text = String::new();
    if let Some(content) = candidate.get("content") {
        let content = content.as_object().ok_or_else(|| {
            invalid_provider_event("Gemini candidate `content` was not an object")
        })?;
        if let Some(parts) = content.get("parts") {
            let parts = parts.as_array().ok_or_else(|| {
                invalid_provider_event("Gemini candidate content `parts` was not an array")
            })?;
            for part in parts {
                if let Some(value) = part.get("text") {
                    let value = value.as_str().ok_or_else(|| {
                        invalid_provider_event("Gemini content part `text` was not a string")
                    })?;
                    text.push_str(value);
                }
            }
        }
    }

    Ok(ProviderStreamEvent {
        text: (!text.is_empty()).then_some(text),
        terminal,
    })
}

/// Repair a possibly-truncated JSON prefix into a parseable JSON value.
///
/// Returns `Some(value)` only when the repaired text actually parses, so callers
/// never see invalid JSON; when the prefix is too incomplete to safely complete
/// (e.g. a half-written number) it returns `None` and the caller simply waits for
/// more input. This is intended for emitting progressive snapshots of streamed
/// structured output — the authoritative final parse always uses the raw buffer.
pub(crate) fn complete_json(s: &str) -> Option<Value> {
    let repaired = repair_json(s)?;
    serde_json::from_str(&repaired).ok()
}

/// Best-effort completion of a truncated JSON prefix: close an open string, drop a
/// dangling key/comma, and close any open objects/arrays. The result is validated
/// by [`complete_json`] before use, so imperfect repairs are simply discarded.
fn repair_json(s: &str) -> Option<String> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    let mut out = String::with_capacity(s.len() + 8);
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for c in s.chars() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
        } else {
            match c {
                '"' => {
                    in_string = true;
                    out.push(c);
                }
                '{' => {
                    stack.push('{');
                    out.push(c);
                }
                '[' => {
                    stack.push('[');
                    out.push(c);
                }
                '}' => {
                    if stack.pop() != Some('{') {
                        return None;
                    }
                    out.push(c);
                }
                ']' => {
                    if stack.pop() != Some('[') {
                        return None;
                    }
                    out.push(c);
                }
                _ => out.push(c),
            }
        }
    }

    // A trailing incomplete escape (`...\`) inside a string: drop the backslash.
    if in_string && escaped {
        out.pop();
    }
    // Close an open string.
    if in_string {
        out.push('"');
    }

    // Trim trailing structural debris that can't be completed: a dangling comma,
    // or a dangling object key (`"key":` with no value yet).
    loop {
        let trimmed_len = out.trim_end().len();
        out.truncate(trimmed_len);
        if out.ends_with(',') {
            out.pop();
            continue;
        }
        if out.ends_with(':') {
            // Drop the dangling `"key":` back to the previous `{` or `,`.
            let cut = out.rfind(['{', ','])?;
            out.truncate(cut + 1);
            continue;
        }
        break;
    }

    // Close any still-open containers, innermost first.
    for &opener in stack.iter().rev() {
        out.push(if opener == '{' { '}' } else { ']' });
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decoder_emits_complete_data_event() {
        let mut d = SseDecoder::default();
        assert_eq!(
            d.push(b"data: {\"a\":1}\n\n").unwrap(),
            vec![SseEvent::Data("{\"a\":1}".to_string())]
        );
    }

    #[test]
    fn decoder_buffers_across_chunk_boundary() {
        let mut d = SseDecoder::default();
        assert_eq!(d.push(b"data: {\"hel").unwrap(), vec![]);
        assert_eq!(d.push(b"lo\":1").unwrap(), vec![]);
        assert_eq!(
            d.push(b"}\n\n").unwrap(),
            vec![SseEvent::Data("{\"hello\":1}".to_string())]
        );
    }

    #[test]
    fn decoder_handles_crlf_and_ignores_non_data_lines() {
        let mut d = SseDecoder::default();
        assert_eq!(
            d.push(b"event: message\r\ndata: {\"x\":1}\r\n\r\n: keep-alive\r\n")
                .unwrap(),
            vec![SseEvent::Data("{\"x\":1}".to_string())]
        );
    }

    #[test]
    fn decoder_recognizes_done_sentinel() {
        let mut d = SseDecoder::default();
        assert_eq!(d.push(b"data: [DONE]\n\n").unwrap(), vec![SseEvent::Done]);
    }

    #[test]
    fn openai_delta_extracts_content() {
        assert_eq!(
            openai_stream_event(&json!({"choices":[{"delta":{"content":"Hi"}}]})).unwrap(),
            ProviderStreamEvent {
                text: Some("Hi".to_string()),
                terminal: false,
            }
        );
        assert_eq!(
            openai_stream_event(&json!({"choices":[{"delta":{"role":"assistant"}}]})).unwrap(),
            ProviderStreamEvent::default()
        );
    }

    #[test]
    fn anthropic_delta_extracts_text_and_partial_json() {
        assert_eq!(
            anthropic_stream_event(
                &json!({"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}})
            )
            .unwrap(),
            ProviderStreamEvent {
                text: Some("Hi".to_string()),
                terminal: false,
            }
        );
        assert_eq!(
            anthropic_stream_event(
                &json!({"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"a\":"}})
            )
            .unwrap(),
            ProviderStreamEvent {
                text: Some("{\"a\":".to_string()),
                terminal: false,
            }
        );
        assert_eq!(
            anthropic_stream_event(&json!({"type":"message_start"})).unwrap(),
            ProviderStreamEvent::default()
        );
        assert_eq!(
            anthropic_stream_event(&json!({"type":"message_stop"})).unwrap(),
            ProviderStreamEvent {
                text: None,
                terminal: true,
            }
        );
    }

    #[test]
    fn gemini_delta_concatenates_parts() {
        assert_eq!(
            gemini_stream_event(
                &json!({"candidates":[{"content":{"parts":[{"text":"a"},{"text":"b"}]}}]})
            )
            .unwrap(),
            ProviderStreamEvent {
                text: Some("ab".to_string()),
                terminal: false,
            }
        );
        assert_eq!(
            gemini_stream_event(
                &json!({"candidates":[{"content":{"parts":[{"text":"c"}]},"finishReason":"STOP"}]})
            )
            .unwrap(),
            ProviderStreamEvent {
                text: Some("c".to_string()),
                terminal: true,
            }
        );
        assert_eq!(
            gemini_stream_event(
                &json!({"candidates":[{"content":{"role":"model"},"finishReason":"STOP"}]})
            )
            .unwrap(),
            ProviderStreamEvent {
                text: None,
                terminal: true,
            }
        );
    }

    #[test]
    fn provider_adapters_reject_malformed_known_events_and_in_band_errors() {
        assert!(matches!(
            openai_stream_event(&json!({"choices":[{"delta":{"content":42}}]})),
            Err(RStructorError::StreamingError {
                kind: StreamErrorKind::InvalidProviderEvent,
                ..
            })
        ));
        assert!(matches!(
            anthropic_stream_event(
                &json!({"type":"content_block_delta","delta":{"type":"text_delta","text":42}})
            ),
            Err(RStructorError::StreamingError {
                kind: StreamErrorKind::InvalidProviderEvent,
                ..
            })
        ));
        assert!(matches!(
            anthropic_stream_event(
                &json!({"type":"error","error":{"type":"overloaded_error","message":"overloaded"}})
            ),
            Err(RStructorError::StreamingError {
                kind: StreamErrorKind::ProviderStreamError,
                ..
            })
        ));
        assert!(matches!(
            openai_stream_event(
                &json!({"error":{"type":"server_error","message":"internal details"}})
            ),
            Err(RStructorError::StreamingError {
                kind: StreamErrorKind::ProviderStreamError,
                ..
            })
        ));
        assert!(matches!(
            gemini_stream_event(&json!({"candidates":[{"finishReason":42}]})),
            Err(RStructorError::StreamingError {
                kind: StreamErrorKind::InvalidProviderEvent,
                ..
            })
        ));
        assert!(matches!(
            gemini_stream_event(&json!({"candidates":[{"content":{"parts":42}}]})),
            Err(RStructorError::StreamingError {
                kind: StreamErrorKind::InvalidProviderEvent,
                ..
            })
        ));
        assert!(matches!(
            gemini_stream_event(&json!({"candidates":[{"content":"invalid"}]})),
            Err(RStructorError::StreamingError {
                kind: StreamErrorKind::InvalidProviderEvent,
                ..
            })
        ));
        assert!(matches!(
            gemini_stream_event(
                &json!({"error":{"code":503,"status":"UNAVAILABLE","message":"details"}})
            ),
            Err(RStructorError::StreamingError {
                kind: StreamErrorKind::ProviderStreamError,
                ..
            })
        ));
    }

    #[test]
    fn stream_errors_do_not_echo_provider_or_model_content() {
        let provider_secret = "sensitive upstream details";
        let error = anthropic_stream_event(
            &json!({"type":"error","error":{"type":"api_error","message":provider_secret}}),
        )
        .unwrap_err();
        assert!(!error.to_string().contains(provider_secret));

        let model_content = "sensitive-position-token";
        let mut streamer = JsonArrayStreamer::default();
        let error = streamer
            .push_str(&format!(r#"{{"items":[{model_content},0]}}"#))
            .expect_err("invalid item should fail");
        assert!(!error.to_string().contains(model_content));
    }

    #[test]
    fn complete_json_closes_open_string_and_object() {
        assert_eq!(
            complete_json(r#"{"name": "Ali"#).unwrap(),
            json!({"name": "Ali"})
        );
    }

    #[test]
    fn complete_json_drops_dangling_key_and_comma() {
        assert_eq!(complete_json(r#"{"a": 1, "b":"#).unwrap(), json!({"a": 1}));
        assert_eq!(complete_json(r#"{"a": 1, "#).unwrap(), json!({"a": 1}));
        assert_eq!(complete_json(r#"{"a": 1,"#).unwrap(), json!({"a": 1}));
    }

    #[test]
    fn complete_json_closes_nested_and_arrays() {
        assert_eq!(
            complete_json(r#"{"items":[{"x":1},{"x":2"#).unwrap(),
            json!({"items":[{"x":1},{"x":2}]})
        );
        assert_eq!(complete_json(r#"[1, 2, 3"#).unwrap(), json!([1, 2, 3]));
        assert_eq!(complete_json(r#"[1, 2, "#).unwrap(), json!([1, 2]));
    }

    #[test]
    fn complete_json_skips_incomplete_primitive() {
        // A half-written number/keyword can't be safely completed → None.
        assert!(complete_json(r#"{"a": tr"#).is_none());
        assert!(complete_json(r#"{"a": 12."#).is_none());
        assert!(complete_json("").is_none());
    }

    #[test]
    fn complete_json_handles_escapes() {
        assert_eq!(
            complete_json(r#"{"s": "line\"#).unwrap(),
            json!({"s": "line"})
        );
        assert_eq!(
            complete_json(r#"{"s": "a\nb"#).unwrap(),
            json!({"s": "a\nb"})
        );
    }

    #[test]
    fn complete_json_progressive_prefixes_converge() {
        let full = r#"{"name":"Alice","age":30,"tags":["x","y"]}"#;
        // Every prefix either yields None or a valid JSON value, and the full
        // string yields the exact object.
        for i in 1..=full.len() {
            if let Some(v) = complete_json(&full[..i]) {
                assert!(v.is_object() || v.is_array());
            }
        }
        assert_eq!(
            complete_json(full).unwrap(),
            json!({"name":"Alice","age":30,"tags":["x","y"]})
        );
    }

    // --- SSE decoder edge cases ---

    #[test]
    fn decoder_splits_crlf_across_chunks() {
        // A `\r` arrives at the end of one chunk and the `\n` (plus the blank
        // separator) only in the next: the data line must not be emitted until
        // its terminating newline is seen.
        let mut d = SseDecoder::default();
        assert_eq!(d.push(b"data:{a}\r").unwrap(), vec![]);
        assert_eq!(
            d.push(b"\n\r\n").unwrap(),
            vec![SseEvent::Data("{a}".to_string())]
        );
    }

    #[test]
    fn decoder_emits_multiple_events_in_one_chunk_in_order() {
        let mut d = SseDecoder::default();
        assert_eq!(
            d.push(b"data:{a}\n\ndata:{b}\n\n").unwrap(),
            vec![
                SseEvent::Data("{a}".to_string()),
                SseEvent::Data("{b}".to_string()),
            ]
        );
    }

    #[test]
    fn decoder_reassembles_utf8_multibyte_split_across_chunks() {
        // The euro sign U+20AC is the three bytes E2 82 AC; split it across two
        // chunks and confirm the decoder reassembles a single intact code point.
        let mut d = SseDecoder::default();
        assert_eq!(d.push(b"data:\xe2\x82").unwrap(), vec![]);
        let events = d.push(b"\xac\n\n").unwrap();
        assert_eq!(events, vec![SseEvent::Data("\u{20AC}".to_string())]);
        if let SseEvent::Data(s) = &events[0] {
            assert_eq!(s.chars().next(), Some('\u{20AC}'));
        }
    }

    #[test]
    fn decoder_handles_done_and_data_in_same_push() {
        let mut d = SseDecoder::default();
        assert_eq!(
            d.push(b"data:[DONE]\n\ndata:{a}\n\n").unwrap(),
            vec![SseEvent::Done, SseEvent::Data("{a}".to_string())]
        );
    }

    #[test]
    fn decoder_preserves_empty_and_whitespace_data_events() {
        let mut d = SseDecoder::default();
        assert_eq!(
            d.push(b"data:\n\n").unwrap(),
            vec![SseEvent::Data(String::new())]
        );
        let mut d = SseDecoder::default();
        assert_eq!(
            d.push(b"data:   \n\n").unwrap(),
            vec![SseEvent::Data("  ".to_string())]
        );
    }

    #[test]
    fn decoder_done_sentinel_is_case_sensitive() {
        // Only the exact `[DONE]` sentinel ends the stream; lowercase is data.
        let mut d = SseDecoder::default();
        assert_eq!(
            d.push(b"data:[done]\n\n").unwrap(),
            vec![SseEvent::Data("[done]".to_string())]
        );
    }

    #[test]
    fn decoder_joins_multiline_data_fields_per_sse_spec() {
        let mut d = SseDecoder::default();
        assert_eq!(
            d.push(b"data: {\"a\":\ndata: 1}\n\n").unwrap(),
            vec![SseEvent::Data("{\"a\":\n1}".to_string())]
        );
    }

    #[test]
    fn decoder_accepts_cr_only_line_endings() {
        let mut d = SseDecoder::default();
        let mut events = d.push(b"data: {\"a\":1}\r\rdata: [DONE]\r\r").unwrap();
        events.extend(d.finish().unwrap());
        assert_eq!(
            events,
            vec![SseEvent::Data("{\"a\":1}".to_string()), SseEvent::Done,]
        );
    }

    #[test]
    fn decoder_dispatches_a_final_event_at_eof_without_newline() {
        let mut d = SseDecoder::default();
        assert!(d.push(b"data: {\"a\":1}").unwrap().is_empty());
        assert_eq!(
            d.finish().unwrap(),
            vec![SseEvent::Data("{\"a\":1}".to_string())]
        );
    }

    #[test]
    fn decoder_rejects_invalid_utf8_without_lossy_replacement() {
        let mut d = SseDecoder::default();
        let error = d
            .push(b"data: \xff\n\n")
            .expect_err("invalid UTF-8 must fail the stream");
        assert!(matches!(
            error,
            RStructorError::StreamingError {
                kind: StreamErrorKind::InvalidEventEncoding,
                ..
            }
        ));
    }

    #[test]
    fn decoder_is_invariant_across_every_byte_split() {
        let fixture = b"\xef\xbb\xbfdata: {\"symbol\":\"EUR/USD\"}\r\n\r\ndata: [DONE]\n\n";
        let expected = vec![
            SseEvent::Data("{\"symbol\":\"EUR/USD\"}".to_string()),
            SseEvent::Done,
        ];

        for split in 0..=fixture.len() {
            let mut decoder = SseDecoder::default();
            let mut events = decoder.push(&fixture[..split]).unwrap();
            events.extend(decoder.push(&fixture[split..]).unwrap());
            events.extend(decoder.finish().unwrap());
            assert_eq!(events, expected, "failed at byte split {split}");
        }
    }

    #[test]
    fn malformed_event_json_has_a_machine_readable_error_kind() {
        let error = parse_event_json(r#"{"choices":["#).unwrap_err();
        assert!(matches!(
            error,
            RStructorError::StreamingError {
                kind: StreamErrorKind::InvalidEventJson,
                ..
            }
        ));
        assert!(!error.to_string().contains(r#"{"choices":["#));
    }

    // --- complete_json edge cases ---

    #[test]
    fn complete_json_rejects_truncated_unicode_escape() {
        // The braces/quotes are balanced so repair produces a string, but the
        // truncated `\u00` escape makes it invalid JSON → None.
        assert!(complete_json(r#"{"s":"\u00"#).is_none());
    }

    #[test]
    fn complete_json_rejects_unbalanced_or_extra_closers() {
        // Extra `}` and a mismatched `]` can't be repaired.
        assert!(complete_json(r#"{"a":1}}"#).is_none());
        assert!(complete_json(r#"{"a":1]"#).is_none());
    }

    #[test]
    fn complete_json_handles_odd_and_even_trailing_backslashes() {
        // Even backslashes: `a\\` is a complete escaped backslash; the value is a
        // single backslash. (`json!({"s": "a\\"})` is the string `a\`.)
        assert_eq!(complete_json(r#"{"s":"a\\"#).unwrap(), json!({"s": "a\\"}));
        // Odd backslashes: the dangling final `\` is an incomplete escape and is
        // dropped, leaving the same completed value.
        assert_eq!(complete_json(r#"{"s":"a\\\"#).unwrap(), json!({"s": "a\\"}));
    }

    #[test]
    fn complete_json_rejects_dangling_minus_but_allows_negative_exponent() {
        assert!(complete_json(r#"{"a":-"#).is_none());
        assert_eq!(
            complete_json(r#"{"a":-1.2e10"#).unwrap(),
            json!({"a": -1.2e10})
        );
    }

    #[test]
    fn complete_json_rejects_dangling_colon_without_container() {
        // A dangling key/colon with no surrounding `{`/`,` to cut back to → None.
        assert!(complete_json(r#""key":"#).is_none());
        assert!(complete_json("x:").is_none());
    }

    #[test]
    fn complete_json_passes_top_level_scalars_through() {
        assert_eq!(complete_json("42").unwrap(), json!(42));
        assert_eq!(complete_json("true").unwrap(), json!(true));
        assert_eq!(complete_json(r#""hello"#).unwrap(), json!("hello"));
        assert_eq!(complete_json("[1,2,3]").unwrap(), json!([1, 2, 3]));
    }

    #[test]
    fn complete_json_trims_trailing_whitespace_and_comma() {
        assert_eq!(complete_json("{\"a\":1,  \n  ").unwrap(), json!({"a": 1}));
    }

    // --- StreamedObject helper ---

    #[test]
    fn streamed_object_complete_accessor() {
        assert_eq!(StreamedObject::Complete(42).complete(), Some(42));
        assert_eq!(
            StreamedObject::<i32>::Partial(json!({"a": 1})).complete(),
            None
        );
    }
}

/// A boxed stream of fully-parsed items, one per element of a streamed JSON array.
pub type ItemStream<'a, T> = Pin<Box<dyn Stream<Item = Result<T>> + Send + 'a>>;

/// Incrementally extracts complete top-level elements from a streaming JSON array.
///
/// Used by `materialize_iter` to yield each element of a list as soon as it is
/// fully received, rather than buffering the whole array. The model is asked for a
/// wrapper object `{"items": [ ... ]}`; the streamer skips to the first `[` (the
/// `items` array) and then emits each complete element (object or scalar) as a
/// `serde_json::Value`.
#[derive(Debug, Default, PartialEq, Eq)]
enum ArrayState {
    #[default]
    Seeking,
    Streaming,
    ClosingEnvelope,
    Complete,
}

#[derive(Default)]
pub(crate) struct JsonArrayStreamer {
    state: ArrayState,
    prefix_in_string: bool,
    prefix_escaped: bool,
    envelope: Vec<char>,
    containers: Vec<char>,
    in_string: bool,
    escaped: bool,
    started_element: bool,
    just_saw_comma: bool,
    next_index: usize,
    current: String,
}

struct ArrayPush {
    values: Vec<Value>,
    error: Option<RStructorError>,
}

#[cfg(test)]
impl ArrayPush {
    fn unwrap(self) -> Vec<Value> {
        if let Some(error) = self.error {
            panic!("called ArrayPush::unwrap() on an error: {error}");
        }
        self.values
    }

    fn expect_err(self, message: &str) -> RStructorError {
        self.error.unwrap_or_else(|| panic!("{message}"))
    }
}

impl JsonArrayStreamer {
    fn push_str(&mut self, s: &str) -> ArrayPush {
        let mut values = Vec::new();
        for c in s.chars() {
            if let Err(error) = self.push_char(c, &mut values) {
                return ArrayPush {
                    values,
                    error: Some(error),
                };
            }
        }
        ArrayPush {
            values,
            error: None,
        }
    }

    fn push_char(&mut self, c: char, out: &mut Vec<Value>) -> Result<()> {
        match self.state {
            ArrayState::Seeking => {
                self.seek_array(c);
                return Ok(());
            }
            ArrayState::ClosingEnvelope => {
                return self.close_envelope(c);
            }
            ArrayState::Complete => {
                if !c.is_whitespace() {
                    return Err(streaming_error(
                        StreamErrorKind::InvalidArrayEnvelope,
                        "unexpected data followed the completed streamed array envelope",
                    ));
                }
                return Ok(());
            }
            ArrayState::Streaming => {}
        }

        if self.in_string {
            self.current.push(c);
            if self.escaped {
                self.escaped = false;
            } else if c == '\\' {
                self.escaped = true;
            } else if c == '"' {
                self.in_string = false;
            }
            return Ok(());
        }
        match c {
            '"' => {
                self.start_element();
                self.in_string = true;
                self.current.push(c);
            }
            '{' => {
                self.start_element();
                self.containers.push('}');
                self.current.push(c);
            }
            '[' => {
                self.start_element();
                self.containers.push(']');
                self.current.push(c);
            }
            ']' => {
                if let Some(expected) = self.containers.last().copied() {
                    if expected != ']' {
                        return Err(self.invalid_element(format!(
                            "mismatched closing delimiter `]`; expected `{expected}`"
                        )));
                    }
                    self.containers.pop();
                    self.current.push(c);
                } else {
                    if self.just_saw_comma {
                        return Err(
                            self.invalid_element("trailing commas are not valid JSON array syntax")
                        );
                    }
                    if let Some(value) = self.finish_element()? {
                        out.push(value);
                    }
                    self.state = if self.envelope.is_empty() {
                        ArrayState::Complete
                    } else {
                        ArrayState::ClosingEnvelope
                    };
                }
            }
            '}' => {
                let Some(expected) = self.containers.last().copied() else {
                    return Err(
                        self.invalid_element("unexpected `}` before the streamed array was closed")
                    );
                };
                if expected != '}' {
                    return Err(self.invalid_element(format!(
                        "mismatched closing delimiter `}}`; expected `{expected}`"
                    )));
                }
                self.containers.pop();
                self.current.push(c);
            }
            ',' if self.containers.is_empty() => {
                let Some(value) = self.finish_element()? else {
                    return Err(self.invalid_element("expected an array element before `,`"));
                };
                out.push(value);
                self.just_saw_comma = true;
            }
            c if c.is_whitespace() && !self.started_element => {}
            _ => {
                self.start_element();
                self.current.push(c);
            }
        }
        Ok(())
    }

    /// Verify that HTTP EOF or `[DONE]` arrived after a complete array.
    pub(crate) fn finish(&self) -> Result<()> {
        match self.state {
            ArrayState::Seeking => Err(streaming_error(
                StreamErrorKind::MissingArray,
                "structured stream ended before an `items` array was received",
            )),
            ArrayState::Streaming => Err(streaming_error(
                StreamErrorKind::IncompleteArray {
                    next_index: self.next_index,
                },
                format!(
                    "structured stream ended before the array and element {} were complete",
                    self.next_index
                ),
            )),
            ArrayState::ClosingEnvelope => Err(streaming_error(
                StreamErrorKind::IncompleteArrayEnvelope,
                "streamed items array closed, but its surrounding JSON object did not",
            )),
            ArrayState::Complete => Ok(()),
        }
    }

    fn seek_array(&mut self, c: char) {
        if self.prefix_in_string {
            if self.prefix_escaped {
                self.prefix_escaped = false;
            } else if c == '\\' {
                self.prefix_escaped = true;
            } else if c == '"' {
                self.prefix_in_string = false;
            }
        } else if c == '"' {
            self.prefix_in_string = true;
        } else if c == '{' {
            self.envelope.push('}');
        } else if c == '}' {
            if self.envelope.last() == Some(&'}') {
                self.envelope.pop();
            }
        } else if c == '[' {
            self.state = ArrayState::Streaming;
        }
    }

    fn close_envelope(&mut self, c: char) -> Result<()> {
        if c.is_whitespace() {
            return Ok(());
        }
        let Some(expected) = self.envelope.last().copied() else {
            return Err(streaming_error(
                StreamErrorKind::InvalidArrayEnvelope,
                "unexpected data followed the streamed items array",
            ));
        };
        if c != expected {
            return Err(streaming_error(
                StreamErrorKind::InvalidArrayEnvelope,
                format!("unexpected `{c}` after streamed items array; expected `{expected}`"),
            ));
        }
        self.envelope.pop();
        if self.envelope.is_empty() {
            self.state = ArrayState::Complete;
        }
        Ok(())
    }

    fn start_element(&mut self) {
        self.started_element = true;
        self.just_saw_comma = false;
    }

    fn finish_element(&mut self) -> Result<Option<Value>> {
        let text = std::mem::take(&mut self.current);
        self.started_element = false;
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        let value = serde_json::from_str(trimmed).map_err(|error| {
            self.invalid_element(format!(
                "element {} was not valid JSON at line {}, column {}: {}",
                self.next_index,
                error.line(),
                error.column(),
                error
            ))
        })?;
        self.next_index += 1;
        Ok(Some(value))
    }

    fn invalid_element(&self, message: impl Into<Box<str>>) -> RStructorError {
        streaming_error(
            StreamErrorKind::InvalidArrayElement {
                index: self.next_index,
            },
            message,
        )
    }
}

/// Build a streaming-array request: yields each element of the response's `items`
/// array as a validated `T`, as soon as it is fully received.
///
/// `finalize_item` turns each element's `serde_json::Value` into a validated `T`
/// (deserialize + validate, plus any provider-specific transform).
pub(crate) fn iter_stream<'a, T, Fut, F, Fin>(
    send: Fut,
    extract: F,
    marker: TerminalMarker,
    finalize_item: Fin,
) -> ItemStream<'a, T>
where
    T: Send + 'a,
    Fut: Future<Output = Result<reqwest::Response>> + Send + 'a,
    F: Fn(&Value) -> Result<ProviderStreamEvent> + Send + 'a,
    Fin: Fn(Value) -> Result<T> + Send + 'a,
{
    Box::pin(try_stream! {
        let mut array = JsonArrayStreamer::default();
        let mut deltas = sse_text_stream(send, extract, marker);

        while let Some(delta) = deltas.next().await {
            let pushed = array.push_str(&delta?);
            for element in pushed.values {
                yield finalize_item(element)?;
            }
            if let Some(error) = pushed.error {
                Err(error)?;
            }
        }
        array.finish()?;
    })
}

/// Default per-element finalizer for [`iter_stream`]: deserialize a streamed array
/// element into `T` and run its validation.
pub(crate) fn finalize_item<T: Instructor + DeserializeOwned>(value: Value) -> Result<T> {
    let item: T = crate::decode::output_from_value(value)?;
    item.validate()?;
    Ok(item)
}

/// Wrap a (prepared) item schema into the `{ "items": [ <item> ] }` object schema
/// used for streaming arrays. `strict` adds `additionalProperties: false`
/// (OpenAI/Anthropic); Gemini passes `false`.
pub(crate) fn array_wrapper_schema(item_schema: Value, strict: bool) -> Value {
    let mut wrapper = serde_json::json!({
        "type": "object",
        "properties": { "items": { "type": "array", "items": item_schema } },
        "required": ["items"],
    });
    if strict {
        wrapper["additionalProperties"] = Value::Bool(false);
    }
    wrapper
}

#[cfg(test)]
mod array_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn streams_object_elements_as_they_complete() {
        let mut s = JsonArrayStreamer::default();
        // Wrapper prefix + first complete element, split across pushes.
        assert_eq!(s.push_str(r#"{"items":["#).unwrap(), Vec::<Value>::new());
        assert_eq!(
            s.push_str(r#"{"n":1},{"n":2}"#).unwrap(),
            vec![json!({"n":1})]
        );
        assert_eq!(
            s.push_str(r#",{"n":3}]}"#).unwrap(),
            vec![json!({"n":2}), json!({"n":3})]
        );
        s.finish().unwrap();
    }

    #[test]
    fn handles_scalars_strings_and_nesting() {
        let mut s = JsonArrayStreamer::default();
        let got = s
            .push_str(r#"{"items":[1, "a,b", {"x":[1,2]}, true]}"#)
            .unwrap();
        assert_eq!(
            got,
            vec![json!(1), json!("a,b"), json!({"x":[1,2]}), json!(true)]
        );
    }

    #[test]
    fn ignores_strings_containing_brackets_before_array() {
        let mut s = JsonArrayStreamer::default();
        // The first '[' is the items array; nothing emitted until an element completes.
        assert_eq!(
            s.push_str(r#"{"items":[{"v":"#).unwrap(),
            Vec::<Value>::new()
        );
    }

    // --- JsonArrayStreamer edge cases ---

    #[test]
    fn handles_escaped_quotes_in_string_element() {
        let mut s = JsonArrayStreamer::default();
        assert_eq!(
            s.push_str(r#"{"items":["he said \"hi\"","x"]}"#).unwrap(),
            vec![json!("he said \"hi\""), json!("x")]
        );
    }

    #[test]
    fn handles_string_containing_closing_bracket() {
        let mut s = JsonArrayStreamer::default();
        // The `]` inside the string must not be treated as the array terminator.
        assert_eq!(
            s.push_str(r#"{"items":["a]b","c"]}"#).unwrap(),
            vec![json!("a]b"), json!("c")]
        );
    }

    #[test]
    fn handles_array_of_arrays_elements() {
        let mut s = JsonArrayStreamer::default();
        assert_eq!(
            s.push_str(r#"{"items":[[1,2],[3,4]]}"#).unwrap(),
            vec![json!([1, 2]), json!([3, 4])]
        );
    }

    #[test]
    fn handles_null_and_bare_top_level_array() {
        let mut s = JsonArrayStreamer::default();
        assert_eq!(
            s.push_str(r#"{"items":[null,1]}"#).unwrap(),
            vec![json!(null), json!(1)]
        );
        // A bare top-level array (no `{"items": ...}` wrapper): the first `[` is
        // still taken as the array start.
        let mut s = JsonArrayStreamer::default();
        assert_eq!(
            s.push_str(r#"[1,2,3]"#).unwrap(),
            vec![json!(1), json!(2), json!(3)]
        );
    }

    #[test]
    fn rejects_invalid_and_missing_elements() {
        let mut s = JsonArrayStreamer::default();
        let error = s
            .push_str(r#"{"items":[1abc,2]}"#)
            .expect_err("malformed elements must not be dropped");
        assert!(matches!(
            error,
            RStructorError::StreamingError {
                kind: StreamErrorKind::InvalidArrayElement { index: 0 },
                ..
            }
        ));

        let mut s = JsonArrayStreamer::default();
        assert!(matches!(
            s.push_str(r#"{"items":[,1]}"#).error,
            Some(RStructorError::StreamingError {
                kind: StreamErrorKind::InvalidArrayElement { index: 0 },
                ..
            })
        ));

        let mut s = JsonArrayStreamer::default();
        assert!(matches!(
            s.push_str(r#"{"items":[1,]}"#).error,
            Some(RStructorError::StreamingError {
                kind: StreamErrorKind::InvalidArrayElement { index: 1 },
                ..
            })
        ));

        let mut s = JsonArrayStreamer::default();
        let pushed = s.push_str(r#"{"items":[1,not-json,2]}"#);
        assert_eq!(pushed.values, vec![json!(1)]);
        assert!(matches!(
            pushed.error,
            Some(RStructorError::StreamingError {
                kind: StreamErrorKind::InvalidArrayElement { index: 1 },
                ..
            })
        ));
    }

    #[test]
    fn empty_items_array_yields_nothing() {
        let mut s = JsonArrayStreamer::default();
        assert_eq!(s.push_str(r#"{"items":[]}"#).unwrap(), Vec::<Value>::new());
        s.finish().unwrap();
    }

    #[test]
    fn element_split_across_push_str_calls() {
        // Scalar split mid-number: the two halves concatenate into one element.
        let mut s = JsonArrayStreamer::default();
        assert_eq!(s.push_str(r#"{"items":[12"#).unwrap(), Vec::<Value>::new());
        assert_eq!(
            s.push_str(r#"34,5]}"#).unwrap(),
            vec![json!(1234), json!(5)]
        );
    }

    #[test]
    fn escape_flag_persists_across_push_str_calls() {
        // The escaping backslash ends chunk 1 (inside a string); the escaped flag
        // must persist so the following `\b` decodes to a single literal backslash
        // + `b`, not a control escape, and the element parses correctly.
        let mut s = JsonArrayStreamer::default();
        assert_eq!(s.push_str(r#"{"items":["a\"#).unwrap(), Vec::<Value>::new());
        assert_eq!(
            s.push_str(r#"\b","c"]}"#).unwrap(),
            vec![json!("a\\b"), json!("c")]
        );
    }

    #[test]
    fn does_not_reenter_after_array_close() {
        let mut s = JsonArrayStreamer::default();
        assert_eq!(s.push_str(r#"{"items":[1]}"#).unwrap(), vec![json!(1)]);
        assert!(matches!(
            s.push_str(r#"[9,9]"#).error,
            Some(RStructorError::StreamingError {
                kind: StreamErrorKind::InvalidArrayEnvelope,
                ..
            })
        ));
    }

    #[test]
    fn ignores_brackets_inside_wrapper_strings_while_seeking() {
        let mut s = JsonArrayStreamer::default();
        assert_eq!(
            s.push_str(r#"{"note":"risk [gross]","items":[1,2]}"#)
                .unwrap(),
            vec![json!(1), json!(2)]
        );
        s.finish().unwrap();
    }

    #[test]
    fn finish_distinguishes_missing_and_incomplete_arrays() {
        let mut missing = JsonArrayStreamer::default();
        missing.push_str(r#"{"items":"not an array"}"#).unwrap();
        assert!(matches!(
            missing.finish(),
            Err(RStructorError::StreamingError {
                kind: StreamErrorKind::MissingArray,
                ..
            })
        ));

        let mut incomplete = JsonArrayStreamer::default();
        incomplete
            .push_str(r#"{"items":[{"symbol":"AAPL"},"#)
            .unwrap();
        assert!(matches!(
            incomplete.finish(),
            Err(RStructorError::StreamingError {
                kind: StreamErrorKind::IncompleteArray { next_index: 1 },
                ..
            })
        ));

        let mut incomplete_envelope = JsonArrayStreamer::default();
        incomplete_envelope
            .push_str(r#"{"items":[{"symbol":"AAPL"}]"#)
            .unwrap();
        assert!(matches!(
            incomplete_envelope.finish(),
            Err(RStructorError::StreamingError {
                kind: StreamErrorKind::IncompleteArrayEnvelope,
                ..
            })
        ));
    }

    #[test]
    fn array_streamer_is_invariant_across_every_utf8_byte_split() {
        let fixture =
            r#"{"items":[{"symbol":"AAPL","quantity":125000},{"symbol":"ES","quantity":-240}]}"#;
        let expected = vec![
            json!({"symbol":"AAPL","quantity":125000}),
            json!({"symbol":"ES","quantity":-240}),
        ];

        for split in 0..=fixture.len() {
            if !fixture.is_char_boundary(split) {
                continue;
            }
            let mut streamer = JsonArrayStreamer::default();
            let mut values = streamer.push_str(&fixture[..split]).unwrap();
            values.extend(streamer.push_str(&fixture[split..]).unwrap());
            streamer.finish().unwrap();
            assert_eq!(values, expected, "failed at byte split {split}");
        }
    }

    // --- array_wrapper_schema ---

    #[test]
    fn array_wrapper_schema_strict_adds_additional_properties_and_required() {
        let item = json!({"type": "object"});
        let wrapper = array_wrapper_schema(item.clone(), true);
        assert_eq!(wrapper["additionalProperties"], json!(false));
        assert_eq!(wrapper["required"], json!(["items"]));
        assert_eq!(wrapper["type"], json!("object"));
        assert_eq!(wrapper["properties"]["items"]["type"], json!("array"));
        assert_eq!(wrapper["properties"]["items"]["items"], item);
    }

    #[test]
    fn array_wrapper_schema_non_strict_omits_additional_properties() {
        let item = json!({"type": "string"});
        let wrapper = array_wrapper_schema(item.clone(), false);
        assert!(wrapper.get("additionalProperties").is_none());
        // `required` and the array shape are still present in non-strict mode.
        assert_eq!(wrapper["required"], json!(["items"]));
        assert_eq!(wrapper["properties"]["items"]["items"], item);
    }

    // --- finalize_item failure branches ---

    #[cfg(feature = "derive")]
    #[test]
    fn finalize_item_validation_and_deserialize_failures() {
        use crate::Instructor;
        use serde::{Deserialize, Serialize};

        #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
        #[llm(validate = "validate_ticket")]
        struct Ticket {
            title: String,
            priority: u8,
        }

        fn validate_ticket(t: &Ticket) -> crate::Result<()> {
            if !(1..=5).contains(&t.priority) {
                return Err(RStructorError::ValidationError(format!(
                    "priority must be 1-5, got {}",
                    t.priority
                )));
            }
            Ok(())
        }

        // Deserializes fine but fails validation → ValidationError from validate().
        let err = finalize_item::<Ticket>(json!({"title": "x", "priority": 99}))
            .expect_err("priority 99 should fail validation");
        assert!(
            matches!(err, RStructorError::ValidationError(_)),
            "expected ValidationError, got {err:?}"
        );

        // Wrong type for `title` reports the exact output path.
        let err = finalize_item::<Ticket>(json!({"title": 123, "priority": 1}))
            .expect_err("non-string title should fail deserialization");
        assert!(
            matches!(
                err,
                RStructorError::OutputDecodeError { ref path, .. } if path == "$.title"
            ),
            "expected path-aware OutputDecodeError, got {err:?}"
        );

        // Sanity: a valid element succeeds.
        let ok = finalize_item::<Ticket>(json!({"title": "x", "priority": 3})).unwrap();
        assert_eq!(
            ok,
            Ticket {
                title: "x".into(),
                priority: 3
            }
        );
    }
}

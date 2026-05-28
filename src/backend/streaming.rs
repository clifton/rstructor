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

use crate::error::{RStructorError, Result};
use crate::model::Instructor;

/// A boxed stream of text deltas. Each item is either an incremental piece of the
/// model's text output or a transport/decode error.
pub type TextStream<'a> = Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>>;

/// A boxed stream of [`StreamedObject`] items for a streaming structured request.
pub type ObjectStream<'a, T> = Pin<Box<dyn Stream<Item = Result<StreamedObject<T>>> + Send + 'a>>;

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
#[derive(Default)]
pub(crate) struct SseDecoder {
    buf: Vec<u8>,
}

impl SseDecoder {
    /// Feed a chunk of bytes, returning any complete `data:` events it completed.
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(chunk);
        let mut events = Vec::new();

        while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = self.buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches(['\r', '\n']);

            // SSE: only `data:` fields carry content. Ignore `event:`, `id:`,
            // `retry:`, comment lines (`:`...), and blank separators.
            if let Some(rest) = line.strip_prefix("data:") {
                let data = rest.trim();
                if data == "[DONE]" {
                    events.push(SseEvent::Done);
                } else if !data.is_empty() {
                    events.push(SseEvent::Data(data.to_string()));
                }
            }
        }

        events
    }
}

/// Build a raw-text stream from an SSE HTTP response.
///
/// `send` is the (async) request that yields the streaming response; deferring it
/// lets this function return a `Stream` synchronously. `extract` pulls the
/// incremental text out of each parsed `data:` JSON event.
pub(crate) fn sse_text_stream<'a, Fut, F>(send: Fut, extract: F) -> TextStream<'a>
where
    Fut: Future<Output = Result<reqwest::Response>> + Send + 'a,
    F: Fn(&Value) -> Option<String> + Send + 'a,
{
    Box::pin(try_stream! {
        let response = send.await?;
        let mut bytes = response.bytes_stream();
        let mut decoder = SseDecoder::default();

        'outer: while let Some(chunk) = bytes.next().await {
            let chunk = chunk.map_err(RStructorError::from)?;
            for event in decoder.push(chunk.as_ref()) {
                match event {
                    SseEvent::Done => break 'outer,
                    SseEvent::Data(data) => {
                        if let Ok(json) = serde_json::from_str::<Value>(&data)
                            && let Some(text) = extract(&json)
                            && !text.is_empty()
                        {
                            yield text;
                        }
                    }
                }
            }
        }
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
pub(crate) fn object_stream<'a, T, Fut, F>(send: Fut, extract: F) -> ObjectStream<'a, T>
where
    T: Instructor + DeserializeOwned + Send + 'a,
    Fut: Future<Output = Result<reqwest::Response>> + Send + 'a,
    F: Fn(&Value) -> Option<String> + Send + 'a,
{
    object_stream_with(send, extract, |raw: &str| {
        super::utils::parse_and_validate_response::<T>(raw).map_err(|(err, _ctx)| err)
    })
}

/// Like [`object_stream`], but with a caller-supplied `finalize` that turns the
/// complete raw buffer into the validated `T`. Used by providers (e.g. Gemini)
/// that must transform the response before deserializing.
pub(crate) fn object_stream_with<'a, T, Fut, F, Fin>(
    send: Fut,
    extract: F,
    finalize: Fin,
) -> ObjectStream<'a, T>
where
    T: Send + 'a,
    Fut: Future<Output = Result<reqwest::Response>> + Send + 'a,
    F: Fn(&Value) -> Option<String> + Send + 'a,
    Fin: FnOnce(&str) -> Result<T> + Send + 'a,
{
    Box::pin(try_stream! {
        let response = send.await?;
        let mut bytes = response.bytes_stream();
        let mut decoder = SseDecoder::default();
        let mut buf = String::new();
        let mut last_partial: Option<Value> = None;

        'outer: while let Some(chunk) = bytes.next().await {
            let chunk = chunk.map_err(RStructorError::from)?;
            for event in decoder.push(chunk.as_ref()) {
                match event {
                    SseEvent::Done => break 'outer,
                    SseEvent::Data(data) => {
                        if let Ok(json) = serde_json::from_str::<Value>(&data)
                            && let Some(text) = extract(&json)
                        {
                            buf.push_str(&text);
                            if let Some(partial) = complete_json(&buf)
                                && last_partial.as_ref() != Some(&partial)
                            {
                                last_partial = Some(partial.clone());
                                yield StreamedObject::Partial(partial);
                            }
                        }
                    }
                }
            }
        }

        let value: T = finalize(buf.trim())?;
        yield StreamedObject::Complete(value);
    })
}

/// Extract the text delta from an OpenAI/Grok streaming chunk
/// (`{"choices":[{"delta":{"content":"..."}}]}`).
pub(crate) fn openai_delta(event: &Value) -> Option<String> {
    event
        .get("choices")?
        .get(0)?
        .get("delta")?
        .get("content")?
        .as_str()
        .map(str::to_owned)
}

/// Extract the text delta from an Anthropic streaming event
/// (`{"type":"content_block_delta","delta":{"text":"..."}}`). Also accepts
/// `input_json_delta.partial_json`, used when streaming structured output.
pub(crate) fn anthropic_delta(event: &Value) -> Option<String> {
    if event.get("type")?.as_str()? != "content_block_delta" {
        return None;
    }
    let delta = event.get("delta")?;
    delta
        .get("text")
        .and_then(Value::as_str)
        .or_else(|| delta.get("partial_json").and_then(Value::as_str))
        .map(str::to_owned)
}

/// Extract the text delta from a Gemini streaming chunk
/// (`{"candidates":[{"content":{"parts":[{"text":"..."}]}}]}`). Concatenates the
/// text of every part in the chunk.
pub(crate) fn gemini_delta(event: &Value) -> Option<String> {
    let parts = event
        .get("candidates")?
        .get(0)?
        .get("content")?
        .get("parts")?
        .as_array()?;

    let text: String = parts
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect();

    if text.is_empty() { None } else { Some(text) }
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
            if let Some(cut) = out.rfind(['{', ',']) {
                out.truncate(cut + 1);
            } else {
                return None;
            }
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
            d.push(b"data: {\"a\":1}\n\n"),
            vec![SseEvent::Data("{\"a\":1}".to_string())]
        );
    }

    #[test]
    fn decoder_buffers_across_chunk_boundary() {
        let mut d = SseDecoder::default();
        assert_eq!(d.push(b"data: {\"hel"), vec![]);
        assert_eq!(d.push(b"lo\":1"), vec![]);
        assert_eq!(
            d.push(b"}\n"),
            vec![SseEvent::Data("{\"hello\":1}".to_string())]
        );
    }

    #[test]
    fn decoder_handles_crlf_and_ignores_non_data_lines() {
        let mut d = SseDecoder::default();
        assert_eq!(
            d.push(b"event: message\r\ndata: {\"x\":1}\r\n\r\n: keep-alive\r\n"),
            vec![SseEvent::Data("{\"x\":1}".to_string())]
        );
    }

    #[test]
    fn decoder_recognizes_done_sentinel() {
        let mut d = SseDecoder::default();
        assert_eq!(d.push(b"data: [DONE]\n\n"), vec![SseEvent::Done]);
    }

    #[test]
    fn openai_delta_extracts_content() {
        assert_eq!(
            openai_delta(&json!({"choices":[{"delta":{"content":"Hi"}}]})),
            Some("Hi".to_string())
        );
        assert_eq!(
            openai_delta(&json!({"choices":[{"delta":{"role":"assistant"}}]})),
            None
        );
    }

    #[test]
    fn anthropic_delta_extracts_text_and_partial_json() {
        assert_eq!(
            anthropic_delta(
                &json!({"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}})
            ),
            Some("Hi".to_string())
        );
        assert_eq!(
            anthropic_delta(
                &json!({"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"a\":"}})
            ),
            Some("{\"a\":".to_string())
        );
        assert_eq!(anthropic_delta(&json!({"type":"message_start"})), None);
    }

    #[test]
    fn gemini_delta_concatenates_parts() {
        assert_eq!(
            gemini_delta(
                &json!({"candidates":[{"content":{"parts":[{"text":"a"},{"text":"b"}]}}]})
            ),
            Some("ab".to_string())
        );
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
}

//! Path-aware deserialization for structured model output and tool arguments.

use serde::de::DeserializeOwned;
#[cfg(any(feature = "streaming", feature = "tools", test))]
use serde_json::Value;
use serde_path_to_error::{Error as PathError, Path, Segment};

use crate::error::{RStructorError, Result};

#[derive(Clone, Copy)]
enum DecodeTarget {
    Output,
    #[cfg(feature = "tools")]
    ToolArguments,
}

/// Deserialize structured model output while retaining the failing field path.
///
/// `serde_path_to_error::deserialize` consumes one JSON value but, unlike
/// `serde_json::from_str`, does not reject trailing input. The explicit `end`
/// call preserves the crate's existing strict whole-document behavior.
pub(crate) fn output_from_str<T: DeserializeOwned>(raw: &str) -> Result<T> {
    from_str(raw, DecodeTarget::Output)
}

#[cfg(feature = "tools")]
pub(crate) fn tool_arguments_from_str<T: DeserializeOwned>(raw: &str) -> Result<T> {
    from_str(raw, DecodeTarget::ToolArguments)
}

fn from_str<T: DeserializeOwned>(raw: &str, target: DecodeTarget) -> Result<T> {
    let mut deserializer = serde_json::Deserializer::from_str(raw);
    let value = serde_path_to_error::deserialize(&mut deserializer)
        .map_err(|error| map_path_error(error, target))?;
    deserializer
        .end()
        .map_err(|error| map_root_error(error, target))?;
    Ok(value)
}

/// Deserialize a pre-parsed structured output value while retaining the field path.
#[cfg(any(feature = "streaming", test))]
pub(crate) fn output_from_value<T: DeserializeOwned>(value: Value) -> Result<T> {
    serde_path_to_error::deserialize(value)
        .map_err(|error| map_path_error(error, DecodeTarget::Output))
}

/// Deserialize tool-call arguments while retaining the failing field path.
#[cfg(feature = "tools")]
pub(crate) fn tool_arguments_from_value<T: DeserializeOwned>(value: Value) -> Result<T> {
    serde_path_to_error::deserialize(value)
        .map_err(|error| map_path_error(error, DecodeTarget::ToolArguments))
}

fn map_path_error(error: PathError<serde_json::Error>, target: DecodeTarget) -> RStructorError {
    let path = json_path(error.path());
    map_error(path, error.inner().to_string(), target)
}

fn map_root_error(error: serde_json::Error, target: DecodeTarget) -> RStructorError {
    map_error("$".to_string(), error.to_string(), target)
}

fn map_error(path: String, message: String, target: DecodeTarget) -> RStructorError {
    match target {
        DecodeTarget::Output => RStructorError::OutputDecodeError { path, message },
        #[cfg(feature = "tools")]
        DecodeTarget::ToolArguments => RStructorError::ToolArgumentDecodeError { path, message },
    }
}

/// Render Serde's structured path as an unambiguous JSONPath-style location.
///
/// Struct fields and identifier-like map keys use dot notation. Other map keys
/// use JSON-escaped bracket notation, so keys such as `BRK.B` remain one segment.
fn json_path(path: &Path) -> String {
    let mut rendered = String::from("$");
    for segment in path {
        match segment {
            Segment::Seq { index } => rendered.push_str(&format!("[{index}]")),
            Segment::Map { key } => push_key(&mut rendered, key),
            Segment::Enum { variant } => push_key(&mut rendered, variant),
            Segment::Unknown => rendered.push_str("[\"?\"]"),
        }
    }
    rendered
}

fn push_key(path: &mut String, key: &str) {
    if is_identifier(key) {
        path.push('.');
        path.push_str(key);
    } else {
        path.push('[');
        path.push_str(
            &serde_json::to_string(key).expect("serializing a Rust string to JSON cannot fail"),
        );
        path.push(']');
    }
}

fn is_identifier(key: &str) -> bool {
    let mut chars = key.chars();
    matches!(chars.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && chars.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde::Deserialize;
    use serde_json::json;

    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Portfolio {
        portfolio_id: String,
        positions: Vec<Position>,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Position {
        symbol: String,
        quantity: i64,
    }

    #[derive(Debug, Deserialize)]
    struct Book {
        #[serde(rename = "holdings")]
        _holdings: BTreeMap<String, Position>,
    }

    #[test]
    fn output_string_reports_nested_sequence_path() {
        let raw = include_str!("../tests/fixtures/structured/portfolio_invalid_quantity.json");
        let error = output_from_str::<Portfolio>(raw).unwrap_err();

        match error {
            RStructorError::OutputDecodeError { path, message } => {
                assert_eq!(path, "$.positions[1].quantity");
                assert!(message.contains("invalid type"));
                assert!(!message.contains("HF-ALPHA-001"));
            }
            other => panic!("expected OutputDecodeError, got {other:?}"),
        }
    }

    #[test]
    fn output_value_reports_nested_sequence_path() {
        let value: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/structured/portfolio_invalid_quantity.json"
        ))
        .unwrap();
        let error = output_from_value::<Portfolio>(value).unwrap_err();

        assert!(matches!(
            error,
            RStructorError::OutputDecodeError { path, .. }
                if path == "$.positions[1].quantity"
        ));
    }

    #[test]
    fn complex_map_keys_use_escaped_bracket_notation() {
        let value = json!({
            "holdings": {
                "BRK.B": {
                    "symbol": "BRK.B",
                    "quantity": "one hundred"
                }
            }
        });
        let error = output_from_value::<Book>(value).unwrap_err();

        assert!(matches!(
            error,
            RStructorError::OutputDecodeError { path, .. }
                if path == "$.holdings[\"BRK.B\"].quantity"
        ));
    }

    #[test]
    fn malformed_json_reports_the_active_nested_path() {
        let raw = r#"{"portfolio_id":"HF-ALPHA-001","positions":["#;
        let error = output_from_str::<Portfolio>(raw).unwrap_err();
        assert!(
            matches!(
                error,
                RStructorError::OutputDecodeError { ref path, .. }
                    if path == "$.positions"
            ),
            "unexpected malformed-JSON error: {error:?}"
        );
    }

    #[test]
    fn trailing_json_reports_the_root() {
        let raw = r#"{"portfolio_id":"HF-ALPHA-001","positions":[]} trailing"#;
        let error = output_from_str::<Portfolio>(raw).unwrap_err();
        assert!(
            matches!(
                error,
                RStructorError::OutputDecodeError { ref path, .. } if path == "$"
            ),
            "unexpected trailing-input error: {error:?}"
        );
    }

    #[test]
    fn valid_output_still_deserializes() {
        let raw = include_str!("../tests/fixtures/structured/portfolio_valid.json");
        let portfolio = output_from_str::<Portfolio>(raw).unwrap();

        assert_eq!(portfolio.portfolio_id, "HF-ALPHA-001");
        assert_eq!(portfolio.positions[1].symbol, "ESU6");
        assert_eq!(portfolio.positions[1].quantity, -240);
    }

    #[cfg(feature = "tools")]
    #[test]
    fn tool_arguments_have_a_phase_specific_error() {
        #[derive(Debug, Deserialize)]
        struct RebalanceArgs {
            #[serde(rename = "order")]
            _order: RebalanceOrder,
        }

        #[derive(Debug, Deserialize)]
        struct RebalanceOrder {
            #[serde(rename = "quantity")]
            _quantity: i64,
        }

        let error = tool_arguments_from_value::<RebalanceArgs>(json!({
            "order": { "quantity": "ten thousand shares" }
        }))
        .unwrap_err();

        assert!(matches!(
            error,
            RStructorError::ToolArgumentDecodeError { path, .. }
                if path == "$.order.quantity"
        ));
    }

    #[cfg(feature = "tools")]
    #[test]
    fn malformed_tool_arguments_report_the_active_path() {
        let error = tool_arguments_from_str::<Value>(r#"{"order":"#).unwrap_err();

        assert!(
            matches!(
                error,
                RStructorError::ToolArgumentDecodeError {
                    ref path,
                    ref message,
                } if path == "$.order" && message.contains("EOF")
            ),
            "unexpected malformed-tool-argument error: {error:?}"
        );
    }
}

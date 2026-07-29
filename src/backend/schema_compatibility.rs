//! Provider schema compatibility checks.
//!
//! The canonical schema must continue to describe the values accepted by the
//! requested Rust type. Provider-specific compilation may remove unsupported
//! annotations, but it must not silently narrow a dynamic map into an object
//! that permits no keys.

use std::fmt;

use serde_json::Value;

use crate::backend::utils::prepare_strict_schema;
use crate::error::{RStructorError, Result};
use crate::schema::Schema;

/// Providers whose constrained-output dialect closes every object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StrictSchemaProvider {
    OpenAI,
    Anthropic,
    Grok,
}

impl fmt::Display for StrictSchemaProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OpenAI => "OpenAI",
            Self::Anthropic => "Anthropic",
            Self::Grok => "Grok",
        })
    }
}

/// Compile a canonical schema for a provider that requires closed objects.
///
/// Dynamic maps are rejected before the strict transform can overwrite their
/// schema-valued `additionalProperties` with `false`.
pub(crate) fn compile_strict_schema(
    schema: &Schema,
    provider: StrictSchemaProvider,
    context: impl Into<String>,
) -> Result<Value> {
    let schema_json = schema.to_json();
    if let Some(path) = dynamic_object_path(&schema_json) {
        return Err(RStructorError::SchemaCompatibilityError {
            provider: provider.to_string().into_boxed_str(),
            context: context.into().into_boxed_str(),
            path: path.into_boxed_str(),
            message: concat!(
                "dynamic object keys require `additionalProperties`, but this provider's ",
                "strict structured-output dialect closes objects with ",
                "`additionalProperties: false`; use a provider with native typed-map ",
                "support because automatic fallback encodings are intentionally not selected"
            )
            .into(),
        });
    }

    Ok(prepare_strict_schema(schema))
}

/// Find the first schema node that permits dynamic properties.
///
/// Returns the path of the object schema rather than the path of its
/// `additionalProperties` keyword, which makes errors point at the affected
/// Rust field (for example `$.properties.positions`).
fn dynamic_object_path(schema: &Value) -> Option<String> {
    find_dynamic_object(schema, "$")
}

fn find_dynamic_object(schema: &Value, path: &str) -> Option<String> {
    match schema {
        Value::Object(object) => {
            if object
                .get("additionalProperties")
                .is_some_and(|additional| additional != &Value::Bool(false))
            {
                return Some(path.to_string());
            }

            for (key, value) in object {
                let child_path = append_key(path, key);
                if let Some(path) = find_dynamic_object(value, &child_path) {
                    return Some(path);
                }
            }
            None
        }
        Value::Array(array) => array
            .iter()
            .enumerate()
            .find_map(|(index, value)| find_dynamic_object(value, &format!("{path}[{index}]"))),
        _ => None,
    }
}

fn append_key(path: &str, key: &str) -> String {
    if is_identifier(key) {
        format!("{path}.{key}")
    } else {
        let quoted = serde_json::to_string(key).expect("serializing a string cannot fail");
        format!("{path}[{quoted}]")
    }
}

fn is_identifier(key: &str) -> bool {
    let mut chars = key.chars();
    matches!(chars.next(), Some(first) if first == '_' || first.is_ascii_alphabetic())
        && chars.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema(value: Value) -> Schema {
        Schema::new(value)
    }

    #[test]
    fn root_dynamic_map_is_rejected() {
        let error = compile_strict_schema(
            &schema(json!({
                "type": "object",
                "additionalProperties": { "type": "integer" }
            })),
            StrictSchemaProvider::OpenAI,
            "structured output",
        )
        .unwrap_err();

        assert!(matches!(
            error,
            RStructorError::SchemaCompatibilityError {
                provider,
                context,
                path,
                message,
            } if provider.as_ref() == "OpenAI"
                && context.as_ref() == "structured output"
                && path.as_ref() == "$"
                && message.contains("additionalProperties: false")
        ));
    }

    #[test]
    fn nested_map_reports_the_affected_field() {
        let error = compile_strict_schema(
            &schema(json!({
                "type": "object",
                "properties": {
                    "portfolio": {
                        "type": "object",
                        "properties": {
                            "positions": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "object",
                                    "properties": {
                                        "quantity": { "type": "integer" }
                                    }
                                }
                            }
                        }
                    }
                }
            })),
            StrictSchemaProvider::Anthropic,
            "structured output",
        )
        .unwrap_err();

        assert!(matches!(
            error,
            RStructorError::SchemaCompatibilityError {
                provider,
                path,
                ..
            } if provider.as_ref() == "Anthropic"
                && path.as_ref() == "$.properties.portfolio.properties.positions"
        ));
    }

    #[test]
    fn maps_inside_arrays_and_unions_are_detected() {
        let array_error = compile_strict_schema(
            &schema(json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": { "type": "number" }
                }
            })),
            StrictSchemaProvider::Grok,
            "streamed item",
        )
        .unwrap_err();
        assert!(matches!(
            array_error,
            RStructorError::SchemaCompatibilityError { path, .. }
                if path.as_ref() == "$.items"
        ));

        let union_error = compile_strict_schema(
            &schema(json!({
                "anyOf": [
                    { "type": "null" },
                    {
                        "type": "object",
                        "additionalProperties": { "type": "string" }
                    }
                ]
            })),
            StrictSchemaProvider::OpenAI,
            "structured output",
        )
        .unwrap_err();
        assert!(matches!(
            union_error,
            RStructorError::SchemaCompatibilityError { path, .. }
                if path.as_ref() == "$.anyOf[1]"
        ));
    }

    #[test]
    fn definitions_and_complex_property_names_use_unambiguous_paths() {
        let error = compile_strict_schema(
            &schema(json!({
                "$defs": {
                    "Risk.Bucket": {
                        "type": "object",
                        "additionalProperties": true
                    }
                }
            })),
            StrictSchemaProvider::OpenAI,
            "structured output",
        )
        .unwrap_err();

        assert!(matches!(
            error,
            RStructorError::SchemaCompatibilityError { path, .. }
                if path.as_ref() == r#"$["$defs"]["Risk.Bucket"]"#
        ));
    }

    #[test]
    fn closed_objects_still_receive_the_normal_strict_transform() {
        let compiled = compile_strict_schema(
            &schema(json!({
                "type": "object",
                "properties": {
                    "symbol": { "type": "string" },
                    "quantity": { "type": "integer" }
                },
                "required": ["symbol"]
            })),
            StrictSchemaProvider::OpenAI,
            "structured output",
        )
        .unwrap();

        assert_eq!(compiled["additionalProperties"], false);
        assert_eq!(compiled["required"], json!(["quantity", "symbol"]));
        assert_eq!(
            compiled["properties"]["quantity"]["type"],
            json!(["integer", "null"])
        );
    }

    #[test]
    fn mutual_recursion_definitions_receive_the_strict_transform() {
        let canonical = schema(json!({
            "$ref": "#/$defs/Fund",
            "$defs": {
                "Fund": {
                    "type": "object",
                    "properties": {
                        "lei": { "type": "string" },
                        "prime_broker": { "$ref": "#/$defs/PrimeBroker" }
                    },
                    "required": ["lei"]
                },
                "PrimeBroker": {
                    "type": "object",
                    "properties": {
                        "lei": { "type": "string" },
                        "funds": {
                            "type": "array",
                            "items": { "$ref": "#/$defs/Fund" }
                        }
                    },
                    "required": ["lei", "funds"]
                }
            }
        }));

        for provider in [
            StrictSchemaProvider::OpenAI,
            StrictSchemaProvider::Anthropic,
            StrictSchemaProvider::Grok,
        ] {
            let compiled =
                compile_strict_schema(&canonical, provider, "mutually recursive output").unwrap();

            assert_eq!(compiled["$ref"], "#/$defs/Fund");
            assert_eq!(
                compiled["$defs"]["Fund"]["properties"]["prime_broker"]["anyOf"][0]["$ref"],
                "#/$defs/PrimeBroker"
            );
            assert_eq!(
                compiled["$defs"]["PrimeBroker"]["properties"]["funds"]["items"]["$ref"],
                "#/$defs/Fund"
            );

            for definition in ["Fund", "PrimeBroker"] {
                assert_eq!(
                    compiled["$defs"][definition]["additionalProperties"], false,
                    "{provider} must close the {definition} definition"
                );
                let property_count = compiled["$defs"][definition]["properties"]
                    .as_object()
                    .unwrap()
                    .len();
                assert_eq!(
                    compiled["$defs"][definition]["required"]
                        .as_array()
                        .unwrap()
                        .len(),
                    property_count,
                    "{provider} must require every strict property in {definition}"
                );
            }
        }
    }

    #[test]
    fn explicitly_closed_additional_properties_is_compatible() {
        let compiled = compile_strict_schema(
            &schema(json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            })),
            StrictSchemaProvider::Anthropic,
            "tool `rebalance` arguments",
        )
        .unwrap();

        assert_eq!(compiled["additionalProperties"], false);
    }
}

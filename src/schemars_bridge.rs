//! Bridge for model types that already derive [`schemars::JsonSchema`].
//!
//! Wrap an existing Serde + schemars type in [`Schemars`] to use it with
//! [`LLMClient::materialize`](crate::LLMClient::materialize) without deriving
//! rstructor's `Instructor` macro:
//!
//! ```no_run
//! use rstructor::{LLMClient, Schemars};
//! use schemars::JsonSchema;
//! use serde::{Deserialize, Serialize};
//!
//! /// A quote observed on a public exchange.
//! #[derive(Debug, JsonSchema, Serialize, Deserialize)]
//! struct Quote {
//!     /// Exchange ticker symbol.
//!     symbol: String,
//!     /// Best bid price in USD.
//!     bid: f64,
//! }
//!
//! async fn extract_quote<C: LLMClient + Sync>(
//!     client: &C,
//! ) -> rstructor::Result<Quote> {
//!     let quote = client
//!         .materialize::<Schemars<Quote>>("AAPL bid 211.42")
//!         .await?;
//!     Ok(quote.into_inner())
//! }
//! ```
//!
//! Schemas use the JSON Schema draft-07 deserialization contract and inline
//! every acyclic nested schema. Recursive types remain unsupported because
//! strict provider schemas cannot accept the remaining references; attempts to
//! materialize one return a clear [`RStructorError::SchemaError`](crate::RStructorError::SchemaError)
//! before a provider request is sent.

use schemars::JsonSchema;
use schemars::generate::SchemaSettings;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Instructor, RStructorError, Result, Schema, SchemaType};

/// Transparent adapter from a schemars model to rstructor's [`Instructor`].
///
/// `Schemars<T>` serializes and deserializes exactly like `T`; the wrapper only
/// supplies schema generation and rstructor's default no-op validation.
///
/// # Example
///
/// ```
/// use rstructor::Schemars;
///
/// let wrapped = Schemars(String::from("AAPL"));
/// assert_eq!(wrapped.into_inner(), "AAPL");
/// ```
#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Schemars<T>(pub T);

impl<T> Schemars<T> {
    /// Remove the adapter and return the wrapped model.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> SchemaType for Schemars<T>
where
    T: JsonSchema + Serialize + DeserializeOwned,
{
    fn schema() -> Schema {
        Self::try_schema().unwrap_or_else(|error| {
            panic!("failed to generate an rstructor schema from schemars: {error}")
        })
    }

    fn try_schema() -> Result<Schema> {
        generate_schema::<T>()
    }

    fn schema_name() -> Option<String> {
        Some(T::schema_name().into_owned())
    }
}

impl<T> Instructor for Schemars<T> where T: JsonSchema + Serialize + DeserializeOwned {}

fn generate_schema<T: JsonSchema>() -> Result<Schema> {
    let settings = SchemaSettings::draft07().with(|settings| {
        settings.inline_subschemas = true;
        // The dialect is fixed by `draft07`; the declaration itself is
        // metadata that strict provider transports do not need.
        settings.meta_schema = None;
    });
    let generated = settings.into_generator().into_root_schema_for::<T>();
    let mut schema = generated.to_value();

    if let Some((path, reference)) = find_reference(&schema, "$") {
        return Err(RStructorError::SchemaError(format!(
            "cannot generate a reference-free schemars schema for recursive type `{}`: \
             found `{reference}` at {path}; recursive types are not supported by \
             `Schemars<T>`",
            std::any::type_name::<T>()
        )));
    }

    // Inlining leaves no consumers for definition tables. Remove an empty or
    // otherwise unused table so neither spelling can reach provider APIs.
    if let Some(object) = schema.as_object_mut() {
        object.remove("$defs");
        object.remove("definitions");
    }

    Ok(Schema::new(schema))
}

fn find_reference<'a>(schema: &'a Value, path: &str) -> Option<(String, &'a str)> {
    let mut pending = vec![(schema, path.to_string())];

    while let Some((value, path)) = pending.pop() {
        match value {
            Value::Object(object) => {
                if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
                    return Some((format!("{path}.$ref"), reference));
                }

                pending.extend(
                    object
                        .iter()
                        .map(|(key, value)| (value, format!("{path}.{}", json_path_key(key)))),
                );
            }
            Value::Array(values) => {
                pending.extend(
                    values
                        .iter()
                        .enumerate()
                        .map(|(index, value)| (value, format!("{path}[{index}]"))),
                );
            }
            _ => {}
        }
    }

    None
}

fn json_path_key(key: &str) -> String {
    let mut characters = key.chars();
    if matches!(
        characters.next(),
        Some(first) if first == '_' || first.is_ascii_alphabetic()
    ) && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
    {
        key.to_string()
    } else {
        serde_json::to_string(key).expect("serializing a string cannot fail")
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "_client")]
    use std::collections::BTreeSet;

    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};

    use super::*;

    /// A security listed on a public exchange.
    #[derive(JsonSchema, Serialize, Deserialize)]
    struct Instrument {
        /// Exchange ticker symbol.
        symbol: String,
        /// ISO 10383 market identifier code, when known.
        venue: Option<String>,
    }

    /// A real-time top-of-book market quote.
    #[derive(JsonSchema, Serialize, Deserialize)]
    struct Quote {
        /// Instrument whose order book was observed.
        instrument: Instrument,
        /// Best displayed bid in USD.
        bid: f64,
        /// Best displayed offer in USD.
        ask: f64,
    }

    fn canonical_quote_schema() -> Schema {
        Schemars::<Quote>::try_schema().expect("acyclic schemars model")
    }

    fn assert_no_keyword(schema: &Value, keyword: &str) {
        match schema {
            Value::Object(object) => {
                assert!(
                    !object.contains_key(keyword),
                    "unexpected `{keyword}` in {schema}"
                );
                for value in object.values() {
                    assert_no_keyword(value, keyword);
                }
            }
            Value::Array(values) => {
                for value in values {
                    assert_no_keyword(value, keyword);
                }
            }
            _ => {}
        }
    }

    #[cfg(feature = "_client")]
    fn assert_strict_objects(schema: &Value, provider: StrictSchemaProvider) {
        match schema {
            Value::Object(object) => {
                if let Some(properties) = object.get("properties").and_then(Value::as_object) {
                    assert_eq!(
                        object.get("additionalProperties"),
                        Some(&Value::Bool(false)),
                        "{provider} must close every object: {schema}"
                    );

                    let property_names = properties.keys().cloned().collect::<BTreeSet<_>>();
                    let required_names = object
                        .get("required")
                        .and_then(Value::as_array)
                        .expect("strict object must have required fields")
                        .iter()
                        .map(|name| {
                            name.as_str()
                                .expect("required field names must be strings")
                                .to_string()
                        })
                        .collect::<BTreeSet<_>>();
                    assert_eq!(
                        required_names, property_names,
                        "{provider} must require every object property"
                    );
                }

                for value in object.values() {
                    assert_strict_objects(value, provider);
                }
            }
            Value::Array(values) => {
                for value in values {
                    assert_strict_objects(value, provider);
                }
            }
            _ => {}
        }
    }

    #[cfg(feature = "_client")]
    fn assert_strict_provider_schema(provider: StrictSchemaProvider) {
        let compiled =
            compile_strict_schema(&canonical_quote_schema(), provider, "schemars test output")
                .expect("quote schema is compatible with strict providers");
        assert_strict_objects(&compiled, provider);
        assert_no_keyword(&compiled, "$ref");
        assert_no_keyword(&compiled, "$defs");
        assert_no_keyword(&compiled, "definitions");

        assert_eq!(
            compiled["properties"]["instrument"]["properties"]["symbol"]["description"],
            "Exchange ticker symbol.",
            "{provider} must preserve schemars field descriptions"
        );
        assert_eq!(
            compiled["properties"]["instrument"]["properties"]["venue"]["type"],
            json!(["string", "null"]),
            "{provider} must retain schemars optional-field semantics"
        );
    }

    #[test]
    fn draft07_schema_is_inline_and_preserves_doc_comments() {
        let schema = canonical_quote_schema().to_json();

        assert_eq!(Schemars::<Quote>::schema_name().as_deref(), Some("Quote"));
        assert_no_keyword(&schema, "$schema");
        assert_no_keyword(&schema, "$ref");
        assert_no_keyword(&schema, "$defs");
        assert_no_keyword(&schema, "definitions");
        assert_eq!(
            schema["description"],
            "A real-time top-of-book market quote."
        );
        assert_eq!(
            schema["properties"]["instrument"]["description"],
            "Instrument whose order book was observed."
        );
        assert_eq!(
            schema["properties"]["instrument"]["properties"]["symbol"]["description"],
            "Exchange ticker symbol."
        );
        assert_eq!(schema["required"], json!(["instrument", "bid", "ask"]));
    }

    #[cfg(feature = "_client")]
    use crate::backend::{StrictSchemaProvider, compile_strict_schema};

    #[cfg(feature = "_client")]
    #[test]
    fn openai_normalization_produces_a_strict_schema() {
        assert_strict_provider_schema(StrictSchemaProvider::OpenAI);
    }

    #[cfg(feature = "_client")]
    #[test]
    fn anthropic_normalization_produces_a_strict_schema() {
        assert_strict_provider_schema(StrictSchemaProvider::Anthropic);
    }

    #[cfg(feature = "_client")]
    #[test]
    fn grok_normalization_produces_a_strict_schema() {
        assert_strict_provider_schema(StrictSchemaProvider::Grok);
    }

    #[cfg(feature = "_client")]
    #[test]
    fn gemini_normalization_preserves_the_supported_schema_shape() {
        let compiled = crate::backend::prepare_gemini_schema(
            &canonical_quote_schema(),
            "schemars test output",
        )
        .expect("quote schema is compatible with Gemini");

        for keyword in ["$schema", "$ref", "$defs", "definitions", "title"] {
            assert_no_keyword(&compiled, keyword);
        }
        assert_eq!(compiled["required"], json!(["instrument", "bid", "ask"]));
        assert_eq!(
            compiled["properties"]["instrument"]["properties"]["symbol"]["description"],
            "Exchange ticker symbol."
        );
    }
}

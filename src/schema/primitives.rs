use super::{Schema, SchemaType};
use serde_json::{Value, json};
use std::collections::HashMap;

// ============================================================================
// Box<T> - Transparent wrapper, delegates to inner type
// ============================================================================

impl<T: SchemaType> SchemaType for Box<T> {
    fn schema() -> Schema {
        T::schema()
    }

    fn schema_name() -> Option<String> {
        T::schema_name()
    }
}

// ============================================================================
// serde_json::Value - Any valid JSON
// ============================================================================

impl SchemaType for Value {
    fn schema() -> Schema {
        // For serde_json::Value, we use an empty object schema which allows any JSON
        Schema::new(json!({}))
    }

    fn schema_name() -> Option<String> {
        Some("JsonValue".to_string())
    }
}

// ============================================================================
// HashMap<String, V> - Objects with dynamic keys
// ============================================================================

impl<V: SchemaType> SchemaType for HashMap<String, V> {
    fn schema() -> Schema {
        let value_schema = V::schema().to_json();
        Schema::new(json!({
            "type": "object",
            "additionalProperties": value_schema
        }))
    }

    fn schema_name() -> Option<String> {
        let value_name = V::schema_name().unwrap_or_else(|| "Unknown".to_string());
        Some(format!("HashMap<String, {}>", value_name))
    }
}

// Also implement for HashMap with other string-like keys that serialize to strings
impl<V: SchemaType> SchemaType for std::collections::BTreeMap<String, V> {
    fn schema() -> Schema {
        let value_schema = V::schema().to_json();
        Schema::new(json!({
            "type": "object",
            "additionalProperties": value_schema
        }))
    }

    fn schema_name() -> Option<String> {
        let value_name = V::schema_name().unwrap_or_else(|| "Unknown".to_string());
        Some(format!("BTreeMap<String, {}>", value_name))
    }
}

// ============================================================================
// Tuples - Fixed-length arrays with typed elements
// ============================================================================

// Helper macro to implement SchemaType for tuples of various sizes
macro_rules! impl_tuple_schema {
    ($($idx:tt $T:ident),+) => {
        impl<$($T: SchemaType),+> SchemaType for ($($T,)+) {
            fn schema() -> Schema {
                let items = vec![
                    $($T::schema().to_json()),+
                ];
                let count = items.len();
                Schema::new(json!({
                    "type": "array",
                    "prefixItems": items,
                    "minItems": count,
                    "maxItems": count
                }))
            }

            fn schema_name() -> Option<String> {
                let names = vec![
                    $($T::schema_name().unwrap_or_else(|| "Unknown".to_string())),+
                ];
                Some(format!("({})", names.join(", ")))
            }
        }
    };
}

// Implement for tuples of size 1-12
impl_tuple_schema!(0 T0);
impl_tuple_schema!(0 T0, 1 T1);
impl_tuple_schema!(0 T0, 1 T1, 2 T2);
impl_tuple_schema!(0 T0, 1 T1, 2 T2, 3 T3);
impl_tuple_schema!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4);
impl_tuple_schema!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5);
impl_tuple_schema!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6);
impl_tuple_schema!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7);
impl_tuple_schema!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7, 8 T8);
impl_tuple_schema!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7, 8 T8, 9 T9);
impl_tuple_schema!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7, 8 T8, 9 T9, 10 T10);
impl_tuple_schema!(0 T0, 1 T1, 2 T2, 3 T3, 4 T4, 5 T5, 6 T6, 7 T7, 8 T8, 9 T9, 10 T10, 11 T11);

// ============================================================================
// Primitive types - String, integers, floats, bool
// ============================================================================

impl SchemaType for String {
    fn schema() -> Schema {
        Schema::new(json!({"type": "string"}))
    }

    fn schema_name() -> Option<String> {
        Some("String".to_string())
    }
}

impl SchemaType for &str {
    fn schema() -> Schema {
        Schema::new(json!({"type": "string"}))
    }

    fn schema_name() -> Option<String> {
        Some("str".to_string())
    }
}

impl SchemaType for bool {
    fn schema() -> Schema {
        Schema::new(json!({"type": "boolean"}))
    }

    fn schema_name() -> Option<String> {
        Some("bool".to_string())
    }
}

// Integer types
macro_rules! impl_integer_schema {
    ($($ty:ty),+) => {
        $(
            impl SchemaType for $ty {
                fn schema() -> Schema {
                    Schema::new(json!({"type": "integer"}))
                }

                fn schema_name() -> Option<String> {
                    Some(stringify!($ty).to_string())
                }
            }
        )+
    };
}

impl_integer_schema!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

// Float types
macro_rules! impl_float_schema {
    ($($ty:ty),+) => {
        $(
            impl SchemaType for $ty {
                fn schema() -> Schema {
                    Schema::new(json!({"type": "number"}))
                }

                fn schema_name() -> Option<String> {
                    Some(stringify!($ty).to_string())
                }
            }
        )+
    };
}

impl_float_schema!(f32, f64);

// ============================================================================
// Vec<T> - Arrays
// ============================================================================

impl<T: SchemaType> SchemaType for Vec<T> {
    fn schema() -> Schema {
        let item_schema = T::schema().to_json();
        Schema::new(json!({
            "type": "array",
            "items": item_schema
        }))
    }

    fn schema_name() -> Option<String> {
        let item_name = T::schema_name().unwrap_or_else(|| "Unknown".to_string());
        Some(format!("Vec<{}>", item_name))
    }
}

// ============================================================================
// Option<T> - Nullable values
// ============================================================================

impl<T: SchemaType> SchemaType for Option<T> {
    fn schema() -> Schema {
        // For Option<T>, we just return the inner type's schema
        // The "required" handling is done at the struct level
        T::schema()
    }

    fn schema_name() -> Option<String> {
        let inner_name = T::schema_name().unwrap_or_else(|| "Unknown".to_string());
        Some(format!("Option<{}>", inner_name))
    }
}

// ============================================================================
// HashSet and BTreeSet - Arrays with unique items
// ============================================================================

impl<T: SchemaType> SchemaType for std::collections::HashSet<T> {
    fn schema() -> Schema {
        let item_schema = T::schema().to_json();
        Schema::new(json!({
            "type": "array",
            "items": item_schema,
            "uniqueItems": true
        }))
    }

    fn schema_name() -> Option<String> {
        let item_name = T::schema_name().unwrap_or_else(|| "Unknown".to_string());
        Some(format!("HashSet<{}>", item_name))
    }
}

impl<T: SchemaType> SchemaType for std::collections::BTreeSet<T> {
    fn schema() -> Schema {
        let item_schema = T::schema().to_json();
        Schema::new(json!({
            "type": "array",
            "items": item_schema,
            "uniqueItems": true
        }))
    }

    fn schema_name() -> Option<String> {
        let item_name = T::schema_name().unwrap_or_else(|| "Unknown".to_string());
        Some(format!("BTreeSet<{}>", item_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_schema() {
        let schema = <Box<String>>::schema();
        let json = schema.to_json();
        assert_eq!(json["type"], "string");
    }

    #[test]
    fn test_value_schema() {
        let schema = Value::schema();
        let json = schema.to_json();
        // Empty object means any JSON is valid
        assert!(json.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_hashmap_schema() {
        let schema = <HashMap<String, i32>>::schema();
        let json = schema.to_json();
        assert_eq!(json["type"], "object");
        assert_eq!(json["additionalProperties"]["type"], "integer");
    }

    #[test]
    fn test_tuple_schema() {
        let schema = <(i32, String)>::schema();
        let json = schema.to_json();
        assert_eq!(json["type"], "array");
        assert_eq!(json["minItems"], 2);
        assert_eq!(json["maxItems"], 2);
        assert_eq!(json["prefixItems"][0]["type"], "integer");
        assert_eq!(json["prefixItems"][1]["type"], "string");
    }

    #[test]
    fn test_vec_schema() {
        let schema = <Vec<String>>::schema();
        let json = schema.to_json();
        assert_eq!(json["type"], "array");
        assert_eq!(json["items"]["type"], "string");
    }

    #[test]
    fn test_nested_hashmap_schema() {
        let schema = <HashMap<String, Vec<String>>>::schema();
        let json = schema.to_json();
        assert_eq!(json["type"], "object");
        assert_eq!(json["additionalProperties"]["type"], "array");
        assert_eq!(json["additionalProperties"]["items"]["type"], "string");
    }

    // ========================================================================
    // HashSet / BTreeSet schema (uniqueItems)
    // ========================================================================

    #[test]
    fn test_hashset_schema() {
        use std::collections::HashSet;
        let json = <HashSet<String>>::schema().to_json();
        assert_eq!(json["type"], "array");
        assert_eq!(json["items"]["type"], "string");
        assert_eq!(json["uniqueItems"], true);
    }

    #[test]
    fn test_btreeset_schema() {
        use std::collections::BTreeSet;
        let json = <BTreeSet<i64>>::schema().to_json();
        assert_eq!(json["type"], "array");
        assert_eq!(json["items"]["type"], "integer");
        assert_eq!(json["uniqueItems"], true);
    }

    // ========================================================================
    // BTreeMap<String, V> schema
    // ========================================================================

    #[test]
    fn test_btreemap_schema() {
        use std::collections::BTreeMap;
        let json = <BTreeMap<String, bool>>::schema().to_json();
        assert_eq!(json["type"], "object");
        assert_eq!(json["additionalProperties"]["type"], "boolean");
    }

    // ========================================================================
    // Deeply-nested generic combinations
    // ========================================================================

    #[test]
    fn test_nested_vec_option_schema() {
        // Vec<Option<i32>>: Option is transparent, so items.type == integer
        let json = <Vec<Option<i32>>>::schema().to_json();
        assert_eq!(json["type"], "array");
        assert_eq!(json["items"]["type"], "integer");
    }

    #[test]
    fn test_nested_option_vec_box_schema() {
        // Option<Vec<Box<i32>>>: Option transparent -> array; Box transparent -> integer items
        let json = <Option<Vec<Box<i32>>>>::schema().to_json();
        assert_eq!(json["type"], "array");
        assert_eq!(json["items"]["type"], "integer");
    }

    #[test]
    fn test_nested_hashset_vec_schema() {
        use std::collections::HashSet;
        // HashSet<Vec<i32>>: set -> uniqueItems array of arrays of integers
        let json = <HashSet<Vec<i32>>>::schema().to_json();
        assert_eq!(json["type"], "array");
        assert_eq!(json["uniqueItems"], true);
        assert_eq!(json["items"]["type"], "array");
        assert_eq!(json["items"]["items"]["type"], "integer");
    }

    // ========================================================================
    // Tuple arities 1 and 3..=12 (prefixItems / minItems / maxItems)
    // ========================================================================

    #[test]
    fn test_tuple_arity_one_schema() {
        let json = <(bool,)>::schema().to_json();
        assert_eq!(json["type"], "array");
        assert_eq!(json["minItems"], 1);
        assert_eq!(json["maxItems"], 1);
        assert_eq!(json["prefixItems"][0]["type"], "boolean");
        assert_eq!(json["prefixItems"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_tuple_arity_three_schema() {
        let json = <(i32, String, f64)>::schema().to_json();
        assert_eq!(json["type"], "array");
        assert_eq!(json["minItems"], 3);
        assert_eq!(json["maxItems"], 3);
        assert_eq!(json["prefixItems"][0]["type"], "integer");
        assert_eq!(json["prefixItems"][1]["type"], "string");
        assert_eq!(json["prefixItems"][2]["type"], "number");
        assert_eq!(json["prefixItems"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_tuple_arity_twelve_schema() {
        // Largest supported tuple arity.
        type T12 = (i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, String, bool);
        let json = <T12>::schema().to_json();
        assert_eq!(json["type"], "array");
        assert_eq!(json["minItems"], 12);
        assert_eq!(json["maxItems"], 12);
        let prefix = json["prefixItems"].as_array().unwrap();
        assert_eq!(prefix.len(), 12);
        assert_eq!(prefix[0]["type"], "integer");
        assert_eq!(prefix[8]["type"], "number"); // f32
        assert_eq!(prefix[10]["type"], "string"); // String
        assert_eq!(prefix[11]["type"], "boolean"); // bool
    }

    // ========================================================================
    // schema_name() for all collection impls + "Unknown" fallback
    // ========================================================================

    #[test]
    fn test_schema_name_collections() {
        use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
        assert_eq!(
            <Vec<String>>::schema_name(),
            Some("Vec<String>".to_string())
        );
        assert_eq!(
            <HashMap<String, bool>>::schema_name(),
            Some("HashMap<String, bool>".to_string())
        );
        assert_eq!(
            <BTreeMap<String, i64>>::schema_name(),
            Some("BTreeMap<String, i64>".to_string())
        );
        assert_eq!(
            <HashSet<String>>::schema_name(),
            Some("HashSet<String>".to_string())
        );
        assert_eq!(
            <BTreeSet<i32>>::schema_name(),
            Some("BTreeSet<i32>".to_string())
        );
        assert_eq!(
            <(i32, String)>::schema_name(),
            Some("(i32, String)".to_string())
        );
        assert_eq!(
            <Option<String>>::schema_name(),
            Some("Option<String>".to_string())
        );
    }

    // A type whose `schema_name()` falls back to the default `None`, exercising
    // the "Unknown" interpolation branch in the collection name formatters.
    struct Anon;

    impl SchemaType for Anon {
        fn schema() -> Schema {
            Schema::new(json!({"type": "object"}))
        }
        // schema_name() intentionally left as the trait default (None).
    }

    #[test]
    fn test_schema_name_unknown_fallback() {
        use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
        // Inner name is None -> formatted as "Unknown".
        assert_eq!(<Vec<Anon>>::schema_name(), Some("Vec<Unknown>".to_string()));
        assert_eq!(
            <Option<Anon>>::schema_name(),
            Some("Option<Unknown>".to_string())
        );
        assert_eq!(
            <HashSet<Anon>>::schema_name(),
            Some("HashSet<Unknown>".to_string())
        );
        assert_eq!(
            <BTreeSet<Anon>>::schema_name(),
            Some("BTreeSet<Unknown>".to_string())
        );
        assert_eq!(
            <HashMap<String, Anon>>::schema_name(),
            Some("HashMap<String, Unknown>".to_string())
        );
        assert_eq!(
            <BTreeMap<String, Anon>>::schema_name(),
            Some("BTreeMap<String, Unknown>".to_string())
        );
        // Tuple element name None -> "Unknown" inside the joined list.
        assert_eq!(
            <(i32, Anon)>::schema_name(),
            Some("(i32, Unknown)".to_string())
        );
    }

    // ========================================================================
    // Option<T> transparency
    // ========================================================================

    #[test]
    fn test_option_schema_transparency() {
        // Option<T>::schema() must be exactly the inner type's schema:
        // no anyOf, no "null" type wrapping.
        let opt = <Option<String>>::schema().to_json();
        let inner = String::schema().to_json();
        assert_eq!(opt, inner);
        assert_eq!(opt["type"], "string");
        assert!(opt.get("anyOf").is_none());
        // The "null" string must not appear anywhere in the schema.
        assert!(!opt.to_string().contains("null"));
    }

    // ========================================================================
    // Box<T> non-String inner + name passthrough
    // ========================================================================

    #[test]
    fn test_box_non_string_inner_schema() {
        // Box<Vec<i32>>::schema() == Vec<i32>::schema() (array of integers).
        let json = <Box<Vec<i32>>>::schema().to_json();
        assert_eq!(json["type"], "array");
        assert_eq!(json["items"]["type"], "integer");
        // Box<i32> -> plain integer.
        let int_json = <Box<i32>>::schema().to_json();
        assert_eq!(int_json["type"], "integer");
    }

    #[test]
    fn test_box_name_passthrough() {
        // Box is invisible in schema_name: it forwards the inner name verbatim.
        assert_eq!(<Box<Vec<i32>>>::schema_name(), Some("Vec<i32>".to_string()));
        assert_eq!(<Box<String>>::schema_name(), Some("String".to_string()));
        assert_eq!(<Box<i32>>::schema_name(), Some("i32".to_string()));
    }

    // ========================================================================
    // CustomTypeSchema default-None branches
    // ========================================================================

    // Minimal impl: only schema_type overridden, every other method left as the
    // trait default (None). json_schema() must emit only {"type": ...}.
    struct MinimalCustom;

    impl crate::schema::CustomTypeSchema for MinimalCustom {
        fn schema_type() -> &'static str {
            "string"
        }
    }

    #[test]
    fn test_custom_type_schema_minimal_defaults() {
        use crate::schema::CustomTypeSchema;
        let json = MinimalCustom::json_schema();
        let obj = json.as_object().unwrap();
        assert_eq!(obj.len(), 1);
        assert_eq!(json["type"], "string");
        assert!(json.get("format").is_none());
        assert!(json.get("description").is_none());
        // Default accessor methods all return None.
        assert_eq!(MinimalCustom::schema_format(), None);
        assert_eq!(MinimalCustom::schema_description(), None);
        assert!(MinimalCustom::schema_additional_properties().is_none());
    }

    // Fully-populated impl: every optional method overridden so json_schema()
    // takes each "Some" branch, including additional-properties merging.
    struct FullCustom;

    impl crate::schema::CustomTypeSchema for FullCustom {
        fn schema_type() -> &'static str {
            "string"
        }
        fn schema_format() -> Option<&'static str> {
            Some("date-time")
        }
        fn schema_description() -> Option<String> {
            Some("a timestamp".to_string())
        }
        fn schema_additional_properties() -> Option<Value> {
            Some(json!({"x-custom": 1}))
        }
    }

    #[test]
    fn test_custom_type_schema_full_branches() {
        use crate::schema::CustomTypeSchema;
        let json = FullCustom::json_schema();
        assert_eq!(json["type"], "string");
        assert_eq!(json["format"], "date-time");
        assert_eq!(json["description"], "a timestamp");
        assert_eq!(json["x-custom"], 1);
    }

    #[test]
    fn test_custom_type_schema_non_object_additional_ignored() {
        // additional_properties that is not a JSON object is silently ignored
        // (the `if let Some(obj) = additional.as_object()` guard fails) and
        // must not panic.
        struct NonObjectAdditional;
        impl crate::schema::CustomTypeSchema for NonObjectAdditional {
            fn schema_type() -> &'static str {
                "string"
            }
            fn schema_additional_properties() -> Option<Value> {
                Some(json!([1, 2, 3]))
            }
        }
        use crate::schema::CustomTypeSchema;
        let json = NonObjectAdditional::json_schema();
        let obj = json.as_object().unwrap();
        // Only "type" survives; the array additional is dropped.
        assert_eq!(obj.len(), 1);
        assert_eq!(json["type"], "string");
    }

    // ========================================================================
    // SchemaBuilder example single-vs-multi
    // ========================================================================

    #[test]
    fn test_schema_builder_single_example() {
        let json = crate::schema::SchemaBuilder::object()
            .example(json!({"a": 1}))
            .build()
            .to_json();
        // Single example -> "example" key (scalar), no "examples" array.
        assert_eq!(json["example"], json!({"a": 1}));
        assert!(json.get("examples").is_none());
    }

    #[test]
    fn test_schema_builder_multi_examples() {
        let json = crate::schema::SchemaBuilder::object()
            .example(json!({"a": 1}))
            .example(json!({"a": 2}))
            .build()
            .to_json();
        // Multiple examples -> "examples" array, no "example" key.
        assert!(json.get("example").is_none());
        let examples = json["examples"].as_array().unwrap();
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0], json!({"a": 1}));
        assert_eq!(examples[1], json!({"a": 2}));
    }

    // ========================================================================
    // Primitive schema_name() coverage: integer widths, &str, bool, String,
    // floats, serde_json::Value (LOW rows)
    // ========================================================================

    #[test]
    fn test_integer_width_schema_and_names() {
        // All integer widths emit {"type":"integer"} and name == stringified type.
        macro_rules! check_int {
            ($($t:ty => $name:literal),+ $(,)?) => {
                $(
                    assert_eq!(<$t>::schema().to_json()["type"], "integer", "{} type", $name);
                    assert_eq!(<$t>::schema_name(), Some($name.to_string()), "{} name", $name);
                )+
            };
        }
        check_int!(
            i8 => "i8",
            i16 => "i16",
            i32 => "i32",
            i64 => "i64",
            i128 => "i128",
            isize => "isize",
            u8 => "u8",
            u16 => "u16",
            u32 => "u32",
            u64 => "u64",
            u128 => "u128",
            usize => "usize",
        );
    }

    #[test]
    fn test_float_schema_and_names() {
        assert_eq!(f32::schema().to_json()["type"], "number");
        assert_eq!(f32::schema_name(), Some("f32".to_string()));
        assert_eq!(f64::schema().to_json()["type"], "number");
        assert_eq!(f64::schema_name(), Some("f64".to_string()));
    }

    #[test]
    fn test_str_bool_string_names_and_schema() {
        assert_eq!(<&str>::schema().to_json()["type"], "string");
        assert_eq!(<&str>::schema_name(), Some("str".to_string()));
        assert_eq!(bool::schema().to_json()["type"], "boolean");
        assert_eq!(bool::schema_name(), Some("bool".to_string()));
        assert_eq!(String::schema().to_json()["type"], "string");
        assert_eq!(String::schema_name(), Some("String".to_string()));
    }

    #[test]
    fn test_json_value_schema_name() {
        assert_eq!(
            <serde_json::Value>::schema_name(),
            Some("JsonValue".to_string())
        );
        // schema() is an empty object (any JSON allowed).
        assert!(
            <serde_json::Value>::schema()
                .to_json()
                .as_object()
                .unwrap()
                .is_empty()
        );
    }
}

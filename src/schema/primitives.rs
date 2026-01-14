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
}

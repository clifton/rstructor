use serde_json::{Value, json};

/// Trait for types that can provide their own JSON Schema representation
///
/// This trait should be implemented for custom types that don't have a
/// direct JSON representation but can be serialized to JSON, such as dates,
/// UUIDs, and other special types.
///
/// # Examples
///
/// ```
/// use rstructor::schema::CustomTypeSchema;
/// use serde_json::json;
/// use serde::{Serialize, Deserialize};
///
/// // Create a custom date type
/// #[derive(Serialize, Deserialize)]
/// struct MyCustomDate {
///     year: u16,
///     month: u8,
///     day: u8,
/// }
///
/// // Implement CustomTypeSchema for our custom date type
/// impl CustomTypeSchema for MyCustomDate {
///     fn schema_type() -> &'static str {
///         "string"
///     }
///     
///     fn schema_format() -> Option<&'static str> {
///         Some("date-time")
///     }
///     
///     fn schema_description() -> Option<String> {
///         Some("ISO-8601 formatted date and time".to_string())
///     }
/// }
/// ```
pub trait CustomTypeSchema {
    /// Returns the JSON Schema type for this custom type
    ///
    /// This is typically "string" for dates, UUIDs, etc.
    fn schema_type() -> &'static str;

    /// Returns the JSON Schema format for this custom type
    ///
    /// Common formats include "date-time", "uuid", "email", etc.
    fn schema_format() -> Option<&'static str> {
        None
    }

    /// Returns a description of this custom type for documentation
    fn schema_description() -> Option<String> {
        None
    }

    /// Returns any additional JSON Schema properties for this type
    fn schema_additional_properties() -> Option<Value> {
        None
    }

    /// Generate a complete JSON Schema object for this type
    fn json_schema() -> Value {
        let mut schema = json!({
            "type": Self::schema_type(),
        });

        // Add format if present
        if let Some(format) = Self::schema_format() {
            schema
                .as_object_mut()
                .unwrap()
                .insert("format".to_string(), Value::String(format.to_string()));
        }

        // Add description if present
        if let Some(description) = Self::schema_description() {
            schema
                .as_object_mut()
                .unwrap()
                .insert("description".to_string(), Value::String(description));
        }

        // Add any additional properties
        if let Some(additional) = Self::schema_additional_properties()
            && let Some(additional_obj) = additional.as_object() {
                for (key, value) in additional_obj {
                    schema
                        .as_object_mut()
                        .unwrap()
                        .insert(key.clone(), value.clone());
                }
            }

        schema
    }
}

mod builder;
mod custom_type;
mod primitives;
pub use builder::SchemaBuilder;
pub use custom_type::CustomTypeSchema;

use crate::error::Result;
use serde_json::Value;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// Helper function to call a struct's validate method if it exists
/// This is used by the derive macro to prevent dead code warnings on struct validate methods
pub fn call_validate_if_exists<T>(_obj: &T) -> Result<()> {
    // This function is intentionally a no-op in the base implementation
    // The derive macro will generate specialized versions that call the actual validate method
    // for types that have one
    Ok(())
}

/// Schema is a representation of a JSON Schema that describes the structure
/// an LLM should return.
///
/// The Schema struct wraps a JSON object that follows the JSON Schema specification.
/// It provides methods to access and manipulate the schema.
///
/// # Examples
///
/// Creating a schema manually:
///
/// ```
/// use rstructor::Schema;
/// use serde_json::json;
///
/// // Create a schema for a person with name and age
/// let schema = Schema::new(json!({
///     "type": "object",
///     "title": "Person",
///     "properties": {
///         "name": {
///             "type": "string",
///             "description": "Person's name"
///         },
///         "age": {
///             "type": "integer",
///             "description": "Person's age"
///         }
///     },
///     "required": ["name", "age"]
/// }));
///
/// // Convert to JSON or string
/// let json = schema.to_json();
/// assert_eq!(json["title"], "Person");
///
/// let schema_str = schema.to_string();
/// assert!(schema_str.contains("Person"));
/// ```
///
/// Using the builder:
///
/// ```
/// use rstructor::Schema;
/// use serde_json::json;
///
/// // Create a schema using the builder
/// let schema = Schema::builder()
///     .title("Person")
///     .property("name", json!({"type": "string", "description": "Person's name"}), true)
///     .property("age", json!({"type": "integer", "description": "Person's age"}), true)
///     .build();
///
/// let json = schema.to_json();
/// assert_eq!(json["title"], "Person");
/// ```
#[derive(Debug, Clone)]
pub struct Schema {
    pub schema: Value,
}

impl Schema {
    pub fn new(schema: Value) -> Self {
        Self { schema }
    }

    /// Return a reference to the raw unenhanced schema
    ///
    /// This method exists for backward compatibility with code expecting a reference.
    /// Most internal code should use to_enhanced_json() instead.
    pub fn original_schema(&self) -> &Value {
        &self.schema
    }

    /// Get the JSON representation of this schema
    ///
    /// Returns the schema as-is without enhancement to prevent stack overflow
    /// with complex nested structures. The derive macro should generate complete
    /// schemas that don't need runtime enhancement.
    pub fn to_json(&self) -> Value {
        // Return schema directly without enhancement to prevent stack overflow
        // The derive macro should generate complete schemas
        self.schema.clone()
    }

    // Format the schema as a pretty-printed JSON string
    pub fn to_pretty_json(&self) -> String {
        // Get the schema with array enhancements
        let schema_json = self.to_json();
        // CRITICAL: Use serde_json directly to avoid recursion - never call self.schema.to_string()
        // which would use Display impl and cause infinite recursion
        serde_json::to_string_pretty(&schema_json).unwrap_or_else(|_| {
            serde_json::to_string_pretty(&self.schema).unwrap_or_else(|_| "{}".to_string())
        })
    }

    /// Create a schema builder for an object type
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::object()
    }
}

// Display implementation for Schema
// NOTE: This can cause stack overflow with very complex schemas.
// Prefer using serde_json::to_string_pretty(&schema.to_json()) directly
impl Display for Schema {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        // Use serde_json directly to avoid any potential recursion
        let json = self.to_json();
        let json_str = serde_json::to_string_pretty(&json).unwrap_or_else(|_| "{}".to_string());
        write!(f, "{}", json_str)
    }
}

/// SchemaType trait defines a type that can be converted to a JSON Schema.
///
/// This trait is implemented for types that can generate a JSON Schema representation
/// of themselves. It's typically implemented by the derive macro for structs and enums.
///
/// # Examples
///
/// Manual implementation for a custom type:
///
/// ```
/// use rstructor::{Schema, SchemaType};
/// use serde_json::json;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Serialize, Deserialize)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// // Manually implement SchemaType for Person
/// impl SchemaType for Person {
///     fn schema() -> Schema {
///         Schema::new(json!({
///             "type": "object",
///             "title": "Person",
///             "properties": {
///                 "name": {
///                     "type": "string"
///                 },
///                 "age": {
///                     "type": "integer"
///                 }
///             },
///             "required": ["name", "age"]
///         }))
///     }
///
///     fn schema_name() -> Option<String> {
///         Some("Person".to_string())
///     }
/// }
///
/// // Use the schema
/// let schema = Person::schema();
/// let json = schema.to_json();
/// assert_eq!(json["title"], "Person");
/// assert_eq!(Person::schema_name(), Some("Person".to_string()));
/// ```
///
/// With the derive macro (typically how you'd use it):
///
/// ```no_run
/// use rstructor::Instructor;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Instructor, Serialize, Deserialize)]
/// struct Person {
///     #[llm(description = "Person's name")]
///     name: String,
///
///     #[llm(description = "Person's age")]
///     age: u32,
/// }
///
/// // SchemaType is implemented by the Instructor derive macro
/// // (This would work in real code, but doctest doesn't have access to the macro)
/// // let schema = Person::schema();
/// // let json = schema.to_json();
/// // assert_eq!(json["properties"]["name"]["description"], "Person's name");
/// ```
pub trait SchemaType {
    /// Generate a JSON Schema representation of this type
    fn schema() -> Schema;

    /// Optional name for the schema
    ///
    /// This method returns an optional name for the schema. It's used by the LLM clients
    /// to reference the schema in their requests.
    fn schema_name() -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests;

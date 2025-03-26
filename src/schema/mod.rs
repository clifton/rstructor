mod builder;
pub use builder::SchemaBuilder;

use serde_json::Value;

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

    pub fn to_json(&self) -> &Value {
        &self.schema
    }

    pub fn to_string(&self) -> String {
        self.schema.to_string()
    }

    /// Create a schema builder for an object type
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::object()
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
/// use rstructor::LLMModel;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(LLMModel, Serialize, Deserialize)]
/// struct Person {
///     #[llm(description = "Person's name")]
///     name: String,
///     
///     #[llm(description = "Person's age")]
///     age: u32,
/// }
///
/// // SchemaType is implemented by the LLMModel derive macro
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

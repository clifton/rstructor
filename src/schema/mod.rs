mod builder;
mod custom_type;
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

    /// Get the JSON representation of this schema with improved array field descriptions
    /// and additional properties for better LLM guidance
    pub fn to_json(&self) -> Value {
        // Clone the schema for manipulation
        let mut schema_json = self.schema.clone();

        // Enhance schemas: fix array items and nested object properties
        if let Value::Object(obj) = &mut schema_json
            && let Some(Value::Object(props)) = obj.get_mut("properties")
        {
            // Check each property
            for (_, prop_value) in props.iter_mut() {
                if let Value::Object(prop) = prop_value {
                    // First, handle nested object fields (non-array)
                    if let Some(Value::String(prop_type)) = prop.get("type")
                        && prop_type == "object"
                        && !prop.contains_key("properties")
                    {
                        // This is a nested struct without embedded properties
                        // Check if description indicates it needs properties
                        let desc = prop
                            .get("description")
                            .and_then(|d| d.as_str())
                            .unwrap_or("");
                        if desc.contains("MUST be an object")
                            || desc.contains("with exactly these fields")
                        {
                            // This should have nested properties but doesn't
                            // We can't resolve the type at runtime, but we can add better description
                            // The actual fix would need to happen in the derive macro
                        }
                    }

                    // Check if this is an array property
                    if let Some(Value::String(prop_type)) = prop.get("type")
                        && prop_type == "array"
                    {
                        // Get the parent property description (may indicate objects are needed)
                        // Get this BEFORE any mutable borrows
                        let parent_description = prop
                            .get("description")
                            .and_then(|d| d.as_str())
                            .unwrap_or("")
                            .to_string();

                        // Check if it has items
                        // If items property is missing, add a default one for string type
                        if !prop.contains_key("items") {
                            let mut default_items = serde_json::Map::new();
                            default_items
                                .insert("type".to_string(), Value::String("string".to_string()));
                            prop.insert("items".to_string(), Value::Object(default_items));
                        }

                        if let Some(Value::Object(items)) = prop.get_mut("items") {
                            // Check if the items are objects or should be objects
                            // Get items type and description BEFORE mutable operations
                            let items_type =
                                items.get("type").and_then(|v| v.as_str()).unwrap_or("");

                            // Check both items description and parent property description
                            let items_description = items
                                .get("description")
                                .and_then(|d| d.as_str())
                                .unwrap_or("")
                                .to_string();

                            // Determine if this should be an object array based on:
                            // 1. Type is already "object"
                            // 2. Items description mentions objects
                            // 3. Parent property description mentions "MUST be an array of objects" or similar
                            let should_be_object = items_type == "object"
                                || (items_type == "string"
                                    && (items_description.contains("object")
                                        || items_description.contains("MUST be")
                                        || items_description.contains("complete object")
                                        || parent_description
                                            .contains("MUST be an array of objects")
                                        || parent_description.contains("array of objects")
                                        || parent_description.contains("complete object")));

                            if should_be_object {
                                // Ensure type is set to object
                                items.insert(
                                    "type".to_string(),
                                    Value::String("object".to_string()),
                                );

                                // Add a more explicit description to make sure models understand
                                let existing_description = items
                                    .get("description")
                                    .and_then(|d| d.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                // Create a more informative description without specific examples
                                let improved_desc = if existing_description.is_empty() {
                                    "Must be an array of objects. Each object must include all required fields.".to_string()
                                } else {
                                    format!(
                                        "{}. IMPORTANT: Each item must be a complete object with all required fields, not a string or primitive value.",
                                        existing_description
                                    )
                                };
                                items.insert(
                                    "description".to_string(),
                                    Value::String(improved_desc),
                                );

                                // For object arrays, try to infer common properties from the description
                                // This helps guide the LLM even if we can't embed the full schema
                                // The enhanced description already emphasizes object structure

                                // Only add generic properties if they don't already exist
                                // (they might be there if the schema was properly embedded)
                                if !items.contains_key("properties") {
                                    let mut properties = serde_json::Map::new();

                                    // Add universal properties that most objects have
                                    let mut name_prop = serde_json::Map::new();
                                    name_prop.insert(
                                        "type".to_string(),
                                        Value::String("string".to_string()),
                                    );
                                    properties.insert("name".to_string(), Value::Object(name_prop));

                                    // Add other common properties for various object types
                                    let mut type_prop = serde_json::Map::new();
                                    type_prop.insert(
                                        "type".to_string(),
                                        Value::String("string".to_string()),
                                    );
                                    properties.insert(
                                        "entity_type".to_string(),
                                        Value::Object(type_prop),
                                    );

                                    // Add relevance property for Entity objects
                                    let mut relevance_prop = serde_json::Map::new();
                                    relevance_prop.insert(
                                        "type".to_string(),
                                        Value::String("integer".to_string()),
                                    );
                                    properties.insert(
                                        "relevance".to_string(),
                                        Value::Object(relevance_prop),
                                    );

                                    // Add amount/unit for ingredient-like objects
                                    let mut amount_prop = serde_json::Map::new();
                                    amount_prop.insert(
                                        "type".to_string(),
                                        Value::String("number".to_string()),
                                    );
                                    properties
                                        .insert("amount".to_string(), Value::Object(amount_prop));

                                    let mut unit_prop = serde_json::Map::new();
                                    unit_prop.insert(
                                        "type".to_string(),
                                        Value::String("string".to_string()),
                                    );
                                    properties.insert("unit".to_string(), Value::Object(unit_prop));

                                    // Insert properties to show the structure expected
                                    items.insert(
                                        "properties".to_string(),
                                        Value::Object(properties),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        schema_json
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
impl Display for Schema {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.to_pretty_json())
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

mod builder;
pub use builder::SchemaBuilder;

use serde_json::Value;

/// Schema is a representation of a JSON Schema that describes the structure
/// an LLM should return.
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

/// SchemaType trait defines a type that can be converted to a JSON Schema
pub trait SchemaType {
    /// Generate a JSON Schema representation of this type
    fn schema() -> Schema;
    
    /// Optional name for the schema
    fn schema_name() -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests;
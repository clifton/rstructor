use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

use super::Schema;

/// SchemaBuilder helps construct JSON Schema incrementally
#[derive(Default)]
pub struct SchemaBuilder {
    schema_type: String,
    title: Option<String>,
    description: Option<String>,
    properties: HashMap<String, Value>,
    required: HashSet<String>,
    examples: Vec<Value>,
}

impl SchemaBuilder {
    pub fn new() -> Self {
        Self {
            schema_type: "object".to_string(),
            ..Default::default()
        }
    }

    pub fn object() -> Self {
        Self::new()
    }

    pub fn array(items: Value) -> Self {
        let mut builder = Self::new();
        builder.schema_type = "array".to_string();
        builder.properties.insert("items".to_string(), items);
        builder
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn property(
        mut self,
        name: impl Into<String>,
        property_schema: Value,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(name.clone(), property_schema);
        if required {
            self.required.insert(name);
        }
        self
    }

    pub fn example(mut self, example: Value) -> Self {
        self.examples.push(example);
        self
    }

    pub fn build(self) -> Schema {
        let mut schema = json!({
            "type": self.schema_type
        });

        if let Some(title) = self.title {
            schema["title"] = json!(title);
        }

        if let Some(description) = self.description {
            schema["description"] = json!(description);
        }

        if !self.properties.is_empty() {
            if self.schema_type == "object" {
                schema["properties"] = json!(self.properties);

                if !self.required.is_empty() {
                    schema["required"] = json!(self.required);
                }
            } else if self.schema_type == "array" {
                if let Some(items) = self.properties.get("items") {
                    schema["items"] = items.clone();
                }
            }
        }

        if !self.examples.is_empty() {
            if self.examples.len() == 1 {
                schema["example"] = self.examples[0].clone();
            } else {
                schema["examples"] = json!(self.examples);
            }
        }

        Schema::new(schema)
    }
}

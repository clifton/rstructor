use crate::backend::{
    ChatMessage, MaterializeInternalOutput, TokenUsage, ValidationFailureContext,
};
use crate::error::{ApiErrorKind, RStructorError, Result};
use crate::model::Instructor;
use reqwest::Response;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, error, info, trace, warn};

/// Prepare a JSON schema for strict mode by recursively adding required fields
/// to all object types in the schema.
///
/// This is required by providers like OpenAI that use strict structured outputs, where
/// every object in the schema (including nested objects and array items) must have:
/// 1. `additionalProperties: false`
/// 2. A `required` array listing all property keys
///
/// # Arguments
///
/// * `schema` - The JSON schema to modify
///
/// # Returns
///
/// A new schema Value with strict mode requirements added to all objects
pub fn prepare_strict_schema(schema: &crate::schema::Schema) -> Value {
    let mut schema_json = schema.to_json();
    add_additional_properties_false(&mut schema_json);
    schema_json
}

/// Recursively prepares a JSON schema for strict mode by adding:
/// 1. `additionalProperties: false` to all object types
/// 2. `required` array with all property keys (if not already present)
fn add_additional_properties_false(schema: &mut Value) {
    if let Some(obj) = schema.as_object_mut() {
        // Check if this is an object type schema
        let is_object_type = obj
            .get("type")
            .and_then(|t| t.as_str())
            .is_some_and(|t| t == "object");

        // Also check if it has properties (even without explicit type: object)
        let has_properties = obj.contains_key("properties");

        if is_object_type || has_properties {
            obj.insert("additionalProperties".to_string(), serde_json::json!(false));

            // OpenAI strict mode requires ALL properties to be listed in `required`
            // This overrides any existing `required` array since the derive macro
            // only includes non-optional fields, but strict mode needs all of them
            if let Some(properties) = obj.get("properties")
                && let Some(props_obj) = properties.as_object()
            {
                let required_keys: Vec<Value> =
                    props_obj.keys().map(|k| serde_json::json!(k)).collect();
                if !required_keys.is_empty() {
                    obj.insert("required".to_string(), Value::Array(required_keys));
                }
            }
        }

        // Recursively process nested schemas
        // Process 'properties' object
        if let Some(properties) = obj.get_mut("properties")
            && let Some(props_obj) = properties.as_object_mut()
        {
            for (_key, prop_schema) in props_obj.iter_mut() {
                add_additional_properties_false(prop_schema);
            }
        }

        // Process 'items' for arrays
        if let Some(items) = obj.get_mut("items") {
            add_additional_properties_false(items);
        }

        // Process 'additionalItems' for arrays
        if let Some(additional_items) = obj.get_mut("additionalItems") {
            add_additional_properties_false(additional_items);
        }

        // Process 'allOf' array
        if let Some(all_of) = obj.get_mut("allOf")
            && let Some(arr) = all_of.as_array_mut()
        {
            for item in arr.iter_mut() {
                add_additional_properties_false(item);
            }
        }

        // Process 'anyOf' array
        if let Some(any_of) = obj.get_mut("anyOf")
            && let Some(arr) = any_of.as_array_mut()
        {
            for item in arr.iter_mut() {
                add_additional_properties_false(item);
            }
        }

        // Process 'oneOf' array
        if let Some(one_of) = obj.get_mut("oneOf")
            && let Some(arr) = one_of.as_array_mut()
        {
            for item in arr.iter_mut() {
                add_additional_properties_false(item);
            }
        }

        // Process 'definitions' / '$defs' for reusable schemas
        if let Some(definitions) = obj.get_mut("definitions")
            && let Some(defs_obj) = definitions.as_object_mut()
        {
            for (_key, def_schema) in defs_obj.iter_mut() {
                add_additional_properties_false(def_schema);
            }
        }

        if let Some(defs) = obj.get_mut("$defs")
            && let Some(defs_obj) = defs.as_object_mut()
        {
            for (_key, def_schema) in defs_obj.iter_mut() {
                add_additional_properties_false(def_schema);
            }
        }

        // Process 'not' schema
        if let Some(not_schema) = obj.get_mut("not") {
            add_additional_properties_false(not_schema);
        }

        // Process 'if', 'then', 'else' schemas
        if let Some(if_schema) = obj.get_mut("if") {
            add_additional_properties_false(if_schema);
        }
        if let Some(then_schema) = obj.get_mut("then") {
            add_additional_properties_false(then_schema);
        }
        if let Some(else_schema) = obj.get_mut("else") {
            add_additional_properties_false(else_schema);
        }

        // Process 'patternProperties' object
        if let Some(pattern_props) = obj.get_mut("patternProperties")
            && let Some(pattern_obj) = pattern_props.as_object_mut()
        {
            for (_pattern, pattern_schema) in pattern_obj.iter_mut() {
                add_additional_properties_false(pattern_schema);
            }
        }

        // Process 'contains' for arrays
        if let Some(contains) = obj.get_mut("contains") {
            add_additional_properties_false(contains);
        }

        // Process 'propertyNames' schema
        if let Some(property_names) = obj.get_mut("propertyNames") {
            add_additional_properties_false(property_names);
        }
    }
}

/// Information about adjacently tagged enum transformations for response conversion
#[derive(Debug, Clone)]
pub struct AdjacentlyTaggedEnumInfo {
    pub tag_key: String,
    pub content_key: String,
    pub tag_values: Vec<String>,
}

/// Extract adjacently tagged enum info from a schema (before Gemini transformation)
/// Searches recursively through the schema tree
pub fn extract_adjacently_tagged_info(schema: &Value) -> Option<AdjacentlyTaggedEnumInfo> {
    // First check if this level has oneOf
    if let Some(one_of) = schema.get("oneOf").and_then(|v| v.as_array()) {
        let mut tag_key = None;
        let mut content_key = None;
        let mut tag_values = Vec::new();

        for variant in one_of {
            if let Some((t, c, v)) = detect_adjacently_tagged_variant(variant) {
                if tag_key.is_none() {
                    tag_key = Some(t);
                    content_key = Some(c);
                }
                tag_values.push(v);
            }
        }

        if let (Some(tag), Some(content)) = (tag_key, content_key) {
            return Some(AdjacentlyTaggedEnumInfo {
                tag_key: tag,
                content_key: content,
                tag_values,
            });
        }
    }

    // Recursively search in properties
    if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
        for (_key, prop_schema) in properties {
            if let Some(info) = extract_adjacently_tagged_info(prop_schema) {
                return Some(info);
            }
        }
    }

    // Recursively search in array items
    if let Some(items) = schema.get("items")
        && let Some(info) = extract_adjacently_tagged_info(items)
    {
        return Some(info);
    }

    // Search in allOf, anyOf, oneOf
    for key in &["allOf", "anyOf"] {
        if let Some(arr) = schema.get(key).and_then(|v| v.as_array()) {
            for item in arr {
                if let Some(info) = extract_adjacently_tagged_info(item) {
                    return Some(info);
                }
            }
        }
    }

    None
}

/// Transform internally tagged JSON back to adjacently tagged format
pub fn transform_internally_to_adjacently_tagged(
    json: &mut Value,
    enum_info: &AdjacentlyTaggedEnumInfo,
) {
    match json {
        Value::Object(obj) => {
            // Check if this object is an internally tagged enum instance
            if let Some(tag_value) = obj.get(&enum_info.tag_key).and_then(|v| v.as_str())
                && enum_info.tag_values.contains(&tag_value.to_string())
            {
                // This is an enum instance - extract all fields except the tag
                let mut content_fields = serde_json::Map::new();
                let mut keys_to_move: Vec<String> = Vec::new();

                for (key, _value) in obj.iter() {
                    if key != &enum_info.tag_key {
                        keys_to_move.push(key.clone());
                    }
                }

                // Move fields to content (unless it's a unit variant with only tag)
                if !keys_to_move.is_empty() {
                    for key in &keys_to_move {
                        if let Some(value) = obj.remove(key) {
                            content_fields.insert(key.clone(), value);
                        }
                    }

                    // Add content field
                    obj.insert(enum_info.content_key.clone(), Value::Object(content_fields));
                }
                // For unit variants (only tag), don't add content field
                return;
            }

            // Recursively process nested objects and arrays
            for value in obj.values_mut() {
                transform_internally_to_adjacently_tagged(value, enum_info);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                transform_internally_to_adjacently_tagged(item, enum_info);
            }
        }
        _ => {}
    }
}

/// Prepare a JSON schema for Gemini by stripping unsupported keywords.
///
/// Gemini's structured outputs API doesn't support certain JSON Schema keywords like
/// `examples`, `additionalProperties`, `title`, etc. This function recursively removes
/// them from the schema.
///
/// # Arguments
///
/// * `schema` - The JSON schema to modify
///
/// # Returns
///
/// A new schema Value with unsupported keywords removed
pub fn prepare_gemini_schema(schema: &crate::schema::Schema) -> Value {
    let mut schema_json = schema.to_json();
    strip_gemini_unsupported_keywords(&mut schema_json);
    schema_json
}

/// Recursively removes keywords unsupported by Gemini's structured outputs.
fn strip_gemini_unsupported_keywords(schema: &mut Value) {
    // First, resolve any $ref references by inlining definitions
    resolve_refs_for_gemini(schema);

    strip_gemini_unsupported_keywords_recursive(schema);
}

/// Resolves $ref references by inlining definitions for Gemini compatibility.
/// This handles recursive schemas by inlining to a limited depth.
fn resolve_refs_for_gemini(schema: &mut Value) {
    // Extract $defs if present
    let defs = if let Some(obj) = schema.as_object_mut() {
        obj.remove("$defs").or_else(|| obj.remove("definitions"))
    } else {
        None
    };

    if let Some(defs) = defs {
        // If schema has $ref at root, replace with the referenced definition
        if let Some(obj) = schema.as_object_mut()
            && let Some(ref_value) = obj.remove("$ref")
            && let Some(ref_str) = ref_value.as_str()
            && let Some(def_name) = ref_str
                .strip_prefix("#/$defs/")
                .or_else(|| ref_str.strip_prefix("#/definitions/"))
            && let Some(defs_obj) = defs.as_object()
            && let Some(definition) = defs_obj.get(def_name)
        {
            // Replace schema with the definition
            *schema = definition.clone();

            // Recursively inline refs in the new schema (with depth limit)
            inline_refs_recursive(schema, &defs, 3);
        }
    }
}

/// Recursively inlines $ref references with a depth limit to prevent infinite recursion.
fn inline_refs_recursive(schema: &mut Value, defs: &Value, depth: usize) {
    if depth == 0 {
        // At max depth, replace self-references with a simple object schema
        if let Some(obj) = schema.as_object_mut()
            && obj.contains_key("$ref")
        {
            obj.remove("$ref");
            obj.insert("type".to_string(), Value::String("object".to_string()));
            obj.insert(
                "description".to_string(),
                Value::String("Recursive reference (depth limit reached)".to_string()),
            );
        }
        return;
    }

    if let Some(obj) = schema.as_object_mut() {
        // Replace $ref with inline definition
        if let Some(ref_value) = obj.remove("$ref")
            && let Some(ref_str) = ref_value.as_str()
            && let Some(def_name) = ref_str
                .strip_prefix("#/$defs/")
                .or_else(|| ref_str.strip_prefix("#/definitions/"))
            && let Some(defs_obj) = defs.as_object()
            && let Some(definition) = defs_obj.get(def_name)
        {
            // Replace with inline definition
            *schema = definition.clone();
            // Continue recursively with reduced depth
            inline_refs_recursive(schema, defs, depth - 1);
            return;
        }

        // Process nested schemas
        if let Some(properties) = obj.get_mut("properties")
            && let Some(props_obj) = properties.as_object_mut()
        {
            for prop_schema in props_obj.values_mut() {
                inline_refs_recursive(prop_schema, defs, depth);
            }
        }

        if let Some(items) = obj.get_mut("items") {
            inline_refs_recursive(items, defs, depth);
        }

        if let Some(prefix_items) = obj.get_mut("prefixItems")
            && let Some(arr) = prefix_items.as_array_mut()
        {
            for item in arr.iter_mut() {
                inline_refs_recursive(item, defs, depth);
            }
        }

        if let Some(one_of) = obj.get_mut("oneOf")
            && let Some(arr) = one_of.as_array_mut()
        {
            for item in arr.iter_mut() {
                inline_refs_recursive(item, defs, depth);
            }
        }

        if let Some(any_of) = obj.get_mut("anyOf")
            && let Some(arr) = any_of.as_array_mut()
        {
            for item in arr.iter_mut() {
                inline_refs_recursive(item, defs, depth);
            }
        }

        if let Some(all_of) = obj.get_mut("allOf")
            && let Some(arr) = all_of.as_array_mut()
        {
            for item in arr.iter_mut() {
                inline_refs_recursive(item, defs, depth);
            }
        }

        // Handle additionalProperties if it's a schema object (for maps)
        if let Some(additional) = obj.get_mut("additionalProperties")
            && additional.is_object()
        {
            inline_refs_recursive(additional, defs, depth);
        }
    }
}

/// Detects if a oneOf variant looks like an adjacently tagged enum variant.
/// Returns Some((tag_key, content_key, tag_value)) if it matches the pattern.
fn detect_adjacently_tagged_variant(variant: &Value) -> Option<(String, String, String)> {
    let obj = variant.as_object()?;

    // Must be an object type
    if obj.get("type")?.as_str()? != "object" {
        return None;
    }

    let properties = obj.get("properties")?.as_object()?;
    let required = obj.get("required")?.as_array()?;

    // Must have exactly 2 required fields
    if required.len() != 2 {
        return None;
    }

    // Find the tag field (has enum with single value) and content field (is object)
    let mut tag_key = None;
    let mut tag_value = None;
    let mut content_key = None;

    for (key, prop) in properties.iter() {
        if let Some(prop_obj) = prop.as_object() {
            // Check if it's a tag field (has enum with single value)
            if let Some(enum_array) = prop_obj.get("enum").and_then(|e| e.as_array())
                && enum_array.len() == 1
                && let Some(val) = enum_array[0].as_str()
            {
                tag_key = Some(key.clone());
                tag_value = Some(val.to_string());
                continue;
            }

            // Check if it's a content field (is object type)
            if prop_obj.get("type").and_then(|t| t.as_str()) == Some("object")
                && prop_obj.contains_key("properties")
            {
                content_key = Some(key.clone());
            }
        }
    }

    // Must have found both tag and content
    if let (Some(tag), Some(content), Some(value)) = (tag_key, content_key, tag_value) {
        Some((tag, content, value))
    } else {
        None
    }
}

/// Transforms adjacently tagged enum variants to internally tagged format for Gemini.
/// This is a workaround for Gemini's limitation with nested content objects.
fn transform_adjacently_tagged_to_internally_tagged(
    variant: &Value,
    _tag_key: &str,
    content_key: &str,
    _tag_value: &str,
) -> Value {
    let mut obj = variant.as_object().unwrap().clone();

    // Clone content properties and required fields before modifying
    let content_props_to_add: Vec<(String, Value)>;
    let content_required: Vec<Value>;

    {
        let properties = obj.get("properties").unwrap().as_object().unwrap();

        // Get the content object properties
        if let Some(content_obj) = properties.get(content_key).and_then(|c| c.as_object()) {
            if let Some(content_props) = content_obj.get("properties").and_then(|p| p.as_object()) {
                // Collect properties to add
                content_props_to_add = content_props
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
            } else {
                content_props_to_add = Vec::new();
            }

            // Collect required fields
            if let Some(req) = content_obj.get("required").and_then(|r| r.as_array()) {
                content_required = req.clone();
            } else {
                content_required = Vec::new();
            }
        } else {
            content_props_to_add = Vec::new();
            content_required = Vec::new();
        }
    }

    // Now modify properties
    if let Some(properties) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
        // Add flattened properties
        for (key, value) in content_props_to_add {
            properties.insert(key, value);
        }

        // Remove the content field itself
        properties.remove(content_key);
    }

    // Update required array
    if let Some(required_array) = obj.get_mut("required").and_then(|r| r.as_array_mut()) {
        // Remove content_key from required
        required_array.retain(|v| v.as_str() != Some(content_key));

        // Add content's required fields
        for field in content_required {
            if !required_array.contains(&field) {
                required_array.push(field);
            }
        }
    }

    // Update description to note the transformation
    if let Some(desc) = obj.get("description").and_then(|d| d.as_str()) {
        let new_desc = format!("{} (flattened for Gemini compatibility)", desc);
        obj.insert("description".to_string(), Value::String(new_desc));
    }

    Value::Object(obj)
}

/// Internal function that strips unsupported keywords after refs are resolved.
fn strip_gemini_unsupported_keywords_recursive(schema: &mut Value) {
    if let Some(obj) = schema.as_object_mut() {
        // Remove unsupported keywords
        obj.remove("examples");
        obj.remove("title");
        obj.remove("$schema");
        obj.remove("$id");
        obj.remove("default");
        obj.remove("$defs");
        obj.remove("definitions");
        obj.remove("$ref"); // Should be resolved by now, but remove if any remain

        // Handle additionalProperties: remove if boolean, keep if it's a schema for maps
        if let Some(additional) = obj.get("additionalProperties")
            && additional.is_boolean()
        {
            obj.remove("additionalProperties");
        }

        // For object types with additionalProperties but no properties, this is a Map type
        // Gemini requires properties to be non-empty for object types
        // Since Gemini doesn't support map types natively, we remove type constraint
        // and add a description. The response won't be strictly validated but should
        // parse correctly.
        let is_object = obj.get("type").and_then(|t| t.as_str()) == Some("object");
        let has_properties = obj.contains_key("properties");
        let has_additional_props = obj.contains_key("additionalProperties");

        if is_object && !has_properties && has_additional_props {
            // This is a map type - Gemini doesn't support this natively
            // We need to generate a workaround schema that Gemini can understand
            let additional = obj.remove("additionalProperties");

            // Get existing description if any
            let existing_desc = obj
                .get("description")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string())
                .unwrap_or_default();

            // For Gemini, we'll define a few placeholder property keys that represent
            // the expected dynamic key structure. This is a workaround since Gemini
            // requires properties to be defined.
            let value_schema = additional.unwrap_or(serde_json::json!({}));

            // Try to extract specific keys from description (e.g., "Keys: [info, warn, error]")
            // This allows us to provide concrete enum variants as keys instead of generic placeholders
            let mut keys = vec!["key1".to_string(), "key2".to_string(), "key3".to_string()];
            if let Some(start) = existing_desc.find("Keys: [")
                && let Some(end) = existing_desc[start..].find(']')
            {
                let keys_str = &existing_desc[start + 7..start + end];
                let extracted_keys: Vec<String> = keys_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if !extracted_keys.is_empty() {
                    keys = extracted_keys;
                }
            }

            // Create placeholder properties with the value schema
            // Using extracted keys if available, otherwise generic key names
            let mut placeholder_props = serde_json::Map::new();
            for key in &keys {
                placeholder_props.insert(key.clone(), value_schema.clone());
            }

            obj.insert("properties".to_string(), Value::Object(placeholder_props));

            // Update description to explain this is a map with specific or example keys
            let map_desc = if existing_desc.contains("Keys: [") {
                // If keys were specified, keep the original description as-is
                existing_desc
            } else if existing_desc.is_empty() {
                format!(
                    "Object with any string keys ({} are examples - use actual meaningful key names)",
                    keys.join(", ")
                )
            } else {
                format!(
                    "{} ({} are example keys - use actual meaningful key names)",
                    existing_desc,
                    keys.join(", ")
                )
            };
            obj.insert("description".to_string(), Value::String(map_desc));
        }

        // Recursively process nested schemas
        if let Some(properties) = obj.get_mut("properties")
            && let Some(props_obj) = properties.as_object_mut()
        {
            for prop_schema in props_obj.values_mut() {
                strip_gemini_unsupported_keywords_recursive(prop_schema);
            }
        }

        // Process 'items' for arrays
        if let Some(items) = obj.get_mut("items") {
            strip_gemini_unsupported_keywords_recursive(items);
        }

        // Handle tuples (prefixItems) - Gemini doesn't support prefixItems
        // Convert to a regular array with oneOf for the item types
        if let Some(prefix_items) = obj.remove("prefixItems")
            && let Some(arr) = prefix_items.as_array()
        {
            // Recursively process each item schema
            let mut processed_items: Vec<Value> = arr
                .iter()
                .map(|item| {
                    let mut item_clone = item.clone();
                    strip_gemini_unsupported_keywords_recursive(&mut item_clone);
                    item_clone
                })
                .collect();

            // Remove duplicates for cleaner schema
            processed_items.dedup();

            // If all items are the same type, use single items schema
            if processed_items.len() == 1 {
                obj.insert(
                    "items".to_string(),
                    processed_items.into_iter().next().unwrap(),
                );
            } else {
                // Use anyOf for mixed types
                obj.insert(
                    "items".to_string(),
                    serde_json::json!({
                        "anyOf": processed_items
                    }),
                );
            }

            // Remove minItems/maxItems since they're not strictly enforced without prefixItems
            obj.remove("minItems");
            obj.remove("maxItems");

            // Add description about tuple structure
            let existing_desc = obj
                .get("description")
                .and_then(|d| d.as_str())
                .map(|s| format!("{}. ", s))
                .unwrap_or_default();
            let tuple_len = arr.len();
            obj.insert(
                "description".to_string(),
                Value::String(format!(
                    "{}Fixed-length array (tuple) with {} elements",
                    existing_desc, tuple_len
                )),
            );
        }

        // Process 'allOf' array
        if let Some(all_of) = obj.get_mut("allOf")
            && let Some(arr) = all_of.as_array_mut()
        {
            for item in arr.iter_mut() {
                strip_gemini_unsupported_keywords_recursive(item);
            }
        }

        // Process 'anyOf' array
        if let Some(any_of) = obj.get_mut("anyOf")
            && let Some(arr) = any_of.as_array_mut()
        {
            for item in arr.iter_mut() {
                strip_gemini_unsupported_keywords_recursive(item);
            }
        }

        // Process 'oneOf' array
        if let Some(one_of) = obj.get_mut("oneOf")
            && let Some(arr) = one_of.as_array_mut()
        {
            // First, check if this looks like an adjacently tagged enum
            // All variants should have the same tag/content keys
            let mut adjacently_tagged_info: Option<(String, String)> = None;
            let mut all_adjacently_tagged = true;

            for item in arr.iter() {
                if let Some((tag_key, content_key, _tag_value)) =
                    detect_adjacently_tagged_variant(item)
                {
                    if let Some((ref existing_tag, ref existing_content)) = adjacently_tagged_info {
                        // Check if keys match
                        if tag_key != *existing_tag || content_key != *existing_content {
                            all_adjacently_tagged = false;
                            break;
                        }
                    } else {
                        adjacently_tagged_info = Some((tag_key, content_key));
                    }
                } else {
                    // Unit variant (only tag, no content) is still okay
                    // Check if it has just one required field with enum
                    if let Some(variant_obj) = item.as_object()
                        && let Some(props) =
                            variant_obj.get("properties").and_then(|p| p.as_object())
                        && props.len() == 1
                        && variant_obj
                            .get("required")
                            .and_then(|r| r.as_array())
                            .map(|a| a.len())
                            == Some(1)
                    {
                        // This is a unit variant, keep checking
                        continue;
                    }
                    all_adjacently_tagged = false;
                    break;
                }
            }

            // If all variants are adjacently tagged, transform them
            if all_adjacently_tagged && adjacently_tagged_info.is_some() {
                // Transform each variant
                *arr = arr
                    .iter()
                    .map(|item| {
                        if let Some((t, c, v)) = detect_adjacently_tagged_variant(item) {
                            transform_adjacently_tagged_to_internally_tagged(item, &t, &c, &v)
                        } else {
                            // Unit variant - leave as is
                            item.clone()
                        }
                    })
                    .collect();
            }

            // Now recursively process all variants
            for item in arr.iter_mut() {
                strip_gemini_unsupported_keywords_recursive(item);
            }
        }

        // Handle additionalProperties if it's a schema object (for maps) - recurse into it
        if let Some(additional) = obj.get_mut("additionalProperties")
            && additional.is_object()
        {
            strip_gemini_unsupported_keywords_recursive(additional);
        }
    }
}

/// JSON Schema format specification for structured outputs.
///
/// This struct is used by OpenAI and Grok (and potentially other OpenAI-compatible APIs)
/// for their native structured outputs feature.
#[derive(Debug, Serialize)]
pub struct JsonSchemaFormat {
    /// Name of the schema (usually the type name)
    pub name: String,
    /// Optional description of the schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The JSON schema itself
    pub schema: Value,
    /// Whether to use strict mode (required for structured outputs)
    pub strict: bool,
}

/// Response format for structured outputs (OpenAI-compatible).
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    /// JSON Schema structured output format
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// The JSON schema specification
        json_schema: JsonSchemaFormat,
    },
}

impl ResponseFormat {
    /// Create a new JSON schema response format for structured outputs.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the schema (usually the type name)
    /// * `schema` - The JSON schema for the expected output
    /// * `description` - Optional description of what the output should contain
    pub fn json_schema(name: String, schema: Value, description: Option<String>) -> Self {
        ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name,
                description,
                schema,
                strict: true,
            },
        }
    }
}

/// Parse a raw JSON response and validate it against the Instructor trait.
///
/// This function handles:
/// 1. JSON parsing with detailed error messages
/// 2. Custom validation via the Instructor trait
///
/// # Arguments
///
/// * `raw_response` - The raw JSON string from the LLM
///
/// # Returns
///
/// The parsed and validated data, or an error with validation context
pub fn parse_and_validate_response<T>(
    raw_response: &str,
) -> std::result::Result<T, (RStructorError, Option<ValidationFailureContext>)>
where
    T: Instructor + DeserializeOwned,
{
    // Parse the JSON content into our target type
    let result: T = match serde_json::from_str(raw_response) {
        Ok(parsed) => parsed,
        Err(e) => {
            let error_msg = format!(
                "Failed to parse response as JSON: {}\nPartial JSON: {}",
                e, raw_response
            );
            error!(
                error = %e,
                content = %raw_response,
                "JSON parsing error"
            );
            return Err((
                RStructorError::ValidationError(error_msg.clone()),
                Some(ValidationFailureContext::new(
                    error_msg,
                    raw_response.to_string(),
                )),
            ));
        }
    };

    // Apply any custom validation (business logic beyond schema)
    if let Err(e) = result.validate() {
        error!(error = ?e, "Custom validation failed");
        let error_msg = e.to_string();
        return Err((
            e,
            Some(ValidationFailureContext::new(
                error_msg,
                raw_response.to_string(),
            )),
        ));
    }

    Ok(result)
}

/// Helper to create a successful MaterializeInternalOutput from parsed data.
///
/// This is a convenience function that combines parsing, validation, and
/// output construction in one step.
///
/// # Arguments
///
/// * `raw_response` - The raw JSON string from the LLM
/// * `usage` - Optional token usage information
///
/// # Returns
///
/// A MaterializeInternalOutput with the parsed data, or an error
pub fn parse_validate_and_create_output<T>(
    raw_response: String,
    usage: Option<TokenUsage>,
) -> std::result::Result<
    MaterializeInternalOutput<T>,
    (RStructorError, Option<ValidationFailureContext>),
>
where
    T: Instructor + DeserializeOwned,
{
    let result = parse_and_validate_response::<T>(&raw_response)?;
    info!("Successfully generated and validated structured data");
    Ok(MaterializeInternalOutput::new(result, raw_response, usage))
}

/// Convert a reqwest error to a RStructorError, handling timeout errors specially.
pub fn handle_http_error(e: reqwest::Error, provider_name: &str) -> RStructorError {
    error!(error = %e, "HTTP request to {} failed", provider_name);
    if e.is_timeout() {
        RStructorError::Timeout
    } else {
        RStructorError::HttpError(e)
    }
}

/// Parse retry-after header value to Duration.
fn parse_retry_after(value: &str) -> Option<Duration> {
    // Try parsing as seconds (most common)
    if let Ok(secs) = value.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    // Could also parse HTTP-date format, but seconds is most common
    None
}

/// Classify an API error based on HTTP status code and response body.
fn classify_api_error(
    status: reqwest::StatusCode,
    error_text: &str,
    retry_after: Option<Duration>,
    model_hint: Option<&str>,
) -> ApiErrorKind {
    let code = status.as_u16();
    let error_lower = error_text.to_lowercase();

    match code {
        // Authentication errors
        401 => ApiErrorKind::AuthenticationFailed,

        // Permission errors
        403 => ApiErrorKind::PermissionDenied,

        // Not found - check if it's a model error
        404 => {
            // Check if the error message mentions "model"
            if error_lower.contains("model") {
                let model = model_hint
                    .map(|s| s.to_string())
                    .or_else(|| extract_model_from_error(&error_lower))
                    .unwrap_or_else(|| "unknown".to_string());
                ApiErrorKind::InvalidModel {
                    model,
                    suggestion: suggest_model(&error_lower),
                }
            } else {
                ApiErrorKind::Other {
                    code,
                    message: error_text.to_string(),
                }
            }
        }

        // Bad request
        400 => ApiErrorKind::BadRequest {
            details: truncate_message(error_text, 200),
        },

        // Payload too large
        413 => ApiErrorKind::RequestTooLarge,

        // Rate limited
        429 => ApiErrorKind::RateLimited { retry_after },

        // Server errors
        500 | 502 => ApiErrorKind::ServerError { code },

        // Service unavailable
        503 => ApiErrorKind::ServiceUnavailable,

        // Gateway/Cloudflare errors
        520..=524 => ApiErrorKind::GatewayError { code },

        // Other errors
        _ => ApiErrorKind::Other {
            code,
            message: truncate_message(error_text, 500),
        },
    }
}

/// Extract model name from error message if present.
fn extract_model_from_error(error_text: &str) -> Option<String> {
    // Look for quoted model names like 'gpt-4' or "gpt-4"
    for quote in ['\'', '"'] {
        if let Some(start) = error_text.find(quote) {
            let rest = &error_text[start + 1..];
            if let Some(end) = rest.find(quote) {
                let candidate = &rest[..end];
                // Model names typically have alphanumeric chars, dots, or dashes
                if candidate.len() > 2
                    && candidate
                        .chars()
                        .all(|c| c.is_alphanumeric() || c == '-' || c == '.' || c == '_')
                {
                    return Some(candidate.to_string());
                }
            }
        }
    }
    None
}

/// Suggest an alternative model based on error context.
fn suggest_model(error_text: &str) -> Option<String> {
    // Common model name patterns and their suggestions
    if error_text.contains("gpt") {
        Some("gpt-5.2".to_string())
    } else if error_text.contains("claude") || error_text.contains("sonnet") {
        Some("claude-sonnet-4-5-20250929".to_string())
    } else if error_text.contains("gemini") {
        Some("gemini-3-flash-preview".to_string())
    } else {
        None
    }
}

/// Truncate a message to a maximum length.
///
/// Uses `floor_char_boundary` to ensure we don't slice in the middle of a
/// multi-byte UTF-8 character, which would cause a panic.
fn truncate_message(msg: &str, max_len: usize) -> String {
    if msg.len() <= max_len {
        msg.to_string()
    } else {
        // Find a valid UTF-8 character boundary at or before max_len
        let boundary = msg.floor_char_boundary(max_len);
        format!("{}...", &msg[..boundary])
    }
}

/// Check HTTP response status and extract error message if unsuccessful.
///
/// This function classifies errors into actionable types (rate limit, auth failure, etc.)
/// and provides user-friendly error messages with suggested actions.
pub async fn check_response_status(response: Response, provider_name: &str) -> Result<Response> {
    if !response.status().is_success() {
        let status = response.status();

        // Extract retry-after header if present
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_retry_after);

        let error_text = response.text().await?;

        let kind = classify_api_error(status, &error_text, retry_after, None);

        error!(
            status = %status,
            error = %error_text,
            kind = %kind,
            "{} API returned error response", provider_name
        );

        return Err(RStructorError::api_error(provider_name, kind));
    }
    Ok(response)
}

/// Helper function to execute generation with retry logic using conversation history.
///
/// This function maintains a conversation history across retry attempts, which enables:
/// - **Prompt caching**: Providers like Anthropic and OpenAI can cache the prefix of the
///   conversation, reducing token costs and latency on retries.
/// - **Better error correction**: The model sees its previous (failed) response and the
///   specific error, making it more likely to produce a correct response.
///
/// # How it works
///
/// 1. On first attempt: Sends `[User(prompt)]`
/// 2. On validation failure: Appends `[Assistant(failed_response), User(error_feedback)]`
/// 3. On retry: Sends the full conversation history
///
/// This approach preserves the original prompt exactly, maximizing cache hit rates.
///
/// # Arguments
///
/// * `generate_fn` - Function that takes a conversation history and returns the result plus raw response
/// * `prompt` - The initial user prompt
/// * `max_retries` - Maximum number of retry attempts (None or 0 means no retries)
pub async fn generate_with_retry_with_history<F, Fut, T>(
    mut generate_fn: F,
    prompt: &str,
    max_retries: Option<usize>,
) -> Result<MaterializeInternalOutput<T>>
where
    F: FnMut(Vec<ChatMessage>) -> Fut,
    Fut: std::future::Future<
            Output = std::result::Result<
                MaterializeInternalOutput<T>,
                (RStructorError, Option<ValidationFailureContext>),
            >,
        >,
{
    let Some(max_retries) = max_retries.filter(|&n| n > 0) else {
        // No retries configured - just run once with a single user message
        let messages = vec![ChatMessage::user(prompt)];
        return generate_fn(messages).await.map_err(|(err, _)| err);
    };

    let max_attempts = max_retries + 1; // +1 for initial attempt

    // Initialize conversation history with the original user prompt
    let mut messages = vec![ChatMessage::user(prompt)];

    trace!(
        "Starting structured generation with conversation history: max_attempts={}",
        max_attempts
    );

    for attempt in 0..max_attempts {
        // Log attempt information
        info!(
            attempt = attempt + 1,
            total_attempts = max_attempts,
            history_len = messages.len(),
            "Generation attempt with conversation history"
        );

        // Attempt to generate structured data
        match generate_fn(messages.clone()).await {
            Ok(result) => {
                if attempt > 0 {
                    info!(
                        attempts_used = attempt + 1,
                        "Successfully generated after {} retries (with conversation history)",
                        attempt
                    );
                } else {
                    debug!("Successfully generated on first attempt");
                }
                return Ok(result);
            }
            Err((err, validation_ctx)) => {
                let is_last_attempt = attempt >= max_attempts - 1;

                // Handle validation errors with conversation history
                if let RStructorError::ValidationError(ref msg) = err {
                    if !is_last_attempt {
                        warn!(
                            attempt = attempt + 1,
                            error = msg,
                            "Validation error in generation attempt"
                        );

                        // Build conversation history for retry with error feedback
                        if let Some(ctx) = validation_ctx {
                            // Add the failed assistant response to history
                            messages.push(ChatMessage::assistant(&ctx.raw_response));

                            // Add user message with error feedback
                            let error_feedback = format!(
                                "Your previous response contained validation errors. Please provide a complete, valid JSON response that includes ALL required fields and follows the schema exactly.\n\nError details:\n{}\n\nPlease fix the issues in your response. Make sure to:\n1. Include ALL required fields exactly as specified in the schema\n2. For enum fields, use EXACTLY one of the allowed values from the description\n3. CRITICAL: For arrays where items.type = 'object':\n   - You MUST provide an array of OBJECTS, not strings or primitive values\n   - Each object must be a complete JSON object with all its required fields\n   - Include multiple items (at least 2-3) in arrays of objects\n4. Verify all nested objects have their complete structure\n5. Follow ALL type specifications (string, number, boolean, array, object)",
                                ctx.error_message
                            );
                            messages.push(ChatMessage::user(error_feedback));

                            debug!(
                                history_len = messages.len(),
                                "Updated conversation history for retry"
                            );
                        } else {
                            // Fallback: no raw response context available.
                            // We cannot add error feedback without the raw response because:
                            // 1. Adding only a user message would create consecutive user messages,
                            //    violating the alternating user/assistant pattern expected by LLM APIs
                            // 2. The error message references "your previous response" but we can't show it
                            // Instead, we retry with the original conversation (no history modification)
                            warn!(
                                "Validation error occurred but no raw response context available. \
                                 Retrying without error feedback in conversation history."
                            );
                        }

                        // Wait briefly before retrying
                        sleep(Duration::from_millis(500)).await;
                        continue;
                    } else {
                        error!(
                            attempts = max_attempts,
                            error = msg,
                            "Failed after maximum retry attempts with validation errors"
                        );
                    }
                }
                // Handle retryable API errors (rate limits, transient failures)
                else if err.is_retryable() && !is_last_attempt {
                    let delay = err.retry_delay().unwrap_or(Duration::from_secs(1));
                    warn!(
                        attempt = attempt + 1,
                        error = ?err,
                        delay_ms = delay.as_millis(),
                        "Retryable API error, waiting before retry"
                    );
                    // For API errors, we don't modify the conversation history
                    // Just retry with the same messages
                    sleep(delay).await;
                    continue;
                }
                // Non-retryable errors or last attempt
                else if is_last_attempt {
                    error!(
                        attempts = max_attempts,
                        error = ?err,
                        "Failed after maximum retry attempts"
                    );
                } else {
                    error!(
                        error = ?err,
                        "Non-retryable error occurred during generation"
                    );
                }

                return Err(err);
            }
        }
    }

    // This should never be reached due to the returns in the loop
    unreachable!()
}

/// Macro to generate standard builder methods for LLM clients.
///
/// This macro generates `model()`, `temperature()`, `max_tokens()`, and `timeout()` methods
/// that are identical across all LLM client implementations.
#[macro_export]
macro_rules! impl_client_builder_methods {
    (
        client_type: $client:ty,
        config_type: $config:ty,
        model_type: $model:ty,
        provider_name: $provider:expr
    ) => {
        impl $client {
            /// Set the model to use. Accepts either a Model enum variant or a string.
            ///
            /// When a string is provided, it will be converted to a Model enum. If the string
            /// matches a known model variant, that variant is used; otherwise, it becomes `Custom(name)`.
            /// This allows using any model name, including new models or local LLMs, without needing
            /// to update the enum.
            #[tracing::instrument(skip(self, model))]
            pub fn model<M: Into<$model>>(mut self, model: M) -> Self {
                let model = model.into();
                tracing::debug!(
                    previous_model = ?self.config.model,
                    new_model = ?model,
                    "Setting {} model", $provider
                );
                self.config.model = model;
                self
            }

            /// Set the temperature (0.0 to 1.0, lower = more deterministic)
            #[tracing::instrument(skip(self))]
            pub fn temperature(mut self, temp: f32) -> Self {
                tracing::debug!(
                    previous_temp = self.config.temperature,
                    new_temp = temp,
                    "Setting temperature"
                );
                self.config.temperature = temp;
                self
            }

            /// Set the maximum tokens to generate
            #[tracing::instrument(skip(self))]
            pub fn max_tokens(mut self, max: u32) -> Self {
                tracing::debug!(
                    previous_max = ?self.config.max_tokens,
                    new_max = max,
                    "Setting max_tokens"
                );
                // Ensure max_tokens is at least 1 to avoid API errors
                self.config.max_tokens = Some(max.max(1));
                self
            }

            /// Set the timeout for HTTP requests.
            ///
            /// This sets the timeout for both the connection and the entire request.
            /// The timeout applies to each HTTP request made by the client.
            ///
            /// # Arguments
            ///
            /// * `timeout` - Timeout duration (e.g., `Duration::from_secs(30)` for 30 seconds)
            #[tracing::instrument(skip(self))]
            pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
                tracing::debug!(
                    previous_timeout = ?self.config.timeout,
                    new_timeout = ?timeout,
                    "Setting timeout"
                );
                self.config.timeout = Some(timeout);

                // Rebuild reqwest client with timeout immediately
                self.client = reqwest::Client::builder()
                    .timeout(timeout)
                    .build()
                    .unwrap_or_else(|e| {
                        tracing::warn!(
                            error = %e,
                            "Failed to build reqwest client with timeout, using default"
                        );
                        reqwest::Client::new()
                    });

                self
            }

            /// Set the maximum number of retry attempts for validation errors.
            ///
            /// When `materialize` encounters a validation error, it will automatically
            /// retry up to this many times, including the validation error message in subsequent attempts.
            ///
            /// The default is 3 retries. Use `.no_retries()` to disable retries entirely.
            ///
            /// # Arguments
            ///
            /// * `max_retries` - Maximum number of retry attempts (0 = no retries, only single attempt)
            ///
            /// # Examples
            ///
            /// ```no_run
            /// # use rstructor::OpenAIClient;
            /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
            /// let client = OpenAIClient::new("api-key")?
            ///     .max_retries(5);  // Increase to 5 retries (default is 3)
            /// # Ok(())
            /// # }
            /// ```
            #[tracing::instrument(skip(self))]
            pub fn max_retries(mut self, max_retries: usize) -> Self {
                tracing::debug!(
                    previous_max_retries = ?self.config.max_retries,
                    new_max_retries = max_retries,
                    "Setting max_retries"
                );
                self.config.max_retries = Some(max_retries);
                self
            }

            /// Disable automatic retries on validation errors.
            ///
            /// By default, the client retries up to 3 times when validation errors occur.
            /// Use this method to disable retries and fail immediately on the first error.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// # use rstructor::OpenAIClient;
            /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
            /// let client = OpenAIClient::new("api-key")?
            ///     .no_retries();  // Fail immediately on validation errors
            /// # Ok(())
            /// # }
            /// ```
            #[tracing::instrument(skip(self))]
            pub fn no_retries(mut self) -> Self {
                tracing::debug!(
                    previous_max_retries = ?self.config.max_retries,
                    "Disabling retries"
                );
                self.config.max_retries = Some(0);
                self
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_additional_properties_simple_object() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });
        add_additional_properties_false(&mut schema);
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
    }

    #[test]
    fn test_add_additional_properties_nested_object() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    }
                }
            }
        });
        add_additional_properties_false(&mut schema);
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
        assert_eq!(
            schema["properties"]["user"]["additionalProperties"],
            serde_json::json!(false)
        );
    }

    #[test]
    fn test_add_additional_properties_array_items() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "ingredients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "amount": { "type": "string" }
                        }
                    }
                }
            }
        });
        add_additional_properties_false(&mut schema);
        // Top-level should have additionalProperties: false
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
        // Array items object should also have additionalProperties: false
        assert_eq!(
            schema["properties"]["ingredients"]["items"]["additionalProperties"],
            serde_json::json!(false)
        );
    }

    #[test]
    fn test_add_additional_properties_deeply_nested() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "recipe": {
                    "type": "object",
                    "properties": {
                        "ingredients": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "details": {
                                        "type": "object",
                                        "properties": {
                                            "brand": { "type": "string" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        add_additional_properties_false(&mut schema);

        // All object levels should have additionalProperties: false
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
        assert_eq!(
            schema["properties"]["recipe"]["additionalProperties"],
            serde_json::json!(false)
        );
        assert_eq!(
            schema["properties"]["recipe"]["properties"]["ingredients"]["items"]["additionalProperties"],
            serde_json::json!(false)
        );
        assert_eq!(
            schema["properties"]["recipe"]["properties"]["ingredients"]["items"]["properties"]["details"]
                ["additionalProperties"],
            serde_json::json!(false)
        );
    }

    #[test]
    fn test_add_additional_properties_anyof() {
        let mut schema = serde_json::json!({
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    }
                },
                {
                    "type": "object",
                    "properties": {
                        "id": { "type": "number" }
                    }
                }
            ]
        });
        add_additional_properties_false(&mut schema);

        assert_eq!(
            schema["anyOf"][0]["additionalProperties"],
            serde_json::json!(false)
        );
        assert_eq!(
            schema["anyOf"][1]["additionalProperties"],
            serde_json::json!(false)
        );
    }

    #[test]
    fn test_add_additional_properties_definitions() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "item": { "$ref": "#/definitions/Item" }
            },
            "definitions": {
                "Item": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    }
                }
            }
        });
        add_additional_properties_false(&mut schema);

        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
        assert_eq!(
            schema["definitions"]["Item"]["additionalProperties"],
            serde_json::json!(false)
        );
    }

    #[test]
    fn test_add_additional_properties_preserves_existing() {
        // If additionalProperties is already set, it should be overwritten
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "additionalProperties": true
        });
        add_additional_properties_false(&mut schema);
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
    }

    #[test]
    fn test_add_additional_properties_no_type() {
        // Object with properties but no explicit type should still get additionalProperties: false
        let mut schema = serde_json::json!({
            "properties": {
                "name": { "type": "string" }
            }
        });
        add_additional_properties_false(&mut schema);
        assert_eq!(schema["additionalProperties"], serde_json::json!(false));
    }

    #[test]
    fn test_adds_required_array() {
        // Schema without required array should get one added with all property keys
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "number" }
            }
        });
        add_additional_properties_false(&mut schema);

        let required = schema["required"]
            .as_array()
            .expect("required should be an array");
        assert_eq!(required.len(), 2);
        assert!(required.contains(&serde_json::json!("name")));
        assert!(required.contains(&serde_json::json!("age")));
    }

    #[test]
    fn test_overrides_existing_required_array() {
        // Schema with existing required array should be overridden to include all properties
        // (OpenAI strict mode requires ALL properties in required, even optional ones)
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "number" }
            },
            "required": ["name"]
        });
        add_additional_properties_false(&mut schema);

        let required = schema["required"]
            .as_array()
            .expect("required should be an array");
        // Now it should include ALL properties, not just the original
        assert_eq!(required.len(), 2);
        assert!(required.contains(&serde_json::json!("name")));
        assert!(required.contains(&serde_json::json!("age")));
    }

    #[test]
    fn test_adds_required_array_to_nested_objects() {
        // Nested objects should also get required arrays
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "number": { "type": "integer" },
                            "description": { "type": "string" }
                        }
                    }
                }
            }
        });
        add_additional_properties_false(&mut schema);

        // Top-level should have required
        let required = schema["required"]
            .as_array()
            .expect("required should be an array");
        assert!(required.contains(&serde_json::json!("steps")));

        // Nested array items should also have required
        let nested_required = schema["properties"]["steps"]["items"]["required"]
            .as_array()
            .expect("nested required should be an array");
        assert!(nested_required.contains(&serde_json::json!("number")));
        assert!(nested_required.contains(&serde_json::json!("description")));
    }

    #[test]
    fn truncate_message_ascii_within_limit() {
        let msg = "Hello, world!";
        assert_eq!(truncate_message(msg, 20), "Hello, world!");
    }

    #[test]
    fn truncate_message_ascii_exact_limit() {
        let msg = "Hello";
        assert_eq!(truncate_message(msg, 5), "Hello");
    }

    #[test]
    fn truncate_message_ascii_exceeds_limit() {
        let msg = "Hello, world!";
        assert_eq!(truncate_message(msg, 5), "Hello...");
    }

    #[test]
    fn truncate_message_utf8_within_limit() {
        let msg = "你好世界"; // 12 bytes (3 bytes per character)
        assert_eq!(truncate_message(msg, 20), "你好世界");
    }

    #[test]
    fn truncate_message_utf8_boundary_safe() {
        // "你好世界" is 12 bytes total (3 bytes per character)
        // Truncating at 5 bytes would fall in the middle of the second character
        // floor_char_boundary(5) should return 3 (end of first character)
        let msg = "你好世界";
        let result = truncate_message(msg, 5);
        assert_eq!(result, "你...");
    }

    #[test]
    fn truncate_message_utf8_exact_boundary() {
        // Truncating at exactly 6 bytes should include first two characters
        let msg = "你好世界";
        let result = truncate_message(msg, 6);
        assert_eq!(result, "你好...");
    }

    #[test]
    fn truncate_message_emoji() {
        // Emojis are typically 4 bytes each
        let msg = "🎉🎊🎈";
        // Truncating at 5 bytes falls in the middle of second emoji
        // floor_char_boundary(5) should return 4 (end of first emoji)
        let result = truncate_message(msg, 5);
        assert_eq!(result, "🎉...");
    }

    #[test]
    fn truncate_message_mixed_utf8() {
        let msg = "Error: 无效的请求";
        // "Error: " is 7 bytes, then Chinese characters are 3 bytes each
        // Truncating at 10 bytes falls at the boundary after the first Chinese char
        // floor_char_boundary(10) should return 10 (end of first Chinese char after "Error: ")
        let result = truncate_message(msg, 10);
        assert_eq!(result, "Error: 无...");
    }

    #[test]
    fn truncate_message_empty_string() {
        let msg = "";
        assert_eq!(truncate_message(msg, 10), "");
    }

    #[test]
    fn truncate_message_zero_limit() {
        let msg = "Hello";
        // floor_char_boundary(0) returns 0, so we get just "..."
        assert_eq!(truncate_message(msg, 0), "...");
    }

    #[test]
    fn test_gemini_schema_strips_unsupported_keywords() {
        use crate::schema::Schema;

        // Create a schema with examples and other unsupported keywords
        let schema = Schema::new(serde_json::json!({
            "type": "object",
            "title": "Movie",
            "properties": {
                "title": { "type": "string", "description": "Movie title" },
                "year": { "type": "integer", "description": "Release year" }
            },
            "examples": [{
                "title": "The Matrix",
                "year": 1999
            }]
        }));

        let gemini_schema = prepare_gemini_schema(&schema);

        // Verify examples is stripped
        assert!(
            gemini_schema.get("examples").is_none(),
            "examples should be stripped from Gemini schema"
        );

        // Verify title is stripped (Gemini doesn't support it)
        assert!(
            gemini_schema.get("title").is_none(),
            "title should be stripped from Gemini schema"
        );

        // Verify the basic schema structure is preserved
        assert_eq!(gemini_schema["type"], "object");
        assert!(gemini_schema["properties"]["title"].is_object());
        assert!(gemini_schema["properties"]["year"].is_object());
    }

    #[test]
    fn test_gemini_schema_strips_nested_examples() {
        use crate::schema::Schema;

        // Create a schema with nested objects that have examples
        let schema = Schema::new(serde_json::json!({
            "type": "object",
            "properties": {
                "recipe_name": { "type": "string" },
                "ingredients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "amount": { "type": "number" }
                        },
                        "examples": [{
                            "name": "flour",
                            "amount": 2.5
                        }]
                    }
                }
            },
            "examples": [{
                "recipe_name": "Cookies",
                "ingredients": []
            }]
        }));

        let gemini_schema = prepare_gemini_schema(&schema);

        // Verify examples is stripped at root
        assert!(
            gemini_schema.get("examples").is_none(),
            "root examples should be stripped"
        );

        // Verify examples is stripped from array items (nested object)
        assert!(
            gemini_schema["properties"]["ingredients"]["items"]
                .get("examples")
                .is_none(),
            "nested examples should be stripped"
        );
    }

    #[test]
    fn test_gemini_schema_strips_additional_properties() {
        let mut schema_json = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "additionalProperties": false
        });

        strip_gemini_unsupported_keywords(&mut schema_json);

        assert!(
            schema_json.get("additionalProperties").is_none(),
            "additionalProperties should be stripped"
        );
    }

    #[test]
    fn test_gemini_schema_strips_title_and_schema() {
        let mut schema_json = serde_json::json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Movie",
            "type": "object",
            "properties": {
                "name": {
                    "title": "MovieName",
                    "type": "string"
                }
            }
        });

        strip_gemini_unsupported_keywords(&mut schema_json);

        assert!(
            schema_json.get("$schema").is_none(),
            "$schema should be stripped"
        );
        assert!(
            schema_json.get("title").is_none(),
            "title should be stripped"
        );
        assert!(
            schema_json["properties"]["name"].get("title").is_none(),
            "nested title should be stripped"
        );
    }
}

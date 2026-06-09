use proc_macro2::TokenStream;
use quote::quote;
use syn::{DataStruct, Fields, Ident, Type};

use crate::container_attrs::ContainerAttributes;
use crate::parsers::field_parser::parse_field_attributes;
use crate::type_utils::{
    get_array_inner_type, get_box_inner_type, get_map_types, get_option_inner_type,
    get_schema_type_from_rust_type, get_tuple_element_types, get_type_name, is_array_type,
    is_box_type, is_json_value_type, is_map_type, is_option_type, is_self_reference, is_tuple_type,
};

/// Generate the schema implementation for a struct
pub fn generate_struct_schema(
    name: &Ident,
    data_struct: &DataStruct,
    container_attrs: &ContainerAttributes,
) -> TokenStream {
    let mut property_setters = Vec::new();
    let mut required_setters = Vec::new();
    let mut has_self_reference = false;
    let struct_name_str = name.to_string();

    match &data_struct.fields {
        Fields::Named(fields) => {
            // First pass: check for self-references
            for field in &fields.named {
                if is_self_reference(&field.ty, &struct_name_str) {
                    has_self_reference = true;
                    break;
                }
            }

            for field in &fields.named {
                // Parse field attributes first to check for serde rename
                let attrs = parse_field_attributes(field);

                let original_field_name = field.ident.as_ref().unwrap().to_string();
                // Priority: 1) field-level #[serde(rename)], 2) container #[serde(rename_all)], 3) original name
                let field_name = if let Some(ref rename) = attrs.serde_rename {
                    rename.clone()
                } else if let Some(rename_all) = &container_attrs.serde_rename_all {
                    // Apply the serde rename_all transformation
                    apply_rename_all(&original_field_name, rename_all)
                } else {
                    original_field_name
                };
                let is_optional = is_option_type(&field.ty);

                // Get schema type
                let schema_type = get_schema_type_from_rust_type(&field.ty);

                // Extract type name for well-known library types only (exact matches, no heuristics)
                let type_name = get_type_name(&field.ty);

                // Also extract the inner type name when the field is Optional,
                // so Option<NaiveDate>, Option<DateTime>, Option<Uuid> etc. get proper format metadata
                let inner_type_name = if is_optional {
                    get_type_name(get_option_inner_type(&field.ty))
                } else {
                    None
                };

                // Check for well-known library types by exact match only (no contains checks)
                let is_datetime_type = matches!(
                    type_name.as_deref(),
                    Some("DateTime") | Some("NaiveDateTime")
                ) || matches!(
                    inner_type_name.as_deref(),
                    Some("DateTime") | Some("NaiveDateTime")
                );
                let is_date_only_type =
                    matches!(type_name.as_deref(), Some("NaiveDate") | Some("Date"))
                        || matches!(inner_type_name.as_deref(), Some("NaiveDate") | Some("Date"));
                let is_uuid_type = matches!(type_name.as_deref(), Some("Uuid"))
                    || matches!(inner_type_name.as_deref(), Some("Uuid"));

                // Create field property
                // IMPORTANT: Default to treating unknown types as structs (objects)
                // Structs are far more common than enums, and this is the safest default
                let field_prop = if is_datetime_type || is_date_only_type || is_uuid_type {
                    // Well-known library type *names* (chrono's DateTime/NaiveDate/...,
                    // uuid's Uuid). These names are only a heuristic: a user-defined
                    // `struct Date` deriving Instructor must keep its real schema. So
                    // probe at compile time (via autoref specialization): if the field
                    // type implements SchemaType, its own schema wins; otherwise fall
                    // back to the sniffed string/format schema.
                    let actual_type = if is_optional {
                        get_option_inner_type(&field.ty)
                    } else {
                        &field.ty
                    };
                    let fallback = sniffed_fallback_schema(is_datetime_type, is_date_only_type);
                    quote! {
                        // Create property for this well-known (date/uuid) field,
                        // preferring the type's own SchemaType impl when present
                        let probed = {
                            #[allow(unused_imports)]
                            use ::rstructor::schema::__private::SchemaProbeFallback as _;
                            ::rstructor::schema::__private::SchemaProbe::<#actual_type>::new()
                                .rstructor_schema_or(#fallback)
                        };
                        let mut props = if let ::serde_json::Value::Object(m) = probed {
                            m
                        } else {
                            let mut m = ::serde_json::Map::new();
                            m.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                            m
                        };
                    }
                } else if is_array_type(&field.ty)
                    || (is_optional && is_array_type(get_option_inner_type(&field.ty)))
                {
                    // For array types (including Option<Vec<T>>), we need to add the 'items' property
                    let actual_array_type =
                        if is_optional && is_array_type(get_option_inner_type(&field.ty)) {
                            get_option_inner_type(&field.ty)
                        } else {
                            &field.ty
                        };
                    if let Some(inner_type) = get_array_inner_type(actual_array_type) {
                        // Generate the items schema, recursing into nested
                        // collections so e.g. Vec<Vec<i32>> keeps its inner items
                        let items_expr = generate_array_items_schema(inner_type, &struct_name_str);
                        quote! {
                            // Create property for this array field
                            let mut props = ::serde_json::Map::new();
                            props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));

                            // Add items schema (recursive for nested collections)
                            props.insert("items".to_string(), #items_expr);
                        }
                    } else {
                        // Fallback for array without detectable item type
                        quote! {
                            // Create property for this array field (fallback)
                            let mut props = ::serde_json::Map::new();
                            props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));

                            // Add default items schema
                            let mut items_schema = ::serde_json::Map::new();
                            items_schema.insert("type".to_string(), ::serde_json::Value::String("string".to_string()));
                            props.insert("items".to_string(), ::serde_json::Value::Object(items_schema));
                        }
                    }
                } else if is_json_value_type(&field.ty)
                    || (is_optional && is_json_value_type(get_option_inner_type(&field.ty)))
                {
                    // For serde_json::Value, use an empty schema (any JSON is valid)
                    quote! {
                        let mut props = ::serde_json::Map::new();
                        // Empty object schema means any JSON value is accepted
                    }
                } else if is_map_type(&field.ty)
                    || (is_optional && is_map_type(get_option_inner_type(&field.ty)))
                {
                    // For HashMap<K, V> or BTreeMap<K, V>
                    let actual_type = if is_optional {
                        get_option_inner_type(&field.ty)
                    } else {
                        &field.ty
                    };
                    if let Some((key_ty, val_ty)) = get_map_types(actual_type) {
                        // Use SchemaType::schema() for all value types to get complete schema
                        // This ensures arrays get proper `items`, objects get properties, etc.
                        // For enum keys, extract the enum variants and add them to description
                        // so that Gemini can use the correct keys instead of generic placeholders
                        quote! {
                            let mut props = ::serde_json::Map::new();
                            props.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                            let value_schema = <#val_ty as ::rstructor::schema::SchemaType>::schema();
                            props.insert("additionalProperties".to_string(), value_schema.to_json());

                            // Try to extract enum keys from key type schema (for enum keys)
                            let key_schema = <#key_ty as ::rstructor::schema::SchemaType>::schema();
                            if let Some(enum_values) = key_schema.to_json().get("enum")
                                .and_then(|e| e.as_array())
                            {
                                let keys: Vec<String> = enum_values
                                    .iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect();
                                if !keys.is_empty() {
                                    let keys_hint = format!("Keys: [{}]", keys.join(", "));
                                    props.insert("description".to_string(), ::serde_json::Value::String(keys_hint));
                                    // Also store enum keys as a structured extension field
                                    // so backends can extract them without parsing the description
                                    let keys_json: Vec<::serde_json::Value> = keys.iter()
                                        .map(|k| ::serde_json::Value::String(k.clone()))
                                        .collect();
                                    props.insert("x-enum-keys".to_string(), ::serde_json::Value::Array(keys_json));
                                }
                            }
                        }
                    } else {
                        // Fallback for map without detectable types
                        quote! {
                            let mut props = ::serde_json::Map::new();
                            props.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                        }
                    }
                } else if is_box_type(&field.ty)
                    || (is_optional && is_box_type(get_option_inner_type(&field.ty)))
                {
                    // For Box<T>, unwrap and use the inner type's schema
                    let actual_type = if is_optional {
                        get_option_inner_type(&field.ty)
                    } else {
                        &field.ty
                    };
                    if let Some(inner_ty) = get_box_inner_type(actual_type) {
                        let inner_schema_type = get_schema_type_from_rust_type(inner_ty);
                        if inner_schema_type == "object" {
                            // Inner type is a complex type, use its schema
                            quote! {
                                let nested_schema = <#inner_ty as ::rstructor::schema::SchemaType>::schema();
                                let props_json = nested_schema.to_json();
                                let mut props = if let ::serde_json::Value::Object(m) = props_json {
                                    m
                                } else {
                                    let mut m = ::serde_json::Map::new();
                                    m.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                                    m
                                };
                            }
                        } else {
                            // Inner type is a primitive
                            quote! {
                                let mut props = ::serde_json::Map::new();
                                props.insert("type".to_string(), ::serde_json::Value::String(#inner_schema_type.to_string()));
                            }
                        }
                    } else {
                        // Fallback
                        quote! {
                            let mut props = ::serde_json::Map::new();
                            props.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                        }
                    }
                } else if is_tuple_type(&field.ty)
                    || (is_optional && is_tuple_type(get_option_inner_type(&field.ty)))
                {
                    // For tuples, generate array with prefixItems
                    let actual_type = if is_optional {
                        get_option_inner_type(&field.ty)
                    } else {
                        &field.ty
                    };
                    if let Some(element_types) = get_tuple_element_types(actual_type) {
                        let element_count = element_types.len();
                        // Generate schema for each element
                        let element_schemas: Vec<TokenStream> = element_types
                            .iter()
                            .map(|elem_ty| {
                                let elem_schema_type = get_schema_type_from_rust_type(elem_ty);
                                if elem_schema_type == "object" {
                                    quote! {
                                        <#elem_ty as ::rstructor::schema::SchemaType>::schema().to_json()
                                    }
                                } else {
                                    quote! {
                                        ::serde_json::json!({"type": #elem_schema_type})
                                    }
                                }
                            })
                            .collect();
                        quote! {
                            let mut props = ::serde_json::Map::new();
                            props.insert("type".to_string(), ::serde_json::Value::String("array".to_string()));
                            let prefix_items = vec![
                                #(#element_schemas),*
                            ];
                            props.insert("prefixItems".to_string(), ::serde_json::Value::Array(prefix_items));
                            props.insert("minItems".to_string(), ::serde_json::Value::Number(::serde_json::Number::from(#element_count)));
                            props.insert("maxItems".to_string(), ::serde_json::Value::Number(::serde_json::Number::from(#element_count)));
                        }
                    } else {
                        // Fallback for tuple without detectable elements
                        quote! {
                            let mut props = ::serde_json::Map::new();
                            props.insert("type".to_string(), ::serde_json::Value::String("array".to_string()));
                        }
                    }
                } else if type_name.is_some() && schema_type == "object" {
                    // For nested struct fields, embed the inner type's schema directly
                    // This requires the inner type to implement SchemaType
                    // If it's an Option<T>, unwrap to get T first
                    let actual_type = if is_optional {
                        get_option_inner_type(&field.ty)
                    } else {
                        &field.ty
                    };
                    quote! {
                        // Get the nested type's schema directly
                        let nested_schema = <#actual_type as ::rstructor::schema::SchemaType>::schema();
                        let props_json = nested_schema.to_json();
                        // Convert to a mutable map so we can add to it
                        let mut props = if let ::serde_json::Value::Object(m) = props_json {
                            m
                        } else {
                            let mut m = ::serde_json::Map::new();
                            m.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                            m
                        };
                    }
                } else {
                    // Regular primitive type
                    quote! {
                        // Create property for this field
                        let mut props = ::serde_json::Map::new();
                        props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));
                    }
                };
                property_setters.push(field_prop);

                // Add description if available - merge with existing keys hint if present
                if let Some(desc) = attrs.description {
                    let desc_prop = quote! {
                        if let Some(existing) = props.get("description").and_then(|v| v.as_str()) {
                            if existing.contains("Keys: [") {
                                // Merge user description with the keys hint
                                let merged = format!("{}. {}", #desc, existing);
                                props.insert("description".to_string(), ::serde_json::Value::String(merged));
                            } else {
                                props.insert("description".to_string(), ::serde_json::Value::String(#desc.to_string()));
                            }
                        } else {
                            props.insert("description".to_string(), ::serde_json::Value::String(#desc.to_string()));
                        }
                    };
                    property_setters.push(desc_prop);
                }

                // Add single example if available
                if let Some(ex_val) = &attrs.example_value {
                    let ex_prop = quote! {
                        let example_value = #ex_val;
                        props.insert("example".to_string(), example_value);
                    };
                    property_setters.push(ex_prop);
                }

                // Add multiple examples if available
                if !attrs.examples_array.is_empty() {
                    let examples_tokens = attrs.examples_array.iter().collect::<Vec<_>>();
                    let exs_prop = quote! {
                        let examples_value = ::serde_json::Value::Array(vec![
                            #(#examples_tokens),*
                        ]);
                        props.insert("examples".to_string(), examples_value);
                    };
                    property_setters.push(exs_prop);
                }

                // Add the property to the schema
                let add_prop = quote! {
                    // Add property to the schema
                    let props_val = ::serde_json::Value::Object(props);
                    if let ::serde_json::Value::Object(obj) = schema_obj.get_mut("properties").unwrap() {
                        obj.insert(#field_name.to_string(), props_val);
                    }
                };
                property_setters.push(add_prop);

                // Add to required fields if not Optional type
                if !is_optional {
                    let required_field = quote! {
                        required.push(::serde_json::Value::String(#field_name.to_string()));
                    };
                    required_setters.push(required_field);
                }
            }
        }
        _ => panic!("Instructor can only be derived for structs with named fields"),
    }

    // Handle container attributes
    let mut container_setters = Vec::new();

    // Description
    if let Some(desc) = &container_attrs.description {
        container_setters.push(quote! {
            schema_obj["description"] = ::serde_json::Value::String(#desc.to_string());
        });
    }

    // Title (override default)
    if let Some(title) = &container_attrs.title {
        container_setters.push(quote! {
            schema_obj["title"] = ::serde_json::Value::String(#title.to_string());
        });
    }

    // Examples
    if !container_attrs.examples.is_empty() {
        let examples_values = &container_attrs.examples;
        container_setters.push(quote! {
            let examples_array = vec![
                #(#examples_values),*
            ];
            schema_obj["examples"] = ::serde_json::Value::Array(examples_array);
        });
    }

    // Combine all container attribute setters
    let container_setter = if !container_setters.is_empty() {
        quote! {
            #(#container_setters)*
        }
    } else {
        quote! {}
    };

    // Generate implementation with $defs support for recursive types
    if has_self_reference {
        quote! {
            impl ::rstructor::schema::SchemaType for #name {
                fn schema() -> ::rstructor::schema::Schema {
                    // Create base schema object (properties will be added to $defs)
                    let mut schema_obj = ::serde_json::json!({
                        "type": "object",
                        "title": stringify!(#name),
                        "properties": {}
                    });

                    // Add container attributes if available
                    #container_setter

                    // Fill properties
                    #(#property_setters)*

                    // Add required fields
                    let mut required = Vec::new();
                    #(#required_setters)*
                    schema_obj["required"] = ::serde_json::Value::Array(required);

                    // Create root schema with $defs for recursive types
                    let struct_name = stringify!(#name);
                    let root_schema = ::serde_json::json!({
                        "$defs": {
                            struct_name: schema_obj
                        },
                        "$ref": format!("#/$defs/{}", struct_name)
                    });

                    ::rstructor::schema::Schema::new(root_schema)
                }

                fn schema_name() -> Option<String> {
                    Some(stringify!(#name).to_string())
                }
            }
        }
    } else {
        quote! {
            impl ::rstructor::schema::SchemaType for #name {
                fn schema() -> ::rstructor::schema::Schema {
                    // Create base schema object
                    let mut schema_obj = ::serde_json::json!({
                        "type": "object",
                        "title": stringify!(#name),
                        "properties": {}
                    });

                    // Add container attributes if available
                    #container_setter

                    // Fill properties
                    #(#property_setters)*

                    // Add required fields
                    let mut required = Vec::new();
                    #(#required_setters)*
                    schema_obj["required"] = ::serde_json::Value::Array(required);

                    ::rstructor::schema::Schema::new(schema_obj)
                }

                fn schema_name() -> Option<String> {
                    Some(stringify!(#name).to_string())
                }
            }
        }
    }
}

/// Generate an expression evaluating to the sniffed fallback schema for a
/// well-known library type name: date-time (`DateTime`/`NaiveDateTime`), date
/// (`NaiveDate`/`Date`), or uuid (`Uuid`).
fn sniffed_fallback_schema(is_datetime: bool, is_date_only: bool) -> TokenStream {
    if is_datetime {
        quote! {
            ::serde_json::json!({
                "type": "string",
                "format": "date-time",
                "description": "ISO-8601 formatted date and time"
            })
        }
    } else if is_date_only {
        quote! {
            ::serde_json::json!({
                "type": "string",
                "format": "date",
                "description": "ISO-8601 formatted date (YYYY-MM-DD)"
            })
        }
    } else {
        quote! {
            ::serde_json::json!({
                "type": "string",
                "format": "uuid",
                "description": "UUID identifier string"
            })
        }
    }
}

/// Generate an expression evaluating to the JSON Schema `items` value for an
/// array whose element type is `inner_type`.
///
/// Recurses into nested collections (`Vec<Vec<i32>>`, `Vec<HashSet<String>>`,
/// ...) so every nesting level carries its own `items` schema, and emits a
/// `$ref` for self-referential element types to prevent infinite recursion.
fn generate_array_items_schema(inner_type: &Type, struct_name_str: &str) -> TokenStream {
    // Check for well-known library types by exact match only (no heuristics)
    let inner_type_name = get_type_name(inner_type);
    let is_datetime = matches!(
        inner_type_name.as_deref(),
        Some("DateTime") | Some("NaiveDateTime")
    );
    let is_date_only = matches!(inner_type_name.as_deref(), Some("NaiveDate") | Some("Date"));
    let is_uuid = matches!(inner_type_name.as_deref(), Some("Uuid"));

    if is_datetime || is_date_only || is_uuid {
        // Name match is only a heuristic: prefer the element type's own
        // SchemaType impl (user-defined `struct Date`), falling back to the
        // sniffed string/format schema (chrono/uuid types).
        let fallback = sniffed_fallback_schema(is_datetime, is_date_only);
        return quote! {
            {
                #[allow(unused_imports)]
                use ::rstructor::schema::__private::SchemaProbeFallback as _;
                ::rstructor::schema::__private::SchemaProbe::<#inner_type>::new()
                    .rstructor_schema_or(#fallback)
            }
        };
    }

    // Nested collections: recurse so the inner `items` schema is preserved
    if is_array_type(inner_type)
        && let Some(next_inner) = get_array_inner_type(inner_type)
    {
        let nested_items = generate_array_items_schema(next_inner, struct_name_str);
        return quote! {
            {
                let mut items_schema = ::serde_json::Map::new();
                items_schema.insert("type".to_string(), ::serde_json::Value::String("array".to_string()));
                items_schema.insert("items".to_string(), #nested_items);
                ::serde_json::Value::Object(items_schema)
            }
        };
    }

    // Self-referential element types use $ref to prevent infinite recursion
    if is_self_reference(inner_type, struct_name_str) {
        return quote! {
            ::serde_json::json!({ "$ref": format!("#/$defs/{}", #struct_name_str) })
        };
    }

    let inner_schema_type = get_schema_type_from_rust_type(inner_type);
    if inner_schema_type == "object" {
        // For nested structs (and Box/HashMap wrappers), embed the inner
        // type's schema directly. This requires the inner type to implement
        // SchemaType.
        quote! {
            <#inner_type as ::rstructor::schema::SchemaType>::schema().to_json()
        }
    } else {
        // Standard handling for primitive types
        quote! {
            ::serde_json::json!({ "type": #inner_schema_type })
        }
    }
}

/// Apply serde rename_all transformation to a field/variant name
pub fn apply_rename_all(name: &str, rename_all: &str) -> String {
    match rename_all {
        "lowercase" => name.to_lowercase(),
        "UPPERCASE" => name.to_uppercase(),
        "camelCase" => {
            // Convert snake_case to camelCase, or PascalCase to camelCase
            if name.contains('_') {
                // snake_case input
                let parts: Vec<&str> = name.split('_').collect();
                if parts.is_empty() {
                    name.to_string()
                } else {
                    let mut result = parts[0].to_lowercase();
                    for part in &parts[1..] {
                        if !part.is_empty() {
                            let mut chars = part.chars();
                            if let Some(first) = chars.next() {
                                result.push(first.to_ascii_uppercase());
                                result.extend(chars.map(|c| c.to_ascii_lowercase()));
                            }
                        }
                    }
                    result
                }
            } else {
                // PascalCase input - just lowercase the first char
                let mut chars = name.chars();
                match chars.next() {
                    Some(first) => first.to_ascii_lowercase().to_string() + chars.as_str(),
                    None => name.to_string(),
                }
            }
        }
        "PascalCase" => {
            // Convert snake_case to PascalCase
            let parts: Vec<&str> = name.split('_').collect();
            let mut result = String::new();
            for part in parts {
                if !part.is_empty() {
                    let mut chars = part.chars();
                    if let Some(first) = chars.next() {
                        result.push(first.to_ascii_uppercase());
                        result.extend(chars);
                    }
                }
            }
            result
        }
        "snake_case" => {
            // Convert PascalCase/camelCase to snake_case
            pascal_to_snake_case(name)
        }
        "SCREAMING_SNAKE_CASE" => {
            // Convert to SCREAMING_SNAKE_CASE
            pascal_to_snake_case(name).to_uppercase()
        }
        "kebab-case" => {
            // Convert to kebab-case
            pascal_to_snake_case(name).replace('_', "-")
        }
        "SCREAMING-KEBAB-CASE" => {
            // Convert to SCREAMING-KEBAB-CASE
            pascal_to_snake_case(name).to_uppercase().replace('_', "-")
        }
        _ => name.to_string(),
    }
}

/// Convert PascalCase or camelCase to snake_case
fn pascal_to_snake_case(name: &str) -> String {
    let mut result = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(c.to_ascii_lowercase());
    }
    result
}

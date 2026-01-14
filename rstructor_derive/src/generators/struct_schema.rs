use proc_macro2::TokenStream;
use quote::quote;
use syn::{DataStruct, Fields, Ident, Type};

use crate::container_attrs::ContainerAttributes;
use crate::parsers::field_parser::parse_field_attributes;
use crate::type_utils::{
    get_array_inner_type, get_box_inner_type, get_map_types, get_option_inner_type,
    get_schema_type_from_rust_type, get_tuple_element_types, is_array_type, is_box_type,
    is_json_value_type, is_map_type, is_option_type, is_self_reference, is_tuple_type,
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
                let type_name = if let Type::Path(type_path) = &field.ty {
                    type_path
                        .path
                        .segments
                        .first()
                        .map(|segment| segment.ident.to_string())
                } else {
                    None
                };

                // Check for well-known library types by exact match only (no contains checks)
                let is_date_type = matches!(
                    type_name.as_deref(),
                    Some("DateTime") | Some("NaiveDateTime") | Some("NaiveDate") | Some("Date")
                );
                let is_uuid_type = matches!(type_name.as_deref(), Some("Uuid"));

                // Create field property
                // IMPORTANT: Default to treating unknown types as structs (objects)
                // Structs are far more common than enums, and this is the safest default
                let field_prop = if is_date_type {
                    // For date types
                    quote! {
                        // Create property for this date field
                        let mut props = ::serde_json::Map::new();
                        props.insert("type".to_string(), ::serde_json::Value::String("string".to_string()));
                        props.insert("format".to_string(), ::serde_json::Value::String("date-time".to_string()));
                        props.insert("description".to_string(),
                                    ::serde_json::Value::String("ISO-8601 formatted date and time".to_string()));
                    }
                } else if is_uuid_type {
                    // For UUID types
                    quote! {
                        // Create property for this UUID field
                        let mut props = ::serde_json::Map::new();
                        props.insert("type".to_string(), ::serde_json::Value::String("string".to_string()));
                        props.insert("format".to_string(), ::serde_json::Value::String("uuid".to_string()));
                        props.insert("description".to_string(),
                                    ::serde_json::Value::String("UUID identifier string".to_string()));
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
                        // Get the inner schema type
                        let inner_schema_type = get_schema_type_from_rust_type(inner_type);

                        // Extract inner type name for well-known library types only
                        let inner_type_name = if let Type::Path(type_path) = inner_type {
                            type_path
                                .path
                                .segments
                                .first()
                                .map(|segment| segment.ident.to_string())
                        } else {
                            None
                        };

                        // Check for well-known library types by exact match only
                        let is_date = matches!(
                            inner_type_name.as_deref(),
                            Some("DateTime")
                                | Some("NaiveDateTime")
                                | Some("NaiveDate")
                                | Some("Date")
                        );
                        let is_uuid = matches!(inner_type_name.as_deref(), Some("Uuid"));

                        // Generate items schema
                        if is_date {
                            // Handle array of dates
                            quote! {
                                // Create property for this array field with date items
                                let mut props = ::serde_json::Map::new();
                                props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));

                                // Add items schema for dates
                                let mut items_schema = ::serde_json::Map::new();
                                items_schema.insert("type".to_string(), ::serde_json::Value::String("string".to_string()));
                                items_schema.insert("format".to_string(), ::serde_json::Value::String("date-time".to_string()));
                                items_schema.insert("description".to_string(),
                                    ::serde_json::Value::String("ISO-8601 formatted date and time".to_string()));

                                props.insert("items".to_string(), ::serde_json::Value::Object(items_schema));
                            }
                        } else if is_uuid {
                            // Handle array of UUIDs
                            quote! {
                                // Create property for this array field with UUID items
                                let mut props = ::serde_json::Map::new();
                                props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));

                                // Add items schema for UUIDs
                                let mut items_schema = ::serde_json::Map::new();
                                items_schema.insert("type".to_string(), ::serde_json::Value::String("string".to_string()));
                                items_schema.insert("format".to_string(), ::serde_json::Value::String("uuid".to_string()));
                                items_schema.insert("description".to_string(),
                                    ::serde_json::Value::String("UUID identifier string".to_string()));

                                props.insert("items".to_string(), ::serde_json::Value::Object(items_schema));
                            }
                        } else if inner_schema_type == "object" {
                            // Check if this is a self-reference (recursive type)
                            let struct_name_str = name.to_string();
                            if is_self_reference(inner_type, &struct_name_str) {
                                // For self-referential types, use $ref to prevent infinite recursion
                                quote! {
                                    // Create property for this array field with recursive reference
                                    let mut props = ::serde_json::Map::new();
                                    props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));

                                    // Use $ref for recursive type
                                    let mut items_schema = ::serde_json::Map::new();
                                    items_schema.insert("$ref".to_string(), ::serde_json::Value::String(format!("#/$defs/{}", #struct_name_str)));
                                    props.insert("items".to_string(), ::serde_json::Value::Object(items_schema));
                                }
                            } else {
                                // For arrays of nested structs, embed the inner type's schema directly
                                // This requires the inner type to implement SchemaType
                                quote! {
                                    // Create property for this array field with nested struct items
                                    let mut props = ::serde_json::Map::new();
                                    props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));

                                    // Get the inner type's schema directly
                                    // This embeds the full schema with properties at compile time
                                    let inner_schema = <#inner_type as ::rstructor::schema::SchemaType>::schema();
                                    let items_schema = inner_schema.to_json();

                                    props.insert("items".to_string(), items_schema);
                                }
                            }
                        } else {
                            // Standard handling for primitive types
                            quote! {
                                // Create property for this array field
                                let mut props = ::serde_json::Map::new();
                                props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));

                                // Add items schema
                                let mut items_schema = ::serde_json::Map::new();
                                items_schema.insert("type".to_string(), ::serde_json::Value::String(#inner_schema_type.to_string()));
                                props.insert("items".to_string(), ::serde_json::Value::Object(items_schema));
                            }
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
                    if let Some((_key_ty, val_ty)) = get_map_types(actual_type) {
                        // Use SchemaType::schema() for all value types to get complete schema
                        // This ensures arrays get proper `items`, objects get properties, etc.
                        quote! {
                            let mut props = ::serde_json::Map::new();
                            props.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                            let value_schema = <#val_ty as ::rstructor::schema::SchemaType>::schema();
                            props.insert("additionalProperties".to_string(), value_schema.to_json());
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

                // Add description if available
                if let Some(desc) = attrs.description {
                    let desc_prop = quote! {
                        props.insert("description".to_string(), ::serde_json::Value::String(#desc.to_string()));
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

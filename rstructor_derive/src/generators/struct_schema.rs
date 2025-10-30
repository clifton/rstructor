#![allow(clippy::collapsible_if)]

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DataStruct, Fields, Ident, Type};

use crate::container_attrs::ContainerAttributes;
use crate::parsers::field_parser::parse_field_attributes;
use crate::type_utils::{
    get_array_inner_type, get_schema_type_from_rust_type, is_array_type, is_option_type,
};

/// Generate the schema implementation for a struct
pub fn generate_struct_schema(
    name: &Ident,
    data_struct: &DataStruct,
    container_attrs: &ContainerAttributes,
) -> TokenStream {
    let mut property_setters = Vec::new();
    let mut required_setters = Vec::new();

    match &data_struct.fields {
        Fields::Named(fields) => {
            for field in &fields.named {
                let original_field_name = field.ident.as_ref().unwrap().to_string();
                let field_name = if let Some(rename_all) = &container_attrs.serde_rename_all {
                    // Apply the serde rename_all transformation
                    match rename_all.as_str() {
                        "lowercase" => original_field_name.to_lowercase(),
                        "UPPERCASE" => original_field_name.to_uppercase(),
                        "camelCase" => {
                            // Convert snake_case to camelCase
                            let parts: Vec<&str> = original_field_name.split('_').collect();
                            if parts.is_empty() {
                                original_field_name
                            } else {
                                let mut result = parts[0].to_string();
                                for part in &parts[1..] {
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
                        }
                        "PascalCase" => {
                            // Convert snake_case to PascalCase
                            let parts: Vec<&str> = original_field_name.split('_').collect();
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
                        "snake_case" => original_field_name,
                        _ => original_field_name,
                    }
                } else {
                    original_field_name
                };
                let is_optional = is_option_type(&field.ty);

                // Parse field attributes
                let attrs = parse_field_attributes(field);

                // Get schema type
                let schema_type = get_schema_type_from_rust_type(&field.ty);

                // For custom types, check if they're enums by looking at the type name
                let type_name = if let Type::Path(type_path) = &field.ty {
                    type_path
                        .path
                        .segments
                        .first()
                        .map(|segment| segment.ident.to_string())
                } else {
                    None
                };

                // Special handling for enums and custom types used as fields
                let (is_likely_enum, is_date_type, is_uuid_type, is_custom_type) = if let Some(
                    name,
                ) = &type_name
                {
                    // Check for special types
                    let is_date = name == "DateTime"
                        || name == "NaiveDateTime"
                        || name == "NaiveDate"
                        || name == "Date"
                        || name.contains("Date")
                        || name.contains("Time");
                    let is_uuid = name == "Uuid";

                    // Check if this could be a custom type implementing CustomTypeSchema
                    // Heuristic: Custom types often have "meaningful" names like CustomDate, EmailAddress, etc.
                    let is_custom = name.contains("Date")
                        || name.contains("Time")
                        || name.contains("Email")
                        || name.contains("Uuid")
                        || name.contains("Phone")
                        || name.contains("Custom");

                    // Check if it's likely an enum (starts with uppercase, is an object, not an array)
                    // CRITICAL: Be EXTREMELY conservative - only flag as enum if it's clearly enum-like
                    // Nested structs are MUCH more common than enums as fields, so default to struct
                    // True enums are usually: VERY short single PascalCase word (Status, Type, Color, State)
                    // Structs have descriptive names (Address, Person, ContactInfo, etc.)
                    let first_char = name.chars().next();
                    let uppercase_count = name.chars().filter(|c| c.is_uppercase()).count();
                    let is_enum = first_char.is_some_and(|c| c.is_uppercase())
                            && schema_type == "object"
                            && !is_array_type(&field.ty)
                            && !is_date
                            && !is_uuid
                            && !is_custom
                            // EXTREMELY strict criteria - only match very short single-word enums:
                            && name.len() <= 6  // Very short names only (Status=6, Type=4, Color=5, State=5)
                            && uppercase_count == 1  // Single capital letter (true PascalCase single word)
                            && !name.contains("_")  // No underscores
                            && name.chars().all(|c| c.is_alphanumeric()) // Only alphanumeric
                            // Additional check: common enum names (whitelist approach)
                            && (name == "Status" || name == "Type" || name == "State" || name == "Color" 
                                || name == "Kind" || name == "Mode" || name == "Role" || name == "Level");

                    (is_enum, is_date, is_uuid, is_custom)
                } else {
                    (false, false, false, false)
                };

                // Create field property
                // CRITICAL: Check for nested structs FIRST - they should be type "object"
                // Only treat as enum if it's clearly not a struct (very short, single PascalCase word)
                let field_prop = if type_name.is_some()
                    && schema_type == "object"
                    && !is_array_type(&field.ty)
                    && !is_date_type
                    && !is_uuid_type
                    && !is_custom_type
                {
                    // For nested struct/enum types - prioritize treating as object unless clearly enum
                    if is_likely_enum {
                        // Only if it's VERY likely an enum (short, single word), treat as string
                        quote! {
                            // Create property for this enum field
                            let mut props = ::serde_json::Map::new();
                            // Use string type for enums
                            props.insert("type".to_string(), ::serde_json::Value::String("string".to_string()));
                            // We'll add the enum description separately since we need to handle field attributes
                        }
                    } else {
                        // Treat as nested struct - must be type "object"
                        quote! {
                            // Create property for nested struct field
                            let mut props = ::serde_json::Map::new();
                            // CRITICAL: Must be type "object" for nested structs, not "string"
                            props.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                        }
                    }
                } else if is_likely_enum {
                    // For likely enum types, use String type with a reference to using enum values
                    quote! {
                        // Create property for this enum field
                        let mut props = ::serde_json::Map::new();
                        // Use string type for enums
                        props.insert("type".to_string(), ::serde_json::Value::String("string".to_string()));
                        // We'll add the enum description separately since we need to handle field attributes
                    }
                } else if is_custom_type {
                    // For custom types implementing CustomTypeSchema
                    let type_name_val = if let Some(name) = &type_name {
                        name.clone()
                    } else {
                        "CustomType".to_string()
                    };

                    quote! {
                        // Create property for this custom field
                        let mut props = ::serde_json::Map::new();
                        props.insert("type".to_string(), ::serde_json::Value::String("string".to_string()));

                        // Add some defaults that will be overridden by the field attributes if provided
                        props.insert("description".to_string(),
                                    ::serde_json::Value::String(format!("A custom {} value", #type_name_val)));
                    }
                } else if is_date_type {
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
                } else if type_name.is_some() {
                    // Default handling for other custom types
                    quote! {
                        // Create property for this field
                        let mut props = ::serde_json::Map::new();
                        props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));
                    }
                } else if is_array_type(&field.ty) {
                    // For array types, we need to add the 'items' property
                    if let Some(inner_type) = get_array_inner_type(&field.ty) {
                        // Get the inner schema type
                        let inner_schema_type = get_schema_type_from_rust_type(inner_type);

                        // Check if the inner type might be an enum or custom type
                        let inner_type_name = if let Type::Path(type_path) = inner_type {
                            type_path
                                .path
                                .segments
                                .first()
                                .map(|segment| segment.ident.to_string())
                        } else {
                            None
                        };

                        // Choose the appropriate handling for the array items based on the inner type
                        let items_tokens: proc_macro2::TokenStream = if let Some(type_name) =
                            &inner_type_name
                        {
                            // Check if type name starts with uppercase (likely custom type)
                            let first_char = type_name.chars().next();
                            let is_uppercase = first_char.is_some_and(|c| c.is_uppercase());

                            // Check if this could be an enum
                            let is_likely_enum = is_uppercase &&
                                inner_schema_type == "object" &&
                                !is_array_type(inner_type) &&
                                // Additional heuristic: enums are usually short names without underscores
                                !type_name.contains('_') &&
                                type_name.len() < 20;

                            if is_likely_enum && type_name != "Entity" && type_name != "Item" {
                                // For arrays of enum values (excluding Entity which is a known struct)
                                let type_name_str = type_name.clone();
                                quote! {
                                    // Create property for this array field with enum items
                                    let mut props = ::serde_json::Map::new();
                                    props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));

                                    // Add items schema for enum
                                    let mut items_schema = ::serde_json::Map::new();
                                    items_schema.insert("type".to_string(), ::serde_json::Value::String("string".to_string()));
                                    items_schema.insert("description".to_string(),
                                        ::serde_json::Value::String(format!("Must be one of the allowed values for {}", #type_name_str)));
                                    props.insert("items".to_string(), ::serde_json::Value::Object(items_schema));
                                }
                            } else if type_name == "DateTime"
                                || type_name == "NaiveDateTime"
                                || type_name == "NaiveDate"
                                || type_name == "Date"
                            {
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
                            } else if type_name == "Uuid" {
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
                            } else if is_uppercase && inner_schema_type == "object" {
                                // For arrays of complex objects, embed the full nested schema
                                let type_name_str = type_name.clone();
                                quote! {
                                    // Create property for this array field with complex object items
                                    let mut props = ::serde_json::Map::new();
                                    props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));

                                    // Get the full schema for the nested struct type
                                    // Note: We need to embed this at runtime since we can't resolve types at macro time
                                    // We'll create a placeholder and let the schema enhancement logic handle it
                                    let mut items_schema = ::serde_json::Map::new();
                                    items_schema.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                                    items_schema.insert("description".to_string(),
                                        ::serde_json::Value::String(format!("Each {} must be a complete object with all required fields. MUST be an object, not a string.", #type_name_str)));

                                    // Try to get the nested struct's schema and embed its properties
                                    // This will work if the type implements SchemaType
                                    // We use a helper that will enhance the schema later
                                    props.insert("items".to_string(), ::serde_json::Value::Object(items_schema));
                                }
                            } else {
                                // Standard handling for other types
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
                        };

                        items_tokens
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
                } else {
                    // Regular non-array type
                    quote! {
                        // Create property for this field
                        let mut props = ::serde_json::Map::new();
                        props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));
                    }
                };
                property_setters.push(field_prop);

                // Add description if available
                if let Some(desc) = attrs.description {
                    let desc_prop = if is_likely_enum {
                        // For enum fields, enhance the description to include enum information
                        let type_name_str = type_name.clone().unwrap_or_else(|| "".to_string());
                        quote! {
                            props.insert("description".to_string(),
                                ::serde_json::Value::String(format!("{} (Must be one of the allowed enum values for {})", #desc, #type_name_str)));
                        }
                    } else if is_custom_type {
                        // For custom types, just use the description as is (CustomTypeSchema will be used)
                        quote! {
                            props.insert("description".to_string(), ::serde_json::Value::String(#desc.to_string()));
                        }
                    } else {
                        quote! {
                            props.insert("description".to_string(), ::serde_json::Value::String(#desc.to_string()));
                        }
                    };
                    property_setters.push(desc_prop);
                } else if is_likely_enum {
                    // If no description but it's an enum, add a note about using enum values
                    let type_name_str = type_name.clone().unwrap_or_else(|| "".to_string());
                    let desc_prop = quote! {
                        props.insert("description".to_string(),
                            ::serde_json::Value::String(format!("Must be one of the allowed enum values for {}", #type_name_str)));
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

    // Generate implementation
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

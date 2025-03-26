use proc_macro2::TokenStream;
use quote::quote;
use syn::{DataStruct, Fields, Ident};

use crate::container_attrs::ContainerAttributes;
use crate::parsers::field_parser::parse_field_attributes;
use crate::type_utils::{get_schema_type_from_rust_type, is_option_type};

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

                // Create field property
                let field_prop = quote! {
                    // Create property for this field
                    let mut props = ::serde_json::Map::new();
                    props.insert("type".to_string(), ::serde_json::Value::String(#schema_type.to_string()));
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
        _ => panic!("LLMModel can only be derived for structs with named fields"),
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

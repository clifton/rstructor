use proc_macro2::TokenStream;
use quote::quote;
use syn::{DataEnum, Fields, Ident, Type};

use crate::container_attrs::ContainerAttributes;
use crate::generators::struct_schema::apply_rename_all;
use crate::parsers::field_parser::parse_field_attributes;
use crate::parsers::variant_parser::parse_variant_attributes;
use crate::type_utils::{
    get_array_inner_type, get_schema_type_from_rust_type, is_array_type, is_option_type,
};

/// Generate the schema implementation for an enum
pub fn generate_enum_schema(
    name: &Ident,
    data_enum: &DataEnum,
    container_attrs: &ContainerAttributes,
) -> TokenStream {
    // Check if it's a simple enum (no data)
    let all_simple = data_enum.variants.iter().all(|v| v.fields.is_empty());

    if all_simple {
        // Generate implementation for simple enum as before
        generate_simple_enum_schema(name, data_enum, container_attrs)
    } else {
        // Generate implementation for enum with associated data
        generate_complex_enum_schema(name, data_enum, container_attrs)
    }
}

/// Generate schema for a simple enum (no associated data)
fn generate_simple_enum_schema(
    name: &Ident,
    data_enum: &DataEnum,
    container_attrs: &ContainerAttributes,
) -> TokenStream {
    // Generate implementation for simple enum with serde rename support
    let variant_values: Vec<_> = data_enum
        .variants
        .iter()
        .map(|v| {
            let attrs = parse_variant_attributes(v);
            let original_name = v.ident.to_string();
            // Priority: 1) variant #[serde(rename)], 2) container #[serde(rename_all)], 3) original name
            if let Some(ref rename) = attrs.serde_rename {
                rename.clone()
            } else if let Some(ref rename_all) = container_attrs.serde_rename_all {
                apply_rename_all(&original_name, rename_all)
            } else {
                original_name
            }
        })
        .collect();

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

    quote! {
        impl ::rstructor::schema::SchemaType for #name {
            fn schema() -> ::rstructor::schema::Schema {
                // Create array of enum values
                let enum_values = vec![
                    #(::serde_json::Value::String(#variant_values.to_string())),*
                ];

                let mut schema_obj = ::serde_json::json!({
                    "type": "string",
                    "enum": enum_values,
                    "title": stringify!(#name)
                });

                // Add container attributes if available
                #container_setter

                ::rstructor::schema::Schema::new(schema_obj)
            }

            fn schema_name() -> Option<String> {
                Some(stringify!(#name).to_string())
            }
        }
    }
}

/// Generate schema for a complex enum (with associated data)
fn generate_complex_enum_schema(
    name: &Ident,
    data_enum: &DataEnum,
    container_attrs: &ContainerAttributes,
) -> TokenStream {
    // Create variants for oneOf schema
    let mut variant_schemas = Vec::new();

    // Process each variant
    for variant in &data_enum.variants {
        // Get description and rename from variant attributes
        let attrs = parse_variant_attributes(variant);

        let original_variant_name = variant.ident.to_string();
        // Priority: 1) variant #[serde(rename)], 2) container #[serde(rename_all)], 3) original name
        let variant_name = if let Some(ref rename) = attrs.serde_rename {
            rename.clone()
        } else if let Some(ref rename_all) = container_attrs.serde_rename_all {
            apply_rename_all(&original_variant_name, rename_all)
        } else {
            original_variant_name.clone()
        };

        let description = attrs
            .description
            .unwrap_or_else(|| format!("Variant {}", variant_name));

        match &variant.fields {
            // For variants with no fields (simple enum variants)
            Fields::Unit => {
                let variant_name_str = variant_name.clone();
                let description_str = description.clone();
                variant_schemas.push(quote! {
                    // Simple variant with no data
                    ::serde_json::json!({
                        "type": "string",
                        "enum": [#variant_name_str],
                        "description": #description_str
                    })
                });
            }

            // For tuple-like variants with unnamed fields e.g., Variant(Type1, Type2)
            Fields::Unnamed(fields) => {
                let has_single_field = fields.unnamed.len() == 1;

                if has_single_field {
                    // Handle single unnamed field specially (more natural JSON)
                    let field = fields.unnamed.first().unwrap();

                    // Extract field schema based on its type
                    let field_schema = generate_field_schema(&field.ty, &None);
                    let variant_name_str = variant_name.clone();
                    let description_str = description.clone();

                    variant_schemas.push(quote! {
                        // Tuple variant with single field - { "variant": value }
                        {
                            let field_schema_value = #field_schema;
                            let mut properties_map = ::serde_json::Map::new();
                            properties_map.insert(#variant_name_str.to_string(), field_schema_value);

                            let mut required_array = Vec::new();
                            required_array.push(::serde_json::Value::String(#variant_name_str.to_string()));

                            let mut schema_obj = ::serde_json::Map::new();
                            schema_obj.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                            schema_obj.insert("properties".to_string(), ::serde_json::Value::Object(properties_map));
                            schema_obj.insert("required".to_string(), ::serde_json::Value::Array(required_array));
                            schema_obj.insert("description".to_string(), ::serde_json::Value::String(#description_str.to_string()));
                            schema_obj.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                            ::serde_json::Value::Object(schema_obj)
                        }
                    });
                } else {
                    // Multiple unnamed fields - use array format
                    let mut field_schemas = Vec::new();

                    for field in fields.unnamed.iter() {
                        let field_schema = generate_field_schema(&field.ty, &None);
                        field_schemas.push(field_schema);
                    }

                    let variant_name_str = variant_name.clone();
                    let description_str = description.clone();
                    let field_count = fields.unnamed.len();
                    variant_schemas.push(quote! {
                        // Tuple variant with multiple fields - { "variant": [values...] }
                        {
                            let field_schema_values: Vec<::serde_json::Value> = vec![
                                #(#field_schemas),*
                            ];

                            let mut items_array = ::serde_json::Map::new();
                            items_array.insert("type".to_string(), ::serde_json::Value::String("array".to_string()));
                            items_array.insert("items".to_string(), ::serde_json::Value::Array(field_schema_values));
                            let field_count_u64 = #field_count as u64;
                            items_array.insert("minItems".to_string(), ::serde_json::Value::Number(::serde_json::Number::from(field_count_u64)));
                            items_array.insert("maxItems".to_string(), ::serde_json::Value::Number(::serde_json::Number::from(field_count_u64)));

                            let mut variant_properties = ::serde_json::Map::new();
                            variant_properties.insert(#variant_name_str.to_string(), ::serde_json::Value::Object(items_array));

                            let mut required_array = Vec::new();
                            required_array.push(::serde_json::Value::String(#variant_name_str.to_string()));

                            let mut schema_obj = ::serde_json::Map::new();
                            schema_obj.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                            schema_obj.insert("properties".to_string(), ::serde_json::Value::Object(variant_properties));
                            schema_obj.insert("required".to_string(), ::serde_json::Value::Array(required_array));
                            schema_obj.insert("description".to_string(), ::serde_json::Value::String(#description_str.to_string()));
                            schema_obj.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                            ::serde_json::Value::Object(schema_obj)
                        }
                    });
                }
            }

            // For struct-like variants with named fields e.g., Variant { field1: Type1, field2: Type2 }
            Fields::Named(fields) => {
                let mut prop_setters = Vec::new();
                let mut required_fields = Vec::new();

                for field in &fields.named {
                    if let Some(field_ident) = &field.ident {
                        let original_field_name = field_ident.to_string();
                        let field_attrs = parse_field_attributes(field);

                        // Apply serde rename if present
                        let field_name_str = if let Some(ref rename) = field_attrs.serde_rename {
                            rename.clone()
                        } else {
                            original_field_name.clone()
                        };

                        let field_desc = field_attrs
                            .description
                            .unwrap_or_else(|| format!("Field {}", field_name_str));

                        let is_optional = is_option_type(&field.ty);
                        let field_schema = generate_field_schema(&field.ty, &Some(field_desc));

                        let field_name_str_owned = field_name_str.clone();
                        prop_setters.push(quote! {
                            {
                                let field_schema_value = #field_schema;
                                properties_map.insert(#field_name_str_owned.to_string(), field_schema_value);
                            }
                        });

                        if !is_optional {
                            required_fields.push(quote! {
                                ::serde_json::Value::String(#field_name_str.to_string())
                            });
                        }
                    }
                }

                let variant_name_str = variant_name.clone();
                let description_str = description.clone();
                let required_array_code = if !required_fields.is_empty() {
                    quote! {
                        let mut required_vec = Vec::new();
                        #(required_vec.push(#required_fields);)*
                        variant_properties.insert("required".to_string(), ::serde_json::Value::Array(required_vec));
                    }
                } else {
                    quote! {}
                };

                variant_schemas.push(quote! {
                    // Struct variant with named fields
                    {
                        let mut properties_map = ::serde_json::Map::new();
                        #(#prop_setters)*

                        let mut variant_properties = ::serde_json::Map::new();
                        variant_properties.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                        variant_properties.insert("properties".to_string(), ::serde_json::Value::Object(properties_map));
                        #required_array_code
                        variant_properties.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                        let mut outer_properties = ::serde_json::Map::new();
                        outer_properties.insert(#variant_name_str.to_string(), ::serde_json::Value::Object(variant_properties));

                        let mut schema_obj = ::serde_json::Map::new();
                        schema_obj.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                        schema_obj.insert("properties".to_string(), ::serde_json::Value::Object(outer_properties));

                        let mut required_array = Vec::new();
                        required_array.push(::serde_json::Value::String(#variant_name_str.to_string()));
                        schema_obj.insert("required".to_string(), ::serde_json::Value::Array(required_array));
                        schema_obj.insert("description".to_string(), ::serde_json::Value::String(#description_str.to_string()));
                        schema_obj.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                        ::serde_json::Value::Object(schema_obj)
                    }
                });
            }
        }
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

    // Generate the final schema implementation
    quote! {
        impl ::rstructor::schema::SchemaType for #name {
            fn schema() -> ::rstructor::schema::Schema {
                // Create oneOf schema for enum variants
                let variant_schemas = vec![
                    #(#variant_schemas),*
                ];

                let mut schema_obj = ::serde_json::json!({
                    "oneOf": variant_schemas,
                    "title": stringify!(#name)
                });

                // Add container attributes if available
                #container_setter

                ::rstructor::schema::Schema::new(schema_obj)
            }

            fn schema_name() -> Option<String> {
                Some(stringify!(#name).to_string())
            }
        }
    }
}

/// Generate schema for a field based on its type
fn generate_field_schema(field_type: &Type, description: &Option<String>) -> TokenStream {
    let schema_type = get_schema_type_from_rust_type(field_type);

    if is_array_type(field_type) {
        // For array types
        if let Some(inner_type) = get_array_inner_type(field_type) {
            let inner_schema_type = get_schema_type_from_rust_type(inner_type);

            let desc_prop = if let Some(desc) = description {
                quote! {
                    "description": #desc,
                }
            } else {
                quote! {}
            };

            quote! {
                ::serde_json::json!({
                    "type": #schema_type,
                    #desc_prop
                    "items": {
                        "type": #inner_schema_type
                    }
                })
            }
        } else {
            // Fallback for array without detectable item type
            let desc_prop = if let Some(desc) = description {
                quote! {
                    "description": #desc,
                }
            } else {
                quote! {}
            };

            quote! {
                ::serde_json::json!({
                    "type": #schema_type,
                    #desc_prop
                    "items": {
                        "type": "string"
                    }
                })
            }
        }
    } else if schema_type == "object" {
        // For object types (custom types)
        // Try to access schema of nested type
        match field_type {
            Type::Path(type_path) => {
                let last_segment = type_path.path.segments.last();
                if let Some(_segment) = last_segment {
                    // We don't need the type name for now, but this structure is useful for future enhancements

                    // Use the type's schema if it implements SchemaType
                    // Note: This assumes the type implements SchemaType (which it will if it has #[derive(Instructor)])
                    if let Some(desc) = description {
                        let desc_str = desc.clone();
                        quote! {
                            {
                                // Use the type's schema directly (it must implement SchemaType)
                                let mut obj = <#type_path as ::rstructor::schema::SchemaType>::schema().to_json().clone();

                                // Add description if provided
                                if let ::serde_json::Value::Object(map) = &mut obj {
                                    map.insert("description".to_string(), ::serde_json::Value::String(#desc_str.to_string()));
                                }

                                obj
                            }
                        }
                    } else {
                        quote! {
                            {
                                // Use the type's schema directly (it must implement SchemaType)
                                <#type_path as ::rstructor::schema::SchemaType>::schema().to_json()
                            }
                        }
                    }
                } else {
                    // Fallback for unidentifiable object type
                    let desc_prop = if let Some(desc) = description {
                        quote! {
                            "description": #desc,
                        }
                    } else {
                        quote! {}
                    };

                    quote! {
                        ::serde_json::json!({
                            "type": "object",
                            #desc_prop
                        })
                    }
                }
            }
            _ => {
                // Fallback for non-path type
                let desc_prop = if let Some(desc) = description {
                    quote! {
                        "description": #desc,
                    }
                } else {
                    quote! {}
                };

                quote! {
                    ::serde_json::json!({
                        "type": "object",
                        #desc_prop
                    })
                }
            }
        }
    } else {
        // For primitive types
        let desc_prop = if let Some(desc) = description {
            quote! {
                "description": #desc,
            }
        } else {
            quote! {}
        };

        quote! {
            ::serde_json::json!({
                "type": #schema_type,
                #desc_prop
            })
        }
    }
}

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DataEnum, Ident, Fields, Type};

use crate::container_attrs::ContainerAttributes;
use crate::parsers::field_parser::parse_field_attributes;
use crate::parsers::variant_parser::parse_variant_attributes;
use crate::type_utils::{get_schema_type_from_rust_type, is_option_type, is_array_type, get_array_inner_type};

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
    // Generate implementation for simple enum
    let variant_values: Vec<_> = data_enum
        .variants
        .iter()
        .map(|v| v.ident.to_string())
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
        let variant_name = variant.ident.to_string();
        
        // Get description from variant attributes if available
        let attrs = parse_variant_attributes(variant);
        let description = attrs.description.unwrap_or_else(|| format!("Variant {}", variant_name));
        
        match &variant.fields {
            // For variants with no fields (simple enum variants)
            Fields::Unit => {
                variant_schemas.push(quote! {
                    // Simple variant with no data
                    ::serde_json::json!({
                        "type": "string",
                        "enum": [#variant_name],
                        "description": #description
                    })
                });
            },
            
            // For tuple-like variants with unnamed fields e.g., Variant(Type1, Type2)
            Fields::Unnamed(fields) => {
                let has_single_field = fields.unnamed.len() == 1;
                
                if has_single_field {
                    // Handle single unnamed field specially (more natural JSON)
                    let field = fields.unnamed.first().unwrap();
                    let field_type = get_schema_type_from_rust_type(&field.ty);
                    let is_opt = is_option_type(&field.ty);
                    
                    // Extract field schema based on its type
                    let field_schema = generate_field_schema(&field.ty, &None);
                    
                    variant_schemas.push(quote! {
                        // Tuple variant with single field - { "variant": value }
                        ::serde_json::json!({
                            "type": "object",
                            "properties": {
                                #variant_name: #field_schema
                            },
                            "required": [#variant_name],
                            "description": #description,
                            "additionalProperties": false
                        })
                    });
                } else {
                    // Multiple unnamed fields - use array format
                    let mut field_schemas = Vec::new();
                    
                    for (i, field) in fields.unnamed.iter().enumerate() {
                        let field_schema = generate_field_schema(&field.ty, &None);
                        field_schemas.push(field_schema);
                    }
                    
                    variant_schemas.push(quote! {
                        // Tuple variant with multiple fields - { "variant": [values...] }
                        ::serde_json::json!({
                            "type": "object",
                            "properties": {
                                #variant_name: {
                                    "type": "array",
                                    "items": [
                                        #(#field_schemas),*
                                    ],
                                    "minItems": #fields.unnamed.len(),
                                    "maxItems": #fields.unnamed.len()
                                }
                            },
                            "required": [#variant_name],
                            "description": #description,
                            "additionalProperties": false
                        })
                    });
                }
            },
            
            // For struct-like variants with named fields e.g., Variant { field1: Type1, field2: Type2 }
            Fields::Named(fields) => {
                let mut prop_schemas = Vec::new();
                let mut required_fields = Vec::new();
                
                for field in &fields.named {
                    if let Some(field_name) = &field.ident {
                        let field_name_str = field_name.to_string();
                        let field_attrs = parse_field_attributes(field);
                        let field_desc = field_attrs.description.unwrap_or_else(|| format!("Field {}", field_name_str));
                        
                        let is_optional = is_option_type(&field.ty);
                        let field_schema = generate_field_schema(&field.ty, &Some(field_desc));
                        
                        prop_schemas.push(quote! {
                            #field_name_str: #field_schema
                        });
                        
                        if !is_optional {
                            required_fields.push(quote! {
                                ::serde_json::Value::String(#field_name_str.to_string())
                            });
                        }
                    }
                }
                
                let required_array = if !required_fields.is_empty() {
                    quote! {
                        "required": [#(#required_fields),*],
                    }
                } else {
                    quote! { }
                };
                
                variant_schemas.push(quote! {
                    // Struct variant with named fields
                    ::serde_json::json!({
                        "type": "object",
                        "properties": {
                            #variant_name: {
                                "type": "object",
                                "properties": {
                                    #(#prop_schemas),*
                                },
                                #required_array
                                "additionalProperties": false
                            }
                        },
                        "required": [#variant_name],
                        "description": #description,
                        "additionalProperties": false
                    })
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
                if let Some(segment) = last_segment {
                    let type_name = &segment.ident;
                    
                    let desc_prop = if let Some(desc) = description {
                        quote! {
                            "description": #desc,
                        }
                    } else {
                        quote! {}
                    };
                    
                    // Use the type's schema if it implements SchemaType
                    quote! {
                        {
                            // Try to use the type's schema if available
                            if let Some(schema) = <#type_path as ::rstructor::schema::SchemaType>::schema_name() {
                                let mut obj = <#type_path as ::rstructor::schema::SchemaType>::schema().to_json().clone();
                                
                                // Add description if provided
                                if let ::serde_json::Value::Object(ref mut map) = obj {
                                    #desc_prop
                                }
                                
                                obj
                            } else {
                                // Fallback to simple object schema
                                ::serde_json::json!({
                                    "type": "object",
                                    #desc_prop
                                })
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
            },
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

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DataEnum, Fields, Ident, Type};

use crate::container_attrs::ContainerAttributes;
use crate::generators::struct_schema::apply_rename_all;
use crate::parsers::field_parser::parse_field_attributes;
use crate::parsers::variant_parser::parse_variant_attributes;
use crate::type_utils::{
    get_array_inner_type, get_box_inner_type, get_map_types, get_option_inner_type,
    get_schema_type_from_rust_type, get_tuple_element_types, is_array_type, is_box_type,
    is_json_value_type, is_map_type, is_option_type, is_tuple_type,
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
    // Dispatch to appropriate generator based on serde tagging mode
    if container_attrs.serde_untagged {
        return generate_untagged_enum_schema(name, data_enum, container_attrs);
    } else if let Some(tag) = &container_attrs.serde_tag {
        if let Some(content) = &container_attrs.serde_content {
            // Adjacent tagging: #[serde(tag = "...", content = "...")]
            return generate_adjacently_tagged_enum_schema(
                name,
                data_enum,
                container_attrs,
                tag,
                content,
            );
        } else {
            // Internal tagging: #[serde(tag = "...")]
            return generate_internally_tagged_enum_schema(
                name,
                data_enum,
                container_attrs,
                tag,
            );
        }
    }

    // Default: External tagging (current behavior)
    generate_externally_tagged_enum_schema(name, data_enum, container_attrs)
}

/// Generate schema for externally tagged enums (default serde behavior)
/// Format: {"VariantName": ...data...}
fn generate_externally_tagged_enum_schema(
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
    let is_optional = is_option_type(field_type);
    let actual_type = if is_optional {
        get_option_inner_type(field_type)
    } else {
        field_type
    };

    let desc_prop = if let Some(desc) = description {
        quote! { "description": #desc, }
    } else {
        quote! {}
    };

    // Handle serde_json::Value
    if is_json_value_type(actual_type) {
        return quote! {
            ::serde_json::json!({
                #desc_prop
            })
        };
    }

    // Handle HashMap/BTreeMap
    if is_map_type(actual_type)
        && let Some((_key_ty, val_ty)) = get_map_types(actual_type) {
            let val_schema_type = get_schema_type_from_rust_type(val_ty);
            if val_schema_type == "object" {
                return quote! {
                    {
                        let mut schema = ::serde_json::json!({
                            "type": "object",
                            #desc_prop
                        });
                        let value_schema = <#val_ty as ::rstructor::schema::SchemaType>::schema();
                        if let ::serde_json::Value::Object(map) = &mut schema {
                            map.insert("additionalProperties".to_string(), value_schema.to_json());
                        }
                        schema
                    }
                };
            } else {
                return quote! {
                    ::serde_json::json!({
                        "type": "object",
                        #desc_prop
                        "additionalProperties": {
                            "type": #val_schema_type
                        }
                    })
                };
            }
        }

    // Handle Box<T>
    if is_box_type(actual_type)
        && let Some(inner_ty) = get_box_inner_type(actual_type) {
            return generate_field_schema(inner_ty, description);
        }

    // Handle tuples
    if is_tuple_type(actual_type)
        && let Some(element_types) = get_tuple_element_types(actual_type) {
            let element_count = element_types.len();
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
            return quote! {
                {
                    let prefix_items = vec![
                        #(#element_schemas),*
                    ];
                    ::serde_json::json!({
                        "type": "array",
                        #desc_prop
                        "prefixItems": prefix_items,
                        "minItems": #element_count,
                        "maxItems": #element_count
                    })
                }
            };
        }

    // Handle array types
    if is_array_type(actual_type) {
        if let Some(inner_type) = get_array_inner_type(actual_type) {
            let inner_schema_type = get_schema_type_from_rust_type(inner_type);

            if inner_schema_type == "object" {
                // For arrays of complex types
                return quote! {
                    {
                        let items_schema = <#inner_type as ::rstructor::schema::SchemaType>::schema().to_json();
                        ::serde_json::json!({
                            "type": "array",
                            #desc_prop
                            "items": items_schema
                        })
                    }
                };
            } else {
                return quote! {
                    ::serde_json::json!({
                        "type": "array",
                        #desc_prop
                        "items": {
                            "type": #inner_schema_type
                        }
                    })
                };
            }
        } else {
            return quote! {
                ::serde_json::json!({
                    "type": "array",
                    #desc_prop
                    "items": {
                        "type": "string"
                    }
                })
            };
        }
    }

    // Handle custom object types
    if schema_type == "object"
        && let Type::Path(type_path) = actual_type
        && type_path.path.segments.last().is_some()
    {
        if let Some(desc) = description {
            let desc_str = desc.clone();
            return quote! {
                {
                    let mut obj = <#type_path as ::rstructor::schema::SchemaType>::schema().to_json().clone();
                    if let ::serde_json::Value::Object(map) = &mut obj {
                        map.insert("description".to_string(), ::serde_json::Value::String(#desc_str.to_string()));
                    }
                    obj
                }
            };
        } else {
            return quote! {
                <#type_path as ::rstructor::schema::SchemaType>::schema().to_json()
            };
        }
    }

    // Fallback for primitive types
    quote! {
        ::serde_json::json!({
            "type": #schema_type,
            #desc_prop
        })
    }
}

/// Generate schema for internally tagged enums
/// Format: {"tag_name": "VariantName", ...fields...}
/// Only works with struct variants (named fields)
fn generate_internally_tagged_enum_schema(
    name: &Ident,
    data_enum: &DataEnum,
    container_attrs: &ContainerAttributes,
    tag_name: &str,
) -> TokenStream {
    let mut variant_schemas = Vec::new();

    for variant in &data_enum.variants {
        let attrs = parse_variant_attributes(variant);
        let original_variant_name = variant.ident.to_string();
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
            Fields::Unit => {
                // Unit variant: {"tag": "VariantName"}
                let variant_name_str = variant_name.clone();
                let description_str = description.clone();
                let tag_name_str = tag_name.to_string();
                variant_schemas.push(quote! {
                    {
                        let mut schema_obj = ::serde_json::Map::new();
                        schema_obj.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));

                        let mut properties = ::serde_json::Map::new();
                        properties.insert(#tag_name_str.to_string(), ::serde_json::json!({
                            "type": "string",
                            "enum": [#variant_name_str]
                        }));
                        schema_obj.insert("properties".to_string(), ::serde_json::Value::Object(properties));

                        let required = vec![::serde_json::Value::String(#tag_name_str.to_string())];
                        schema_obj.insert("required".to_string(), ::serde_json::Value::Array(required));
                        schema_obj.insert("description".to_string(), ::serde_json::Value::String(#description_str.to_string()));
                        schema_obj.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                        ::serde_json::Value::Object(schema_obj)
                    }
                });
            }
            Fields::Named(fields) => {
                // Struct variant: {"tag": "VariantName", field1: ..., field2: ...}
                let mut prop_setters = Vec::new();
                let mut required_fields = vec![quote! {
                    ::serde_json::Value::String(#tag_name.to_string())
                }];

                for field in &fields.named {
                    if let Some(field_ident) = &field.ident {
                        let original_field_name = field_ident.to_string();
                        let field_attrs = parse_field_attributes(field);

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
                                properties.insert(#field_name_str_owned.to_string(), field_schema_value);
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
                let tag_name_str = tag_name.to_string();
                variant_schemas.push(quote! {
                    {
                        let mut schema_obj = ::serde_json::Map::new();
                        schema_obj.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));

                        let mut properties = ::serde_json::Map::new();
                        // Add the tag property
                        properties.insert(#tag_name_str.to_string(), ::serde_json::json!({
                            "type": "string",
                            "enum": [#variant_name_str]
                        }));

                        // Add variant fields
                        #(#prop_setters)*

                        schema_obj.insert("properties".to_string(), ::serde_json::Value::Object(properties));

                        let required = vec![#(#required_fields),*];
                        schema_obj.insert("required".to_string(), ::serde_json::Value::Array(required));
                        schema_obj.insert("description".to_string(), ::serde_json::Value::String(#description_str.to_string()));
                        schema_obj.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                        ::serde_json::Value::Object(schema_obj)
                    }
                });
            }
            Fields::Unnamed(_) => {
                // Internal tagging doesn't support tuple variants in serde
                // Fall back to treating it as a unit variant with the tag only
                let variant_name_str = variant_name.clone();
                let description_str = description.clone();
                let tag_name_str = tag_name.to_string();
                variant_schemas.push(quote! {
                    {
                        let mut schema_obj = ::serde_json::Map::new();
                        schema_obj.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));

                        let mut properties = ::serde_json::Map::new();
                        properties.insert(#tag_name_str.to_string(), ::serde_json::json!({
                            "type": "string",
                            "enum": [#variant_name_str]
                        }));
                        schema_obj.insert("properties".to_string(), ::serde_json::Value::Object(properties));

                        let required = vec![::serde_json::Value::String(#tag_name_str.to_string())];
                        schema_obj.insert("required".to_string(), ::serde_json::Value::Array(required));
                        schema_obj.insert("description".to_string(), ::serde_json::Value::String(#description_str.to_string()));
                        schema_obj.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                        ::serde_json::Value::Object(schema_obj)
                    }
                });
            }
        }
    }

    // Container attributes
    let container_setter = generate_container_setters(container_attrs);

    quote! {
        impl ::rstructor::schema::SchemaType for #name {
            fn schema() -> ::rstructor::schema::Schema {
                let variant_schemas = vec![
                    #(#variant_schemas),*
                ];

                let mut schema_obj = ::serde_json::json!({
                    "oneOf": variant_schemas,
                    "title": stringify!(#name)
                });

                #container_setter

                ::rstructor::schema::Schema::new(schema_obj)
            }

            fn schema_name() -> Option<String> {
                Some(stringify!(#name).to_string())
            }
        }
    }
}

/// Generate schema for adjacently tagged enums
/// Format: {"tag_name": "VariantName", "content_name": ...data...}
fn generate_adjacently_tagged_enum_schema(
    name: &Ident,
    data_enum: &DataEnum,
    container_attrs: &ContainerAttributes,
    tag_name: &str,
    content_name: &str,
) -> TokenStream {
    let mut variant_schemas = Vec::new();

    for variant in &data_enum.variants {
        let attrs = parse_variant_attributes(variant);
        let original_variant_name = variant.ident.to_string();
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
            Fields::Unit => {
                // Unit variant: {"tag": "VariantName"} - no content field
                let variant_name_str = variant_name.clone();
                let description_str = description.clone();
                let tag_name_str = tag_name.to_string();
                variant_schemas.push(quote! {
                    {
                        let mut schema_obj = ::serde_json::Map::new();
                        schema_obj.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));

                        let mut properties = ::serde_json::Map::new();
                        properties.insert(#tag_name_str.to_string(), ::serde_json::json!({
                            "type": "string",
                            "enum": [#variant_name_str]
                        }));
                        schema_obj.insert("properties".to_string(), ::serde_json::Value::Object(properties));

                        let required = vec![::serde_json::Value::String(#tag_name_str.to_string())];
                        schema_obj.insert("required".to_string(), ::serde_json::Value::Array(required));
                        schema_obj.insert("description".to_string(), ::serde_json::Value::String(#description_str.to_string()));
                        schema_obj.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                        ::serde_json::Value::Object(schema_obj)
                    }
                });
            }
            Fields::Unnamed(fields) => {
                let variant_name_str = variant_name.clone();
                let description_str = description.clone();
                let tag_name_str = tag_name.to_string();
                let content_name_str = content_name.to_string();

                if fields.unnamed.len() == 1 {
                    // Single field: {"tag": "Variant", "content": value}
                    let field = fields.unnamed.first().unwrap();
                    let field_schema = generate_field_schema(&field.ty, &None);

                    variant_schemas.push(quote! {
                        {
                            let mut schema_obj = ::serde_json::Map::new();
                            schema_obj.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));

                            let mut properties = ::serde_json::Map::new();
                            properties.insert(#tag_name_str.to_string(), ::serde_json::json!({
                                "type": "string",
                                "enum": [#variant_name_str]
                            }));
                            properties.insert(#content_name_str.to_string(), #field_schema);
                            schema_obj.insert("properties".to_string(), ::serde_json::Value::Object(properties));

                            let required = vec![
                                ::serde_json::Value::String(#tag_name_str.to_string()),
                                ::serde_json::Value::String(#content_name_str.to_string())
                            ];
                            schema_obj.insert("required".to_string(), ::serde_json::Value::Array(required));
                            schema_obj.insert("description".to_string(), ::serde_json::Value::String(#description_str.to_string()));
                            schema_obj.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                            ::serde_json::Value::Object(schema_obj)
                        }
                    });
                } else {
                    // Multiple fields: {"tag": "Variant", "content": [values...]}
                    let mut field_schemas = Vec::new();
                    for field in fields.unnamed.iter() {
                        let field_schema = generate_field_schema(&field.ty, &None);
                        field_schemas.push(field_schema);
                    }
                    let field_count = fields.unnamed.len();

                    variant_schemas.push(quote! {
                        {
                            let mut schema_obj = ::serde_json::Map::new();
                            schema_obj.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));

                            let mut properties = ::serde_json::Map::new();
                            properties.insert(#tag_name_str.to_string(), ::serde_json::json!({
                                "type": "string",
                                "enum": [#variant_name_str]
                            }));

                            let field_schema_values: Vec<::serde_json::Value> = vec![
                                #(#field_schemas),*
                            ];
                            let mut content_schema = ::serde_json::Map::new();
                            content_schema.insert("type".to_string(), ::serde_json::Value::String("array".to_string()));
                            content_schema.insert("items".to_string(), ::serde_json::Value::Array(field_schema_values));
                            let field_count_u64 = #field_count as u64;
                            content_schema.insert("minItems".to_string(), ::serde_json::Value::Number(::serde_json::Number::from(field_count_u64)));
                            content_schema.insert("maxItems".to_string(), ::serde_json::Value::Number(::serde_json::Number::from(field_count_u64)));

                            properties.insert(#content_name_str.to_string(), ::serde_json::Value::Object(content_schema));
                            schema_obj.insert("properties".to_string(), ::serde_json::Value::Object(properties));

                            let required = vec![
                                ::serde_json::Value::String(#tag_name_str.to_string()),
                                ::serde_json::Value::String(#content_name_str.to_string())
                            ];
                            schema_obj.insert("required".to_string(), ::serde_json::Value::Array(required));
                            schema_obj.insert("description".to_string(), ::serde_json::Value::String(#description_str.to_string()));
                            schema_obj.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                            ::serde_json::Value::Object(schema_obj)
                        }
                    });
                }
            }
            Fields::Named(fields) => {
                // Struct variant: {"tag": "Variant", "content": {field1: ..., field2: ...}}
                let mut prop_setters = Vec::new();
                let mut required_content_fields = Vec::new();

                for field in &fields.named {
                    if let Some(field_ident) = &field.ident {
                        let original_field_name = field_ident.to_string();
                        let field_attrs = parse_field_attributes(field);

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
                                content_properties.insert(#field_name_str_owned.to_string(), field_schema_value);
                            }
                        });

                        if !is_optional {
                            required_content_fields.push(quote! {
                                ::serde_json::Value::String(#field_name_str.to_string())
                            });
                        }
                    }
                }

                let variant_name_str = variant_name.clone();
                let description_str = description.clone();
                let tag_name_str = tag_name.to_string();
                let content_name_str = content_name.to_string();
                let required_content_code = if !required_content_fields.is_empty() {
                    quote! {
                        let required_content = vec![#(#required_content_fields),*];
                        content_schema.insert("required".to_string(), ::serde_json::Value::Array(required_content));
                    }
                } else {
                    quote! {}
                };

                variant_schemas.push(quote! {
                    {
                        let mut schema_obj = ::serde_json::Map::new();
                        schema_obj.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));

                        let mut properties = ::serde_json::Map::new();
                        properties.insert(#tag_name_str.to_string(), ::serde_json::json!({
                            "type": "string",
                            "enum": [#variant_name_str]
                        }));

                        // Build content schema as an object
                        let mut content_properties = ::serde_json::Map::new();
                        #(#prop_setters)*

                        let mut content_schema = ::serde_json::Map::new();
                        content_schema.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));
                        content_schema.insert("properties".to_string(), ::serde_json::Value::Object(content_properties));
                        #required_content_code
                        content_schema.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                        properties.insert(#content_name_str.to_string(), ::serde_json::Value::Object(content_schema));
                        schema_obj.insert("properties".to_string(), ::serde_json::Value::Object(properties));

                        let required = vec![
                            ::serde_json::Value::String(#tag_name_str.to_string()),
                            ::serde_json::Value::String(#content_name_str.to_string())
                        ];
                        schema_obj.insert("required".to_string(), ::serde_json::Value::Array(required));
                        schema_obj.insert("description".to_string(), ::serde_json::Value::String(#description_str.to_string()));
                        schema_obj.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                        ::serde_json::Value::Object(schema_obj)
                    }
                });
            }
        }
    }

    // Container attributes
    let container_setter = generate_container_setters(container_attrs);

    quote! {
        impl ::rstructor::schema::SchemaType for #name {
            fn schema() -> ::rstructor::schema::Schema {
                let variant_schemas = vec![
                    #(#variant_schemas),*
                ];

                let mut schema_obj = ::serde_json::json!({
                    "oneOf": variant_schemas,
                    "title": stringify!(#name)
                });

                #container_setter

                ::rstructor::schema::Schema::new(schema_obj)
            }

            fn schema_name() -> Option<String> {
                Some(stringify!(#name).to_string())
            }
        }
    }
}

/// Generate schema for untagged enums
/// Format: Just the data, discriminated by structure
fn generate_untagged_enum_schema(
    name: &Ident,
    data_enum: &DataEnum,
    container_attrs: &ContainerAttributes,
) -> TokenStream {
    let mut variant_schemas = Vec::new();

    for variant in &data_enum.variants {
        let attrs = parse_variant_attributes(variant);
        let variant_name = variant.ident.to_string();
        let description = attrs
            .description
            .unwrap_or_else(|| format!("Variant {}", variant_name));

        match &variant.fields {
            Fields::Unit => {
                // Unit variants are problematic in untagged enums
                // They serialize as null
                let description_str = description.clone();
                variant_schemas.push(quote! {
                    ::serde_json::json!({
                        "type": "null",
                        "description": #description_str
                    })
                });
            }
            Fields::Unnamed(fields) => {
                if fields.unnamed.len() == 1 {
                    // Single field - just the value
                    let field = fields.unnamed.first().unwrap();
                    let field_schema = generate_field_schema(&field.ty, &Some(description.clone()));
                    variant_schemas.push(quote! { #field_schema });
                } else {
                    // Multiple fields - array
                    let mut field_schemas = Vec::new();
                    for field in fields.unnamed.iter() {
                        let field_schema = generate_field_schema(&field.ty, &None);
                        field_schemas.push(field_schema);
                    }
                    let field_count = fields.unnamed.len();
                    let description_str = description.clone();
                    variant_schemas.push(quote! {
                        {
                            let field_schema_values: Vec<::serde_json::Value> = vec![
                                #(#field_schemas),*
                            ];
                            let mut schema_obj = ::serde_json::Map::new();
                            schema_obj.insert("type".to_string(), ::serde_json::Value::String("array".to_string()));
                            schema_obj.insert("items".to_string(), ::serde_json::Value::Array(field_schema_values));
                            let field_count_u64 = #field_count as u64;
                            schema_obj.insert("minItems".to_string(), ::serde_json::Value::Number(::serde_json::Number::from(field_count_u64)));
                            schema_obj.insert("maxItems".to_string(), ::serde_json::Value::Number(::serde_json::Number::from(field_count_u64)));
                            schema_obj.insert("description".to_string(), ::serde_json::Value::String(#description_str.to_string()));

                            ::serde_json::Value::Object(schema_obj)
                        }
                    });
                }
            }
            Fields::Named(fields) => {
                // Struct - just the object with fields
                let mut prop_setters = Vec::new();
                let mut required_fields = Vec::new();

                for field in &fields.named {
                    if let Some(field_ident) = &field.ident {
                        let original_field_name = field_ident.to_string();
                        let field_attrs = parse_field_attributes(field);

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
                                properties.insert(#field_name_str_owned.to_string(), field_schema_value);
                            }
                        });

                        if !is_optional {
                            required_fields.push(quote! {
                                ::serde_json::Value::String(#field_name_str.to_string())
                            });
                        }
                    }
                }

                let description_str = description.clone();
                let required_code = if !required_fields.is_empty() {
                    quote! {
                        let required = vec![#(#required_fields),*];
                        schema_obj.insert("required".to_string(), ::serde_json::Value::Array(required));
                    }
                } else {
                    quote! {}
                };

                variant_schemas.push(quote! {
                    {
                        let mut schema_obj = ::serde_json::Map::new();
                        schema_obj.insert("type".to_string(), ::serde_json::Value::String("object".to_string()));

                        let mut properties = ::serde_json::Map::new();
                        #(#prop_setters)*

                        schema_obj.insert("properties".to_string(), ::serde_json::Value::Object(properties));
                        #required_code
                        schema_obj.insert("description".to_string(), ::serde_json::Value::String(#description_str.to_string()));
                        schema_obj.insert("additionalProperties".to_string(), ::serde_json::Value::Bool(false));

                        ::serde_json::Value::Object(schema_obj)
                    }
                });
            }
        }
    }

    // Container attributes
    let container_setter = generate_container_setters(container_attrs);

    quote! {
        impl ::rstructor::schema::SchemaType for #name {
            fn schema() -> ::rstructor::schema::Schema {
                let variant_schemas = vec![
                    #(#variant_schemas),*
                ];

                let mut schema_obj = ::serde_json::json!({
                    "oneOf": variant_schemas,
                    "title": stringify!(#name)
                });

                #container_setter

                ::rstructor::schema::Schema::new(schema_obj)
            }

            fn schema_name() -> Option<String> {
                Some(stringify!(#name).to_string())
            }
        }
    }
}

/// Generate container attribute setters (shared helper)
fn generate_container_setters(container_attrs: &ContainerAttributes) -> TokenStream {
    let mut container_setters = Vec::new();

    if let Some(desc) = &container_attrs.description {
        container_setters.push(quote! {
            schema_obj["description"] = ::serde_json::Value::String(#desc.to_string());
        });
    }

    if let Some(title) = &container_attrs.title {
        container_setters.push(quote! {
            schema_obj["title"] = ::serde_json::Value::String(#title.to_string());
        });
    }

    if !container_attrs.examples.is_empty() {
        let examples_values = &container_attrs.examples;
        container_setters.push(quote! {
            let examples_array = vec![
                #(#examples_values),*
            ];
            schema_obj["examples"] = ::serde_json::Value::Array(examples_array);
        });
    }

    if !container_setters.is_empty() {
        quote! {
            #(#container_setters)*
        }
    } else {
        quote! {}
    }
}

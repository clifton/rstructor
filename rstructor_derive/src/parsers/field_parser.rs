use proc_macro2::TokenStream;
use quote::quote;
use syn::Field;

use crate::parsers::array_parser::parse_array_literal;
use crate::type_utils::{TypeCategory, get_option_inner_type, get_type_category, is_option_type};

/// Represents parsed field attributes
pub struct FieldAttributes {
    pub description: Option<String>,
    pub example_value: Option<TokenStream>,
    pub examples_array: Vec<TokenStream>,
    /// Field rename from #[serde(rename = "...")]
    pub serde_rename: Option<String>,
}

/// Parse a single field's llm and serde attributes
pub fn parse_field_attributes(field: &Field) -> FieldAttributes {
    let mut description = None;
    let mut example_value = None;
    let mut examples_array = Vec::new();
    let mut serde_rename = None;

    // Get the base type (unwrapping Option if present)
    let is_optional = is_option_type(&field.ty);
    let base_type = if is_optional {
        get_option_inner_type(&field.ty)
    } else {
        &field.ty
    };

    // Extract attributes
    for attr in &field.attrs {
        // Parse serde attributes for rename
        if attr.path().is_ident("serde") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("rename") {
                    let value = meta.value()?;
                    let content: syn::LitStr = value.parse()?;
                    serde_rename = Some(content.value());
                }
                Ok(())
            });
        }

        if attr.path().is_ident("llm") {
            // Parse attribute arguments
            let _result = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("description") {
                    let value = meta.value()?;
                    let content: syn::LitStr = value.parse()?;
                    description = Some(content.value());
                } else if meta.path.is_ident("example") {
                    let value = meta.value()?;

                    // First, try to parse as an array literal for array types
                    if let TypeCategory::Array = get_type_category(base_type) {
                        // Try to parse as an array literal
                        if let Some(array_tokens) = parse_array_literal(value) {
                            example_value = Some(quote! {
                                ::serde_json::Value::Array(vec![#(#array_tokens),*])
                            });
                            return Ok(());
                        }
                    }

                    // If not an array literal or not an array type, parse based on field type
                    match get_type_category(base_type) {
                        TypeCategory::String => {
                            let content: syn::LitStr = value.parse()?;
                            example_value = Some(quote! {
                                ::serde_json::Value::String(#content.to_string())
                            });
                        },
                        TypeCategory::Integer => {
                            if let Ok(content) = value.parse::<syn::LitInt>() {
                                example_value = Some(quote! {
                                    ::serde_json::Value::Number(::serde_json::Number::from(#content))
                                });
                            } else if let Ok(content) = value.parse::<syn::LitStr>() {
                                // Allow string for integer too
                                let int_str = content.value();
                                example_value = Some(quote! {
                                    match #int_str.parse::<i64>() {
                                        Ok(num) => ::serde_json::Value::Number(::serde_json::Number::from(num)),
                                        Err(_) => ::serde_json::Value::String(#int_str.to_string())
                                    }
                                });
                            }
                        },
                        TypeCategory::Float => {
                            if let Ok(content) = value.parse::<syn::LitFloat>() {
                                let float_str = content.to_string();
                                example_value = Some(quote! {
                                    match #float_str.parse::<f64>() {
                                        Ok(num) => ::serde_json::json!(num),
                                        Err(_) => ::serde_json::Value::String(#float_str.to_string())
                                    }
                                });
                            } else if let Ok(content) = value.parse::<syn::LitStr>() {
                                // Allow string for float too
                                let float_str = content.value();
                                example_value = Some(quote! {
                                    match #float_str.parse::<f64>() {
                                        Ok(num) => ::serde_json::json!(num),
                                        Err(_) => ::serde_json::Value::String(#float_str.to_string())
                                    }
                                });
                            }
                        },
                        TypeCategory::Boolean => {
                            if let Ok(content) = value.parse::<syn::LitBool>() {
                                let bool_val = content.value;
                                example_value = Some(quote! {
                                    ::serde_json::Value::Bool(#bool_val)
                                });
                            } else if let Ok(content) = value.parse::<syn::LitStr>() {
                                // Allow string for boolean too
                                let bool_str = content.value();
                                example_value = Some(quote! {
                                    match #bool_str.parse::<bool>() {
                                        Ok(val) => ::serde_json::Value::Bool(val),
                                        Err(_) => ::serde_json::Value::String(#bool_str.to_string())
                                    }
                                });
                            }
                        },
                        TypeCategory::Array => {
                            // For array types, we need to handle the special array syntax
                            // Parse the attribute value as a string first
                            if let Ok(content) = value.parse::<syn::LitStr>() {
                                let array_str = content.value();

                                // Check if it's a bracketed array like ["a", "b", "c"]
                                if array_str.starts_with('[') && array_str.ends_with(']') {
                                    // Parse as JSON array, but convert single quotes to double quotes
                                    let json_array = array_str.replace('\'', "\"");
                                    example_value = Some(quote! {
                                        match ::serde_json::from_str(#json_array) {
                                            Ok(val) => val,
                                            Err(_) => ::serde_json::Value::String(#array_str.to_string())
                                        }
                                    });
                                } else {
                                    // Treat as a single string
                                    example_value = Some(quote! {
                                        ::serde_json::Value::Array(vec![::serde_json::Value::String(#array_str.to_string())])
                                    });
                                }
                            }
                        },
                        TypeCategory::Object => {
                            // For object types, parse as JSON string
                            if let Ok(content) = value.parse::<syn::LitStr>() {
                                let json_str = content.value();
                                example_value = Some(quote! {
                                    match ::serde_json::from_str(#json_str) {
                                        Ok(val) => val,
                                        Err(_) => ::serde_json::Value::String(#json_str.to_string())
                                    }
                                });
                            }
                        }
                    }
                } else if meta.path.is_ident("examples") {
                    // First, try to parse as an array literal
                    let value = meta.value()?;

                    if let Some(array_tokens) = parse_array_literal(value) {
                        // Use the parsed array tokens directly
                        examples_array = array_tokens;
                        return Ok(());
                    }

                    // If not an array literal, handle examples based on type
                    match get_type_category(base_type) {
                        TypeCategory::String => {
                            // Parse array of string literals for string types
                            meta.parse_nested_meta(|nested_meta| {
                                if let Ok(lit_str) = nested_meta.value()?.parse::<syn::LitStr>() {
                                    let value = lit_str.value();
                                    examples_array.push(quote! {
                                        ::serde_json::Value::String(#value.to_string())
                                    });
                                }
                                Ok(())
                            })?;
                        },
                        TypeCategory::Integer => {
                            // Parse array of integer literals
                            meta.parse_nested_meta(|nested_meta| {
                                if let Ok(lit_int) = nested_meta.value()?.parse::<syn::LitInt>() {
                                    examples_array.push(quote! {
                                        ::serde_json::Value::Number(::serde_json::Number::from(#lit_int))
                                    });
                                } else if let Ok(lit_str) = nested_meta.value()?.parse::<syn::LitStr>() {
                                    let value = lit_str.value();
                                    examples_array.push(quote! {
                                        match #value.parse::<i64>() {
                                            Ok(num) => ::serde_json::Value::Number(::serde_json::Number::from(num)),
                                            Err(_) => ::serde_json::Value::String(#value.to_string())
                                        }
                                    });
                                }
                                Ok(())
                            })?;
                        },
                        TypeCategory::Float => {
                            // Parse array of float literals
                            meta.parse_nested_meta(|nested_meta| {
                                if let Ok(lit_float) = nested_meta.value()?.parse::<syn::LitFloat>() {
                                    let value = lit_float.to_string();
                                    examples_array.push(quote! {
                                        match #value.parse::<f64>() {
                                            Ok(num) => ::serde_json::json!(num),
                                            Err(_) => ::serde_json::Value::String(#value.to_string())
                                        }
                                    });
                                } else if let Ok(lit_str) = nested_meta.value()?.parse::<syn::LitStr>() {
                                    let value = lit_str.value();
                                    examples_array.push(quote! {
                                        match #value.parse::<f64>() {
                                            Ok(num) => ::serde_json::json!(num),
                                            Err(_) => ::serde_json::Value::String(#value.to_string())
                                        }
                                    });
                                }
                                Ok(())
                            })?;
                        },
                        TypeCategory::Boolean => {
                            // Parse array of boolean literals
                            meta.parse_nested_meta(|nested_meta| {
                                if let Ok(lit_bool) = nested_meta.value()?.parse::<syn::LitBool>() {
                                    let value = lit_bool.value;
                                    examples_array.push(quote! {
                                        ::serde_json::Value::Bool(#value)
                                    });
                                } else if let Ok(lit_str) = nested_meta.value()?.parse::<syn::LitStr>() {
                                    let value = lit_str.value();
                                    examples_array.push(quote! {
                                        match #value.parse::<bool>() {
                                            Ok(val) => ::serde_json::Value::Bool(val),
                                            Err(_) => ::serde_json::Value::String(#value.to_string())
                                        }
                                    });
                                }
                                Ok(())
                            })?;
                        },
                        TypeCategory::Array | TypeCategory::Object => {
                            // For arrays and objects, accept a JSON string
                            let value = meta.value()?;
                            let content: syn::LitStr = value.parse()?;
                            let json_str = content.value();

                            examples_array.push(quote! {
                                match ::serde_json::from_str::<::serde_json::Value>(#json_str) {
                                    Ok(val) => val,
                                    Err(_) => ::serde_json::Value::String(#json_str.to_string())
                                }
                            });
                        }
                    }
                }
                Ok(())
            });
        }
    }

    FieldAttributes {
        description,
        example_value,
        examples_array,
        serde_rename,
    }
}

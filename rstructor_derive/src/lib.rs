use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Type, PathArguments, GenericArgument};

/// Derive macro for implementing LLMModel and SchemaType
#[proc_macro_derive(LLMModel, attributes(llm))]
pub fn derive_llm_model(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    
    let name = &input.ident;
    
    // Extract attributes based on the type
    match &input.data {
        Data::Struct(data_struct) => {
            // Process struct fields
            let mut property_setters = Vec::new();
            let mut required_setters = Vec::new();
            
            match &data_struct.fields {
                Fields::Named(fields) => {
                    for field in &fields.named {
                        let field_name = field.ident.as_ref().unwrap().to_string();
                        let is_optional = is_option_type(&field.ty);
                        
                        // Default properties
                        let mut description = None;
                        let mut example_value = None;
                        let mut examples_array = Vec::new();
                        
                        // Get the base type (unwrapping Option if present)
                        let base_type = if is_optional {
                            get_option_inner_type(&field.ty)
                        } else {
                            &field.ty
                        };
                        
                        // Extract attributes
                        for attr in &field.attrs {
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
                                            if let Some(array_tokens) = parse_array_literal(&value) {
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
                                        
                                        if let Some(array_tokens) = parse_array_literal(&value) {
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
                        if let Some(desc) = description {
                            let desc_prop = quote! {
                                props.insert("description".to_string(), ::serde_json::Value::String(#desc.to_string()));
                            };
                            property_setters.push(desc_prop);
                        }
                        
                        // Add single example if available
                        if let Some(ex_val) = &example_value {
                            let ex_prop = quote! {
                                let example_value = #ex_val;
                                props.insert("example".to_string(), example_value);
                            };
                            property_setters.push(ex_prop);
                        }
                        
                        // Add multiple examples if available
                        if !examples_array.is_empty() {
                            let examples_tokens = examples_array.iter().collect::<Vec<_>>();
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
            
            // Generate implementation
            let schema_impl = quote! {
                impl ::rstructor::schema::SchemaType for #name {
                    fn schema() -> ::rstructor::schema::Schema {
                        // Create base schema object
                        let mut schema_obj = ::serde_json::json!({
                            "type": "object",
                            "title": stringify!(#name),
                            "properties": {}
                        });
                        
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
            };
            
            schema_impl.into()
        }
        Data::Enum(data_enum) => {
            // Check if it's a simple enum (no data)
            let all_simple = data_enum.variants.iter().all(|v| v.fields.is_empty());
            
            if all_simple {
                // Generate implementation for simple enum
                let variant_values: Vec<_> = data_enum.variants.iter()
                    .map(|v| v.ident.to_string())
                    .collect();
                
                let schema_impl = quote! {
                    impl ::rstructor::schema::SchemaType for #name {
                        fn schema() -> ::rstructor::schema::Schema {
                            // Create array of enum values
                            let enum_values = vec![
                                #(::serde_json::Value::String(#variant_values.to_string())),*
                            ];
                            
                            let schema_obj = ::serde_json::json!({
                                "type": "string",
                                "enum": enum_values,
                                "title": stringify!(#name)
                            });
                            
                            ::rstructor::schema::Schema::new(schema_obj)
                        }
                        
                        fn schema_name() -> Option<String> {
                            Some(stringify!(#name).to_string())
                        }
                    }
                };
                
                schema_impl.into()
            } else {
                panic!("Enums with associated data are not supported yet by LLMModel derive");
            }
        }
        _ => panic!("LLMModel can only be derived for structs and enums"),
    }
}

// Enum to categorize Rust types
enum TypeCategory {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
}

// Determine if a type is an Option<T>
fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            return segment.ident == "Option";
        }
    }
    false
}

// Get the inner type of an Option<T>
fn get_option_inner_type(ty: &Type) -> &Type {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            if segment.ident == "Option" {
                if let PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(GenericArgument::Type(inner_ty)) = args.args.first() {
                        return inner_ty;
                    }
                }
            }
        }
    }
    ty
}

// Get type category from Rust type
fn get_type_category(ty: &Type) -> TypeCategory {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            let type_name = segment.ident.to_string();
            match type_name.as_str() {
                "String" | "str" | "char" => return TypeCategory::String,
                "bool" => return TypeCategory::Boolean,
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" |
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => return TypeCategory::Integer,
                "f32" | "f64" => return TypeCategory::Float,
                "Vec" | "Array" | "HashSet" | "BTreeSet" => return TypeCategory::Array,
                "HashMap" | "BTreeMap" => return TypeCategory::Object,
                _ => return TypeCategory::Object, // Default to object for custom types
            }
        }
    }
    TypeCategory::Object // Default
}

// Parse an array literal from an attribute value
fn parse_array_literal(value: &syn::parse::ParseBuffer) -> Option<Vec<proc_macro2::TokenStream>> {
    use syn::{bracketed, Expr, ExprArray, Token, parse::Parse};
    use quote::ToTokens;
    
    struct ArrayAttr {
        expr_array: ExprArray
    }
    
    impl Parse for ArrayAttr {
        fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
            let content;
            bracketed!(content in input);
            let mut elements = Vec::new();
            
            // Parse comma-separated expressions inside brackets
            while !content.is_empty() {
                let elem: Expr = content.parse()?;
                elements.push(elem);
                
                if content.is_empty() {
                    break;
                }
                
                content.parse::<Token![,]>()?;
            }
            
            Ok(ArrayAttr {
                expr_array: ExprArray {
                    attrs: Vec::new(),
                    bracket_token: syn::token::Bracket::default(),
                    elems: elements.into_iter().collect(),
                }
            })
        }
    }
    
    // Try to parse as a bracketed array
    if let Ok(array_attr) = value.parse::<ArrayAttr>() {
        // Process each element of the array
        let mut tokens = Vec::new();
        
        for elem in &array_attr.expr_array.elems {
            match elem {
                Expr::Lit(lit) => {
                    match &lit.lit {
                        syn::Lit::Str(lit_str) => {
                            let s = lit_str.value();
                            tokens.push(quote! {
                                ::serde_json::Value::String(#s.to_string())
                            });
                        },
                        syn::Lit::Int(lit_int) => {
                            tokens.push(quote! {
                                ::serde_json::Value::Number(::serde_json::Number::from(#lit_int))
                            });
                        },
                        syn::Lit::Float(lit_float) => {
                            let s = lit_float.to_string();
                            tokens.push(quote! {
                                ::serde_json::json!(#s.parse::<f64>().unwrap())
                            });
                        },
                        syn::Lit::Bool(lit_bool) => {
                            let b = lit_bool.value;
                            tokens.push(quote! {
                                ::serde_json::Value::Bool(#b)
                            });
                        },
                        _ => {
                            // For other literals, convert to string
                            let elem_tokens = elem.to_token_stream();
                            tokens.push(quote! {
                                ::serde_json::Value::String(#elem_tokens.to_string())
                            });
                        }
                    }
                },
                _ => {
                    // For non-literals, convert to string
                    let elem_tokens = elem.to_token_stream();
                    tokens.push(quote! {
                        ::serde_json::Value::String(format!("{}", #elem_tokens))
                    });
                }
            }
        }
        
        Some(tokens)
    } else {
        None
    }
}

// Get JSON Schema type from Rust type
fn get_schema_type_from_rust_type(ty: &Type) -> &'static str {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            let type_name = segment.ident.to_string();
            match type_name.as_str() {
                "String" | "str" | "char" => return "string",
                "bool" => return "boolean",
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" |
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => return "integer",
                "f32" | "f64" => return "number",
                "Vec" | "Array" | "HashSet" | "BTreeSet" => return "array",
                "HashMap" | "BTreeMap" => return "object",
                "Option" => {
                    // For Option<T>, we need to look at the inner type
                    if let PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let Some(GenericArgument::Type(inner_ty)) = args.args.first() {
                            return get_schema_type_from_rust_type(inner_ty);
                        }
                    }
                    return "null";
                }
                _ => return "object", // Default to object for custom types
            }
        }
    }
    "object" // Default
}
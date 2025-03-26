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
                        let mut example = None;
                        let mut is_attr_optional = false;
                        
                        // Extract attributes
                        for attr in &field.attrs {
                            if attr.path().is_ident("llm") {
                                // Parse attribute arguments
                                let _result = attr.parse_nested_meta(|meta| {
                                    if meta.path.is_ident("description") {
                                        let value = meta.value()?;
                                        let content: syn::LitStr = value.parse()?;
                                        description = Some(content.value());
                                    } else if meta.path.is_ident("examples") {
                                        let value = meta.value()?;
                                        let content: syn::LitStr = value.parse()?;
                                        example = Some(content.value());
                                    } else if meta.path.is_ident("optional") {
                                        is_attr_optional = true;
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
                        
                        // Add examples if available
                        if let Some(ex) = example {
                            let ex_prop = quote! {
                                // Try to parse as JSON, or use as string
                                let example_value = match ::serde_json::from_str::<::serde_json::Value>(#ex) {
                                    Ok(val) => val,
                                    Err(_) => ::serde_json::Value::String(#ex.to_string()),
                                };
                                props.insert("examples".to_string(), example_value);
                            };
                            property_setters.push(ex_prop);
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
                        
                        // Add to required fields if not optional
                        if !is_optional && !is_attr_optional {
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
                        
                        // Add properties
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

// Determine if a type is an Option<T>
fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            return segment.ident == "Option";
        }
    }
    false
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
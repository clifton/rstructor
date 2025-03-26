use proc_macro2::TokenStream;
use quote::quote;
use syn::{DataStruct, Fields, Ident};

use crate::type_utils::{get_schema_type_from_rust_type, is_option_type};
use crate::parsers::field_parser::parse_field_attributes;

/// Generate the schema implementation for a struct
pub fn generate_struct_schema(name: &Ident, data_struct: &DataStruct) -> TokenStream {
    let mut property_setters = Vec::new();
    let mut required_setters = Vec::new();
    
    match &data_struct.fields {
        Fields::Named(fields) => {
            for field in &fields.named {
                let field_name = field.ident.as_ref().unwrap().to_string();
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
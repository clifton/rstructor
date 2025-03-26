use proc_macro2::TokenStream;
use quote::quote;
use syn::{DataEnum, Ident};

use crate::container_attrs::ContainerAttributes;

/// Generate the schema implementation for an enum
pub fn generate_enum_schema(name: &Ident, data_enum: &DataEnum, container_attrs: &ContainerAttributes) -> TokenStream {
    // Check if it's a simple enum (no data)
    let all_simple = data_enum.variants.iter().all(|v| v.fields.is_empty());
    
    if all_simple {
        // Generate implementation for simple enum
        let variant_values: Vec<_> = data_enum.variants.iter()
            .map(|v| v.ident.to_string())
            .collect();
        
        // Handle container description
        let description_setter = if let Some(desc) = &container_attrs.description {
            quote! {
                schema_obj["description"] = ::serde_json::Value::String(#desc.to_string());
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
                    
                    // Add description if available
                    #description_setter
                    
                    ::rstructor::schema::Schema::new(schema_obj)
                }
                
                fn schema_name() -> Option<String> {
                    Some(stringify!(#name).to_string())
                }
            }
        }
    } else {
        panic!("Enums with associated data are not supported yet by LLMModel derive");
    }
}
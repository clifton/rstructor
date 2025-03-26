/*! 
 Procedural macros for the rstructor library.
 
 This crate provides the derive macro for implementing LLMModel and SchemaType
 traits from the rstructor library. It automatically generates JSON Schema 
 representations of Rust types.
*/
mod type_utils;
mod parsers;
mod generators;
mod container_attrs;

use proc_macro::TokenStream;
use syn::{parse_macro_input, Data, DeriveInput};
use container_attrs::ContainerAttributes;

/// Derive macro for implementing LLMModel and SchemaType
///
/// This macro automatically implements the SchemaType trait for a struct or enum,
/// generating a JSON Schema representation based on the Rust type.
///
/// # Examples
///
/// ## Field-level attributes
///
/// ```
/// use rstructor::LLMModel;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(LLMModel, Serialize, Deserialize, Debug)]
/// struct Person {
///     #[llm(description = "Full name of the person")]
///     name: String,
///     
///     #[llm(description = "Age of the person in years", example = 30)]
///     age: u32,
///     
///     #[llm(description = "List of skills", example = ["Programming", "Writing", "Design"])]
///     skills: Vec<String>,
/// }
/// ```
///
/// ## Container-level attributes
///
/// You can also add a description to the struct or enum itself:
///
/// ```
/// use rstructor::LLMModel;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(LLMModel, Serialize, Deserialize, Debug)]
/// #[llm(description = "Represents a person with their basic information")]
/// struct Person {
///     #[llm(description = "Full name of the person")]
///     name: String,
///     
///     #[llm(description = "Age of the person in years")]
///     age: u32,
/// }
///
/// #[derive(LLMModel, Serialize, Deserialize, Debug)]
/// #[llm(description = "Represents a person's role in an organization")]
/// enum Role {
///     Employee,
///     Manager,
///     Director,
///     Executive,
/// }
/// ```
#[proc_macro_derive(LLMModel, attributes(llm))]
pub fn derive_llm_model(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    
    // First, extract container-level attributes
    let container_attrs = extract_container_attributes(&input.attrs);
    
    // Then generate schema based on the type
    match &input.data {
        Data::Struct(data_struct) => {
            generators::generate_struct_schema(name, data_struct, &container_attrs).into()
        }
        Data::Enum(data_enum) => {
            generators::generate_enum_schema(name, data_enum, &container_attrs).into()
        }
        _ => panic!("LLMModel can only be derived for structs and enums"),
    }
}

/// Extract container level attributes like descriptions for structs and enums
fn extract_container_attributes(attrs: &[syn::Attribute]) -> ContainerAttributes {
    let mut description = None;
    
    for attr in attrs {
        if attr.path().is_ident("llm") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("description") {
                    let value = meta.value()?;
                    let content: syn::LitStr = value.parse()?;
                    description = Some(content.value());
                }
                Ok(())
            });
        }
    }
    
    ContainerAttributes::new(description)
}
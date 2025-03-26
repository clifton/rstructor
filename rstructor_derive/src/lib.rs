/*! 
 Procedural macros for the rstructor library.
 
 This crate provides the derive macro for implementing LLMModel and SchemaType
 traits from the rstructor library. It automatically generates JSON Schema 
 representations of Rust types.
*/
mod type_utils;
mod parsers;
mod generators;

use proc_macro::TokenStream;
use syn::{parse_macro_input, Data, DeriveInput};

/// Derive macro for implementing LLMModel and SchemaType
///
/// This macro automatically implements the SchemaType trait for a struct or enum,
/// generating a JSON Schema representation based on the Rust type.
///
/// # Example
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
#[proc_macro_derive(LLMModel, attributes(llm))]
pub fn derive_llm_model(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    
    // Extract attributes based on the type
    match &input.data {
        Data::Struct(data_struct) => {
            generators::generate_struct_schema(name, data_struct).into()
        }
        Data::Enum(data_enum) => {
            generators::generate_enum_schema(name, data_enum).into()
        }
        _ => panic!("LLMModel can only be derived for structs and enums"),
    }
}
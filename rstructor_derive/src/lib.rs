/*!
 Procedural macros for the rstructor library.

 This crate provides the derive macro for implementing LLMModel and SchemaType
 traits from the rstructor library. It automatically generates JSON Schema
 representations of Rust types.
*/
mod container_attrs;
mod generators;
mod parsers;
mod type_utils;

use container_attrs::ContainerAttributes;
use proc_macro::TokenStream;
use syn::{Data, DeriveInput, parse_macro_input};

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
/// You can add additional information to the struct or enum itself:
///
/// ```
/// use rstructor::LLMModel;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(LLMModel, Serialize, Deserialize, Debug)]
/// #[llm(description = "Represents a person with their basic information",
///       title = "PersonDetail",
///       examples = [
///         ::serde_json::json!({"name": "John Doe", "age": 30}),
///         ::serde_json::json!({"name": "Jane Smith", "age": 25})
///       ])]
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
/// #[serde(rename_all = "camelCase")]
/// struct Employee {
///     first_name: String,
///     last_name: String,
///     employee_id: u32,
/// }
///
/// #[derive(LLMModel, Serialize, Deserialize, Debug)]
/// #[llm(description = "Represents a person's role in an organization",
///       examples = ["Manager", "Director"])]
/// enum Role {
///     Employee,
///     Manager,
///     Director,
///     Executive,
/// }
/// ```
///
/// ### Container Attributes
///
/// - `description`: A description of the struct or enum
/// - `title`: A custom title for the JSON Schema (defaults to the type name)
/// - `examples`: Example instances of the struct or enum
///
/// ### Serde Integration
///
/// - Respects `#[serde(rename_all = "...")]` for transforming property names
///   - Supported values: "lowercase", "UPPERCASE", "camelCase", "PascalCase", "snake_case"
///   - Example: With `#[serde(rename_all = "camelCase")]`, a field `user_id` becomes `userId` in the schema
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

use quote::ToTokens;

/// Extract container level attributes like descriptions for structs and enums
fn extract_container_attributes(attrs: &[syn::Attribute]) -> ContainerAttributes {
    let mut description = None;
    let mut title = None;
    let mut examples = Vec::new();
    let mut serde_rename_all = None;

    // First, check for llm-specific attributes
    for attr in attrs {
        if attr.path().is_ident("llm") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("description") {
                    let value = meta.value()?;
                    let content: syn::LitStr = value.parse()?;
                    description = Some(content.value());
                } else if meta.path.is_ident("title") {
                    let value = meta.value()?;
                    let content: syn::LitStr = value.parse()?;
                    title = Some(content.value());
                } else if meta.path.is_ident("examples") {
                    // Handle array syntax like examples = ["one", "two"]
                    let value = meta.value()?;

                    // Try to parse as array expression
                    if let Ok(expr) = value.parse::<syn::Expr>() {
                        if let syn::Expr::Array(array) = expr {
                            // For each element in the array, convert to TokenStream
                            for elem in array.elems.iter() {
                                // For string literals, wrap them in serde_json::Value::String constructors
                                if let syn::Expr::Lit(lit_expr) = elem {
                                    if let syn::Lit::Str(lit_str) = &lit_expr.lit {
                                        let str_val = lit_str.value();
                                        let json_str = quote::quote! {
                                            ::serde_json::Value::String(#str_val.to_string())
                                        };
                                        examples.push(json_str);
                                    } else {
                                        // For other literals, pass them through
                                        examples.push(elem.to_token_stream());
                                    }
                                } else {
                                    // For non-literals (like objects), pass them through
                                    examples.push(elem.to_token_stream());
                                }
                            }
                        }
                    }
                }
                Ok(())
            });
        }
    }

    // Then, check for serde attributes
    for attr in attrs {
        if attr.path().is_ident("serde") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("rename_all") {
                    let value = meta.value()?;
                    let content: syn::LitStr = value.parse()?;
                    serde_rename_all = Some(content.value());
                }
                Ok(())
            });
        }
    }

    ContainerAttributes::new(description, title, examples, serde_rename_all)
}

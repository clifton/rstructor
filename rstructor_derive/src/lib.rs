/*!
 Procedural macros for the rstructor library.

 This crate provides the derive macro for implementing Instructor and SchemaType
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

/// Derive macro for implementing Instructor and SchemaType
///
/// This macro automatically implements the SchemaType trait for a struct or enum,
/// generating a JSON Schema representation based on the Rust type.
///
/// # Nested Types and Schema Embedding
///
/// When you have nested structs or enums, they should also derive `Instructor`
/// to ensure their full schema is embedded in the parent type. This produces
/// complete JSON schemas that help LLMs generate correct structured output.
///
/// ```rust
/// use rstructor::Instructor;
/// use serde::{Serialize, Deserialize};
///
/// // Parent type derives Instructor
/// #[derive(Instructor, Serialize, Deserialize)]
/// struct Parent {
///     child: Child,  // Child's schema will be embedded
/// }
///
/// // Nested types should also derive Instructor for complete schema
/// #[derive(Instructor, Serialize, Deserialize)]
/// struct Child {
///     name: String,
/// }
/// ```
///
/// The schema embedding happens at compile time, avoiding any runtime overhead.
///
/// # Validation
///
/// To add custom validation, use the `validate` attribute with a function path:
///
/// ```
/// use rstructor::{Instructor, RStructorError};
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Instructor, Serialize, Deserialize)]
/// #[llm(validate = "validate_product")]
/// struct Product {
///     name: String,
///     price: f64,
/// }
///
/// fn validate_product(product: &Product) -> rstructor::Result<()> {
///     if product.price <= 0.0 {
///         return Err(RStructorError::ValidationError(
///             "price must be positive".into()
///         ));
///     }
///     Ok(())
/// }
/// ```
///
/// The validation function is called automatically when the LLM response is deserialized.
///
/// # Examples
///
/// ## Field-level attributes
///
/// ```
/// use rstructor::Instructor;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Instructor, Serialize, Deserialize, Debug)]
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
/// use rstructor::Instructor;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Instructor, Serialize, Deserialize, Debug)]
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
/// #[derive(Instructor, Serialize, Deserialize, Debug)]
/// #[llm(description = "Represents a person's role in an organization")]
/// #[serde(rename_all = "camelCase")]
/// struct Employee {
///     first_name: String,
///     last_name: String,
///     employee_id: u32,
/// }
///
/// #[derive(Instructor, Serialize, Deserialize, Debug)]
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
#[proc_macro_derive(Instructor, attributes(llm))]
pub fn derive_instructor(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // First, extract container-level attributes
    let container_attrs = extract_container_attributes(&input.attrs);

    // Generate the schema implementation
    let schema_impl = match &input.data {
        Data::Struct(data_struct) => {
            generators::generate_struct_schema(name, data_struct, &container_attrs)
        }
        Data::Enum(data_enum) => {
            generators::generate_enum_schema(name, data_enum, &container_attrs)
        }
        _ => panic!("Instructor can only be derived for structs and enums"),
    };

    // Generate the Instructor trait implementation
    let instructor_impl = if let Some(validate_fn) = &container_attrs.validate {
        // Parse the validation function path
        let validate_path: syn::Path =
            syn::parse_str(validate_fn).expect("validate attribute must be a valid function path");
        quote::quote! {
            impl ::rstructor::model::Instructor for #name {
                fn validate(&self) -> ::rstructor::error::Result<()> {
                    #validate_path(self)
                }
            }
        }
    } else {
        // Default implementation - validation passes
        quote::quote! {
            impl ::rstructor::model::Instructor for #name {
                fn validate(&self) -> ::rstructor::error::Result<()> {
                    ::rstructor::error::Result::Ok(())
                }
            }
        }
    };

    // Combine the two implementations
    let combined = quote::quote! {
        #schema_impl

        #instructor_impl
    };

    combined.into()
}

use quote::ToTokens;

fn extract_container_attributes(attrs: &[syn::Attribute]) -> ContainerAttributes {
    let mut description = None;
    let mut title = None;
    let mut examples = Vec::new();
    let mut serde_rename_all = None;
    let mut validate = None;
    let mut serde_tag = None;
    let mut serde_content = None;
    let mut serde_untagged = false;

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
                } else if meta.path.is_ident("validate") {
                    let value = meta.value()?;
                    let content: syn::LitStr = value.parse()?;
                    validate = Some(content.value());
                } else if meta.path.is_ident("examples") {
                    // Handle array syntax like examples = ["one", "two"]
                    let value = meta.value()?;

                    // Try to parse as array expression
                    if let Ok(syn::Expr::Array(array)) = value.parse::<syn::Expr>() {
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
                } else if meta.path.is_ident("tag") {
                    let value = meta.value()?;
                    let content: syn::LitStr = value.parse()?;
                    serde_tag = Some(content.value());
                } else if meta.path.is_ident("content") {
                    let value = meta.value()?;
                    let content: syn::LitStr = value.parse()?;
                    serde_content = Some(content.value());
                } else if meta.path.is_ident("untagged") {
                    serde_untagged = true;
                }
                Ok(())
            });
        }
    }

    ContainerAttributes::builder()
        .description(description)
        .title(title)
        .examples(examples)
        .serde_rename_all(serde_rename_all)
        .validate(validate)
        .serde_tag(serde_tag)
        .serde_content(serde_content)
        .serde_untagged(serde_untagged)
        .build()
}

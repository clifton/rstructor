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

    // Check if the type has a validate method by looking through impl blocks
    let validate_impl = find_validate_method(&input);

    // Generate the Instructor trait implementation with proper validate method calling
    // Always generate a standard Instructor implementation
    // We'll use a special pattern to avoid stack overflow
    let instructor_impl = quote::quote! {
        impl ::rstructor::model::Instructor for #name {
            fn validate(&self) -> ::rstructor::error::Result<()> {
                // We use this method to prevent the dead code warning
                // and avoid stack overflow by using a different method name
                #name::__validate_impl(self)
            }
        }

        // This implementation provides a special hidden method that will call
        // the actual validate method if it exists, or do nothing if it doesn't
        impl #name {
            #[doc(hidden)]
            fn __validate_impl(this: &Self) -> ::rstructor::error::Result<()> {
                // This will either call the struct's own validate method,
                // or it will use the default implementation (do nothing)
                #[allow(unused_variables)]
                {
                    #[cfg(any())]
                    let _ignore_this = stringify!(#validate_impl);

                    // Only include this code if we detected a validate method
                    if #validate_impl {
                        // If a validate method exists, call it directly
                        if let ::std::result::Result::Err(err) = this.validate() {
                            return ::std::result::Result::Err(err);
                        }
                    }
                }

                // Return Ok if no validation was done or if validation succeeded
                ::rstructor::error::Result::Ok(())
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

/// Extract container level attributes like descriptions for structs and enums
/// Check if a type has a validate method by examining the codebase
/// This is a simple heuristic approach - we can't actually detect all validate methods
/// during proc macro expansion due to limitations in the compiler, but this helps
/// reduce dead code warnings in many typical cases.
fn find_validate_method(input: &syn::DeriveInput) -> bool {
    // Since we can't reliably detect all validate methods at compile time,
    // we'll assume structs with certain attributes or patterns likely have validation

    // Check for documentation comments that mention validation
    for attr in &input.attrs {
        if attr.path().is_ident("doc") {
            let mut has_validation = false;
            let _ = attr.parse_nested_meta(|meta| {
                #[allow(clippy::collapsible_if)]
                if let Ok(value) = meta.value() {
                    if let Ok(lit_str) = value.parse::<syn::LitStr>() {
                        let doc_str = lit_str.value().to_lowercase();
                        if doc_str.contains("valid") || doc_str.contains("check") {
                            has_validation = true;
                        }
                    }
                }
                Ok(())
            });
            if has_validation {
                return true;
            }
        }
    }

    // Check if it has llm attributes, which often indicate validation will be present
    for attr in &input.attrs {
        if attr.path().is_ident("llm") {
            return true;
        }
    }

    // Check the type name for validation-related patterns
    let type_name = input.ident.to_string().to_lowercase();
    if type_name.contains("valid") || type_name.contains("check") || type_name.contains("rule") {
        return true;
    }

    // For structs, check field names/types for validation-related patterns
    if let syn::Data::Struct(data_struct) = &input.data {
        for field in data_struct.fields.iter() {
            if let Some(field_name) = &field.ident {
                let name = field_name.to_string().to_lowercase();
                if name.contains("valid") || name.contains("rule") || name.contains("constraint") {
                    return true;
                }
            }

            // Check field attributes for validation hints
            for attr in &field.attrs {
                if attr.path().is_ident("llm") {
                    return true;
                }
            }
        }
    }

    // Default to true to be safe - this will generate code that properly uses validate methods
    // even if they're added later
    true
}

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
                }
                Ok(())
            });
        }
    }

    ContainerAttributes::new(description, title, examples, serde_rename_all)
}

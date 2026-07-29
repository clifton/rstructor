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
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

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
/// - `validate`: A quoted Rust path to a custom validation function
///
/// ### Field and Variant Attributes
///
/// Fields accept `description`, `example`, and `examples`. Enum variants accept
/// `description`. Field optionality is inferred from `Option<T>`; there is no
/// `optional` attribute.
///
/// The `llm` namespace is checked strictly. Unknown attributes, malformed
/// values, invalid validation paths, and unsupported tuple/unit structs produce
/// errors at the relevant source span instead of being ignored.
///
/// ### Serde Integration
///
/// Supported Serde name and skip metadata is interpreted from the
/// deserialization side of the wire contract:
///
/// - Respects `rename`, `rename_all`, and `rename_all_fields`
/// - Uses `deserialize = "..."` when names differ by direction
/// - Omits fields and variants marked `skip` or `skip_deserializing`
/// - Keeps `skip_serializing` fields because they remain valid inputs
///
/// Supported case transformations include "lowercase", "UPPERCASE",
/// "camelCase", "PascalCase", and "snake_case". For example, with
/// `#[serde(rename_all = "camelCase")]`, `user_id` becomes `userId`.
#[proc_macro_derive(Instructor, attributes(llm))]
pub fn derive_instructor(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    validate_derive_input(&input)?;

    let name = &input.ident;
    let container_attrs = extract_container_attributes(&input.attrs)?;

    // Generate the schema implementation
    let schema_impl = match &input.data {
        Data::Struct(data_struct) => generators::generate_struct_schema(
            name,
            data_struct,
            &container_attrs,
            &input.generics,
        )?,
        Data::Enum(data_enum) => {
            generators::generate_enum_schema(name, data_enum, &container_attrs, &input.generics)?
        }
        Data::Union(_) => {
            return Err(syn::Error::new_spanned(
                &input,
                "Instructor can only be derived for structs with named fields and enums",
            ));
        }
    };

    // Generate the Instructor trait implementation.
    //
    // `validate` first recurses into every field whose type implements
    // `Instructor` (nested structs/enums, and their contents through `Option`,
    // `Vec`, `Box`, and string-keyed maps), then runs this type's own
    // `#[llm(validate = "...")]` function, if any.
    let field_validation = generate_field_validation(&input.data);
    let container_validate = if let Some(validate_path) = &container_attrs.validate {
        quote::quote! { #validate_path(self)?; }
    } else {
        quote::quote! {}
    };
    // `Instructor` requires `SchemaType + Serialize + DeserializeOwned` as
    // supertraits, so for generic types every type parameter must be bound by
    // those traits for the impl to typecheck (mirroring serde's own derive,
    // which bounds type parameters by `Serialize`/`Deserialize`).
    let mut instructor_bounds = vec![
        syn::parse_quote!(::rstructor::schema::SchemaType),
        syn::parse_quote!(::serde::Serialize),
        syn::parse_quote!(::serde::de::DeserializeOwned),
    ];
    if input.generics.lifetimes().next().is_none() {
        instructor_bounds.push(syn::parse_quote!('static));
    }
    let instructor_generics = type_utils::generics_with_bounds(&input.generics, &instructor_bounds);
    let (impl_generics, ty_generics, where_clause) = instructor_generics.split_for_impl();
    let instructor_impl = quote::quote! {
        impl #impl_generics ::rstructor::model::Instructor for #name #ty_generics #where_clause {
            fn validate(&self) -> ::rstructor::error::Result<()> {
                #[allow(unused_imports)]
                use ::rstructor::model::__private::ProbeFallback as _;
                #field_validation
                #container_validate
                ::rstructor::error::Result::Ok(())
            }
        }
    };

    // Combine the two implementations
    let combined = quote::quote! {
        #schema_impl

        #instructor_impl
    };

    Ok(combined)
}

fn validate_derive_input(input: &DeriveInput) -> syn::Result<()> {
    let mut errors = None;

    if let Err(error) = extract_container_attributes(&input.attrs) {
        combine_error(&mut errors, error);
    }

    match &input.data {
        Data::Struct(data) => {
            if !matches!(data.fields, Fields::Named(_)) {
                let span = match &data.fields {
                    Fields::Unit => input.ident.span(),
                    fields => fields.span(),
                };
                combine_error(
                    &mut errors,
                    syn::Error::new(span, "Instructor requires a struct with named fields"),
                );
            }
            for field in &data.fields {
                if let Err(error) = parsers::field_parser::parse_field_attributes(field) {
                    combine_error(&mut errors, error);
                }
            }
        }
        Data::Enum(data) => {
            for variant in &data.variants {
                if let Err(error) = parsers::variant_parser::parse_variant_attributes(variant) {
                    combine_error(&mut errors, error);
                }
                for field in &variant.fields {
                    if let Err(error) = parsers::field_parser::parse_field_attributes(field) {
                        combine_error(&mut errors, error);
                    }
                }
            }
        }
        Data::Union(_) => combine_error(
            &mut errors,
            syn::Error::new_spanned(
                input,
                "Instructor can only be derived for structs with named fields and enums",
            ),
        ),
    }

    match errors {
        Some(errors) => Err(errors),
        None => Ok(()),
    }
}

fn combine_error(errors: &mut Option<syn::Error>, error: syn::Error) {
    match errors {
        Some(errors) => errors.combine(error),
        None => *errors = Some(error),
    }
}

/// Generate statements that recursively validate every field of a struct or the
/// active variant of an enum.
///
/// Each field is wrapped in `__private::Probe`, whose `rstructor_probe` resolves
/// (via autoref specialization) to the field's `Instructor::validate` when the
/// field type implements `Instructor`, and to a no-op otherwise — so primitive
/// fields cost nothing while nested `Instructor` values are validated.
fn generate_field_validation(data: &Data) -> proc_macro2::TokenStream {
    match data {
        Data::Struct(data_struct) => {
            match &data_struct.fields {
                Fields::Named(named) => {
                    let probes = named.named.iter().filter_map(|f| f.ident.as_ref()).map(|ident| {
                    quote::quote! {
                        ::rstructor::model::__private::Probe(&self.#ident).rstructor_probe()?;
                    }
                });
                    quote::quote! { #(#probes)* }
                }
                Fields::Unnamed(unnamed) => {
                    let probes = unnamed.unnamed.iter().enumerate().map(|(i, _)| {
                        let index = syn::Index::from(i);
                        quote::quote! {
                            ::rstructor::model::__private::Probe(&self.#index).rstructor_probe()?;
                        }
                    });
                    quote::quote! { #(#probes)* }
                }
                Fields::Unit => quote::quote! {},
            }
        }
        Data::Enum(data_enum) => {
            let arms = data_enum.variants.iter().map(|variant| {
                let vname = &variant.ident;
                match &variant.fields {
                    Fields::Named(named) => {
                        let binds: Vec<_> = named
                            .named
                            .iter()
                            .filter_map(|f| f.ident.clone())
                            .collect();
                        quote::quote! {
                            Self::#vname { #(#binds),* } => {
                                #( ::rstructor::model::__private::Probe(#binds).rstructor_probe()?; )*
                            }
                        }
                    }
                    Fields::Unnamed(unnamed) => {
                        let binds: Vec<_> = (0..unnamed.unnamed.len())
                            .map(|i| quote::format_ident!("field{}", i))
                            .collect();
                        quote::quote! {
                            Self::#vname( #(#binds),* ) => {
                                #( ::rstructor::model::__private::Probe(#binds).rstructor_probe()?; )*
                            }
                        }
                    }
                    Fields::Unit => quote::quote! { Self::#vname => {} },
                }
            });
            quote::quote! {
                match self {
                    #(#arms)*
                }
            }
        }
        _ => quote::quote! {},
    }
}

fn extract_container_attributes(attrs: &[syn::Attribute]) -> syn::Result<ContainerAttributes> {
    let mut description = None;
    let mut title = None;
    let mut examples = Vec::new();
    let mut validate = None;
    let mut errors = None;

    for attr in attrs.iter().filter(|attr| attr.path().is_ident("llm")) {
        if let Err(error) = attr.parse_nested_meta(|meta| {
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
                    validate = Some(content.parse::<syn::Path>().map_err(|error| {
                        syn::Error::new(
                            content.span(),
                            format!("invalid validation function path: {error}"),
                        )
                    })?);
                } else if meta.path.is_ident("examples") {
                    let expressions =
                        parsers::array_parser::parse_expression_list(&meta, "examples")?;
                    examples.extend(parsers::array_parser::expression_tokens(&expressions));
                } else {
                    return Err(meta.error(
                        "unsupported container `llm` attribute; expected one of: `description`, `title`, `examples`, `validate`",
                    ));
                }
                Ok(())
            })
        {
            combine_error(&mut errors, error);
        }
    }

    if let Some(errors) = errors {
        return Err(errors);
    }

    let serde = parsers::serde_parser::parse_serde_attributes(attrs);
    Ok(ContainerAttributes::builder()
        .description(description)
        .title(title)
        .examples(examples)
        .serde_rename_all(serde.rename_all)
        .serde_rename_all_fields(serde.rename_all_fields)
        .validate(validate)
        .serde_tag(serde.tag)
        .serde_content(serde.content)
        .serde_untagged(serde.untagged)
        .build())
}

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Expr, ExprLit, Field, Lit};

use crate::parsers::array_parser::{
    array_expression_tokens, expression_tokens, parse_expression_list,
};
use crate::parsers::serde_parser::parse_serde_attributes;
use crate::type_utils::{TypeCategory, get_option_inner_type, get_type_category, is_option_type};

/// Represents parsed field attributes.
pub struct FieldAttributes {
    pub description: Option<String>,
    pub example_value: Option<TokenStream>,
    pub examples_array: Vec<TokenStream>,
    /// Field rename from `#[serde(rename = "...")]`.
    pub serde_rename: Option<String>,
    /// Whether Serde excludes this field from deserialization.
    pub serde_skip_deserializing: bool,
}

/// Parse a single field's `llm` attributes and the schema-relevant subset of
/// Serde metadata.
pub fn parse_field_attributes(field: &Field) -> syn::Result<FieldAttributes> {
    let mut description = None;
    let mut example_value = None;
    let mut examples_array = Vec::new();

    let base_type = if is_option_type(&field.ty) {
        get_option_inner_type(&field.ty)
    } else {
        &field.ty
    };
    let category = get_type_category(base_type);

    for attr in field
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("llm"))
    {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("description") {
                let value = meta.value()?;
                let content: syn::LitStr = value.parse()?;
                description = Some(content.value());
            } else if meta.path.is_ident("example") {
                let value = meta.value()?;
                let expression: Expr = value.parse()?;
                example_value = Some(example_tokens(expression, category)?);
            } else if meta.path.is_ident("examples") {
                let expressions = parse_expression_list(&meta, "examples")?;
                examples_array = expression_tokens(&expressions);
            } else {
                return Err(meta.error(
                    "unsupported field `llm` attribute; expected one of: `description`, `example`, `examples`",
                ));
            }
            Ok(())
        })?;
    }

    let serde = parse_serde_attributes(&field.attrs);
    Ok(FieldAttributes {
        description,
        example_value,
        examples_array,
        serde_rename: serde.rename,
        serde_skip_deserializing: serde.skip_deserializing,
    })
}

fn example_tokens(expression: Expr, category: TypeCategory) -> syn::Result<TokenStream> {
    match category {
        TypeCategory::String => {
            let value = string_literal(expression, "a string literal")?;
            Ok(quote! {
                ::serde_json::Value::String(#value.to_string())
            })
        }
        TypeCategory::Integer => match expression {
            Expr::Lit(ExprLit {
                lit: Lit::Int(value),
                ..
            }) => Ok(quote! {
                ::serde_json::Value::Number(::serde_json::Number::from(#value))
            }),
            expression if is_negative_integer(&expression) => Ok(quote! {
                ::serde_json::Value::Number(::serde_json::Number::from(#expression))
            }),
            expression => Err(expected_example(expression, "an integer literal")),
        },
        TypeCategory::Float => match expression {
            Expr::Lit(ExprLit {
                lit: Lit::Float(value),
                ..
            }) => Ok(quote! {
                ::serde_json::json!(#value)
            }),
            Expr::Lit(ExprLit {
                lit: Lit::Int(value),
                ..
            }) => Ok(quote! {
                ::serde_json::json!(#value)
            }),
            expression if is_negative_number(&expression) => Ok(quote! {
                ::serde_json::json!(#expression)
            }),
            expression => Err(expected_example(expression, "a numeric literal")),
        },
        TypeCategory::Boolean => match expression {
            Expr::Lit(ExprLit {
                lit: Lit::Bool(value),
                ..
            }) => {
                let value = value.value;
                Ok(quote! {
                    ::serde_json::Value::Bool(#value)
                })
            }
            expression => Err(expected_example(expression, "a boolean literal")),
        },
        TypeCategory::Array => match expression {
            Expr::Array(array) => {
                let values = array_expression_tokens(&array);
                Ok(quote! {
                    ::serde_json::Value::Array(vec![#(#values),*])
                })
            }
            Expr::Lit(ExprLit {
                lit: Lit::Str(value),
                ..
            }) => {
                let value = value.value();
                if value.starts_with('[') && value.ends_with(']') {
                    let json = value.replace('\'', "\"");
                    Ok(quote! {
                        match ::serde_json::from_str(#json) {
                            Ok(value) => value,
                            Err(_) => ::serde_json::Value::String(#value.to_string()),
                        }
                    })
                } else {
                    Ok(quote! {
                        ::serde_json::Value::Array(vec![
                            ::serde_json::Value::String(#value.to_string())
                        ])
                    })
                }
            }
            expression => Err(expected_example(
                expression,
                "an array expression or string literal",
            )),
        },
        TypeCategory::Object => match expression {
            Expr::Macro(expression)
                if expression
                    .mac
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "json") =>
            {
                Ok(quote! { #expression })
            }
            expression => {
                let value = string_literal(expression, "a JSON string literal or `json!` macro")?;
                Ok(quote! {
                    match ::serde_json::from_str(#value) {
                        Ok(value) => value,
                        Err(_) => ::serde_json::Value::String(#value.to_string()),
                    }
                })
            }
        },
    }
}

fn is_negative_integer(expression: &Expr) -> bool {
    let Expr::Unary(unary) = expression else {
        return false;
    };
    if !matches!(unary.op, syn::UnOp::Neg(_)) {
        return false;
    }
    matches!(
        unary.expr.as_ref(),
        Expr::Lit(ExprLit {
            lit: Lit::Int(_),
            ..
        })
    )
}

fn is_negative_number(expression: &Expr) -> bool {
    let Expr::Unary(unary) = expression else {
        return false;
    };
    if !matches!(unary.op, syn::UnOp::Neg(_)) {
        return false;
    }
    matches!(
        unary.expr.as_ref(),
        Expr::Lit(ExprLit {
            lit: Lit::Int(_) | Lit::Float(_),
            ..
        })
    )
}

fn string_literal(expression: Expr, expected: &str) -> syn::Result<String> {
    match expression {
        Expr::Lit(ExprLit {
            lit: Lit::Str(value),
            ..
        }) => Ok(value.value()),
        expression => Err(expected_example(expression, expected)),
    }
}

fn expected_example(expression: Expr, expected: &str) -> syn::Error {
    syn::Error::new_spanned(
        expression,
        format!("invalid `example` value: expected {expected}"),
    )
}

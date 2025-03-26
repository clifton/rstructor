use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{Expr, ExprArray, Lit, Token, bracketed, parse::Parse};

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    use syn::parse_str;

    // Test the ArrayAttr structure directly
    #[test]
    fn test_array_attr_parse() {
        // Create a string array
        let input = "[\"apple\", \"banana\", \"cherry\"]";
        let array_expr: syn::ExprArray = parse_str(input).unwrap();

        // Create an ArrayAttr
        let array_attr = ArrayAttr {
            expr_array: array_expr,
        };

        // Check the array elements
        assert_eq!(array_attr.expr_array.elems.len(), 3);
    }

    #[test]
    fn test_array_attr_types() {
        // Test with different types
        let string_array = "[\"apple\", \"banana\"]";
        let int_array = "[1, 2, 3]";
        let bool_array = "[true, false]";
        let mixed_array = "[\"string\", 42, true]";

        // Parse each type
        let string_expr: syn::ExprArray = parse_str(string_array).unwrap();
        let int_expr: syn::ExprArray = parse_str(int_array).unwrap();
        let bool_expr: syn::ExprArray = parse_str(bool_array).unwrap();
        let mixed_expr: syn::ExprArray = parse_str(mixed_array).unwrap();

        // Check lengths
        assert_eq!(string_expr.elems.len(), 2);
        assert_eq!(int_expr.elems.len(), 3);
        assert_eq!(bool_expr.elems.len(), 2);
        assert_eq!(mixed_expr.elems.len(), 3);
    }

    #[test]
    fn test_tokenize_array_elements() {
        // Test tokenizing array elements for strings
        let string_array = "[\"apple\", \"banana\"]";
        let string_expr: syn::ExprArray = parse_str(string_array).unwrap();

        // Check first element using quote
        let first_elem = &string_expr.elems[0];
        let tokens = quote! { #first_elem };
        let token_string = tokens.to_string();

        // The tokenized string should include quotes
        assert!(token_string.contains("apple"));
    }
}

/// Utility struct to parse array literal expressions
/// Handles array literals like [1, 2, 3] or ["a", "b", "c"]
pub struct ArrayAttr {
    pub expr_array: ExprArray,
}

impl Parse for ArrayAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        bracketed!(content in input);
        let mut elements = Vec::new();

        // Parse comma-separated expressions inside brackets
        while !content.is_empty() {
            let elem: Expr = content.parse()?;
            elements.push(elem);

            if content.is_empty() {
                break;
            }

            content.parse::<Token![,]>()?;
        }

        Ok(ArrayAttr {
            expr_array: ExprArray {
                attrs: Vec::new(),
                bracket_token: syn::token::Bracket::default(),
                elems: elements.into_iter().collect(),
            },
        })
    }
}

/// Parse an array literal from an attribute value
/// Returns a vector of TokenStreams representing JSON values for each array element
pub fn parse_array_literal(value: &syn::parse::ParseBuffer) -> Option<Vec<TokenStream>> {
    // Try to parse as a bracketed array
    if let Ok(array_attr) = value.parse::<ArrayAttr>() {
        // Process each element of the array
        let mut tokens = Vec::new();

        for elem in &array_attr.expr_array.elems {
            match elem {
                Expr::Lit(lit) => {
                    match &lit.lit {
                        Lit::Str(lit_str) => {
                            let s = lit_str.value();
                            tokens.push(quote! {
                                ::serde_json::Value::String(#s.to_string())
                            });
                        }
                        Lit::Int(lit_int) => {
                            tokens.push(quote! {
                                ::serde_json::Value::Number(::serde_json::Number::from(#lit_int))
                            });
                        }
                        Lit::Float(lit_float) => {
                            let s = lit_float.to_string();
                            tokens.push(quote! {
                                ::serde_json::json!(#s.parse::<f64>().unwrap())
                            });
                        }
                        Lit::Bool(lit_bool) => {
                            let b = lit_bool.value;
                            tokens.push(quote! {
                                ::serde_json::Value::Bool(#b)
                            });
                        }
                        _ => {
                            // For other literals, convert to string
                            let elem_tokens = elem.to_token_stream();
                            tokens.push(quote! {
                                ::serde_json::Value::String(#elem_tokens.to_string())
                            });
                        }
                    }
                }
                _ => {
                    // For non-literals, convert to string
                    let elem_tokens = elem.to_token_stream();
                    tokens.push(quote! {
                        ::serde_json::Value::String(format!("{}", #elem_tokens))
                    });
                }
            }
        }

        Some(tokens)
    } else {
        None
    }
}

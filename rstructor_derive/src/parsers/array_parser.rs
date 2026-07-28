use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::meta::ParseNestedMeta;
use syn::punctuated::Punctuated;
use syn::{Expr, ExprArray, Lit, Token, parenthesized};

/// Parse a multi-value attribute in either native supported form:
/// `examples = [one, two]` or `examples(one, two)`.
pub fn parse_expression_list(
    meta: &ParseNestedMeta<'_>,
    attribute_name: &str,
) -> syn::Result<Vec<Expr>> {
    if meta.input.peek(Token![=]) {
        let value = meta.value()?;
        let expression: Expr = value.parse()?;
        let Expr::Array(array) = expression else {
            return Err(syn::Error::new_spanned(
                expression,
                format!("`{attribute_name}` must be an array expression"),
            ));
        };
        return Ok(array.elems.into_iter().collect());
    }

    if meta.input.peek(syn::token::Paren) {
        let content;
        parenthesized!(content in meta.input);
        return Ok(Punctuated::<Expr, Token![,]>::parse_terminated(&content)?
            .into_iter()
            .collect());
    }

    Err(meta.error(format!("`{attribute_name}` must use `= [...]` or `(...)`")))
}

/// Convert an already-parsed Rust array expression into generated
/// `serde_json::Value` expressions.
pub fn array_expression_tokens(array: &ExprArray) -> Vec<TokenStream> {
    expression_tokens(array.elems.iter())
}

/// Convert parsed Rust expressions into generated `serde_json::Value`
/// expressions.
pub fn expression_tokens<'a>(expressions: impl IntoIterator<Item = &'a Expr>) -> Vec<TokenStream> {
    let mut tokens = Vec::new();

    for elem in expressions {
        match elem {
            Expr::Macro(expr_macro)
                if expr_macro
                    .mac
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "json") =>
            {
                let elem_tokens = elem.to_token_stream();
                tokens.push(quote! {
                    #elem_tokens
                });
            }
            Expr::Lit(lit) => match &lit.lit {
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
                    tokens.push(quote! {
                        ::serde_json::json!(#lit_float)
                    });
                }
                Lit::Bool(lit_bool) => {
                    let b = lit_bool.value;
                    tokens.push(quote! {
                        ::serde_json::Value::Bool(#b)
                    });
                }
                _ => {
                    let elem_tokens = elem.to_token_stream();
                    tokens.push(quote! {
                        ::serde_json::Value::String(#elem_tokens.to_string())
                    });
                }
            },
            _ => {
                let elem_tokens = elem.to_token_stream();
                tokens.push(quote! {
                    ::serde_json::Value::String(format!("{}", #elem_tokens))
                });
            }
        }
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    use syn::parse_str;

    #[test]
    fn test_array_expression_tokens() {
        let input = "[\"apple\", \"banana\", \"cherry\"]";
        let array_expr: syn::ExprArray = parse_str(input).unwrap();
        assert_eq!(array_expression_tokens(&array_expr).len(), 3);
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

    #[test]
    fn preserves_json_macro_array_elements_as_values() {
        let array_expr: syn::ExprArray =
            parse_str(r#"[::serde_json::json!({"symbol": "AAPL", "weight": 0.5})]"#).unwrap();

        let tokens = array_expression_tokens(&array_expr);
        assert_eq!(tokens.len(), 1);

        let token_string = tokens[0].to_string();
        assert!(token_string.contains("serde_json"));
        assert!(token_string.contains("json"));
        assert!(!token_string.contains("Value :: String"));
    }

    #[test]
    fn parses_both_multi_value_attribute_forms() {
        let array_form: syn::Attribute = syn::parse_quote!(#[llm(examples = ["SPY", "QQQ"])]);
        let parenthesized_form: syn::Attribute = syn::parse_quote!(#[llm(examples("SPY", "QQQ"))]);

        for attribute in [array_form, parenthesized_form] {
            let mut values = None;
            attribute
                .parse_nested_meta(|meta| {
                    values = Some(parse_expression_list(&meta, "examples")?);
                    Ok(())
                })
                .unwrap();
            assert_eq!(values.unwrap().len(), 2);
        }
    }
}

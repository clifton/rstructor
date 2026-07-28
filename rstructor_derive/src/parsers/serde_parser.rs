use syn::punctuated::Punctuated;
use syn::{Attribute, Expr, Lit, Meta, Token};

/// The subset of Serde metadata that currently affects generated schemas.
///
/// Parsing is deliberately permissive: Serde owns this namespace, so
/// `#[derive(Instructor)]` must ignore metadata it does not understand instead
/// of rejecting valid Serde options such as `default`, `flatten`, or `with`.
#[derive(Default)]
pub struct SerdeAttributes {
    /// The field/variant name accepted during deserialization.
    pub rename: Option<String>,
    /// The case rule applied during deserialization.
    pub rename_all: Option<String>,
    /// The case rule applied to all struct-variant fields during
    /// deserialization.
    pub rename_all_fields: Option<String>,
    pub tag: Option<String>,
    pub content: Option<String>,
    pub untagged: bool,
    /// Whether Serde excludes this field or variant from deserialization.
    pub skip_deserializing: bool,
}

pub fn parse_serde_attributes(attrs: &[Attribute]) -> SerdeAttributes {
    let mut parsed = SerdeAttributes::default();

    for attr in attrs.iter().filter(|attr| attr.path().is_ident("serde")) {
        let Ok(items) = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
        else {
            // Serde's own derive reports malformed Serde metadata. Ignoring it
            // here prevents this macro from claiming ownership of that syntax.
            continue;
        };

        for item in items {
            match item {
                Meta::NameValue(meta) if meta.path.is_ident("rename") => {
                    parsed.rename = string_value(&meta.value);
                }
                Meta::NameValue(meta) if meta.path.is_ident("rename_all") => {
                    parsed.rename_all = string_value(&meta.value);
                }
                Meta::NameValue(meta) if meta.path.is_ident("rename_all_fields") => {
                    parsed.rename_all_fields = string_value(&meta.value);
                }
                Meta::List(meta) if meta.path.is_ident("rename") => {
                    parsed.rename = directional_string(&meta, "deserialize");
                }
                Meta::List(meta) if meta.path.is_ident("rename_all") => {
                    parsed.rename_all = directional_string(&meta, "deserialize");
                }
                Meta::List(meta) if meta.path.is_ident("rename_all_fields") => {
                    parsed.rename_all_fields = directional_string(&meta, "deserialize");
                }
                Meta::NameValue(meta) if meta.path.is_ident("tag") => {
                    parsed.tag = string_value(&meta.value);
                }
                Meta::NameValue(meta) if meta.path.is_ident("content") => {
                    parsed.content = string_value(&meta.value);
                }
                Meta::Path(path) if path.is_ident("untagged") => {
                    parsed.untagged = true;
                }
                Meta::Path(path)
                    if path.is_ident("skip") || path.is_ident("skip_deserializing") =>
                {
                    parsed.skip_deserializing = true;
                }
                _ => {
                    // All other Serde metadata is intentionally left to Serde.
                }
            }
        }
    }

    parsed
}

fn directional_string(meta: &syn::MetaList, direction: &str) -> Option<String> {
    let items = meta
        .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
        .ok()?;

    items.into_iter().find_map(|item| match item {
        Meta::NameValue(meta) if meta.path.is_ident(direction) => string_value(&meta.value),
        _ => None,
    })
}

fn string_value(value: &Expr) -> Option<String> {
    let Expr::Lit(value) = value else {
        return None;
    };
    let Lit::Str(value) = &value.lit else {
        return None;
    };
    Some(value.value())
}

#[cfg(test)]
mod tests {
    use syn::parse_quote;

    use super::parse_serde_attributes;

    #[test]
    fn extracts_supported_values_and_ignores_other_valid_serde_metadata() {
        let input: syn::DeriveInput = parse_quote! {
            #[serde(
                rename(serialize = "writeName", deserialize = "readName"),
                rename_all(serialize = "PascalCase", deserialize = "camelCase"),
                rename_all_fields(serialize = "UPPERCASE", deserialize = "snake_case"),
                tag = "kind",
                content = "payload",
                untagged,
                skip_deserializing,
                default,
                alias = "legacy"
            )]
            enum Example {}
        };

        let attrs = parse_serde_attributes(&input.attrs);
        assert_eq!(attrs.rename.as_deref(), Some("readName"));
        assert_eq!(attrs.rename_all.as_deref(), Some("camelCase"));
        assert_eq!(attrs.rename_all_fields.as_deref(), Some("snake_case"));
        assert_eq!(attrs.tag.as_deref(), Some("kind"));
        assert_eq!(attrs.content.as_deref(), Some("payload"));
        assert!(attrs.untagged);
        assert!(attrs.skip_deserializing);
    }

    #[test]
    fn serialize_only_metadata_does_not_change_deserialization_names() {
        let input: syn::DeriveInput = parse_quote! {
            #[serde(
                rename(serialize = "writeName"),
                rename_all(serialize = "PascalCase"),
                rename_all_fields(serialize = "UPPERCASE"),
                skip_serializing
            )]
            enum Example {}
        };

        let attrs = parse_serde_attributes(&input.attrs);
        assert_eq!(attrs.rename, None);
        assert_eq!(attrs.rename_all, None);
        assert_eq!(attrs.rename_all_fields, None);
        assert!(!attrs.skip_deserializing);
    }
}

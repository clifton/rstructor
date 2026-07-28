use syn::Variant;

use crate::parsers::serde_parser::parse_serde_attributes;

/// Represents parsed variant attributes
pub struct VariantAttributes {
    pub description: Option<String>,
    /// Variant rename from #[serde(rename = "...")]
    pub serde_rename: Option<String>,
}

/// Parse a single enum variant's llm and serde attributes
pub fn parse_variant_attributes(variant: &Variant) -> syn::Result<VariantAttributes> {
    let mut description = None;

    for attr in variant
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("llm"))
    {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("description") {
                let value = meta.value()?;
                let content: syn::LitStr = value.parse()?;
                description = Some(content.value());
            } else {
                return Err(
                    meta.error("unsupported variant `llm` attribute; expected `description`")
                );
            }
            Ok(())
        })?;
    }

    Ok(VariantAttributes {
        description,
        serde_rename: parse_serde_attributes(&variant.attrs).rename,
    })
}

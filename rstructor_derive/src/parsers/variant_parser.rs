use syn::Variant;

use crate::parsers::serde_parser::parse_serde_attributes;

/// Represents parsed variant attributes
pub struct VariantAttributes {
    pub description: Option<String>,
    /// Variant rename from #[serde(rename = "...")]
    pub serde_rename: Option<String>,
    /// Case rule for named fields inside this variant.
    pub serde_rename_all: Option<String>,
    /// Whether Serde excludes this variant from deserialization.
    pub serde_skip_deserializing: bool,
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

    let serde = parse_serde_attributes(&variant.attrs);
    Ok(VariantAttributes {
        description,
        serde_rename: serde.rename,
        serde_rename_all: serde.rename_all,
        serde_skip_deserializing: serde.skip_deserializing,
    })
}

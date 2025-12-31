use syn::Variant;

/// Represents parsed variant attributes
pub struct VariantAttributes {
    pub description: Option<String>,
    /// Variant rename from #[serde(rename = "...")]
    pub serde_rename: Option<String>,
}

/// Parse a single enum variant's llm and serde attributes
pub fn parse_variant_attributes(variant: &Variant) -> VariantAttributes {
    let mut description = None;
    let mut serde_rename = None;

    // Extract attributes
    for attr in &variant.attrs {
        // Parse serde attributes for rename
        if attr.path().is_ident("serde") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("rename") {
                    let value = meta.value()?;
                    let content: syn::LitStr = value.parse()?;
                    serde_rename = Some(content.value());
                }
                Ok(())
            });
        }

        if attr.path().is_ident("llm") {
            // Parse attribute arguments
            let _result = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("description") {
                    let value = meta.value()?;
                    let content: syn::LitStr = value.parse()?;
                    description = Some(content.value());
                }
                Ok(())
            });
        }
    }

    VariantAttributes {
        description,
        serde_rename,
    }
}

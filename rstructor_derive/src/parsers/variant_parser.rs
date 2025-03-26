use syn::Variant;

/// Represents parsed variant attributes
pub struct VariantAttributes {
    pub description: Option<String>,
}

/// Parse a single enum variant's llm attributes
pub fn parse_variant_attributes(variant: &Variant) -> VariantAttributes {
    let mut description = None;

    // Extract attributes
    for attr in &variant.attrs {
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

    VariantAttributes { description }
}

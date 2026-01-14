/// Container-level attributes for structs and enums
#[derive(Debug, Clone)]
pub struct ContainerAttributes {
    /// Description of the struct or enum
    pub description: Option<String>,

    /// Custom title for the schema (overrides the default type name)
    pub title: Option<String>,

    /// Examples of valid instances (as tokenstreams)
    pub examples: Vec<proc_macro2::TokenStream>,

    /// Serde rename_all case style (from serde attribute)
    pub serde_rename_all: Option<String>,

    /// Custom validation function path (e.g., "validate_product" or "my_module::validate")
    pub validate: Option<String>,

    /// Serde tag field name for internally/adjacently tagged enums
    pub serde_tag: Option<String>,

    /// Serde content field name for adjacently tagged enums
    pub serde_content: Option<String>,

    /// Whether the enum is untagged
    pub serde_untagged: bool,
}

/// Builder for constructing ContainerAttributes
#[derive(Default)]
pub struct ContainerAttributesBuilder {
    description: Option<String>,
    title: Option<String>,
    examples: Vec<proc_macro2::TokenStream>,
    serde_rename_all: Option<String>,
    validate: Option<String>,
    serde_tag: Option<String>,
    serde_content: Option<String>,
    serde_untagged: bool,
}

impl ContainerAttributesBuilder {
    pub fn description(mut self, desc: Option<String>) -> Self {
        self.description = desc;
        self
    }

    pub fn title(mut self, title: Option<String>) -> Self {
        self.title = title;
        self
    }

    pub fn examples(mut self, examples: Vec<proc_macro2::TokenStream>) -> Self {
        self.examples = examples;
        self
    }

    pub fn serde_rename_all(mut self, rename_all: Option<String>) -> Self {
        self.serde_rename_all = rename_all;
        self
    }

    pub fn validate(mut self, validate: Option<String>) -> Self {
        self.validate = validate;
        self
    }

    pub fn serde_tag(mut self, tag: Option<String>) -> Self {
        self.serde_tag = tag;
        self
    }

    pub fn serde_content(mut self, content: Option<String>) -> Self {
        self.serde_content = content;
        self
    }

    pub fn serde_untagged(mut self, untagged: bool) -> Self {
        self.serde_untagged = untagged;
        self
    }

    pub fn build(self) -> ContainerAttributes {
        ContainerAttributes {
            description: self.description,
            title: self.title,
            examples: self.examples,
            serde_rename_all: self.serde_rename_all,
            validate: self.validate,
            serde_tag: self.serde_tag,
            serde_content: self.serde_content,
            serde_untagged: self.serde_untagged,
        }
    }
}

impl ContainerAttributes {
    /// Create a new builder for ContainerAttributes
    pub fn builder() -> ContainerAttributesBuilder {
        ContainerAttributesBuilder::default()
    }

    /// Returns true if there are no attributes set - currently unused but kept for future use
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.description.is_none()
            && self.title.is_none()
            && self.examples.is_empty()
            && self.serde_rename_all.is_none()
            && self.validate.is_none()
            && self.serde_tag.is_none()
            && self.serde_content.is_none()
            && !self.serde_untagged
    }
}

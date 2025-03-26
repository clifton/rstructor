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
}

impl ContainerAttributes {
    /// Create a new container attributes object
    pub fn new(
        description: Option<String>,
        title: Option<String>,
        examples: Vec<proc_macro2::TokenStream>,
        serde_rename_all: Option<String>,
    ) -> Self {
        Self {
            description,
            title,
            examples,
            serde_rename_all,
        }
    }

    /// Returns true if there are no attributes set
    pub fn is_empty(&self) -> bool {
        self.description.is_none()
            && self.title.is_none()
            && self.examples.is_empty()
            && self.serde_rename_all.is_none()
    }
}

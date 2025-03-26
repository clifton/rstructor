/// Container-level attributes for structs and enums
#[derive(Debug, Clone)]
pub struct ContainerAttributes {
    pub description: Option<String>,
}

impl ContainerAttributes {
    /// Create a new container attributes object
    pub fn new(description: Option<String>) -> Self {
        Self { description }
    }
}
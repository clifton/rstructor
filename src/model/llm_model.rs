use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::Result;
use crate::schema::SchemaType;

/// LLMModel trait combines schemas, serialization, and optionally validation
pub trait LLMModel: SchemaType + DeserializeOwned + Serialize {
    /// Optional validation logic beyond type checking
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Implement LLMModel for any type that implements the required traits
impl<T: SchemaType + DeserializeOwned + Serialize> LLMModel for T {}
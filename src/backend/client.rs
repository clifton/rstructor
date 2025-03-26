use async_trait::async_trait;
use serde::de::DeserializeOwned;

use crate::error::Result;
use crate::model::LLMModel;

/// LLMClient trait defines the interface for all LLM API clients
#[async_trait]
pub trait LLMClient {
    /// Generate a structured object of type T from a prompt
    async fn generate_struct<T>(&self, prompt: &str) -> Result<T>
    where
        T: LLMModel + DeserializeOwned + Send + 'static;
    
    /// Raw completion without structure (returns plain text)
    async fn generate(&self, prompt: &str) -> Result<String>;
}
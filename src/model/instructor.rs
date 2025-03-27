use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::Result;
use crate::schema::SchemaType;

/// The `Instructor` trait combines JSON schema generation, serialization, and validation.
///
/// This trait is automatically implemented for any type that implements the required traits
/// (SchemaType, DeserializeOwned, and Serialize), but you can also provide a custom
/// implementation to add your own validation logic.
///
/// # Validation
///
/// The `validate` method is called automatically when an LLM generates a structured response,
/// allowing you to apply domain-specific validation logic beyond what type checking provides.
///
/// To add custom validation:
///
/// ```
/// use rstructor::{Instructor, RStructorError};
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Instructor, Serialize, Deserialize, Debug)]
/// struct Product {
///     name: String,
///     price: f64,
///     quantity: u32,
/// }
///
/// // Add a custom validate implementation
/// impl Product {
///     fn validate(&self) -> rstructor::Result<()> {
///         // Price must be positive
///         if self.price <= 0.0 {
///             return Err(RStructorError::ValidationError(
///                 format!("Product price must be positive, got {}", self.price)
///             ));
///         }
///         
///         // Name can't be empty
///         if self.name.trim().is_empty() {
///             return Err(RStructorError::ValidationError(
///                 "Product name cannot be empty".to_string()
///             ));
///         }
///         
///         Ok(())
///     }
/// }
/// ```
///
/// # Example: Using with LLM clients
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use rstructor::{LLMClient, OpenAIClient, OpenAIModel, Instructor};
/// use serde::{Serialize, Deserialize};
///
/// // Define a product model
/// #[derive(Instructor, Serialize, Deserialize, Debug)]
/// struct ProductInfo {
///     name: String,
///     price: f64,
/// }
///
/// // Create a client
/// let client = OpenAIClient::new("your-api-key")?
///     .model(OpenAIModel::Gpt35Turbo)
///     .build();
///
/// // Get structured data with automatic validation
/// let product = client.generate_struct::<ProductInfo>("Describe a laptop").await?;
/// # Ok(())
/// # }
/// ```
pub trait Instructor: SchemaType + DeserializeOwned + Serialize {
    /// Optional validation logic beyond type checking
    ///
    /// This method is called automatically by `generate_struct` to validate
    /// the data returned by the LLM. The default implementation does nothing
    /// and returns Ok(()), but you can override it to add your own validation logic.
    ///
    /// # Example
    ///
    /// ```
    /// # use rstructor::{Instructor, RStructorError};
    /// # use serde::{Serialize, Deserialize};
    /// #
    /// # #[derive(Instructor, Serialize, Deserialize, Debug)]
    /// # struct Product {
    /// #     name: String,
    /// #     price: f64,
    /// # }
    /// #
    /// impl Product {
    ///     fn validate(&self) -> rstructor::Result<()> {
    ///         // Price must be positive
    ///         if self.price < 0.0 {
    ///             return Err(RStructorError::ValidationError(
    ///                 format!("Price must be positive, got {}", self.price)
    ///             ));
    ///         }
    ///         Ok(())
    ///     }
    /// }
    /// ```
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Implement Instructor for any type that implements the required traits
impl<T: SchemaType + DeserializeOwned + Serialize> Instructor for T {}

/// Helper trait to mark a type as implementing custom validation.
///
/// This is a marker trait only used for documentation purposes to indicate
/// that a type provides a custom validation implementation beyond the default.
/// You don't need to implement this trait directly - it's only for clarity
/// in documentation.
pub trait Validatable {}

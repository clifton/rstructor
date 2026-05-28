use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::Result;
use crate::schema::SchemaType;

/// The `Instructor` trait combines JSON schema generation, serialization, and validation.
///
/// It is implemented for your type by `#[derive(Instructor)]`, which also generates the
/// [`SchemaType`] implementation. Add custom validation with the `#[llm(validate = "path")]`
/// attribute (see [`validate`](Instructor::validate)); the derive wires it into the trait.
///
/// # Nested Types and Schema Embedding
///
/// When you have nested structs or enums, they should also derive `Instructor` to ensure
/// their full schema is embedded in the parent type. This produces complete JSON schemas
/// that help LLMs generate correct structured output.
///
/// ```rust
/// # use rstructor::Instructor;
/// # use serde::{Serialize, Deserialize};
/// // Parent type derives Instructor
/// #[derive(Instructor, Serialize, Deserialize)]
/// struct Parent {
///     child: Child,  // Child's schema will be embedded
/// }
///
/// // Nested types should also derive Instructor for complete schema
/// #[derive(Instructor, Serialize, Deserialize)]
/// struct Child {
///     name: String,
/// }
/// ```
///
/// This ensures the generated schema includes all nested properties.
///
/// # Validation
///
/// The `validate` method is called automatically when an LLM generates a structured response,
/// allowing you to apply domain-specific validation logic beyond what type checking provides.
///
/// To add custom validation, use the `validate` attribute with a function path:
///
/// ```
/// use rstructor::{Instructor, RStructorError};
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Instructor, Serialize, Deserialize)]
/// #[llm(validate = "validate_product")]
/// struct Product {
///     name: String,
///     price: f64,
///     quantity: u32,
/// }
///
/// fn validate_product(product: &Product) -> rstructor::Result<()> {
///     // Price must be positive
///     if product.price <= 0.0 {
///         return Err(RStructorError::ValidationError(
///             format!("Product price must be positive, got {}", product.price)
///         ));
///     }
///
///     // Name can't be empty
///     if product.name.trim().is_empty() {
///         return Err(RStructorError::ValidationError(
///             "Product name cannot be empty".to_string()
///         ));
///     }
///
///     Ok(())
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
///     .model(OpenAIModel::Gpt55);
///
/// // Get structured data with automatic validation
/// let product = client.materialize::<ProductInfo>("Describe a laptop").await?;
/// # Ok(())
/// # }
/// ```
pub trait Instructor: SchemaType + DeserializeOwned + Serialize {
    /// Optional validation logic beyond type checking.
    ///
    /// This method is called automatically by `materialize` to validate the data
    /// returned by the LLM; a failure triggers an automatic re-ask with the error
    /// fed back to the model. The default implementation returns `Ok(())`.
    ///
    /// To add custom validation, use the `#[llm(validate = "path")]` container
    /// attribute — the derive macro wires your function into this trait method:
    ///
    /// ```
    /// use rstructor::{Instructor, RStructorError};
    /// use serde::{Serialize, Deserialize};
    ///
    /// #[derive(Instructor, Serialize, Deserialize, Debug)]
    /// #[llm(validate = "validate_product")]
    /// struct Product {
    ///     name: String,
    ///     price: f64,
    /// }
    ///
    /// fn validate_product(product: &Product) -> rstructor::Result<()> {
    ///     if product.price < 0.0 {
    ///         return Err(RStructorError::ValidationError(
    ///             format!("Price must be positive, got {}", product.price),
    ///         ));
    ///     }
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Important: inherent methods do not work
    ///
    /// Writing an *inherent* `impl Product { fn validate(&self) {...} }` does **not**
    /// hook into validation. `#[derive(Instructor)]` always generates this trait
    /// method (defaulting to `Ok(())`), and trait dispatch ignores the inherent
    /// method — so an inherent `validate` would silently never run. Always use the
    /// `#[llm(validate = "...")]` attribute, or hand-write `impl Instructor`.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

// The blanket implementation is removed
// Instead, the derive macro will handle implementing Instructor for each type
// This avoids the conflicting implementation errors

/// Helper trait to mark a type as implementing custom validation.
///
/// This is a marker trait only used for documentation purposes to indicate
/// that a type provides a custom validation implementation beyond the default.
/// You don't need to implement this trait directly - it's only for clarity
/// in documentation.
pub trait Validatable {}

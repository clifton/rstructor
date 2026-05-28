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
    /// The derive-generated implementation automatically recurses into nested
    /// `Instructor` fields — directly, and through `Option`, `Vec`, `Box`, and
    /// string-keyed maps — before running this type's own validator, so validating
    /// a parent validates its entire tree.
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

// Container implementations so that validation recurses into nested values.
// `#[derive(Instructor)]` validates each field via these impls, so a `Vec`,
// `Option`, `Box`, or string-keyed map of validating types is validated
// element-by-element as part of its parent.

impl<T: Instructor> Instructor for Option<T> {
    fn validate(&self) -> Result<()> {
        match self {
            Some(value) => value.validate(),
            None => Ok(()),
        }
    }
}

impl<T: Instructor> Instructor for Vec<T> {
    fn validate(&self) -> Result<()> {
        for value in self {
            value.validate()?;
        }
        Ok(())
    }
}

impl<T: Instructor> Instructor for Box<T> {
    fn validate(&self) -> Result<()> {
        (**self).validate()
    }
}

impl<V: Instructor> Instructor for std::collections::HashMap<String, V> {
    fn validate(&self) -> Result<()> {
        for value in self.values() {
            value.validate()?;
        }
        Ok(())
    }
}

/// Internal helpers used by `#[derive(Instructor)]`. Not part of the public API
/// and exempt from semver guarantees.
#[doc(hidden)]
pub mod __private {
    use super::Instructor;
    use crate::error::Result;

    /// Autoref-specialization wrapper that lets generated code validate a field
    /// **iff** its type implements [`Instructor`], and otherwise do nothing —
    /// without the derive macro having to know which field types are `Instructor`.
    ///
    /// `#[derive(Instructor)]` emits `Probe(&self.field).rstructor_probe()?` for
    /// each field. When the field's type implements `Instructor`, the inherent
    /// method below is selected (inherent methods take priority over trait
    /// methods); otherwise method resolution falls back to the [`ProbeFallback`]
    /// trait, which is a no-op.
    pub struct Probe<'a, T>(pub &'a T);

    /// Fallback for non-`Instructor` field types (e.g. `String`, `u32`).
    pub trait ProbeFallback {
        fn rstructor_probe(&self) -> Result<()>;
    }

    impl<T> ProbeFallback for Probe<'_, T> {
        fn rstructor_probe(&self) -> Result<()> {
            Ok(())
        }
    }

    impl<T: Instructor> Probe<'_, T> {
        /// Validate the wrapped value (selected over the trait method when
        /// `T: Instructor`).
        pub fn rstructor_probe(&self) -> Result<()> {
            self.0.validate()
        }
    }
}

/// Helper trait to mark a type as implementing custom validation.
///
/// This is a marker trait only used for documentation purposes to indicate
/// that a type provides a custom validation implementation beyond the default.
/// You don't need to implement this trait directly - it's only for clarity
/// in documentation.
pub trait Validatable {}

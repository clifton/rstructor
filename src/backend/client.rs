use async_trait::async_trait;
use serde::de::DeserializeOwned;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::error::{RStructorError, Result};
use crate::model::Instructor;

/// LLMClient trait defines the interface for all LLM API clients.
///
/// This trait is the core abstraction for interacting with different LLM providers
/// like OpenAI or Anthropic. It provides methods for generating structured data
/// and raw text completions.
///
/// The library includes implementations for popular LLM providers:
/// - `OpenAIClient` for OpenAI's GPT models (gpt-3.5-turbo, gpt-4, etc.)
/// - `AnthropicClient` for Anthropic's Claude models
///
/// # Examples
///
/// Using OpenAI client:
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use rstructor::{LLMClient, Instructor, OpenAIClient, OpenAIModel};
/// use serde::{Serialize, Deserialize};
///
/// // Define your data model
/// #[derive(Instructor, Serialize, Deserialize, Debug)]
/// struct Movie {
///     title: String,
///     director: String,
///     year: u16,
/// }
///
/// // Create a client
/// let client = OpenAIClient::new("your-openai-api-key")?
///     .model(OpenAIModel::Gpt35Turbo)
///     .temperature(0.0)
///     .build();
///
/// // Generate a structured response
/// let prompt = "Describe the movie Inception";
/// let movie: Movie = client.generate_struct(prompt).await?;
///
/// println!("Title: {}", movie.title);
/// println!("Director: {}", movie.director);
/// println!("Year: {}", movie.year);
/// # Ok(())
/// # }
/// ```
///
/// Using Anthropic client:
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use rstructor::{LLMClient, Instructor, AnthropicClient, AnthropicModel};
/// use serde::{Serialize, Deserialize};
///
/// // Define your data model
/// #[derive(Instructor, Serialize, Deserialize, Debug)]
/// struct MovieReview {
///     movie_title: String,
///     rating: f32,
///     review: String,
/// }
///
/// // Create a client
/// let client = AnthropicClient::new("your-anthropic-api-key")?
///     .model(AnthropicModel::Claude3Haiku)
///     .temperature(0.0)
///     .build();
///
/// // Generate a structured response
/// let prompt = "Write a short review of the movie The Matrix";
/// let review: MovieReview = client.generate_struct(prompt).await?;
///
/// println!("Movie: {}", review.movie_title);
/// println!("Rating: {}/10", review.rating);
/// println!("Review: {}", review.review);
/// # Ok(())
/// # }
/// ```
#[async_trait]
pub trait LLMClient {
    /// Generate a structured object of type T from a prompt.
    ///
    /// This method takes a text prompt and returns a structured object
    /// of type T, where T implements the `Instructor` trait. The LLM is guided
    /// to produce output that conforms to the JSON schema defined by T.
    ///
    /// If the returned data doesn't match the expected schema or fails validation,
    /// an error is returned.
    async fn generate_struct<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static;

    /// Generate a structured object with automatic retry for validation errors.
    ///
    /// Similar to `generate_struct`, but will automatically retry if validation fails,
    /// including the validation error message in subsequent attempts.
    ///
    /// Parameters:
    /// - `prompt`: The initial prompt
    /// - `max_retries`: Maximum number of retry attempts (default: 3)
    /// - `include_errors`: Whether to include prior validation errors in retry prompts (default: true)
    ///
    /// Note: This will retry only for validation errors, not for API or other errors.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// use rstructor::{LLMClient, OpenAIClient, OpenAIModel, Instructor};
    /// use serde::{Serialize, Deserialize};
    ///
    /// #[derive(Instructor, Serialize, Deserialize, Debug)]
    /// struct Recipe {
    ///     name: String,
    ///     ingredients: Vec<String>,
    ///     steps: Vec<String>,
    /// }
    ///
    /// // Implement custom validation
    /// impl Recipe {
    ///     fn validate(&self) -> rstructor::Result<()> {
    ///         // Your validation logic here
    ///         Ok(())
    ///     }
    /// }
    ///
    /// let client = OpenAIClient::new("your-api-key")?
    ///     .model(OpenAIModel::Gpt4O)
    ///     .build();
    ///
    /// // This will automatically retry up to 3 times if validation fails
    /// let recipe = client.generate_struct_with_retry::<Recipe>(
    ///     "Give me a chocolate cake recipe",
    ///     Some(3),     // max_retries
    ///     Some(true),  // include_errors
    /// ).await?;
    ///
    /// println!("Recipe: {}", recipe.name);
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(
        name = "generate_struct_with_retry",
        skip(self, prompt),
        fields(
            type_name = std::any::type_name::<T>(),
            max_retries,
            include_errors,
            prompt_len = prompt.len()
        )
    )]
    async fn generate_struct_with_retry<T>(
        &self,
        prompt: &str,
        max_retries: Option<usize>,
        include_errors: Option<bool>,
    ) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        let max_attempts = max_retries.unwrap_or(3) + 1; // +1 for initial attempt
        let include_errors = include_errors.unwrap_or(true);
        let mut validation_errors: Option<String> = None;
        let mut current_prompt = prompt.to_string();

        trace!(
            "Starting structured generation with retry: max_attempts={}, include_errors={}",
            max_attempts, include_errors
        );

        for attempt in 0..max_attempts {
            // Add validation errors to the prompt if available and enabled
            if attempt > 0 && include_errors && validation_errors.is_some() {
                debug!(
                    attempt,
                    error = validation_errors.as_ref().unwrap(),
                    "Retrying with validation error feedback"
                );

                let error_prompt = format!(
                    "{}\n\nYour previous response contained validation errors. Please provide a complete, valid JSON response that includes ALL required fields and follows the schema exactly.\n\nError details:\n{}\n\nPlease fix the issues in your response. Make sure to:\n1. Include ALL required fields exactly as specified in the schema\n2. For enum fields, use EXACTLY one of the allowed values from the description\n3. CRITICAL: For arrays where items.type = 'object':\n   - You MUST provide an array of OBJECTS, not strings or primitive values\n   - Each object must be a complete JSON object with all its required fields\n   - Include multiple items (at least 2-3) in arrays of objects\n4. Verify all nested objects have their complete structure\n5. Follow ALL type specifications (string, number, boolean, array, object)",
                    prompt,
                    validation_errors.as_ref().unwrap()
                );
                current_prompt = error_prompt;
            }

            // Log attempt information
            info!(
                attempt = attempt + 1,
                total_attempts = max_attempts,
                "Generation attempt"
            );

            // Attempt to generate structured data
            match self.generate_struct::<T>(&current_prompt).await {
                Ok(result) => {
                    if attempt > 0 {
                        // If we succeeded after retries
                        info!(
                            attempts_used = attempt + 1,
                            "Successfully generated after {} retries", attempt
                        );
                    } else {
                        debug!("Successfully generated on first attempt");
                    }
                    return Ok(result);
                }
                Err(err) => {
                    // Only retry for validation errors
                    if let RStructorError::ValidationError(msg) = &err {
                        if attempt < max_attempts - 1 {
                            warn!(
                                attempt = attempt + 1,
                                error = msg,
                                "Validation error in generation attempt"
                            );
                            // Store error for next attempt
                            validation_errors = Some(msg.clone());
                            // Wait briefly before retrying
                            sleep(Duration::from_millis(500)).await;
                            continue;
                        } else {
                            // Last attempt failed
                            error!(
                                attempts = max_attempts,
                                error = msg,
                                "Failed after maximum retry attempts with validation errors"
                            );
                        }
                    } else {
                        // For non-validation errors
                        error!(
                            error = ?err,
                            "Non-validation error occurred during generation"
                        );
                    }

                    // For non-validation errors or last attempt, return the error
                    return Err(err);
                }
            }
        }

        // This should never be reached due to the returns in the loop
        unreachable!()
    }

    /// Raw completion without structure (returns plain text).
    ///
    /// This method provides a simpler interface for getting raw text completions
    /// from the LLM without enforcing any structure.
    async fn generate(&self, prompt: &str) -> Result<String>;
}

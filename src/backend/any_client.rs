//! A provider-agnostic client selectable at runtime.
//!
//! [`LLMClient::materialize`](crate::LLMClient::materialize) is generic over the
//! target type, which makes the `LLMClient` trait non-object-safe — you cannot
//! store a provider behind `Box<dyn LLMClient>` or pick one dynamically through a
//! trait object. [`AnyClient`] solves the common need behind that limitation
//! ("choose a provider at runtime and keep it in a single type") by wrapping each
//! concrete client in an enum that itself implements [`LLMClient`].

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use std::fmt;
use std::str::FromStr;

use crate::backend::usage::{
    GenerateResult, MaterializeFailure, MaterializeReport, MaterializeResult,
};
use crate::backend::{LLMClient, MediaFile, ModelInfo};
use crate::error::{ApiErrorKind, RStructorError, Result};
use crate::model::Instructor;

#[cfg(feature = "anthropic")]
use crate::backend::anthropic::AnthropicClient;
#[cfg(feature = "gemini")]
use crate::backend::gemini::GeminiClient;
#[cfg(feature = "grok")]
use crate::backend::grok::GrokClient;
#[cfg(feature = "openai")]
use crate::backend::openai::OpenAIClient;

/// Identifies an LLM provider for runtime selection via [`AnyClient`].
///
/// Only providers enabled via Cargo features are present as variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
    /// OpenAI (reads `OPENAI_API_KEY`).
    #[cfg(feature = "openai")]
    OpenAI,
    /// Anthropic (reads `ANTHROPIC_API_KEY`).
    #[cfg(feature = "anthropic")]
    Anthropic,
    /// Google Gemini (reads `GEMINI_API_KEY`, then `GOOGLE_API_KEY`).
    #[cfg(feature = "gemini")]
    Gemini,
    /// xAI / Grok (reads `XAI_API_KEY`).
    #[cfg(feature = "grok")]
    Grok,
}

impl FromStr for Provider {
    type Err = RStructorError;

    /// Parse a case-insensitive provider name.
    ///
    /// `xai` is accepted as an alias for `grok`. A recognized provider whose
    /// Cargo feature is disabled returns an error naming the feature to enable.
    fn from_str(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "openai" => {
                #[cfg(feature = "openai")]
                {
                    Ok(Self::OpenAI)
                }
                #[cfg(not(feature = "openai"))]
                {
                    Err(provider_feature_disabled("openai", "openai"))
                }
            }
            "anthropic" => {
                #[cfg(feature = "anthropic")]
                {
                    Ok(Self::Anthropic)
                }
                #[cfg(not(feature = "anthropic"))]
                {
                    Err(provider_feature_disabled("anthropic", "anthropic"))
                }
            }
            "gemini" => {
                #[cfg(feature = "gemini")]
                {
                    Ok(Self::Gemini)
                }
                #[cfg(not(feature = "gemini"))]
                {
                    Err(provider_feature_disabled("gemini", "gemini"))
                }
            }
            "grok" | "xai" => {
                #[cfg(feature = "grok")]
                {
                    Ok(Self::Grok)
                }
                #[cfg(not(feature = "grok"))]
                {
                    Err(provider_feature_disabled(name, "grok"))
                }
            }
            _ => Err(RStructorError::Unsupported(format!(
                "unknown provider `{name}`; valid providers: openai, anthropic, gemini, grok (alias: xai)"
            ))),
        }
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "openai")]
            Self::OpenAI => formatter.write_str("openai"),
            #[cfg(feature = "anthropic")]
            Self::Anthropic => formatter.write_str("anthropic"),
            #[cfg(feature = "gemini")]
            Self::Gemini => formatter.write_str("gemini"),
            #[cfg(feature = "grok")]
            Self::Grok => formatter.write_str("grok"),
        }
    }
}

#[cfg(any(
    not(feature = "openai"),
    not(feature = "anthropic"),
    not(feature = "gemini"),
    not(feature = "grok")
))]
fn provider_feature_disabled(provider: &str, feature: &str) -> RStructorError {
    RStructorError::Unsupported(format!(
        "provider `{provider}` is disabled; enable the `{feature}` Cargo feature"
    ))
}

/// A provider-agnostic client chosen at runtime.
///
/// Because [`LLMClient`] has a generic `materialize` method it is not
/// object-safe, so `Box<dyn LLMClient>` is impossible. `AnyClient` is an enum
/// over the concrete clients that itself implements [`LLMClient`], giving you a
/// single, `Clone`, `Send + Sync` type that can hold whichever provider you
/// selected at runtime (from a CLI flag, config file, env, etc.).
///
/// Construct it with [`from_env_for`](Self::from_env_for), with
/// [`LLMClient::from_env`] (which auto-detects from the environment), or with
/// `From<ConcreteClient>` when you need custom configuration:
///
/// ```no_run
/// # async fn ex() -> Result<(), Box<dyn std::error::Error>> {
/// use rstructor::{AnyClient, Provider, LLMClient, Instructor};
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Instructor, Serialize, Deserialize, Debug)]
/// struct Movie {
///     title: String,
/// }
///
/// // Provider decided at runtime.
/// let provider = Provider::Anthropic;
/// let client = AnyClient::from_env_for(provider)?;
///
/// let movie: Movie = client.materialize("Describe Inception").await?;
/// println!("{}", movie.title);
/// # Ok(())
/// # }
/// ```
///
/// Wrapping a pre-configured client:
///
/// ```no_run
/// # fn ex() -> Result<(), Box<dyn std::error::Error>> {
/// use rstructor::{AnyClient, OpenAIClient, OpenAIModel};
///
/// let configured = OpenAIClient::from_env()?.model(OpenAIModel::Gpt56);
/// let client: AnyClient = configured.into();
/// # let _ = client;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub enum AnyClient {
    /// An OpenAI client.
    #[cfg(feature = "openai")]
    OpenAI(OpenAIClient),
    /// An Anthropic client.
    #[cfg(feature = "anthropic")]
    Anthropic(AnthropicClient),
    /// A Grok client.
    #[cfg(feature = "grok")]
    Grok(GrokClient),
    /// A Gemini client.
    #[cfg(feature = "gemini")]
    Gemini(GeminiClient),
}

impl AnyClient {
    /// Build a client for `provider`, reading its API key from the environment.
    ///
    /// # Errors
    ///
    /// Returns an error if the provider's environment variable is not set.
    pub fn from_env_for(provider: Provider) -> Result<Self> {
        match provider {
            #[cfg(feature = "openai")]
            Provider::OpenAI => Ok(Self::OpenAI(OpenAIClient::from_env()?)),
            #[cfg(feature = "anthropic")]
            Provider::Anthropic => Ok(Self::Anthropic(AnthropicClient::from_env()?)),
            #[cfg(feature = "grok")]
            Provider::Grok => Ok(Self::Grok(GrokClient::from_env()?)),
            #[cfg(feature = "gemini")]
            Provider::Gemini => Ok(Self::Gemini(GeminiClient::from_env()?)),
        }
    }

    /// Return the [`Provider`] backing this client.
    #[must_use]
    pub fn provider(&self) -> Provider {
        match self {
            #[cfg(feature = "openai")]
            Self::OpenAI(_) => Provider::OpenAI,
            #[cfg(feature = "anthropic")]
            Self::Anthropic(_) => Provider::Anthropic,
            #[cfg(feature = "grok")]
            Self::Grok(_) => Provider::Grok,
            #[cfg(feature = "gemini")]
            Self::Gemini(_) => Provider::Gemini,
        }
    }
}

#[cfg(feature = "openai")]
impl From<OpenAIClient> for AnyClient {
    fn from(client: OpenAIClient) -> Self {
        Self::OpenAI(client)
    }
}

#[cfg(feature = "anthropic")]
impl From<AnthropicClient> for AnyClient {
    fn from(client: AnthropicClient) -> Self {
        Self::Anthropic(client)
    }
}

#[cfg(feature = "grok")]
impl From<GrokClient> for AnyClient {
    fn from(client: GrokClient) -> Self {
        Self::Grok(client)
    }
}

#[cfg(feature = "gemini")]
impl From<GeminiClient> for AnyClient {
    fn from(client: GeminiClient) -> Self {
        Self::Gemini(client)
    }
}

/// Dispatch a method call to whichever provider this `AnyClient` wraps.
macro_rules! dispatch {
    ($self:expr, $client:ident => $call:expr) => {
        match $self {
            #[cfg(feature = "openai")]
            Self::OpenAI($client) => $call,
            #[cfg(feature = "anthropic")]
            Self::Anthropic($client) => $call,
            #[cfg(feature = "grok")]
            Self::Grok($client) => $call,
            #[cfg(feature = "gemini")]
            Self::Gemini($client) => $call,
        }
    };
}

#[async_trait]
impl LLMClient for AnyClient {
    async fn materialize<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        dispatch!(self, c => c.materialize(prompt).await)
    }

    async fn materialize_with_media<T>(&self, prompt: &str, media: &[MediaFile]) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        dispatch!(self, c => c.materialize_with_media(prompt, media).await)
    }

    async fn materialize_with_metadata<T>(&self, prompt: &str) -> Result<MaterializeResult<T>>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        dispatch!(self, c => c.materialize_with_metadata(prompt).await)
    }

    async fn materialize_with_attempts<T>(
        &self,
        prompt: &str,
    ) -> std::result::Result<MaterializeReport<T>, MaterializeFailure>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        dispatch!(self, c => c.materialize_with_attempts(prompt).await)
    }

    async fn materialize_with_media_and_attempts<T>(
        &self,
        prompt: &str,
        media: &[MediaFile],
    ) -> std::result::Result<MaterializeReport<T>, MaterializeFailure>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        dispatch!(self, c => c.materialize_with_media_and_attempts(prompt, media).await)
    }

    async fn generate(&self, prompt: &str) -> Result<String> {
        dispatch!(self, c => c.generate(prompt).await)
    }

    async fn generate_with_media(&self, prompt: &str, media: &[MediaFile]) -> Result<String> {
        dispatch!(self, c => c.generate_with_media(prompt, media).await)
    }

    async fn generate_with_metadata(&self, prompt: &str) -> Result<GenerateResult> {
        dispatch!(self, c => c.generate_with_metadata(prompt).await)
    }

    /// Auto-detect a provider from the environment.
    ///
    /// Enabled providers are tried in order: `OPENAI_API_KEY`,
    /// `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`, then
    /// `XAI_API_KEY`. The first configured provider wins. For deterministic
    /// selection, prefer [`AnyClient::from_env_for`].
    ///
    /// # Errors
    ///
    /// Returns an [`ApiErrorKind::AuthenticationFailed`] error if none of the
    /// enabled providers' API-key variables are set.
    fn from_env() -> Result<Self> {
        #[cfg(feature = "openai")]
        if std::env::var("OPENAI_API_KEY").is_ok() {
            return Ok(Self::OpenAI(OpenAIClient::from_env()?));
        }
        #[cfg(feature = "anthropic")]
        if std::env::var("ANTHROPIC_API_KEY").is_ok() {
            return Ok(Self::Anthropic(AnthropicClient::from_env()?));
        }
        #[cfg(feature = "gemini")]
        if std::env::var("GEMINI_API_KEY").is_ok() || std::env::var("GOOGLE_API_KEY").is_ok() {
            return Ok(Self::Gemini(GeminiClient::from_env()?));
        }
        #[cfg(feature = "grok")]
        if std::env::var("XAI_API_KEY").is_ok() {
            return Ok(Self::Grok(GrokClient::from_env()?));
        }
        Err(RStructorError::api_error(
            "AnyClient",
            ApiErrorKind::AuthenticationFailed,
        ))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        dispatch!(self, c => c.list_models().await)
    }
}

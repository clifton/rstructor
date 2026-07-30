//! A fluent request builder over any [`LLMClient`].
//!
//! Attach context with `with_system`, images with `with_media`, and tools with
//! `with_tools`, then choose a terminal: `materialize` (structured), `generate`
//! (text), `run` (text, using tools if attached), or — with the `streaming`
//! feature — `materialize_iter` / `materialize_stream` / `generate_stream`.
//! Media-bearing requests must use a non-streaming terminal; streaming terminals
//! yield [`RStructorError::Unsupported`](crate::RStructorError::Unsupported)
//! instead of silently discarding attachments.
//!
//! ```no_run
//! # use rstructor::{OpenAIClient, RequestExt, Instructor};
//! # use serde::{Serialize, Deserialize};
//! # #[derive(Instructor, Serialize, Deserialize)] struct Movie { title: String }
//! # async fn ex() -> Result<(), Box<dyn std::error::Error>> {
//! let client = OpenAIClient::from_env()?;
//! let movie: Movie = client
//!     .with_system("Assume USD; dates as ISO-8601.")
//!     .materialize("Describe Inception")
//!     .await?;
//! # Ok(()) }
//! ```

use serde::de::DeserializeOwned;

use crate::backend::{LLMClient, MaterializeFailure, MaterializeReport, MediaFile};
#[cfg(feature = "streaming")]
use crate::error::RStructorError;
use crate::error::Result;
use crate::model::Instructor;

#[cfg(feature = "streaming")]
const STREAMING_MEDIA_UNSUPPORTED: &str = "streaming requests with media are not supported; \
    remove the media attachment or use a non-streaming request terminal";

/// A fluent request being built against a client. Created via [`RequestExt`].
pub struct Request<'a, C: ?Sized> {
    client: &'a C,
    system: Option<String>,
    media: Vec<MediaFile>,
    #[cfg(feature = "tools")]
    tools: Option<&'a crate::backend::tools::Toolbox>,
    #[cfg(feature = "tools")]
    max_iterations: usize,
}

impl<'a, C: ?Sized> Request<'a, C> {
    fn new(client: &'a C) -> Self {
        Self {
            client,
            system: None,
            media: Vec::new(),
            #[cfg(feature = "tools")]
            tools: None,
            #[cfg(feature = "tools")]
            max_iterations: crate::backend::tools::DEFAULT_MAX_TOOL_ITERATIONS,
        }
    }

    /// Attach system/context instructions.
    ///
    /// Built-in provider clients send this through the provider's first-class
    /// system field or message role for every terminal. This keeps the static
    /// instruction prefix separate from the dynamic user prompt, which also
    /// improves provider prompt-cache reuse. Custom [`LLMClient`]
    /// implementations inherit a backwards-compatible concatenation fallback
    /// unless they override the request hooks on that trait.
    #[must_use]
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Attach media (images, or PDFs where the provider supports them) to the
    /// request. Used by `materialize`, `generate`, and `run`. Streaming
    /// terminals reject attached media because their client APIs are text-only.
    #[must_use]
    pub fn media(mut self, media: impl Into<Vec<MediaFile>>) -> Self {
        self.media = media.into();
        self
    }

    /// Attach a [`Toolbox`](crate::Toolbox); `run` will let the model call its
    /// tools. Requires the `tools` feature.
    #[cfg(feature = "tools")]
    #[must_use]
    pub fn tools(mut self, toolbox: &'a crate::backend::tools::Toolbox) -> Self {
        self.tools = Some(toolbox);
        self
    }

    /// Maximum number of tool round-trips for `run` (default 10).
    #[cfg(feature = "tools")]
    #[must_use]
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
}

impl<C: LLMClient + Sync + ?Sized> Request<'_, C> {
    /// Materialize a structured `T`, applying any attached system context and media.
    pub async fn materialize<T>(self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        self.client
            .materialize_request(self.system.as_deref(), prompt, &self.media)
            .await
    }

    /// Materialize a structured `T` with cumulative usage and every provider attempt.
    pub async fn materialize_with_attempts<T>(
        self,
        prompt: &str,
    ) -> std::result::Result<MaterializeReport<T>, MaterializeFailure>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        self.client
            .materialize_request_with_attempts(self.system.as_deref(), prompt, &self.media)
            .await
    }

    /// Generate raw text, applying any attached system context and media.
    pub async fn generate(self, prompt: &str) -> Result<String> {
        self.client
            .generate_request(self.system.as_deref(), prompt, &self.media)
            .await
    }
}

#[cfg(feature = "streaming")]
impl<'a, C: LLMClient + Sync + ?Sized> Request<'a, C> {
    /// Stream a **list** of structured `T`, yielding each item as soon as it is
    /// fully generated and validated, with any attached system context applied.
    ///
    /// If media is attached, the stream yields one
    /// [`RStructorError::Unsupported`](crate::RStructorError::Unsupported) and
    /// ends without calling the client. Streaming APIs are currently text-only.
    pub fn materialize_iter<T>(self, prompt: &str) -> crate::backend::streaming::ItemStream<'a, T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        if !self.media.is_empty() {
            return crate::backend::streaming::error_stream(RStructorError::Unsupported(
                STREAMING_MEDIA_UNSUPPORTED.to_string(),
            ));
        }
        self.client
            .materialize_iter_request::<T>(self.system, prompt.to_string())
    }

    /// Stream raw text deltas, with any attached system context applied.
    ///
    /// If media is attached, the stream yields one
    /// [`RStructorError::Unsupported`](crate::RStructorError::Unsupported) and
    /// ends without calling the client. Streaming APIs are currently text-only.
    pub fn generate_stream(self, prompt: &str) -> crate::backend::streaming::TextStream<'a> {
        if !self.media.is_empty() {
            return crate::backend::streaming::error_stream(RStructorError::Unsupported(
                STREAMING_MEDIA_UNSUPPORTED.to_string(),
            ));
        }
        self.client
            .generate_stream_request(self.system, prompt.to_string())
    }

    /// Stream a single structured object as its JSON fills in, with any attached
    /// system context applied.
    ///
    /// If media is attached, the stream yields one
    /// [`RStructorError::Unsupported`](crate::RStructorError::Unsupported) and
    /// ends without calling the client. Streaming APIs are currently text-only.
    pub fn materialize_stream<T>(
        self,
        prompt: &str,
    ) -> crate::backend::streaming::ObjectStream<'a, T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        if !self.media.is_empty() {
            return crate::backend::streaming::error_stream(RStructorError::Unsupported(
                STREAMING_MEDIA_UNSUPPORTED.to_string(),
            ));
        }
        self.client
            .materialize_stream_request::<T>(self.system, prompt.to_string())
    }
}

#[cfg(feature = "tools")]
impl<C: crate::backend::tools::ToolRunner + LLMClient + Sync + ?Sized> Request<'_, C> {
    /// Get a text answer, letting the model call attached tools (if any) in a loop
    /// until it produces a final response. Attached media is included in the
    /// initial user turn. With no tools attached this is equivalent to
    /// [`generate`](Self::generate).
    pub async fn run(self, prompt: &str) -> Result<String> {
        match self.tools {
            Some(toolbox) => {
                self.client
                    .run_tool_loop(
                        self.system.as_deref(),
                        prompt,
                        &self.media,
                        toolbox,
                        self.max_iterations,
                    )
                    .await
            }
            None => {
                self.client
                    .generate_request(self.system.as_deref(), prompt, &self.media)
                    .await
            }
        }
    }
}

/// Fluent request entry points, available on every [`LLMClient`].
///
/// `use rstructor::RequestExt;` to call `client.with_system(..)`,
/// `client.with_media(..)`, `client.with_tools(..)`, or `client.request()`.
pub trait RequestExt: LLMClient {
    /// Start an empty request.
    fn request(&self) -> Request<'_, Self> {
        Request::new(self)
    }

    /// Start a request with system/context instructions.
    fn with_system(&self, system: impl Into<String>) -> Request<'_, Self> {
        Request::new(self).system(system)
    }

    /// Start a request with attached media (images, or PDFs where the provider
    /// supports them).
    fn with_media<'a>(&'a self, media: &'a [MediaFile]) -> Request<'a, Self> {
        Request::new(self).media(media.to_vec())
    }

    /// Start a request with a [`Toolbox`](crate::Toolbox); call `.run(prompt)` to
    /// run the agentic loop. Requires the `tools` feature.
    #[cfg(feature = "tools")]
    fn with_tools<'a>(&'a self, toolbox: &'a crate::backend::tools::Toolbox) -> Request<'a, Self> {
        Request::new(self).tools(toolbox)
    }
}

impl<C: LLMClient + ?Sized> RequestExt for C {}

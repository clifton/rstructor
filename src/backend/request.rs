//! A fluent request builder over any [`LLMClient`].
//!
//! Attach context with `with_system`, images with `with_media`, and tools with
//! `with_tools`, then choose a terminal: `materialize` (structured), `generate`
//! (text), `run` (text, using tools if attached), or — with the `streaming`
//! feature — `materialize_iter` / `materialize_stream` / `generate_stream`.
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

use crate::backend::{LLMClient, MediaFile};
use crate::error::Result;
use crate::model::Instructor;

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

    /// Attach system/context instructions, prepended to the prompt (for
    /// `materialize`/`generate`) or sent as the provider's system prompt (for
    /// tool `run`).
    #[must_use]
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Attach media (images, or PDFs where the provider supports them) to the
    /// request. Used by `materialize`, `generate`, and `run`.
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

    /// Prompt with the system context prepended, if any.
    fn combined(&self, prompt: &str) -> String {
        match &self.system {
            Some(system) => format!("{system}\n\n{prompt}"),
            None => prompt.to_string(),
        }
    }
}

impl<C: LLMClient + Sync + ?Sized> Request<'_, C> {
    /// Materialize a structured `T`, applying any attached system context and media.
    pub async fn materialize<T>(self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        let prompt = self.combined(prompt);
        if self.media.is_empty() {
            self.client.materialize(&prompt).await
        } else {
            self.client
                .materialize_with_media(&prompt, &self.media)
                .await
        }
    }

    /// Generate raw text, applying any attached system context and media.
    pub async fn generate(self, prompt: &str) -> Result<String> {
        let prompt = self.combined(prompt);
        if self.media.is_empty() {
            self.client.generate(&prompt).await
        } else {
            self.client.generate_with_media(&prompt, &self.media).await
        }
    }
}

#[cfg(feature = "streaming")]
impl<'a, C: LLMClient + Sync + ?Sized> Request<'a, C> {
    /// Stream a **list** of structured `T`, yielding each item as soon as it is
    /// fully generated and validated, with any attached system context prepended.
    ///
    /// Attached media is ignored — the streaming APIs are text-only.
    pub fn materialize_iter<T>(self, prompt: &str) -> crate::backend::streaming::ItemStream<'a, T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        use futures_util::StreamExt;
        let combined = self.combined(prompt);
        let client = self.client;
        Box::pin(async_stream::try_stream! {
            let mut inner = client.materialize_iter::<T>(&combined);
            while let Some(item) = inner.next().await {
                yield item?;
            }
        })
    }

    /// Stream raw text deltas, with any attached system context prepended.
    pub fn generate_stream(self, prompt: &str) -> crate::backend::streaming::TextStream<'a> {
        use futures_util::StreamExt;
        let combined = self.combined(prompt);
        let client = self.client;
        Box::pin(async_stream::try_stream! {
            let mut inner = client.generate_stream(&combined);
            while let Some(chunk) = inner.next().await {
                yield chunk?;
            }
        })
    }

    /// Stream a single structured object as its JSON fills in, with any attached
    /// system context prepended. Attached media is ignored.
    pub fn materialize_stream<T>(
        self,
        prompt: &str,
    ) -> crate::backend::streaming::ObjectStream<'a, T>
    where
        T: Instructor + DeserializeOwned + Send + 'static,
    {
        use futures_util::StreamExt;
        let combined = self.combined(prompt);
        let client = self.client;
        Box::pin(async_stream::try_stream! {
            let mut inner = client.materialize_stream::<T>(&combined);
            while let Some(obj) = inner.next().await {
                yield obj?;
            }
        })
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
                let prompt = self.combined(prompt);
                if self.media.is_empty() {
                    self.client.generate(&prompt).await
                } else {
                    self.client.generate_with_media(&prompt, &self.media).await
                }
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

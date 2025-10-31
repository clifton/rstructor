pub mod client;
mod utils;

#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "gemini")]
pub mod gemini;
#[cfg(feature = "grok")]
pub mod grok;
#[cfg(feature = "openai")]
pub mod openai;

pub use client::LLMClient;
pub(crate) use utils::{check_response_status, extract_json_from_markdown, handle_http_error};

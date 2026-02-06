//! Chat message types for conversation history support.
//!
//! This module provides types for representing chat messages in a provider-agnostic way,
//! enabling conversation history to be maintained across retry attempts for prompt caching benefits.

use super::client::MediaFile;

/// Role of a chat message participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    /// System message (initial instructions)
    System,
    /// User message (prompts/questions)
    User,
    /// Assistant message (model responses)
    Assistant,
}

impl ChatRole {
    /// Returns the role as a string for API requests.
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        }
    }
}

/// A chat message for conversation history.
///
/// This is a provider-agnostic representation of a chat message that can be
/// converted to the specific format required by each LLM provider.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// The role of the message sender
    pub role: ChatRole,
    /// The message content
    pub content: String,
    /// Media references attached to this message (if supported by provider)
    pub media: Vec<MediaFile>,
}

impl ChatMessage {
    /// Create a new chat message.
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            media: Vec::new(),
        }
    }

    /// Create a user message.
    ///
    /// # Example
    ///
    /// ```
    /// use rstructor::ChatMessage;
    ///
    /// let msg = ChatMessage::user("What is the capital of France?");
    /// assert_eq!(msg.role.as_str(), "user");
    /// ```
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User, content)
    }

    /// Create a user message with attached media references.
    pub fn user_with_media(content: impl Into<String>, media: Vec<MediaFile>) -> Self {
        let mut msg = Self::new(ChatRole::User, content);
        msg.media = media;
        msg
    }

    /// Create an assistant message.
    ///
    /// # Example
    ///
    /// ```
    /// use rstructor::ChatMessage;
    ///
    /// let msg = ChatMessage::assistant("The capital of France is Paris.");
    /// assert_eq!(msg.role.as_str(), "assistant");
    /// ```
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }

    /// Create a system message.
    ///
    /// # Example
    ///
    /// ```
    /// use rstructor::ChatMessage;
    ///
    /// let msg = ChatMessage::system("You are a helpful assistant.");
    /// assert_eq!(msg.role.as_str(), "system");
    /// ```
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System, content)
    }
}

/// Result from a materialize internal call, including the raw response for retry handling.
///
/// This struct captures both the successfully parsed data and the raw response string,
/// which is needed for building conversation history during retries.
#[derive(Debug)]
pub struct MaterializeInternalOutput<T> {
    /// The parsed and validated data
    pub data: T,
    /// The raw response string from the model (JSON or text).
    /// This is stored for debugging and future features, even though it may not be
    /// directly accessed in the success path.
    #[allow(dead_code)]
    pub raw_response: String,
    /// Token usage information if available
    pub usage: Option<crate::backend::TokenUsage>,
}

impl<T> MaterializeInternalOutput<T> {
    /// Create a new output with all fields.
    pub fn new(data: T, raw_response: String, usage: Option<crate::backend::TokenUsage>) -> Self {
        Self {
            data,
            raw_response,
            usage,
        }
    }
}

/// Error context for validation failures that preserves the raw response.
///
/// This allows the retry logic to include the failed response in the conversation
/// history, enabling the model to see what it generated wrong.
#[derive(Debug, Clone)]
pub struct ValidationFailureContext {
    /// The validation error message
    pub error_message: String,
    /// The raw response that failed validation
    pub raw_response: String,
}

impl ValidationFailureContext {
    /// Create a new validation failure context.
    pub fn new(error_message: impl Into<String>, raw_response: impl Into<String>) -> Self {
        Self {
            error_message: error_message.into(),
            raw_response: raw_response.into(),
        }
    }
}

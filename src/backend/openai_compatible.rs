use serde::{Deserialize, Serialize};

use crate::backend::{
    ChatMessage, OpenAICompatibleMessageContent, ResponseFormat,
    build_openai_compatible_message_content,
};
use crate::error::Result;

#[derive(Debug, Serialize)]
pub(crate) struct OpenAICompatibleChatMessage {
    pub role: String,
    pub content: OpenAICompatibleMessageContent,
}

pub(crate) fn convert_openai_compatible_chat_messages(
    messages: &[ChatMessage],
    provider_name: &str,
) -> Result<Vec<OpenAICompatibleChatMessage>> {
    messages
        .iter()
        .map(|msg| {
            Ok(OpenAICompatibleChatMessage {
                role: msg.role.as_str().to_string(),
                content: build_openai_compatible_message_content(msg, provider_name)?,
            })
        })
        .collect()
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAICompatibleChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OpenAICompatibleChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    pub temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Reasoning effort for GPT-5.x models (OpenAI only).
    /// Omitted for providers that don't support it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct OpenAICompatibleResponseMessage {
    pub role: String,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct OpenAICompatibleChatCompletionChoice {
    pub message: OpenAICompatibleResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct OpenAICompatibleUsageInfo {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAICompatibleChatCompletionResponse {
    pub choices: Vec<OpenAICompatibleChatCompletionChoice>,
    #[serde(default)]
    pub usage: Option<OpenAICompatibleUsageInfo>,
    pub model: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MediaFile;

    #[test]
    fn test_convert_openai_compatible_chat_messages_text_only() {
        let messages = vec![ChatMessage::user("hello")];
        let converted = convert_openai_compatible_chat_messages(&messages, "OpenAI")
            .expect("conversion should succeed");

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
        let json = serde_json::to_value(&converted[0]).expect("serialization should succeed");
        assert_eq!(json["content"], serde_json::json!("hello"));
    }

    #[test]
    fn test_convert_openai_compatible_chat_messages_with_media() {
        let messages = vec![ChatMessage::user_with_media(
            "describe image",
            vec![MediaFile::from_bytes(b"abc", "image/png")],
        )];
        let converted = convert_openai_compatible_chat_messages(&messages, "OpenAI")
            .expect("conversion should succeed");

        assert_eq!(converted.len(), 1);
        let json = serde_json::to_value(&converted[0]).expect("serialization should succeed");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][1]["type"], "image_url");
    }
}

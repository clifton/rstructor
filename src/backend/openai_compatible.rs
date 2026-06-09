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
    /// Sampling temperature. `None` omits the key entirely, which is required
    /// for OpenAI o-series reasoning models (they reject `temperature` with a
    /// 400 error).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Completion-token limit for OpenAI o-series reasoning models, which
    /// reject `max_tokens` and require `max_completion_tokens` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Reasoning effort for OpenAI reasoning-capable models (GPT-5.x and the
    /// o-series). Omitted for providers that don't support it.
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

    /// Build a minimal request with all `Option` fields set to `None`.
    fn request_with_none_options() -> OpenAICompatibleChatCompletionRequest {
        OpenAICompatibleChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![OpenAICompatibleChatMessage {
                role: "user".to_string(),
                content: OpenAICompatibleMessageContent::Text("hi".to_string()),
            }],
            response_format: None,
            temperature: None,
            max_tokens: None,
            max_completion_tokens: None,
            reasoning_effort: None,
        }
    }

    /// When `temperature` is `None`, the `temperature` key must be absent from
    /// the serialized request body. o-series reasoning models reject the
    /// parameter with a 400 error, so omitting it must be possible.
    #[test]
    fn test_request_omits_temperature_when_none() {
        let req = request_with_none_options();
        let json = serde_json::to_value(&req).expect("serialization should succeed");

        let obj = json.as_object().expect("request serializes to an object");
        assert!(
            !obj.contains_key("temperature"),
            "temperature key must be omitted when None, got: {json}"
        );
    }

    /// When `temperature` is `Some(..)`, the serialized request body must carry
    /// the numeric value under the `temperature` key.
    #[test]
    fn test_request_includes_temperature_when_some() {
        let mut req = request_with_none_options();
        req.temperature = Some(0.5);
        let json = serde_json::to_value(&req).expect("serialization should succeed");

        assert_eq!(json["temperature"], serde_json::json!(0.5));
    }

    /// When `max_completion_tokens` is `None`, the key must be absent from the
    /// serialized request body.
    #[test]
    fn test_request_omits_max_completion_tokens_when_none() {
        let req = request_with_none_options();
        let json = serde_json::to_value(&req).expect("serialization should succeed");

        let obj = json.as_object().expect("request serializes to an object");
        assert!(
            !obj.contains_key("max_completion_tokens"),
            "max_completion_tokens key must be omitted when None, got: {json}"
        );
    }

    /// When `max_completion_tokens` is `Some(..)`, the serialized request body
    /// must carry the numeric value under the `max_completion_tokens` key
    /// (the limit parameter o-series reasoning models require).
    #[test]
    fn test_request_includes_max_completion_tokens_when_some() {
        let mut req = request_with_none_options();
        req.max_completion_tokens = Some(1024);
        let json = serde_json::to_value(&req).expect("serialization should succeed");

        assert_eq!(json["max_completion_tokens"], serde_json::json!(1024));
    }

    /// When `max_tokens` is `None`, the `max_tokens` key must be absent from the
    /// serialized request body (`skip_serializing_if = "Option::is_none"`).
    #[test]
    fn test_request_omits_max_tokens_when_none() {
        let req = request_with_none_options();
        let json = serde_json::to_value(&req).expect("serialization should succeed");

        let obj = json.as_object().expect("request serializes to an object");
        assert!(
            !obj.contains_key("max_tokens"),
            "max_tokens key must be omitted when None, got: {json}"
        );
    }

    /// When `max_tokens` is `Some(1)`, the serialized request body must carry the
    /// numeric value `1` under the `max_tokens` key.
    #[test]
    fn test_request_includes_max_tokens_when_some() {
        let mut req = request_with_none_options();
        req.max_tokens = Some(1);
        let json = serde_json::to_value(&req).expect("serialization should succeed");

        assert_eq!(json["max_tokens"], serde_json::json!(1));
    }

    /// When `reasoning_effort` is `None`, the key must be absent from the
    /// serialized request body.
    #[test]
    fn test_request_omits_reasoning_effort_when_none() {
        let req = request_with_none_options();
        let json = serde_json::to_value(&req).expect("serialization should succeed");

        let obj = json.as_object().expect("request serializes to an object");
        assert!(
            !obj.contains_key("reasoning_effort"),
            "reasoning_effort key must be omitted when None, got: {json}"
        );
    }

    /// When `reasoning_effort` is `Some(..)`, the serialized request body must
    /// carry the string value under the `reasoning_effort` key.
    #[test]
    fn test_request_includes_reasoning_effort_when_some() {
        let mut req = request_with_none_options();
        req.reasoning_effort = Some("high".to_string());
        let json = serde_json::to_value(&req).expect("serialization should succeed");

        assert_eq!(json["reasoning_effort"], serde_json::json!("high"));
    }

    /// When `response_format` is `None`, the key must be absent from the
    /// serialized request body.
    #[test]
    fn test_request_omits_response_format_when_none() {
        let req = request_with_none_options();
        let json = serde_json::to_value(&req).expect("serialization should succeed");

        let obj = json.as_object().expect("request serializes to an object");
        assert!(
            !obj.contains_key("response_format"),
            "response_format key must be omitted when None, got: {json}"
        );
    }

    /// When `response_format` is `Some(..)`, the serialized request body must
    /// carry the `response_format` object with its `type` discriminant.
    #[test]
    fn test_request_includes_response_format_when_some() {
        let mut req = request_with_none_options();
        req.response_format = Some(ResponseFormat::json_schema(
            "Movie".to_string(),
            serde_json::json!({"type": "object"}),
            None,
        ));
        let json = serde_json::to_value(&req).expect("serialization should succeed");

        assert_eq!(json["response_format"]["type"], "json_schema");
    }

    /// Sanity check: the always-serialized fields (`model`, `messages`) remain
    /// present even when every `Option` field is `None`. `temperature` is no
    /// longer unconditional: it is omitted for o-series reasoning models,
    /// which reject the parameter.
    #[test]
    fn test_request_required_fields_present_with_all_none() {
        let req = request_with_none_options();
        let json = serde_json::to_value(&req).expect("serialization should succeed");
        let obj = json.as_object().expect("request serializes to an object");

        assert!(obj.contains_key("model"), "model must always be present");
        assert!(
            obj.contains_key("messages"),
            "messages must always be present"
        );
    }

    /// `OpenAICompatibleUsageInfo::total_tokens` is `#[serde(default)]`, so a
    /// response body that omits `total_tokens` must deserialize it to `0`.
    #[test]
    fn test_usage_info_total_tokens_defaults_to_zero_when_missing() {
        let json = serde_json::json!({
            "prompt_tokens": 3,
            "completion_tokens": 5,
        });
        let usage: OpenAICompatibleUsageInfo =
            serde_json::from_value(json).expect("deserialization should succeed");

        assert_eq!(usage.prompt_tokens, 3);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 0);
    }

    /// When `total_tokens` is present in the response body it must be preserved
    /// (the `#[serde(default)]` only kicks in for the missing case).
    #[test]
    fn test_usage_info_total_tokens_preserved_when_present() {
        let json = serde_json::json!({
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8,
        });
        let usage: OpenAICompatibleUsageInfo =
            serde_json::from_value(json).expect("deserialization should succeed");

        assert_eq!(usage.total_tokens, 8);
    }
}

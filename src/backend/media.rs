use serde::Serialize;

use crate::backend::ChatMessage;
use crate::error::{ApiErrorKind, RStructorError, Result};

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum OpenAICompatibleMessageContent {
    Text(String),
    Parts(Vec<OpenAICompatibleMessagePart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OpenAICompatibleMessagePart {
    Text { text: String },
    ImageUrl { image_url: OpenAICompatibleImageUrl },
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAICompatibleImageUrl {
    pub(crate) url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) detail: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicContentBlock {
    Text { text: String },
    Image { source: AnthropicImageSource },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

pub(crate) fn build_openai_compatible_message_content(
    msg: &ChatMessage,
    provider_name: &str,
) -> Result<OpenAICompatibleMessageContent> {
    if msg.media.is_empty() {
        return Ok(OpenAICompatibleMessageContent::Text(msg.content.clone()));
    }

    let mut parts = Vec::new();
    if !msg.content.is_empty() {
        parts.push(OpenAICompatibleMessagePart::Text {
            text: msg.content.clone(),
        });
    }

    for media in &msg.media {
        let url = media_to_url(media, provider_name)?;
        parts.push(OpenAICompatibleMessagePart::ImageUrl {
            image_url: OpenAICompatibleImageUrl {
                url,
                detail: Some("auto".to_string()),
            },
        });
    }

    Ok(OpenAICompatibleMessageContent::Parts(parts))
}

pub(crate) fn build_anthropic_message_content(
    msg: &ChatMessage,
) -> Result<AnthropicMessageContent> {
    if msg.media.is_empty() {
        return Ok(AnthropicMessageContent::Text(msg.content.clone()));
    }

    let mut blocks = Vec::new();
    if !msg.content.is_empty() {
        blocks.push(AnthropicContentBlock::Text {
            text: msg.content.clone(),
        });
    }

    for media in &msg.media {
        if let Some(data) = media.data.as_ref() {
            if data.is_empty() {
                return Err(RStructorError::api_error(
                    "Anthropic",
                    ApiErrorKind::BadRequest {
                        details: "MediaFile inline data cannot be empty".to_string(),
                    },
                ));
            }
            if media.mime_type.is_empty() {
                return Err(RStructorError::api_error(
                    "Anthropic",
                    ApiErrorKind::BadRequest {
                        details: "MediaFile mime_type cannot be empty".to_string(),
                    },
                ));
            }
            blocks.push(AnthropicContentBlock::Image {
                source: AnthropicImageSource::Base64 {
                    media_type: media.mime_type.clone(),
                    data: data.clone(),
                },
            });
        } else if !media.uri.is_empty() {
            blocks.push(AnthropicContentBlock::Image {
                source: AnthropicImageSource::Url {
                    url: media.uri.clone(),
                },
            });
        } else {
            return Err(RStructorError::api_error(
                "Anthropic",
                ApiErrorKind::BadRequest {
                    details: "MediaFile must include either inline data or uri".to_string(),
                },
            ));
        }
    }

    Ok(AnthropicMessageContent::Blocks(blocks))
}

fn media_to_url(media: &crate::backend::client::MediaFile, provider_name: &str) -> Result<String> {
    if let Some(data) = media.data.as_ref() {
        if data.is_empty() {
            return Err(RStructorError::api_error(
                provider_name,
                ApiErrorKind::BadRequest {
                    details: "MediaFile inline data cannot be empty".to_string(),
                },
            ));
        }
        if media.mime_type.is_empty() {
            return Err(RStructorError::api_error(
                provider_name,
                ApiErrorKind::BadRequest {
                    details: "MediaFile mime_type cannot be empty".to_string(),
                },
            ));
        }
        Ok(format!("data:{};base64,{}", media.mime_type, data))
    } else if !media.uri.is_empty() {
        Ok(media.uri.clone())
    } else {
        Err(RStructorError::api_error(
            provider_name,
            ApiErrorKind::BadRequest {
                details: "MediaFile must include either inline data or uri".to_string(),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MediaFile;

    #[test]
    fn test_openai_compatible_content_text_only() {
        let msg = ChatMessage::user("hello");
        let content =
            build_openai_compatible_message_content(&msg, "OpenAI").expect("content should build");
        let json = serde_json::to_value(&content).expect("content should serialize");
        assert_eq!(json, serde_json::json!("hello"));
    }

    #[test]
    fn test_openai_compatible_content_with_media() {
        let msg = ChatMessage::user_with_media(
            "describe image",
            vec![MediaFile::from_bytes(b"abc", "image/png")],
        );
        let content =
            build_openai_compatible_message_content(&msg, "OpenAI").expect("content should build");
        let json = serde_json::to_value(&content).expect("content should serialize");
        assert_eq!(json[0]["type"], "text");
        assert_eq!(json[1]["type"], "image_url");
        assert_eq!(json[1]["image_url"]["url"], "data:image/png;base64,YWJj");
    }

    #[test]
    fn test_anthropic_content_text_only() {
        let msg = ChatMessage::user("hello");
        let content = build_anthropic_message_content(&msg).expect("content should build");
        let json = serde_json::to_value(&content).expect("content should serialize");
        assert_eq!(json, serde_json::json!("hello"));
    }

    #[test]
    fn test_anthropic_content_with_inline_media() {
        let msg = ChatMessage::user_with_media(
            "describe image",
            vec![MediaFile::from_bytes(b"abc", "image/png")],
        );
        let content = build_anthropic_message_content(&msg).expect("content should build");
        let json = serde_json::to_value(&content).expect("content should serialize");
        assert_eq!(json[0]["type"], "text");
        assert_eq!(json[1]["type"], "image");
        assert_eq!(json[1]["source"]["type"], "base64");
        assert_eq!(json[1]["source"]["media_type"], "image/png");
        assert_eq!(json[1]["source"]["data"], "YWJj");
    }
}

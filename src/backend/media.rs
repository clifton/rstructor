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
    File { file: OpenAICompatibleFile },
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAICompatibleImageUrl {
    pub(crate) url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) detail: Option<String>,
}

/// A file content part for OpenAI chat completions (PDF input).
///
/// See <https://platform.openai.com/docs/guides/pdf-files>: the part is
/// `{"type": "file", "file": {"filename": ..., "file_data": "data:application/pdf;base64,..."}}`.
#[derive(Debug, Serialize)]
pub(crate) struct OpenAICompatibleFile {
    pub(crate) filename: String,
    pub(crate) file_data: String,
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
    Text {
        text: String,
    },
    Image {
        source: AnthropicMediaSource,
    },
    /// A PDF document block, see
    /// <https://docs.anthropic.com/en/docs/build-with-claude/pdf-support>:
    /// `{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": ...}}`
    /// or `{"type": "document", "source": {"type": "url", "url": ...}}`.
    Document {
        source: AnthropicMediaSource,
    },
}

/// Source of an Anthropic `image` or `document` block (both share this shape).
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicMediaSource {
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
        if media.mime_type.starts_with("image/") {
            let url = media_to_url(media, provider_name)?;
            parts.push(OpenAICompatibleMessagePart::ImageUrl {
                image_url: OpenAICompatibleImageUrl {
                    url,
                    detail: Some("auto".to_string()),
                },
            });
        } else if media.mime_type == "application/pdf" {
            parts.push(openai_compatible_pdf_part(media, provider_name)?);
        } else {
            return Err(unsupported_media_type(media, provider_name));
        }
    }

    Ok(OpenAICompatibleMessageContent::Parts(parts))
}

/// Build the PDF content part for an OpenAI-compatible chat completions request,
/// or a clear error for providers/sources without a documented PDF pathway.
fn openai_compatible_pdf_part(
    media: &crate::backend::client::MediaFile,
    provider_name: &str,
) -> Result<OpenAICompatibleMessagePart> {
    // xAI's chat completions API only documents text and image content parts
    // (https://docs.x.ai/docs/guides/image-understanding); there is no file or
    // document part, so sending a PDF would be silently mislabeled or rejected.
    if provider_name == "Grok" {
        return Err(RStructorError::api_error(
            provider_name,
            ApiErrorKind::BadRequest {
                details: "PDF attachments are not supported for Grok: the xAI chat \
                          completions API only accepts text and image content parts. \
                          Extract the PDF's text or render its pages to images and \
                          attach those instead"
                    .to_string(),
            },
        ));
    }

    if let Some(data) = media.data.as_ref() {
        if data.is_empty() {
            return Err(RStructorError::api_error(
                provider_name,
                ApiErrorKind::BadRequest {
                    details: "MediaFile inline data cannot be empty".to_string(),
                },
            ));
        }
        // Chat completions accept PDFs as a `file` part with base64 `file_data`
        // (https://platform.openai.com/docs/guides/pdf-files).
        Ok(OpenAICompatibleMessagePart::File {
            file: OpenAICompatibleFile {
                filename: "document.pdf".to_string(),
                file_data: format!("data:{};base64,{}", media.mime_type, data),
            },
        })
    } else if !media.uri.is_empty() {
        // Chat completions do not accept remote file URLs (only `file_data` or an
        // uploaded `file_id`); see https://platform.openai.com/docs/guides/pdf-files.
        Err(RStructorError::api_error(
            provider_name,
            ApiErrorKind::BadRequest {
                details: format!(
                    "{provider_name} chat completions does not support URL-based PDF \
                     attachments; download the file and attach the bytes inline with \
                     MediaFile::from_bytes(bytes, \"application/pdf\") instead"
                ),
            },
        ))
    } else {
        Err(RStructorError::api_error(
            provider_name,
            ApiErrorKind::BadRequest {
                details: "MediaFile must include either inline data or uri".to_string(),
            },
        ))
    }
}

/// Error for MIME types with no documented attachment pathway on this provider.
fn unsupported_media_type(
    media: &crate::backend::client::MediaFile,
    provider_name: &str,
) -> RStructorError {
    let supported = if provider_name == "Grok" {
        "image/*"
    } else {
        "image/* and application/pdf"
    };
    RStructorError::api_error(
        provider_name,
        ApiErrorKind::BadRequest {
            details: format!(
                "unsupported media type {:?} for {provider_name}: only {supported} \
                 attachments are supported on this provider",
                media.mime_type,
            ),
        },
    )
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
        let is_image = media.mime_type.starts_with("image/");
        let is_pdf = media.mime_type == "application/pdf";

        let source = if let Some(data) = media.data.as_ref() {
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
            AnthropicMediaSource::Base64 {
                media_type: media.mime_type.clone(),
                data: data.clone(),
            }
        } else if !media.uri.is_empty() {
            AnthropicMediaSource::Url {
                url: media.uri.clone(),
            }
        } else {
            return Err(RStructorError::api_error(
                "Anthropic",
                ApiErrorKind::BadRequest {
                    details: "MediaFile must include either inline data or uri".to_string(),
                },
            ));
        };

        if is_image {
            blocks.push(AnthropicContentBlock::Image { source });
        } else if is_pdf {
            // PDFs go in a `document` block, never an `image` block; see
            // https://docs.anthropic.com/en/docs/build-with-claude/pdf-support.
            blocks.push(AnthropicContentBlock::Document { source });
        } else {
            return Err(unsupported_media_type(media, "Anthropic"));
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

    #[test]
    fn test_anthropic_content_with_url_image() {
        let msg = ChatMessage::user_with_media(
            "describe image",
            vec![MediaFile::new("https://example.com/cat.png", "image/png")],
        );
        let content = build_anthropic_message_content(&msg).expect("content should build");
        let json = serde_json::to_value(&content).expect("content should serialize");
        assert_eq!(json[1]["type"], "image");
        assert_eq!(
            json[1]["source"],
            serde_json::json!({"type": "url", "url": "https://example.com/cat.png"})
        );
    }

    // ---- PDF routing: OpenAI ----

    #[test]
    fn test_openai_inline_pdf_becomes_file_part() {
        let msg = ChatMessage::user_with_media(
            "summarize",
            vec![MediaFile::from_bytes(b"%PDF", "application/pdf")],
        );
        let content =
            build_openai_compatible_message_content(&msg, "OpenAI").expect("content should build");
        let json = serde_json::to_value(&content).expect("content should serialize");
        assert_eq!(json[0]["type"], "text");
        assert_eq!(
            json[1],
            serde_json::json!({
                "type": "file",
                "file": {
                    "filename": "document.pdf",
                    "file_data": "data:application/pdf;base64,JVBERg==",
                }
            })
        );
    }

    #[test]
    fn test_openai_url_pdf_is_a_clear_error() {
        let msg = ChatMessage::user_with_media(
            "summarize",
            vec![MediaFile::new(
                "https://example.com/report.pdf",
                "application/pdf",
            )],
        );
        let err = build_openai_compatible_message_content(&msg, "OpenAI")
            .expect_err("URL-based PDFs are not supported by chat completions");
        let text = err.to_string();
        assert!(
            text.contains("URL-based PDF") && text.contains("MediaFile::from_bytes"),
            "error should explain the fix, got: {text}"
        );
    }

    #[test]
    fn test_openai_non_image_non_pdf_is_a_clear_error() {
        let msg = ChatMessage::user_with_media(
            "transcribe",
            vec![MediaFile::from_bytes(b"abc", "audio/mpeg")],
        );
        let err = build_openai_compatible_message_content(&msg, "OpenAI")
            .expect_err("audio attachments have no chat-completions pathway");
        let text = err.to_string();
        assert!(
            text.contains("unsupported media type") && text.contains("audio/mpeg"),
            "error should name the offending MIME type, got: {text}"
        );
    }

    // ---- PDF routing: Grok ----

    #[test]
    fn test_grok_inline_pdf_is_a_clear_error_not_an_image_url() {
        let msg = ChatMessage::user_with_media(
            "summarize",
            vec![MediaFile::from_bytes(b"%PDF", "application/pdf")],
        );
        let err = build_openai_compatible_message_content(&msg, "Grok")
            .expect_err("Grok has no documented PDF pathway");
        let text = err.to_string();
        assert!(
            text.contains("PDF attachments are not supported for Grok"),
            "error should say PDFs are unsupported on Grok, got: {text}"
        );
    }

    #[test]
    fn test_grok_url_pdf_is_a_clear_error() {
        let msg = ChatMessage::user_with_media(
            "summarize",
            vec![MediaFile::new(
                "https://example.com/report.pdf",
                "application/pdf",
            )],
        );
        let err = build_openai_compatible_message_content(&msg, "Grok")
            .expect_err("Grok has no documented PDF pathway");
        assert!(
            err.to_string()
                .contains("PDF attachments are not supported for Grok")
        );
    }

    #[test]
    fn test_grok_images_still_use_image_url() {
        let msg = ChatMessage::user_with_media(
            "describe image",
            vec![MediaFile::from_bytes(b"abc", "image/jpeg")],
        );
        let content =
            build_openai_compatible_message_content(&msg, "Grok").expect("content should build");
        let json = serde_json::to_value(&content).expect("content should serialize");
        assert_eq!(json[1]["type"], "image_url");
        assert_eq!(json[1]["image_url"]["url"], "data:image/jpeg;base64,YWJj");
    }

    // ---- PDF routing: Anthropic ----

    #[test]
    fn test_anthropic_inline_pdf_becomes_document_block() {
        let msg = ChatMessage::user_with_media(
            "summarize",
            vec![MediaFile::from_bytes(b"%PDF", "application/pdf")],
        );
        let content = build_anthropic_message_content(&msg).expect("content should build");
        let json = serde_json::to_value(&content).expect("content should serialize");
        assert_eq!(json[0]["type"], "text");
        assert_eq!(
            json[1],
            serde_json::json!({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERg==",
                }
            })
        );
    }

    #[test]
    fn test_anthropic_url_pdf_becomes_url_document_block() {
        let msg = ChatMessage::user_with_media(
            "summarize",
            vec![MediaFile::new(
                "https://example.com/report.pdf",
                "application/pdf",
            )],
        );
        let content = build_anthropic_message_content(&msg).expect("content should build");
        let json = serde_json::to_value(&content).expect("content should serialize");
        assert_eq!(
            json[1],
            serde_json::json!({
                "type": "document",
                "source": {
                    "type": "url",
                    "url": "https://example.com/report.pdf",
                }
            })
        );
    }

    #[test]
    fn test_anthropic_non_image_non_pdf_is_a_clear_error() {
        let msg = ChatMessage::user_with_media(
            "transcribe",
            vec![MediaFile::from_bytes(b"abc", "audio/mpeg")],
        );
        let err = build_anthropic_message_content(&msg)
            .expect_err("audio attachments have no Messages API pathway");
        let text = err.to_string();
        assert!(
            text.contains("unsupported media type") && text.contains("audio/mpeg"),
            "error should name the offending MIME type, got: {text}"
        );
    }
}

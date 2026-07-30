//! Provider response regressions for prompt-cache usage accounting.
//!
//! Fixtures mirror the usage fields documented by each provider. Cache reads
//! and writes remain subsets of total input usage, allowing callers to inspect
//! cache effectiveness without double-counting billed tokens.

#![cfg(feature = "_client")]

use rstructor::LLMClient;

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_reports_cache_reads_and_writes() {
    let mut server = mockito::Server::new_async().await;
    let response = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(include_str!(
            "fixtures/prompt_caching/openai_cache_hit.json"
        ))
        .expect(1)
        .create_async()
        .await;

    let result = rstructor::OpenAIClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gpt-5.6")
        .generate_with_metadata("Summarize today's risk.")
        .await
        .unwrap();

    let usage = result.usage.unwrap();
    assert_eq!(usage.model, "gpt-5.6");
    assert_eq!(usage.input_tokens, 2048);
    assert_eq!(usage.cached_input_tokens, 1024);
    assert_eq!(usage.cache_write_input_tokens, 768);
    assert_eq!(usage.output_tokens, 125);
    assert_eq!(usage.total_tokens(), 2173);
    response.assert_async().await;
}

#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_reports_automatic_cache_hits() {
    let mut server = mockito::Server::new_async().await;
    let response = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(include_str!("fixtures/prompt_caching/grok_cache_hit.json"))
        .expect(1)
        .create_async()
        .await;

    let result = rstructor::GrokClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("grok-4.1-fast")
        .generate_with_metadata("Summarize today's risk.")
        .await
        .unwrap();

    let usage = result.usage.unwrap();
    assert_eq!(usage.model, "grok-4.1-fast");
    assert_eq!(usage.input_tokens, 3072);
    assert_eq!(usage.cached_input_tokens, 2048);
    assert_eq!(usage.cache_write_input_tokens, 0);
    assert_eq!(usage.output_tokens, 125);
    assert_eq!(usage.total_tokens(), 3197);
    response.assert_async().await;
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_includes_cache_reads_and_writes_in_total_input() {
    let mut server = mockito::Server::new_async().await;
    let response = server
        .mock("POST", "/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(include_str!(
            "fixtures/prompt_caching/anthropic_cache_hit.json"
        ))
        .expect(1)
        .create_async()
        .await;

    let result = rstructor::AnthropicClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("claude-opus-5-20260701")
        .generate_with_metadata("Summarize today's risk.")
        .await
        .unwrap();

    let usage = result.usage.unwrap();
    assert_eq!(usage.model, "claude-opus-5-20260701");
    assert_eq!(usage.input_tokens, 3104);
    assert_eq!(usage.cached_input_tokens, 1024);
    assert_eq!(usage.cache_write_input_tokens, 2048);
    assert_eq!(usage.output_tokens, 125);
    assert_eq!(usage.total_tokens(), 3229);
    response.assert_async().await;
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_reports_implicit_cache_hits() {
    let mut server = mockito::Server::new_async().await;
    let response = server
        .mock("POST", "/models/gemini-3.1-pro-preview:generateContent")
        .match_query(mockito::Matcher::UrlEncoded(
            "key".to_string(),
            "test-key".to_string(),
        ))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(include_str!(
            "fixtures/prompt_caching/gemini_cache_hit.json"
        ))
        .expect(1)
        .create_async()
        .await;

    let result = rstructor::GeminiClient::new("test-key")
        .unwrap()
        .base_url(server.url())
        .model("gemini-3.1-pro-preview")
        .generate_with_metadata("Summarize today's risk.")
        .await
        .unwrap();

    let usage = result.usage.unwrap();
    assert_eq!(usage.model, "gemini-3.1-pro-preview");
    assert_eq!(usage.input_tokens, 3072);
    assert_eq!(usage.cached_input_tokens, 2048);
    assert_eq!(usage.cache_write_input_tokens, 0);
    assert_eq!(usage.output_tokens, 125);
    assert_eq!(usage.total_tokens(), 3197);
    response.assert_async().await;
}

//! Live streaming tests. Only compiled with `--features streaming` (not a default
//! feature), so the default `cargo test` run does not hit the network.
#![cfg(feature = "streaming")]

use futures_util::StreamExt;
use rstructor::{Instructor, LLMClient, StreamedObject};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Movie {
    title: String,
    year: u16,
    director: String,
}

/// Drive an object stream to completion, returning (partial_count, final_value).
async fn drive<C: LLMClient + Sync>(client: &C, prompt: &str) -> (usize, Movie) {
    let mut stream = client.materialize_stream::<Movie>(prompt);
    let mut partials = 0usize;
    let mut complete: Option<Movie> = None;
    while let Some(item) = stream.next().await {
        match item.expect("stream item should not error") {
            StreamedObject::Partial(value) => {
                assert!(value.is_object(), "partial should be a JSON object");
                partials += 1;
            }
            StreamedObject::Complete(movie) => complete = Some(movie),
        }
    }
    (partials, complete.expect("stream should end with Complete"))
}

const PROMPT: &str = "Describe the movie Inception: title, year, director.";

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_text_stream_yields_text() {
    use rstructor::OpenAIClient;
    let client = OpenAIClient::from_env().unwrap().model("gpt-4.1-mini");
    let mut stream = client.generate_stream("Say hello in exactly three words.");
    let mut chunks = 0usize;
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        text.push_str(&item.expect("stream item should not error"));
        chunks += 1;
    }
    assert!(chunks >= 1, "expected at least one streamed chunk");
    assert!(!text.trim().is_empty());
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_object_stream() {
    use rstructor::OpenAIClient;
    let client = OpenAIClient::from_env().unwrap().model("gpt-4.1-mini");
    let (partials, movie) = drive(&client, PROMPT).await;
    assert!(!movie.title.trim().is_empty());
    assert!(movie.year > 1900, "unexpected year: {}", movie.year);
    assert!(partials >= 1, "expected partial snapshots, got {partials}");
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_object_stream() {
    use rstructor::AnthropicClient;
    let client = AnthropicClient::from_env()
        .unwrap()
        .model("claude-haiku-4-5-20251001");
    let (_partials, movie) = drive(&client, PROMPT).await;
    assert!(!movie.title.trim().is_empty());
    assert!(movie.year > 1900, "unexpected year: {}", movie.year);
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_object_stream() {
    use rstructor::GeminiClient;
    let client = GeminiClient::from_env().unwrap().model("gemini-2.5-flash");
    let (_partials, movie) = drive(&client, PROMPT).await;
    assert!(!movie.title.trim().is_empty());
    assert!(movie.year > 1900, "unexpected year: {}", movie.year);
}

#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_object_stream() {
    use rstructor::GrokClient;
    let client = GrokClient::from_env().unwrap();
    let (_partials, movie) = drive(&client, PROMPT).await;
    assert!(!movie.title.trim().is_empty());
    assert!(movie.year > 1900, "unexpected year: {}", movie.year);
}

// ---- materialize_iter: stream a list, one validated item at a time ----

const LIST_PROMPT: &str = "List 3 acclaimed movies, each with title, year, and director.";

async fn collect_iter<C: LLMClient + Sync>(client: &C) -> Vec<Movie> {
    let mut stream = client.materialize_iter::<Movie>(LIST_PROMPT);
    let mut movies = Vec::new();
    while let Some(item) = stream.next().await {
        movies.push(item.expect("iter item should not error"));
    }
    movies
}

fn assert_movie_list(movies: &[Movie]) {
    assert!(
        movies.len() >= 2,
        "expected several streamed movies, got {}",
        movies.len()
    );
    assert!(
        movies
            .iter()
            .all(|m| !m.title.trim().is_empty() && m.year > 1900),
        "every streamed movie should be valid: {movies:?}"
    );
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_materialize_iter() {
    use rstructor::OpenAIClient;
    let client = OpenAIClient::from_env().unwrap().model("gpt-4.1-mini");
    assert_movie_list(&collect_iter(&client).await);
}

#[cfg(feature = "grok")]
#[tokio::test]
async fn grok_materialize_iter() {
    use rstructor::GrokClient;
    let client = GrokClient::from_env().unwrap();
    assert_movie_list(&collect_iter(&client).await);
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_materialize_iter() {
    use rstructor::AnthropicClient;
    let client = AnthropicClient::from_env()
        .unwrap()
        .model("claude-haiku-4-5-20251001")
        .max_tokens(2048);
    assert_movie_list(&collect_iter(&client).await);
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_materialize_iter() {
    use rstructor::GeminiClient;
    let client = GeminiClient::from_env().unwrap().model("gemini-2.5-flash");
    assert_movie_list(&collect_iter(&client).await);
}

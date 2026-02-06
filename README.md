# rstructor: Structured LLM Outputs for Rust

<p align="center">
  <a href="https://crates.io/crates/rstructor"><img src="https://img.shields.io/crates/v/rstructor" alt="crates.io"/></a>
  <a href="https://crates.io/crates/rstructor"><img src="https://img.shields.io/crates/d/rstructor" alt="downloads"/></a>
  <a href="https://github.com/clifton/rstructor/actions"><img src="https://github.com/clifton/rstructor/actions/workflows/test.yml/badge.svg" alt="CI"/></a>
  <img src="https://img.shields.io/badge/rust-2024-orange" alt="Rust 2024"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"/>
</p>

Extract structured, validated data from LLMs using native Rust types. Define your schema as structs/enums, and rstructor handles JSON Schema generation, API communication, parsing, and validation.

The Rust equivalent of [Instructor](https://github.com/jxnl/instructor) for Python.

## Features

- **Type-safe schemas** — Define models as Rust structs/enums with derive macros
- **Multi-provider** — OpenAI, Anthropic, Grok (xAI), and Gemini with unified API
- **Auto-validation** — Type checking plus custom business rules with automatic retry
- **Complex types** — Nested objects, arrays, optionals, enums with associated data
- **Extended thinking** — Native support for reasoning models (GPT-5.2, Claude 4.5, Gemini 3)

## Installation

```toml
[dependencies]
rstructor = "0.2"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["rt-multi-thread", "macros"] }
```

## Quick Start

```rust
use rstructor::{Instructor, LLMClient, OpenAIClient};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Movie {
    #[llm(description = "Title of the movie")]
    title: String,
    #[llm(description = "Director of the movie")]
    director: String,
    #[llm(description = "Year released", example = 2010)]
    year: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAIClient::from_env()?
        .temperature(0.0);

    let movie: Movie = client.materialize("Tell me about Inception").await?;
    println!("{}: {} ({})", movie.title, movie.director, movie.year);
    Ok(())
}
```

## Providers

```rust
use rstructor::{OpenAIClient, AnthropicClient, GrokClient, GeminiClient, LLMClient};

// OpenAI (reads OPENAI_API_KEY)
let client = OpenAIClient::from_env()?.model("gpt-5.2");

// Anthropic (reads ANTHROPIC_API_KEY)
let client = AnthropicClient::from_env()?.model("claude-opus-4-6");

// Grok/xAI (reads XAI_API_KEY)
let client = GrokClient::from_env()?.model("grok-4-1-fast-non-reasoning");

// Gemini (reads GEMINI_API_KEY)
let client = GeminiClient::from_env()?.model("gemini-3-flash-preview");

// Custom endpoint (local LLMs, proxies)
let client = OpenAIClient::new("key")?
    .base_url("http://localhost:1234/v1")
    .model("llama-3.1-70b");
```

## Validation

Add custom validation with automatic retry on failure:

```rust
use rstructor::{Instructor, RStructorError, Result};

#[derive(Instructor, Serialize, Deserialize)]
#[llm(validate = "validate_movie")]
struct Movie {
    title: String,
    year: u16,
    rating: f32,
}

fn validate_movie(movie: &Movie) -> Result<()> {
    if movie.year < 1888 || movie.year > 2030 {
        return Err(RStructorError::ValidationError(
            format!("Invalid year: {}", movie.year)
        ));
    }
    if movie.rating < 0.0 || movie.rating > 10.0 {
        return Err(RStructorError::ValidationError(
            format!("Rating must be 0-10, got {}", movie.rating)
        ));
    }
    Ok(())
}

// Retries are enabled by default (3 attempts with error feedback)
// To increase retries:
let client = OpenAIClient::from_env()?.max_retries(5);

// To disable retries:
let client = OpenAIClient::from_env()?.no_retries();
```

## Complex Types

### Nested Structures

```rust
#[derive(Instructor, Serialize, Deserialize)]
struct Ingredient {
    name: String,
    amount: f32,
    unit: String,
}

#[derive(Instructor, Serialize, Deserialize)]
struct Recipe {
    name: String,
    ingredients: Vec<Ingredient>,
    prep_time_minutes: u16,
}
```

### Enums with Data

```rust
#[derive(Instructor, Serialize, Deserialize)]
enum PaymentMethod {
    #[llm(description = "Credit card payment")]
    Card { number: String, expiry: String },
    #[llm(description = "PayPal account")]
    PayPal(String),
    #[llm(description = "Cash on delivery")]
    CashOnDelivery,
}
```

### Serde Rename Support

rstructor respects `#[serde(rename)]` and `#[serde(rename_all)]` attributes:

```rust
#[derive(Instructor, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UserProfile {
    first_name: String,      // becomes "firstName" in schema
    last_name: String,       // becomes "lastName" in schema
    email_address: String,   // becomes "emailAddress" in schema
}

#[derive(Instructor, Serialize, Deserialize)]
struct CommitMessage {
    #[serde(rename = "type")]  // use "type" as JSON key
    commit_type: String,
    description: String,
}

#[derive(Instructor, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum CommitType {
    Fix,       // becomes "fix"
    Feat,      // becomes "feat"
    Refactor,  // becomes "refactor"
}
```

Supported case conversions: `lowercase`, `UPPERCASE`, `camelCase`, `PascalCase`, `snake_case`, `SCREAMING_SNAKE_CASE`, `kebab-case`, `SCREAMING-KEBAB-CASE`.

### Custom Types (Dates, UUIDs)

```rust
use chrono::{DateTime, Utc};
use rstructor::schema::CustomTypeSchema;

impl CustomTypeSchema for DateTime<Utc> {
    fn schema_type() -> &'static str { "string" }
    fn schema_format() -> Option<&'static str> { Some("date-time") }
}

#[derive(Instructor, Serialize, Deserialize)]
struct Event {
    name: String,
    start_time: DateTime<Utc>,
}
```

## Multimodal (Image Input)

Analyze images with structured extraction using Gemini's inline data support:

```rust
use rstructor::{Instructor, LLMClient, GeminiClient, MediaFile};

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct ImageAnalysis {
    subject: String,
    colors: Vec<String>,
    is_logo: bool,
    description: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Download or load image bytes
    let image_bytes = reqwest::get("https://example.com/image.png")
        .await?.bytes().await?;

    // Create inline media from bytes (base64-encoded automatically)
    let media = MediaFile::from_bytes(&image_bytes, "image/png");

    let client = GeminiClient::from_env()?;
    let analysis: ImageAnalysis = client
        .materialize_with_media("Describe this image", &[media])
        .await?;
    println!("{:?}", analysis);
    Ok(())
}
```

`MediaFile::new(uri, mime_type)` is also available for Gemini Files API / GCS URIs.

## Extended Thinking

Configure reasoning depth for supported models:

```rust
use rstructor::ThinkingLevel;

// GPT-5.2, Claude 4.5 (Sonnet/Opus), Gemini 3
let client = OpenAIClient::from_env()?
    .model("gpt-5.2")
    .thinking_level(ThinkingLevel::High);

// Levels: Off, Minimal, Low, Medium, High
```

## Token Usage

```rust
let result = client.materialize_with_metadata::<Movie>("...").await?;
println!("Movie: {}", result.data.title);
if let Some(usage) = result.usage {
    println!("Tokens: {} in, {} out", usage.input_tokens, usage.output_tokens);
}
```

## Error Handling

```rust
use rstructor::{ApiErrorKind, RStructorError};

match client.materialize::<Movie>("...").await {
    Ok(movie) => println!("{:?}", movie),
    Err(e) if e.is_retryable() => {
        println!("Transient error: {}", e);
        if let Some(delay) = e.retry_delay() {
            tokio::time::sleep(delay).await;
        }
    }
    Err(e) => match e.api_error_kind() {
        Some(ApiErrorKind::RateLimited { retry_after }) => { /* ... */ }
        Some(ApiErrorKind::AuthenticationFailed) => { /* ... */ }
        _ => eprintln!("Error: {}", e),
    }
}
```

## Feature Flags

```toml
[dependencies]
rstructor = { version = "0.2", features = ["openai", "anthropic", "grok", "gemini"] }
```

- `openai`, `anthropic`, `grok`, `gemini` — Provider backends
- `derive` — Derive macro (default)
- `logging` — Tracing integration

## Examples

See `examples/` for complete working examples:

```bash
export OPENAI_API_KEY=your_key
cargo run --example structured_movie_info
cargo run --example nested_objects_example
cargo run --example enum_with_data_example
cargo run --example serde_rename_example
cargo run --example gemini_multimodal_example
```

## For Python Developers

If you're coming from Python and searching for:
- **"pydantic rust"** or **"rust pydantic"** — rstructor provides similar schema validation and type safety
- **"instructor rust"** or **"rust instructor"** — same structured LLM output extraction pattern
- **"structured output rust"** or **"llm structured output"** — exactly what rstructor does
- **"type-safe llm rust"** — ensures type safety from LLM responses to Rust structs

## License

MIT — see [LICENSE](LICENSE)

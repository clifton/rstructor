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
- **Extended thinking** — Native support for reasoning models (GPT-5.x, Claude 4.x, Gemini 3)

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
        .temperature(0.0)
        .max_retries(3);

    let movie: Movie = client.materialize("Tell me about Inception").await?;
    println!("{}: {} ({})", movie.title, movie.director, movie.year);
    Ok(())
}
```

## Providers

```rust
use rstructor::{OpenAIClient, AnthropicClient, GrokClient, GeminiClient, LLMClient};

// OpenAI (reads OPENAI_API_KEY)
let client = OpenAIClient::from_env()?.model("gpt-4o");

// Anthropic (reads ANTHROPIC_API_KEY)
let client = AnthropicClient::from_env()?.model("claude-sonnet-4-5-20250929");

// Grok/xAI (reads XAI_API_KEY)
let client = GrokClient::from_env()?.model("grok-3");

// Gemini (reads GEMINI_API_KEY)
let client = GeminiClient::from_env()?.model("gemini-2.5-flash");

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

// Configure retry behavior
let client = OpenAIClient::from_env()?
    .max_retries(3)
    .include_error_feedback(true);  // Send error context on retry
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

## Extended Thinking

Configure reasoning depth for supported models:

```rust
use rstructor::ThinkingLevel;

// GPT-5.x, Claude 4.x, Gemini 3
let client = OpenAIClient::from_env()?
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
```

## For Python Developers

If you're coming from Python and searching for:
- **"pydantic rust"** or **"rust pydantic"** — rstructor provides similar schema validation and type safety
- **"instructor rust"** or **"rust instructor"** — same structured LLM output extraction pattern
- **"structured output rust"** or **"llm structured output"** — exactly what rstructor does
- **"type-safe llm rust"** — ensures type safety from LLM responses to Rust structs

## License

MIT — see [LICENSE](LICENSE)

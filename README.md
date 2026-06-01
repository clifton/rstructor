# rstructor: Structured LLM Outputs for Rust

<p align="center">
  <a href="https://crates.io/crates/rstructor"><img src="https://img.shields.io/crates/v/rstructor" alt="crates.io"/></a>
  <a href="https://crates.io/crates/rstructor"><img src="https://img.shields.io/crates/d/rstructor" alt="downloads"/></a>
  <a href="https://github.com/clifton/rstructor/actions"><img src="https://github.com/clifton/rstructor/actions/workflows/test.yml/badge.svg" alt="CI"/></a>
  <img src="https://img.shields.io/badge/rust-2024-orange" alt="Rust 2024"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"/>
</p>

Get structured, validated data out of any LLM as native Rust structs and enums. Define the shape you want as plain Rust types — rstructor generates the JSON Schema, prompts the model, parses the response, and retries on validation errors until the data fits.

## Features

- **Type-safe schemas from Rust types** — Derive `Instructor` on structs and enums; rstructor generates the JSON Schema and validated parser for you, no hand-written prompts or DTOs
- **Multi-provider, one API** — OpenAI, Anthropic, Grok (xAI), and Gemini behind a single `materialize()` call with swappable clients
- **Validation with automatic re-ask** — Built-in type checking plus custom business rules; validation failures are fed back to the model and retried until the data is correct
- **Rich, nested data** — Nested objects, arrays, optionals, maps, and enums with associated data, with validation that recurses through the whole tree
- **Familiar if you know Pydantic + Instructor** — The same structured-output workflow as Python's [Instructor](https://github.com/jxnl/instructor) + [Pydantic](https://github.com/pydantic/pydantic), with Rust's compile-time type safety

## Installation

```toml
[dependencies]
rstructor = "0.3"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["rt-multi-thread", "macros"] }
```

## Quick Start

Describe the shape you want as plain Rust types, then turn a line of free-form text into a fully-typed, validated value:

```rust
use rstructor::{Instructor, LLMClient, OpenAIClient};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
enum Priority {
    Low,
    Medium,
    High,
    Urgent,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "A support ticket triaged from a free-form message")]
struct Ticket {
    #[llm(description = "Short, imperative summary of what needs to be done")]
    title: String,
    #[llm(description = "How urgent this is, inferred from tone and deadlines")]
    priority: Priority,
    #[llm(description = "Email of the person on it, or null if unassigned")]
    assignee: Option<String>,
    #[llm(description = "Relevant topic tags", examples = ["billing", "auth", "outage"])]
    tags: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAIClient::from_env()?.temperature(0.0);

    let ticket: Ticket = client
        .materialize(
            "Hey, the login page is throwing 500s for half our users since the deploy. \
             Sarah (sarah@acme.io) is on it but we need this fixed before the demo at 3pm!",
        )
        .await?;

    println!("{ticket:#?}");
    // Ticket {
    //     title: "Login page returning 500 errors after deploy",
    //     priority: Urgent,
    //     assignee: Some("sarah@acme.io"),
    //     tags: ["auth", "outage"],
    // }
    Ok(())
}
```

Every field is *inferred*, not transcribed: the urgency is read from the tone and deadline, the email is plucked out of mid-sentence text, and the tags are synthesized — all parsed into the exact types you declared.

## Request Builder

`materialize`, `generate`, and (with the `tools` feature) tool `run` are also
available through a fluent builder that attaches context, images, and tools to a
single request. Bring `RequestExt` into scope and chain the pieces you need:

```rust
use rstructor::{Instructor, OpenAIClient, RequestExt};

let client = OpenAIClient::from_env()?;

// Add context that is prepended to the prompt, then materialize a struct.
let movie: Movie = client
    .with_system("Assume USD; format dates as ISO-8601.")
    .materialize("Describe Inception")
    .await?;

// Or start from `.request()` and combine builders before a terminal.
let summary = client
    .request()
    .system("Be concise.")
    .generate("Summarize the plot of Inception")
    .await?;
```

The terminals are `materialize::<T>(prompt)` (structured), `generate(prompt)`
(text), and — with the `tools` feature — `run(prompt)` (text, calling any
attached tools in a loop). Builders compose: `with_system`, `with_media`, and
`with_tools` can be chained in any order before the terminal.

## Providers

```rust
use rstructor::{OpenAIClient, AnthropicClient, GrokClient, GeminiClient, LLMClient};

// OpenAI (reads OPENAI_API_KEY)
let client = OpenAIClient::from_env()?.model("gpt-5.5");

// Anthropic (reads ANTHROPIC_API_KEY)
let client = AnthropicClient::from_env()?.model("claude-sonnet-4-6");

// Grok/xAI (reads XAI_API_KEY)
let client = GrokClient::from_env()?.model("grok-4.3");

// Gemini (reads GEMINI_API_KEY)
let client = GeminiClient::from_env()?.model("gemini-3.5-flash");

// Custom endpoint (local LLMs, proxies)
let client = OpenAIClient::new("key")?
    .base_url("http://localhost:1234/v1")
    .model("llama-3.1-70b");
```

### Selecting a provider at runtime

`LLMClient::materialize` is generic, so the trait isn't object-safe (`Box<dyn LLMClient>` is impossible). Use `AnyClient` when the provider is decided at runtime (CLI flag, config, env) and you want to store it in a single type:

```rust
use rstructor::{AnyClient, Provider, LLMClient};

// Pick a provider dynamically, reading its key from the environment.
let provider = Provider::Anthropic; // e.g. parsed from a config file
let client = AnyClient::from_env_for(provider)?;
let movie: Movie = client.materialize("Describe Inception").await?;

// Or auto-detect from whichever API key is set:
let client = AnyClient::from_env()?;

// Or wrap a pre-configured client:
let client: AnyClient = OpenAIClient::from_env()?.model("gpt-5.5").into();
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

### Dates, UUIDs, and Custom Types

```rust
use chrono::{DateTime, NaiveDate, Utc};
use rstructor::Instructor;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Instructor, Serialize, Deserialize)]
struct JobRun {
    id: Uuid,                         // schema format: "uuid"
    trade_date: NaiveDate,            // schema format: "date"
    started_at: DateTime<Utc>,        // schema format: "date-time"
    parent_id: Option<Uuid>,          // optional UUID keeps format metadata
    related_ids: Vec<Uuid>,           // array items keep format metadata
}
```

For your own domain-specific scalar types, implement `CustomTypeSchema` plus `SchemaType`:

```rust
use rstructor::schema::CustomTypeSchema;
use rstructor::{Schema, SchemaType};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct SecurityId(String);

impl CustomTypeSchema for SecurityId {
    fn schema_type() -> &'static str { "string" }
    fn schema_format() -> Option<&'static str> { Some("security-id") }
}

impl SchemaType for SecurityId {
    fn schema() -> Schema { Schema::new(Self::json_schema()) }
    fn schema_name() -> Option<String> { Some("SecurityId".to_string()) }
}
```

## Multimodal (Image Input)

Analyze images with structured extraction across all major providers by
attaching media to a request with `with_media`:

```rust
use rstructor::{Instructor, OpenAIClient, MediaFile, RequestExt};

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct ImageAnalysis {
    subject: String,
    summary: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Download or load image bytes (real-world fixture)
    let image_bytes = reqwest::get("https://example.com/image.png")
        .await?.bytes().await?;

    // Inline media is base64-encoded automatically
    let media = [MediaFile::from_bytes(&image_bytes, "image/png")];

    // Works with OpenAI, Anthropic, Grok, and Gemini clients
    let client = OpenAIClient::from_env()?;
    let analysis: ImageAnalysis = client
        .with_media(&media)
        .materialize("Describe this image")
        .await?;
    println!("{:?}", analysis);
    Ok(())
}
```

`MediaFile::new(uri, mime_type)` is also available for URL/URI-based media input.
The lower-level `LLMClient::materialize_with_media(prompt, &media)` method does
the same thing in one call when you do not need the builder.

Provider examples:
- `cargo run --example openai_multimodal_example --features openai`
- `cargo run --example anthropic_multimodal_example --features anthropic`
- `cargo run --example grok_multimodal_example --features grok`
- `cargo run --example gemini_multimodal_example --features gemini`

## Extended Thinking

Configure reasoning depth for supported models:

```rust
use rstructor::ThinkingLevel;

// GPT-5.5, Claude 4.6 Sonnet, Gemini 3.1
let client = OpenAIClient::from_env()?
    .model("gpt-5.5")
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

## Streaming

Enable the `streaming` feature to stream responses as they are generated.

```toml
rstructor = { version = "0.3", features = ["streaming"] }
```

`materialize_iter` streams a **list of structured objects**, yielding each item as soon as it is fully generated and validated — the common case where you want a long list without buffering the whole response:

```rust
use futures_util::StreamExt;
use rstructor::{LLMClient, OpenAIClient, Instructor};

let client = OpenAIClient::from_env()?;
let mut stream = client.materialize_iter::<Invention>("List 10 important inventions.");

while let Some(item) = stream.next().await {
    let invention = item?;          // each item: fully parsed + validated
    println!("{} ({})", invention.name, invention.year);
}
```

`generate_stream` streams raw text deltas:

```rust
let mut stream = client.generate_stream("Write a haiku");
while let Some(chunk) = stream.next().await {
    print!("{}", chunk?);
}
```

There is also `materialize_stream`, which streams a single object as progressive `StreamedObject::Partial(json)` snapshots followed by a validated `Complete(T)`.

All are available on every provider (OpenAI, Anthropic, Grok, Gemini). See `examples/streaming_example.rs`.

## Tool Calling

Enable the `tools` feature to let the model call your typed Rust functions and feed the results back, looping until it produces a final answer. Tool argument types derive `Instructor`, so their JSON Schema is generated automatically.

```toml
rstructor = { version = "0.3", features = ["tools"] }
```

```rust
use rstructor::{OpenAIClient, Toolbox, FnTool, Instructor};
use serde::{Serialize, Deserialize};
use serde_json::json;

#[derive(Instructor, Serialize, Deserialize)]
struct WeatherArgs {
    #[llm(description = "City name")]
    city: String,
}

let toolbox = Toolbox::new().with(FnTool::new(
    "get_weather",
    "Get the current weather for a city",
    |args: WeatherArgs| async move {
        Ok(json!({ "city": args.city, "temp_f": 72 }))   // call a real API here
    },
));

let client = OpenAIClient::from_env()?;
let answer = client
    .with_tools(&toolbox)
    .system("Use tools when relevant.")   // optional
    .run("What's the weather in Paris?")
    .await?;
```

Works with all providers (OpenAI, Anthropic, Grok, Gemini). See `examples/tool_calling_example.rs`.

## Feature Flags

```toml
[dependencies]
rstructor = { version = "0.3", features = ["openai", "anthropic", "grok", "gemini"] }
```

- `openai`, `anthropic`, `grok`, `gemini` — Provider backends (each pulls in the shared HTTP/`tokio` stack)
- `derive` — Derive macro (default)
- `logging` — Tracing integration
- `streaming` — Streaming via `generate_stream` / `materialize_iter` / `materialize_stream` (opt-in)
- `tools` — Tool/function calling via `Toolbox` + `client.with_tools(..).run(..)` (opt-in)

All features are on by default. For a **schema-only build** — generate JSON Schema from your types with no networking, `tokio`, or `reqwest` — disable the providers:

```toml
[dependencies]
rstructor = { version = "0.3", default-features = false, features = ["derive"] }
```

This keeps the derive macro, `SchemaType`, the `Instructor` trait, and the `LLMClient` trait (so you can implement your own backend) without the async/HTTP dependency tree.

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

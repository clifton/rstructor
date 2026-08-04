# rstructor: Structured LLM Outputs for Rust

<p align="center">
  <a href="https://crates.io/crates/rstructor"><img src="https://img.shields.io/crates/v/rstructor" alt="crates.io"/></a>
  <a href="https://crates.io/crates/rstructor"><img src="https://img.shields.io/crates/d/rstructor" alt="downloads"/></a>
  <a href="https://github.com/clifton/rstructor/actions"><img src="https://github.com/clifton/rstructor/actions/workflows/test.yml/badge.svg" alt="CI"/></a>
  <img src="https://img.shields.io/badge/rust-2024-orange" alt="Rust 2024"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"/>
</p>

Get structured, validated data from supported LLM providers as native Rust structs and enums. Define the shape you want as plain Rust types — rstructor generates the JSON Schema, prompts the model, parses the response, and either returns a validated value or a typed error after the configured retries.

## Features

- **Type-safe schemas from Rust types** — Derive `Instructor` on structs and enums; rstructor generates the JSON Schema and validated parser for you, no hand-written prompts or DTOs
- **Multi-provider, one API** — OpenAI, Anthropic, Grok (xAI), and Gemini behind a single `extract()` call with swappable clients
- **Validation with automatic re-ask** — Built-in type checking plus custom business rules; validation failures are fed back to the model and retried within an explicit bound
- **Rich, nested data** — Nested objects, arrays, optionals, maps, and enums with associated data, with validation that recurses through the whole tree
- **Familiar if you know Pydantic + Instructor** — The same structured-output workflow as Python's [Instructor](https://github.com/jxnl/instructor) + [Pydantic](https://github.com/pydantic/pydantic), with Rust's compile-time type safety

## Installation

```toml
[dependencies]
rstructor = "0.5.1"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["rt-multi-thread", "macros"] }
```

## 60-Second Extraction

Set the API key for your provider—for this example, `OPENAI_API_KEY`—then
describe the shape you want as plain Rust types. One `extract` call turns
free-form text into a fully typed, validated value:

```rust
use rstructor::{Instructor, LLMClient};
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
    let text = "Hey, the login page is throwing 500s for half our users since the deploy. \
                Sarah (sarah@acme.io) is on it but we need this fixed before the demo at 3pm!";
    let ticket: Ticket = rstructor::client("openai/gpt-5.6-sol")?
        .extract(text)
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

If you want provider-specific configuration, use the explicit client form:

```rust
use rstructor::{LLMClient, OpenAIClient};

let client = OpenAIClient::from_env()?.temperature(0.0);
let ticket: Ticket = client.extract(text).await?;
```

`materialize` is retained as an equivalent name for existing code. New code
can use `extract` when that better describes the operation.

### Already using schemars?

Enable rstructor's `schemars` feature and wrap your existing
`JsonSchema + Serialize + Deserialize` model; no `Instructor` derive is needed:

```rust
let ticket = client
    .extract::<rstructor::Schemars<Ticket>>(text)
    .await?
    .into_inner();
```

Nested schemas and doc-comment descriptions are inlined automatically.
Recursive schemars models are rejected before a provider request because v1 of
the bridge intentionally sends only reference-free schemas.

## Request Builder

`extract`, `generate`, and (with the `tools` feature) tool `run` are also
available through a fluent builder that attaches context, images, and tools to a
single request. Bring `RequestExt` into scope and chain the pieces you need:

```rust
use rstructor::{Instructor, OpenAIClient, RequestExt};

let client = OpenAIClient::from_env()?;

// Keep stable instructions separate from the dynamic user prompt.
let movie: Movie = client
    .with_system("Assume USD; format dates as ISO-8601.")
    .extract("Describe Inception")
    .await?;

// Or start from `.request()` and combine builders before a terminal.
let summary = client
    .request()
    .system("Be concise.")
    .generate("Summarize the plot of Inception")
    .await?;
```

The primary terminals are `extract::<T>(prompt)` (structured), `generate(prompt)`
(text), and — with the `tools` feature — `run(prompt)` (text, calling any
attached tools in a loop). Builders compose: `with_system`, `with_media`, and
`with_tools` can be chained in any order before the terminal. The original
`materialize::<T>(prompt)` name remains an equivalent structured terminal.

### System prompts and prompt caching

Built-in clients send `with_system(...)` through each provider's native
instruction channel for every request terminal:

- OpenAI-compatible APIs and xAI receive an initial `system` message.
- Anthropic receives the top-level `system` field.
- Gemini receives `systemInstruction`.

This preserves instruction semantics and keeps a stable prefix eligible for the
provider's prompt cache instead of merging it into each changing user message.
It applies to structured materialization, raw generation, streaming, media
requests, retry ledgers, and `run` with or without tools.

Put stable policy and examples in the system prompt, then keep request-specific
data in the user prompt:

```rust
use rstructor::{LLMClient, OpenAIClient, RequestExt};

const RISK_POLICY: &str =
    "Use the fund's base currency. Report exposure as a multiple of NAV.";

let client = OpenAIClient::from_env()?;
let extraction = client
    .with_system(RISK_POLICY)
    .extract_with_report::<Portfolio>(daily_positions)
    .await?;

if let Some(usage) = extraction.report.cumulative_usage {
    println!(
        "{} of {} input tokens came from cache",
        usage.cached_input_tokens,
        usage.input_tokens,
    );
}
```

OpenAI, Gemini, and xAI can apply implicit prefix caching when a request meets
their model and token thresholds. Anthropic requires cache-control configuration
to create a cache, which rstructor does not enable implicitly because it can
change billing. Likewise, rstructor does not currently create Gemini explicit
cache objects or set OpenAI cache-routing keys. Provider-reported cache reads and
writes are exposed as `cached_input_tokens` and `cache_write_input_tokens`;
both are subsets of `input_tokens` and are not added again by `total_tokens()`.
See the official [OpenAI](https://developers.openai.com/api/docs/guides/prompt-caching),
[Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-caching),
[Gemini](https://ai.google.dev/gemini-api/docs/caching), and
[xAI](https://docs.x.ai/developers/advanced-api-usage/prompt-caching)
prompt-caching guides for current eligibility and retention rules.

## Recipes

Start with the task you need, then open the linked runnable example. The
[task-oriented cookbook](docs/COOKBOOK.md) expands the core workflows into
complete copy-paste recipes.

| I want to… | Example | What it shows |
|---|---|---|
| Extract typed data from free text | [`structured_movie_info.rs`](examples/structured_movie_info.rs) | Turns one sentence into a validated Rust struct with field descriptions and business rules. |
| Classify into an enum | [`news_article_categorizer.rs`](examples/news_article_categorizer.rs) | Selects a typed category while extracting sentiment, entities, and keywords. |
| Extract from an image or PDF | [`openai_multimodal_example.rs`](examples/openai_multimodal_example.rs) ([Anthropic](examples/anthropic_multimodal_example.rs), [Gemini](examples/gemini_multimodal_example.rs), [Grok](examples/grok_multimodal_example.rs)) | Sends inline media with a prompt and materializes the answer into a struct. |
| Read a chart with Kimi K3 | [`kimi_k3_multimodal_example.rs`](examples/kimi_k3_multimodal_example.rs) | Downloads a labeled revenue chart, sends it through Moonshot's OpenAI-compatible endpoint, and returns typed values plus calculated insights. |
| Put extraction behind an axum handler | [`axum_handler_example.rs`](examples/axum_handler_example.rs) | Injects any `LLMClient` into typed JSON request and response handling, tested in-process. |
| Test without network | [`mock_testing_example.rs`](examples/mock_testing_example.rs) | Scripts realistic responses through the real deserialization, validation, and re-ask path. |
| Use a local model with Ollama | [`ollama_local_example.rs`](examples/ollama_local_example.rs) | Connects to the keyless local endpoint through the same structured-output API. |
| Choose a provider at runtime | [`runtime_provider_example.rs`](examples/runtime_provider_example.rs) | Parses `provider/model` into one `AnyClient`, including aggregator model IDs with slashes. |
| Reuse an existing schemars model | [`schemars_bridge_example.rs`](examples/schemars_bridge_example.rs) | Materializes `JsonSchema + Serde` types through the transparent `Schemars<T>` adapter. |
| Call tools in a loop | [`tool_calling_example.rs`](examples/tool_calling_example.rs) | Runs schema-validated tool calls until the model produces its final answer. |
| Stream partial output | [`streaming_example.rs`](examples/streaming_example.rs) | Yields validated list items incrementally and explains completion integrity. |
| Add custom validation and re-ask | [`validation_example.rs`](examples/validation_example.rs) | Rejects domain-invalid output with a custom validator that providers can retry. |
| Inspect retries and token cost | [`retry_attempt_ledger.rs`](examples/retry_attempt_ledger.rs) | Reports every attempt, disposition, per-response usage, and cumulative known usage. |
| Model nested or recursive schemas | [`nested_objects_example.rs`](examples/nested_objects_example.rs), [`recursive_schema_graph.rs`](examples/recursive_schema_graph.rs) | Builds deeply nested values and finite `$defs` graphs for recursive domain types. |

## Providers

```rust
use rstructor::{OpenAIClient, AnthropicClient, GrokClient, GeminiClient, LLMClient};

// OpenAI (reads OPENAI_API_KEY)
let client = OpenAIClient::from_env()?.model("gpt-5.6-sol");

// Anthropic (reads ANTHROPIC_API_KEY)
let client = AnthropicClient::from_env()?.model("claude-opus-5");

// Grok/xAI (reads XAI_API_KEY)
let client = GrokClient::from_env()?.model("grok-4.5");

// Gemini (reads GEMINI_API_KEY)
let client = GeminiClient::from_env()?.model("gemini-3.6-flash");

// Local Ollama (no API key)
let client = OpenAIClient::ollama()?.model("llama3.3");
```

### Local models & aggregators

Ollama and LM Studio use the OpenAI-compatible client without an API key or
`Authorization` header:

```rust
use rstructor::{LLMClient, OpenAIClient};

let local = rstructor::client("ollama/llama3.3")?;
let movie: Movie = local.materialize("Describe Inception").await?;

// The named constructor is equivalent and supports all normal builders.
let local = OpenAIClient::lm_studio()?.model("your-loaded-model");
```

Hosted aggregators read their own keys instead of `OPENAI_API_KEY`. Model IDs
may contain `/`; only the first slash separates the route prefix from the model:

```rust
use rstructor::{LLMClient, MediaFile, OpenAIClient};

// Reads MOONSHOT_API_KEY. Kimi K3 fixes temperature at 1.0.
let image = MediaFile::from_bytes(chart_bytes, "image/png");
let kimi = OpenAIClient::moonshot()?
    .model("kimi-k3")
    .temperature(1.0);
let report: RevenueChart = kimi
    .materialize_with_media("Read every labeled bar and calculate the total.", &[image])
    .await?;

// Reads OPENROUTER_API_KEY.
let router = rstructor::client("openrouter/moonshotai/kimi-k3")?;
let movie: Movie = router.materialize("Describe Inception").await?;

// GROQ_API_KEY works the same way.
let groq = rstructor::client("groq/openai/gpt-oss-120b")?;
```

See the runnable
[`kimi_k3_multimodal_example.rs`](examples/kimi_k3_multimodal_example.rs) for
the complete chart schema and output. Moonshot documents
[`kimi-k3`](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart) as a native
vision model with strict JSON Schema output. Public image URLs are not supported,
so attach base64 bytes as above (or use a Moonshot `ms://` file ID). Supported
image types are JPEG, PNG, GIF, WebP, BMP, HEIC, and HEIF; SVG is rejected, and
Moonshot recommends no more than 4096×2160 resolution.

The named constructors are `OpenAIClient::ollama()`, `lm_studio()`,
`openrouter()`, `groq()`, and `moonshot()`. They all require the `openai` Cargo
feature. Strict `response_format` / JSON Schema support varies between compatible
servers; rstructor keeps the same schema request and validation/re-ask retry
loop, without endpoint-specific schema-dialect rewriting.

### Selecting a provider at runtime

`LLMClient::materialize` is generic, so the trait isn't object-safe (`Box<dyn LLMClient>` is impossible). Use `AnyClient` when the provider is decided at runtime (CLI flag, config, env) and you want to store it in a single type:

```rust
use rstructor::{AnyClient, Provider, LLMClient};

// Parse a case-insensitive "provider/model" string.
let client = rstructor::client("anthropic/claude-opus-5")?;
let movie: Movie = client.materialize("Describe Inception").await?;

// Or auto-detect in deterministic environment-key order:
let client = rstructor::client_from_env()?;

// Pick a provider dynamically, reading its key from the environment.
let provider = Provider::Anthropic; // e.g. parsed from a config file
let client = AnyClient::from_env_for(provider)?;
let movie: Movie = client.materialize("Describe Inception").await?;

// The equivalent trait-level constructor is also available:
let client = AnyClient::from_env()?;

// Or wrap a pre-configured client:
let client: AnyClient = OpenAIClient::from_env()?.model("gpt-5.6-sol").into();
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

// Retries are enabled by default (3 retries, 4 total attempts)
// To increase retries:
let client = OpenAIClient::from_env()?.max_retries(5);

// To disable retries:
let client = OpenAIClient::from_env()?.no_retries();
```

### Derive attributes

The `llm` attribute accepts a small, checked API:

- Structs and enums: `description`, `title`, `examples`, `validate`
- Fields: `description`, `example`, `examples`
- Enum variants: `description`

Examples use native Rust expressions. Use `serde_json::json!` for object values
instead of embedding serialized JSON strings. Multi-value examples accept both
`examples = [one, two]` and `examples(one, two)`.

Optionality comes from the Rust type itself:

```rust
#[derive(Instructor, Serialize, Deserialize)]
#[llm(
    description = "A portfolio allocation",
    examples = [::serde_json::json!({
        "name": "Long quality",
        "benchmark": null
    })]
)]
struct Allocation {
    #[llm(description = "Strategy name", example = "Long quality")]
    name: String,

    // No `#[llm(optional)]` marker is needed.
    #[llm(description = "Optional benchmark ticker")]
    benchmark: Option<String>,
}
```

Unknown `llm` keys, malformed values, invalid validation paths, and unsupported
tuple or unit structs are compile errors at the relevant attribute or item.
Serde attributes outside the schema subset remain owned by Serde and are not
rejected by `Instructor`.

## Complex Types

### Dynamic Maps

Use `HashMap<String, V>` when keys are runtime data such as ticker symbols,
account IDs, or category names. The map values remain fully typed:

```rust
use std::collections::HashMap;

use rstructor::{GeminiClient, Instructor, LLMClient};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Position {
    asset_class: String,
    quantity: i64,
    mark_price: f64,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Portfolio {
    portfolio_id: String,
    positions: HashMap<String, Position>,
}

let client = GeminiClient::from_env()?;
let portfolio: Portfolio = client
    .materialize(
        "AAPL: 125,000 equity shares at 213.76; \
         ESU6: short 240 futures at 6378.25",
    )
    .await?;

assert_eq!(portfolio.positions["AAPL"].quantity, 125_000);
assert_eq!(portfolio.positions["ESU6"].quantity, -240);
```

Gemini accepts native typed dynamic-map schemas. OpenAI, Anthropic, and Grok
strict structured-output dialects cannot currently represent arbitrary keys
without weakening or changing the Rust contract. Those clients return a
non-retryable `SchemaCompatibilityError` before making an HTTP request instead
of silently constraining the map to `{}`:

```rust
use rstructor::{OpenAIClient, RStructorError};

let result = OpenAIClient::from_env()?
    .materialize::<Portfolio>("Extract the positions")
    .await;

match result {
    Err(RStructorError::SchemaCompatibilityError {
        provider,
        path,
        ..
    }) => {
        assert_eq!(provider.as_ref(), "OpenAI");
        assert_eq!(path.as_ref(), "$.properties.positions");
    }
    other => panic!("expected a map compatibility error, got {other:?}"),
}
```

Use a struct instead when the keys are a fixed part of the contract. Dynamic-map
fallback encodings are deliberately not selected automatically because doing so
would change the provider wire schema and structured-output guarantee.

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

Derived schemas also support direct and mutual recursion. Recursive definitions
are hoisted to the document root, and concrete Rust type identity keeps
same-named types from different modules or generic instantiations distinct:

```rust
#[derive(Instructor, Serialize, Deserialize)]
struct Fund {
    lei: String,
    prime_broker: Option<Box<PrimeBroker>>,
}

#[derive(Instructor, Serialize, Deserialize)]
struct PrimeBroker {
    lei: String,
    sponsored_funds: Vec<Fund>,
}

let schema = Fund::schema().to_json();
assert!(schema["$defs"].is_object());
```

OpenAI, Anthropic, and Grok preserve those recursive references in their
structured-output schemas. Gemini cannot currently represent an unbounded
recursive schema. Its client returns a local, non-retryable compatibility error
instead of silently replacing the deepest recursive branch with `{}`:

```rust
use rstructor::{GeminiClient, LLMClient, RStructorError};

let result = GeminiClient::from_env()?
    .materialize::<Fund>("Extract the fund and prime-broker hierarchy")
    .await;

assert!(matches!(
    result,
    Err(RStructorError::SchemaCompatibilityError { provider, .. })
        if provider.as_ref() == "Gemini"
));
```

Run the complete example with
`cargo run --example recursive_schema_graph`.

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

### Serde Deserialization Support

rstructor interprets supported Serde metadata from the deserialization side of
the wire contract. It respects `rename`, `rename_all`, `rename_all_fields`,
`skip`, and `skip_deserializing`. When Serde has separate directions, the
deserialize-side name is used:

```rust
#[derive(Instructor, Serialize, Deserialize)]
#[serde(rename_all(serialize = "snake_case", deserialize = "camelCase"))]
struct BrokerOrder {
    // Omitted from the input schema; Serde supplies String::default().
    #[serde(skip_deserializing)]
    server_timestamp: String,

    // Included because it is accepted during deserialization.
    #[serde(skip_serializing)]
    client_order_id: String,

    #[serde(rename(serialize = "symbol", deserialize = "ticker"))]
    instrument: String,
}
```

For symmetric names, the usual shorthand remains unchanged:

```rust
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

## Multimodal (Image & PDF Input)

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
Attached media is honored by `extract`, `generate`, and tool `run` alike. The
request builder is the primary way to combine media with system instructions or
run reporting.

PDFs are supported too: pass `"application/pdf"` as the MIME type and the
attachment is routed to each provider's documented document format (OpenAI
`file` part, Anthropic `document` block, Gemini `inlineData`/`fileData`).
Combinations a provider does not support — PDFs on Grok, or URL-based PDFs on
OpenAI chat completions — return a clear error instead of a broken request.

Provider examples:
- `cargo run --example openai_multimodal_example --features openai`
- `cargo run --example anthropic_multimodal_example --features anthropic`
- `cargo run --example grok_multimodal_example --features grok`
- `cargo run --example gemini_multimodal_example --features gemini`

## Extended Thinking

Configure reasoning depth for supported models:

```rust
use rstructor::ThinkingLevel;

// GPT-5.6 and Gemini 3.6 use named effort levels; Claude 4.x uses
// extended-thinking token budgets.
let client = OpenAIClient::from_env()?
    .model("gpt-5.6-sol")
    .thinking_level(ThinkingLevel::High);

// Levels: Off, Minimal, Low, Medium, High
```

## Extraction Reports and Token Usage

Use `extract_with_report` when retry cost or failure observability matters. The
same `ExtractionReport` shape is attached to success and failure, including an
ordered attempt ledger and cumulative known usage from provider responses that
failed decoding or validation:

```rust
match client.extract_with_report::<Portfolio>("Reconcile the fund book").await {
    Ok(extraction) => {
        println!("Portfolio: {:?}", extraction.data);
        println!("Provider attempts: {}", extraction.report.attempts.len());
        if let Some(usage) = extraction.report.cumulative_usage {
            println!("Known run tokens: {}", usage.total_tokens());
            for (model, model_usage) in usage.by_model {
                println!("{model}: {} tokens", model_usage.total_tokens());
            }
        }
    }
    Err(failure) => {
        eprintln!("Final error: {}", failure.error());
        eprintln!("Attempts made: {}", failure.report.attempts.len());
        if let Some(usage) = failure.report.cumulative_usage {
            eprintln!("Known tokens before failure: {}", usage.total_tokens());
        }
    }
}
```

Usage is conservative: attempts remain in the ledger when a provider omits
token metadata, while cumulative totals include only responses with reported
usage. Local schema/media preflight failures record zero provider attempts.
Cache read and write counters are provider-reported subsets of input usage, so
they provide cache observability without inflating `total_tokens()`.
Built-in clients and `MockClient` set `report.attempts_complete` to `true`; the
compatibility fallback for custom clients sets it to `false` rather than
inventing provider attempts it cannot observe.
See `examples/retry_attempt_ledger.rs` for a complete success-and-failure
example.

## Error Handling

```rust
use rstructor::{ApiErrorKind, RStructorError};

match client.extract::<Movie>("...").await {
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
rstructor = { version = "0.5.1", features = ["streaming"] }
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

Streaming uses strict integrity checks by default. A stream can yield validated
items and later report malformed provider data or a truncated response, so the
full collection is authoritative only after the stream drains to `None` without
an error. OpenAI/Grok must send `[DONE]`, Anthropic must send `message_stop`, and
Gemini must provide a non-empty `finishReason`.

For irreversible side effects, stage items until clean completion and handle the
machine-readable error kind:

```rust
use rstructor::{RStructorError, StreamErrorKind};

let mut staged = Vec::new();
while let Some(item) = stream.next().await {
    match item {
        Ok(invention) => staged.push(invention),
        Err(RStructorError::StreamingError {
            kind: StreamErrorKind::IncompleteEventStream,
            ..
        }) => {
            staged.clear(); // do not commit a possibly truncated collection
            return Err("provider stream ended without its terminal event".into());
        }
        Err(error) => return Err(error.into()),
    }
}
commit_inventions(staged).await?;
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

Streaming terminals are currently text-only. A fluent request with attached
media returns one `Unsupported` stream error instead of silently dropping the
attachments:

```rust
use futures_util::StreamExt;
use rstructor::{MediaFile, RequestExt, RStructorError};

let media = [MediaFile::new("https://example.com/chart.png", "image/png")];
let mut stream = client.with_media(&media).generate_stream("Analyze this chart");

assert!(matches!(
    stream.next().await,
    Some(Err(RStructorError::Unsupported(_)))
));
assert!(stream.next().await.is_none());
```

Use `generate_with_media` or `materialize_with_media` when attachments are
required.

## Tool Calling

Enable the `tools` feature to let the model call your typed Rust functions and feed the results back, looping until it produces a final answer. Tool argument types derive `Instructor`, so their JSON Schema is generated automatically.

```toml
rstructor = { version = "0.5.1", features = ["tools"] }
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

## Testing (offline)

Enable the `mock` feature to unit-test code that extracts structured data without any
network or API key. `MockClient` implements `LLMClient`, so it drops into any
`C: LLMClient` slot; scripted responses flow through the **real** deserialize +
`validate()` path, so you can test schema/validation failures, not just happy paths.

```toml
[dev-dependencies]
rstructor = { version = "0.5.1", features = ["mock"] }
```

```rust
use rstructor::{Instructor, LLMClient, MockClient};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Movie { title: String, year: u16 }

// Your code under test is generic over the client:
async fn extract<C: LLMClient + Sync>(client: &C) -> rstructor::Result<Movie> {
    client.materialize("Describe Inception").await
}

#[tokio::test]
async fn extracts_a_movie() {
    let client = MockClient::new().with_response(r#"{"title": "Inception", "year": 2010}"#);
    let movie = extract(&client).await.unwrap();
    assert_eq!(movie.title, "Inception");
    // Every call is recorded for assertions:
    assert_eq!(client.last_request().unwrap().schema_name.as_deref(), Some("Movie"));
}
```

Script multiple responses with `with_response`/`with_responses` (a FIFO queue), branch
on the request with `with_responder`, simulate the validation re-ask loop with
`with_retries`, attach final/default token usage with `with_usage`, or attach
per-attempt usage with `with_response_and_usage`. Assert on captured requests via
`requests()` / `last_request()`. `RequestKind` is non-exhaustive, so downstream
matches should include a wildcard arm as new client terminals are added. The `mock`
feature pulls in only the lightweight path-aware decoder and works without the HTTP
client; streaming and tool-loop mocking light up when the `streaming` / `tools`
features are also enabled. See `examples/mock_testing_example.rs`.

## Feature Flags

```toml
[dependencies]
rstructor = { version = "0.5.1", features = ["openai", "anthropic", "grok", "gemini"] }
```

- `openai`, `anthropic`, `grok`, `gemini` — Provider backends (each pulls in the shared HTTP/`tokio` stack)
- `derive` — Derive macro (default)
- `logging` — Tracing integration
- `streaming` — Streaming via `generate_stream` / `materialize_iter` / `materialize_stream` (opt-in)
- `tools` — Tool/function calling via `Toolbox` + `client.with_tools(..).run(..)` (opt-in)
- `mock` — `MockClient` for offline unit testing (opt-in; see [Testing](#testing-offline))

All features are on by default. For a **schema-only build** — generate JSON Schema from your types with no networking, `tokio`, or `reqwest` — disable the providers:

```toml
[dependencies]
rstructor = { version = "0.5.1", default-features = false, features = ["derive"] }
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
cargo run --example retry_attempt_ledger --features openai
```

## For Python Developers

If you're coming from Python and searching for:
- **"pydantic rust"** or **"rust pydantic"** — rstructor provides similar schema validation and type safety
- **"instructor rust"** or **"rust instructor"** — same structured LLM output extraction pattern
- **"structured output rust"** or **"llm structured output"** — exactly what rstructor does
- **"type-safe llm rust"** — ensures type safety from LLM responses to Rust structs

## License

MIT — see [LICENSE](LICENSE)

# rstructor cookbook

These recipes start from a task instead of a provider. Every snippet is a
complete binary, and every section links to a runnable example that CI builds
and executes. Unless a section says otherwise, use the default rstructor
features plus:

```toml
[dependencies]
rstructor = "0.4"
serde = { version = "1", features = ["derive"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Provider examples read the API-key environment variable named in the recipe.
Offline examples use `MockClient` and never touch the network.

## Extract typed data from an image or PDF

The runnable
[OpenAI multimodal example](../examples/openai_multimodal_example.rs) uses an
image; equivalent examples cover
[Anthropic](../examples/anthropic_multimodal_example.rs),
[Gemini](../examples/gemini_multimodal_example.rs), and
[Grok](../examples/grok_multimodal_example.rs). `MediaFile::from_bytes` also
accepts PDF bytes for OpenAI, Anthropic, and Gemini. Grok rejects PDFs during
preflight because its documented media path is image-only.

For an OpenAI-compatible hosted endpoint, the
[Kimi K3 chart example](../examples/kimi_k3_multimodal_example.rs) downloads a
labeled revenue chart, attaches the PNG bytes through
`OpenAIClient::moonshot()`, and materializes every bar plus aggregate insights.
Moonshot's exact model ID is
[`kimi-k3`](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart); it accepts
base64 image data but not public image URLs. Supported formats are JPEG, PNG,
GIF, WebP, BMP, HEIC, and HEIF (not SVG), with a recommended maximum resolution
of 4096×2160. K3 fixes temperature at `1.0`, which the example sets explicitly.

```rust
use std::{env, fs, path::Path};

use rstructor::{Instructor, LLMClient, MediaFile};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Instructor, Serialize)]
struct DocumentFacts {
    title: String,
    summary: String,
    organizations: Vec<String>,
    total_usd: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("pass an image or PDF path as the first argument")?;
    let mime_type = match Path::new(&path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("pdf") => "application/pdf",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("png") => "image/png",
        _ => return Err("supported extensions are pdf, jpg, jpeg, and png".into()),
    };

    let media = MediaFile::from_bytes(fs::read(path)?, mime_type);
    let client = rstructor::client("openai/gpt-5.6-sol")?;
    let facts: DocumentFacts = client
        .materialize_with_media(
            "Extract the document title, a concise summary, named organizations, \
             and the total USD amount when present.",
            &[media],
        )
        .await?;

    println!("{facts:#?}");
    Ok(())
}
```

Run it with `OPENAI_API_KEY` set and pass a local file path:

```text
cargo run -- invoice.pdf
```

Run the Kimi chart variant with:

```text
MOONSHOT_API_KEY=your_key_here \
  cargo run --example kimi_k3_multimodal_example --features openai
```

## Classify text into an enum

The full [news categorizer](../examples/news_article_categorizer.rs) combines
classification with entity and sentiment extraction. An enum in the target
type constrains the provider to the variants your application handles.

```rust
use rstructor::{Instructor, LLMClient};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Instructor, Serialize)]
#[serde(rename_all = "snake_case")]
enum MarketEvent {
    Earnings,
    Macro,
    Merger,
    Regulatory,
    Other,
}

#[derive(Debug, Deserialize, Instructor, Serialize)]
struct Classification {
    event: MarketEvent,
    confidence: f64,
    rationale: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = rstructor::client("grok/grok-4.5")?;
    let result: Classification = client
        .materialize(
            "The FTC requested additional information about the proposed \
             acquisition before the statutory waiting period expires.",
        )
        .await?;

    println!("{result:#?}");
    Ok(())
}
```

This route reads `XAI_API_KEY`; change only the `provider/model` string to use a
different enabled provider.

## Put typed extraction behind an axum handler

The runnable [axum handler example](../examples/axum_handler_example.rs) uses a
real `Router` and a deterministic in-process request. Add the `mock` feature for
the self-contained demo; production can pass an `AnyClient` to the same generic
`app` function.

```toml
[dependencies]
axum = { version = "0.8", features = ["json"] }
rstructor = { version = "0.6.0", default-features = false, features = ["derive", "mock"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["macros", "rt"] }
tower = { version = "0.5", features = ["util"] }
```

```rust
use std::sync::Arc;

use axum::{
    body::{Body, to_bytes},
    extract::State,
    http::{Method, Request, StatusCode, header},
    routing::post,
    Json, Router,
};
use rstructor::{Instructor, LLMClient, MockClient};
use serde::{Deserialize, Serialize};
use tower::ServiceExt;

#[derive(Debug, Deserialize, Serialize)]
struct ExtractRequest {
    text: String,
}

#[derive(Debug, Deserialize, Instructor, PartialEq, Serialize)]
struct PositionBreak {
    portfolio_id: String,
    symbol: String,
    expected_quantity: i64,
    actual_quantity: i64,
    as_of: String,
}

struct AppState<C> {
    client: C,
}

async fn extract<C>(
    State(state): State<Arc<AppState<C>>>,
    Json(request): Json<ExtractRequest>,
) -> Result<Json<PositionBreak>, (StatusCode, String)>
where
    C: LLMClient + Send + Sync + 'static,
{
    state
        .client
        .materialize(&request.text)
        .await
        .map(Json)
        .map_err(|error| (StatusCode::UNPROCESSABLE_ENTITY, error.to_string()))
}

fn app<C>(client: C) -> Router
where
    C: LLMClient + Send + Sync + 'static,
{
    Router::new()
        .route("/position-breaks/extract", post(extract::<C>))
        .with_state(Arc::new(AppState { client }))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = MockClient::new().with_response(
        r#"{
            "portfolio_id": "HF-ALPHA-001",
            "symbol": "ESU6",
            "expected_quantity": -240,
            "actual_quantity": -238,
            "as_of": "2026-07-29T14:31:00Z"
        }"#,
    );
    let payload = ExtractRequest {
        text: "HF-ALPHA-001 expected short 240 ESU6; clearing shows short 238."
            .to_string(),
    };

    let response = app(client)
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/position-breaks/extract")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&payload)?))?,
        )
        .await?;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024).await?;
    let position_break: PositionBreak = serde_json::from_slice(&body)?;
    println!("{position_break:#?}");
    Ok(())
}
```

`State` comes before `Json` because body-consuming extractors must be last. The
one-shot request validates the whole HTTP boundary without opening a port.

## Test extraction offline with `MockClient`

The [offline testing example](../examples/mock_testing_example.rs) covers the
success, validation failure, recorded-request, and re-ask paths. Enable the
off-by-default `mock` feature:

```toml
rstructor = { version = "0.6.0", default-features = false, features = ["derive", "mock"] }
```

```rust
use rstructor::{Instructor, LLMClient, MockClient, RStructorError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Instructor, PartialEq, Serialize)]
#[llm(validate = "validate_fill")]
struct Fill {
    symbol: String,
    quantity: u64,
    price: f64,
}

fn validate_fill(fill: &Fill) -> rstructor::Result<()> {
    if fill.quantity == 0 || fill.price <= 0.0 {
        return Err(RStructorError::ValidationError(
            "quantity and price must be positive".to_string(),
        ));
    }
    Ok(())
}

async fn parse_fill<C: LLMClient + Sync>(
    client: &C,
    message: &str,
) -> rstructor::Result<Fill> {
    client.materialize(message).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = MockClient::new()
        .with_response(r#"{"symbol":"AAPL","quantity":0,"price":238.0}"#)
        .with_response(r#"{"symbol":"AAPL","quantity":5000,"price":238.0}"#)
        .with_retries(1);

    let fill = parse_fill(&client, "Bought 5,000 AAPL at 238").await?;
    assert_eq!(fill.quantity, 5_000);
    assert_eq!(client.request_count(), 2);
    assert_eq!(
        client.last_request().and_then(|request| request.schema_name),
        Some("Fill".to_string())
    );
    Ok(())
}
```

Scripted payloads still pass through the production Serde and `Instructor`
validation path; only the provider transport is replaced.

## Record and replay a sanitized fixture

The [fixture example](../examples/fixture_record_replay.rs) records a complete
non-streaming interaction and replays it without a key or network. The required
sanitizer runs before values enter the in-memory fixture; inline media bytes are
never persisted.

```rust
use rstructor::{Fixture, FixtureRecorder, FixtureSanitizer, LLMClient, OpenAIClient};

fn sanitizer() -> FixtureSanitizer {
    FixtureSanitizer::new(|text| text.replace("PRIVATE-ACCOUNT", "[ACCOUNT]"))
}

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let recorder = FixtureRecorder::new(OpenAIClient::from_env()?, sanitizer());
let fill: Fill = recorder.extract("PRIVATE-ACCOUNT bought 5,000 AAPL at 238").await?;
recorder.save("tests/fixtures/fill.fixture.json")?;

let fixture = Fixture::load("tests/fixtures/fill.fixture.json")?;
let replay = fixture.replay_with_sanitizer(sanitizer());
let replayed: Fill = replay.extract("PRIVATE-ACCOUNT bought 5,000 AAPL at 238").await?;
assert_eq!(replayed, fill);
replay.assert_finished()?;
# Ok(())
# }
```

Replay matches the sanitized operation, prompt, schema, and media metadata in
order. A mismatch leaves the interaction unconsumed and reports only the field
that differed, so assertion failures do not echo fixture contents.

## Use a local model through Ollama

The [Ollama example](../examples/ollama_local_example.rs) is safe to run in CI:
it only connects when `OLLAMA_MODEL` is set. The route is keyless and therefore
does not send an `Authorization` header.

```rust
use rstructor::{Instructor, LLMClient};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Instructor, Serialize)]
struct Summary {
    summary: String,
    topics: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = rstructor::client("ollama/llama3.3")?;
    let summary: Summary = client
        .materialize(
            "Rust combines memory safety, zero-cost abstractions, and fearless concurrency.",
        )
        .await?;

    println!("{summary:#?}");
    Ok(())
}
```

Start Ollama and ensure the model is present before running:

```text
ollama pull llama3.3
cargo run
```

LM Studio uses the same pattern with `lm_studio/your-loaded-model`.

## Choose a provider at runtime

The runnable
[runtime provider example](../examples/runtime_provider_example.rs) accepts a
CLI argument or `RSTRUCTOR_CLIENT`. `rstructor::client` parses the first path
segment as the route and preserves the rest as the provider-native model ID.

```rust
use rstructor::{Instructor, LLMClient};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Instructor, Serialize)]
struct Position {
    symbol: String,
    quantity: i64,
    market_value_usd: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec = std::env::var("RSTRUCTOR_CLIENT")
        .unwrap_or_else(|_| "ollama/llama3.3".to_string());
    let client = rstructor::client(&spec)?;
    let position: Position = client
        .materialize("AAPL: long 125,000 shares, market value $29,750,000")
        .await?;

    println!("{position:#?}");
    Ok(())
}
```

Examples of valid values are `openai/gpt-5.6-sol`,
`anthropic/claude-opus-5`, `ollama/llama3.3`, and
`openrouter/moonshotai/kimi-k3`. Hosted routes read their provider-specific key;
Ollama and LM Studio are keyless.

## Reuse a schemars model

The runnable [schemars bridge example](../examples/schemars_bridge_example.rs)
proves that no `Instructor` derive is needed. Enable both `schemars` and `mock`
for this offline version:

```toml
rstructor = { version = "0.6.0", default-features = false, features = ["mock", "schemars"] }
schemars = "1"
```

```rust
use rstructor::{LLMClient, MockClient, Schemars};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A position reported by a prime broker.
#[derive(Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
struct Position {
    /// Fund or account identifier.
    portfolio_id: String,
    /// Exchange ticker or contract symbol.
    symbol: String,
    /// Signed quantity: positive is long and negative is short.
    quantity: i64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = MockClient::new().with_response(
        r#"{
            "portfolio_id": "HF-ALPHA-001",
            "symbol": "ESU6",
            "quantity": -240
        }"#,
    );
    let position = client
        .materialize::<Schemars<Position>>("HF-ALPHA-001 is short 240 ESU6")
        .await?
        .into_inner();

    assert_eq!(position.quantity, -240);
    Ok(())
}
```

Doc comments become schema descriptions, and acyclic nested schemas are
inlined. Recursive schemars types return a schema error before any provider
request because strict provider schemas must be reference-free.

## Inspect cumulative retry cost

The [attempt-ledger example](../examples/retry_attempt_ledger.rs) shows both
success and terminal-failure reporting. `extract_with_report` attaches the same
report shape to both outcomes, preserving every provider attempt and summing all
usage the provider reported.

```rust
use rstructor::{AttemptOutcome, Instructor, LLMClient};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Instructor, Serialize)]
struct Portfolio {
    portfolio_id: String,
    positions: Vec<Position>,
}

#[derive(Debug, Deserialize, Instructor, Serialize)]
struct Position {
    symbol: String,
    quantity: i64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = rstructor::client("openai/gpt-5.6-sol")?;
    match client
        .extract_with_report::<Portfolio>(
            "HF-ALPHA-001: short 240 ESU6 and long 125,000 AAPL",
        )
        .await
    {
        Ok(extraction) => {
            for attempt in &extraction.report.attempts {
                println!(
                    "attempt {}: {:?}, {:?}, usage={:?}",
                    attempt.number, attempt.kind, attempt.outcome, attempt.usage
                );
                if let AttemptOutcome::Failed { message, .. } = &attempt.outcome {
                    eprintln!("provider failure: {message}");
                }
            }
            println!(
                "known cumulative usage: {:?}",
                extraction.report.cumulative_usage
            );
            println!("portfolio: {:#?}", extraction.data);
        }
        Err(failure) => {
            eprintln!("final error: {}", failure.error());
            eprintln!("attempts: {:#?}", failure.report.attempts);
            eprintln!(
                "known cumulative usage: {:?}",
                failure.report.cumulative_usage
            );
        }
    }
    Ok(())
}
```

Usage is deliberately conservative: if a provider omits usage for one response,
the attempt remains in the ledger while cumulative totals include only reported
tokens. Local preflight failures have an empty ledger.

## Reuse stable instructions with prompt caching

The request builder keeps stable instructions in the provider's native system
channel while the changing request stays in the user turn. This gives implicit
prefix caches a reusable prefix and applies equally to `materialize`,
`generate`, streaming, media, and `run`:

```rust
use rstructor::{Instructor, LLMClient, OpenAIClient, RequestExt};
use serde::{Deserialize, Serialize};

const RISK_POLICY: &str = "\
Report gross exposure as a multiple of NAV.
Use the portfolio's base currency.
Return status as within_limits or breached.";

#[derive(Debug, Deserialize, Instructor, Serialize)]
struct RiskSummary {
    status: String,
    gross_exposure: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAIClient::from_env()?;
    let extraction = client
        .with_system(RISK_POLICY)
        .extract_with_report::<RiskSummary>(
            "NAV USD 100mm; long USD 92mm; short USD 50mm",
        )
        .await?;

    if let Some(usage) = extraction.report.cumulative_usage {
        println!(
            "input={} cache_read={} cache_write={}",
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.cache_write_input_tokens,
        );
    }
    println!("{:#?}", extraction.data);
    Ok(())
}
```

Cache counters are subsets of input tokens, not extra tokens. OpenAI, Gemini,
and xAI can cache eligible exact prefixes implicitly. Anthropic requires
cache-control configuration; rstructor does not enable paid cache writes by
default. Current provider-specific thresholds and retention policies are in the
official [OpenAI](https://developers.openai.com/api/docs/guides/prompt-caching),
[Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-caching),
[Gemini](https://ai.google.dev/gemini-api/docs/caching), and
[xAI](https://docs.x.ai/developers/advanced-api-usage/prompt-caching)
documentation.

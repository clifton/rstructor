# rstructor: Structured LLM Outputs for Rust

<p align="center">
  <img src="https://img.shields.io/badge/rust-2024-orange" alt="Rust 2024"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"/>
  <img src="https://github.com/clifton/rstructor/actions/workflows/test.yml/badge.svg" alt="Tests Status"/>
  <img src="https://github.com/clifton/rstructor/actions/workflows/clippy.yml/badge.svg" alt="Clippy Status"/>
</p>

rstructor is a Rust library for extracting structured data from Large Language Models (LLMs) with built-in validation. Define your schemas as Rust structs/enums, and rstructor will handle the rest—generating JSON Schemas, communicating with LLMs, parsing responses, and validating the results.

Think of it as the Rust equivalent of [Instructor + Pydantic](https://github.com/jxnl/instructor) for Python, bringing the same structured output capabilities to the Rust ecosystem.

## ✨ Features

- **📝 Type-Safe Definitions**: Define data models as standard Rust structs/enums with attributes
- **🔄 JSON Schema Generation**: Auto-generates JSON Schema from your Rust types
- **✅ Built-in Validation**: Type checking plus custom business rule validation
- **🔌 Multiple LLM Providers**: Support for OpenAI, Anthropic, Grok (xAI), and Gemini (Google), with an extensible backend system
- **🧩 Complex Data Structures**: Support for nested objects, arrays, optional fields, and deeply nested enums
- **🧠 Schema Fidelity**: Heuristic-free JSON Schema generation that preserves nested struct and enum detail
- **🔍 Custom Validation Rules**: Add domain-specific validation with automatically detected `validate` methods
- **🔁 Async API**: Fully asynchronous API for efficient operations
- **⚙️ Builder Pattern**: Fluent API for configuring LLM clients (temperature retries, timeouts, etc)
- **📊 Feature Flags**: Optional backends via feature flags

## 📦 Installation

Add rstructor to your `Cargo.toml`:

```toml
[dependencies]
rstructor = "0.1.0"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["rt-multi-thread", "macros"] }
```

## 🚀 Quick Start

Here's a simple example of extracting structured information about a movie from an LLM:

```rust
use rstructor::{Instructor, LLMClient, OpenAIClient, OpenAIModel};
use serde::{Serialize, Deserialize};
use std::env;
use std::time::Duration;

// Define your data model
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Movie {
    #[llm(description = "Title of the movie")]
    title: String,

    #[llm(description = "Director of the movie")]
    director: String,

    #[llm(description = "Year the movie was released", example = 2010)]
    year: u16,

    #[llm(description = "Brief plot summary")]
    plot: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY")?;

    // Create an OpenAI client
    let client = OpenAIClient::new(api_key)?
        .model(OpenAIModel::Gpt4OMini)
        .temperature(0.0)
        .with_timeout(Duration::from_secs(30))  // Optional: set 30 second timeout
        .build();

    // Generate structured information with a simple prompt
    // For production use, prefer generate_struct_with_retry for automatic error recovery
    let movie: Movie = client.generate_struct("Tell me about the movie Inception").await?;

    // Use the structured data
    println!("Title: {}", movie.title);
    println!("Director: {}", movie.director);
    println!("Year: {}", movie.year);
    println!("Plot: {}", movie.plot);

    Ok(())
}
```

## 📝 Detailed Examples

### Production Example with Automatic Retry

For production use, prefer `generate_struct_with_retry` which automatically retries on validation errors:

```rust
use rstructor::{Instructor, LLMClient, OpenAIClient, OpenAIModel};
use std::time::Duration;
use serde::{Serialize, Deserialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "Information about a movie")]
struct Movie {
    #[llm(description = "Title of the movie")]
    title: String,

    #[llm(description = "Year the movie was released", example = 2010)]
    year: u16,

    #[llm(description = "IMDB rating out of 10", example = 8.5)]
    rating: f32,
}

// Generate with automatic retry (recommended for production)
let movie: Movie = client
    .generate_struct_with_retry::<Movie>(
        "Tell me about Inception",
        Some(3),    // max retries
        Some(true),  // include error feedback in retries
    )
    .await?;
```

### Basic Example with Validation

Add custom validation rules to enforce business logic beyond type checking:

```rust
use rstructor::{Instructor, LLMClient, OpenAIClient, OpenAIModel, RStructorError, Result};
use serde::{Serialize, Deserialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "Information about a movie")]
struct Movie {
    #[llm(description = "Title of the movie")]
    title: String,

    #[llm(description = "Year the movie was released", example = 2010)]
    year: u16,

    #[llm(description = "IMDB rating out of 10", example = 8.5)]
    rating: f32,
}

// Add custom validation
impl Movie {
    fn validate(&self) -> Result<()> {
        // Title can't be empty
        if self.title.trim().is_empty() {
            return Err(RStructorError::ValidationError(
                "Movie title cannot be empty".to_string()
            ));
        }

        // Year must be in a reasonable range
        if self.year < 1888 || self.year > 2030 {
            return Err(RStructorError::ValidationError(
                format!("Movie year must be between 1888 and 2030, got {}", self.year)
            ));
        }

        // Rating must be between 0 and 10
        if self.rating < 0.0 || self.rating > 10.0 {
            return Err(RStructorError::ValidationError(
                format!("Rating must be between 0 and 10, got {}", self.rating)
            ));
        }

        Ok(())
    }
}

// The derive macro automatically wires this method into the generated implementation,
// so you won't see `dead_code` warnings even if the method is only called by rstructor.
```

### Complex Nested Structures

rstructor supports complex nested data structures:

```rust
use rstructor::{Instructor, LLMClient, OpenAIClient, OpenAIModel};
use std::time::Duration;
use serde::{Serialize, Deserialize};

// Define a nested data model for a recipe
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Ingredient {
    #[llm(description = "Name of the ingredient", example = "flour")]
    name: String,

    #[llm(description = "Amount of the ingredient", example = 2.5)]
    amount: f32,

    #[llm(description = "Unit of measurement", example = "cups")]
    unit: String,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Step {
    #[llm(description = "Order number of this step", example = 1)]
    number: u16,

    #[llm(description = "Description of this step",
          example = "Mix the flour and sugar together")]
    description: String,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "A cooking recipe with ingredients and instructions")]
struct Recipe {
    #[llm(description = "Name of the recipe", example = "Chocolate Chip Cookies")]
    name: String,

    #[llm(description = "List of ingredients needed")]
    ingredients: Vec<Ingredient>,

    #[llm(description = "Step-by-step cooking instructions")]
    steps: Vec<Step>,
}

// Usage:
// let recipe: Recipe = client.generate_struct("Give me a recipe for chocolate chip cookies").await?;
```

### Working with Enums

rstructor supports both simple enums and enums with associated data.

#### Simple Enums

Use enums for categorical data:

```rust
use rstructor::{Instructor, LLMClient, AnthropicClient, AnthropicModel};
use serde::{Serialize, Deserialize};

// Define an enum for sentiment analysis
#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "The sentiment of a text")]
enum Sentiment {
    #[llm(description = "Positive or favorable sentiment")]
    Positive,

    #[llm(description = "Negative or unfavorable sentiment")]
    Negative,

    #[llm(description = "Neither clearly positive nor negative")]
    Neutral,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct SentimentAnalysis {
    #[llm(description = "The text to analyze")]
    text: String,

    #[llm(description = "The detected sentiment of the text")]
    sentiment: Sentiment,

    #[llm(description = "Confidence score between 0.0 and 1.0",
          example = 0.85)]
    confidence: f32,
}

// Usage:
// let analysis: SentimentAnalysis = client.generate_struct("Analyze the sentiment of: I love this product!").await?;
```

#### Enums with Associated Data (Tagged Unions)

rstructor also supports more complex enums with associated data:

```rust
use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};

// Enum with different types of associated data
#[derive(Instructor, Serialize, Deserialize, Debug)]
enum UserStatus {
    #[llm(description = "The user is online")]
    Online,

    #[llm(description = "The user is offline")]
    Offline,

    #[llm(description = "The user is away with an optional message")]
    Away(String),

    #[llm(description = "The user is busy until a specific time in minutes")]
    Busy(u32),
}

// Using struct variants for more complex associated data
#[derive(Instructor, Serialize, Deserialize, Debug)]
enum PaymentMethod {
    #[llm(description = "Payment with credit card")]
    Card {
        #[llm(description = "Credit card number")]
        number: String,

        #[llm(description = "Expiration date in MM/YY format")]
        expiry: String,
    },

    #[llm(description = "Payment via PayPal account")]
    PayPal(String),

    #[llm(description = "Payment will be made on delivery")]
    CashOnDelivery,
}

// Usage:
// let user_status: UserStatus = client.generate_struct("What's the user's status?").await?;
```

#### Nested Enums Across Structs

Enums can be freely nested inside other enums and structs—`#[derive(Instructor)]` now
generates the correct schema without requiring manual `SchemaType` implementations:

```rust
#[derive(Instructor, Serialize, Deserialize, Debug)]
enum TaskState {
    #[llm(description = "Task is pending with a priority level")]
    Pending { priority: Priority },
    #[llm(description = "Task is in progress")]
    InProgress { priority: Priority, status: Status },
    #[llm(description = "Task is completed")]
    Completed { status: Status },
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Task {
    #[llm(description = "Task title")]
    title: String,

    #[llm(description = "Current task state with nested enums")]
    state: TaskState,
}

// Works automatically – TaskState::schema() includes the nested enum structure.
```

See `examples/nested_enum_example.rs` for a complete runnable walkthrough that
also exercises deserialization of nested enum variants.

When serialized to JSON, these enum variants with data become tagged unions:

```json
// UserStatus::Away("Back in 10 minutes")
{
  "Away": "Back in 10 minutes"
}

// PaymentMethod::Card { number: "4111...", expiry: "12/25" }
{
  "Card": {
    "number": "4111 1111 1111 1111",
    "expiry": "12/25"
  }
}
```

### Working with Custom Types (Dates, UUIDs, etc.)

rstructor provides the `CustomTypeSchema` trait to handle types that don't have direct JSON representations but need specific schema formats. This is particularly useful for:

- Date/time types (e.g., `chrono::DateTime`)
- UUIDs (e.g., `uuid::Uuid`)
- Email addresses
- URLs
- Custom domain-specific types

#### Basic Implementation

```rust
use rstructor::{Instructor, schema::CustomTypeSchema};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use serde_json::json;
use uuid::Uuid;

// Implement CustomTypeSchema for chrono::DateTime<Utc>
impl CustomTypeSchema for DateTime<Utc> {
    fn schema_type() -> &'static str {
        "string"
    }

    fn schema_format() -> Option<&'static str> {
        Some("date-time")
    }

    fn schema_description() -> Option<String> {
        Some("ISO-8601 formatted date and time".to_string())
    }
}

// Implement CustomTypeSchema for UUID
impl CustomTypeSchema for Uuid {
    fn schema_type() -> &'static str {
        "string"
    }

    fn schema_format() -> Option<&'static str> {
        Some("uuid")
    }
}
```

#### Usage in Structs

Once implemented, these custom types can be used directly in your structs:

```rust
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Event {
    #[llm(description = "Unique identifier for the event")]
    id: Uuid,

    #[llm(description = "Name of the event")]
    name: String,

    #[llm(description = "When the event starts")]
    start_time: DateTime<Utc>,

    #[llm(description = "When the event ends (optional)")]
    end_time: Option<DateTime<Utc>>,

    #[llm(description = "Recurring event dates")]
    recurring_dates: Vec<DateTime<Utc>>, // Even works with arrays!
}
```

#### Advanced Customization

You can add additional schema properties for more complex validation:

```rust
impl CustomTypeSchema for EmailAddress {
    fn schema_type() -> &'static str {
        "string"
    }

    fn schema_format() -> Option<&'static str> {
        Some("email")
    }

    fn schema_additional_properties() -> Option<Value> {
        Some(json!({
            "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
            "examples": ["user@example.com", "contact@company.org"]
        }))
    }
}
```

The macro automatically detects these custom types and generates appropriate JSON Schema with format specifications that guide LLMs to produce correctly formatted values. The library includes built-in recognition of common date and UUID types, but you can implement the trait for any custom type.

### Configuring Different LLM Providers

Choose between different providers:

```rust
// Using OpenAI
let openai_client = OpenAIClient::new(openai_api_key)?
    .model(OpenAIModel::Gpt5)
    .temperature(0.2)
    .max_tokens(1500)
    .with_timeout(Duration::from_secs(60))  // Optional: set 60 second timeout
    .build();

// Using Anthropic
let anthropic_client = AnthropicClient::new(anthropic_api_key)?
    .model(AnthropicModel::ClaudeSonnet4)
    .temperature(0.0)
    .max_tokens(2000)
    .with_timeout(Duration::from_secs(60))  // Optional: set 60 second timeout
    .build();

// Using Grok (xAI) - reads from XAI_API_KEY environment variable
let grok_client = GrokClient::from_env()?  // Reads from XAI_API_KEY env var
    .model(GrokModel::Grok4)
    .temperature(0.0)
    .max_tokens(1500)
    .with_timeout(Duration::from_secs(60))  // Optional: set 60 second timeout
    .build();

// Using Gemini (Google) - reads from GEMINI_API_KEY environment variable
let gemini_client = GeminiClient::from_env()?  // Reads from GEMINI_API_KEY env var
    .model(GeminiModel::Gemini25Flash)
    .temperature(0.0)
    .max_tokens(1500)
    .with_timeout(Duration::from_secs(60))  // Optional: set 60 second timeout
    .build();
```

### Configuring Request Timeouts

All clients (`OpenAIClient`, `AnthropicClient`, `GrokClient`, and `GeminiClient`) support configurable timeouts for HTTP requests using the builder pattern:

```rust
use std::time::Duration;

let client = OpenAIClient::new(api_key)?
    .model(OpenAIModel::Gpt4O)
    .temperature(0.0)
    .with_timeout(Duration::from_secs(30))  // Set 30 second timeout
    .build();
```

**Timeout Behavior:**
- The timeout applies to each HTTP request made by the client
- If a request exceeds the timeout, it will return `RStructorError::Timeout`
- If no timeout is specified, the client uses reqwest's default timeout behavior
- Timeout values are specified as `std::time::Duration` (e.g., `Duration::from_secs(30)` or `Duration::from_millis(2500)`)

**Example with timeout error handling:**

```rust
use rstructor::{OpenAIClient, OpenAIModel, RStructorError};
use std::time::Duration;

match client.generate_struct::<Movie>("prompt").await {
    Ok(movie) => println!("Success: {:?}", movie),
    Err(RStructorError::Timeout) => eprintln!("Request timed out"),
    Err(e) => eprintln!("Other error: {}", e),
}
```

### Handling Container-Level Attributes

Add metadata and examples at the container level:

```rust
#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "Detailed information about a movie",
      title = "MovieDetails",
      examples = [
        ::serde_json::json!({
            "title": "The Matrix",
            "director": "Lana and Lilly Wachowski",
            "year": 1999,
            "genres": ["Sci-Fi", "Action"],
            "rating": 8.7,
            "plot": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."
        })
      ])]
struct Movie {
    // fields...
}
```

## 📚 API Reference

### Instructor Trait

The `Instructor` trait is the core of rstructor. It's implemented automatically via the derive macro and provides schema generation and validation:

```rust
pub trait Instructor: SchemaType + DeserializeOwned + Serialize {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}
```

Override the `validate` method to add custom validation logic.

### CustomTypeSchema Trait

The `CustomTypeSchema` trait allows you to define JSON Schema representations for types that don't have direct JSON equivalents, like dates and UUIDs:

```rust
pub trait CustomTypeSchema {
    /// Returns the JSON Schema type for this custom type
    ///
    /// This is typically "string" for dates, UUIDs, etc.
    fn schema_type() -> &'static str;

    /// Returns the JSON Schema format for this custom type
    ///
    /// Common formats include "date-time", "uuid", "email", etc.
    fn schema_format() -> Option<&'static str> {
        None
    }

    /// Returns a description of this custom type for documentation
    fn schema_description() -> Option<String> {
        None
    }

    /// Returns any additional JSON Schema properties for this type
    ///
    /// This can include patterns, examples, minimum/maximum values, etc.
    fn schema_additional_properties() -> Option<Value> {
        None
    }

    /// Generate a complete JSON Schema object for this type
    fn json_schema() -> Value {
        // Default implementation that combines all properties
        // (You don't normally need to override this)
        let mut schema = json!({
            "type": Self::schema_type(),
        });

        // Add format if present
        if let Some(format) = Self::schema_format() {
            schema.as_object_mut().unwrap()
                .insert("format".to_string(), Value::String(format.to_string()));
        }

        // Add description if present
        if let Some(description) = Self::schema_description() {
            schema.as_object_mut().unwrap()
                .insert("description".to_string(), Value::String(description));
        }

        // Add any additional properties
        if let Some(additional) = Self::schema_additional_properties() {
            // Merge additional properties into the schema
            if let Some(additional_obj) = additional.as_object() {
                for (key, value) in additional_obj {
                    schema.as_object_mut().unwrap()
                        .insert(key.clone(), value.clone());
                }
            }
        }

        schema
    }
}
```

Implement this trait for custom types like `DateTime<Utc>` or `Uuid` to control their JSON Schema representation. Most implementations only need to specify `schema_type()` and `schema_format()`, with the remaining methods providing additional schema customization when needed.

### LLMClient Trait

The `LLMClient` trait defines the interface for all LLM providers:

```rust
#[async_trait]
pub trait LLMClient {
    /// Generate a structured object from a prompt (single attempt)
    async fn generate_struct<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static;

    /// Generate a structured object with automatic retry on validation errors
    ///
    /// This is the recommended method for production use as it automatically
    /// retries failed generations with error feedback to improve success rates.
    async fn generate_struct_with_retry<T>(
        &self,
        prompt: &str,
        max_retries: Option<usize>,
        include_errors: Option<bool>,
    ) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static;

    /// Generate raw text without structure
    async fn generate(&self, prompt: &str) -> Result<String>;
}
```

**Note**: For production applications, prefer `generate_struct_with_retry` over `generate_struct` as it automatically handles validation errors by retrying with error feedback. This significantly improves success rates with complex schemas.

### Supported Attributes

#### Field Attributes
- `description`: Text description of the field
- `example`: A single example value
- `examples`: Multiple example values

#### Container Attributes
- `description`: Text description of the struct or enum
- `title`: Custom title for the JSON Schema
- `examples`: Example instances as JSON objects

## 🔧 Feature Flags

Configure rstructor with feature flags:

```toml
[dependencies]
rstructor = { version = "0.1.0", features = ["openai", "anthropic", "grok", "gemini"] }
```

Available features:
- `openai`: Include the OpenAI client
- `anthropic`: Include the Anthropic client
- `grok`: Include the Grok (xAI) client
- `gemini`: Include the Gemini (Google) client
- `derive`: Include the derive macro (enabled by default)
- `logging`: Enable tracing integration with default subscriber

## 📊 Logging and Tracing

rstructor includes structured logging via the `tracing` crate:

```rust
use rstructor::logging::{init_logging, LogLevel};

// Initialize with desired level
init_logging(LogLevel::Debug);

// Or use filter strings for granular control
// init_logging_with_filter("rstructor=info,rstructor::backend=trace");
```

Override with environment variables:
```bash
RSTRUCTOR_LOG=debug cargo run
```

Validation errors, retries, and API interactions are thoroughly logged at appropriate levels.

## 📋 Examples

See the `examples/` directory for complete, working examples:

- `structured_movie_info.rs`: Basic example of getting movie information with validation
- `nested_objects_example.rs`: Working with complex nested structures for recipe data
- `news_article_categorizer.rs`: Using enums for categorization
- `enum_with_data_example.rs`: Working with enums that have associated data (tagged unions)
- `event_planner.rs`: Interactive event planning with user input
- `weather_example.rs`: Simple model with validation demonstration
- `validation_example.rs`: Demonstrates custom validation without dead code warnings
- `custom_type_example.rs`: Using custom types like dates and UUIDs with JSON Schema format support
- `logging_example.rs`: Demonstrates tracing integration with custom log levels
- `nested_enum_example.rs`: Shows automatic schema generation for nested enums inside structs

## ▶️ Running the Examples

```bash
# Set environment variables
export OPENAI_API_KEY=your_openai_key_here
# or
export ANTHROPIC_API_KEY=your_anthropic_key_here
# or
export XAI_API_KEY=your_xai_key_here
# or
export GEMINI_API_KEY=your_gemini_key_here

# Run examples
cargo run --example structured_movie_info
cargo run --example news_article_categorizer
```

## ⚠️ Current Limitations

rstructor currently focuses on single-turn, synchronous structured output generation. The following features are planned but not yet implemented:

- **Streaming Responses**: Real-time streaming of partial results as they're generated
- **Conversation History**: Multi-turn conversations with message history (currently only single prompts supported)
- **System Messages**: Explicit system prompts for role-based interactions
- **Response Modes**: Different validation strategies (strict, partial, etc.)
- **Rate Limiting**: Built-in rate limit handling and backoff strategies

## 🛣️ Roadmap

- [x] Core traits and interfaces
- [x] OpenAI backend implementation
- [x] Anthropic backend implementation
- [x] Procedural macro for deriving `Instructor`
- [x] Schema generation functionality
- [x] Custom validation capabilities
- [x] Support for nested structures
- [x] Rich validation API with custom domain rules
- [x] Support for enums with associated data (tagged unions)
- [x] Support for custom types (dates, UUIDs, etc.)
- [x] Structured logging and tracing
- [x] Automatic retry with validation error feedback
- [ ] Streaming responses
- [ ] Conversation history / multi-turn support
- [ ] System messages and role-based prompts
- [ ] Response modes (strict, partial, retry)
- [ ] Rate limiting and backoff strategies
- [ ] Support for additional LLM providers
- [ ] Integration with web frameworks (Axum, Actix)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

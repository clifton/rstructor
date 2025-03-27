# RStructor: Structured LLM Outputs for Rust

<p align="center">
  <img src="https://img.shields.io/badge/rust-2024-orange" alt="Rust 2024"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"/>
  <img src="https://github.com/clifton/rstructor/actions/workflows/test.yml/badge.svg" alt="Tests Status"/>
  <img src="https://github.com/clifton/rstructor/actions/workflows/clippy.yml/badge.svg" alt="Clippy Status"/>
</p>

RStructor is a Rust library for extracting structured data from Large Language Models (LLMs) with built-in validation. Define your schemas as Rust structs/enums, and RStructor will handle the rest—generating JSON Schemas, communicating with LLMs, parsing responses, and validating the results.

Think of it as the Rust equivalent of [Instructor + Pydantic](https://github.com/jxnl/instructor) for Python, bringing the same structured output capabilities to the Rust ecosystem.

## ✨ Features

- **📝 Type-Safe Definitions**: Define data models as standard Rust structs/enums with attributes
- **🔄 JSON Schema Generation**: Auto-generates JSON Schema from your Rust types
- **✅ Built-in Validation**: Type checking plus custom business rule validation
- **🔌 Multiple LLM Providers**: Support for OpenAI and Anthropic, with an extensible backend system
- **🧩 Complex Data Structures**: Support for nested objects, arrays, and optional fields
- **🔍 Custom Validation Rules**: Add domain-specific validation for reliable results
- **🔁 Async API**: Fully asynchronous API for efficient operations
- **⚙️ Builder Pattern**: Fluent API for configuring LLM clients
- **📊 Feature Flags**: Optional backends via feature flags

## 📦 Installation

Add RStructor to your `Cargo.toml`:

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
        .model(OpenAIModel::Gpt35Turbo)
        .temperature(0.0)
        .build();

    // Generate structured information with a simple prompt
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
```

### Complex Nested Structures

RStructor supports complex nested data structures:

```rust
use rstructor::{Instructor, LLMClient, OpenAIClient, OpenAIModel};
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

RStructor supports both simple enums and enums with associated data.

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

RStructor also supports more complex enums with associated data:

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

### Configuring Different LLM Providers

Choose between different providers:

```rust
// Using OpenAI
let openai_client = OpenAIClient::new(openai_api_key)?
    .model(OpenAIModel::Gpt4)
    .temperature(0.2)
    .max_tokens(1500)
    .build();

// Using Anthropic
let anthropic_client = AnthropicClient::new(anthropic_api_key)?
    .model(AnthropicModel::Claude3Sonnet)
    .temperature(0.0)
    .max_tokens(2000)
    .build();
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

The `Instructor` trait is the core of RStructor. It's implemented automatically via the derive macro and provides schema generation and validation:

```rust
pub trait Instructor: SchemaType + DeserializeOwned + Serialize {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}
```

Override the `validate` method to add custom validation logic.

### LLMClient Trait

The `LLMClient` trait defines the interface for all LLM providers:

```rust
#[async_trait]
pub trait LLMClient {
    async fn generate_struct<T>(&self, prompt: &str) -> Result<T>
    where
        T: Instructor + DeserializeOwned + Send + 'static;

    async fn generate(&self, prompt: &str) -> Result<String>;
}
```

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

Configure RStructor with feature flags:

```toml
[dependencies]
rstructor = { version = "0.1.0", features = ["openai", "anthropic"] }
```

Available features:
- `openai`: Include the OpenAI client
- `anthropic`: Include the Anthropic client
- `derive`: Include the derive macro (enabled by default)

## 📋 Examples

See the `examples/` directory for complete, working examples:

- `structured_movie_info.rs`: Basic example of getting movie information with validation
- `nested_objects_example.rs`: Working with complex nested structures for recipe data
- `news_article_categorizer.rs`: Using enums for categorization
- `enum_with_data_example.rs`: Working with enums that have associated data (tagged unions)
- `event_planner.rs`: Interactive event planning with user input
- `weather_example.rs`: Simple model with validation demonstration

## ▶️ Running the Examples

```bash
# Set environment variables
export OPENAI_API_KEY=your_openai_key_here
# or
export ANTHROPIC_API_KEY=your_anthropic_key_here

# Run examples
cargo run --example structured_movie_info
cargo run --example news_article_categorizer
```

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
- [ ] Streaming responses
- [ ] Support for additional LLM providers
- [ ] Integration with web frameworks (Axum, Actix)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
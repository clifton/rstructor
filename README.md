# RStructor: Structured LLM Outputs for Rust

RStructor is a Rust library for defining structured data models and getting Large Language Models (LLMs) to fill them, with automatic validation. It provides a type-safe way to obtain structured data from LLMs by defining your schemas with Rust structs and enums.

## Features

- **📝 Structured Definitions**: Define your data models as Rust structs/enums with annotations for LLM guidance
- **🔄 JSON Schema Generation**: Automatically generate JSON Schema for your models to guide LLM outputs
- **✅ Validation**: Ensure LLM responses match your defined structure and types
- **🔌 Multiple LLM Providers**: Support for OpenAI and Anthropic APIs, with an extensible backend system
- **🔄 Async API**: Fully async API for efficient network operations
- **🧩 Nested Structures**: Support for complex nested data structures
- **🔍 Validation Rules**: Custom validation rules for enhanced type safety

## Example Usage

```rust
use rstructor::{LLMModel, OpenAIClient, OpenAIModel}; 
use serde::{Serialize, Deserialize};
use std::env;

// Define your data model
#[derive(LLMModel, Serialize, Deserialize, Debug)]
#[llm(description = "Information about a movie", 
      title = "DetailedMovieInfo",
      examples = [
        ::serde_json::json!({"title": "Inception", "director": "Christopher Nolan", "year": 2010, "genres": ["Sci-Fi", "Action"], "rating": 8.8}),
        ::serde_json::json!({"title": "The Godfather", "director": "Francis Ford Coppola", "year": 1972, "genres": ["Crime", "Drama"]})
      ])]
#[serde(rename_all = "camelCase")]
struct MovieInfo {
    #[llm(description = "Title of the movie")]
    title: String,
    
    #[llm(description = "Name of the director", example = "Christopher Nolan")]
    director: String,
    
    #[llm(description = "Year the movie was released", example = 2010)]
    year: u16,
    
    #[llm(description = "Genres of the movie", example = ["Action", "Sci-Fi"])]
    genres: Vec<String>,
    
    #[llm(description = "IMDB rating from 0.0 to 10.0", example = 8.8)]
    rating: Option<f32>,
}

// Custom validation logic
impl MovieInfo {
    fn validate(&self) -> rstructor::Result<()> {
        // Check that the rating is between 0 and 10
        if let Some(rating) = self.rating {
            if rating < 0.0 || rating > 10.0 {
                return Err(rstructor::RStructorError::ValidationError(
                    format!("Rating must be between 0 and 10, got {}", rating)
                ));
            }
        }
        Ok(())
    }
}

// Use with OpenAI
async fn get_movie_info() -> Result<MovieInfo, Box<dyn std::error::Error>> {
    let api_key = env::var("OPENAI_API_KEY")?;
    
    let client = OpenAIClient::new(api_key)?
        .model(OpenAIModel::Gpt4)
        .temperature(0.0)
        .build();
    
    let prompt = "Tell me about the movie Inception";
    let movie: MovieInfo = client.generate_struct(prompt).await?;
    
    println!("Title: {}", movie.title);
    println!("Director: {}", movie.director);
    println!("Year: {}", movie.year);
    println!("Genres: {:?}", movie.genres);
    if let Some(rating) = movie.rating {
        println!("Rating: {:.1}", rating);
    }
    
    Ok(movie)
}
```

## Supported Attributes

### Field-level Attributes
- `description`: Text description of the field
- `example`: A single example value (supports simple types, objects, and arrays)
- `examples`: Multiple example values

### Container-level Attributes
- `description`: Text description of the struct or enum
- `title`: Custom title for the JSON Schema (defaults to type name)
- `examples`: Example instances (complete objects for structs, string values for enums)

### Serde Integration
- Respects `#[serde(rename_all = "...")]` for property names (camelCase, snake_case, etc.)

## Custom Validation

RStructor supports custom validation rules beyond simple type checking. This allows you to enforce domain-specific constraints on your data:

```rust
impl MovieInfo {
    fn validate(&self) -> rstructor::Result<()> {
        // Validate year is reasonable
        if self.year < 1888 || self.year > 2030 {
            return Err(rstructor::RStructorError::ValidationError(
                format!("Movie year must be between 1888 and 2030, got {}", self.year)
            ));
        }
        
        // Check that rating is valid
        if let Some(rating) = self.rating {
            if rating < 0.0 || rating > 10.0 {
                return Err(rstructor::RStructorError::ValidationError(
                    format!("Rating must be between 0 and 10, got {}", rating)
                ));
            }
        }
        
        Ok(())
    }
}
```

Validation is automatically applied when using `generate_struct()` to get structured data from an LLM, ensuring the output meets your requirements.

## Examples

See the `examples/` directory for various use cases:

- `structured_movie_info.rs`: Basic example of getting movie information with validation
- `nested_objects_example.rs`: Working with complex nested structures (recipe data)
- `news_article_categorizer.rs`: Using enums and categorization 
- `event_planner.rs`: User input processing for event planning
- `weather_example.rs`: Simple model with validation demonstration

## Running the Examples

Set the appropriate environment variables:
```bash
export OPENAI_API_KEY=your_openai_key_here
# or
export ANTHROPIC_API_KEY=your_anthropic_key_here

# Then run the examples
cargo run --example structured_movie_info
cargo run --example nested_objects_example
```

## Features & Roadmap

- [x] Core traits and interfaces
- [x] OpenAI backend implementation
- [x] Anthropic backend implementation
- [x] Procedural macro for deriving `LLMModel`
- [x] Schema generation functionality
- [x] Field-level attributes (description, example, examples)
- [x] Container-level attributes (description, title, examples)
- [x] Serde integration (rename_all)
- [x] Array literal support for examples
- [x] Custom validation capabilities
- [x] Support for nested structures
- [x] Rich validation API with custom domain rules
- [ ] Support for enums with associated data
- [ ] Streaming responses
- [ ] Support for more LLM providers

## License

This project is licensed under the MIT License - see the LICENSE file for details.
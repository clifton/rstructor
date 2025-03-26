# RStructor: Structured LLM Outputs for Rust

RStructor is a Rust library for defining structured data models and getting Large Language Models (LLMs) to fill them, with automatic validation. It provides a type-safe way to obtain structured data from LLMs by defining your schemas with Rust structs and enums.

## Features

- **📝 Structured Definitions**: Define your data models as Rust structs/enums with annotations for LLM guidance
- **🔄 JSON Schema Generation**: Automatically generate JSON Schema for your models to guide LLM outputs
- **✅ Validation**: Ensure LLM responses match your defined structure and types
- **🔌 Multiple LLM Providers**: Support for OpenAI and Anthropic APIs, with an extensible backend system
- **🔄 Async API**: Fully async API for efficient network operations

## Status: Early Development

This library is in early development. The core interfaces and traits are defined, but the procedural macro for deriving `LLMModel` is not yet implemented. Stay tuned for updates!

## Example (Future API)

```rust
use rstructor::{LLMModel, OpenAIClient}; 
use serde::{Serialize, Deserialize};

// Define your data model
#[derive(LLMModel, Serialize, Deserialize)]
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

// Use with OpenAI
async fn get_movie_info() -> Result<MovieInfo, Box<dyn std::error::Error>> {
    let client = OpenAIClient::new("YOUR_API_KEY")?
        .model(OpenAIModel::Gpt4)
        .temperature(0.0)
        .build();
    
    let prompt = "Tell me about the movie Inception";
    let movie: MovieInfo = client.generate_struct(prompt).await?;
    
    println!("Title: {}", movie.title);
    println!("Director: {}", movie.director);
    println!("Year: {}", movie.year);
    
    Ok(movie)
}
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
- [x] Basic validation capabilities
- [ ] Enhanced validation with custom validators
- [ ] Support for enums with associated data
- [ ] Streaming responses
- [ ] Support for more LLM providers

## License

This project is licensed under the MIT License - see the LICENSE file for details.
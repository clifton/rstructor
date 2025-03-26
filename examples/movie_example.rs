use rstructor::{LLMModel, SchemaType};
use serde::{Deserialize, Serialize};

#[derive(LLMModel, Serialize, Deserialize, Debug)]
struct Person {
    #[llm(description = "Full name of the person")]
    name: String,
    
    #[llm(description = "Role of the person in the production", optional)]
    role: Option<String>,
}

#[derive(LLMModel, Serialize, Deserialize, Debug)]
struct MovieInfo {
    #[llm(description = "Title of the movie")]
    title: String,
    
    #[llm(description = "Name of the director", example = "Christopher Nolan")]
    director: String,
    
    #[llm(description = "Year the movie was released", example = 2010)]
    year: u16,
    
    #[llm(description = "Genres of the movie", example = ["Action", "Sci-Fi", "Adventure"])]
    genres: Vec<String>,
    
    #[llm(description = "IMDB rating from 0.0 to 10.0", examples = [7.5, 8.8, 9.2])]
    rating: Option<f32>,
    
    #[llm(description = "List of main cast members")]
    cast: Vec<Person>,
}

fn main() {
    // Get the schema for the movie info
    let schema = MovieInfo::schema();
    
    println!("Movie Info Schema:");
    println!("{}", serde_json::to_string_pretty(schema.to_json()).unwrap());
    
    // Sample movie instance
    let inception = MovieInfo {
        title: "Inception".to_string(),
        director: "Christopher Nolan".to_string(),
        year: 2010,
        genres: vec!["Action".to_string(), "Sci-Fi".to_string(), "Thriller".to_string()],
        rating: Some(8.8),
        cast: vec![
            Person {
                name: "Leonardo DiCaprio".to_string(),
                role: Some("Cobb".to_string()),
            },
            Person {
                name: "Joseph Gordon-Levitt".to_string(),
                role: Some("Arthur".to_string()),
            },
            Person {
                name: "Ellen Page".to_string(),
                role: Some("Ariadne".to_string()),
            },
        ],
    };
    
    // Serialize to JSON
    println!("\nSample Movie Info:");
    println!("{}", serde_json::to_string_pretty(&inception).unwrap());
}
use rstructor::{Instructor, LLMClient, OpenAIClient};
use serde::{Deserialize, Serialize};
use std::env;

/// This example demonstrates extracting structured movie information.

// Define our data model
#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "Detailed information about a movie",
      title = "MovieDetails",
      validate = "validate_movie",
      examples = [
        ::serde_json::json!({
            "title": "The Matrix",
            "director": "Lana and Lilly Wachowski",
            "release_year": 1999,
            "genre": ["Sci-Fi", "Action"],
            "rating": 8.7,
            "plot_summary": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."
        })
      ])]
struct Movie {
    #[llm(description = "Title of the movie")]
    title: String,

    #[llm(
        description = "Director(s) of the movie",
        example = "Christopher Nolan"
    )]
    director: String,

    #[llm(description = "Year the movie was released", example = 2010)]
    release_year: u16,

    #[llm(description = "List of genres for the movie",
          example = ["Drama", "Thriller"])]
    genre: Vec<String>,

    #[llm(description = "IMDB rating out of 10", example = 8.5)]
    rating: f32,

    #[llm(description = "Brief summary of the movie plot")]
    plot_summary: String,
}

// Custom validation function referenced by #[llm(validate = "validate_movie")]
fn validate_movie(movie: &Movie) -> rstructor::Result<()> {
    // Check that the rating is between 0 and 10
    if movie.rating < 0.0 || movie.rating > 10.0 {
        return Err(rstructor::RStructorError::ValidationError(format!(
            "Rating must be between 0 and 10, got {}",
            movie.rating
        )));
    }

    // Check that the release year is reasonable
    if movie.release_year < 1888 || movie.release_year > 2030 {
        return Err(rstructor::RStructorError::ValidationError(format!(
            "Release year must be between 1888 and 2030, got {}",
            movie.release_year
        )));
    }

    Ok(())
}

fn print_movie(provider: &str, movie: &Movie) {
    println!("\n{} Response:", provider);
    println!("  Title: {}", movie.title);
    println!("  Director: {}", movie.director);
    println!("  Year: {}", movie.release_year);
    println!("  Genres: {:?}", movie.genre);
    println!("  Rating: {:.1}", movie.rating);
    println!("  Plot: {}", movie.plot_summary);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = "Tell me about the movie Inception";

    let api_key =
        env::var("OPENAI_API_KEY").expect("Please set OPENAI_API_KEY environment variable");

    let client = OpenAIClient::new(api_key)?;
    let movie = client.materialize::<Movie>(prompt).await?;
    print_movie("OpenAI", &movie);

    Ok(())
}

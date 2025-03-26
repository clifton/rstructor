use rstructor::{LLMModel, LLMClient, OpenAIClient, OpenAIModel, AnthropicClient, AnthropicModel};
use serde::{Serialize, Deserialize};
use std::env;

// Define our data model
#[derive(LLMModel, Serialize, Deserialize, Debug)]
#[llm(description = "Detailed information about a movie",
      title = "MovieDetails",
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
    
    #[llm(description = "Director(s) of the movie", 
          example = "Christopher Nolan")]
    director: String,
    
    #[llm(description = "Year the movie was released", 
          example = 2010)]
    release_year: u16,
    
    #[llm(description = "List of genres for the movie", 
          example = ["Drama", "Thriller"])]
    genre: Vec<String>,
    
    #[llm(description = "IMDB rating out of 10", 
          example = 8.5)]
    rating: f32,
    
    #[llm(description = "Brief summary of the movie plot")]
    plot_summary: String,
}

// Custom validation logic
// Manually provide the validate method with custom validation logic
impl Movie {
    // Custom validation method
    fn validate(&self) -> rstructor::Result<()> {
        // Check that the rating is between 0 and 10
        if self.rating < 0.0 || self.rating > 10.0 {
            return Err(rstructor::RStructorError::ValidationError(
                format!("Rating must be between 0 and 10, got {}", self.rating)
            ));
        }
        
        // Check that the release year is reasonable
        if self.release_year < 1888 || self.release_year > 2030 {
            return Err(rstructor::RStructorError::ValidationError(
                format!("Release year must be between 1888 and 2030, got {}", self.release_year)
            ));
        }
        
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API keys from environment
    let openai_key = env::var("OPENAI_API_KEY").ok();
    let anthropic_key = env::var("ANTHROPIC_API_KEY").ok();
    
    // User prompt
    let prompt = "Tell me about the movie Inception";
    
    // Try OpenAI if key is available
    if let Some(api_key) = openai_key {
        println!("Using OpenAI...");
        
        let client = OpenAIClient::new(api_key)?
            .model(OpenAIModel::Gpt35Turbo)
            .temperature(0.0)
            .build();
        
        match client.generate_struct::<Movie>(prompt).await {
            Ok(movie) => {
                println!("\nOpenAI Response:");
                println!("Title: {}", movie.title);
                println!("Director: {}", movie.director);
                println!("Year: {}", movie.release_year);
                println!("Genres: {:?}", movie.genre);
                println!("Rating: {:.1}", movie.rating);
                println!("Plot: {}", movie.plot_summary);
            },
            Err(e) => println!("Error with OpenAI: {}", e),
        }
    } else {
        println!("Skipping OpenAI (API key not found)");
    }
    
    // Try Anthropic if key is available
    if let Some(api_key) = anthropic_key {
        println!("\nUsing Anthropic...");
        
        let client = AnthropicClient::new(api_key)?
            .model(AnthropicModel::Claude3Haiku) // Using smaller model for faster results
            .temperature(0.0)
            .build();
        
        match client.generate_struct::<Movie>(prompt).await {
            Ok(movie) => {
                println!("\nAnthropic Response:");
                println!("Title: {}", movie.title);
                println!("Director: {}", movie.director);
                println!("Year: {}", movie.release_year);
                println!("Genres: {:?}", movie.genre);
                println!("Rating: {:.1}", movie.rating);
                println!("Plot: {}", movie.plot_summary);
            },
            Err(e) => println!("Error with Anthropic: {}", e),
        }
    } else {
        println!("Skipping Anthropic (API key not found)");
    }
    
    Ok(())
}
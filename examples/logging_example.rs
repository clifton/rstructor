use rstructor::{
    AnthropicClient, AnthropicModel, Instructor, LLMClient,
    logging::{LogLevel, init_logging},
};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct WeatherForecast {
    location: String,
    current_temperature: f32,
    forecast: Vec<DayForecast>,
}

#[derive(Serialize, Deserialize, Debug)]
struct DayForecast {
    day: String,
    temperature: f32,
    #[serde(alias = "weather")]
    conditions: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with INFO level
    // You can also set RSTRUCTOR_LOG environment variable to override this
    // E.g.: RSTRUCTOR_LOG=debug,rstructor::backend=trace cargo run --example logging_example
    init_logging(LogLevel::Debug);

    // Get API key from environment variable
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set");

    // Create a client
    let client = AnthropicClient::new(api_key)?
        .model(AnthropicModel::Claude3Haiku)
        .temperature(0.2)
        .build();

    // Generate a structured forecast
    let prompt = "What's the weather forecast for Tokyo for the next 3 days?";

    // This line will be logged with spans and info
    let forecast: WeatherForecast = client
        .generate_struct_with_retry(prompt, Some(2), Some(true))
        .await?;

    println!("\n\nGenerated forecast for {}", forecast.location);
    println!("Current temperature: {}°C", forecast.current_temperature);
    println!("\nUpcoming forecast:");

    for day in forecast.forecast {
        println!("- {}: {}°C, {}", day.day, day.temperature, day.conditions);
    }

    Ok(())
}

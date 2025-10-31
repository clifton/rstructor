use rstructor::{
    AnthropicClient, AnthropicModel, Instructor, LLMClient,
    logging::{LogLevel, init_logging},
};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "Weather forecast for a location")]
struct WeatherForecast {
    #[llm(description = "Location/city name")]
    location: String,

    #[llm(description = "Current temperature in Celsius", example = 25.5)]
    current_temperature: f32,

    #[llm(description = "Forecast for upcoming days")]
    forecast: Vec<DayForecast>,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "Weather forecast for a specific day")]
struct DayForecast {
    #[llm(description = "Day of the week", example = "Monday")]
    day: String,

    #[llm(description = "Temperature in Celsius", example = 28.5)]
    temperature: f32,

    #[llm(description = "Weather conditions", example = "Sunny")]
    #[serde(alias = "weather")]
    conditions: String,
}

// Implement custom validation directly on the WeatherForecast struct
// The Instructor derive macro will call this method
impl WeatherForecast {
    fn validate(&self) -> rstructor::Result<()> {
        // Check that location is not empty
        if self.location.trim().is_empty() {
            return Err(rstructor::RStructorError::ValidationError(
                "Location cannot be empty".to_string(),
            ));
        }

        // Check temperature is in reasonable range
        if self.current_temperature < -100.0 || self.current_temperature > 70.0 {
            return Err(rstructor::RStructorError::ValidationError(format!(
                "Current temperature must be between -100 and 70°C, got {}",
                self.current_temperature
            )));
        }

        // Check that we have at least one forecast day
        if self.forecast.is_empty() {
            return Err(rstructor::RStructorError::ValidationError(
                "Forecast must include at least one day".to_string(),
            ));
        }

        // Validate each day forecast
        for day_forecast in &self.forecast {
            // Check that day is not empty
            if day_forecast.day.trim().is_empty() {
                return Err(rstructor::RStructorError::ValidationError(
                    "Day cannot be empty".to_string(),
                ));
            }

            // Check temperature is in reasonable range
            if day_forecast.temperature < -100.0 || day_forecast.temperature > 70.0 {
                return Err(rstructor::RStructorError::ValidationError(format!(
                    "Forecast temperature must be between -100 and 70°C, got {}",
                    day_forecast.temperature
                )));
            }

            // Check that conditions is not empty
            if day_forecast.conditions.trim().is_empty() {
                return Err(rstructor::RStructorError::ValidationError(
                    "Weather conditions cannot be empty".to_string(),
                ));
            }
        }

        Ok(())
    }
}

// Implement validation for DayForecast
// This will be called by the derived Instructor implementation
impl DayForecast {
    fn validate(&self) -> rstructor::Result<()> {
        // Check that day is not empty
        if self.day.trim().is_empty() {
            return Err(rstructor::RStructorError::ValidationError(
                "Day cannot be empty".to_string(),
            ));
        }

        // Check temperature is in reasonable range
        if self.temperature < -100.0 || self.temperature > 70.0 {
            return Err(rstructor::RStructorError::ValidationError(format!(
                "Forecast temperature must be between -100 and 70°C, got {}",
                self.temperature
            )));
        }

        // Check that conditions is not empty
        if self.conditions.trim().is_empty() {
            return Err(rstructor::RStructorError::ValidationError(
                "Weather conditions cannot be empty".to_string(),
            ));
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with DEBUG level to show detailed logs including retries
    // You can also set RSTRUCTOR_LOG environment variable to override this
    // E.g.: RSTRUCTOR_LOG=debug,rstructor::backend=trace cargo run --example logging_example
    init_logging(LogLevel::Debug);

    println!("Starting weather forecast example with detailed logging...");
    println!("This example demonstrates retry logic with validation errors.");

    // Get API key from environment variable with better error handling
    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("⚠️  ANTHROPIC_API_KEY environment variable not set!");
            eprintln!("Please set it with: export ANTHROPIC_API_KEY=your_api_key");
            eprintln!(
                "For demonstration purposes, using a fake key (will show API errors in logs)"
            );
            "dummy-key-for-demonstration".to_string()
        }
    };

    // Create a client with higher temperature to increase chances of validation errors
    let client = AnthropicClient::new(api_key)?
        .model(AnthropicModel::ClaudeSonnet4)
        .temperature(0.7); // Higher temperature = more creativity = more validation errors

    println!("\nSending request to Anthropic API with increased randomness...");
    println!("Will retry up to 3 times on validation errors with detailed logging.\n");

    // Generate a structured forecast with more specific prompt and increased retries
    let prompt = "What's the detailed weather forecast for Tokyo for the next 3 days? Include temperatures in Celsius and weather conditions for each day.

CRITICAL REQUIREMENTS - ALL FIELDS ARE REQUIRED:
1. The 'location' field is REQUIRED (city name).
2. The 'current_temperature' field is REQUIRED (number in Celsius).
3. The 'forecast' field must be an array of objects, where each object has:
   - 'day' (REQUIRED - day of the week like 'Monday', 'Tuesday', etc.)
   - 'temperature' (REQUIRED - number in Celsius)
   - 'conditions' (REQUIRED - weather description like 'Sunny', 'Cloudy', etc.)
   ALL THREE FIELDS ARE REQUIRED FOR EACH FORECAST ITEM.";

    // This line will be logged with spans and info - using 3 retries for more chances to see retry logs
    let forecast_result = client
        .generate_struct_with_retry::<WeatherForecast>(prompt, Some(3), Some(true))
        .await;

    match forecast_result {
        Ok(forecast) => {
            println!("\n✅ Successfully generated forecast after potential retries!");
            println!("\nGenerated forecast for {}", forecast.location);
            println!("Current temperature: {}°C", forecast.current_temperature);
            println!("\nUpcoming forecast:");

            for day in forecast.forecast {
                println!("- {}: {}°C, {}", day.day, day.temperature, day.conditions);
            }
        }
        Err(e) => {
            println!("\n❌ Failed to generate forecast after retries");
            println!("Error: {}", e);
            println!("\nThe logs above show the detailed retry attempts and validation errors.");
        }
    }

    Ok(())
}

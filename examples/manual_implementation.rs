use rstructor::{Schema, SchemaType};
use rstructor::schema::SchemaBuilder;
use serde::{Deserialize, Serialize};
use serde_json::json;

// Define a struct without using the derive macro
#[derive(Debug, Serialize, Deserialize)]
struct WeatherInfo {
    city: String,
    temperature: f32,
    description: Option<String>,
}

// Manually implement SchemaType for the struct
impl SchemaType for WeatherInfo {
    fn schema() -> Schema {
        SchemaBuilder::object()
            .title("WeatherInfo")
            .description("Weather information for a city")
            .property(
                "city",
                json!({
                    "type": "string",
                    "description": "City name to get weather for"
                }),
                true
            )
            .property(
                "temperature",
                json!({
                    "type": "number",
                    "description": "Current temperature in Celsius",
                    "example": 22.5
                }),
                true
            )
            .property(
                "description",
                json!({
                    "type": "string",
                    "description": "Weather description"
                }),
                false
            )
            .build()
    }

    fn schema_name() -> Option<String> {
        Some("weather_info".to_string())
    }
}

fn main() {
    // Get the schema
    let schema = WeatherInfo::schema();
    
    println!("Weather Info Schema:");
    println!("{}", serde_json::to_string_pretty(schema.to_json()).unwrap());
    
    // Create a sample instance
    let weather = WeatherInfo {
        city: "Paris".to_string(),
        temperature: 25.5,
        description: Some("Sunny".to_string()),
    };
    
    // Serialize to JSON
    println!("\nSample Weather Info:");
    println!("{}", serde_json::to_string_pretty(&weather).unwrap());
}
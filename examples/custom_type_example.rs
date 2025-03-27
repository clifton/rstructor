use rstructor::schema::CustomTypeSchema;
use rstructor::{Instructor, Schema, SchemaType};
use serde::{Deserialize, Serialize};
use serde_json::json;

// We'll create a simple date struct to show the implementation pattern
// In a real application, you would typically implement this for chrono::DateTime or other date types

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CustomDate {
    year: u16,
    month: u8,
    day: u8,
}

// Implement the CustomTypeSchema trait for our custom date type
impl CustomTypeSchema for CustomDate {
    fn schema_type() -> &'static str {
        "string"
    }

    fn schema_format() -> Option<&'static str> {
        Some("date")
    }

    fn schema_description() -> Option<String> {
        Some("A date in YYYY-MM-DD format".to_string())
    }

    fn schema_additional_properties() -> Option<serde_json::Value> {
        Some(json!({
            "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
            "examples": ["2023-01-15", "2024-03-27"]
        }))
    }
}

// Now let's create a struct that uses our custom date type
#[derive(Instructor, Serialize, Deserialize)]
struct Event {
    #[llm(description = "The name of the event")]
    name: String,

    #[llm(description = "When the event starts")]
    start_date: CustomDate,

    #[llm(description = "When the event ends (optional)")]
    end_date: Option<CustomDate>,

    #[llm(description = "List of dates for recurring events")]
    recurring_dates: Vec<CustomDate>,
}

// Debugging: Display direct schema for CustomDate
fn print_custom_date_schema() {
    println!("\nDirect CustomDate Schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(&CustomDate::json_schema()).unwrap()
    );
}

// For demonstration, manually implement SchemaType for CustomDate
// (This would typically be handled by the CustomTypeSchema trait in real usage)
impl SchemaType for CustomDate {
    fn schema() -> Schema {
        Schema::new(CustomDate::json_schema())
    }

    fn schema_name() -> Option<String> {
        Some("CustomDate".to_string())
    }
}

fn main() {
    // Display the schema for our Event type
    let schema = Event::schema();

    println!("Event Schema with Custom Date Types:");
    println!(
        "{}",
        serde_json::to_string_pretty(&schema.to_json()).unwrap()
    );

    // Print the direct schema from CustomTypeSchema
    print_custom_date_schema();

    // Create an example Event
    let event = Event {
        name: "Conference".to_string(),
        start_date: CustomDate {
            year: 2024,
            month: 3,
            day: 27,
        },
        end_date: Some(CustomDate {
            year: 2024,
            month: 3,
            day: 29,
        }),
        recurring_dates: vec![
            CustomDate {
                year: 2025,
                month: 3,
                day: 15,
            },
            CustomDate {
                year: 2025,
                month: 9,
                day: 15,
            },
        ],
    };

    // Serialize to JSON
    println!("\nExample Event:");
    println!("{}", serde_json::to_string_pretty(&event).unwrap());
}

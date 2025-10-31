use rstructor::{
    AnthropicClient, AnthropicModel, Instructor, OpenAIClient, OpenAIModel, RStructorError,
};
type Result<T> = rstructor::Result<T>;
use chrono::{NaiveDate, NaiveTime};
use serde::{Deserialize, Serialize};
use std::{env, io::stdin};

// Define data structures for event planning

#[derive(Instructor, Serialize, Deserialize, Debug, Clone)]
#[llm(description = "Represents a contact person")]
struct Contact {
    #[llm(description = "Name of the contact person", example = "John Smith")]
    name: String,

    #[llm(description = "Email address", example = "john.smith@example.com")]
    email: Option<String>,

    #[llm(description = "Phone number", example = "555-123-4567")]
    phone: Option<String>,

    #[llm(description = "Role or relationship", example = "Event organizer")]
    role: Option<String>,
}

#[derive(Instructor, Serialize, Deserialize, Debug, Clone)]
#[llm(description = "Represents a location")]
struct Location {
    #[llm(
        description = "Name of the venue or location",
        example = "Grand Plaza Hotel"
    )]
    name: String,

    #[llm(description = "Street address", example = "123 Main St")]
    address: String,

    #[llm(description = "City", example = "New York")]
    city: String,

    #[llm(description = "State or province", example = "NY")]
    state: Option<String>,

    #[llm(description = "Postal/ZIP code", example = "10001")]
    zip: Option<String>,

    #[llm(description = "Country", example = "USA")]
    country: Option<String>,

    #[llm(
        description = "Any special instructions for finding or accessing the location",
        example = "Enter through the south entrance"
    )]
    instructions: Option<String>,
}

#[derive(Instructor, Serialize, Deserialize, Debug, Clone)]
#[llm(description = "Represents a scheduled activity within an event")]
struct Activity {
    #[llm(
        description = "Name or title of the activity",
        example = "Welcome Reception"
    )]
    name: String,

    #[llm(description = "Start time in HH:MM format", example = "18:30")]
    start_time: String,

    #[llm(description = "End time in HH:MM format", example = "20:00")]
    end_time: String,

    #[llm(
        description = "Description of the activity",
        example = "Casual networking with drinks and appetizers"
    )]
    description: Option<String>,

    #[llm(
        description = "Location of this activity, if different from main event",
        example = "Garden Terrace"
    )]
    location: Option<String>,
}

#[derive(Instructor, Serialize, Deserialize, Debug, Clone)]
#[llm(description = "Information about an event to be organized",
      examples = [
        ::serde_json::json!({
            "event_name": "Annual Tech Conference",
            "event_type": "Conference",
            "description": "A gathering of industry professionals to discuss emerging trends in technology",
            "date": "2025-06-15",
            "start_time": "09:00",
            "end_time": "17:00",
            "location": {
                "name": "Metro Convention Center",
                "address": "400 Convention Way",
                "city": "San Francisco",
                "state": "CA",
                "zip": "94103",
                "country": "USA",
                "instructions": "Parking available in the south garage"
            },
            "estimated_attendees": 250,
            "contact": {
                "name": "Jane Smith",
                "email": "jane.smith@example.com",
                "phone": "555-987-6543",
                "role": "Event Coordinator"
            },
            "activities": [
                {
                    "name": "Registration & Breakfast",
                    "start_time": "08:00",
                    "end_time": "09:00",
                    "description": "Check-in and continental breakfast",
                    "location": "Main Lobby"
                },
                {
                    "name": "Keynote Speech",
                    "start_time": "09:15",
                    "end_time": "10:30",
                    "description": "Opening address by CEO",
                    "location": "Grand Ballroom"
                }
            ],
            "special_requirements": "Vegetarian lunch options, AV equipment for presentations",
            "estimated_budget": 15000
        })
      ])]
struct EventPlan {
    #[llm(description = "Name of the event", example = "Company Holiday Party")]
    event_name: String,

    #[llm(description = "Type of event", example = "Party")]
    event_type: String,

    #[llm(
        description = "Description of the event",
        example = "Annual celebration for employees and their families"
    )]
    description: String,

    #[llm(
        description = "Date of the event in YYYY-MM-DD format",
        example = "2023-12-15"
    )]
    date: String,

    #[llm(description = "Start time in HH:MM format", example = "18:00")]
    start_time: String,

    #[llm(description = "End time in HH:MM format", example = "22:00")]
    end_time: String,

    #[llm(description = "Location details for the event")]
    location: Location,

    #[llm(description = "Estimated number of attendees", example = 100)]
    estimated_attendees: u32,

    #[llm(description = "Primary contact person for the event")]
    contact: Contact,

    #[llm(description = "Schedule of activities during the event")]
    activities: Vec<Activity>,

    #[llm(
        description = "Any special requirements or notes",
        example = "Need vegetarian food options"
    )]
    special_requirements: Option<String>,

    #[llm(description = "Estimated budget in dollars", example = 5000)]
    estimated_budget: Option<f32>,
}

// Custom validation implementation
impl EventPlan {
    fn validate(&self) -> rstructor::Result<()> {
        // Validate date format
        if NaiveDate::parse_from_str(&self.date, "%Y-%m-%d").is_err() {
            return Err(RStructorError::ValidationError(format!(
                "Invalid date format: {}. Expected YYYY-MM-DD",
                self.date
            )));
        }

        // Validate times
        let validate_time = |time: &str| -> rstructor::Result<()> {
            if NaiveTime::parse_from_str(time, "%H:%M").is_err() {
                return Err(RStructorError::ValidationError(format!(
                    "Invalid time format: {}. Expected HH:MM",
                    time
                )));
            }
            Ok(())
        };

        validate_time(&self.start_time)?;
        validate_time(&self.end_time)?;

        // Validate activity times
        for activity in &self.activities {
            validate_time(&activity.start_time)?;
            validate_time(&activity.end_time)?;
        }

        // Make sure activities are within event timeframe
        let event_start = NaiveTime::parse_from_str(&self.start_time, "%H:%M").unwrap();
        let event_end = NaiveTime::parse_from_str(&self.end_time, "%H:%M").unwrap();

        for activity in &self.activities {
            let activity_start = NaiveTime::parse_from_str(&activity.start_time, "%H:%M").unwrap();
            let activity_end = NaiveTime::parse_from_str(&activity.end_time, "%H:%M").unwrap();

            if activity_start < event_start || activity_end > event_end {
                return Err(RStructorError::ValidationError(format!(
                    "Activity '{}' time ({}-{}) is outside event hours ({}-{})",
                    activity.name,
                    activity.start_time,
                    activity.end_time,
                    self.start_time,
                    self.end_time
                )));
            }
        }

        // Validate contact information
        if self.contact.email.is_none() && self.contact.phone.is_none() {
            return Err(RStructorError::ValidationError(
                "Contact must have either email or phone specified".to_string(),
            ));
        }

        Ok(())
    }
}

async fn process_event_request(
    client: &(impl rstructor::LLMClient + std::marker::Sync),
    description: &str,
) -> Result<EventPlan> {
    let prompt = format!(
        "Target JSON: EventPlan\n\nCRITICAL REQUIREMENTS - ALL FIELDS ARE REQUIRED:
1. 'event_name' (REQUIRED - string)
2. 'event_type' (REQUIRED - string)
3. 'description' (REQUIRED - string)
4. 'date' (REQUIRED - YYYY-MM-DD format)
5. 'start_time' (REQUIRED - HH:MM format)
6. 'end_time' (REQUIRED - HH:MM format)
7. 'location' (REQUIRED - object with name, address, city at minimum)
8. 'estimated_attendees' (REQUIRED - number)
9. 'contact' (REQUIRED - object with name and either email or phone)
10. 'activities' (REQUIRED - array of objects, each with name, start_time, end_time)

Based on the following description, create a detailed event plan:\n\n{}",
        description
    );

    // Use retry with up to 5 attempts if validation fails
    client
        .generate_struct_with_retry::<EventPlan>(&prompt, Some(5), Some(true))
        .await
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Get user input
    println!("Welcome to the AI Event Planner!");
    println!(
        "Please describe the event you want to plan (type 'done' on a new line when finished):"
    );

    let mut description = String::new();
    let mut line = String::new();

    loop {
        line.clear();
        stdin().read_line(&mut line)?;

        if line.trim() == "done" {
            break;
        }

        description.push_str(&line);
    }

    if description.trim().is_empty() {
        println!("No description provided. Using a sample description instead.");
        description = "I need to plan a team-building retreat for my company's marketing department. \
                     We have about 20 people and want to do it next month on a Friday. \
                     We'd like some outdoor activities and team exercises, ideally at a nice location \
                     near nature. Our budget is approximately $5000.".to_string();
    }

    // Select LLM client based on available API keys
    if let Ok(api_key) = env::var("OPENAI_API_KEY") {
        println!("\nProcessing your request with OpenAI...\n");

        let client = OpenAIClient::new(api_key)?
            .model(OpenAIModel::Gpt5)
            .temperature(0.3);

        match process_event_request(&client, &description).await {
            Ok(plan) => print_event_plan(&plan),
            Err(e) => {
                println!("Error: {}", e);
                if let rstructor::RStructorError::ValidationError(msg) = &e {
                    println!("\nValidation error details: {}", msg);
                }
            }
        }
    } else if let Ok(api_key) = env::var("ANTHROPIC_API_KEY") {
        println!("\nProcessing your request with Anthropic...\n");

        let client = AnthropicClient::new(api_key)?
            .model(AnthropicModel::ClaudeSonnet4)
            .temperature(0.3);

        match process_event_request(&client, &description).await {
            Ok(plan) => print_event_plan(&plan),
            Err(e) => {
                println!("Error: {}", e);
                if let rstructor::RStructorError::ValidationError(msg) = &e {
                    println!("\nValidation error details: {}", msg);
                }
            }
        }
    } else {
        println!("\nNo API keys found in environment variables.");
        println!("Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY to use this example.");
    }

    Ok(())
}

fn print_event_plan(plan: &EventPlan) {
    println!("===== EVENT PLAN =====");
    println!("Name: {}", plan.event_name);
    println!("Type: {}", plan.event_type);
    println!("Description: {}", plan.description);
    println!("Date: {}", plan.date);
    println!("Time: {} to {}", plan.start_time, plan.end_time);
    println!("Estimated Attendees: {}", plan.estimated_attendees);

    println!("\n--- LOCATION ---");
    println!("Venue: {}", plan.location.name);
    println!("Address: {}", plan.location.address);
    println!("City: {}", plan.location.city);
    if let Some(state) = &plan.location.state {
        println!("State: {}", state);
    }
    if let Some(zip) = &plan.location.zip {
        println!("ZIP: {}", zip);
    }
    if let Some(country) = &plan.location.country {
        println!("Country: {}", country);
    }
    if let Some(instructions) = &plan.location.instructions {
        println!("Instructions: {}", instructions);
    }

    println!("\n--- CONTACT ---");
    println!("Name: {}", plan.contact.name);
    if let Some(email) = &plan.contact.email {
        println!("Email: {}", email);
    }
    if let Some(phone) = &plan.contact.phone {
        println!("Phone: {}", phone);
    }
    if let Some(role) = &plan.contact.role {
        println!("Role: {}", role);
    }

    println!("\n--- SCHEDULE ---");
    for activity in &plan.activities {
        println!(
            "• {} to {}: {}",
            activity.start_time, activity.end_time, activity.name
        );
        if let Some(desc) = &activity.description {
            println!("  {}", desc);
        }
        if let Some(loc) = &activity.location {
            println!("  Location: {}", loc);
        }
    }

    if let Some(requirements) = &plan.special_requirements {
        println!("\nSpecial Requirements: {}", requirements);
    }

    if let Some(budget) = plan.estimated_budget {
        println!("\nEstimated Budget: ${:.2}", budget);
    }
}

use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};

// Simple enum with primitive associated data
#[derive(Instructor, Serialize, Deserialize, Debug)]
enum UserStatus {
    #[llm(description = "The user is online")]
    Online,

    #[llm(description = "The user is offline")]
    Offline,

    #[llm(description = "The user is away with an optional message")]
    Away(String),

    #[llm(description = "The user is busy until a specific time")]
    Busy(u32),
}

fn main() {
    // Print schema for UserStatus enum with associated data
    let user_status_schema = UserStatus::schema();
    println!("UserStatus Schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(&user_status_schema.to_json()).unwrap()
    );

    // Example instances
    let online_status = UserStatus::Online;
    let away_status = UserStatus::Away("Back in 10 minutes".to_string());
    let busy_status = UserStatus::Busy(60);

    // Print serialized representations
    println!("\nSerialized UserStatus instances:");
    println!(
        "Online: {}",
        serde_json::to_string_pretty(&online_status).unwrap()
    );
    println!(
        "Away: {}",
        serde_json::to_string_pretty(&away_status).unwrap()
    );
    println!(
        "Busy: {}",
        serde_json::to_string_pretty(&busy_status).unwrap()
    );
}

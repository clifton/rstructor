//! Example demonstrating serde rename attribute support.
//!
//! rstructor respects `#[serde(rename)]` and `#[serde(rename_all)]` attributes,
//! ensuring the generated JSON schema matches serde's serialization behavior.

use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};

/// Commit type enum with lowercase serialization
#[derive(Instructor, Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
enum CommitType {
    Fix,
    Feat,
    Docs,
    Refactor,
    Test,
    Chore,
}

/// Commit message struct demonstrating field renames
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct CommitMessage {
    /// Using #[serde(rename)] to use "type" (a Rust keyword) as the JSON key
    #[serde(rename = "type")]
    #[llm(description = "The type of commit (fix, feat, docs, etc.)")]
    commit_type: CommitType,

    #[llm(description = "Brief description of the change")]
    description: String,

    #[serde(rename = "breaking_change")]
    #[llm(description = "Whether this commit introduces a breaking change")]
    is_breaking: bool,
}

/// User profile with camelCase field names for JavaScript interop
#[derive(Instructor, Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct UserProfile {
    #[llm(description = "User's first name")]
    first_name: String,

    #[llm(description = "User's last name")]
    last_name: String,

    #[llm(description = "User's email address")]
    email_address: String,

    #[llm(description = "Whether the user is verified")]
    is_verified: bool,
}

/// API response with SCREAMING_SNAKE_CASE status codes
#[derive(Instructor, Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum StatusCode {
    Ok,
    NotFound,
    InternalError,
    BadRequest,
}

fn main() {
    // Demonstrate enum with rename_all = "lowercase"
    println!("=== CommitType Enum (lowercase) ===");
    let commit_type_schema = CommitType::schema();
    println!(
        "{}",
        serde_json::to_string_pretty(&commit_type_schema.to_json()).unwrap()
    );

    // Show that serialization matches the schema
    let fix = CommitType::Fix;
    println!(
        "\nSerialized CommitType::Fix: {}",
        serde_json::to_string(&fix).unwrap()
    );

    // Demonstrate struct with field renames
    println!("\n=== CommitMessage Struct (field renames) ===");
    let commit_schema = CommitMessage::schema();
    println!(
        "{}",
        serde_json::to_string_pretty(&commit_schema.to_json()).unwrap()
    );

    // Show that serialization matches the schema
    let commit = CommitMessage {
        commit_type: CommitType::Feat,
        description: "Add serde rename support".to_string(),
        is_breaking: false,
    };
    println!("\nSerialized CommitMessage:");
    println!("{}", serde_json::to_string_pretty(&commit).unwrap());

    // Demonstrate struct with rename_all = "camelCase"
    println!("\n=== UserProfile Struct (camelCase) ===");
    let user_schema = UserProfile::schema();
    println!(
        "{}",
        serde_json::to_string_pretty(&user_schema.to_json()).unwrap()
    );

    // Show that serialization matches the schema
    let user = UserProfile {
        first_name: "John".to_string(),
        last_name: "Doe".to_string(),
        email_address: "john@example.com".to_string(),
        is_verified: true,
    };
    println!("\nSerialized UserProfile:");
    println!("{}", serde_json::to_string_pretty(&user).unwrap());

    // Demonstrate enum with rename_all = "SCREAMING_SNAKE_CASE"
    println!("\n=== StatusCode Enum (SCREAMING_SNAKE_CASE) ===");
    let status_schema = StatusCode::schema();
    println!(
        "{}",
        serde_json::to_string_pretty(&status_schema.to_json()).unwrap()
    );

    let status = StatusCode::NotFound;
    println!(
        "\nSerialized StatusCode::NotFound: {}",
        serde_json::to_string(&status).unwrap()
    );
}

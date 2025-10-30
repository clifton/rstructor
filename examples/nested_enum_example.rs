//! Example demonstrating nested enum support
//!
//! This example shows that nested enums now work automatically with #[derive(Instructor)]
//! without requiring manual SchemaType implementations.
//!
//! Previously, users had to manually implement SchemaType for nested enums:
//! ```rust
//! impl SchemaType for NestedEnumType {
//!     fn schema() -> Schema { ... }
//! }
//! ```
//!
//! Now, nested enums work automatically with #[derive(Instructor)]!

use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};

// Define a simple enum
#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

// Define another enum
#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
enum Status {
    Pending,
    InProgress,
    Completed,
    Cancelled,
}

// Define a nested enum - enum variant containing another enum
// This now works automatically with #[derive(Instructor)]!
#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
enum TaskState {
    #[llm(description = "Task is pending with a priority level")]
    Pending {
        #[llm(description = "Priority level")]
        priority: Priority,
    },
    #[llm(description = "Task is in progress")]
    InProgress {
        #[llm(description = "Priority level")]
        priority: Priority,
        #[llm(description = "Current status")]
        status: Status,
    },
    #[llm(description = "Task is completed")]
    Completed {
        #[llm(description = "Completion status")]
        status: Status,
    },
}

// Define a struct that uses the nested enum
#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
struct Task {
    #[llm(description = "Task title")]
    title: String,

    #[llm(description = "Current task state with nested enums")]
    state: TaskState,
}

fn main() {
    // Demonstrate that schema generation works automatically
    println!("=== Nested Enum Schema Generation ===\n");

    // Generate schema for the nested enum - works automatically!
    let task_state_schema = TaskState::schema();
    let schema_json = task_state_schema.to_json();

    println!("TaskState schema (automatically generated):");
    println!("{}", serde_json::to_string_pretty(&schema_json).unwrap());
    println!();

    // Demonstrate deserialization works correctly
    println!("=== Deserialization Test ===\n");

    let json1 = serde_json::json!({
        "Pending": {
            "priority": "High"
        }
    });

    let task_state1: TaskState = serde_json::from_value(json1).unwrap();
    match task_state1 {
        TaskState::Pending { priority } => {
            println!(
                "✓ Deserialized Pending variant with priority: {:?}",
                priority
            );
            assert_eq!(priority, Priority::High);
        }
        _ => panic!("Expected Pending variant"),
    }

    let json2 = serde_json::json!({
        "InProgress": {
            "priority": "Critical",
            "status": "InProgress"
        }
    });

    let task_state2: TaskState = serde_json::from_value(json2).unwrap();
    match task_state2 {
        TaskState::InProgress { priority, status } => {
            println!(
                "✓ Deserialized InProgress variant with priority: {:?}, status: {:?}",
                priority, status
            );
            assert_eq!(priority, Priority::Critical);
            assert_eq!(status, Status::InProgress);
        }
        _ => panic!("Expected InProgress variant"),
    }

    // Demonstrate it works in structs too
    println!("\n=== Nested Enum in Struct ===\n");

    let task_json = serde_json::json!({
        "title": "Fix bug",
        "state": {
            "Pending": {
                "priority": "High"
            }
        }
    });

    let task: Task = serde_json::from_value(task_json).unwrap();
    println!("✓ Deserialized Task with nested enum state");
    assert_eq!(task.title, "Fix bug");
    match task.state {
        TaskState::Pending { priority } => {
            assert_eq!(priority, Priority::High);
            println!("  Task state: Pending with {:?} priority", priority);
        }
        _ => panic!("Expected Pending variant"),
    }

    println!("\n=== Success! ===");
    println!("Nested enums now work automatically with #[derive(Instructor)]!");
    println!("No manual SchemaType implementation needed!");
}

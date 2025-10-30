//! Comprehensive tests for nested enum functionality
//!
//! These tests ensure that nested enums work correctly in all scenarios:
//! - Enum fields in structs
//! - Nested enums in enum variants
//! - Arrays of enums
//! - Optional nested enums
//! - Schema generation
//! - Deserialization

#[cfg(test)]
mod nested_enum_tests {
    use rstructor::{Instructor, SchemaType};
    use serde::{Deserialize, Serialize};

    // ====== Simple nested enum tests ======

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    enum Status {
        Active,
        Inactive,
        Pending,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct User {
        #[llm(description = "User name")]
        name: String,

        #[llm(description = "User status")]
        status: Status,
    }

    #[test]
    fn test_enum_field_in_struct_schema() {
        let schema = User::schema();
        let schema_json = schema.to_json();

        // Verify parent struct schema
        assert_eq!(schema_json["type"], "object");
        assert_eq!(schema_json["title"], "User");

        // Verify status field exists and is an object (since Status is a custom type)
        let status_prop = &schema_json["properties"]["status"];
        assert!(status_prop.is_object());

        // Status should have a type (either "object" for custom types or use SchemaType)
        assert!(status_prop["type"].is_string());
    }

    #[test]
    fn test_enum_field_in_struct_deserialization() {
        let json = serde_json::json!({
            "name": "John Doe",
            "status": "Active"
        });

        let user: User = serde_json::from_value(json).unwrap();
        assert_eq!(user.name, "John Doe");
        assert_eq!(user.status, Status::Active);
    }

    // ====== Nested enum in enum variant tests ======

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    enum Priority {
        Low,
        Medium,
        High,
        Critical,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    enum TaskState {
        #[llm(description = "Task is pending")]
        Pending {
            #[llm(description = "Task priority")]
            priority: Priority,
        },
        #[llm(description = "Task is in progress")]
        InProgress {
            #[llm(description = "Task priority")]
            priority: Priority,
            #[llm(description = "Assignee name")]
            assignee: String,
        },
        #[llm(description = "Task is completed")]
        Completed {
            #[llm(description = "Completion time")]
            completed_at: String,
        },
    }

    #[test]
    fn test_nested_enum_in_enum_variant_schema() {
        let schema = TaskState::schema();
        let schema_json = schema.to_json();

        // Verify it's a oneOf schema (complex enum)
        assert!(schema_json["oneOf"].is_array());
        let variants = schema_json["oneOf"].as_array().unwrap();
        assert_eq!(variants.len(), 3);

        // Verify Pending variant has priority field
        let pending_variant = variants
            .iter()
            .find(|v| v.get("properties").and_then(|p| p.get("Pending")).is_some());
        assert!(pending_variant.is_some());

        let pending_props = pending_variant
            .unwrap()
            .get("properties")
            .unwrap()
            .get("Pending")
            .unwrap()
            .get("properties")
            .unwrap();

        // Priority field should exist
        assert!(pending_props.get("priority").is_some());
    }

    #[test]
    fn test_nested_enum_in_enum_variant_deserialization() {
        let json = serde_json::json!({
            "Pending": {
                "priority": "High"
            }
        });

        let task_state: TaskState = serde_json::from_value(json).unwrap();
        match task_state {
            TaskState::Pending { priority } => {
                assert_eq!(priority, Priority::High);
            }
            _ => panic!("Expected Pending variant"),
        }
    }

    // ====== Array of enums tests ======

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct TaskList {
        #[llm(description = "List name")]
        name: String,

        #[llm(description = "Task states")]
        tasks: Vec<TaskState>,
    }

    #[test]
    fn test_array_of_enums_schema() {
        let schema = TaskList::schema();
        let schema_json = schema.to_json();

        // Verify tasks field is an array
        let tasks_prop = &schema_json["properties"]["tasks"];
        assert_eq!(tasks_prop["type"], "array");

        // Verify items are objects (since TaskState is a complex enum)
        let items = &tasks_prop["items"];
        assert!(items.is_object());
        // Complex enums should have oneOf schema
        assert!(items.get("oneOf").is_some() || items.get("type").is_some());
    }

    #[test]
    fn test_array_of_enums_deserialization() {
        let json = serde_json::json!({
            "name": "My Tasks",
            "tasks": [
                {
                    "Pending": {
                        "priority": "Low"
                    }
                },
                {
                    "InProgress": {
                        "priority": "High",
                        "assignee": "Alice"
                    }
                }
            ]
        });

        let task_list: TaskList = serde_json::from_value(json).unwrap();
        assert_eq!(task_list.name, "My Tasks");
        assert_eq!(task_list.tasks.len(), 2);
    }

    // ====== Optional nested enum tests ======

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct UserProfile {
        #[llm(description = "User name")]
        name: String,

        #[llm(description = "Optional status")]
        status: Option<Status>,
    }

    #[test]
    fn test_optional_nested_enum_schema() {
        let schema = UserProfile::schema();
        let schema_json = schema.to_json();

        // Verify status field exists
        let status_prop = &schema_json["properties"]["status"];
        assert!(status_prop.is_object());

        // Status should not be in required fields
        let required = schema_json["required"].as_array().unwrap();
        assert!(!required.iter().any(|r| r == "status"));
    }

    #[test]
    fn test_optional_nested_enum_deserialization() {
        // With status
        let json1 = serde_json::json!({
            "name": "John",
            "status": "Active"
        });
        let profile1: UserProfile = serde_json::from_value(json1).unwrap();
        assert_eq!(profile1.status, Some(Status::Active));

        // Without status
        let json2 = serde_json::json!({
            "name": "Jane"
        });
        let profile2: UserProfile = serde_json::from_value(json2).unwrap();
        assert_eq!(profile2.status, None);
    }

    // ====== Deeply nested enum tests ======

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    enum DeepNestedEnum {
        Level1 {
            #[llm(description = "Nested status")]
            status: Status,
            #[llm(description = "Nested priority")]
            priority: Priority,
        },
    }

    #[test]
    fn test_deeply_nested_enum_schema() {
        let schema = DeepNestedEnum::schema();
        let schema_json = schema.to_json();

        // Verify it's a oneOf schema
        assert!(schema_json["oneOf"].is_array());
        let variants = schema_json["oneOf"].as_array().unwrap();
        assert_eq!(variants.len(), 1);

        // Verify Level1 variant has both status and priority fields
        let level1_variant = &variants[0];
        let level1_props = level1_variant
            .get("properties")
            .unwrap()
            .get("Level1")
            .unwrap()
            .get("properties")
            .unwrap();

        assert!(level1_props.get("status").is_some());
        assert!(level1_props.get("priority").is_some());
    }

    #[test]
    fn test_deeply_nested_enum_deserialization() {
        let json = serde_json::json!({
            "Level1": {
                "status": "Active",
                "priority": "High"
            }
        });

        let nested: DeepNestedEnum = serde_json::from_value(json).unwrap();
        match nested {
            DeepNestedEnum::Level1 { status, priority } => {
                assert_eq!(status, Status::Active);
                assert_eq!(priority, Priority::High);
            }
        }
    }
}

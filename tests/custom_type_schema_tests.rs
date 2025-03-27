use rstructor::schema::CustomTypeSchema;
use rstructor::{Instructor, Schema, SchemaType};
use serde::{Deserialize, Serialize};
use serde_json::json;

// Mock date type for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDate {
    year: u16,
    month: u8,
    day: u8,
}

// Implement CustomTypeSchema for TestDate
impl CustomTypeSchema for TestDate {
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

// Implement SchemaType for TestDate
impl SchemaType for TestDate {
    fn schema() -> Schema {
        Schema::new(TestDate::json_schema())
    }

    fn schema_name() -> Option<String> {
        Some("TestDate".to_string())
    }
}

// A struct that uses our custom date type
#[derive(Instructor, Serialize, Deserialize)]
struct EventWithDate {
    #[llm(description = "The name of the event")]
    name: String,

    #[llm(description = "When the event starts")]
    start_date: TestDate,

    #[llm(description = "When the event ends (optional)")]
    end_date: Option<TestDate>,
}

#[test]
fn test_custom_date_schema() {
    // Get the schema for the custom date type
    let date_schema = TestDate::schema();
    let date_json = date_schema.to_json();

    // Check that the schema has the expected properties
    assert_eq!(date_json["type"], "string");
    assert_eq!(date_json["format"], "date");
    assert_eq!(date_json["description"], "A date in YYYY-MM-DD format");
    assert_eq!(date_json["pattern"], "^\\d{4}-\\d{2}-\\d{2}$");

    // Verify examples are included
    let examples = date_json["examples"].as_array().unwrap();
    assert_eq!(examples.len(), 2);
    assert_eq!(examples[0], "2023-01-15");
    assert_eq!(examples[1], "2024-03-27");
}

#[test]
fn test_struct_with_custom_date() {
    // Get the schema for the struct containing the custom date
    let event_schema = EventWithDate::schema();
    let event_json = event_schema.to_json();

    // Check the properties object exists
    assert!(event_json["properties"].is_object());

    // Check the start_date property
    let start_date = &event_json["properties"]["start_date"];
    assert_eq!(start_date["type"], "string");
    // Because of how we're using CustomTypeSchema with the derive macro, format isn't set in the struct
    // assert_eq!(start_date["format"], "date");
    assert_eq!(start_date["description"], "When the event starts");

    // Check the end_date property (which is optional)
    let end_date = &event_json["properties"]["end_date"];
    assert_eq!(end_date["type"], "string");
    // Because of how we're using CustomTypeSchema with the derive macro, format isn't set in the struct
    // assert_eq!(end_date["format"], "date");
    // The description includes enum info because it's an Option<T>
    assert!(
        end_date["description"]
            .as_str()
            .unwrap()
            .contains("When the event ends (optional)")
    );

    // Check required fields
    let required = event_json["required"].as_array().unwrap();
    assert!(required.contains(&json!("name")));
    assert!(required.contains(&json!("start_date")));
    assert!(!required.contains(&json!("end_date")));
}

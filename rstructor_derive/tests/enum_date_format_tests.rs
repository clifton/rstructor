// Tests for date/UUID format handling in enum schema generation
use chrono::{NaiveDate, NaiveDateTime};
use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};

// Externally tagged enum (default) with struct variants containing date fields
#[derive(Instructor, Serialize, Deserialize, Debug)]
enum EventType {
    #[llm(description = "A single-day event")]
    SingleDay { name: String, date: NaiveDate },
    #[llm(description = "A multi-day event with start and end dates")]
    MultiDay {
        name: String,
        start: NaiveDate,
        end: NaiveDate,
    },
    #[llm(description = "A timestamped event")]
    Timestamped {
        name: String,
        timestamp: NaiveDateTime,
    },
}

// Internally tagged enum with struct variant containing a NaiveDate field
#[derive(Instructor, Serialize, Deserialize, Debug)]
#[serde(tag = "kind")]
enum InternallyTaggedEvent {
    #[llm(description = "A birthday event")]
    Birthday { person: String, date: NaiveDate },
    #[llm(description = "A meeting event")]
    Meeting {
        title: String,
        scheduled_at: NaiveDateTime,
    },
}

// Externally tagged enum with a Vec<NaiveDate> field
#[derive(Instructor, Serialize, Deserialize, Debug)]
enum Schedule {
    #[llm(description = "A recurring schedule with multiple dates")]
    Recurring { name: String, dates: Vec<NaiveDate> },
    #[llm(description = "A one-off event")]
    OneOff { name: String, date: NaiveDate },
}

// Externally tagged enum using fully qualified date/UUID paths.
#[derive(Instructor, Serialize, Deserialize, Debug)]
enum QualifiedSchedule {
    Entry {
        date: chrono::NaiveDate,
        created_at: std::option::Option<chrono::NaiveDateTime>,
        id: uuid::Uuid,
        checkpoints: std::vec::Vec<chrono::NaiveDate>,
    },
}

#[test]
fn test_externally_tagged_enum_naive_date_format() {
    let schema = EventType::schema();
    let json = schema.to_json();

    // Externally tagged enums use anyOf
    let variants = json["anyOf"]
        .as_array()
        .expect("Schema should use anyOf for externally tagged enums");

    // Find the SingleDay variant
    let single_day = variants
        .iter()
        .find(|v| {
            v["properties"]
                .as_object()
                .map_or(false, |props| props.contains_key("SingleDay"))
        })
        .expect("Should find SingleDay variant");

    let single_day_props = &single_day["properties"]["SingleDay"]["properties"];
    let date_prop = &single_day_props["date"];

    assert_eq!(
        date_prop["type"], "string",
        "NaiveDate should be string type"
    );
    assert_eq!(
        date_prop["format"], "date",
        "NaiveDate in externally tagged enum should have format 'date'"
    );
}

#[test]
fn test_externally_tagged_enum_naive_datetime_format() {
    let schema = EventType::schema();
    let json = schema.to_json();

    let variants = json["anyOf"]
        .as_array()
        .expect("Schema should use anyOf for externally tagged enums");

    // Find the Timestamped variant
    let timestamped = variants
        .iter()
        .find(|v| {
            v["properties"]
                .as_object()
                .map_or(false, |props| props.contains_key("Timestamped"))
        })
        .expect("Should find Timestamped variant");

    let timestamped_props = &timestamped["properties"]["Timestamped"]["properties"];
    let ts_prop = &timestamped_props["timestamp"];

    assert_eq!(
        ts_prop["type"], "string",
        "NaiveDateTime should be string type"
    );
    assert_eq!(
        ts_prop["format"], "date-time",
        "NaiveDateTime in externally tagged enum should have format 'date-time'"
    );
}

#[test]
fn test_internally_tagged_enum_naive_date_format() {
    let schema = InternallyTaggedEvent::schema();
    let json = schema.to_json();
    // Internally tagged enums also use anyOf
    let variants = json["anyOf"]
        .as_array()
        .expect("Schema should use anyOf for internally tagged enums");

    // Find the Birthday variant (has "kind" with enum: ["Birthday"] in properties)
    let birthday = variants
        .iter()
        .find(|v| {
            v["properties"]["kind"]["enum"]
                .as_array()
                .and_then(|arr| arr.first())
                .and_then(|c| c.as_str())
                == Some("Birthday")
        })
        .expect("Should find Birthday variant");

    let date_prop = &birthday["properties"]["date"];

    assert_eq!(
        date_prop["type"], "string",
        "NaiveDate in internally tagged enum should be string type"
    );
    assert_eq!(
        date_prop["format"], "date",
        "NaiveDate in internally tagged enum should have format 'date'"
    );
}

#[test]
fn test_enum_vec_naive_date_items_format() {
    let schema = Schedule::schema();
    let json = schema.to_json();

    let variants = json["anyOf"]
        .as_array()
        .expect("Schema should use anyOf for externally tagged enums");

    // Find the Recurring variant
    let recurring = variants
        .iter()
        .find(|v| {
            v["properties"]
                .as_object()
                .map_or(false, |props| props.contains_key("Recurring"))
        })
        .expect("Should find Recurring variant");

    let recurring_props = &recurring["properties"]["Recurring"]["properties"];
    let dates_prop = &recurring_props["dates"];

    assert_eq!(
        dates_prop["type"], "array",
        "Vec<NaiveDate> should be array type"
    );

    let items = &dates_prop["items"];
    assert_eq!(
        items["type"], "string",
        "Vec<NaiveDate> items should be string type"
    );
    assert_eq!(
        items["format"], "date",
        "Vec<NaiveDate> items in enum should have format 'date'"
    );
}

#[test]
fn test_enum_fully_qualified_date_uuid_formats() {
    let schema = QualifiedSchedule::schema();
    let json = schema.to_json();
    let variants = json["anyOf"]
        .as_array()
        .expect("Schema should use anyOf for externally tagged enums");

    let entry = variants
        .iter()
        .find(|v| {
            v["properties"]
                .as_object()
                .is_some_and(|props| props.contains_key("Entry"))
        })
        .expect("Should find Entry variant");

    let props = &entry["properties"]["Entry"]["properties"];
    assert_eq!(props["date"]["format"], "date");
    assert_eq!(props["created_at"]["format"], "date-time");
    assert_eq!(props["id"]["format"], "uuid");
    assert_eq!(props["checkpoints"]["items"]["format"], "date");

    let required = entry["properties"]["Entry"]["required"]
        .as_array()
        .expect("Entry should have required fields");
    assert!(!required.iter().any(|v| v == "created_at"));
}

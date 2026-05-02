// Tests for date/UUID format handling in schema generation
use chrono::{NaiveDate, NaiveDateTime};
use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Struct with NaiveDate (should get format "date")
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct WithNaiveDate {
    date: NaiveDate,
}

// Struct with NaiveDateTime (should get format "date-time")
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct WithNaiveDateTime {
    #[llm(description = "A date-time field")]
    timestamp: NaiveDateTime,
}

// Struct with Option<NaiveDate> (should still get format "date")
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct WithOptionalNaiveDate {
    date: Option<NaiveDate>,
}

// Struct with Option<NaiveDateTime> (should still get format "date-time")
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct WithOptionalNaiveDateTime {
    #[llm(description = "An optional date-time")]
    timestamp: Option<NaiveDateTime>,
}

// Struct with Option<Uuid> (should still get format "uuid")
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct WithOptionalUuid {
    #[llm(description = "An optional UUID")]
    id: Option<Uuid>,
}

// Struct with Vec<NaiveDate> (items should get format "date")
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct WithVecNaiveDate {
    #[llm(description = "A list of dates")]
    dates: Vec<NaiveDate>,
}

// Struct with Vec<NaiveDateTime> (items should get format "date-time")
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct WithVecNaiveDateTime {
    #[llm(description = "A list of date-times")]
    timestamps: Vec<NaiveDateTime>,
}

// Struct using fully qualified paths instead of imported type names.
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct WithQualifiedDateTypes {
    date: chrono::NaiveDate,
    timestamp: std::option::Option<chrono::NaiveDateTime>,
    id: uuid::Uuid,
    dates: std::vec::Vec<chrono::NaiveDate>,
}

#[test]
fn test_naive_date_gets_date_format() {
    let schema = WithNaiveDate::schema();
    let json = schema.to_json();
    let date_prop = &json["properties"]["date"];

    assert_eq!(date_prop["type"], "string");
    assert_eq!(
        date_prop["format"], "date",
        "NaiveDate should have format 'date', not 'date-time'"
    );
    assert!(
        date_prop["description"]
            .as_str()
            .unwrap()
            .contains("YYYY-MM-DD"),
        "NaiveDate description should mention YYYY-MM-DD"
    );
}

#[test]
fn test_naive_datetime_gets_datetime_format() {
    let schema = WithNaiveDateTime::schema();
    let json = schema.to_json();
    let ts_prop = &json["properties"]["timestamp"];

    assert_eq!(ts_prop["type"], "string");
    assert_eq!(
        ts_prop["format"], "date-time",
        "NaiveDateTime should have format 'date-time'"
    );
}

#[test]
fn test_option_naive_date_gets_date_format() {
    let schema = WithOptionalNaiveDate::schema();
    let json = schema.to_json();
    let date_prop = &json["properties"]["date"];

    assert_eq!(date_prop["type"], "string");
    assert_eq!(
        date_prop["format"], "date",
        "Option<NaiveDate> should have format 'date', not be missing"
    );
    assert!(
        date_prop["description"]
            .as_str()
            .unwrap()
            .contains("YYYY-MM-DD"),
        "Option<NaiveDate> description should mention YYYY-MM-DD"
    );

    // Option fields should NOT be in required
    let required = json["required"].as_array().unwrap();
    assert!(
        !required.iter().any(|v| v == "date"),
        "Option<NaiveDate> should not be required"
    );
}

#[test]
fn test_option_naive_datetime_gets_datetime_format() {
    let schema = WithOptionalNaiveDateTime::schema();
    let json = schema.to_json();
    let ts_prop = &json["properties"]["timestamp"];

    assert_eq!(ts_prop["type"], "string");
    assert_eq!(
        ts_prop["format"], "date-time",
        "Option<NaiveDateTime> should have format 'date-time', not be missing"
    );

    // Option fields should NOT be in required
    let required = json["required"].as_array().unwrap();
    assert!(
        !required.iter().any(|v| v == "timestamp"),
        "Option<NaiveDateTime> should not be required"
    );
}

#[test]
fn test_option_uuid_gets_uuid_format() {
    let schema = WithOptionalUuid::schema();
    let json = schema.to_json();
    let id_prop = &json["properties"]["id"];

    assert_eq!(id_prop["type"], "string");
    assert_eq!(
        id_prop["format"], "uuid",
        "Option<Uuid> should have format 'uuid', not be missing"
    );

    // Option fields should NOT be in required
    let required = json["required"].as_array().unwrap();
    assert!(
        !required.iter().any(|v| v == "id"),
        "Option<Uuid> should not be required"
    );
}

#[test]
fn test_vec_naive_date_items_get_date_format() {
    let schema = WithVecNaiveDate::schema();
    let json = schema.to_json();
    let dates_prop = &json["properties"]["dates"];

    assert_eq!(dates_prop["type"], "array");

    let items = &dates_prop["items"];
    assert_eq!(items["type"], "string");
    assert_eq!(
        items["format"], "date",
        "Vec<NaiveDate> items should have format 'date', not 'date-time'"
    );
    assert!(
        items["description"]
            .as_str()
            .unwrap()
            .contains("YYYY-MM-DD"),
        "Vec<NaiveDate> items description should mention YYYY-MM-DD"
    );
}

#[test]
fn test_vec_naive_datetime_items_get_datetime_format() {
    let schema = WithVecNaiveDateTime::schema();
    let json = schema.to_json();
    let ts_prop = &json["properties"]["timestamps"];

    assert_eq!(ts_prop["type"], "array");

    let items = &ts_prop["items"];
    assert_eq!(items["type"], "string");
    assert_eq!(
        items["format"], "date-time",
        "Vec<NaiveDateTime> items should have format 'date-time'"
    );
}

#[test]
fn test_fully_qualified_date_uuid_types_get_formats() {
    let schema = WithQualifiedDateTypes::schema();
    let json = schema.to_json();

    assert_eq!(json["properties"]["date"]["type"], "string");
    assert_eq!(json["properties"]["date"]["format"], "date");

    assert_eq!(json["properties"]["timestamp"]["type"], "string");
    assert_eq!(json["properties"]["timestamp"]["format"], "date-time");

    assert_eq!(json["properties"]["id"]["type"], "string");
    assert_eq!(json["properties"]["id"]["format"], "uuid");

    assert_eq!(json["properties"]["dates"]["type"], "array");
    assert_eq!(json["properties"]["dates"]["items"]["type"], "string");
    assert_eq!(json["properties"]["dates"]["items"]["format"], "date");

    let required = json["required"].as_array().unwrap();
    assert!(!required.iter().any(|v| v == "timestamp"));
}

// Tests for edge cases in the derive macro
use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};

// Test struct with empty attributes
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct EmptyAttributes {
    // Field with no attributes
    field_no_attrs: String,

    // Field with empty description
    #[llm(description = "")]
    field_empty_desc: String,

    // Optional field with no attributes
    optional_no_attrs: Option<String>,
}

// Struct with only one field
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct SingleField {
    #[llm(description = "The only field")]
    only_field: String,
}

// Struct with unusual but valid field names
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct UnusualFieldNames {
    #[llm(description = "Field with underscores")]
    field_with_underscores: String,

    #[llm(description = "Field with numbers")]
    field123: String,

    #[serde(rename = "renamed-field")]
    #[llm(description = "Field with rename attribute")]
    internal_name: String,
}

// Test numeric edge cases
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct NumericEdgeCases {
    #[llm(description = "Large integer", example = 2147483647)]
    large_int: i32,

    #[llm(description = "Small integer", example = -2147483648)]
    small_int: i32,

    #[llm(description = "Integer zero", example = 0)]
    zero_int: i32,

    #[llm(description = "Float zero", example = 0.0)]
    zero_float: f64,

    #[llm(description = "Very small float", example = 1e-10)]
    very_small_float: f64,

    #[llm(description = "Very large float", example = 1e+10)]
    very_large_float: f64,
}

#[test]
fn test_empty_attributes() {
    let schema = EmptyAttributes::schema();
    let schema_json = schema.to_json();

    // Fields with no attributes should still be in properties
    assert!(
        schema_json["properties"]
            .as_object()
            .unwrap()
            .contains_key("field_no_attrs")
    );

    // Field with empty description should have an empty description
    assert_eq!(
        schema_json["properties"]["field_empty_desc"]["description"],
        ""
    );

    // Optional field should not be in required list
    let required = schema_json["required"].as_array().unwrap();
    assert!(!required.iter().any(|v| v == "optional_no_attrs"));
}

#[test]
fn test_single_field() {
    let schema = SingleField::schema();
    let schema_json = schema.to_json();

    // Schema should have only one property
    assert_eq!(schema_json["properties"].as_object().unwrap().len(), 1);
    assert!(
        schema_json["properties"]
            .as_object()
            .unwrap()
            .contains_key("only_field")
    );

    // Required array should have only one element
    assert_eq!(schema_json["required"].as_array().unwrap().len(), 1);
    assert_eq!(schema_json["required"][0], "only_field");
}

#[test]
fn test_unusual_field_names() {
    let schema = UnusualFieldNames::schema();
    let schema_json = schema.to_json();

    // Check fields exist
    assert!(
        schema_json["properties"]
            .as_object()
            .unwrap()
            .contains_key("field_with_underscores")
    );
    assert!(
        schema_json["properties"]
            .as_object()
            .unwrap()
            .contains_key("field123")
    );

    // Our implementation currently doesn't respect serde rename attributes
    assert!(
        schema_json["properties"]
            .as_object()
            .unwrap()
            .contains_key("internal_name")
    );
}

#[test]
fn test_numeric_edge_cases() {
    let schema = NumericEdgeCases::schema();
    let schema_json = schema.to_json();

    // Check numeric values
    assert_eq!(
        schema_json["properties"]["large_int"]["example"],
        2147483647
    );
    assert_eq!(
        schema_json["properties"]["small_int"]["example"],
        -2147483648
    );
    assert_eq!(schema_json["properties"]["zero_int"]["example"], 0);
    assert_eq!(schema_json["properties"]["zero_float"]["example"], 0.0);

    // Very small float should be representable
    assert!(
        schema_json["properties"]["very_small_float"]["example"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(
        schema_json["properties"]["very_small_float"]["example"]
            .as_f64()
            .unwrap()
            < 0.000001
    );

    // Very large float should be representable
    assert!(
        schema_json["properties"]["very_large_float"]["example"]
            .as_f64()
            .unwrap()
            > 1000000000.0
    );
}

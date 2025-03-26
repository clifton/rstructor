// Integration tests for array literal syntax in the derive macro
use rstructor::{LLMModel, SchemaType};
use serde::{Serialize, Deserialize};

// Test struct with various array literal syntaxes
#[derive(LLMModel, Serialize, Deserialize, Debug)]
struct ArrayLiteralTests {
    // String array with array literal syntax
    #[llm(description = "Array of strings", example = ["apple", "banana", "cherry"])]
    string_array: Vec<String>,
    
    // Integer array with array literal syntax
    #[llm(description = "Array of integers", example = [1, 2, 3, 4])]
    int_array: Vec<i32>,
    
    // Float array with array literal syntax
    #[llm(description = "Array of floats", example = [1.1, 2.2, 3.3])]
    float_array: Vec<f64>,
    
    // Boolean array with array literal syntax
    #[llm(description = "Array of booleans", example = [true, false, true])]
    bool_array: Vec<bool>,
    
    // Mixed array (even though Rust wouldn't allow this directly, the schema can)
    #[llm(description = "Array of mixed types", example = ["string", 42, true, 3.14])]
    mixed_array: Vec<serde_json::Value>,
    
    // Empty array
    #[llm(description = "Empty array", example = [])]
    empty_array: Vec<String>,
    
    // Multiple examples using array literal syntax
    #[llm(description = "Float values", examples = [10.5, 20.5, 30.5])]
    example_floats: f64,
    
    // Multiple string examples
    #[llm(description = "Name examples", examples = ["John", "Jane", "Alice", "Bob"])]
    example_names: String,
}

#[test]
fn test_array_literal_string_array() {
    let schema = ArrayLiteralTests::schema();
    let schema_json = schema.to_json();
    
    // Check string array
    let string_array_example = &schema_json["properties"]["string_array"]["example"];
    assert!(string_array_example.is_array());
    let array = string_array_example.as_array().unwrap();
    assert_eq!(array.len(), 3);
    assert_eq!(array[0], "apple");
    assert_eq!(array[1], "banana");
    assert_eq!(array[2], "cherry");
}

#[test]
fn test_array_literal_int_array() {
    let schema = ArrayLiteralTests::schema();
    let schema_json = schema.to_json();
    
    // Check int array
    let int_array_example = &schema_json["properties"]["int_array"]["example"];
    assert!(int_array_example.is_array());
    let array = int_array_example.as_array().unwrap();
    assert_eq!(array.len(), 4);
    assert_eq!(array[0], 1);
    assert_eq!(array[1], 2);
    assert_eq!(array[2], 3);
    assert_eq!(array[3], 4);
}

#[test]
fn test_array_literal_float_array() {
    let schema = ArrayLiteralTests::schema();
    let schema_json = schema.to_json();
    
    // Check float array
    let float_array_example = &schema_json["properties"]["float_array"]["example"];
    assert!(float_array_example.is_array());
    let array = float_array_example.as_array().unwrap();
    assert_eq!(array.len(), 3);
    assert_eq!(array[0], 1.1);
    assert_eq!(array[1], 2.2);
    assert_eq!(array[2], 3.3);
}

#[test]
fn test_array_literal_bool_array() {
    let schema = ArrayLiteralTests::schema();
    let schema_json = schema.to_json();
    
    // Check bool array
    let bool_array_example = &schema_json["properties"]["bool_array"]["example"];
    assert!(bool_array_example.is_array());
    let array = bool_array_example.as_array().unwrap();
    assert_eq!(array.len(), 3);
    assert_eq!(array[0], true);
    assert_eq!(array[1], false);
    assert_eq!(array[2], true);
}

#[test]
fn test_array_literal_mixed_array() {
    let schema = ArrayLiteralTests::schema();
    let schema_json = schema.to_json();
    
    // Check mixed array
    let mixed_array_example = &schema_json["properties"]["mixed_array"]["example"];
    assert!(mixed_array_example.is_array());
    let array = mixed_array_example.as_array().unwrap();
    assert_eq!(array.len(), 4);
    assert_eq!(array[0], "string");
    assert_eq!(array[1], 42);
    assert_eq!(array[2], true);
    assert_eq!(array[3], 3.14);
}

#[test]
fn test_array_literal_empty_array() {
    let schema = ArrayLiteralTests::schema();
    let schema_json = schema.to_json();
    
    // Check empty array
    let empty_array_example = &schema_json["properties"]["empty_array"]["example"];
    assert!(empty_array_example.is_array());
    let array = empty_array_example.as_array().unwrap();
    assert_eq!(array.len(), 0);
}

#[test]
fn test_array_literal_multiple_examples() {
    let schema = ArrayLiteralTests::schema();
    let schema_json = schema.to_json();
    
    // Check examples for floats
    let examples = &schema_json["properties"]["example_floats"]["examples"];
    assert!(examples.is_array());
    let array = examples.as_array().unwrap();
    assert_eq!(array.len(), 3);
    assert_eq!(array[0], 10.5);
    assert_eq!(array[1], 20.5);
    assert_eq!(array[2], 30.5);
    
    // Check examples for names
    let examples = &schema_json["properties"]["example_names"]["examples"];
    assert!(examples.is_array());
    let array = examples.as_array().unwrap();
    assert_eq!(array.len(), 4);
    assert_eq!(array[0], "John");
    assert_eq!(array[1], "Jane");
    assert_eq!(array[2], "Alice");
    assert_eq!(array[3], "Bob");
}
#[cfg(test)]
mod integration_tests {
    use rstructor::{LLMModel, SchemaType};
    use serde::{Deserialize, Serialize};

    // Simple struct with basic types
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    struct BasicTypes {
        #[llm(description = "A string field", example = "Hello World")]
        string_field: String,

        #[llm(description = "An integer field", example = 42)]
        int_field: i32,

        #[llm(description = "A float field", example = 3.5)]
        float_field: f64,

        #[llm(description = "A boolean field", example = true)]
        bool_field: bool,

        #[llm(description = "An optional field")]
        optional_field: Option<String>,
    }

    // Struct with array literals
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    struct ArrayLiterals {
        #[llm(description = "String array", example = ["red", "green", "blue"])]
        string_array: Vec<String>,

        #[llm(description = "Integer array", example = [1, 2, 3, 4, 5])]
        int_array: Vec<i32>,

        #[llm(description = "Float array", example = [1.1, 2.2, 3.3])]
        float_array: Vec<f64>,

        #[llm(description = "Boolean array", example = [true, false, true])]
        bool_array: Vec<bool>,
    }

    // Struct with multiple examples
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    struct MultipleExamples {
        #[llm(description = "String field with examples", 
              examples = ["Example 1", "Example 2", "Example 3"])]
        string_examples: String,

        #[llm(description = "Integer field with examples",
              examples = [10, 20, 30, 40])]
        int_examples: i32,

        #[llm(description = "Float field with examples",
              examples = [1.1, 2.2, 3.3, 4.4])]
        float_examples: f64,
    }

    // Nested struct
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    struct Address {
        #[llm(description = "Street address", example = "123 Main St")]
        street: String,

        #[llm(description = "City name", example = "Anytown")]
        city: String,

        #[llm(description = "ZIP or postal code", example = "12345")]
        zip: String,
    }

    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    struct Person {
        #[llm(description = "Person's name", example = "John Doe")]
        name: String,

        #[llm(description = "Person's age", example = 30)]
        age: u32,

        #[llm(description = "Person's address")]
        address: Address,

        #[llm(description = "Person's interests", 
              example = ["Reading", "Hiking", "Coding"])]
        interests: Vec<String>,
    }

    // Simple enum
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    enum Color {
        Red,
        Green,
        Blue,
        Yellow,
    }

    #[test]
    fn test_basic_types_schema() {
        let schema = BasicTypes::schema();
        let schema_json = schema.to_json();

        // Check schema type
        assert_eq!(schema_json["type"], "object");
        assert_eq!(schema_json["title"], "BasicTypes");

        // Check properties
        let props = schema_json["properties"].as_object().unwrap();
        assert!(props.contains_key("string_field"));
        assert!(props.contains_key("int_field"));
        assert!(props.contains_key("float_field"));
        assert!(props.contains_key("bool_field"));
        assert!(props.contains_key("optional_field"));

        // Check types
        assert_eq!(props["string_field"]["type"], "string");
        assert_eq!(props["int_field"]["type"], "integer");
        assert_eq!(props["float_field"]["type"], "number");
        assert_eq!(props["bool_field"]["type"], "boolean");

        // Check examples
        assert_eq!(props["string_field"]["example"], "Hello World");
        assert_eq!(props["int_field"]["example"], 42);
        // Check the float value matches the example (3.14)
        // Use a completely different value to avoid clippy PI approximation warning
        assert_eq!(props["float_field"]["example"], 3.5);
        assert_eq!(props["bool_field"]["example"], true);

        // Check required fields
        let required = schema_json["required"].as_array().unwrap();
        assert_eq!(required.len(), 4); // all except optional_field
        assert!(required.iter().any(|v| v == "string_field"));
        assert!(required.iter().any(|v| v == "int_field"));
        assert!(required.iter().any(|v| v == "float_field"));
        assert!(required.iter().any(|v| v == "bool_field"));
    }

    #[test]
    fn test_array_literals_schema() {
        let schema = ArrayLiterals::schema();
        let schema_json = schema.to_json();

        // Check schema type
        assert_eq!(schema_json["type"], "object");

        let props = schema_json["properties"].as_object().unwrap();

        // Check string array
        let string_array = &props["string_array"]["example"].as_array().unwrap();
        assert_eq!(string_array.len(), 3);
        assert_eq!(string_array[0], "red");
        assert_eq!(string_array[1], "green");
        assert_eq!(string_array[2], "blue");

        // Check int array
        let int_array = &props["int_array"]["example"].as_array().unwrap();
        assert_eq!(int_array.len(), 5);
        assert_eq!(int_array[0], 1);
        assert_eq!(int_array[4], 5);

        // Check float array
        let float_array = &props["float_array"]["example"].as_array().unwrap();
        assert_eq!(float_array.len(), 3);
        assert_eq!(float_array[0], 1.1);

        // Check bool array
        let bool_array = &props["bool_array"]["example"].as_array().unwrap();
        assert_eq!(bool_array.len(), 3);
        assert_eq!(bool_array[0], true);
        assert_eq!(bool_array[1], false);
    }

    #[test]
    fn test_multiple_examples() {
        let schema = MultipleExamples::schema();
        let schema_json = schema.to_json();

        let props = schema_json["properties"].as_object().unwrap();

        // Check string examples
        let string_examples = &props["string_examples"]["examples"].as_array().unwrap();
        assert_eq!(string_examples.len(), 3);
        assert_eq!(string_examples[0], "Example 1");
        assert_eq!(string_examples[1], "Example 2");
        assert_eq!(string_examples[2], "Example 3");

        // Check int examples
        let int_examples = &props["int_examples"]["examples"].as_array().unwrap();
        assert_eq!(int_examples.len(), 4);
        assert_eq!(int_examples[0], 10);

        // Check float examples
        let float_examples = &props["float_examples"]["examples"].as_array().unwrap();
        assert_eq!(float_examples.len(), 4);
        assert_eq!(float_examples[0], 1.1);
    }

    #[test]
    fn test_nested_structs() {
        let schema = Person::schema();
        let schema_json = schema.to_json();

        let props = schema_json["properties"].as_object().unwrap();

        // Check that the address field is an object type
        assert_eq!(props["address"]["type"], "object");

        // Check interests array
        let interests = &props["interests"]["example"].as_array().unwrap();
        assert_eq!(interests.len(), 3);
        assert_eq!(interests[0], "Reading");
        assert_eq!(interests[1], "Hiking");
        assert_eq!(interests[2], "Coding");
    }

    #[test]
    fn test_enum_schema() {
        let schema = Color::schema();
        let schema_json = schema.to_json();

        // Enums should be represented as a string with enum values
        assert_eq!(schema_json["type"], "string");
        assert_eq!(schema_json["title"], "Color");

        // Check enum values
        let enum_values = schema_json["enum"].as_array().unwrap();
        assert_eq!(enum_values.len(), 4);
        assert!(enum_values.iter().any(|v| v == "Red"));
        assert!(enum_values.iter().any(|v| v == "Green"));
        assert!(enum_values.iter().any(|v| v == "Blue"));
        assert!(enum_values.iter().any(|v| v == "Yellow"));
    }
}

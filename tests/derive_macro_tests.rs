#[cfg(test)]
mod derive_macro_tests {
    use rstructor::{Instructor, SchemaType};
    use serde::{Deserialize, Serialize};

    // Test string formatting from examples
    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct FormatExamples {
        // String with quotes and special characters
        #[llm(
            description = "String with quotes",
            example = "This is a \"quoted\" string with 'apostrophes' and backslash \\ characters"
        )]
        quoted_string: String,

        // String with multiline
        #[llm(description = "Multiline string", example = "Line 1\nLine 2\nLine 3")]
        multiline_string: String,

        // String with JSON-like content
        #[llm(
            description = "JSON-like string",
            example = "{\"key\": \"value\", \"array\": [1, 2, 3]}"
        )]
        json_string: String,
    }

    // Test attribute combinations
    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct AttributeCombinations {
        // Field with only description
        #[llm(description = "Field with only description")]
        desc_only: String,

        // Field with description and example
        #[llm(
            description = "Field with description and example",
            example = "Example value"
        )]
        desc_and_example: String,

        // Field with description and examples
        #[llm(description = "Field with description and examples", 
              examples = ["Example 1", "Example 2"])]
        desc_and_examples: String,

        // Optional field with example
        #[llm(
            description = "Optional field with example",
            example = "Optional example"
        )]
        optional_with_example: Option<String>,
    }

    // Test backwards compatibility with string-based array syntax
    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct BackwardsCompatibility {
        // Using the old string-based array syntax with single quotes
        #[llm(
            description = "Old array syntax with single quotes",
            example = "['item1', 'item2', 'item3']"
        )]
        old_single_quotes: Vec<String>,

        // Using the old string-based array syntax with double quotes
        #[llm(
            description = "Old array syntax with double quotes",
            example = "[\"item1\", \"item2\", \"item3\"]"
        )]
        old_double_quotes: Vec<String>,
    }

    #[test]
    fn test_string_formatting() {
        let schema = FormatExamples::schema();
        let schema_json = schema.to_json();

        let props = schema_json["properties"].as_object().unwrap();

        // Check quoted string
        let quoted = props["quoted_string"]["example"].as_str().unwrap();
        assert_eq!(
            quoted,
            "This is a \"quoted\" string with 'apostrophes' and backslash \\ characters"
        );

        // Check multiline string
        let multiline = props["multiline_string"]["example"].as_str().unwrap();
        assert_eq!(multiline, "Line 1\nLine 2\nLine 3");

        // Check JSON-like string
        let json_str = props["json_string"]["example"].as_str().unwrap();
        assert_eq!(json_str, "{\"key\": \"value\", \"array\": [1, 2, 3]}");
    }

    #[test]
    fn test_attribute_combinations() {
        let schema = AttributeCombinations::schema();
        let schema_json = schema.to_json();

        let props = schema_json["properties"].as_object().unwrap();

        // Field with only description
        assert!(
            props["desc_only"]
                .as_object()
                .unwrap()
                .contains_key("description")
        );
        assert!(
            !props["desc_only"]
                .as_object()
                .unwrap()
                .contains_key("example")
        );
        assert!(
            !props["desc_only"]
                .as_object()
                .unwrap()
                .contains_key("examples")
        );

        // Field with description and example
        assert!(
            props["desc_and_example"]
                .as_object()
                .unwrap()
                .contains_key("description")
        );
        assert!(
            props["desc_and_example"]
                .as_object()
                .unwrap()
                .contains_key("example")
        );
        assert!(
            !props["desc_and_example"]
                .as_object()
                .unwrap()
                .contains_key("examples")
        );

        // Field with description and examples
        assert!(
            props["desc_and_examples"]
                .as_object()
                .unwrap()
                .contains_key("description")
        );
        assert!(
            !props["desc_and_examples"]
                .as_object()
                .unwrap()
                .contains_key("example")
        );
        assert!(
            props["desc_and_examples"]
                .as_object()
                .unwrap()
                .contains_key("examples")
        );

        // Optional field with example
        assert!(
            props["optional_with_example"]
                .as_object()
                .unwrap()
                .contains_key("description")
        );
        assert!(
            props["optional_with_example"]
                .as_object()
                .unwrap()
                .contains_key("example")
        );

        // Check required fields (optional field should not be in required list)
        let required = schema_json["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "desc_only"));
        assert!(required.iter().any(|v| v == "desc_and_example"));
        assert!(required.iter().any(|v| v == "desc_and_examples"));
        assert!(!required.iter().any(|v| v == "optional_with_example"));
    }

    #[test]
    fn test_backwards_compatibility() {
        let schema = BackwardsCompatibility::schema();
        let schema_json = schema.to_json();

        let props = schema_json["properties"].as_object().unwrap();

        // Check array with single quotes
        let single_quotes = &props["old_single_quotes"]["example"].as_array().unwrap();
        assert_eq!(single_quotes.len(), 3);
        assert_eq!(single_quotes[0], "item1");
        assert_eq!(single_quotes[1], "item2");
        assert_eq!(single_quotes[2], "item3");

        // Check array with double quotes
        let double_quotes = &props["old_double_quotes"]["example"].as_array().unwrap();
        assert_eq!(double_quotes.len(), 3);
        assert_eq!(double_quotes[0], "item1");
        assert_eq!(double_quotes[1], "item2");
        assert_eq!(double_quotes[2], "item3");
    }
}

#[cfg(test)]
mod container_attributes_tests {
    use rstructor::{LLMModel, SchemaType};
    use serde::{Deserialize, Serialize};

    // Test struct with description
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    #[llm(description = "This is a struct description")]
    struct StructWithDescription {
        #[llm(description = "Field description")]
        field1: String,

        #[llm(description = "Another field description")]
        field2: i32,
    }

    // Test struct with title
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    #[llm(title = "CustomTitle")]
    struct StructWithTitle {
        field: String,
    }

    // Test struct with examples
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    #[llm(examples = [
        ::serde_json::json!({"name": "John", "age": 30}),
        ::serde_json::json!({"name": "Jane", "age": 25})
    ])]
    struct StructWithExamples {
        name: String,
        age: u32,
    }

    // Test struct with multiple container attributes
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    #[llm(
        description = "A person with their details",
        title = "PersonDetails",
        examples = [
            ::serde_json::json!({"name": "John", "age": 30, "is_active": true}),
            ::serde_json::json!({"name": "Jane", "age": 25, "is_active": false})
        ]
    )]
    struct StructWithMultipleAttrs {
        name: String,
        age: u32,
        is_active: bool,
    }

    // Test struct with serde rename_all
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    #[serde(rename_all = "camelCase")]
    struct StructWithRenameAll {
        first_name: String,
        last_name: String,
        user_id: u32,
    }

    // Test enum with description
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    #[llm(description = "This is an enum description")]
    enum EnumWithDescription {
        Option1,
        Option2,
        Option3,
    }

    // Test enum with title
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    #[llm(title = "CustomEnumTitle")]
    enum EnumWithTitle {
        Value1,
        Value2,
    }

    // Test enum with examples
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    #[llm(examples = ["Value1", "Value2"])]
    enum EnumWithExamples {
        Value1,
        Value2,
        Value3,
    }

    // Test enum with multiple attributes
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    #[llm(
        description = "A status enumeration",
        title = "StatusEnum",
        examples = ["Active", "Pending"]
    )]
    enum EnumWithMultipleAttrs {
        Active,
        Pending,
        Inactive,
    }

    // Struct without attributes
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    struct StructWithoutAttributes {
        field: String,
    }

    // Enum without attributes
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    enum EnumWithoutAttributes {
        Value1,
        Value2,
    }

    // Tests for struct with description
    #[test]
    fn test_struct_with_description() {
        let schema = StructWithDescription::schema();
        let schema_json = schema.to_json();

        // Check that description was added to the schema
        assert_eq!(schema_json["description"], "This is a struct description");

        // Check that the field descriptions are still there
        assert_eq!(
            schema_json["properties"]["field1"]["description"],
            "Field description"
        );
        assert_eq!(
            schema_json["properties"]["field2"]["description"],
            "Another field description"
        );
    }

    // Tests for struct with title
    #[test]
    fn test_struct_with_title() {
        let schema = StructWithTitle::schema();
        let schema_json = schema.to_json();

        // Check that title was added to the schema
        assert_eq!(schema_json["title"], "CustomTitle");
    }

    // Tests for struct with examples
    #[test]
    fn test_struct_with_examples() {
        let schema = StructWithExamples::schema();
        let schema_json = schema.to_json();

        // Check that examples were added to the schema
        let examples = schema_json["examples"].as_array().unwrap();
        assert_eq!(examples.len(), 2);

        // Check content of examples
        let example1 = &examples[0];
        assert_eq!(example1["name"], "John");
        assert_eq!(example1["age"], 30);

        let example2 = &examples[1];
        assert_eq!(example2["name"], "Jane");
        assert_eq!(example2["age"], 25);
    }

    // Tests for struct with multiple attributes
    #[test]
    fn test_struct_with_multiple_attrs() {
        let schema = StructWithMultipleAttrs::schema();
        let schema_json = schema.to_json();

        // Check all attributes
        assert_eq!(schema_json["description"], "A person with their details");
        assert_eq!(schema_json["title"], "PersonDetails");

        let examples = schema_json["examples"].as_array().unwrap();
        assert_eq!(examples.len(), 2);

        let example1 = &examples[0];
        assert_eq!(example1["name"], "John");
        assert_eq!(example1["age"], 30);
        assert_eq!(example1["is_active"], true);
    }

    // Tests for struct with serde rename_all
    #[test]
    fn test_struct_with_rename_all() {
        let schema = StructWithRenameAll::schema();
        let schema_json = schema.to_json();

        // Check that property names are camelCase
        let properties = schema_json["properties"].as_object().unwrap();
        assert!(properties.contains_key("firstName"));
        assert!(properties.contains_key("lastName"));
        assert!(properties.contains_key("userId"));

        // Check that required fields use the camelCase names
        let required = schema_json["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v.as_str() == Some("firstName")));
        assert!(required.iter().any(|v| v.as_str() == Some("lastName")));
        assert!(required.iter().any(|v| v.as_str() == Some("userId")));
    }

    // Tests for enum with description
    #[test]
    fn test_enum_with_description() {
        let schema = EnumWithDescription::schema();
        let schema_json = schema.to_json();

        // Check that description was added to the schema
        assert_eq!(schema_json["description"], "This is an enum description");

        // Check that the enum values are still there
        let enum_values = schema_json["enum"].as_array().unwrap();
        assert_eq!(enum_values.len(), 3);
    }

    // Tests for enum with title
    #[test]
    fn test_enum_with_title() {
        let schema = EnumWithTitle::schema();
        let schema_json = schema.to_json();

        // Check that title was added to the schema
        assert_eq!(schema_json["title"], "CustomEnumTitle");
    }

    // Tests for enum with examples
    #[test]
    fn test_enum_with_examples() {
        let schema = EnumWithExamples::schema();
        let schema_json = schema.to_json();

        // Check that examples were added to the schema
        let examples = schema_json["examples"].as_array().unwrap();
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0], "Value1");
        assert_eq!(examples[1], "Value2");
    }

    // Tests for enum with multiple attributes
    #[test]
    fn test_enum_with_multiple_attrs() {
        let schema = EnumWithMultipleAttrs::schema();
        let schema_json = schema.to_json();

        // Check all attributes
        assert_eq!(schema_json["description"], "A status enumeration");
        assert_eq!(schema_json["title"], "StatusEnum");

        let examples = schema_json["examples"].as_array().unwrap();
        assert_eq!(examples.len(), 2);
        assert_eq!(examples[0], "Active");
        assert_eq!(examples[1], "Pending");
    }

    // Tests for struct without attributes
    #[test]
    fn test_struct_without_attributes() {
        let schema = StructWithoutAttributes::schema();
        let schema_json = schema.to_json();

        // Check that there's no description or examples in the schema
        assert!(!schema_json.as_object().unwrap().contains_key("description"));
        assert!(!schema_json.as_object().unwrap().contains_key("examples"));

        // Title should default to the struct name
        assert_eq!(schema_json["title"], "StructWithoutAttributes");
    }

    // Tests for enum without attributes
    #[test]
    fn test_enum_without_attributes() {
        let schema = EnumWithoutAttributes::schema();
        let schema_json = schema.to_json();

        // Check that there's no description or examples in the schema
        assert!(!schema_json.as_object().unwrap().contains_key("description"));
        assert!(!schema_json.as_object().unwrap().contains_key("examples"));

        // Title should default to the enum name
        assert_eq!(schema_json["title"], "EnumWithoutAttributes");
    }
}

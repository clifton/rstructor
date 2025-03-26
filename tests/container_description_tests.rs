#[cfg(test)]
mod container_description_tests {
    use rstructor::{LLMModel, SchemaType};
    use serde::{Deserialize, Serialize};

    /// A test struct with a container-level description
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    #[llm(description = "This is a struct description")]
    struct StructWithDescription {
        #[llm(description = "Field description")]
        field1: String,

        #[llm(description = "Another field description")]
        field2: i32,
    }

    /// A test enum with a container-level description
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    #[llm(description = "This is an enum description")]
    enum EnumWithDescription {
        Option1,
        Option2,
        Option3,
    }

    // Struct without a description
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    struct StructWithoutDescription {
        #[llm(description = "Field description")]
        field: String,
    }

    // Enum without a description
    #[derive(LLMModel, Serialize, Deserialize, Debug)]
    enum EnumWithoutDescription {
        Value1,
        Value2,
    }

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

    #[test]
    fn test_enum_with_description() {
        let schema = EnumWithDescription::schema();
        let schema_json = schema.to_json();

        // Check that description was added to the schema
        assert_eq!(schema_json["description"], "This is an enum description");

        // Check that the enum values are still there
        let enum_values = schema_json["enum"].as_array().unwrap();
        assert_eq!(enum_values.len(), 3);
        assert!(enum_values.iter().any(|v| v == "Option1"));
        assert!(enum_values.iter().any(|v| v == "Option2"));
        assert!(enum_values.iter().any(|v| v == "Option3"));
    }

    #[test]
    fn test_struct_without_description() {
        let schema = StructWithoutDescription::schema();
        let schema_json = schema.to_json();

        // Check that there's no description in the schema
        assert!(!schema_json.as_object().unwrap().contains_key("description"));
    }

    #[test]
    fn test_enum_without_description() {
        let schema = EnumWithoutDescription::schema();
        let schema_json = schema.to_json();

        // Check that there's no description in the schema
        assert!(!schema_json.as_object().unwrap().contains_key("description"));
    }
}

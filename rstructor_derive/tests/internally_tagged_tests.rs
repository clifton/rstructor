use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};

// ── Helper structs used as newtype inner types ──────────────────────────

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
struct PersonInfo {
    #[llm(description = "Person's full name")]
    name: String,
    #[llm(description = "Person's age in years")]
    age: u32,
}

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
struct CompanyInfo {
    #[llm(description = "Company name")]
    company_name: String,
    #[llm(description = "Number of employees")]
    employee_count: u32,
}

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
struct OptionalFieldsData {
    #[llm(description = "A required field")]
    required_field: String,
    #[llm(description = "An optional field")]
    optional_field: Option<String>,
}

// ── Test 1: Basic internally tagged enum with newtype variants ──────────

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "kind")]
enum Contact {
    #[llm(description = "A person contact")]
    Person(PersonInfo),
    #[llm(description = "A company contact")]
    Company(CompanyInfo),
}

#[test]
fn test_newtype_variant_fields_appear_in_schema() {
    let schema = Contact::schema().to_json();
    let any_of = schema["anyOf"].as_array().expect("should have anyOf");

    // Find the Person variant
    let person_variant = any_of
        .iter()
        .find(|v| v["properties"]["kind"]["enum"][0] == "Person")
        .expect("should have Person variant");

    // The inner struct fields must appear alongside the tag
    assert!(
        person_variant["properties"].get("name").is_some(),
        "Person variant should have 'name' property from PersonInfo"
    );
    assert!(
        person_variant["properties"].get("age").is_some(),
        "Person variant should have 'age' property from PersonInfo"
    );

    // Find the Company variant
    let company_variant = any_of
        .iter()
        .find(|v| v["properties"]["kind"]["enum"][0] == "Company")
        .expect("should have Company variant");

    assert!(
        company_variant["properties"].get("company_name").is_some(),
        "Company variant should have 'company_name' property from CompanyInfo"
    );
    assert!(
        company_variant["properties"]
            .get("employee_count")
            .is_some(),
        "Company variant should have 'employee_count' property from CompanyInfo"
    );
}

// ── Test 2: Tag property has correct enum constraint ────────────────────

#[test]
fn test_tag_property_enum_constraint() {
    let schema = Contact::schema().to_json();
    let any_of = schema["anyOf"].as_array().unwrap();

    for variant in any_of {
        let tag_prop = &variant["properties"]["kind"];
        assert_eq!(
            tag_prop["type"], "string",
            "tag property should be string type"
        );
        let enum_vals = tag_prop["enum"]
            .as_array()
            .expect("tag should have enum constraint");
        assert_eq!(
            enum_vals.len(),
            1,
            "each variant tag should have exactly one allowed value"
        );
    }

    let tag_values: Vec<&str> = any_of
        .iter()
        .map(|v| v["properties"]["kind"]["enum"][0].as_str().unwrap())
        .collect();
    assert!(tag_values.contains(&"Person"));
    assert!(tag_values.contains(&"Company"));
}

// ── Test 3: Mixed enum with unit, newtype, and struct variants ──────────

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "type")]
enum MixedEvent {
    #[llm(description = "A ping with no data")]
    Ping,

    #[llm(description = "A message carrying a person")]
    PersonMessage(PersonInfo),

    #[llm(description = "An inline error")]
    Error {
        #[llm(description = "Error code")]
        code: u32,
        #[llm(description = "Error message")]
        message: String,
    },
}

#[test]
fn test_mixed_variant_types() {
    let schema = MixedEvent::schema().to_json();
    let any_of = schema["anyOf"].as_array().expect("should have anyOf");
    assert_eq!(any_of.len(), 3, "should have 3 variants");

    // Unit variant: only tag property
    let ping = any_of
        .iter()
        .find(|v| v["properties"]["type"]["enum"][0] == "Ping")
        .expect("should have Ping variant");
    // Ping should have the tag and nothing else
    let ping_props = ping["properties"].as_object().unwrap();
    assert_eq!(
        ping_props.len(),
        1,
        "Ping should only have the tag property"
    );

    // Newtype variant: tag + inner struct fields
    let person_msg = any_of
        .iter()
        .find(|v| v["properties"]["type"]["enum"][0] == "PersonMessage")
        .expect("should have PersonMessage variant");
    assert!(person_msg["properties"].get("name").is_some());
    assert!(person_msg["properties"].get("age").is_some());

    // Struct variant: tag + named fields
    let error = any_of
        .iter()
        .find(|v| v["properties"]["type"]["enum"][0] == "Error")
        .expect("should have Error variant");
    assert!(error["properties"].get("code").is_some());
    assert!(error["properties"].get("message").is_some());
}

// ── Test 4: Serde roundtrip matches schema shape ────────────────────────

#[test]
fn test_serde_roundtrip_matches_schema() {
    // Serialize a Contact::Person and verify the shape matches what the schema describes
    let contact = Contact::Person(PersonInfo {
        name: "Alice".to_string(),
        age: 30,
    });
    let json = serde_json::to_value(&contact).unwrap();

    // serde with internal tagging produces {"kind": "Person", "name": "Alice", "age": 30}
    assert_eq!(json["kind"], "Person");
    assert_eq!(json["name"], "Alice");
    assert_eq!(json["age"], 30);

    // The schema should describe exactly these keys
    let schema = Contact::schema().to_json();
    let person_variant = schema["anyOf"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["properties"]["kind"]["enum"][0] == "Person")
        .unwrap();

    let schema_props: Vec<&str> = person_variant["properties"]
        .as_object()
        .unwrap()
        .keys()
        .map(|k| k.as_str())
        .collect();

    for key in json.as_object().unwrap().keys() {
        assert!(
            schema_props.contains(&key.as_str()),
            "serialized key '{}' should appear in schema properties",
            key
        );
    }

    // Roundtrip deserialization
    let deserialized: Contact = serde_json::from_value(json).unwrap();
    assert_eq!(
        deserialized,
        Contact::Person(PersonInfo {
            name: "Alice".to_string(),
            age: 30,
        })
    );
}

// ── Test 5: Newtype variant with optional fields ────────────────────────

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "tag")]
enum OptionalFieldsEnum {
    #[llm(description = "Variant with optional fields")]
    Data(OptionalFieldsData),
}

#[test]
fn test_optional_fields_not_required() {
    let schema = OptionalFieldsEnum::schema().to_json();
    let variant = &schema["anyOf"].as_array().unwrap()[0];

    let required: Vec<&str> = variant["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();

    assert!(required.contains(&"tag"), "tag should be required");
    assert!(
        required.contains(&"required_field"),
        "required_field should be required"
    );
    assert!(
        !required.contains(&"optional_field"),
        "optional_field should NOT be required"
    );

    // But the optional field should still appear in properties
    assert!(
        variant["properties"].get("optional_field").is_some(),
        "optional_field should still appear in properties"
    );
}

// ── Test 6: rename_all support with tagged enums ────────────────────────

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RenamedVariants {
    #[llm(description = "First variant")]
    MyFirstVariant,
    #[llm(description = "Second variant")]
    MySecondVariant,
}

#[test]
fn test_rename_all_with_tagged_enum() {
    let schema = RenamedVariants::schema().to_json();
    let any_of = schema["anyOf"].as_array().unwrap();

    let tag_values: Vec<&str> = any_of
        .iter()
        .map(|v| v["properties"]["type"]["enum"][0].as_str().unwrap())
        .collect();

    assert!(
        tag_values.contains(&"my_first_variant"),
        "should have snake_case variant name 'my_first_variant', got: {:?}",
        tag_values
    );
    assert!(
        tag_values.contains(&"my_second_variant"),
        "should have snake_case variant name 'my_second_variant', got: {:?}",
        tag_values
    );
}

// ── Test 7: All-unit internally tagged enum produces object schemas ─────

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "status")]
enum Status {
    #[llm(description = "Currently active")]
    Active,
    #[llm(description = "Currently inactive")]
    Inactive,
    #[llm(description = "Account is pending")]
    Pending,
}

#[test]
fn test_all_unit_tagged_enum_produces_object_schemas() {
    let schema = Status::schema().to_json();

    // Should NOT be a simple {"type": "string", "enum": [...]} schema
    assert!(
        schema.get("enum").is_none(),
        "tagged all-unit enum should NOT have top-level 'enum' key"
    );

    // Should have anyOf with object schemas
    let any_of = schema["anyOf"]
        .as_array()
        .expect("tagged all-unit enum should have 'anyOf' with object variant schemas");
    assert_eq!(any_of.len(), 3);

    for variant in any_of {
        assert_eq!(
            variant["type"], "object",
            "each variant should be an object"
        );
        assert!(
            variant["properties"].get("status").is_some(),
            "each variant should have the 'status' tag property"
        );
    }

    // Verify serde serialization matches: Status::Active -> {"status": "Active"}
    let json = serde_json::to_value(Status::Active).unwrap();
    assert_eq!(json, serde_json::json!({"status": "Active"}));
}

// ── Test 8: Untagged all-unit enum still produces simple string enum ────

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
enum SimpleColor {
    #[llm(description = "Red color")]
    Red,
    #[llm(description = "Green color")]
    Green,
    #[llm(description = "Blue color")]
    Blue,
}

#[test]
fn test_untagged_all_unit_enum_is_string_enum() {
    let schema = SimpleColor::schema().to_json();

    // Should be a simple {"type": "string", "enum": [...]} schema
    assert_eq!(
        schema["type"], "string",
        "untagged all-unit enum should be string type"
    );
    let enum_vals = schema["enum"].as_array().expect("should have enum values");
    let values: Vec<&str> = enum_vals.iter().map(|v| v.as_str().unwrap()).collect();
    assert!(values.contains(&"Red"));
    assert!(values.contains(&"Green"));
    assert!(values.contains(&"Blue"));
}

// ── Test 9: Box<T> newtype variant works correctly ──────────────────────

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "kind")]
enum BoxedContact {
    #[llm(description = "A boxed person contact")]
    Person(Box<PersonInfo>),
    #[llm(description = "A boxed company contact")]
    Company(Box<CompanyInfo>),
}

#[test]
fn test_boxed_newtype_variant_fields_appear() {
    let schema = BoxedContact::schema().to_json();
    let any_of = schema["anyOf"].as_array().expect("should have anyOf");

    let person_variant = any_of
        .iter()
        .find(|v| v["properties"]["kind"]["enum"][0] == "Person")
        .expect("should have Person variant");

    assert!(
        person_variant["properties"].get("name").is_some(),
        "Boxed Person variant should have 'name' property"
    );
    assert!(
        person_variant["properties"].get("age").is_some(),
        "Boxed Person variant should have 'age' property"
    );

    // Required should include tag + inner struct required fields
    let required: Vec<&str> = person_variant["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(required.contains(&"kind"));
    assert!(required.contains(&"name"));
    assert!(required.contains(&"age"));

    // Serde roundtrip
    let contact = BoxedContact::Person(Box::new(PersonInfo {
        name: "Bob".to_string(),
        age: 42,
    }));
    let json = serde_json::to_value(&contact).unwrap();
    assert_eq!(json["kind"], "Person");
    assert_eq!(json["name"], "Bob");
    assert_eq!(json["age"], 42);

    let deserialized: BoxedContact = serde_json::from_value(json).unwrap();
    assert_eq!(
        deserialized,
        BoxedContact::Person(Box::new(PersonInfo {
            name: "Bob".to_string(),
            age: 42,
        }))
    );
}

// -- Test 10: Recursive inner structs preserve fields through $defs --------

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
struct RecursiveNode {
    #[llm(description = "Node label")]
    label: String,
    #[llm(description = "Child nodes")]
    children: Vec<Box<RecursiveNode>>,
}

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "kind")]
enum RecursiveTree {
    #[llm(description = "A recursive tree root")]
    Root(RecursiveNode),
}

#[test]
fn test_recursive_newtype_variant_fields_appear() {
    let schema = RecursiveTree::schema().to_json();
    let any_of = schema["anyOf"].as_array().expect("should have anyOf");

    let root_variant = any_of
        .iter()
        .find(|v| v["properties"]["kind"]["enum"][0] == "Root")
        .expect("should have Root variant");

    assert!(
        root_variant["properties"].get("label").is_some(),
        "recursive newtype variant should include fields from the referenced inner schema"
    );
    assert!(
        root_variant["properties"].get("children").is_some(),
        "recursive newtype variant should include recursive child field"
    );
    assert!(
        root_variant["$defs"].get("RecursiveNode").is_some(),
        "recursive definitions should be preserved for nested $ref values"
    );

    let required: Vec<&str> = root_variant["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(required.contains(&"kind"));
    assert!(required.contains(&"label"));
    assert!(required.contains(&"children"));

    let tree = RecursiveTree::Root(RecursiveNode {
        label: "root".to_string(),
        children: Vec::new(),
    });
    let json = serde_json::to_value(&tree).unwrap();
    assert_eq!(json["kind"], "Root");
    assert_eq!(json["label"], "root");
    assert!(json["children"].is_array());
}

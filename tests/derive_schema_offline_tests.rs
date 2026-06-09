//! Offline schema-shape tests for the `Instructor` derive macro.
//!
//! Every test here derives types and asserts the *generated* JSON Schema via the
//! public API `T::schema().to_json()`. These cover the schema surface that was
//! previously only exercised by live integration tests:
//!
//! - Tagged enum schema shapes (internally-tagged, adjacently-tagged, untagged)
//! - Externally-tagged tuple/struct/unit + mixed variants
//! - Map field schemas (`additionalProperties` chain, `x-enum-keys`, "Keys: [..]" hint)
//! - Tuple field schemas (`prefixItems`/`minItems`/`maxItems`)
//! - `Box<T>` fields
//! - Self-referential `$defs`/`$ref`
//! - `rename_all` styles applied to *struct* fields
//! - `example`/`examples` string-coercion + empty-array edges
//! - `serde_json::Value` field -> any-JSON schema
//!
//! These are pure compile-time/offline assertions; no network access is required.

#![cfg(test)]

use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Internally tagged enums: #[serde(tag = "kind")]
// ============================================================================

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
struct Inner {
    #[llm(description = "An inner string field")]
    b: String,
}

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "kind")]
enum InternalEnum {
    Empty,
    Full {
        #[llm(description = "An integer payload")]
        a: i32,
    },
    Wrap(Inner),
}

#[test]
fn internally_tagged_enum_anyof_and_tag_enum() {
    let schema = InternalEnum::schema().to_json();
    assert_eq!(schema["title"], "InternalEnum");
    let any_of = schema["anyOf"].as_array().expect("anyOf must be array");
    assert_eq!(any_of.len(), 3, "three variants -> three anyOf members");

    // Every member carries the tag property `kind` with a single-value enum.
    let find = |name: &str| {
        any_of
            .iter()
            .find(|v| v["properties"]["kind"]["enum"][0] == name)
            .unwrap_or_else(|| panic!("missing variant {name}"))
    };

    // Unit variant: tag only, required == ["kind"].
    let empty = find("Empty");
    assert_eq!(empty["type"], "object");
    assert_eq!(empty["properties"]["kind"]["type"], "string");
    let empty_required: Vec<&str> = empty["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(empty_required, vec!["kind"]);
    assert!(
        empty["properties"].get("a").is_none(),
        "Empty has no extra fields"
    );

    // Struct variant: named field is flattened beside the tag.
    let full = find("Full");
    assert!(
        full["properties"].get("a").is_some(),
        "Full.a must be flattened beside the tag"
    );
    assert_eq!(full["properties"]["a"]["type"], "integer");
    let full_required: Vec<&str> = full["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(full_required.contains(&"kind"));
    assert!(full_required.contains(&"a"));

    // Newtype-of-struct variant: inner struct fields flattened beside the tag.
    let wrap = find("Wrap");
    assert!(
        wrap["properties"].get("b").is_some(),
        "Wrap must flatten Inner.b beside the tag"
    );
    assert_eq!(wrap["properties"]["b"]["type"], "string");
    let wrap_required: Vec<&str> = wrap["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(wrap_required.contains(&"kind"));
    assert!(wrap_required.contains(&"b"));
}

#[test]
fn internally_tagged_enum_schema_name() {
    assert_eq!(
        InternalEnum::schema_name(),
        Some("InternalEnum".to_string())
    );
}

// ============================================================================
// Adjacently tagged enums: #[serde(tag = "t", content = "c")]
// ============================================================================

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "t", content = "c")]
enum AdjacentEnum {
    Pending,
    One(String),
    Two(i32, i32),
    Obj {
        #[llm(description = "x value")]
        x: i32,
    },
}

#[test]
fn adjacently_tagged_required_is_tag_and_content() {
    let schema = AdjacentEnum::schema().to_json();
    let any_of = schema["anyOf"].as_array().unwrap();

    let find = |name: &str| {
        any_of
            .iter()
            .find(|v| v["properties"]["t"]["enum"][0] == name)
            .unwrap_or_else(|| panic!("missing variant {name}"))
    };

    // Single-field content -> content schema is a string, required == [t, c].
    let one = find("One");
    assert_eq!(one["properties"]["c"]["type"], "string");
    let one_required: Vec<&str> = one["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(one_required, vec!["t", "c"]);

    // Multi-field content -> content is a fixed-length array.
    let two = find("Two");
    assert_eq!(two["properties"]["c"]["type"], "array");
    assert_eq!(two["properties"]["c"]["minItems"], 2);
    assert_eq!(two["properties"]["c"]["maxItems"], 2);
    let two_required: Vec<&str> = two["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(two_required, vec!["t", "c"]);

    // Struct content -> content is a nested object holding the variant fields.
    let obj = find("Obj");
    assert_eq!(obj["properties"]["c"]["type"], "object");
    assert_eq!(obj["properties"]["c"]["properties"]["x"]["type"], "integer");
    let obj_required: Vec<&str> = obj["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(obj_required, vec!["t", "c"]);
}

#[test]
fn adjacently_tagged_unit_variant_has_tag_only() {
    let schema = AdjacentEnum::schema().to_json();
    let any_of = schema["anyOf"].as_array().unwrap();
    let pending = any_of
        .iter()
        .find(|v| v["properties"]["t"]["enum"][0] == "Pending")
        .expect("Pending variant");
    // No content key for a unit variant; required is just the tag.
    assert!(pending["properties"].get("c").is_none());
    let required: Vec<&str> = pending["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(required, vec!["t"]);
}

#[test]
fn adjacently_tagged_round_trips_via_serde() {
    // The generated schema must describe what serde actually accepts.
    let v: AdjacentEnum = serde_json::from_value(serde_json::json!({"t": "One", "c": "hi"}))
        .expect("adjacent single-field deserialize");
    assert_eq!(v, AdjacentEnum::One("hi".to_string()));

    let v: AdjacentEnum = serde_json::from_value(serde_json::json!({"t": "Two", "c": [3, 4]}))
        .expect("adjacent tuple deserialize");
    assert_eq!(v, AdjacentEnum::Two(3, 4));

    let v: AdjacentEnum = serde_json::from_value(serde_json::json!({"t": "Obj", "c": {"x": 9}}))
        .expect("adjacent struct");
    assert_eq!(v, AdjacentEnum::Obj { x: 9 });

    let v: AdjacentEnum =
        serde_json::from_value(serde_json::json!({"t": "Pending"})).expect("adjacent unit");
    assert_eq!(v, AdjacentEnum::Pending);
}

// ============================================================================
// Untagged enums: #[serde(untagged)]
// ============================================================================

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
#[serde(untagged)]
enum Untagged {
    NoneV,
    Single(String),
    Pair(i32, i32),
    Obj {
        #[llm(description = "x value")]
        x: i32,
    },
}

#[test]
fn untagged_enum_anyof_shapes() {
    let schema = Untagged::schema().to_json();
    assert_eq!(schema["title"], "Untagged");
    let any_of = schema["anyOf"].as_array().unwrap();
    assert_eq!(any_of.len(), 4);

    // Unit -> null (serde serializes a unit untagged variant as null).
    assert_eq!(any_of[0]["type"], "null");
    // Single newtype -> the bare inner type (string).
    assert_eq!(any_of[1]["type"], "string");
    // Tuple -> fixed-length array.
    assert_eq!(any_of[2]["type"], "array");
    assert_eq!(any_of[2]["minItems"], 2);
    assert_eq!(any_of[2]["maxItems"], 2);
    // Struct -> object with properties + required + additionalProperties:false.
    assert_eq!(any_of[3]["type"], "object");
    assert_eq!(any_of[3]["properties"]["x"]["type"], "integer");
    let required: Vec<&str> = any_of[3]["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(required, vec!["x"]);
    assert_eq!(any_of[3]["additionalProperties"], false);

    // Untagged enums have no top-level required key.
    assert!(schema.get("required").is_none());
}

// ============================================================================
// Externally tagged enums: tuple / struct / unit + mixed variants
// ============================================================================

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
enum Shape {
    Circle(f64),
    Rect(f64, f64),
}

#[test]
fn externally_tagged_tuple_variants() {
    let schema = Shape::schema().to_json();
    let any_of = schema["anyOf"].as_array().unwrap();

    let circle = any_of
        .iter()
        .find(|v| v["properties"].get("Circle").is_some())
        .expect("Circle member");
    // Single-field tuple variant -> {"Circle": <value>}.
    assert_eq!(circle["properties"]["Circle"]["type"], "number");
    let circle_required: Vec<&str> = circle["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(circle_required, vec!["Circle"]);
    assert_eq!(circle["additionalProperties"], false);

    let rect = any_of
        .iter()
        .find(|v| v["properties"].get("Rect").is_some())
        .expect("Rect member");
    // Multi-field tuple variant -> {"Rect": [<v0>, <v1>]} with fixed length.
    assert_eq!(rect["properties"]["Rect"]["type"], "array");
    assert_eq!(rect["properties"]["Rect"]["minItems"], 2);
    assert_eq!(rect["properties"]["Rect"]["maxItems"], 2);
    let rect_required: Vec<&str> = rect["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(rect_required, vec!["Rect"]);
}

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
enum Msg {
    Ping,
    Data {
        #[llm(description = "the payload")]
        payload: String,
    },
}

#[test]
fn externally_tagged_mixed_unit_and_struct_variants() {
    let schema = Msg::schema().to_json();
    let any_of = schema["anyOf"].as_array().unwrap();

    // Unit variant -> {type:string, enum:[Ping]}.
    let ping = any_of
        .iter()
        .find(|v| v["type"] == "string")
        .expect("Ping member as string");
    let ping_enum: Vec<&str> = ping["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(ping_enum, vec!["Ping"]);

    // Struct variant -> object keyed on the variant name.
    let data = any_of
        .iter()
        .find(|v| v["properties"].get("Data").is_some())
        .expect("Data member");
    assert_eq!(
        data["properties"]["Data"]["properties"]["payload"]["type"],
        "string"
    );
}

#[test]
fn externally_tagged_mixed_round_trips() {
    let v: Msg =
        serde_json::from_value(serde_json::json!("Ping")).expect("unit variant from string");
    assert_eq!(v, Msg::Ping);

    let v: Msg = serde_json::from_value(serde_json::json!({"Data": {"payload": "x"}}))
        .expect("struct variant");
    assert_eq!(
        v,
        Msg::Data {
            payload: "x".to_string()
        }
    );
}

// ============================================================================
// Map field schemas
// ============================================================================

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct MapHolder {
    // HashMap<String, Vec<String>> -> additionalProperties chain.
    #[llm(description = "groups of tags")]
    groups: HashMap<String, Vec<String>>,
}

#[test]
fn map_field_additional_properties_chain() {
    let schema = MapHolder::schema().to_json();
    let f = &schema["properties"]["groups"];
    assert_eq!(f["type"], "object");
    // The map value type (Vec<String>) describes additionalProperties.
    assert_eq!(f["additionalProperties"]["type"], "array");
    assert_eq!(f["additionalProperties"]["items"]["type"], "string");
}

#[derive(Instructor, Serialize, Deserialize, Debug, PartialEq, Eq, Hash, Clone)]
enum Level {
    A,
    B,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct EnumKeyMap {
    counts: HashMap<Level, u32>,
}

#[test]
fn map_field_enum_keys_x_enum_keys_and_keys_hint() {
    let schema = EnumKeyMap::schema().to_json();
    let f = &schema["properties"]["counts"];
    assert_eq!(f["type"], "object");
    // Enum-key extension field.
    let keys: Vec<&str> = f["x-enum-keys"]
        .as_array()
        .expect("x-enum-keys present")
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(keys, vec!["A", "B"]);
    // The auto-generated description embeds the key hint.
    let desc = f["description"].as_str().expect("description present");
    assert_eq!(desc, "Keys: [A, B]");
    // Value schema reaches the map value type.
    assert_eq!(f["additionalProperties"]["type"], "integer");
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct EnumKeyMapWithDesc {
    #[llm(description = "counts per level")]
    counts: HashMap<Level, u32>,
}

#[test]
fn map_field_user_description_merges_with_keys_hint() {
    let schema = EnumKeyMapWithDesc::schema().to_json();
    let desc = schema["properties"]["counts"]["description"]
        .as_str()
        .expect("description present");
    // User description is merged in front of the auto keys hint.
    assert_eq!(desc, "counts per level. Keys: [A, B]");
}

// ============================================================================
// Tuple field schemas
// ============================================================================

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct TupleHolder {
    #[llm(description = "a coordinate")]
    coord: (i32, i32),
    mixed: (u64, String, bool),
}

#[test]
fn tuple_field_prefix_items_and_bounds() {
    let schema = TupleHolder::schema().to_json();

    let coord = &schema["properties"]["coord"];
    assert_eq!(coord["type"], "array");
    assert_eq!(coord["minItems"], 2);
    assert_eq!(coord["maxItems"], 2);
    let coord_prefix = coord["prefixItems"].as_array().unwrap();
    assert_eq!(coord_prefix.len(), 2);
    assert_eq!(coord_prefix[0]["type"], "integer");
    assert_eq!(coord_prefix[1]["type"], "integer");

    let mixed = &schema["properties"]["mixed"];
    assert_eq!(mixed["type"], "array");
    assert_eq!(mixed["minItems"], 3);
    assert_eq!(mixed["maxItems"], 3);
    let mixed_prefix = mixed["prefixItems"].as_array().unwrap();
    assert_eq!(mixed_prefix[0]["type"], "integer");
    assert_eq!(mixed_prefix[1]["type"], "string");
    assert_eq!(mixed_prefix[2]["type"], "boolean");
}

// ============================================================================
// Box<T> field schemas (non-recursive)
// ============================================================================

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Address {
    #[llm(description = "street")]
    street: String,
    #[llm(description = "zip")]
    zip: String,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct BoxHolder {
    boxed: Box<Address>,
    count: Box<i32>,
    opt: Option<Box<Address>>,
}

#[test]
fn box_field_of_struct_inlines_inner_schema() {
    let schema = BoxHolder::schema().to_json();
    let boxed = &schema["properties"]["boxed"];
    // Box<Address> is invisible: the field carries Address's object schema.
    assert_eq!(boxed["type"], "object");
    assert_eq!(boxed["properties"]["street"]["type"], "string");
    assert_eq!(boxed["properties"]["zip"]["type"], "string");
}

#[test]
fn box_field_of_primitive_is_primitive() {
    let schema = BoxHolder::schema().to_json();
    assert_eq!(schema["properties"]["count"]["type"], "integer");
}

#[test]
fn optional_box_field_is_not_required() {
    let schema = BoxHolder::schema().to_json();
    let required: Vec<&str> = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(required.contains(&"boxed"));
    assert!(required.contains(&"count"));
    assert!(
        !required.contains(&"opt"),
        "Option<Box<_>> must not be required"
    );
}

// ============================================================================
// Self-referential $defs / $ref
// ============================================================================

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Node {
    #[llm(description = "node value")]
    value: i32,
    #[llm(description = "child nodes")]
    children: Vec<Node>,
}

#[test]
fn self_referential_struct_uses_defs_and_ref() {
    let schema = Node::schema().to_json();
    // Top-level points at $defs/Node.
    assert_eq!(schema["$ref"], "#/$defs/Node");
    assert!(schema["$defs"]["Node"].is_object(), "$defs.Node exists");
    // The recursive child array references the same definition.
    assert_eq!(
        schema["$defs"]["Node"]["properties"]["children"]["items"]["$ref"],
        "#/$defs/Node"
    );
    // The non-recursive field is still a normal primitive.
    assert_eq!(
        schema["$defs"]["Node"]["properties"]["value"]["type"],
        "integer"
    );
}

#[test]
fn self_referential_schema_name() {
    assert_eq!(Node::schema_name(), Some("Node".to_string()));
}

// ============================================================================
// rename_all STYLES applied to STRUCT fields
// ============================================================================

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[serde(rename_all = "UPPERCASE")]
struct UpperFields {
    user_id: i32,
    name: String,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct PascalFields {
    user_id: i32,
    name: String,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct KebabFields {
    user_id: i32,
    first_name: String,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[serde(rename_all = "SCREAMING-KEBAB-CASE")]
struct ScreamingKebabFields {
    user_id: i32,
    first_name: String,
}

fn prop_keys(schema: &serde_json::Value) -> Vec<String> {
    schema["properties"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

#[test]
fn rename_all_uppercase_on_struct_fields() {
    let schema = UpperFields::schema().to_json();
    let keys = prop_keys(&schema);
    assert!(keys.contains(&"USER_ID".to_string()), "got keys: {keys:?}");
    assert!(keys.contains(&"NAME".to_string()));
    // required uses the renamed keys.
    let required: Vec<&str> = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(required.contains(&"USER_ID"));
}

#[test]
fn rename_all_pascalcase_on_struct_fields() {
    let schema = PascalFields::schema().to_json();
    let keys = prop_keys(&schema);
    assert!(keys.contains(&"UserId".to_string()), "got keys: {keys:?}");
    assert!(keys.contains(&"Name".to_string()));
}

#[test]
fn rename_all_kebab_case_on_struct_fields() {
    let schema = KebabFields::schema().to_json();
    let keys = prop_keys(&schema);
    assert!(keys.contains(&"user-id".to_string()), "got keys: {keys:?}");
    assert!(keys.contains(&"first-name".to_string()));
}

#[test]
fn rename_all_screaming_kebab_case_on_struct_fields() {
    let schema = ScreamingKebabFields::schema().to_json();
    let keys = prop_keys(&schema);
    assert!(keys.contains(&"USER-ID".to_string()), "got keys: {keys:?}");
    assert!(keys.contains(&"FIRST-NAME".to_string()));
}

#[test]
fn rename_all_matches_serde_serialization() {
    // The schema field name must match the actual serde wire format.
    let v = KebabFields {
        user_id: 1,
        first_name: "x".to_string(),
    };
    let json = serde_json::to_value(&v).unwrap();
    let obj = json.as_object().unwrap();
    assert!(obj.contains_key("user-id"));
    assert!(obj.contains_key("first-name"));
}

// ============================================================================
// example / examples string-coercion + empty-array edges
// ============================================================================

// Native-literal examples (the supported, working form): an integer/float/bool
// literal on a matching field renders as the corresponding typed JSON value.
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct LiteralExampleHolder {
    #[llm(description = "an int example", example = 42)]
    int_field: i32,
    #[llm(description = "a float example", example = 3.5)]
    float_field: f64,
    #[llm(description = "a bool example", example = true)]
    bool_field: bool,
    #[llm(description = "a string example", example = "hello")]
    str_field: String,
}

#[test]
fn literal_examples_render_as_typed_values() {
    let schema = LiteralExampleHolder::schema().to_json();

    let int_example = &schema["properties"]["int_field"]["example"];
    assert!(int_example.is_number(), "got: {int_example:?}");
    assert_eq!(int_example.as_i64(), Some(42));

    let float_example = &schema["properties"]["float_field"]["example"];
    assert!(float_example.is_number(), "got: {float_example:?}");
    assert_eq!(float_example.as_f64(), Some(3.5));

    let bool_example = &schema["properties"]["bool_field"]["example"];
    assert_eq!(bool_example, &serde_json::Value::Bool(true));

    // String example on a String field is a plain JSON string.
    let str_example = &schema["properties"]["str_field"]["example"];
    assert_eq!(str_example, &serde_json::Value::String("hello".to_string()));
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct CoercionHolder {
    #[llm(description = "an int from a string", example = "42")]
    int_field: i32,
    #[llm(description = "a float from a string", example = "3.5")]
    float_field: f64,
    #[llm(description = "a bool from a string", example = "true")]
    bool_field: bool,
}

// CONTRACT: `#[llm(example = ...)]` takes a NATIVE typed literal matching the
// field — `example = 42` for an i32, `example = true` for a bool, `example =
// "text"` for a String (all covered by the test above). A *string-form* example
// on a numeric/bool field (`example = "42"`) is a type mismatch and is
// intentionally NOT coerced: no `example` key is emitted, and the field is
// otherwise unaffected. (Auto-coercing string-form literals to the field type
// could be a future enhancement; the native typed form is the supported path.)
#[test]
fn string_form_example_on_numeric_field_is_omitted() {
    let schema = CoercionHolder::schema().to_json();
    assert!(
        schema["properties"]["int_field"]["example"].is_null(),
        "string-form example on an i32 should emit no example key"
    );
    assert!(schema["properties"]["float_field"]["example"].is_null());
    assert!(schema["properties"]["bool_field"]["example"].is_null());
    // The fields themselves are still present and correctly typed.
    assert_eq!(schema["properties"]["int_field"]["type"], "integer");
    assert_eq!(schema["properties"]["bool_field"]["type"], "boolean");
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct EmptyArrayExampleHolder {
    #[llm(description = "tags with an empty example list", example = [])]
    tags: Vec<String>,
}

#[test]
fn empty_array_example_present_but_empty() {
    let schema = EmptyArrayExampleHolder::schema().to_json();
    let example = &schema["properties"]["tags"]["example"];
    assert!(example.is_array(), "empty example must be an empty array");
    assert_eq!(example.as_array().unwrap().len(), 0);
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct EmptyExamplesHolder {
    #[llm(description = "tags with no examples", examples = [])]
    tags: Vec<String>,
}

#[test]
fn empty_examples_array_omits_examples_key() {
    let schema = EmptyExamplesHolder::schema().to_json();
    // An empty `examples = []` produces no "examples" key at all.
    assert!(
        schema["properties"]["tags"].get("examples").is_none(),
        "empty examples list must not emit an examples key"
    );
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct MultiExamplesHolder {
    #[llm(description = "an int with multiple examples", examples = [1, 2, 3])]
    n: i32,
}

#[test]
fn examples_array_renders_all_values() {
    let schema = MultiExamplesHolder::schema().to_json();
    let examples = schema["properties"]["n"]["examples"]
        .as_array()
        .expect("examples present");
    let nums: Vec<i64> = examples.iter().map(|v| v.as_i64().unwrap()).collect();
    assert_eq!(nums, vec![1, 2, 3]);
}

// ============================================================================
// serde_json::Value field -> any-JSON (empty) schema
// ============================================================================

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct ValueHolder {
    #[llm(description = "arbitrary metadata")]
    meta: serde_json::Value,
    name: String,
}

#[test]
fn serde_json_value_field_is_any_json() {
    let schema = ValueHolder::schema().to_json();
    let meta = &schema["properties"]["meta"];
    // Any-JSON: no "type" constraint is emitted.
    assert!(
        meta.get("type").is_none(),
        "serde_json::Value field must not constrain type, got: {meta:?}"
    );
    // The neighbor field is unaffected.
    assert_eq!(schema["properties"]["name"]["type"], "string");
}

// ============================================================================
// Nested collections: Vec<Vec<T>>, Vec<Vec<Vec<T>>>, Vec<HashSet<T>>
// ============================================================================

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct NestedCollections {
    #[llm(description = "A matrix of integers")]
    matrix: Vec<Vec<i32>>,
    #[llm(description = "A cube of strings")]
    cube: Vec<Vec<Vec<String>>>,
    #[llm(description = "A list of unique-tag sets")]
    tag_sets: Vec<std::collections::HashSet<String>>,
    #[llm(description = "Optional matrix")]
    opt_matrix: Option<Vec<Vec<f64>>>,
}

#[test]
fn nested_vec_field_keeps_inner_items() {
    let schema = NestedCollections::schema().to_json();

    // Vec<Vec<i32>>: every nesting level carries its own items schema.
    let matrix = &schema["properties"]["matrix"];
    assert_eq!(matrix["type"], "array");
    assert_eq!(matrix["items"]["type"], "array");
    assert_eq!(
        matrix["items"]["items"]["type"], "integer",
        "Vec<Vec<i32>> must recurse: items.items.type == integer, got: {matrix:?}"
    );
}

#[test]
fn triply_nested_vec_field_keeps_all_items() {
    let schema = NestedCollections::schema().to_json();
    let cube = &schema["properties"]["cube"];
    assert_eq!(cube["type"], "array");
    assert_eq!(cube["items"]["type"], "array");
    assert_eq!(cube["items"]["items"]["type"], "array");
    assert_eq!(
        cube["items"]["items"]["items"]["type"], "string",
        "Vec<Vec<Vec<String>>> must recurse three levels, got: {cube:?}"
    );
}

#[test]
fn vec_of_hashset_field_keeps_inner_items() {
    let schema = NestedCollections::schema().to_json();
    let tag_sets = &schema["properties"]["tag_sets"];
    assert_eq!(tag_sets["type"], "array");
    assert_eq!(tag_sets["items"]["type"], "array");
    assert_eq!(tag_sets["items"]["items"]["type"], "string");
}

#[test]
fn optional_nested_vec_field_keeps_inner_items() {
    let schema = NestedCollections::schema().to_json();
    let opt_matrix = &schema["properties"]["opt_matrix"];
    assert_eq!(opt_matrix["type"], "array");
    assert_eq!(opt_matrix["items"]["type"], "array");
    assert_eq!(opt_matrix["items"]["items"]["type"], "number");
    let required: Vec<&str> = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(!required.contains(&"opt_matrix"));
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct NestedStructMatrix {
    #[llm(description = "Grid of addresses")]
    grid: Vec<Vec<Address>>,
}

#[test]
fn nested_vec_of_structs_embeds_inner_schema() {
    let schema = NestedStructMatrix::schema().to_json();
    let grid = &schema["properties"]["grid"];
    assert_eq!(grid["type"], "array");
    assert_eq!(grid["items"]["type"], "array");
    // Innermost items embed the struct's full object schema.
    assert_eq!(grid["items"]["items"]["type"], "object");
    assert_eq!(
        grid["items"]["items"]["properties"]["street"]["type"],
        "string"
    );
}

// ============================================================================
// Well-known type-name sniffing vs user-defined types named Date/DateTime
// ============================================================================

/// A user-defined struct that happens to share its name with chrono's `Date`.
/// It derives Instructor, so its real object schema must win over the
/// name-sniffed `{type: "string", format: "date"}`.
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Date {
    #[llm(description = "Day of month")]
    day: u8,
    #[llm(description = "Month number")]
    month: u8,
    #[llm(description = "Full year")]
    year: i32,
}

/// Same collision for `DateTime` (no generic parameters, unlike chrono's).
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct DateTime {
    #[llm(description = "Unix timestamp")]
    epoch_seconds: i64,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Appointment {
    title: String,
    date: Date,
    starts_at: DateTime,
    ends_at: Option<Date>,
    reschedule_options: Vec<Date>,
}

#[test]
fn user_defined_date_struct_keeps_its_object_schema() {
    let schema = Appointment::schema().to_json();
    let date = &schema["properties"]["date"];
    assert_eq!(
        date["type"], "object",
        "user struct named Date must keep its derived object schema, got: {date:?}"
    );
    assert_eq!(date["properties"]["day"]["type"], "integer");
    assert_eq!(date["properties"]["month"]["type"], "integer");
    assert_eq!(date["properties"]["year"]["type"], "integer");
    assert!(
        date.get("format").is_none(),
        "user struct named Date must not be sniffed into format: date"
    );
}

#[test]
fn user_defined_datetime_struct_keeps_its_object_schema() {
    let schema = Appointment::schema().to_json();
    let starts_at = &schema["properties"]["starts_at"];
    assert_eq!(starts_at["type"], "object");
    assert_eq!(starts_at["properties"]["epoch_seconds"]["type"], "integer");
    assert!(starts_at.get("format").is_none());
}

#[test]
fn optional_user_defined_date_struct_keeps_its_object_schema() {
    let schema = Appointment::schema().to_json();
    let ends_at = &schema["properties"]["ends_at"];
    assert_eq!(ends_at["type"], "object");
    assert_eq!(ends_at["properties"]["day"]["type"], "integer");
    let required: Vec<&str> = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(!required.contains(&"ends_at"));
}

#[test]
fn vec_of_user_defined_date_struct_embeds_object_items() {
    let schema = Appointment::schema().to_json();
    let opts = &schema["properties"]["reschedule_options"];
    assert_eq!(opts["type"], "array");
    assert_eq!(opts["items"]["type"], "object");
    assert_eq!(opts["items"]["properties"]["year"]["type"], "integer");
}

/// chrono's real date/time types must still be sniffed into string/format
/// schemas (chrono types do not implement SchemaType, so the probe falls back).
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct ChronoEvent {
    name: String,
    on_date: chrono::NaiveDate,
    at: chrono::DateTime<chrono::Utc>,
    local_ts: chrono::NaiveDateTime,
    maybe_date: Option<chrono::NaiveDate>,
    past_dates: Vec<chrono::NaiveDate>,
    past_times: Vec<chrono::DateTime<chrono::Utc>>,
}

#[test]
fn chrono_date_fields_still_sniffed_to_string_format() {
    let schema = ChronoEvent::schema().to_json();

    let on_date = &schema["properties"]["on_date"];
    assert_eq!(on_date["type"], "string");
    assert_eq!(on_date["format"], "date");

    let at = &schema["properties"]["at"];
    assert_eq!(at["type"], "string");
    assert_eq!(at["format"], "date-time");

    let local_ts = &schema["properties"]["local_ts"];
    assert_eq!(local_ts["type"], "string");
    assert_eq!(local_ts["format"], "date-time");

    let maybe_date = &schema["properties"]["maybe_date"];
    assert_eq!(maybe_date["type"], "string");
    assert_eq!(maybe_date["format"], "date");
}

#[test]
fn chrono_date_array_items_still_sniffed_to_string_format() {
    let schema = ChronoEvent::schema().to_json();

    let past_dates = &schema["properties"]["past_dates"];
    assert_eq!(past_dates["type"], "array");
    assert_eq!(past_dates["items"]["type"], "string");
    assert_eq!(past_dates["items"]["format"], "date");

    let past_times = &schema["properties"]["past_times"];
    assert_eq!(past_times["type"], "array");
    assert_eq!(past_times["items"]["type"], "string");
    assert_eq!(past_times["items"]["format"], "date-time");
}

// ============================================================================
// Recursive types through Box: Option<Box<Self>>, Vec<Box<Self>>
// ============================================================================

/// Singly-linked list node: recursion through Option<Box<Self>>.
/// Before the Box-branch $ref guard, calling schema() overflowed the stack.
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct LinkedNode {
    #[llm(description = "Value held by this node")]
    value: i32,
    #[llm(description = "Next node in the list, if any")]
    next: Option<Box<LinkedNode>>,
}

#[test]
fn option_box_self_recursion_terminates_with_ref() {
    // Must terminate (no stack overflow) ...
    let schema = LinkedNode::schema().to_json();
    // ... and use the $defs/$ref indirection like the array guard does.
    assert_eq!(schema["$ref"], "#/$defs/LinkedNode");
    let def = &schema["$defs"]["LinkedNode"];
    assert!(def.is_object(), "$defs.LinkedNode must exist");
    assert_eq!(
        def["properties"]["next"]["$ref"], "#/$defs/LinkedNode",
        "Box<Self> field must be emitted as a $ref, got: {:?}",
        def["properties"]["next"]
    );
    assert_eq!(def["properties"]["value"]["type"], "integer");
    // Option<Box<Self>> must not be required.
    let required: Vec<&str> = def["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(required.contains(&"value"));
    assert!(!required.contains(&"next"));
}

#[test]
fn option_box_self_recursion_schema_name() {
    assert_eq!(LinkedNode::schema_name(), Some("LinkedNode".to_string()));
}

/// Tree node: recursion through Vec<Box<Self>> (array guard with boxed items).
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct BoxedTreeNode {
    #[llm(description = "Node label")]
    label: String,
    #[llm(description = "Child nodes")]
    #[allow(clippy::vec_box)]
    children: Vec<Box<BoxedTreeNode>>,
}

#[test]
fn vec_box_self_recursion_terminates_with_ref() {
    let schema = BoxedTreeNode::schema().to_json();
    assert_eq!(schema["$ref"], "#/$defs/BoxedTreeNode");
    let def = &schema["$defs"]["BoxedTreeNode"];
    let children = &def["properties"]["children"];
    assert_eq!(children["type"], "array");
    assert_eq!(children["items"]["$ref"], "#/$defs/BoxedTreeNode");
}

// ============================================================================
// Generic types deriving Instructor
// ============================================================================

use serde::de::DeserializeOwned;

/// A generic wrapper: the derive must emit generics-aware impl blocks
/// (split_for_impl) for both SchemaType and Instructor.
///
/// The `#[serde(bound(...))]` attribute is required by *serde's own derive*
/// whenever `DeserializeOwned` appears as a struct bound (serde otherwise
/// emits an ambiguous `T: Deserialize<'de>` + `T: DeserializeOwned` impl);
/// it is unrelated to the Instructor derive.
#[derive(Instructor, Serialize, Deserialize, Debug)]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
struct Wrapper<T: SchemaType + Serialize + DeserializeOwned> {
    #[llm(description = "The wrapped value")]
    value: T,
    #[llm(description = "A label for the value")]
    label: String,
}

/// Generics without bounds on the struct itself: the derive adds the
/// SchemaType / serde bounds to its own impls.
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Pair<A, B> {
    first: A,
    second: B,
}

/// Generic enum with a default type parameter and data-carrying variants.
#[derive(Instructor, Serialize, Deserialize, Debug)]
enum Maybe<T = String> {
    Nothing,
    Just(T),
}

#[test]
fn generic_struct_compiles_and_produces_schema() {
    let schema = Wrapper::<i32>::schema().to_json();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["title"], "Wrapper");
    assert_eq!(schema["properties"]["value"]["type"], "integer");
    assert_eq!(schema["properties"]["label"]["type"], "string");
    let required: Vec<&str> = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(required.contains(&"value"));
    assert!(required.contains(&"label"));
}

#[test]
fn generic_struct_with_struct_parameter_embeds_schema() {
    // Instantiated with a derived struct: T's full object schema is embedded.
    let schema = Wrapper::<Address>::schema().to_json();
    let value = &schema["properties"]["value"];
    assert_eq!(value["type"], "object");
    assert_eq!(value["properties"]["street"]["type"], "string");
}

#[test]
fn generic_struct_instructor_impl_validates() {
    // `use rstructor::Instructor` at the top of this file imports both the
    // derive macro and the trait, so `validate` is callable here.
    let wrapped = Wrapper {
        value: 42i64,
        label: "answer".to_string(),
    };
    assert!(wrapped.validate().is_ok());
}

#[test]
fn unbounded_generic_struct_compiles_and_produces_schema() {
    let schema = Pair::<String, f64>::schema().to_json();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["first"]["type"], "string");
    assert_eq!(schema["properties"]["second"]["type"], "number");
}

#[test]
fn generic_enum_compiles_and_produces_schema() {
    let schema = Maybe::<i32>::schema().to_json();
    let any_of = schema["anyOf"].as_array().expect("anyOf for data enum");
    assert_eq!(any_of.len(), 2);
    // The Just variant carries the type parameter's schema.
    let just = any_of
        .iter()
        .find(|v| v["properties"].get("Just").is_some())
        .expect("Just variant present");
    assert_eq!(just["properties"]["Just"]["type"], "integer");
}

/// Pathological combination of the sniffing and recursion fixes: a recursive
/// struct that is itself named `Date`. The self-reference $ref must win over
/// both the name sniff and the SchemaType probe (which would otherwise call
/// its own schema() forever).
mod recursive_date {
    use super::*;

    #[derive(Instructor, Serialize, Deserialize, Debug)]
    pub struct Date {
        #[llm(description = "Node label")]
        pub label: String,
        #[llm(description = "Nested child dates")]
        pub children: Vec<Date>,
    }
}

#[test]
fn recursive_struct_named_date_terminates_with_ref() {
    use rstructor::SchemaType as _;
    let schema = recursive_date::Date::schema().to_json();
    assert_eq!(schema["$ref"], "#/$defs/Date");
    let def = &schema["$defs"]["Date"];
    assert_eq!(def["properties"]["children"]["type"], "array");
    assert_eq!(
        def["properties"]["children"]["items"]["$ref"],
        "#/$defs/Date"
    );
    assert_eq!(def["properties"]["label"]["type"], "string");
}

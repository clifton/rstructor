#![allow(dead_code)]

use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

fn required_names(schema: &Value) -> Vec<&str> {
    schema["required"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect()
}

fn only_variant(schema: &Value) -> &Value {
    let variants = schema["anyOf"].as_array().expect("enum anyOf");
    assert_eq!(variants.len(), 1, "unexpected schema: {schema:#}");
    &variants[0]
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
#[serde(rename_all(serialize = "SCREAMING-KEBAB-CASE", deserialize = "camelCase"))]
struct OrderInput {
    #[serde(skip_deserializing)]
    internal_risk_score: i64,

    #[serde(skip_serializing)]
    client_secret: String,

    #[serde(rename(serialize = "SYMBOL", deserialize = "ticker"))]
    symbol: String,

    order_quantity: i64,

    #[serde(rename(serialize = "LIMIT-PRICE"))]
    limit_price: f64,
}

#[test]
fn struct_schema_uses_deserialization_names_and_omits_skipped_inputs() {
    let schema = OrderInput::schema().to_json();
    let properties = schema["properties"].as_object().expect("properties");

    assert_eq!(
        properties.keys().map(String::as_str).collect::<Vec<_>>(),
        ["clientSecret", "limitPrice", "orderQuantity", "ticker"]
    );
    assert_eq!(
        required_names(&schema),
        ["clientSecret", "ticker", "orderQuantity", "limitPrice"]
    );
    assert!(!properties.contains_key("internalRiskScore"));
    assert!(!properties.contains_key("CLIENT-SECRET"));
    assert!(!properties.contains_key("SYMBOL"));

    let decoded: OrderInput = serde_json::from_value(json!({
        "clientSecret": "desk-a",
        "ticker": "AAPL",
        "orderQuantity": 125_000,
        "limitPrice": 213.75
    }))
    .expect("schema-shaped input must deserialize");
    assert_eq!(
        decoded,
        OrderInput {
            internal_risk_score: 0,
            client_secret: "desk-a".into(),
            symbol: "AAPL".into(),
            order_quantity: 125_000,
            limit_price: 213.75,
        }
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
#[serde(rename_all(serialize = "SCREAMING-KEBAB-CASE"))]
struct SerializeOnlyContainerRename {
    account_id: String,
}

#[test]
fn serialize_only_container_rule_leaves_deserialization_schema_unchanged() {
    let schema = SerializeOnlyContainerRename::schema().to_json();
    assert!(schema["properties"].get("account_id").is_some());
    assert!(schema["properties"].get("ACCOUNT-ID").is_none());

    let decoded: SerializeOnlyContainerRename =
        serde_json::from_value(json!({"account_id": "acct-7"})).unwrap();
    assert_eq!(decoded.account_id, "acct-7");
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
#[serde(rename_all(serialize = "SCREAMING-KEBAB-CASE", deserialize = "snake_case"))]
enum OrderKind {
    #[serde(skip_deserializing)]
    InternalCross,

    #[serde(skip_serializing, rename(serialize = "CLIENT-OUT"))]
    ClientOrder,

    #[serde(rename(serialize = "MARKET-OUT", deserialize = "market"))]
    MarketOrder,
}

#[test]
fn simple_enum_contains_only_deserializable_variant_names() {
    let schema = OrderKind::schema().to_json();
    assert_eq!(schema["enum"], json!(["client_order", "market"]));

    assert_eq!(
        serde_json::from_value::<OrderKind>(json!("client_order")).unwrap(),
        OrderKind::ClientOrder
    );
    assert_eq!(
        serde_json::from_value::<OrderKind>(json!("market")).unwrap(),
        OrderKind::MarketOrder
    );
    assert!(serde_json::from_value::<OrderKind>(json!("internal_cross")).is_err());
    assert!(serde_json::from_value::<OrderKind>(json!("MARKET-OUT")).is_err());
}

#[derive(Debug, Instructor, Serialize, Deserialize)]
enum UnitInputWithSkippedData {
    Active,
    #[serde(skip_deserializing)]
    Internal {
        risk_score: i64,
    },
}

#[derive(Debug, Instructor, Serialize, Deserialize)]
enum NoDeserializableVariant {
    #[serde(skip_deserializing)]
    Internal,
}

#[test]
fn skipped_data_variants_do_not_change_simple_enum_shape() {
    let schema = UnitInputWithSkippedData::schema().to_json();
    assert_eq!(schema["type"], "string");
    assert_eq!(schema["enum"], json!(["Active"]));

    let empty_schema = NoDeserializableVariant::schema().to_json();
    assert_eq!(empty_schema["type"], "string");
    assert_eq!(empty_schema["enum"], json!([]));
}

#[derive(Debug, Instructor, Serialize, Deserialize)]
struct NodeInput {
    value: String,
    #[serde(skip_deserializing)]
    parent: Option<Box<NodeInput>>,
}

#[test]
fn skipped_self_reference_does_not_leave_unused_recursive_definitions() {
    let schema = NodeInput::schema().to_json();
    assert!(schema["properties"].get("value").is_some());
    assert!(schema["properties"].get("parent").is_none());
    assert!(schema.get("$defs").is_none());
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
#[serde(
    rename_all(serialize = "SCREAMING-KEBAB-CASE", deserialize = "snake_case"),
    rename_all_fields(serialize = "UPPERCASE", deserialize = "camelCase")
)]
enum ExternalOrder {
    #[serde(skip_deserializing)]
    Internal { risk_score: i64 },

    #[serde(skip_serializing, rename_all(deserialize = "PascalCase"))]
    Submit {
        #[serde(skip_deserializing)]
        server_id: String,
        client_order_id: String,
        #[serde(rename(serialize = "SIZE", deserialize = "quantity"))]
        size: i64,
    },
}

#[test]
fn external_enum_applies_variant_field_rules_and_skips_read_only_data() {
    let schema = ExternalOrder::schema().to_json();
    let variant = only_variant(&schema);
    let payload = &variant["properties"]["submit"];
    let properties = payload["properties"]
        .as_object()
        .expect("payload properties");

    assert_eq!(
        properties.keys().map(String::as_str).collect::<Vec<_>>(),
        ["ClientOrderId", "quantity"]
    );
    assert_eq!(required_names(payload), ["ClientOrderId", "quantity"]);
    assert!(!properties.contains_key("server_id"));

    let decoded: ExternalOrder = serde_json::from_value(json!({
        "submit": {
            "ClientOrderId": "co-123",
            "quantity": 250
        }
    }))
    .expect("schema-shaped external enum must deserialize");
    assert_eq!(
        decoded,
        ExternalOrder::Submit {
            server_id: String::new(),
            client_order_id: "co-123".into(),
            size: 250,
        }
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
enum ExternalTuple {
    Legs(String, #[serde(skip_deserializing)] bool, i64),
    Ack(#[serde(skip_deserializing)] bool),
}

#[test]
fn external_tuple_schema_removes_skipped_elements_and_collapses_skipped_newtype() {
    let schema = ExternalTuple::schema().to_json();
    let variants = schema["anyOf"].as_array().expect("anyOf");

    let legs = variants
        .iter()
        .find(|variant| variant["properties"].get("Legs").is_some())
        .expect("Legs variant");
    let legs_schema = &legs["properties"]["Legs"];
    assert_eq!(legs_schema["minItems"], 2);
    assert_eq!(legs_schema["maxItems"], 2);
    assert_eq!(legs_schema["items"].as_array().unwrap().len(), 2);

    assert!(
        variants
            .iter()
            .any(|variant| { variant["type"] == "string" && variant["enum"] == json!(["Ack"]) })
    );

    assert_eq!(
        serde_json::from_value::<ExternalTuple>(json!({"Legs": ["AAPL", 100]})).unwrap(),
        ExternalTuple::Legs("AAPL".into(), false, 100)
    );
    assert_eq!(
        serde_json::from_value::<ExternalTuple>(json!("Ack")).unwrap(),
        ExternalTuple::Ack(false)
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all(serialize = "SCREAMING-KEBAB-CASE", deserialize = "snake_case"),
    rename_all_fields(deserialize = "camelCase")
)]
enum InternalOrder {
    #[serde(skip_deserializing)]
    Internal { risk_score: i64 },

    #[serde(skip_serializing)]
    Submit {
        #[serde(skip_deserializing)]
        server_id: String,
        client_order_id: String,
    },
}

#[test]
fn internal_enum_matches_deserialization_tag_and_field_contract() {
    let schema = InternalOrder::schema().to_json();
    let variant = only_variant(&schema);
    let properties = variant["properties"].as_object().expect("properties");

    assert_eq!(properties["kind"]["enum"], json!(["submit"]));
    assert!(properties.contains_key("clientOrderId"));
    assert!(!properties.contains_key("serverId"));
    assert_eq!(required_names(variant), ["kind", "clientOrderId"]);

    let decoded: InternalOrder = serde_json::from_value(json!({
        "kind": "submit",
        "clientOrderId": "co-456"
    }))
    .expect("schema-shaped internal enum must deserialize");
    assert_eq!(
        decoded,
        InternalOrder::Submit {
            server_id: String::new(),
            client_order_id: "co-456".into(),
        }
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "order",
    rename_all(serialize = "SCREAMING-KEBAB-CASE", deserialize = "snake_case"),
    rename_all_fields(deserialize = "camelCase")
)]
enum AdjacentOrder {
    #[serde(skip_deserializing)]
    Internal {
        risk_score: i64,
    },

    #[serde(skip_serializing)]
    Submit {
        #[serde(skip_deserializing)]
        server_id: String,
        client_order_id: String,
    },

    Ack(#[serde(skip_deserializing)] bool),
}

#[test]
fn adjacent_enum_uses_deserialization_contract_for_content_and_skipped_newtypes() {
    let schema = AdjacentOrder::schema().to_json();
    let variants = schema["anyOf"].as_array().expect("anyOf");
    assert_eq!(variants.len(), 2);

    let submit = variants
        .iter()
        .find(|variant| variant["properties"]["kind"]["enum"] == json!(["submit"]))
        .expect("submit variant");
    let content = &submit["properties"]["order"];
    assert!(content["properties"].get("clientOrderId").is_some());
    assert!(content["properties"].get("serverId").is_none());
    assert_eq!(required_names(content), ["clientOrderId"]);

    let ack = variants
        .iter()
        .find(|variant| variant["properties"]["kind"]["enum"] == json!(["ack"]))
        .expect("ack variant");
    assert_eq!(ack["properties"]["order"], json!({"type": "null"}));
    assert_eq!(required_names(ack), ["kind", "order"]);

    let submit_value = json!({
        "kind": "submit",
        "order": {"clientOrderId": "co-789"}
    });
    assert_eq!(
        serde_json::from_value::<AdjacentOrder>(submit_value).unwrap(),
        AdjacentOrder::Submit {
            server_id: String::new(),
            client_order_id: "co-789".into(),
        }
    );
    assert_eq!(
        serde_json::from_value::<AdjacentOrder>(json!({"kind": "ack", "order": null})).unwrap(),
        AdjacentOrder::Ack(false)
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
#[serde(untagged, rename_all_fields(deserialize = "camelCase"))]
enum UntaggedOrder {
    #[serde(skip_deserializing)]
    Internal {
        risk_score: i64,
    },

    Submit {
        #[serde(skip_deserializing)]
        server_id: String,
        client_order_id: String,
    },

    Pair(String, #[serde(skip_deserializing)] bool, i64),

    Ack(#[serde(skip_deserializing)] bool),
}

#[test]
fn untagged_enum_filters_skipped_shapes_fields_and_tuple_elements() {
    let schema = UntaggedOrder::schema().to_json();
    let variants = schema["anyOf"].as_array().expect("anyOf");
    assert_eq!(variants.len(), 3);

    let object = variants
        .iter()
        .find(|variant| variant["type"] == "object")
        .expect("object variant");
    assert!(object["properties"].get("clientOrderId").is_some());
    assert!(object["properties"].get("serverId").is_none());

    let tuple = variants
        .iter()
        .find(|variant| variant["type"] == "array")
        .expect("tuple variant");
    assert_eq!(tuple["minItems"], 2);
    assert_eq!(tuple["maxItems"], 2);
    assert_eq!(tuple["items"].as_array().unwrap().len(), 2);

    assert!(variants.iter().any(Value::is_null) || variants.iter().any(|v| v["type"] == "null"));

    assert_eq!(
        serde_json::from_value::<UntaggedOrder>(json!({"clientOrderId": "co-999"})).unwrap(),
        UntaggedOrder::Submit {
            server_id: String::new(),
            client_order_id: "co-999".into(),
        }
    );
    assert_eq!(
        serde_json::from_value::<UntaggedOrder>(json!(["AAPL", 50])).unwrap(),
        UntaggedOrder::Pair("AAPL".into(), false, 50)
    );
    assert_eq!(
        serde_json::from_value::<UntaggedOrder>(Value::Null).unwrap(),
        UntaggedOrder::Ack(false)
    );
}

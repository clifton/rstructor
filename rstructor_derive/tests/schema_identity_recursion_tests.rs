#![allow(dead_code)]
#![allow(clippy::vec_box)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::marker::PhantomData;
use std::process::Command;
use std::thread;

use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const CONTRACT_CHILD_ENV: &str = "RSTRUCTOR_SCHEMA_GRAPH_CONTRACT_CHILD";

/// Assert that the schema is a closed, minimal local-reference graph:
///
/// - every local `$ref` resolves from the schema document root; and
/// - every `$defs`/`definitions` entry is reachable from the emitted root.
///
/// The reachability walk deliberately does not treat merely being present below
/// `$defs` as use. A definition becomes live only when a reachable schema node
/// references it.
#[track_caller]
fn assert_schema_graph_integrity(schema: &Value) {
    if let Some(definitions) = schema.get("$defs").and_then(Value::as_object) {
        for key in definitions.keys() {
            assert!(
                key.bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')),
                "definition key {key:?} is not URI-fragment safe"
            );
        }
    }

    let definitions = collect_definitions(schema);
    let all_refs = collect_local_refs(schema, true);

    for reference in &all_refs {
        assert!(
            resolve_local_ref(schema, reference).is_some(),
            "local reference {reference:?} does not resolve in schema:\n{schema:#}"
        );
    }

    let mut pending: VecDeque<String> = collect_local_refs(schema, false).into_iter().collect();
    let mut visited_refs = BTreeSet::new();
    let mut used_definitions = BTreeSet::new();

    while let Some(reference) = pending.pop_front() {
        if !visited_refs.insert(reference.clone()) {
            continue;
        }

        let target = resolve_local_ref(schema, &reference).unwrap_or_else(|| {
            panic!("local reference {reference:?} does not resolve in schema:\n{schema:#}")
        });

        for definition_pointer in definitions.keys() {
            if reference == *definition_pointer
                || reference
                    .strip_prefix(definition_pointer.as_str())
                    .is_some_and(|suffix| suffix.starts_with('/'))
            {
                used_definitions.insert(definition_pointer.clone());
            }
        }

        pending.extend(collect_local_refs(target, false));
    }

    let unused: Vec<_> = definitions
        .keys()
        .filter(|pointer| !used_definitions.contains(*pointer))
        .cloned()
        .collect();
    assert!(
        unused.is_empty(),
        "schema contains unreachable definitions {unused:?}:\n{schema:#}"
    );
}

fn collect_definitions(schema: &Value) -> BTreeMap<String, &Value> {
    fn visit<'a>(
        value: &'a Value,
        path: &mut Vec<String>,
        definitions: &mut BTreeMap<String, &'a Value>,
    ) {
        let Some(object) = value.as_object() else {
            return;
        };

        for keyword in ["$defs", "definitions"] {
            if let Some(entries) = object.get(keyword).and_then(Value::as_object) {
                for (name, definition) in entries {
                    path.push(keyword.to_string());
                    path.push(name.clone());
                    definitions.insert(json_pointer(path), definition);
                    visit(definition, path, definitions);
                    path.pop();
                    path.pop();
                }
            }
        }

        for (key, child) in object {
            if key == "$defs" || key == "definitions" {
                continue;
            }
            path.push(key.clone());
            visit(child, path, definitions);
            path.pop();
        }
    }

    let mut definitions = BTreeMap::new();
    visit(schema, &mut Vec::new(), &mut definitions);
    definitions
}

fn collect_local_refs(value: &Value, include_definitions: bool) -> BTreeSet<String> {
    fn visit(value: &Value, include_definitions: bool, references: &mut BTreeSet<String>) {
        match value {
            Value::Object(object) => {
                if let Some(reference) = object.get("$ref").and_then(Value::as_str)
                    && reference.starts_with('#')
                {
                    references.insert(reference.to_string());
                }

                for (key, child) in object {
                    if !include_definitions && (key == "$defs" || key == "definitions") {
                        continue;
                    }
                    visit(child, include_definitions, references);
                }
            }
            Value::Array(values) => {
                for child in values {
                    visit(child, include_definitions, references);
                }
            }
            _ => {}
        }
    }

    let mut references = BTreeSet::new();
    visit(value, include_definitions, &mut references);
    references
}

fn resolve_local_ref<'a>(schema: &'a Value, reference: &str) -> Option<&'a Value> {
    if reference == "#" {
        return Some(schema);
    }

    reference
        .strip_prefix('#')
        .and_then(|pointer| schema.pointer(pointer))
}

fn json_pointer(path: &[String]) -> String {
    let escaped = path
        .iter()
        .map(|segment| segment.replace('~', "~0").replace('/', "~1"))
        .collect::<Vec<_>>()
        .join("/");
    format!("#/{escaped}")
}

fn schema_contains_property(schema: &Value, property: &str) -> bool {
    match schema {
        Value::Object(object) => {
            object
                .get("properties")
                .and_then(Value::as_object)
                .is_some_and(|properties| properties.contains_key(property))
                || object
                    .values()
                    .any(|value| schema_contains_property(value, property))
        }
        Value::Array(values) => values
            .iter()
            .any(|value| schema_contains_property(value, property)),
        _ => false,
    }
}

fn find_tagged_variant<'a>(
    schema: &'a Value,
    tag_name: &str,
    variant_name: &str,
) -> Option<&'a Value> {
    match schema {
        Value::Object(object) => {
            let is_match = object
                .get("properties")
                .and_then(Value::as_object)
                .and_then(|properties| properties.get(tag_name))
                .and_then(|tag| tag.get("enum"))
                .and_then(Value::as_array)
                .is_some_and(|values| {
                    values
                        .iter()
                        .any(|value| value.as_str() == Some(variant_name))
                });
            if is_match {
                return Some(schema);
            }
            object
                .values()
                .find_map(|child| find_tagged_variant(child, tag_name, variant_name))
        }
        Value::Array(values) => values
            .iter()
            .find_map(|child| find_tagged_variant(child, tag_name, variant_name)),
        _ => None,
    }
}

fn find_property_schema<'a>(schema: &'a Value, property: &str) -> Option<&'a Value> {
    match schema {
        Value::Object(object) => {
            if let Some(property_schema) = object
                .get("properties")
                .and_then(Value::as_object)
                .and_then(|properties| properties.get(property))
            {
                return Some(property_schema);
            }

            object
                .values()
                .find_map(|value| find_property_schema(value, property))
        }
        Value::Array(values) => values
            .iter()
            .find_map(|value| find_property_schema(value, property)),
        _ => None,
    }
}

fn count_object_key(value: &Value, key: &str) -> usize {
    match value {
        Value::Object(object) => {
            usize::from(object.contains_key(key))
                + object
                    .values()
                    .map(|child| count_object_key(child, key))
                    .sum::<usize>()
        }
        Value::Array(values) => values
            .iter()
            .map(|child| count_object_key(child, key))
            .sum(),
        _ => 0,
    }
}

fn definitions_with_property(schema: &Value, property: &str) -> Vec<String> {
    collect_definitions(schema)
        .into_iter()
        .filter_map(|(pointer, definition)| {
            definition
                .get("properties")
                .and_then(Value::as_object)
                .is_some_and(|properties| properties.contains_key(property))
                .then_some(pointer)
        })
        .collect()
}

fn definition_property_types(schema: &Value, property: &str) -> BTreeMap<String, String> {
    collect_definitions(schema)
        .into_iter()
        .filter_map(|(pointer, definition)| {
            let property_type = definition
                .get("properties")?
                .get(property)?
                .get("type")?
                .as_str()?;
            Some((pointer, property_type.to_string()))
        })
        .collect()
}

/// Run contracts that intentionally exercise a recursive cycle in a child
/// process. The pre-fix stack overflow then becomes an ordinary assertion
/// failure instead of aborting every test in this integration-test binary.
fn run_recursive_contract(test_name: &str, contract: fn()) {
    if env::var(CONTRACT_CHILD_ENV).as_deref() == Ok(test_name) {
        contract();
        return;
    }

    let output = Command::new(env::current_exe().expect("resolve integration-test executable"))
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .env(CONTRACT_CHILD_ENV, test_name)
        .output()
        .expect("run recursive schema contract in child process");

    assert!(
        output.status.success(),
        "recursive schema contract {test_name:?} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct OrderRevision {
    order_id: String,
    version: u32,
    previous: Option<Box<OrderRevision>>,
}

#[test]
fn direct_recursion_preserves_the_existing_defs_and_ref_contract() {
    let schema = OrderRevision::schema().to_json();

    assert_schema_graph_integrity(&schema);
    assert_eq!(schema["$ref"], "#/$defs/OrderRevision");
    assert_eq!(
        schema["$defs"]["OrderRevision"]["properties"]["previous"]["$ref"],
        "#/$defs/OrderRevision"
    );
    assert_eq!(
        schema["$defs"]["OrderRevision"]["properties"]["version"]["type"],
        "integer"
    );

    let decoded: OrderRevision = serde_json::from_value(json!({
        "order_id": "ORD-20260728-0042",
        "version": 2,
        "previous": {
            "order_id": "ORD-20260728-0042",
            "version": 1,
            "previous": null
        }
    }))
    .expect("a finite order-revision chain must deserialize");
    assert_eq!(decoded.version, 2);
    assert_eq!(decoded.previous.expect("prior revision").version, 1);
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct ManagedFund {
    fund_lei: String,
    nav_usd_cents: i64,
    prime_broker: Option<Box<PrimeBrokerRelationship>>,
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct PrimeBrokerRelationship {
    broker_lei: String,
    sponsored_funds: Vec<ManagedFund>,
}

fn struct_to_struct_contract() {
    let schema = ManagedFund::schema().to_json();

    assert_schema_graph_integrity(&schema);
    assert!(schema_contains_property(&schema, "fund_lei"));
    assert!(schema_contains_property(&schema, "broker_lei"));
    assert!(
        !definitions_with_property(&schema, "fund_lei").is_empty(),
        "the fund type must have a graph definition:\n{schema:#}"
    );
    assert!(
        !definitions_with_property(&schema, "broker_lei").is_empty(),
        "the prime-broker type must have a graph definition:\n{schema:#}"
    );

    let decoded: ManagedFund = serde_json::from_value(json!({
        "fund_lei": "5493001KJTIIGC8Y1R12",
        "nav_usd_cents": 12_575_000_000_i64,
        "prime_broker": {
            "broker_lei": "7H6GLXDRUGQFU57RNE97",
            "sponsored_funds": []
        }
    }))
    .expect("a finite fund/prime-broker graph must deserialize");
    assert_eq!(
        decoded.prime_broker.expect("prime broker").sponsored_funds,
        Vec::<ManagedFund>::new()
    );
}

#[test]
fn struct_to_struct_mutual_recursion_is_a_closed_schema_graph() {
    run_recursive_contract(
        "struct_to_struct_mutual_recursion_is_a_closed_schema_graph",
        struct_to_struct_contract,
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct AllocationNode {
    sleeve_id: String,
    transition: Option<Box<AllocationTransition>>,
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
enum AllocationTransition {
    Hold,
    Rebalance {
        target_weight_bps: i32,
        destination: Box<AllocationNode>,
    },
}

fn struct_to_enum_contract() {
    let schema = AllocationNode::schema().to_json();

    assert_schema_graph_integrity(&schema);
    assert!(schema_contains_property(&schema, "sleeve_id"));
    assert!(schema_contains_property(&schema, "transition"));
    assert!(schema_contains_property(&schema, "target_weight_bps"));
    assert!(schema_contains_property(&schema, "destination"));

    let decoded: AllocationNode = serde_json::from_value(json!({
        "sleeve_id": "GLOBAL-MACRO",
        "transition": {
            "Rebalance": {
                "target_weight_bps": 3_500,
                "destination": {
                    "sleeve_id": "UST-DURATION",
                    "transition": "Hold"
                }
            }
        }
    }))
    .expect("a finite allocation/transition graph must deserialize");
    assert_eq!(decoded.sleeve_id, "GLOBAL-MACRO");
}

#[test]
fn struct_to_enum_mutual_recursion_is_a_closed_schema_graph() {
    run_recursive_contract(
        "struct_to_enum_mutual_recursion_is_a_closed_schema_graph",
        struct_to_enum_contract,
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
enum OrderLifecycle {
    Accepted {
        order_id: String,
        next: Option<Box<SettlementLifecycle>>,
    },
    Rejected {
        reason_code: String,
    },
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
enum SettlementLifecycle {
    Pending {
        custodian_bic: String,
        fallback: Box<OrderLifecycle>,
    },
    Settled {
        settlement_instruction_id: String,
    },
}

fn enum_to_enum_contract() {
    let schema = OrderLifecycle::schema().to_json();

    assert_schema_graph_integrity(&schema);
    assert!(schema_contains_property(&schema, "order_id"));
    assert!(schema_contains_property(&schema, "custodian_bic"));
    assert!(schema_contains_property(&schema, "fallback"));
    assert!(schema_contains_property(
        &schema,
        "settlement_instruction_id"
    ));
    assert!(
        collect_definitions(&schema).len() >= 2,
        "both enum identities must participate in the recursive graph:\n{schema:#}"
    );

    let decoded: OrderLifecycle = serde_json::from_value(json!({
        "Accepted": {
            "order_id": "FIX-UST-20260728-8821",
            "next": {
                "Pending": {
                    "custodian_bic": "IRVTUS3N",
                    "fallback": {
                        "Rejected": {
                            "reason_code": "CUSTODY_CUTOFF"
                        }
                    }
                }
            }
        }
    }))
    .expect("a finite order/settlement lifecycle graph must deserialize");
    assert!(matches!(decoded, OrderLifecycle::Accepted { .. }));
}

#[test]
fn enum_to_enum_mutual_recursion_is_a_closed_schema_graph() {
    run_recursive_contract(
        "enum_to_enum_mutual_recursion_is_a_closed_schema_graph",
        enum_to_enum_contract,
    );
}

mod same_short_name {
    use super::*;

    pub mod cme {
        use super::*;

        #[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
        pub struct Order {
            pub cme_order_id: String,
            pub ice_hedge: Option<Box<super::ice::Order>>,
        }
    }

    pub mod ice {
        use super::*;

        #[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
        pub struct Order {
            pub ice_order_id: String,
            pub originating_cme_order: Option<Box<super::cme::Order>>,
        }
    }
}

#[test]
fn same_short_name_types_keep_distinct_schema_identities() {
    let schema = same_short_name::cme::Order::schema().to_json();

    assert_schema_graph_integrity(&schema);
    assert!(schema_contains_property(&schema, "cme_order_id"));
    assert!(
        schema_contains_property(&schema, "ice_order_id"),
        "the ICE Order must not be mistaken for CME Order merely because both have the short name `Order`:\n{schema:#}"
    );

    let cme_definitions = definitions_with_property(&schema, "cme_order_id");
    let ice_definitions = definitions_with_property(&schema, "ice_order_id");
    assert!(
        !cme_definitions.is_empty() && !ice_definitions.is_empty(),
        "both venue-specific Order types need definitions:\n{schema:#}"
    );
    assert!(
        cme_definitions
            .iter()
            .all(|pointer| !ice_definitions.contains(pointer)),
        "CME and ICE Order resolved to the same definition:\n{schema:#}"
    );

    let decoded: same_short_name::cme::Order = serde_json::from_value(json!({
        "cme_order_id": "CME-ES-202609-91827",
        "ice_hedge": {
            "ice_order_id": "ICE-DX-202609-44102",
            "originating_cme_order": null
        }
    }))
    .expect("a finite cross-venue order graph must deserialize");
    assert_eq!(
        decoded.ice_hedge.expect("ICE hedge").ice_order_id,
        "ICE-DX-202609-44102"
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct ConsolidatedRiskBook {
    desks_by_code: BTreeMap<String, RiskDesk>,
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct RiskDesk {
    portfolios: Vec<RiskPortfolio>,
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct RiskPortfolio {
    hedge_books_by_currency: BTreeMap<String, Box<ConsolidatedRiskBook>>,
}

fn three_type_container_cycle_contract() {
    let schema = ConsolidatedRiskBook::schema().to_json();

    assert_schema_graph_integrity(&schema);

    let desks = find_property_schema(&schema, "desks_by_code")
        .expect("risk book must expose its dynamic desk map");
    assert_eq!(desks["type"], "object");
    assert!(
        desks.get("additionalProperties").is_some(),
        "desk map must describe its value type:\n{schema:#}"
    );

    let portfolios =
        find_property_schema(&schema, "portfolios").expect("risk desk must expose portfolios");
    assert_eq!(portfolios["type"], "array");
    assert!(
        portfolios.get("items").is_some(),
        "portfolio collection must describe its item type:\n{schema:#}"
    );

    let hedge_books = find_property_schema(&schema, "hedge_books_by_currency")
        .expect("portfolio must expose its dynamic hedge-book map");
    assert_eq!(hedge_books["type"], "object");
    assert!(
        hedge_books.get("additionalProperties").is_some(),
        "hedge-book map must describe its recursive value type:\n{schema:#}"
    );
    assert!(
        collect_definitions(&schema).len() >= 3,
        "all three concrete types must have graph identities:\n{schema:#}"
    );

    let decoded: ConsolidatedRiskBook = serde_json::from_value(json!({
        "desks_by_code": {
            "RATES-NY": {
                "portfolios": [{
                    "hedge_books_by_currency": {
                        "JPY": {
                            "desks_by_code": {}
                        }
                    }
                }]
            }
        }
    }))
    .expect("a finite three-type market-risk graph must deserialize");
    assert!(
        decoded.desks_by_code["RATES-NY"].portfolios[0]
            .hedge_books_by_currency
            .contains_key("JPY")
    );
}

#[test]
fn three_type_cycle_through_maps_and_collections_is_closed() {
    run_recursive_contract(
        "three_type_cycle_through_maps_and_collections_is_closed",
        three_type_container_cycle_contract,
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct SettlementLink {
    link_id: String,
    edge: SettlementEdge,
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
#[serde(tag = "edge_type")]
enum SettlementEdge {
    Terminal { settlement_account: String },
    Next(Box<SettlementLink>),
}

fn internally_tagged_recursive_newtype_contract() {
    let schema = SettlementLink::schema().to_json();

    assert_schema_graph_integrity(&schema);
    assert!(
        !serde_json::to_string(&schema)
            .expect("schema serialization")
            .contains("__rstructor_deferred_flatten"),
        "private deferred-flatten markers must never escape schema generation:\n{schema:#}"
    );

    let next = find_tagged_variant(&schema, "edge_type", "Next")
        .expect("recursive internally tagged Next variant");
    let properties = next["properties"]
        .as_object()
        .expect("Next variant properties");
    assert!(properties.contains_key("edge_type"));
    assert!(
        properties.contains_key("link_id"),
        "the active recursive inner struct must be flattened after graph completion:\n{schema:#}"
    );
    assert!(
        properties.contains_key("edge"),
        "recursive inner fields must not be silently dropped:\n{schema:#}"
    );

    let decoded: SettlementLink = serde_json::from_value(json!({
        "link_id": "ALLOC-20260728-001",
        "edge": {
            "edge_type": "Next",
            "link_id": "ALLOC-20260728-002",
            "edge": {
                "edge_type": "Terminal",
                "settlement_account": "DTC-00001234"
            }
        }
    }))
    .expect("a finite recursive settlement link must deserialize");
    assert!(matches!(decoded.edge, SettlementEdge::Next(_)));
}

#[test]
fn internally_tagged_recursive_newtype_keeps_flattened_fields() {
    run_recursive_contract(
        "internally_tagged_recursive_newtype_keeps_flattened_fields",
        internally_tagged_recursive_newtype_contract,
    );
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct ManualFixedIncomeLeg {
    cusip: String,
}

impl SchemaType for ManualFixedIncomeLeg {
    fn schema() -> rstructor::Schema {
        rstructor::Schema::new(json!({
            "$ref": "#/$defs/InstrumentLeg",
            "$defs": {
                "InstrumentLeg": {
                    "type": "object",
                    "properties": {
                        "cusip": {"type": "string"}
                    },
                    "required": ["cusip"]
                }
            }
        }))
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct ManualEquityLeg {
    ticker: String,
}

impl SchemaType for ManualEquityLeg {
    fn schema() -> rstructor::Schema {
        rstructor::Schema::new(json!({
            "$ref": "#/$defs/InstrumentLeg",
            "$defs": {
                "InstrumentLeg": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"}
                    },
                    "required": ["ticker"]
                }
            }
        }))
    }
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct ManualMultiAssetTrade {
    bond: ManualFixedIncomeLeg,
    equity: ManualEquityLeg,
}

#[test]
fn manual_schema_defs_are_hoisted_and_collisions_are_rewritten() {
    let schema = ManualMultiAssetTrade::schema().to_json();

    assert_schema_graph_integrity(&schema);
    assert_eq!(
        count_object_key(&schema, "$defs"),
        1,
        "manual definitions must be hoisted to the document root:\n{schema:#}"
    );
    let definitions = schema["$defs"]
        .as_object()
        .expect("hoisted manual definitions");
    assert_eq!(definitions.len(), 2);
    assert!(
        definitions
            .values()
            .any(|definition| schema_contains_property(definition, "cusip"))
    );
    assert!(
        definitions
            .values()
            .any(|definition| schema_contains_property(definition, "ticker"))
    );

    let bond_ref = schema["properties"]["bond"]["$ref"]
        .as_str()
        .expect("bond reference");
    let equity_ref = schema["properties"]["equity"]["$ref"]
        .as_str()
        .expect("equity reference");
    assert_ne!(
        bond_ref, equity_ref,
        "different manual schemas sharing a local definition name must not collapse"
    );
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct ManualRateNode {
    value: f64,
}

impl SchemaType for ManualRateNode {
    fn schema() -> rstructor::Schema {
        rstructor::Schema::new(json!({
            "$ref": "#/$defs/Node",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "value": {"$ref": "#/$defs/Leaf"}
                    },
                    "required": ["value"]
                },
                "Leaf": {"type": "number"}
            }
        }))
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct ManualCreditNode {
    value: String,
}

impl SchemaType for ManualCreditNode {
    fn schema() -> rstructor::Schema {
        rstructor::Schema::new(json!({
            "$ref": "#/$defs/Node",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "value": {"$ref": "#/$defs/Leaf"}
                    },
                    "required": ["value"]
                },
                "Leaf": {"type": "string"}
            }
        }))
    }
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct ManualCrossDocumentBook {
    rate_shock: ManualRateNode,
    credit_regime: ManualCreditNode,
}

#[test]
fn manual_transitive_definition_collisions_do_not_cross_bind() {
    let schema = ManualCrossDocumentBook::schema().to_json();

    assert_schema_graph_integrity(&schema);
    let leaf_type = |property: &str| {
        let node_reference = schema["properties"][property]["$ref"]
            .as_str()
            .expect("manual node reference");
        let node = resolve_local_ref(&schema, node_reference).expect("manual node definition");
        let leaf_reference = node["properties"]["value"]["$ref"]
            .as_str()
            .expect("manual leaf reference");
        resolve_local_ref(&schema, leaf_reference)
            .and_then(|leaf| leaf.get("type"))
            .and_then(Value::as_str)
            .expect("manual leaf type")
    };

    assert_eq!(leaf_type("rate_shock"), "number");
    assert_eq!(leaf_type("credit_regime"), "string");
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct ManualDeskTree {
    desk: String,
    child: Option<Box<ManualDeskTree>>,
}

impl SchemaType for ManualDeskTree {
    fn schema() -> rstructor::Schema {
        rstructor::Schema::new(json!({
            "title": "Node",
            "$ref": "#/$defs/Node",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "desk": {"type": "string"},
                        "child": {
                            "anyOf": [
                                {"$ref": "#"},
                                {"type": "null"}
                            ]
                        }
                    },
                    "required": ["desk"]
                }
            }
        }))
    }
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct ManualDeskEnvelope {
    as_of_date: String,
    hierarchy: ManualDeskTree,
}

#[test]
fn embedded_manual_root_references_keep_their_original_scope() {
    let schema = ManualDeskEnvelope::schema().to_json();

    assert_schema_graph_integrity(&schema);
    let hierarchy_reference = schema["properties"]["hierarchy"]["$ref"]
        .as_str()
        .expect("embedded manual root reference");
    let hierarchy_root =
        resolve_local_ref(&schema, hierarchy_reference).expect("embedded manual root definition");
    assert_eq!(
        hierarchy_root["title"], "Node",
        "embedded-root sibling constraints must remain on the wrapper"
    );
    let node_reference = hierarchy_root["$ref"]
        .as_str()
        .expect("manual root's local Node reference");
    let hierarchy =
        resolve_local_ref(&schema, node_reference).expect("hoisted manual Node definition");
    assert!(
        hierarchy["properties"]["child"]["anyOf"]
            .as_array()
            .expect("nullable recursive child")
            .iter()
            .any(|branch| {
                branch.get("$ref").and_then(Value::as_str) == Some(hierarchy_reference)
            }),
        "manual `#` must resolve to the embedded manual root, not the derived parent:\n{schema:#}"
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct BorrowedPositionTree<'a> {
    position_id: String,
    children: Vec<Box<BorrowedPositionTree<'a>>>,
    #[serde(skip)]
    marker: PhantomData<&'a ()>,
}

fn lifetime_parameterized_recursion_contract() {
    let schema = BorrowedPositionTree::<'static>::schema().to_json();
    assert_schema_graph_integrity(&schema);
    assert!(schema.get("$ref").is_some());
}

#[test]
fn lifetime_parameterized_recursive_derive_remains_supported() {
    run_recursive_contract(
        "lifetime_parameterized_recursive_derive_remains_supported",
        lifetime_parameterized_recursion_contract,
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct ScenarioNode<T> {
    shock: T,
    children: Vec<Box<ScenarioNode<T>>>,
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct CrossAssetScenarioBook {
    rates_bps: ScenarioNode<i64>,
    volatility_regime: ScenarioNode<String>,
}

fn generic_identity_contract() {
    let schema = CrossAssetScenarioBook::schema().to_json();

    assert_schema_graph_integrity(&schema);
    let shock_types = definition_property_types(&schema, "shock");
    assert!(
        shock_types.values().any(|kind| kind == "integer"),
        "ScenarioNode<i64> needs an integer-valued definition:\n{schema:#}"
    );
    assert!(
        shock_types.values().any(|kind| kind == "string"),
        "ScenarioNode<String> needs a string-valued definition:\n{schema:#}"
    );
    assert!(
        shock_types.len() >= 2,
        "generic instantiations must not collapse onto one definition identity:\n{schema:#}"
    );

    let decoded: CrossAssetScenarioBook = serde_json::from_value(json!({
        "rates_bps": {
            "shock": -75,
            "children": [
                {"shock": 25, "children": []}
            ]
        },
        "volatility_regime": {
            "shock": "VOLMAGEDDON",
            "children": [
                {"shock": "NORMALIZATION", "children": []}
            ]
        }
    }))
    .expect("distinct finite generic scenario graphs must deserialize");
    assert_eq!(decoded.rates_bps.children[0].shock, 25);
    assert_eq!(decoded.volatility_regime.children[0].shock, "NORMALIZATION");
}

#[test]
fn generic_instantiations_keep_distinct_schema_identities() {
    run_recursive_contract(
        "generic_instantiations_keep_distinct_schema_identities",
        generic_identity_contract,
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct PositionTree {
    position_id: String,
    children: Vec<Box<PositionTree>>,
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct PortfolioSnapshot {
    portfolio_id: String,
    root_position: PositionTree,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct ManualPositionWrapper {
    root: PositionTree,
}

impl SchemaType for ManualPositionWrapper {
    fn schema() -> rstructor::Schema {
        rstructor::Schema::new(json!({
            "type": "object",
            "properties": {
                "root": PositionTree::schema().to_json()
            },
            "required": ["root"]
        }))
    }
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
struct WrappedPortfolioSnapshot {
    portfolio_id: String,
    positions: ManualPositionWrapper,
}

#[test]
fn acyclic_parent_hoists_recursive_child_defs_to_the_document_root() {
    let schema = PortfolioSnapshot::schema().to_json();

    assert_schema_graph_integrity(&schema);
    assert!(
        schema.get("$defs").is_some(),
        "the recursive child definition must be attached at the document root:\n{schema:#}"
    );
    assert_eq!(
        count_object_key(&schema, "$defs"),
        1,
        "recursive child definitions must not remain nested under a property:\n{schema:#}"
    );

    let root_position =
        find_property_schema(&schema, "root_position").expect("snapshot root position");
    assert!(
        root_position.get("$ref").is_some(),
        "the acyclic parent must reference the hoisted recursive child:\n{schema:#}"
    );

    let decoded: PortfolioSnapshot = serde_json::from_value(json!({
        "portfolio_id": "MASTER-FUND-USD",
        "root_position": {
            "position_id": "UST-10Y-SWAP",
            "children": [{
                "position_id": "SOFR-HEDGE",
                "children": []
            }]
        }
    }))
    .expect("a finite portfolio position tree must deserialize");
    assert_eq!(decoded.root_position.children[0].position_id, "SOFR-HEDGE");
}

#[test]
fn manual_wrapper_hoists_nested_derived_recursive_defs() {
    let schema = WrappedPortfolioSnapshot::schema().to_json();

    assert_schema_graph_integrity(&schema);
    assert_eq!(
        count_object_key(&schema, "$defs"),
        1,
        "recursive definitions nested inside a manual wrapper must be hoisted:\n{schema:#}"
    );
    assert!(
        find_property_schema(&schema, "root")
            .and_then(|root| root.get("$ref"))
            .is_some(),
        "the manual wrapper must retain its reference to the hoisted position tree:\n{schema:#}"
    );
}

#[derive(Debug, Instructor, PartialEq, Serialize, Deserialize)]
#[serde(rename_all(serialize = "SCREAMING-KEBAB-CASE", deserialize = "camelCase"))]
struct RegulatoryFiling {
    filing_id: String,

    #[serde(rename(serialize = "NOTIONAL-USD-CENTS", deserialize = "notionalCents"))]
    notional_usd_cents: i64,

    #[serde(skip_deserializing)]
    superseded_by: Option<Box<RegulatoryFiling>>,
}

#[test]
fn directional_rename_and_skipped_recursive_edge_emit_no_unused_definition() {
    let schema = RegulatoryFiling::schema().to_json();

    assert_schema_graph_integrity(&schema);
    let properties = schema["properties"]
        .as_object()
        .expect("regulatory filing properties");
    assert!(properties.contains_key("filingId"));
    assert!(properties.contains_key("notionalCents"));
    assert!(!properties.contains_key("FILING-ID"));
    assert!(!properties.contains_key("NOTIONAL-USD-CENTS"));
    assert!(!properties.contains_key("supersededBy"));
    assert!(
        collect_definitions(&schema).is_empty(),
        "a skipped recursive edge must not manufacture an unreachable definition:\n{schema:#}"
    );
    assert!(
        collect_local_refs(&schema, true).is_empty(),
        "a skipped recursive edge must not manufacture a reference:\n{schema:#}"
    );

    let decoded: RegulatoryFiling = serde_json::from_value(json!({
        "filingId": "13F-2026Q2-000184",
        "notionalCents": 8_250_000_000_i64
    }))
    .expect("schema-shaped regulatory filing input must deserialize");
    assert_eq!(decoded.notional_usd_cents, 8_250_000_000);
    assert!(decoded.superseded_by.is_none());
}

fn repeated_determinism_contract() {
    let baseline = ManagedFund::schema().to_json();
    assert_schema_graph_integrity(&baseline);

    for iteration in 0..32 {
        assert_eq!(
            ManagedFund::schema().to_json(),
            baseline,
            "schema changed on sequential generation {iteration}"
        );
    }
}

#[test]
fn repeated_generation_is_deterministic() {
    run_recursive_contract(
        "repeated_generation_is_deterministic",
        repeated_determinism_contract,
    );
}

fn concurrent_determinism_contract() {
    let baseline = ManagedFund::schema().to_json();
    assert_schema_graph_integrity(&baseline);

    let handles: Vec<_> = (0..16)
        .map(|_| thread::spawn(|| ManagedFund::schema().to_json()))
        .collect();

    for (worker, handle) in handles.into_iter().enumerate() {
        assert_eq!(
            handle.join().expect("schema worker must not panic"),
            baseline,
            "schema changed in concurrent worker {worker}"
        );
    }
}

#[test]
fn concurrent_generation_is_deterministic() {
    run_recursive_contract(
        "concurrent_generation_is_deterministic",
        concurrent_determinism_contract,
    );
}

//! Tests for `#[serde(rename_all = ...)]` / `#[serde(rename = ...)]` interaction with the
//! FIELD NAMES of struct-style enum variants in the generated JSON schema.
//!
//! ## Why these tests exist (a reported "bug" that is NOT a bug)
//!
//! A coverage report flagged that container-level `#[serde(rename_all = "...")]` is not applied
//! to the field names inside struct-style enum variants (only to top-level struct fields and to
//! variant *names*), claiming this makes the schema advertise e.g. `card_number` while serde
//! expects `cardNumber`, causing deserialization failures.
//!
//! This was verified against serde directly and the premise is **wrong**: serde's container-level
//! `rename_all` on an enum renames **variant names only**, never the field names inside a
//! struct-style variant. Concretely:
//!
//! ```text
//! #[serde(rename_all = "camelCase")]
//! enum E { CreditCard { card_number: String } }
//!
//! serde_json::to_string(&E::CreditCard { card_number: "x".into() })
//!   == r#"{"creditCard":{"card_number":"x"}}"#
//! //         ^^^^^^^^^^ variant renamed       ^^^^^^^^^^^ field NOT renamed
//! ```
//!
//! To rename the inner fields with serde you must put a *per-variant* `#[serde(rename_all = ...)]`
//! on the variant itself (which rstructor does not currently parse — a separate, genuine gap).
//!
//! Therefore the current rstructor codegen (which leaves variant field names alone and only honors
//! a field-level `#[serde(rename = ...)]`) is **correct**: it matches serde exactly. "Fixing" the
//! schema to emit `cardNumber` would *introduce* the very mismatch the report claimed to prevent.
//!
//! These tests lock in that correct behavior across externally-tagged, internally-tagged, and
//! adjacently-tagged enums, plus a field-level `#[serde(rename)]` override case, and prove the
//! schema field keys match serde's actual round-trip serialization keys.

#[cfg(test)]
mod derive_rename_enum_tests {
    use rstructor::{Instructor, SchemaType};
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    /// Find the `anyOf` member whose `properties` map contains `variant_key` (the renamed variant
    /// name) — i.e. the externally-tagged schema for a given variant.
    fn external_variant<'a>(schema: &'a Value, variant_key: &str) -> &'a Value {
        schema["anyOf"]
            .as_array()
            .expect("complex enum schema should be anyOf")
            .iter()
            .find(|m| {
                m.get("properties")
                    .and_then(|p| p.get(variant_key))
                    .is_some()
            })
            .unwrap_or_else(|| panic!("no anyOf member with variant key {variant_key}"))
    }

    // =====================================================================
    // Externally-tagged: container rename_all renames the VARIANT name only,
    // the inner field name is left as-is (matches serde).
    // =====================================================================

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq, Clone)]
    #[serde(rename_all = "camelCase")]
    enum ExternalPayment {
        CreditCard { card_number: String },
        BankTransfer { account_id: String },
    }

    #[test]
    fn external_tagged_rename_all_renames_variant_not_inner_field() {
        let schema_json = ExternalPayment::schema().to_json();

        // Variant name is renamed by rename_all -> "creditCard".
        let member = external_variant(&schema_json, "creditCard");
        let inner = &member["properties"]["creditCard"];
        let inner_props = inner["properties"]
            .as_object()
            .expect("variant should carry an object with properties");

        // The INNER field name must stay snake_case (serde does not apply rename_all here).
        assert!(
            inner_props.contains_key("card_number"),
            "inner field must remain 'card_number' to match serde; got keys {:?}",
            inner_props.keys().collect::<Vec<_>>()
        );
        assert!(
            !inner_props.contains_key("cardNumber"),
            "inner field must NOT be camelCased (would mismatch serde)"
        );

        // required references the un-renamed inner field name.
        let required = inner["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "card_number"));
    }

    #[test]
    fn external_tagged_schema_keys_match_serde_serialization() {
        // The schema's keys must be exactly the keys serde produces.
        let value = ExternalPayment::CreditCard {
            card_number: "4111".into(),
        };
        let serialized = serde_json::to_value(&value).unwrap();

        // Variant key renamed, inner field key NOT renamed.
        assert_eq!(serialized["creditCard"]["card_number"], Value::from("4111"));
        assert!(serialized["creditCard"].get("cardNumber").is_none());

        // Round-trip: serde deserializes the exact shape the schema advertises.
        let back: ExternalPayment = serde_json::from_value(serialized).unwrap();
        assert_eq!(back, value);

        // And the schema's inner-field key is the same one serde just emitted.
        let schema_json = ExternalPayment::schema().to_json();
        let inner_props = external_variant(&schema_json, "creditCard")["properties"]["creditCard"]
            ["properties"]
            .as_object()
            .unwrap();
        assert!(inner_props.contains_key("card_number"));
    }

    // =====================================================================
    // Internally-tagged: same rule — variant name (the tag value) is renamed,
    // inner fields are flattened alongside the tag with their original names.
    // =====================================================================

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq, Clone)]
    #[serde(tag = "kind", rename_all = "camelCase")]
    enum InternalPayment {
        CreditCard { card_number: String },
        BankTransfer { account_id: String },
    }

    #[test]
    fn internal_tagged_rename_all_renames_tag_value_not_inner_field() {
        let schema_json = InternalPayment::schema().to_json();
        let members = schema_json["anyOf"].as_array().unwrap();

        // Locate the member whose tag enum is the renamed "creditCard".
        let member = members
            .iter()
            .find(|m| m["properties"]["kind"]["enum"][0] == "creditCard")
            .expect("should find creditCard tag member");

        let props = member["properties"].as_object().unwrap();
        // Tag present, inner field flattened with its ORIGINAL name.
        assert!(props.contains_key("kind"));
        assert!(
            props.contains_key("card_number"),
            "flattened inner field must remain 'card_number'; got {:?}",
            props.keys().collect::<Vec<_>>()
        );
        assert!(!props.contains_key("cardNumber"));

        let required = member["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "kind"));
        assert!(required.iter().any(|v| v == "card_number"));
    }

    #[test]
    fn internal_tagged_schema_keys_match_serde_serialization() {
        let value = InternalPayment::CreditCard {
            card_number: "4111".into(),
        };
        let serialized = serde_json::to_value(&value).unwrap();

        // serde flattens: {"kind":"creditCard","card_number":"4111"}.
        assert_eq!(serialized["kind"], Value::from("creditCard"));
        assert_eq!(serialized["card_number"], Value::from("4111"));
        assert!(serialized.get("cardNumber").is_none());

        let back: InternalPayment = serde_json::from_value(serialized).unwrap();
        assert_eq!(back, value);
    }

    // =====================================================================
    // Adjacently-tagged: tag value renamed; content object carries inner
    // fields with their original names.
    // =====================================================================

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq, Clone)]
    #[serde(tag = "t", content = "c", rename_all = "camelCase")]
    enum AdjacentPayment {
        CreditCard { card_number: String },
    }

    #[test]
    fn adjacent_tagged_rename_all_renames_tag_value_not_inner_field() {
        let schema_json = AdjacentPayment::schema().to_json();
        let members = schema_json["anyOf"].as_array().unwrap();
        let member = members
            .iter()
            .find(|m| m["properties"]["t"]["enum"][0] == "creditCard")
            .expect("should find creditCard adjacently-tagged member");

        let content_props = member["properties"]["c"]["properties"]
            .as_object()
            .expect("adjacent content should be an object with properties");
        assert!(
            content_props.contains_key("card_number"),
            "content inner field must remain 'card_number'; got {:?}",
            content_props.keys().collect::<Vec<_>>()
        );
        assert!(!content_props.contains_key("cardNumber"));
    }

    #[test]
    fn adjacent_tagged_schema_keys_match_serde_serialization() {
        let value = AdjacentPayment::CreditCard {
            card_number: "4111".into(),
        };
        let serialized = serde_json::to_value(&value).unwrap();

        // {"t":"creditCard","c":{"card_number":"4111"}}
        assert_eq!(serialized["t"], Value::from("creditCard"));
        assert_eq!(serialized["c"]["card_number"], Value::from("4111"));
        assert!(serialized["c"].get("cardNumber").is_none());

        let back: AdjacentPayment = serde_json::from_value(serialized).unwrap();
        assert_eq!(back, value);
    }

    // =====================================================================
    // Field-level #[serde(rename)] inside a struct-style variant IS honored
    // (this is the one rename that serde actually applies to inner fields).
    // =====================================================================

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq, Clone)]
    #[serde(rename_all = "camelCase")]
    enum FieldRenameVariant {
        CreditCard {
            #[serde(rename = "ccNum")]
            card_number: String,
            // Not renamed: container rename_all must NOT touch this inner field.
            holder_name: String,
        },
    }

    #[test]
    fn field_level_rename_inside_variant_is_applied() {
        let schema_json = FieldRenameVariant::schema().to_json();
        let inner_props = external_variant(&schema_json, "creditCard")["properties"]["creditCard"]
            ["properties"]
            .as_object()
            .unwrap();

        // Field-level rename wins.
        assert!(
            inner_props.contains_key("ccNum"),
            "field-level #[serde(rename)] should produce 'ccNum'; got {:?}",
            inner_props.keys().collect::<Vec<_>>()
        );
        assert!(!inner_props.contains_key("card_number"));

        // The non-renamed field is left untouched by container rename_all.
        assert!(
            inner_props.contains_key("holder_name"),
            "non-renamed inner field must remain 'holder_name'"
        );
        assert!(!inner_props.contains_key("holderName"));
    }

    #[test]
    fn field_level_rename_inside_variant_matches_serde() {
        let value = FieldRenameVariant::CreditCard {
            card_number: "4111".into(),
            holder_name: "Ada".into(),
        };
        let serialized = serde_json::to_value(&value).unwrap();

        // serde: variant renamed to creditCard; inner field "ccNum" (explicit) + "holder_name".
        assert_eq!(serialized["creditCard"]["ccNum"], Value::from("4111"));
        assert_eq!(serialized["creditCard"]["holder_name"], Value::from("Ada"));
        assert!(serialized["creditCard"].get("card_number").is_none());

        // Deserializing the schema-advertised keys round-trips.
        let back: FieldRenameVariant = serde_json::from_value(serialized).unwrap();
        assert_eq!(back, value);
    }
}

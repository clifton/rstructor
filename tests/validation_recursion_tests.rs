//! Recursion coverage for the public `Instructor::validate()` API.
//!
//! These tests exercise the container `Instructor` impls (`Box<T>`,
//! `Vec<T>`, `Option<T>`, `HashMap<String, V>`) and the derive-generated
//! children-then-parent validation order. They complement
//! `nested_validation_tests.rs` (direct/Option/Vec recursion) by covering
//! `Box`, string-keyed maps, combination containers, ordering, error-message
//! identity, first-failure short-circuit, `None` deep short-circuit, and the
//! primitive `Probe` fallback path.
//!
//! Maps the "validation" rows of the coverage-gap report to real assertions
//! grounded in `src/model/instructor.rs` (the container impls run values in
//! `self.values()` order, which is non-deterministic, so map tests assert
//! err/ok only — never which value failed) and `rstructor_derive/src/lib.rs`
//! (struct field probes run in declaration order, before the container's own
//! `#[llm(validate)]` function).

use std::collections::HashMap;

use rstructor::{Instructor, RStructorError, Result};
use serde::{Deserialize, Serialize};

/// Leaf type with a custom validator: `value` must be >= 0.
#[derive(Instructor, Serialize, Deserialize, Debug, Clone)]
#[llm(validate = "validate_child")]
struct Child {
    value: i32,
}

fn validate_child(c: &Child) -> Result<()> {
    if c.value < 0 {
        return Err(RStructorError::ValidationError(format!(
            "child.value must be >= 0, got {}",
            c.value
        )));
    }
    Ok(())
}

fn child(v: i32) -> Child {
    Child { value: v }
}

/// Pull the inner string out of a `ValidationError`, failing the test loudly
/// otherwise so a mis-classified error can't masquerade as a pass.
fn validation_message(err: RStructorError) -> String {
    match err {
        RStructorError::ValidationError(msg) => msg,
        other => panic!("expected ValidationError, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Box<T>::validate recursion
// ---------------------------------------------------------------------------

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct BoxParent {
    boxed: Box<Child>,
}

#[test]
fn box_field_recurses_into_child() {
    let bad = BoxParent {
        boxed: Box::new(child(-1)),
    };
    let err = bad.validate().expect_err("boxed child -1 must fail");
    assert_eq!(validation_message(err), "child.value must be >= 0, got -1");

    let good = BoxParent {
        boxed: Box::new(child(1)),
    };
    assert!(good.validate().is_ok());
}

// ---------------------------------------------------------------------------
// HashMap<String, V>::validate recursion
// ---------------------------------------------------------------------------

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct MapParent {
    map: HashMap<String, Child>,
}

#[test]
fn hashmap_value_recursion_one_bad_errors() {
    // One bad value among several -> error. Iteration order over the map is
    // non-deterministic, so we only assert that it is the child message; we do
    // not assert which key surfaced.
    let mut map = HashMap::new();
    map.insert("a".to_string(), child(0));
    map.insert("b".to_string(), child(-3));
    map.insert("c".to_string(), child(5));
    let err = MapParent { map }
        .validate()
        .expect_err("map with a -3 value must fail");
    assert_eq!(validation_message(err), "child.value must be >= 0, got -3");
}

#[test]
fn hashmap_value_recursion_all_good_ok() {
    let mut map = HashMap::new();
    map.insert("a".to_string(), child(0));
    map.insert("b".to_string(), child(7));
    assert!(MapParent { map }.validate().is_ok());
}

#[test]
fn hashmap_value_recursion_empty_ok() {
    let map: HashMap<String, Child> = HashMap::new();
    assert!(MapParent { map }.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Combination containers: Vec<Option<T>>, Option<Vec<T>>, HashMap<String,Vec<T>>
// ---------------------------------------------------------------------------

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct ComboParent {
    vec_opt: Vec<Option<Child>>,
    opt_vec: Option<Vec<Child>>,
    map_vec: HashMap<String, Vec<Child>>,
}

fn empty_combo() -> ComboParent {
    ComboParent {
        vec_opt: vec![],
        opt_vec: None,
        map_vec: HashMap::new(),
    }
}

#[test]
fn vec_of_option_reaches_innermost_validator() {
    // [Some(1), None, Some(-2)] -> the innermost Child validator is reached
    // through Vec -> Option.
    let mut p = empty_combo();
    p.vec_opt = vec![Some(child(1)), None, Some(child(-2))];
    let err = p
        .validate()
        .expect_err("Vec<Option<Child>> with -2 must fail");
    assert_eq!(validation_message(err), "child.value must be >= 0, got -2");

    // All-good (with a None hole) -> ok.
    let mut ok = empty_combo();
    ok.vec_opt = vec![Some(child(1)), None, Some(child(3))];
    assert!(ok.validate().is_ok());
}

#[test]
fn option_of_vec_recurses() {
    let mut bad = empty_combo();
    bad.opt_vec = Some(vec![child(-1)]);
    assert!(bad.validate().is_err());

    let mut empty = empty_combo();
    empty.opt_vec = Some(vec![]);
    assert!(empty.validate().is_ok());

    let mut none = empty_combo();
    none.opt_vec = None;
    assert!(none.validate().is_ok());
}

#[test]
fn map_of_vec_recurses() {
    let mut bad = empty_combo();
    let mut m = HashMap::new();
    m.insert("k".to_string(), vec![child(2), child(-1)]);
    bad.map_vec = m;
    assert!(bad.validate().is_err());

    let mut good = empty_combo();
    let mut m = HashMap::new();
    m.insert("k".to_string(), vec![child(2), child(4)]);
    good.map_vec = m;
    assert!(good.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Children-then-parent ordering: a #[llm(validate)] parent whose child also
// fails should surface the CHILD message first, because the derive runs field
// probes (children) before the container's own validator (parent).
// ---------------------------------------------------------------------------

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(validate = "validate_order_parent")]
struct OrderParent {
    child: Child,
    name: String,
}

fn validate_order_parent(p: &OrderParent) -> Result<()> {
    if p.name == "bad" {
        return Err(RStructorError::ValidationError(
            "parent name must not be 'bad'".to_string(),
        ));
    }
    Ok(())
}

#[test]
fn child_failure_surfaces_before_parent_failure() {
    // Both child and parent are invalid: the CHILD message wins because field
    // probes run first.
    let p = OrderParent {
        child: child(-1),
        name: "bad".to_string(),
    };
    let err = p.validate().expect_err("invalid child must fail");
    assert_eq!(validation_message(err), "child.value must be >= 0, got -1");
}

#[test]
fn parent_failure_surfaces_when_children_valid() {
    // Child is valid, parent invalid -> the PARENT message is reached.
    let p = OrderParent {
        child: child(1),
        name: "bad".to_string(),
    };
    let err = p.validate().expect_err("invalid parent must fail");
    assert_eq!(validation_message(err), "parent name must not be 'bad'");
}

#[test]
fn order_parent_all_valid_ok() {
    let p = OrderParent {
        child: child(1),
        name: "ok".to_string(),
    };
    assert!(p.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Nested error-message identity through recursion: the message that surfaces is
// the exact, unwrapped child message (no enclosing context is prepended).
// ---------------------------------------------------------------------------

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct IdentityParent {
    direct: Child,
    maybe: Option<Child>,
    list: Vec<Child>,
}

#[test]
fn nested_error_message_is_unmodified_child_message() {
    let p = IdentityParent {
        direct: child(0),
        maybe: None,
        list: vec![child(1), child(2), child(-9)],
    };
    let err = p.validate().expect_err("a -9 in the list must fail");
    assert_eq!(validation_message(err), "child.value must be >= 0, got -9");
}

// ---------------------------------------------------------------------------
// Vec first-failure short-circuit: `Vec::validate` returns on the first failing
// element, so [-1, -2] reports -1 and never reaches -2.
// ---------------------------------------------------------------------------

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct ListParent {
    list: Vec<Child>,
}

#[test]
fn vec_short_circuits_on_first_failure() {
    let p = ListParent {
        list: vec![child(-1), child(-2)],
    };
    let msg = validation_message(p.validate().expect_err("first element must fail"));
    assert!(
        msg.contains("got -1"),
        "expected first failure (-1), got message: {msg}"
    );
    assert!(
        !msg.contains("got -2"),
        "second element must not be reached, got message: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Option::None deep short-circuit: `Option<Vec<Child>>` -> None is ok, while
// Some([bad]) errors. None must never reach the inner validator.
// ---------------------------------------------------------------------------

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct OptVecParent {
    items: Option<Vec<Child>>,
}

#[test]
fn option_none_short_circuits_inner_validation() {
    assert!(
        OptVecParent { items: None }.validate().is_ok(),
        "None must short-circuit and pass"
    );
    assert!(
        OptVecParent {
            items: Some(vec![child(-1)])
        }
        .validate()
        .is_err(),
        "Some([Child{{-1}}]) must reach the inner validator and fail"
    );
}

// ---------------------------------------------------------------------------
// Primitive Probe fallback: a struct whose only field is a primitive container
// (`Vec<String>`) has no Instructor validator on that field. The derive emits a
// Probe for it which resolves to the no-op `ProbeFallback`, so `validate()` is
// ok and the type compiles.
// ---------------------------------------------------------------------------

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Prim {
    label: String,
    count: u32,
    tags: Vec<String>,
}

#[test]
fn primitive_probe_fallback_is_noop() {
    let p = Prim {
        label: "x".to_string(),
        count: 3,
        tags: vec!["a".to_string(), "b".to_string()],
    };
    assert!(p.validate().is_ok());
}

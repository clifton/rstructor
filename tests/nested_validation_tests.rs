//! Nested validation: a derived `validate` recurses into nested `Instructor`
//! fields (directly, and through `Option`/`Vec`), then runs the container's own
//! validator. Before this behavior, only the top-level type's validator ran.

use rstructor::{Instructor, RStructorError, Result};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
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

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Parent {
    // primitive field: validation is a no-op, must not interfere
    name: String,
    // direct nested Instructor field
    child: Child,
    // nested through Option
    maybe_child: Option<Child>,
    // nested through Vec
    children: Vec<Child>,
}

fn parent(child: i32, maybe: Option<i32>, vec: Vec<i32>) -> Parent {
    Parent {
        name: "p".to_string(),
        child: Child { value: child },
        maybe_child: maybe.map(|v| Child { value: v }),
        children: vec.into_iter().map(|v| Child { value: v }).collect(),
    }
}

#[test]
fn direct_nested_field_is_validated() {
    assert!(parent(-1, None, vec![]).validate().is_err());
}

#[test]
fn nested_option_is_validated() {
    assert!(parent(0, Some(-5), vec![]).validate().is_err());
    // None must not error.
    assert!(parent(0, None, vec![]).validate().is_ok());
}

#[test]
fn nested_vec_elements_are_validated() {
    assert!(parent(0, None, vec![1, 2, -9]).validate().is_err());
    assert!(parent(0, None, vec![1, 2, 3]).validate().is_ok());
}

#[test]
fn all_valid_passes() {
    assert!(parent(0, Some(2), vec![3, 4]).validate().is_ok());
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
enum Wrapper {
    Tuple(Child),
    Struct { inner: Child },
    Empty,
}

#[test]
fn enum_variant_fields_are_validated() {
    assert!(Wrapper::Tuple(Child { value: -1 }).validate().is_err());
    assert!(
        Wrapper::Struct {
            inner: Child { value: -1 }
        }
        .validate()
        .is_err()
    );
    assert!(Wrapper::Tuple(Child { value: 1 }).validate().is_ok());
    assert!(Wrapper::Empty.validate().is_ok());
}

/// Deeply nested: Parent inside a Vec inside another struct still validates.
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct GrandParent {
    parents: Vec<Parent>,
}

#[test]
fn deeply_nested_is_validated() {
    let gp = GrandParent {
        parents: vec![parent(0, None, vec![]), parent(0, None, vec![-1])],
    };
    assert!(gp.validate().is_err());
}

mod builder;
mod custom_type;
mod primitives;
pub use builder::SchemaBuilder;
pub use custom_type::CustomTypeSchema;

use crate::error::Result;
use serde_json::Value;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// Helper function to call a struct's validate method if it exists
/// This is used by the derive macro to prevent dead code warnings on struct validate methods
pub fn call_validate_if_exists<T>(_obj: &T) -> Result<()> {
    // This function is intentionally a no-op in the base implementation
    // The derive macro will generate specialized versions that call the actual validate method
    // for types that have one
    Ok(())
}

/// Schema is a representation of a JSON Schema that describes the structure
/// an LLM should return.
///
/// The Schema struct wraps a JSON object that follows the JSON Schema specification.
/// It provides methods to access and manipulate the schema.
///
/// # Examples
///
/// Creating a schema manually:
///
/// ```
/// use rstructor::Schema;
/// use serde_json::json;
///
/// // Create a schema for a person with name and age
/// let schema = Schema::new(json!({
///     "type": "object",
///     "title": "Person",
///     "properties": {
///         "name": {
///             "type": "string",
///             "description": "Person's name"
///         },
///         "age": {
///             "type": "integer",
///             "description": "Person's age"
///         }
///     },
///     "required": ["name", "age"]
/// }));
///
/// // Convert to JSON or string
/// let json = schema.to_json();
/// assert_eq!(json["title"], "Person");
///
/// let schema_str = schema.to_string();
/// assert!(schema_str.contains("Person"));
/// ```
///
/// Using the builder:
///
/// ```
/// use rstructor::Schema;
/// use serde_json::json;
///
/// // Create a schema using the builder
/// let schema = Schema::builder()
///     .title("Person")
///     .property("name", json!({"type": "string", "description": "Person's name"}), true)
///     .property("age", json!({"type": "integer", "description": "Person's age"}), true)
///     .build();
///
/// let json = schema.to_json();
/// assert_eq!(json["title"], "Person");
/// ```
#[derive(Debug, Clone)]
pub struct Schema {
    pub schema: Value,
}

impl Schema {
    pub fn new(schema: Value) -> Self {
        Self { schema }
    }

    /// Return a reference to the raw unenhanced schema
    ///
    /// This method exists for backward compatibility with code expecting a reference.
    /// Most internal code should use to_enhanced_json() instead.
    pub fn original_schema(&self) -> &Value {
        &self.schema
    }

    /// Get the JSON representation of this schema
    ///
    /// Returns the schema as-is without enhancement to prevent stack overflow
    /// with complex nested structures. The derive macro should generate complete
    /// schemas that don't need runtime enhancement.
    pub fn to_json(&self) -> Value {
        // Return schema directly without enhancement to prevent stack overflow
        // The derive macro should generate complete schemas
        self.schema.clone()
    }

    // Format the schema as a pretty-printed JSON string
    pub fn to_pretty_json(&self) -> String {
        // Get the schema with array enhancements
        let schema_json = self.to_json();
        // CRITICAL: Use serde_json directly to avoid recursion - never call self.schema.to_string()
        // which would use Display impl and cause infinite recursion
        serde_json::to_string_pretty(&schema_json).unwrap_or_else(|_| {
            serde_json::to_string_pretty(&self.schema).unwrap_or_else(|_| "{}".to_string())
        })
    }

    /// Create a schema builder for an object type
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::object()
    }
}

// Display implementation for Schema
// NOTE: This can cause stack overflow with very complex schemas.
// Prefer using serde_json::to_string_pretty(&schema.to_json()) directly
impl Display for Schema {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        // Use serde_json directly to avoid any potential recursion
        let json = self.to_json();
        let json_str = serde_json::to_string_pretty(&json).unwrap_or_else(|_| "{}".to_string());
        write!(f, "{}", json_str)
    }
}

/// SchemaType trait defines a type that can be converted to a JSON Schema.
///
/// This trait is implemented for types that can generate a JSON Schema representation
/// of themselves. It's typically implemented by the derive macro for structs and enums.
///
/// # Examples
///
/// Manual implementation for a custom type:
///
/// ```
/// use rstructor::{Schema, SchemaType};
/// use serde_json::json;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Serialize, Deserialize)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// // Manually implement SchemaType for Person
/// impl SchemaType for Person {
///     fn schema() -> Schema {
///         Schema::new(json!({
///             "type": "object",
///             "title": "Person",
///             "properties": {
///                 "name": {
///                     "type": "string"
///                 },
///                 "age": {
///                     "type": "integer"
///                 }
///             },
///             "required": ["name", "age"]
///         }))
///     }
///
///     fn schema_name() -> Option<String> {
///         Some("Person".to_string())
///     }
/// }
///
/// // Use the schema
/// let schema = Person::schema();
/// let json = schema.to_json();
/// assert_eq!(json["title"], "Person");
/// assert_eq!(Person::schema_name(), Some("Person".to_string()));
/// ```
///
/// With the derive macro (typically how you'd use it):
///
/// ```no_run
/// use rstructor::Instructor;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Instructor, Serialize, Deserialize)]
/// struct Person {
///     #[llm(description = "Person's name")]
///     name: String,
///
///     #[llm(description = "Person's age")]
///     age: u32,
/// }
///
/// // SchemaType is implemented by the Instructor derive macro
/// // (This would work in real code, but doctest doesn't have access to the macro)
/// // let schema = Person::schema();
/// // let json = schema.to_json();
/// // assert_eq!(json["properties"]["name"]["description"], "Person's name");
/// ```
pub trait SchemaType {
    /// Generate a JSON Schema representation of this type
    fn schema() -> Schema;

    /// Build this type's schema inside an existing derived-schema graph.
    ///
    /// This hook lets derive-generated implementations share recursion state
    /// while composing nested schemas. Manual implementations may rely on the
    /// default, which preserves their existing [`SchemaType::schema`] behavior.
    #[doc(hidden)]
    fn schema_in(context: &mut __private::SchemaBuildContext) -> Value {
        context.import_schema(Self::schema().to_json())
    }

    /// Optional name for the schema
    ///
    /// This method returns an optional name for the schema. It's used by the LLM clients
    /// to reference the schema in their requests.
    fn schema_name() -> Option<String> {
        None
    }
}

/// Internal helpers used by `#[derive(Instructor)]`. Not part of the public API
/// and exempt from semver guarantees.
#[doc(hidden)]
pub mod __private {
    use super::SchemaType;
    use serde_json::Value;
    use std::any::TypeId;
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::marker::PhantomData;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    enum GraphTypeId {
        Static(TypeId),
        Named(String),
    }

    #[derive(Clone)]
    struct ActiveType {
        type_id: GraphTypeId,
        identity: String,
        preferred_key: String,
    }

    /// Per-call state for composing derive-generated schemas as a type graph.
    ///
    /// The context is owned by one root `SchemaType::schema()` call. It is
    /// deliberately neither global nor thread-local, so concurrent and nested
    /// schema builds cannot contaminate one another.
    #[doc(hidden)]
    #[derive(Default)]
    pub struct SchemaBuildContext {
        active: Vec<ActiveType>,
        recursive: HashSet<GraphTypeId>,
        definition_keys: HashMap<GraphTypeId, String>,
        key_owners: HashMap<String, GraphTypeId>,
        manual_keys: HashSet<String>,
        definitions: BTreeMap<String, Value>,
        completed: HashMap<GraphTypeId, Value>,
    }

    impl SchemaBuildContext {
        /// Create an empty schema-build context.
        pub fn new() -> Self {
            Self::default()
        }

        /// Build one concrete type inside this schema graph.
        ///
        /// `preferred_key` is the derive-visible short type name. Direct,
        /// non-generic self recursion keeps that familiar key. Multi-type and
        /// generic cycles use the fully-qualified concrete Rust type identity
        /// to avoid collisions.
        pub fn schema_for<T, F>(&mut self, preferred_key: &str, build: F) -> Value
        where
            T: ?Sized + 'static,
            F: FnOnce(&mut Self) -> Value,
        {
            let identity = std::any::type_name::<T>().to_string();
            self.schema_for_key(
                GraphTypeId::Static(TypeId::of::<T>()),
                identity,
                preferred_key,
                build,
            )
        }

        /// Build a lifetime-parameterized type inside this schema graph.
        ///
        /// `TypeId` is unavailable for non-`'static` types. Rust lifetimes do
        /// not affect Serde's wire representation, so these types use their
        /// concrete diagnostic name for per-call recursion bookkeeping.
        pub fn schema_for_named<T, F>(&mut self, preferred_key: &str, build: F) -> Value
        where
            T: ?Sized,
            F: FnOnce(&mut Self) -> Value,
        {
            let identity = std::any::type_name::<T>().to_string();
            self.schema_for_key(
                GraphTypeId::Named(identity.clone()),
                identity,
                preferred_key,
                build,
            )
        }

        fn schema_for_key<F>(
            &mut self,
            type_id: GraphTypeId,
            identity: String,
            preferred_key: &str,
            build: F,
        ) -> Value
        where
            F: FnOnce(&mut Self) -> Value,
        {
            if let Some(key) = self.definition_keys.get(&type_id) {
                return schema_reference(key);
            }

            if let Some(index) = self
                .active
                .iter()
                .position(|active| active.type_id == type_id)
            {
                let cycle = self.active[index..].to_vec();
                let use_qualified_keys = cycle.len() > 1;
                for active in cycle {
                    self.mark_recursive(active, use_qualified_keys);
                }
                let key = self
                    .definition_keys
                    .get(&type_id)
                    .expect("active recursive type must have a definition key");
                return schema_reference(key);
            }

            if let Some(schema) = self.completed.get(&type_id) {
                return schema.clone();
            }

            self.active.push(ActiveType {
                type_id: type_id.clone(),
                identity,
                preferred_key: preferred_key.to_string(),
            });
            let schema = build(self);
            let popped = self
                .active
                .pop()
                .expect("schema build stack must contain the current type");
            debug_assert_eq!(popped.type_id, type_id);

            if self.recursive.contains(&type_id) {
                let key = self
                    .definition_keys
                    .get(&type_id)
                    .expect("recursive type must have a definition key")
                    .clone();
                self.definitions.entry(key.clone()).or_insert(schema);
                schema_reference(&key)
            } else {
                self.completed.insert(type_id, schema.clone());
                schema
            }
        }

        /// Import a complete schema document produced by a manual
        /// [`SchemaType`] implementation.
        ///
        /// Local definitions are hoisted into this context and local references
        /// are rewritten when a definition name collides. This keeps references
        /// document-root relative after the manual schema is embedded in a
        /// derive-generated parent.
        pub fn import_schema(&mut self, mut schema: Value) -> Value {
            let needs_embedded_root = contains_embedded_root_reference(&schema);
            let preferred_root_key = schema
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("ManualSchema")
                .to_string();
            let local_definition_names = schema
                .get("$defs")
                .or_else(|| schema.get("definitions"))
                .and_then(Value::as_object)
                .map(|definitions| definitions.keys().cloned().collect::<HashSet<_>>())
                .unwrap_or_default();
            let root_key = needs_embedded_root.then(|| {
                self.manual_definition_key_avoiding(&preferred_root_key, &local_definition_names)
            });
            if let Some(root_key) = &root_key {
                rewrite_embedded_root_refs(&mut schema, root_key);
            }
            self.import_schema_scopes(&mut schema);
            let Some(root_key) = root_key else {
                return schema;
            };

            self.definitions.insert(root_key.clone(), schema);
            schema_reference(&root_key)
        }

        /// Merge the fields from an internally tagged newtype variant's inner
        /// schema into the variant object.
        ///
        /// Active recursive references cannot be resolved until the full type
        /// graph has been built. Those merges are recorded as private markers
        /// and completed by [`SchemaBuildContext::finish`].
        pub fn flatten_into_object(&self, mut outer: Value, inner: Value) -> Value {
            if let Some(inner) = self.resolve(&inner) {
                merge_flattened_object(&mut outer, &inner);
            } else if let Value::Object(outer) = &mut outer {
                outer.insert(DEFERRED_FLATTEN_KEY.to_string(), inner);
            }
            outer
        }

        /// Attach all definitions discovered while building `root`.
        pub fn finish(mut self, mut root: Value) -> Value {
            let definitions_snapshot = self.definitions.clone();
            resolve_deferred_flatten(&mut root, &definitions_snapshot, &mut Vec::new());
            for definition in self.definitions.values_mut() {
                resolve_deferred_flatten(definition, &definitions_snapshot, &mut Vec::new());
            }

            if self.definitions.is_empty() {
                return root;
            }

            let definitions = self
                .definitions
                .into_iter()
                .collect::<serde_json::Map<String, Value>>();

            match root {
                Value::Object(mut root) => {
                    if let Some(Value::Object(existing)) = root.get_mut("$defs") {
                        for (key, schema) in definitions {
                            existing.entry(key).or_insert(schema);
                        }
                    } else {
                        root.insert("$defs".to_string(), Value::Object(definitions));
                    }
                    Value::Object(root)
                }
                root => serde_json::json!({
                    "$defs": definitions,
                    "allOf": [root]
                }),
            }
        }

        /// Resolve a direct local `$ref` to a definition already completed in
        /// this context. Non-reference schemas are returned unchanged.
        pub fn resolve(&self, schema: &Value) -> Option<Value> {
            let Some(reference) = schema.get("$ref").and_then(Value::as_str) else {
                return Some(schema.clone());
            };
            let pointer_key = reference.strip_prefix("#/$defs/")?;
            let key = pointer_key.replace("~1", "/").replace("~0", "~");
            self.definitions.get(&key).cloned()
        }

        fn mark_recursive(&mut self, active: ActiveType, use_qualified_key: bool) {
            self.recursive.insert(active.type_id.clone());
            if self.definition_keys.contains_key(&active.type_id) {
                return;
            }

            let prefer_qualified = use_qualified_key || active.identity.contains('<');
            let mut key = if prefer_qualified {
                qualified_definition_key(&active)
            } else {
                active.preferred_key.clone()
            };

            if self.manual_keys.contains(&key)
                || self
                    .key_owners
                    .get(&key)
                    .is_some_and(|owner| owner != &active.type_id)
            {
                key = qualified_definition_key(&active);
            }

            key = self.unique_derived_key(key, &active.type_id);
            self.key_owners.insert(key.clone(), active.type_id.clone());
            self.definition_keys.insert(active.type_id, key);
        }

        fn unique_derived_key(&self, candidate: String, type_id: &GraphTypeId) -> String {
            if !self.manual_keys.contains(&candidate)
                && self
                    .key_owners
                    .get(&candidate)
                    .is_none_or(|owner| owner == type_id)
            {
                return candidate;
            }

            for suffix in 2usize.. {
                let candidate = format!("{candidate}{suffix}");
                if !self.manual_keys.contains(&candidate)
                    && !self.key_owners.contains_key(&candidate)
                {
                    return candidate;
                }
            }
            unreachable!()
        }

        fn manual_definition_key(&mut self, preferred: &str) -> String {
            self.manual_definition_key_avoiding(preferred, &HashSet::new())
        }

        fn manual_definition_key_avoiding(
            &mut self,
            preferred: &str,
            reserved: &HashSet<String>,
        ) -> String {
            let mut key = preferred.to_string();
            for suffix in 2usize.. {
                if !self.definitions.contains_key(&key)
                    && !self.key_owners.contains_key(&key)
                    && !self.manual_keys.contains(&key)
                    && !reserved.contains(&key)
                {
                    self.manual_keys.insert(key.clone());
                    return key;
                }
                key = format!("{preferred}{suffix}");
            }
            unreachable!()
        }

        fn import_schema_scopes(&mut self, schema: &mut Value) {
            match schema {
                Value::Object(object) => {
                    let definitions = object
                        .remove("$defs")
                        .or_else(|| object.remove("definitions"))
                        .and_then(|definitions| definitions.as_object().cloned());

                    if let Some(definitions) = definitions {
                        let mut renamed = BTreeMap::new();
                        for name in definitions.keys() {
                            let key = self.manual_definition_key(name);
                            renamed.insert(name.clone(), key);
                        }

                        rewrite_definition_refs_in_scope(schema, &renamed, true);
                        for (name, mut definition) in definitions {
                            rewrite_definition_refs_in_scope(&mut definition, &renamed, true);
                            self.import_schema_scopes(&mut definition);
                            let key = renamed
                                .get(&name)
                                .expect("every imported definition must have a key")
                                .clone();
                            self.definitions.entry(key).or_insert(definition);
                        }
                    }

                    if let Value::Object(object) = schema {
                        for child in object.values_mut() {
                            self.import_schema_scopes(child);
                        }
                    }
                }
                Value::Array(array) => {
                    for child in array {
                        self.import_schema_scopes(child);
                    }
                }
                _ => {}
            }
        }
    }

    fn schema_reference(key: &str) -> Value {
        let pointer_key = key.replace('~', "~0").replace('/', "~1");
        serde_json::json!({ "$ref": format!("#/$defs/{pointer_key}") })
    }

    fn qualified_definition_key(active: &ActiveType) -> String {
        let mut encoded_identity = String::with_capacity(active.identity.len());
        for byte in active.identity.bytes() {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.') {
                encoded_identity.push(char::from(byte));
            } else {
                use std::fmt::Write as _;
                write!(&mut encoded_identity, "_{byte:02x}")
                    .expect("writing to a String cannot fail");
            }
        }
        format!("{}__{encoded_identity}", active.preferred_key)
    }

    fn contains_embedded_root_reference(value: &Value) -> bool {
        match value {
            Value::Object(object) => {
                let has_root_reference =
                    object
                        .get("$ref")
                        .and_then(Value::as_str)
                        .is_some_and(|reference| {
                            reference == "#"
                                || (reference.starts_with("#/")
                                    && !reference.starts_with("#/$defs/")
                                    && !reference.starts_with("#/definitions/"))
                        });
                has_root_reference || object.values().any(contains_embedded_root_reference)
            }
            Value::Array(array) => array.iter().any(contains_embedded_root_reference),
            _ => false,
        }
    }

    fn rewrite_embedded_root_refs(value: &mut Value, root_key: &str) {
        match value {
            Value::Object(object) => {
                if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
                    let suffix = if reference == "#" {
                        Some("")
                    } else if reference.starts_with("#/")
                        && !reference.starts_with("#/$defs/")
                        && !reference.starts_with("#/definitions/")
                    {
                        reference.strip_prefix('#')
                    } else {
                        None
                    };
                    if let Some(suffix) = suffix {
                        let encoded_key = root_key.replace('~', "~0").replace('/', "~1");
                        object.insert(
                            "$ref".to_string(),
                            Value::String(format!("#/$defs/{encoded_key}{suffix}")),
                        );
                    }
                }
                for child in object.values_mut() {
                    rewrite_embedded_root_refs(child, root_key);
                }
            }
            Value::Array(array) => {
                for child in array {
                    rewrite_embedded_root_refs(child, root_key);
                }
            }
            _ => {}
        }
    }

    fn rewrite_definition_refs_in_scope(
        value: &mut Value,
        renamed: &BTreeMap<String, String>,
        is_scope_root: bool,
    ) {
        match value {
            Value::Object(object) => {
                if !is_scope_root
                    && (object.contains_key("$defs") || object.contains_key("definitions"))
                {
                    return;
                }
                if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
                    let rewritten = rewrite_definition_ref(reference, renamed);
                    if let Some(rewritten) = rewritten {
                        object.insert("$ref".to_string(), Value::String(rewritten));
                    }
                }
                for child in object.values_mut() {
                    rewrite_definition_refs_in_scope(child, renamed, false);
                }
            }
            Value::Array(array) => {
                for child in array {
                    rewrite_definition_refs_in_scope(child, renamed, false);
                }
            }
            _ => {}
        }
    }

    fn rewrite_definition_ref(
        reference: &str,
        renamed: &BTreeMap<String, String>,
    ) -> Option<String> {
        let (prefix, pointer) = if let Some(pointer) = reference.strip_prefix("#/$defs/") {
            ("#/$defs/", pointer)
        } else {
            ("#/$defs/", reference.strip_prefix("#/definitions/")?)
        };
        let (encoded_key, suffix) = pointer
            .split_once('/')
            .map_or((pointer, ""), |(key, suffix)| (key, suffix));
        let key = encoded_key.replace("~1", "/").replace("~0", "~");
        let renamed = renamed.get(&key)?;
        let encoded = renamed.replace('~', "~0").replace('/', "~1");
        Some(if suffix.is_empty() {
            format!("{prefix}{encoded}")
        } else {
            format!("{prefix}{encoded}/{suffix}")
        })
    }

    const DEFERRED_FLATTEN_KEY: &str = "__rstructor_deferred_flatten";

    fn resolve_deferred_flatten(
        value: &mut Value,
        definitions: &BTreeMap<String, Value>,
        resolving: &mut Vec<String>,
    ) {
        match value {
            Value::Object(object) => {
                if let Some(mut inner) = object.remove(DEFERRED_FLATTEN_KEY) {
                    if let Some(reference) = inner.get("$ref").and_then(Value::as_str) {
                        let pointer_key = reference
                            .strip_prefix("#/$defs/")
                            .unwrap_or_else(|| {
                                panic!(
                                    "deferred internally tagged flatten used unsupported reference {reference}"
                                )
                            });
                        let key = pointer_key.replace("~1", "/").replace("~0", "~");
                        assert!(
                            !resolving.contains(&key),
                            "internally tagged recursive newtypes form an unflattenable cycle at {reference}"
                        );
                        inner = definitions
                            .get(&key)
                            .unwrap_or_else(|| {
                                panic!(
                                    "deferred internally tagged flatten could not resolve {reference}"
                                )
                            })
                            .clone();
                        resolving.push(key);
                        resolve_deferred_flatten(&mut inner, definitions, resolving);
                        resolving.pop();
                    } else {
                        resolve_deferred_flatten(&mut inner, definitions, resolving);
                    }
                    merge_flattened_object(value, &inner);
                }

                if let Value::Object(object) = value {
                    for child in object.values_mut() {
                        resolve_deferred_flatten(child, definitions, resolving);
                    }
                }
            }
            Value::Array(array) => {
                for child in array {
                    resolve_deferred_flatten(child, definitions, resolving);
                }
            }
            _ => {}
        }
    }

    fn merge_flattened_object(outer: &mut Value, inner: &Value) {
        let Some(outer) = outer.as_object_mut() else {
            return;
        };
        let Some(inner) = inner.as_object() else {
            return;
        };

        if let Some(Value::Object(inner_properties)) = inner.get("properties")
            && let Some(Value::Object(outer_properties)) = outer.get_mut("properties")
        {
            for (key, schema) in inner_properties {
                outer_properties.insert(key.clone(), schema.clone());
            }
        }

        if let Some(Value::Array(inner_required)) = inner.get("required")
            && let Some(Value::Array(outer_required)) = outer.get_mut("required")
        {
            outer_required.extend(inner_required.iter().cloned());
        }
    }

    /// Autoref-specialization probe that lets generated code use a field type's
    /// own [`SchemaType`] schema **iff** the type implements it, and otherwise
    /// fall back to a name-based well-known schema (e.g. `{"type": "string",
    /// "format": "date"}` for `chrono::NaiveDate`) — without the derive macro
    /// having to know which crate a type named `Date` comes from.
    ///
    /// `#[derive(Instructor)]` emits
    /// `SchemaProbe::<FieldType>::new().rstructor_schema_in_or(context, fallback)` for
    /// fields whose type *name* matches a well-known library type (`Date`,
    /// `DateTime`, `NaiveDate`, `NaiveDateTime`, `Uuid`). When the field's
    /// type implements `SchemaType` (e.g. a user-defined `struct Date` that
    /// derives `Instructor`), the inherent method below is selected (inherent
    /// methods take priority over trait methods) and the type's real schema
    /// wins; otherwise method resolution falls back to
    /// [`SchemaProbeContextFallback`], which returns the sniffed fallback.
    pub struct SchemaProbe<T: ?Sized>(PhantomData<T>);

    impl<T: ?Sized> SchemaProbe<T> {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            SchemaProbe(PhantomData)
        }
    }

    impl<T: SchemaType + ?Sized> SchemaProbe<T> {
        /// Return the wrapped type's schema within an existing build context.
        pub fn rstructor_schema_in_or(
            &self,
            context: &mut SchemaBuildContext,
            _fallback: Value,
        ) -> Value {
            T::schema_in(context)
        }
    }

    /// Context-aware fallback for field types without [`SchemaType`].
    pub trait SchemaProbeContextFallback {
        fn rstructor_schema_in_or(
            &self,
            context: &mut SchemaBuildContext,
            fallback: Value,
        ) -> Value;
    }

    impl<T: ?Sized> SchemaProbeContextFallback for SchemaProbe<T> {
        fn rstructor_schema_in_or(
            &self,
            context: &mut SchemaBuildContext,
            fallback: Value,
        ) -> Value {
            let _ = context;
            fallback
        }
    }
}

#[cfg(test)]
mod tests;

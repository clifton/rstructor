#![allow(clippy::collapsible_if)]

use syn::{GenericArgument, PathArguments, Type};

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_is_option_type() {
        // Create test types
        let option_type: Type = parse_quote!(Option<String>);
        let string_type: Type = parse_quote!(String);
        let vec_type: Type = parse_quote!(Vec<u32>);

        // Check type detection
        assert!(is_option_type(&option_type));
        assert!(!is_option_type(&string_type));
        assert!(!is_option_type(&vec_type));
    }

    #[test]
    fn test_get_option_inner_type() {
        // Create an Option<String> type
        let option_type: Type = parse_quote!(Option<String>);

        // Get inner type
        let inner_type = get_option_inner_type(&option_type);

        // Should be a String
        if let Type::Path(type_path) = inner_type {
            assert_eq!(
                type_path.path.segments.first().unwrap().ident.to_string(),
                "String"
            );
        } else {
            panic!("Inner type is not a Path");
        }
    }

    #[test]
    fn test_get_type_category() {
        // Create test types
        let string_type: Type = parse_quote!(String);
        let int_type: Type = parse_quote!(i32);
        let float_type: Type = parse_quote!(f64);
        let bool_type: Type = parse_quote!(bool);
        let vec_type: Type = parse_quote!(Vec<String>);
        let map_type: Type = parse_quote!(HashMap<String, i32>);
        let custom_type: Type = parse_quote!(MyCustomType);

        // Check type categories
        assert!(matches!(
            get_type_category(&string_type),
            TypeCategory::String
        ));
        assert!(matches!(
            get_type_category(&int_type),
            TypeCategory::Integer
        ));
        assert!(matches!(
            get_type_category(&float_type),
            TypeCategory::Float
        ));
        assert!(matches!(
            get_type_category(&bool_type),
            TypeCategory::Boolean
        ));
        assert!(matches!(get_type_category(&vec_type), TypeCategory::Array));
        assert!(matches!(get_type_category(&map_type), TypeCategory::Object));
        assert!(matches!(
            get_type_category(&custom_type),
            TypeCategory::Object
        ));
    }

    #[test]
    fn test_get_schema_type_from_rust_type() {
        // Create test types
        let string_type: Type = parse_quote!(String);
        let int_type: Type = parse_quote!(i32);
        let float_type: Type = parse_quote!(f64);
        let bool_type: Type = parse_quote!(bool);
        let vec_type: Type = parse_quote!(Vec<String>);
        let map_type: Type = parse_quote!(HashMap<String, i32>);
        let option_type: Type = parse_quote!(Option<String>);

        // Check schema types
        assert_eq!(get_schema_type_from_rust_type(&string_type), "string");
        assert_eq!(get_schema_type_from_rust_type(&int_type), "integer");
        assert_eq!(get_schema_type_from_rust_type(&float_type), "number");
        assert_eq!(get_schema_type_from_rust_type(&bool_type), "boolean");
        assert_eq!(get_schema_type_from_rust_type(&vec_type), "array");
        assert_eq!(get_schema_type_from_rust_type(&map_type), "object");
        assert_eq!(get_schema_type_from_rust_type(&option_type), "string"); // Unwrapped
    }
}

/// Enum to categorize Rust types for schema generation
#[derive(Debug, Clone, Copy)]
pub enum TypeCategory {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
}

/// Determine if a type is an Option<T>
pub fn is_option_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.first()
    {
        return segment.ident == "Option";
    }
    false
}

/// Get the inner type of an Option<T>
pub fn get_option_inner_type(ty: &Type) -> &Type {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.first()
        && segment.ident == "Option"
        && let PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(GenericArgument::Type(inner_ty)) = args.args.first()
    {
        return inner_ty;
    }
    ty
}

/// Get type category from Rust type
pub fn get_type_category(ty: &Type) -> TypeCategory {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            let type_name = segment.ident.to_string();
            match type_name.as_str() {
                "String" | "str" | "char" => return TypeCategory::String,
                "bool" => return TypeCategory::Boolean,
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64"
                | "u128" | "usize" => return TypeCategory::Integer,
                "f32" | "f64" => return TypeCategory::Float,
                "Vec" | "Array" | "HashSet" | "BTreeSet" => return TypeCategory::Array,
                "HashMap" | "BTreeMap" => return TypeCategory::Object,
                _ => return TypeCategory::Object, // Default to object for custom types
            }
        }
    }
    TypeCategory::Object // Default
}

/// Get JSON Schema type from Rust type
pub fn get_schema_type_from_rust_type(ty: &Type) -> &'static str {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            let type_name = segment.ident.to_string();
            match type_name.as_str() {
                "String" | "str" | "char" => return "string",
                "bool" => return "boolean",
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64"
                | "u128" | "usize" => return "integer",
                "f32" | "f64" => return "number",
                "Vec" | "Array" | "HashSet" | "BTreeSet" => return "array",
                "HashMap" | "BTreeMap" => return "object",
                // Recognize common date types directly
                "DateTime" | "NaiveDateTime" | "NaiveDate" | "Date" | "Utc" | "Local" => {
                    return "string";
                }
                // Recognize UUID type
                "Uuid" | "uuid::Uuid" => return "string",
                "Option" => {
                    // For Option<T>, we need to look at the inner type
                    if let PathArguments::AngleBracketed(args) = &segment.arguments
                        && let Some(GenericArgument::Type(inner_ty)) = args.args.first()
                    {
                        return get_schema_type_from_rust_type(inner_ty);
                    }
                    return "null";
                }
                _ => return "object", // Default to object for custom types
            }
        }
    }
    "object" // Default
}

/// Get the inner type of an array type like Vec<T>
pub fn get_array_inner_type(ty: &Type) -> Option<&Type> {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.first()
    {
        let type_name = segment.ident.to_string();
        if matches!(type_name.as_str(), "Vec" | "Array" | "HashSet" | "BTreeSet") {
            if let PathArguments::AngleBracketed(args) = &segment.arguments {
                if let Some(GenericArgument::Type(inner_ty)) = args.args.first() {
                    return Some(inner_ty);
                }
            }
        }
    }
    None
}

/// Check if a type is an array type (Vec, Array, etc.)
pub fn is_array_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.first()
    {
        let type_name = segment.ident.to_string();
        return matches!(type_name.as_str(), "Vec" | "Array" | "HashSet" | "BTreeSet");
    }
    false
}

/// Check if a type is a HashMap or BTreeMap
pub fn is_map_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.first()
    {
        let type_name = segment.ident.to_string();
        return matches!(type_name.as_str(), "HashMap" | "BTreeMap");
    }
    false
}

/// Get the key and value types from a HashMap<K, V> or BTreeMap<K, V>
pub fn get_map_types(ty: &Type) -> Option<(&Type, &Type)> {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.first()
    {
        let type_name = segment.ident.to_string();
        if matches!(type_name.as_str(), "HashMap" | "BTreeMap") {
            if let PathArguments::AngleBracketed(args) = &segment.arguments {
                let mut args_iter = args.args.iter();
                if let (Some(GenericArgument::Type(key_ty)), Some(GenericArgument::Type(val_ty))) =
                    (args_iter.next(), args_iter.next())
                {
                    return Some((key_ty, val_ty));
                }
            }
        }
    }
    None
}

/// Check if a type is a Box<T>
pub fn is_box_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.first()
    {
        return segment.ident == "Box";
    }
    false
}

/// Get the inner type of a Box<T>
pub fn get_box_inner_type(ty: &Type) -> Option<&Type> {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.first()
        && segment.ident == "Box"
        && let PathArguments::AngleBracketed(args) = &segment.arguments
        && let Some(GenericArgument::Type(inner_ty)) = args.args.first()
    {
        return Some(inner_ty);
    }
    None
}

/// Check if a type is serde_json::Value
pub fn is_json_value_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        // Check for "Value" or "serde_json::Value"
        let path_str = type_path
            .path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect::<Vec<_>>()
            .join("::");
        return path_str == "Value" || path_str == "serde_json::Value";
    }
    false
}

/// Check if a type is a tuple (e.g., (i32, String))
pub fn is_tuple_type(ty: &Type) -> bool {
    matches!(ty, Type::Tuple(_))
}

/// Get the element types of a tuple
pub fn get_tuple_element_types(ty: &Type) -> Option<Vec<&Type>> {
    if let Type::Tuple(tuple) = ty {
        return Some(tuple.elems.iter().collect());
    }
    None
}

/// Get the final type name after unwrapping Option and Box
/// Returns the core type name for detecting self-references
pub fn get_core_type_name(ty: &Type) -> Option<String> {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            let type_name = segment.ident.to_string();
            match type_name.as_str() {
                "Option" | "Box" => {
                    // Unwrap and recurse
                    if let PathArguments::AngleBracketed(args) = &segment.arguments
                        && let Some(GenericArgument::Type(inner_ty)) = args.args.first()
                    {
                        return get_core_type_name(inner_ty);
                    }
                }
                "Vec" | "Array" | "HashSet" | "BTreeSet" => {
                    // For collections, get the inner type
                    if let PathArguments::AngleBracketed(args) = &segment.arguments
                        && let Some(GenericArgument::Type(inner_ty)) = args.args.first()
                    {
                        return get_core_type_name(inner_ty);
                    }
                }
                _ => return Some(type_name),
            }
        }
    }
    None
}

/// Check if a type is a self-reference (matches the given struct name)
/// Unwraps Option, Box, Vec, etc. to find the core type
pub fn is_self_reference(ty: &Type, struct_name: &str) -> bool {
    get_core_type_name(ty).is_some_and(|name| name == struct_name)
}

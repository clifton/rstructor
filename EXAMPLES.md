# RStructor Field Attribute Examples

This document shows how to properly use the various field attributes with different Rust types.

## Type-Specific Examples

### Strings

```rust
// String examples
#[llm(description = "The name of the person", example = "John Smith")]
name: String,

// Multiple examples using native array syntax (recommended)
#[llm(description = "The full name", examples = ["John Smith", "Jane Doe", "Alex Johnson"])]
full_name: String,

// You can also still use JSON array syntax for backward compatibility
#[llm(description = "The full name", example = "[\"John Smith\", \"Jane Doe\"]")]
legacy_name: String,
```

### Numbers

```rust
// Integer example (can use literal without quotes)
#[llm(description = "Age of the person", example = 30)]
age: u32,

// Float example (can use literal without quotes)
#[llm(description = "Height in meters", example = 1.75)]
height: f32,

// Float with native array syntax for multiple examples
#[llm(description = "Possible prices", examples = [45.99, 55.50, 32.99])]
price: f32,
```

### Booleans

```rust
// Boolean example (can use literal without quotes)
#[llm(description = "Whether the user is active", example = true)]
is_active: bool,
```

### Arrays/Vectors

```rust
// For arrays, you can now use native Rust array literals (recommended)
#[llm(description = "List of tags", example = ["important", "urgent", "follow-up"])]
tags: Vec<String>,

// For number arrays with native array syntax
#[llm(description = "List of scores", example = [90, 85, 76, 92])]
scores: Vec<u32>,

// You can also still use the JSON-like array string with single quotes for backward compatibility
#[llm(description = "List of categories", example = "['important', 'urgent', 'follow-up']")]
categories: Vec<String>,

// Mixed types are also supported in array literals
#[llm(description = "Mixed data", example = ["text", 123, true, 45.6])]
mixed_data: Vec<serde_json::Value>,
```

### Objects

```rust
// For complex objects, use JSON object syntax
#[llm(description = "The user's address", 
      example = "{\"street\": \"123 Main St\", \"city\": \"New York\", \"zip\": \"10001\"}")]
address: Address,
```

### Optional Fields

```rust
// Optional fields are automatically detected from the Option<T> type
// No need to add an 'optional' attribute
#[llm(description = "Optional middle name")]
middle_name: Option<String>,

// Example for optional field
#[llm(description = "Optional phone number", example = "+1-555-123-4567")]
phone: Option<String>,
```

## General Guidelines

1. For simple types like strings, numbers, and booleans, you can use literals directly
2. For arrays and objects, use a JSON string with the proper format
3. When providing a literal that doesn't match the field type, the macro will try to convert it or use it as a string
4. `Option<T>` types are automatically detected as optional fields
5. The generated schema respects the types of your Rust struct
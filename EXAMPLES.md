# RStructor Attribute Examples

This document shows how to properly use the various attributes with different Rust types.

## Container-Level Attributes

You can add attributes to the entire struct or enum:

```rust
// Add a description to a struct
#[derive(LLMModel, Serialize, Deserialize, Debug)]
#[llm(description = "Represents a person with their basic information")]
struct Person {
    // Fields...
}

// Add a description to an enum
#[derive(LLMModel, Serialize, Deserialize, Debug)]
#[llm(description = "Represents a person's role in an organization")]
enum Role {
    Employee,
    Manager,
    Director,
}
```

## Field-Level Attributes

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

### Enums

RStructor supports both simple enums and enums with associated data.

#### Simple Enums

```rust
#[derive(LLMModel, Serialize, Deserialize, Debug)]
#[llm(description = "Sentiment of the text analysis")]
enum Sentiment {
    #[llm(description = "The text is positive in tone")]
    Positive,

    #[llm(description = "The text is negative in tone")]
    Negative,

    #[llm(description = "The sentiment is neutral or mixed")]
    Neutral,
}
```

#### Enums with Simple Associated Data

```rust
#[derive(LLMModel, Serialize, Deserialize, Debug)]
enum UserStatus {
    #[llm(description = "The user is online")]
    Online,

    #[llm(description = "The user is offline")]
    Offline,

    #[llm(description = "The user is away with an optional message")]
    Away(String),

    #[llm(description = "The user is busy until a specific time")]
    Busy(u32),
}
```

#### Enums with Complex Associated Data

```rust
// Defining structs used in enum variants
#[derive(Serialize, Deserialize, Debug)]
struct Address {
    street: String,
    city: String,
    zip: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct PaymentCard {
    card_number: String,
    exp_date: String,
    cvv: String,
}

// Enum with various types of associated data
#[derive(LLMModel, Serialize, Deserialize, Debug)]
enum PaymentMethod {
    #[llm(description = "Payment made with a credit or debit card")]
    Card(PaymentCard),

    #[llm(description = "Payment made with PayPal")]
    PayPal(String),

    #[llm(description = "Payment will be made on delivery")]
    CashOnDelivery,

    #[llm(description = "Payment made via bank transfer with account details")]
    BankTransfer {
        account_number: String,
        routing_number: String,
        bank_name: String,
    },
}
```

## General Guidelines

1. For simple types like strings, numbers, and booleans, you can use literals directly
2. For arrays and objects, use a JSON string with the proper format
3. When providing a literal that doesn't match the field type, the macro will try to convert it or use it as a string
4. `Option<T>` types are automatically detected as optional fields
5. The generated schema respects the types of your Rust struct
6. Enums with associated data are represented using JSON Schema's `oneOf` pattern for better LLM understanding
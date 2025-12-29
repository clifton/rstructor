use rstructor::{Instructor, RStructorError, SchemaType};
use serde::{Deserialize, Serialize};

// Example that demonstrates how to use custom validation with rstructor

// Define a Product type with validation
#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(
    description = "A product in an inventory system",
    validate = "validate_product"
)]
struct Product {
    #[llm(description = "Product name", example = "Laptop Pro")]
    name: String,

    #[llm(description = "Product price in USD", example = 999.99)]
    price: f64,

    #[llm(description = "Quantity available in inventory", example = 42)]
    quantity: u32,

    #[llm(description = "Product categories", examples = ["Electronics", "Computers", "Office"])]
    categories: Vec<String>,
}

// Custom validation function referenced by #[llm(validate = "validate_product")]
fn validate_product(product: &Product) -> rstructor::Result<()> {
    // Price must be positive
    if product.price <= 0.0 {
        return Err(RStructorError::ValidationError(format!(
            "Product price must be positive, got {}",
            product.price
        )));
    }

    // Name can't be empty
    if product.name.trim().is_empty() {
        return Err(RStructorError::ValidationError(
            "Product name cannot be empty".to_string(),
        ));
    }

    // Must have at least one category
    if product.categories.is_empty() {
        return Err(RStructorError::ValidationError(
            "Product must have at least one category".to_string(),
        ));
    }

    // All validation passed
    Ok(())
}

fn main() {
    // Create a valid product
    let valid_product = Product {
        name: "Laptop Pro".to_string(),
        price: 1299.99,
        quantity: 10,
        categories: vec!["Electronics".to_string(), "Computers".to_string()],
    };

    // Validate the product - this now goes through the Instructor trait
    match <Product as Instructor>::validate(&valid_product) {
        Ok(_) => println!("Valid product: {:?}", valid_product),
        Err(e) => println!("Validation error: {}", e),
    }

    // Create an invalid product (negative price)
    let invalid_product = Product {
        name: "Broken Item".to_string(),
        price: -10.0, // Invalid price
        quantity: 5,
        categories: vec!["Misc".to_string()],
    };

    // Validation should fail
    match <Product as Instructor>::validate(&invalid_product) {
        Ok(_) => println!("Valid product: {:?}", invalid_product),
        Err(e) => println!("Validation error: {}", e),
    }

    // Create another invalid product (no categories)
    let invalid_product2 = Product {
        name: "Missing Categories".to_string(),
        price: 49.99,
        quantity: 100,
        categories: vec![], // Invalid - empty categories
    };

    // Validation should fail
    match <Product as Instructor>::validate(&invalid_product2) {
        Ok(_) => println!("Valid product: {:?}", invalid_product2),
        Err(e) => println!("Validation error: {}", e),
    }

    println!("\nJSON Schema for Product:");
    println!("{}", Product::schema());
}

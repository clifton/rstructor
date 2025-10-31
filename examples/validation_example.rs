use rstructor::{Instructor, RStructorError, SchemaType};
use serde::{Deserialize, Serialize};

// Example that demonstrates how to use custom validation with rstructor
// without getting dead code warnings for the validate method

// Define a Product type with validation
#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "A product in an inventory system")]
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

// Custom validation implementation - this method won't have dead code warnings
// because the derive macro handles properly linking it to the Instructor trait
impl Product {
    // Note: This method will NOT be warned as dead code
    fn validate(&self) -> rstructor::Result<()> {
        // Price must be positive
        if self.price <= 0.0 {
            return Err(RStructorError::ValidationError(format!(
                "Product price must be positive, got {}",
                self.price
            )));
        }

        // Name can't be empty
        if self.name.trim().is_empty() {
            return Err(RStructorError::ValidationError(
                "Product name cannot be empty".to_string(),
            ));
        }

        // Must have at least one category
        if self.categories.is_empty() {
            return Err(RStructorError::ValidationError(
                "Product must have at least one category".to_string(),
            ));
        }

        // All validation passed
        Ok(())
    }
}

fn main() {
    // Create a valid product
    let valid_product = Product {
        name: "Laptop Pro".to_string(),
        price: 1299.99,
        quantity: 10,
        categories: vec!["Electronics".to_string(), "Computers".to_string()],
    };

    // Validate the product
    match valid_product.validate() {
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
    match invalid_product.validate() {
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
    match invalid_product2.validate() {
        Ok(_) => println!("Valid product: {:?}", invalid_product2),
        Err(e) => println!("Validation error: {}", e),
    }

    println!("\nJSON Schema for Product:");
    println!("{}", Product::schema());
}

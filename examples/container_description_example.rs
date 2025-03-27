use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};

// Define a struct with a container-level description
#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "Represents a book in a library catalog")]
struct Book {
    #[llm(description = "Unique identifier for the book")]
    id: u32,

    #[llm(description = "Title of the book", example = "The Lord of the Rings")]
    title: String,

    #[llm(description = "Author of the book", example = "J.R.R. Tolkien")]
    author: String,

    #[llm(description = "Year the book was published", example = 1954)]
    publication_year: u16,

    #[llm(description = "Genres associated with the book", 
          example = ["Fantasy", "Adventure", "Epic"])]
    genres: Vec<String>,

    #[llm(description = "Brief summary of the book's plot")]
    summary: Option<String>,
}

// Define an enum with a container-level description
#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "Represents the status of a book in the library")]
enum BookStatus {
    Available,
    CheckedOut,
    OnHold,
    InRepair,
}

fn main() {
    // Generate the schema for the Book struct
    let book_schema = Book::schema();
    println!("Book Schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(&book_schema.to_json()).unwrap()
    );

    // Generate the schema for the BookStatus enum
    let status_schema = BookStatus::schema();
    println!("\nBookStatus Schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(&status_schema.to_json()).unwrap()
    );
}

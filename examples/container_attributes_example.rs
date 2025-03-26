use rstructor::{LLMModel, SchemaType};
use serde::{Deserialize, Serialize};

// Define a struct with container-level attributes
#[derive(LLMModel, Serialize, Deserialize, Debug)]
#[llm(description = "Represents a book in a library catalog",
      title = "LibraryBook",
      examples = [
        ::serde_json::json!({"id": 1, "title": "The Hobbit", "author": "J.R.R. Tolkien", "publication_year": 1937, "genres": ["Fantasy", "Adventure"]}),
        ::serde_json::json!({"id": 2, "title": "Dune", "author": "Frank Herbert", "publication_year": 1965, "genres": ["Science Fiction"]})
      ])]
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

// Define a struct with serde rename_all
#[derive(LLMModel, Serialize, Deserialize, Debug)]
#[llm(description = "Information about a book's publisher")]
#[serde(rename_all = "camelCase")]
struct Publisher {
    publisher_name: String,
    publication_date: String,
    isbn_number: String,
    page_count: u16,
    is_hardcover: bool,
}

// Define an enum with container-level attributes
#[derive(LLMModel, Serialize, Deserialize, Debug)]
#[llm(description = "Represents the status of a book in the library",
      examples = ["Available", "CheckedOut"])]
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
        serde_json::to_string_pretty(book_schema.to_json()).unwrap()
    );

    // Generate the schema for the Publisher struct (with serde rename_all)
    let publisher_schema = Publisher::schema();
    println!("\nPublisher Schema (with camelCase property names):");
    println!(
        "{}",
        serde_json::to_string_pretty(publisher_schema.to_json()).unwrap()
    );

    // Generate the schema for the BookStatus enum
    let status_schema = BookStatus::schema();
    println!("\nBookStatus Schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(status_schema.to_json()).unwrap()
    );
}

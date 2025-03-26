use rstructor::{LLMModel, SchemaType};
use serde::{Deserialize, Serialize};

// Simple enum
#[derive(LLMModel, Serialize, Deserialize, Debug)]
enum Sentiment {
    #[llm(description = "The text is positive in tone")]
    Positive,

    #[llm(description = "The text is negative in tone")]
    Negative,

    #[llm(description = "The sentiment is neutral or mixed")]
    Neutral,
}

// Struct with enum field
#[derive(LLMModel, Serialize, Deserialize, Debug)]
struct TextAnalysis {
    #[llm(description = "The input text that was analyzed")]
    text: String,

    #[llm(description = "The detected sentiment of the text")]
    sentiment: Sentiment,

    #[llm(description = "Confidence score from 0.0 to 1.0", example = "0.92")]
    confidence: f32,
}

fn main() {
    // Get the schema for the enum
    let sentiment_schema = Sentiment::schema();

    println!("Sentiment Enum Schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(sentiment_schema.to_json()).unwrap()
    );

    // Get the schema for the struct with enum field
    let analysis_schema = TextAnalysis::schema();

    println!("\nTextAnalysis Schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(analysis_schema.to_json()).unwrap()
    );

    // Sample instance
    let analysis = TextAnalysis {
        text: "I really enjoyed this movie!".to_string(),
        sentiment: Sentiment::Positive,
        confidence: 0.95,
    };

    println!("\nSample TextAnalysis:");
    println!("{:?}", analysis);
    println!("{}", serde_json::to_string_pretty(&analysis).unwrap());
}

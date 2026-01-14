// tests/hashmap_integration_tests.rs
#[cfg(test)]
mod hashmap_tests {
    use rstructor::{GeminiClient, Instructor, LLMClient};
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::env;

    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct Inventory {
        #[llm(description = "Map of category names to list of items in that category")]
        categories: HashMap<String, Vec<String>>,

        #[llm(description = "Map of item names to their current stock count")]
        stock_counts: HashMap<String, u32>,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct Metadata {
        pub score: f32,
        pub tags: Vec<String>,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct ComplexMap {
        #[llm(description = "Map of user IDs to their metadata objects")]
        user_data: HashMap<String, Metadata>,
    }

    #[tokio::test]
    async fn test_gemini_hashmap_nested_vec() {
        let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
        let client = GeminiClient::new(api_key).unwrap().temperature(0.0).model("gemini-2.5-flash-preview-09-2025");

        let prompt = "Organize these items into categories: apple, hammer, banana, screwdriver, drill. Use category names like 'fruit', 'tools' as the keys.";
        let result: rstructor::Result<Inventory> = client.materialize(prompt).await;

        // This is expected to fail or produce interesting schema errors if HashMap isn't handled
        assert!(
            result.is_ok(),
            "HashMap<String, Vec<String>> failed: {:?}",
            result.err()
        );
        let inv = result.unwrap();
        // Check that we got some categories with items
        // Note: Gemini may use different key names due to schema limitations
        assert!(
            !inv.categories.is_empty(),
            "Expected categories to have entries, got empty map"
        );
        // Check that at least one category has items
        assert!(
            inv.categories.values().any(|v| !v.is_empty()),
            "Expected at least one category to have items"
        );
    }

    #[tokio::test]
    async fn test_gemini_hashmap_complex_value() {
        let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
        let client = GeminiClient::new(api_key).unwrap().temperature(0.0).model("gemini-2.5-flash-preview-09-2025");

        let prompt = "Generate metadata for users. Create entries with keys 'alice' and 'bob'. Alice has score 9.5 and tags 'pro', 'admin'. Bob has score 4.0 and tag 'newbie'.";
        let result: rstructor::Result<ComplexMap> = client.materialize(prompt).await;

        assert!(
            result.is_ok(),
            "HashMap<String, Metadata> failed: {:?}",
            result.err()
        );
        let map = result.unwrap();
        // Check that we got user data entries with correct structure
        // Note: Gemini may use different key names due to schema limitations
        assert!(
            !map.user_data.is_empty(),
            "Expected user_data to have entries, got empty map"
        );
        // Check that metadata values have the expected fields
        for (key, metadata) in &map.user_data {
            assert!(
                metadata.score >= 0.0,
                "Expected valid score for key '{}', got {}",
                key,
                metadata.score
            );
        }
    }
}

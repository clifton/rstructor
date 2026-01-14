// tests/weird_types_tests.rs
#[cfg(test)]
mod weird_types_tests {
    use rstructor::{GeminiClient, Instructor, LLMClient};
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::env;

    #[derive(Instructor, Serialize, Deserialize, Debug, Hash, Eq, PartialEq)]
    #[serde(rename_all = "lowercase")]
    enum LogLevel {
        Info,
        Warn,
        Error,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct WeirdStruct {
        #[llm(description = "A tuple representing a 2D coordinate (x, y)")]
        point: (i32, i32),

        #[llm(description = "A more complex tuple: (ID, Name, IsActive)")]
        user_tuple: (u64, String, bool),

        #[llm(description = "Map using enum keys")]
        log_counts: HashMap<LogLevel, u32>,

        #[llm(description = "Ultimate fallback: raw JSON value")]
        raw_metadata: serde_json::Value,
    }

    #[tokio::test]
    async fn test_gemini_weird_types() {
        let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
        let client = GeminiClient::new(api_key).unwrap().temperature(0.0).model("gemini-2.5-flash-preview-09-2025");

        let prompt = "Generate a WeirdStruct where point is at 10,20. User is (1, 'Admin', true). Log counts: 5 info, 2 error. Raw metadata is any JSON object.";
        let result: rstructor::Result<WeirdStruct> = client.materialize(prompt).await;

        // Current implementation likely fails LogLevel as key and Tuple as items
        assert!(result.is_ok(), "Weird types failed: {:?}", result.err());
        let weird = result.unwrap();
        assert_eq!(weird.point.0, 10);
        assert_eq!(weird.user_tuple.1, "Admin");
        assert_eq!(*weird.log_counts.get(&LogLevel::Info).unwrap(), 5);
    }
}

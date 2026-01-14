// tests/complex_enum_integration_tests.rs
#[cfg(test)]
mod complex_enum_tests {
    use rstructor::{GeminiClient, Instructor, LLMClient};
    use serde::{Deserialize, Serialize};
    use std::env;

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    #[serde(tag = "status", content = "data")]
    enum TaskResult {
        #[llm(description = "Task completed successfully")]
        Success { output: String, tokens_used: u32 },
        #[llm(description = "Task failed with an error message")]
        Failure { error_code: i32, reason: String },
        #[llm(description = "Task is still in progress")]
        Pending,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug)]
    struct Workflow {
        pub name: String,
        pub steps: Vec<TaskResult>,
    }

    #[tokio::test]
    async fn test_gemini_adjacently_tagged_enum() {
        let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
        let client = GeminiClient::new(api_key)
            .unwrap()
            .temperature(0.0)
            .no_retries();

        let prompt = "Create a workflow with 3 steps: 1. Success with output='Calculated' and tokens_used=100, 2. A Pending step, 3. A Failure with error_code=500 and reason='Timeout'.";
        let result: rstructor::Result<Workflow> = client.materialize(prompt).await;

        // Gemini often struggles with 'oneOf' if titles or descriptions are misplaced
        assert!(
            result.is_ok(),
            "Adjacently tagged enum failed: {:?}",
            result.err()
        );
        let workflow = result.unwrap();
        assert_eq!(workflow.steps.len(), 3);
        assert!(matches!(workflow.steps[1], TaskResult::Pending));
    }
}

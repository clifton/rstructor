// tests/recursive_structures_tests.rs
#[cfg(test)]
mod recursive_tests {
    use rstructor::{GeminiClient, Instructor, LLMClient};
    use serde::{Deserialize, Serialize};
    use std::env;

    #[derive(Instructor, Serialize, Deserialize, Debug)]
    #[llm(description = "A node in a file system tree")]
    struct FileNode {
        name: String,
        #[llm(description = "If true, this is a directory and can have children")]
        is_dir: bool,
        #[llm(description = "Children nodes if this is a directory")]
        #[allow(clippy::vec_box)]
        children: Option<Vec<Box<FileNode>>>,
        #[llm(description = "Size in bytes if this is a file")]
        size: Option<u64>,
    }

    #[tokio::test]
    async fn test_gemini_recursive_schema() {
        let api_key = env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
        let client = GeminiClient::new(api_key)
            .unwrap()
            .temperature(0.0)
            .no_retries();

        let prompt = "Represent a small directory structure: A root folder 'src' containing a file 'lib.rs' (500 bytes) and a subfolder 'backend' which is empty.";
        let result: rstructor::Result<FileNode> = client.materialize(prompt).await;

        // Recursive schemas often fail if the generator doesn't handle $ref or creates infinite loops
        assert!(
            result.is_ok(),
            "Recursive FileNode failed: {:?}",
            result.err()
        );
        let root = result.unwrap();
        assert_eq!(root.name, "src");
        assert_eq!(root.children.unwrap().len(), 2);
    }
}

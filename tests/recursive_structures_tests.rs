// tests/recursive_structures_tests.rs
#[cfg(test)]
mod recursive_tests {
    #[cfg(feature = "streaming")]
    use futures_util::StreamExt;
    use rstructor::{GeminiClient, GeminiModel, Instructor, LLMClient, RStructorError};
    use serde::{Deserialize, Serialize};

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
    async fn test_gemini_recursive_schema_is_rejected_before_http() {
        let client = GeminiClient::new("offline-test-key")
            .unwrap()
            .model(GeminiModel::Gemini31ProPreview)
            .temperature(0.0)
            .no_retries()
            .base_url("http://127.0.0.1:9");

        let prompt = "Represent a small directory structure: A root folder 'src' containing a file 'lib.rs' (500 bytes) and a subfolder 'backend' which is empty.";
        let result: rstructor::Result<FileNode> = client.materialize(prompt).await;

        assert!(matches!(
            result,
            Err(RStructorError::SchemaCompatibilityError {
                provider,
                context,
                message,
                ..
            }) if provider.as_ref() == "Gemini"
                && context.as_ref() == "structured output"
                && message.contains("finite-depth expansion")
        ));
    }

    #[cfg(feature = "streaming")]
    #[tokio::test]
    async fn test_gemini_recursive_stream_is_rejected_before_http() {
        let client = GeminiClient::new("offline-test-key")
            .unwrap()
            .model(GeminiModel::Gemini31ProPreview)
            .no_retries()
            .base_url("http://127.0.0.1:9");

        let mut stream = client.materialize_stream::<FileNode>("Extract a recursive file tree");
        let error = stream
            .next()
            .await
            .expect("compatibility error item")
            .expect_err("recursive stream must fail locally");

        assert!(matches!(
            error,
            RStructorError::SchemaCompatibilityError {
                provider,
                context,
                ..
            } if provider.as_ref() == "Gemini"
                && context.as_ref() == "structured output"
        ));
        assert!(stream.next().await.is_none());
    }
}

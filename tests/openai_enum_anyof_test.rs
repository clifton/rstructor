//! Regression test for OpenAI `oneOf` rejection bug.
//!
//! OpenAI's structured outputs API rejects `oneOf` in JSON schemas — it only
//! supports `anyOf`. The `rstructor` derive macro must emit `anyOf` for all
//! complex enum schemas so that OpenAI (and other providers) accept them.
//!
//! Run with:
//! ```bash
//! cargo test --test openai_enum_anyof_test
//! ```

#[cfg(test)]
mod openai_enum_anyof_tests {
    use rstructor::Instructor;
    use serde::{Deserialize, Serialize};

    // ── Complex enum: externally tagged (default serde representation) ──

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    #[llm(description = "A type of animal")]
    enum Animal {
        #[llm(description = "A dog")]
        Dog {
            #[llm(description = "The breed of the dog")]
            breed: String,
        },
        #[llm(description = "A cat")]
        Cat {
            #[llm(description = "Whether the cat is indoor-only")]
            indoor: bool,
        },
        #[llm(description = "A mouse")]
        Mouse {
            #[llm(description = "The colour of the mouse")]
            colour: String,
        },
    }

    #[derive(Instructor, Serialize, Deserialize, Debug)]
    #[llm(description = "Identification of a fictional character")]
    struct CharacterId {
        #[llm(description = "Name of the character")]
        name: String,

        #[llm(description = "What kind of animal the character is")]
        animal: Animal,
    }

    // ====================================================================
    // Live OpenAI integration test — requires OPENAI_API_KEY
    // ====================================================================

    /// Send a struct containing a complex enum to OpenAI and confirm the API
    /// accepts the schema and returns valid data.
    ///
    /// Before the fix this would fail with:
    /// > Invalid schema for response_format '...': 'oneOf' is not permitted.
    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn test_openai_accepts_complex_enum_schema() {
        use rstructor::{LLMClient, OpenAIClient, OpenAIModel};
        use std::env;

        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

        let client = OpenAIClient::new(api_key)
            .expect("Failed to create OpenAI client")
            .model(OpenAIModel::Gpt4OMini)
            .temperature(0.0);

        let result = client
            .materialize::<CharacterId>("What kind of animal is Scooby-Doo?")
            .await;

        assert!(
            result.is_ok(),
            "OpenAI should accept the schema with `anyOf`. Error: {:?}",
            result.err()
        );

        let character = result.unwrap();
        assert_eq!(character.name, "Scooby-Doo");
        assert!(
            matches!(character.animal, Animal::Dog { .. }),
            "Scooby-Doo should be identified as a Dog, got: {:?}",
            character.animal
        );
    }
}

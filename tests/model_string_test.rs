#[cfg(test)]
mod tests {
    use rstructor::{AnthropicModel, GeminiModel, GrokModel, OpenAIModel};
    use std::str::FromStr;

    #[test]
    fn test_openai_model_from_string() {
        // Test known model
        let model = OpenAIModel::from_string("gpt-4o");
        assert_eq!(model, OpenAIModel::Gpt4O);

        // Test custom model
        let model = OpenAIModel::from_string("gpt-4-custom");
        match model {
            OpenAIModel::Custom(name) => assert_eq!(name, "gpt-4-custom"),
            _ => panic!("Expected Custom variant"),
        }

        // Test FromStr
        let model = OpenAIModel::from_str("gpt-4o-mini").unwrap();
        assert_eq!(model, OpenAIModel::Gpt4OMini);

        // Test From<&str>
        let model: OpenAIModel = "gpt-3.5-turbo".into();
        assert_eq!(model, OpenAIModel::Gpt35Turbo);
    }

    #[test]
    fn test_anthropic_model_from_string() {
        // Test known model
        let model = AnthropicModel::from_string("claude-sonnet-4-20250514");
        assert_eq!(model, AnthropicModel::ClaudeSonnet4);

        // Test custom model
        let model = AnthropicModel::from_string("claude-custom");
        match model {
            AnthropicModel::Custom(name) => assert_eq!(name, "claude-custom"),
            _ => panic!("Expected Custom variant"),
        }
    }

    #[test]
    fn test_grok_model_as_str() {
        let models = vec![
            GrokModel::Grok4,
            GrokModel::Grok4FastReasoning,
            GrokModel::Grok4FastNonReasoning,
            GrokModel::Grok41FastReasoning,
            GrokModel::Grok41FastNonReasoning,
            GrokModel::Grok3,
            GrokModel::Grok3Mini,
            GrokModel::GrokCodeFast1,
            GrokModel::Grok2Vision,
        ];

        for model in models {
            let model_str = model.as_str();
            let roundtrip_model = GrokModel::from_string(model_str);
            assert_eq!(model, roundtrip_model);
        }
    }

    #[test]
    fn test_grok_model_from_string() {
        let test_strings = vec![
            "grok-4-0709",
            "grok-4-fast-reasoning",
            "grok-4-fast-non-reasoning",
            "grok-4-1-fast-reasoning",
            "grok-4-1-fast-non-reasoning",
            "grok-3",
            "grok-3-mini",
            "grok-code-fast-1",
            "grok-2-vision-1212",
        ];

        for original_string in test_strings {
            let model = GrokModel::from_string(original_string);
            let roundtrip_string = model.as_str();

            assert_eq!(roundtrip_string, original_string);
        }

        // Test custom model
        let model = GrokModel::from_string("grok-custom");
        match model {
            GrokModel::Custom(name) => assert_eq!(name, "grok-custom"),
            _ => panic!("Expected Custom variant"),
        }
    }

    #[test]
    fn test_gemini_model_from_string() {
        // Test known model
        let model = GeminiModel::from_string("gemini-2.5-flash");
        assert_eq!(model, GeminiModel::Gemini25Flash);

        // Test custom model
        let model = GeminiModel::from_string("gemini-custom");
        match model {
            GeminiModel::Custom(name) => assert_eq!(name, "gemini-custom"),
            _ => panic!("Expected Custom variant"),
        }
    }

    #[test]
    fn test_model_as_str_with_custom() {
        let model = OpenAIModel::Custom("my-custom-model".to_string());
        assert_eq!(model.as_str(), "my-custom-model");
    }
}

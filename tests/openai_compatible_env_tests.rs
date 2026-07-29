#![cfg(feature = "openai")]
//! Environment-variable contracts for named OpenAI-compatible constructors.
//!
//! Environment variables are process-global, so all mutations live in this one
//! test and are restored on drop, including non-Unicode values.

use rstructor::{AnyClient, ApiErrorKind, OpenAIClient};

const ENV_KEYS: [&str; 3] = ["OPENROUTER_API_KEY", "GROQ_API_KEY", "MOONSHOT_API_KEY"];

struct EnvGuard {
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl EnvGuard {
    fn capture() -> Self {
        Self {
            saved: ENV_KEYS
                .iter()
                .map(|&key| (key, std::env::var_os(key)))
                .collect(),
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in &self.saved {
            // SAFETY: this is the only test in this binary and restores every
            // environment variable it mutates.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

fn set_only(key: Option<&str>, value: &str) {
    for env_key in ENV_KEYS {
        // SAFETY: environment mutation is confined to the single test below.
        unsafe {
            if Some(env_key) == key {
                std::env::set_var(env_key, value);
            } else {
                std::env::remove_var(env_key);
            }
        }
    }
}

fn assert_authentication_failed(result: rstructor::Result<OpenAIClient>, context: &str) {
    let error = match result {
        Ok(_) => panic!("{context} should fail without its API key"),
        Err(error) => error,
    };
    assert!(
        matches!(
            error.api_error_kind(),
            Some(ApiErrorKind::AuthenticationFailed)
        ),
        "{context}: {error:?}"
    );
}

#[test]
fn compatible_constructor_environment_contracts() {
    let _guard = EnvGuard::capture();

    set_only(None, "");
    OpenAIClient::ollama().expect("Ollama must not require an API key");
    OpenAIClient::lm_studio().expect("LM Studio must not require an API key");
    assert!(matches!(
        rstructor::client("ollama/llama3.3").unwrap(),
        AnyClient::OpenAI(_)
    ));
    assert!(matches!(
        rstructor::client("lm_studio/local-model").unwrap(),
        AnyClient::OpenAI(_)
    ));
    assert_authentication_failed(OpenAIClient::openrouter(), "OpenRouter");
    assert_authentication_failed(OpenAIClient::groq(), "Groq");
    assert_authentication_failed(OpenAIClient::moonshot(), "Moonshot");

    for (variable, constructor) in [
        (
            "OPENROUTER_API_KEY",
            OpenAIClient::openrouter as fn() -> rstructor::Result<OpenAIClient>,
        ),
        ("GROQ_API_KEY", OpenAIClient::groq),
        ("MOONSHOT_API_KEY", OpenAIClient::moonshot),
    ] {
        set_only(Some(variable), &format!("{variable}-value"));
        constructor().unwrap_or_else(|error| {
            panic!("{variable} should construct its compatible client: {error}")
        });

        let route = match variable {
            "OPENROUTER_API_KEY" => "openrouter/vendor/model",
            "GROQ_API_KEY" => "groq/openai/gpt-oss-120b",
            "MOONSHOT_API_KEY" => "moonshot/kimi-k3",
            _ => unreachable!(),
        };
        assert!(
            matches!(rstructor::client(route).unwrap(), AnyClient::OpenAI(_)),
            "{route} should use {variable}"
        );

        set_only(Some(variable), "");
        assert_authentication_failed(constructor(), variable);
    }
}

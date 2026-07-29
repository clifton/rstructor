//! Provider-string routing for runtime client construction.

use std::str::FromStr;

#[cfg(feature = "openai")]
use crate::backend::openai::OpenAIClient;
use crate::backend::{AnyClient, LLMClient, Provider};
use crate::error::{RStructorError, Result};

pub(crate) const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";
pub(crate) const LM_STUDIO_BASE_URL: &str = "http://localhost:1234/v1";
pub(crate) const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
pub(crate) const GROQ_BASE_URL: &str = "https://api.groq.com/openai/v1";
pub(crate) const MOONSHOT_BASE_URL: &str = "https://api.moonshot.ai/v1";

/// How a routed provider obtains its API key.
#[derive(Debug, Clone, Copy)]
pub(crate) enum KeyPolicy {
    ProviderDefault,
    Keyless,
    Environment {
        provider: &'static str,
        variable: &'static str,
    },
}

/// Internal routing metadata kept separate from the public parser result.
///
/// Compatible-provider prefixes carry their endpoint and authentication policy
/// here while preserving the public `(Provider, model)` parser contract.
#[derive(Debug, Clone, Copy)]
struct ClientRoute<'a> {
    provider: Provider,
    model: Option<&'a str>,
    base_url: Option<&'static str>,
    key_policy: KeyPolicy,
}

fn parse_route(spec: &str) -> Result<ClientRoute<'_>> {
    let (provider_name, model) = match spec.split_once('/') {
        Some((provider, model)) => (provider, Some(model)),
        None => (spec, None),
    };

    match provider_name.to_ascii_lowercase().as_str() {
        "openai" | "anthropic" | "gemini" | "grok" | "xai" => Ok(ClientRoute {
            provider: Provider::from_str(provider_name)?,
            model,
            base_url: None,
            key_policy: KeyPolicy::ProviderDefault,
        }),
        "ollama" => {
            openai_compatible_route(provider_name, model, OLLAMA_BASE_URL, KeyPolicy::Keyless)
        }
        "lm_studio" => {
            openai_compatible_route(provider_name, model, LM_STUDIO_BASE_URL, KeyPolicy::Keyless)
        }
        "openrouter" => openai_compatible_route(
            provider_name,
            model,
            OPENROUTER_BASE_URL,
            KeyPolicy::Environment {
                provider: "OpenRouter",
                variable: "OPENROUTER_API_KEY",
            },
        ),
        "groq" => openai_compatible_route(
            provider_name,
            model,
            GROQ_BASE_URL,
            KeyPolicy::Environment {
                provider: "Groq",
                variable: "GROQ_API_KEY",
            },
        ),
        "moonshot" => openai_compatible_route(
            provider_name,
            model,
            MOONSHOT_BASE_URL,
            KeyPolicy::Environment {
                provider: "Moonshot",
                variable: "MOONSHOT_API_KEY",
            },
        ),
        _ => Err(RStructorError::Unsupported(format!(
            "unknown provider `{provider_name}`; valid providers: openai, anthropic, gemini, grok (alias: xai), ollama, lm_studio, openrouter, groq, moonshot"
        ))),
    }
}

fn openai_compatible_route<'a>(
    provider_name: &str,
    model: Option<&'a str>,
    base_url: &'static str,
    key_policy: KeyPolicy,
) -> Result<ClientRoute<'a>> {
    #[cfg(feature = "openai")]
    {
        let _ = provider_name;
        Ok(ClientRoute {
            provider: Provider::OpenAI,
            model,
            base_url: Some(base_url),
            key_policy,
        })
    }
    #[cfg(not(feature = "openai"))]
    {
        if let KeyPolicy::Environment { provider, variable } = key_policy {
            let _ = (provider, variable);
        }
        let _ = (model, base_url);
        Err(RStructorError::Unsupported(format!(
            "provider `{provider_name}` is disabled; enable the `openai` Cargo feature"
        )))
    }
}

/// Parse a `provider/model` client specification without reading the environment.
///
/// The first `/` separates the provider from the model. Any additional `/`
/// characters remain part of the model identifier. Omitting the slash selects
/// the provider's default model.
///
/// Provider names are case-insensitive. Native providers are `openai`,
/// `anthropic`, `gemini`, and `grok` (`xai` is an alias for `grok`).
/// OpenAI-compatible prefixes are `ollama`, `lm_studio`, `openrouter`, `groq`,
/// and `moonshot`; these parse as [`Provider::OpenAI`].
///
/// # Examples
///
/// ```
/// use rstructor::{Provider, parse_client_spec};
///
/// let (provider, model) = parse_client_spec("OpenAI/org/some-model")?;
/// assert_eq!(provider, Provider::OpenAI);
/// assert_eq!(model, Some("org/some-model"));
///
/// let (_, default_model) = parse_client_spec("openai")?;
/// assert_eq!(default_model, None);
///
/// let (provider, model) = parse_client_spec("openrouter/moonshotai/kimi-k3")?;
/// assert_eq!(provider, Provider::OpenAI);
/// assert_eq!(model, Some("moonshotai/kimi-k3"));
/// # Ok::<(), rstructor::RStructorError>(())
/// ```
///
/// # Errors
///
/// Returns an error when the provider is unknown or its Cargo feature is
/// disabled. Disabled-provider errors name the feature to enable.
pub fn parse_client_spec(spec: &str) -> Result<(Provider, Option<&str>)> {
    let route = parse_route(spec)?;
    Ok((route.provider, route.model))
}

/// Build a client from a `provider/model` string.
///
/// Native and hosted providers read their API key from the provider-specific
/// environment variable. Local `ollama` and `lm_studio` routes are keyless.
/// Unknown model identifiers are preserved and sent as custom model strings.
/// Leave off `/model` to retain the OpenAI client's default model; compatible
/// endpoints will normally need an explicit model string.
///
/// # Examples
///
/// ```no_run
/// use rstructor::{Instructor, LLMClient};
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Deserialize, Serialize, Instructor)]
/// struct Ticket {
///     title: String,
/// }
///
/// # async fn example() -> rstructor::Result<()> {
/// let client = rstructor::client("openai/gpt-5.6-sol")?;
/// let ticket: Ticket = client.materialize("Customer cannot sign in").await?;
///
/// // Local endpoints need no API key:
/// let local = rstructor::client("ollama/llama3.3")?;
/// # let _ = ticket;
/// # let _ = local;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an error when the specification is invalid, the provider feature is
/// disabled, or a required provider API key environment variable is unavailable.
pub fn client(spec: &str) -> Result<AnyClient> {
    let route = parse_route(spec)?;
    let client = match route.key_policy {
        KeyPolicy::ProviderDefault => AnyClient::from_env_for(route.provider)?,
        #[cfg(feature = "openai")]
        key_policy @ (KeyPolicy::Keyless | KeyPolicy::Environment { .. }) => {
            AnyClient::OpenAI(OpenAIClient::openai_compatible(
                route
                    .base_url
                    .expect("OpenAI-compatible routes always define a base URL"),
                key_policy,
            )?)
        }
        #[cfg(not(feature = "openai"))]
        KeyPolicy::Keyless | KeyPolicy::Environment { .. } => {
            unreachable!("disabled OpenAI-compatible routes fail during parsing")
        }
    };

    Ok(configure_client(client, route.model, route.base_url))
}

/// Build a client for the first provider whose API key is configured.
///
/// Detection is deterministic. The first available key wins in this order:
/// `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` /
/// `GOOGLE_API_KEY`, then `XAI_API_KEY`.
///
/// # Examples
///
/// ```no_run
/// use rstructor::{Instructor, LLMClient};
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Deserialize, Serialize, Instructor)]
/// struct Ticket {
///     title: String,
/// }
///
/// # async fn example() -> rstructor::Result<()> {
/// let client = rstructor::client_from_env()?;
/// let ticket: Ticket = client.materialize("Customer cannot sign in").await?;
/// # let _ = ticket;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an authentication error when none of the enabled providers has an
/// API key configured.
pub fn client_from_env() -> Result<AnyClient> {
    <AnyClient as LLMClient>::from_env()
}

fn configure_client(client: AnyClient, model: Option<&str>, base_url: Option<&str>) -> AnyClient {
    macro_rules! configure {
        ($client:ident, $variant:path) => {{
            let client = match model {
                Some(model) => $client.model(model),
                None => $client,
            };
            let client = match base_url {
                Some(base_url) => client.base_url(base_url),
                None => client,
            };
            $variant(client)
        }};
    }

    match client {
        #[cfg(feature = "openai")]
        AnyClient::OpenAI(client) => configure!(client, AnyClient::OpenAI),
        #[cfg(feature = "anthropic")]
        AnyClient::Anthropic(client) => configure!(client, AnyClient::Anthropic),
        #[cfg(feature = "gemini")]
        AnyClient::Gemini(client) => configure!(client, AnyClient::Gemini),
        #[cfg(feature = "grok")]
        AnyClient::Grok(client) => configure!(client, AnyClient::Grok),
    }
}

#[cfg(test)]
mod tests {
    use super::parse_client_spec;
    #[cfg(feature = "openai")]
    use super::{KeyPolicy, parse_route};
    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "grok"
    ))]
    use crate::Provider;

    #[cfg(feature = "openai")]
    #[test]
    fn parses_model_default_and_embedded_slashes() {
        assert_eq!(
            parse_client_spec("openai/gpt-5.6-sol").unwrap(),
            (Provider::OpenAI, Some("gpt-5.6-sol"))
        );
        assert_eq!(
            parse_client_spec("openai").unwrap(),
            (Provider::OpenAI, None)
        );
        assert_eq!(
            parse_client_spec("openai/vendor/model/version").unwrap(),
            (Provider::OpenAI, Some("vendor/model/version"))
        );
    }

    #[cfg(feature = "openai")]
    #[test]
    fn parses_openai_compatible_prefixes_and_preserves_model_slashes() {
        for (spec, expected_model) in [
            ("ollama/llama3.3", "llama3.3"),
            ("lm_studio/local-model", "local-model"),
            ("openrouter/moonshotai/kimi-k3", "moonshotai/kimi-k3"),
            ("groq/openai/gpt-oss-120b", "openai/gpt-oss-120b"),
            ("moonshot/kimi-k3", "kimi-k3"),
        ] {
            assert_eq!(
                parse_client_spec(spec).unwrap(),
                (Provider::OpenAI, Some(expected_model)),
                "{spec}"
            );
        }

        assert_eq!(
            parse_client_spec("OlLaMa").unwrap(),
            (Provider::OpenAI, None)
        );
        assert_eq!(
            parse_client_spec("LM_STUDIO/model").unwrap(),
            (Provider::OpenAI, Some("model"))
        );
    }

    #[cfg(feature = "openai")]
    #[test]
    fn compatible_routes_carry_base_url_and_key_policy_metadata() {
        let cases = [
            ("ollama/model", super::OLLAMA_BASE_URL, None),
            ("lm_studio/model", super::LM_STUDIO_BASE_URL, None),
            (
                "openrouter/model",
                super::OPENROUTER_BASE_URL,
                Some(("OpenRouter", "OPENROUTER_API_KEY")),
            ),
            (
                "groq/model",
                super::GROQ_BASE_URL,
                Some(("Groq", "GROQ_API_KEY")),
            ),
            (
                "moonshot/model",
                super::MOONSHOT_BASE_URL,
                Some(("Moonshot", "MOONSHOT_API_KEY")),
            ),
        ];

        for (spec, expected_base_url, expected_env) in cases {
            let route = parse_route(spec).unwrap();
            assert_eq!(route.provider, Provider::OpenAI, "{spec}");
            assert_eq!(route.base_url, Some(expected_base_url), "{spec}");
            match (route.key_policy, expected_env) {
                (KeyPolicy::Keyless, None) => {}
                (
                    KeyPolicy::Environment { provider, variable },
                    Some((expected_provider, expected_variable)),
                ) => {
                    assert_eq!(provider, expected_provider, "{spec}");
                    assert_eq!(variable, expected_variable, "{spec}");
                }
                (actual, expected) => {
                    panic!("unexpected key policy for {spec}: {actual:?}, expected {expected:?}")
                }
            }
        }
    }

    #[cfg(feature = "grok")]
    #[test]
    fn provider_names_are_case_insensitive_and_xai_is_an_alias() {
        assert_eq!(
            parse_client_spec("GrOk/grok-4.5").unwrap(),
            (Provider::Grok, Some("grok-4.5"))
        );
        assert_eq!(
            parse_client_spec("XAI/grok-new").unwrap(),
            (Provider::Grok, Some("grok-new"))
        );
    }

    #[test]
    fn every_enabled_provider_name_is_accepted_case_insensitively() {
        #[cfg(feature = "openai")]
        assert_eq!(
            parse_client_spec("OpEnAi/model").unwrap(),
            (Provider::OpenAI, Some("model"))
        );
        #[cfg(feature = "anthropic")]
        assert_eq!(
            parse_client_spec("AnThRoPiC/model").unwrap(),
            (Provider::Anthropic, Some("model"))
        );
        #[cfg(feature = "gemini")]
        assert_eq!(
            parse_client_spec("GeMiNi/model").unwrap(),
            (Provider::Gemini, Some("model"))
        );
        #[cfg(feature = "grok")]
        assert_eq!(
            parse_client_spec("GrOk/model").unwrap(),
            (Provider::Grok, Some("model"))
        );
    }

    #[test]
    fn unknown_provider_error_lists_every_valid_name() {
        let error = parse_client_spec("mystery/model").unwrap_err().to_string();
        assert!(error.contains("unknown provider `mystery`"));
        for valid_name in [
            "openai",
            "anthropic",
            "gemini",
            "grok",
            "xai",
            "ollama",
            "lm_studio",
            "openrouter",
            "groq",
            "moonshot",
        ] {
            assert!(
                error.contains(valid_name),
                "error should list `{valid_name}`: {error}"
            );
        }
    }

    #[cfg(not(feature = "anthropic"))]
    #[test]
    fn disabled_provider_error_names_the_required_feature() {
        let error = parse_client_spec("anthropic/claude-custom")
            .unwrap_err()
            .to_string();
        assert!(error.contains("provider `anthropic` is disabled"));
        assert!(error.contains("`anthropic` Cargo feature"));
    }

    #[cfg(not(feature = "openai"))]
    #[test]
    fn disabled_compatible_provider_error_names_the_openai_feature() {
        for provider in ["ollama", "lm_studio", "openrouter", "groq", "moonshot"] {
            let error = parse_client_spec(&format!("{provider}/model"))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(&format!("provider `{provider}` is disabled")),
                "{error}"
            );
            assert!(error.contains("`openai` Cargo feature"), "{error}");
        }
    }
}

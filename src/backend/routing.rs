//! Provider-string routing for runtime client construction.

use std::str::FromStr;

use crate::backend::{AnyClient, LLMClient, Provider};
use crate::error::Result;

/// How a routed provider obtains its API key.
///
/// Item 12 can add provider prefixes with optional or prefix-specific key
/// policies without changing the public `provider/model` parser contract.
#[derive(Debug, Clone, Copy)]
enum KeyPolicy {
    ProviderDefault,
}

/// Internal routing metadata kept separate from the public parser result.
///
/// The `base_url` and `key_policy` fields are intentionally present before any
/// compatible-provider prefixes use them. Adding `ollama/`, `openrouter/`, or
/// similar routes can therefore remain a table/parser extension.
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

    Ok(ClientRoute {
        provider: Provider::from_str(provider_name)?,
        model,
        base_url: None,
        key_policy: KeyPolicy::ProviderDefault,
    })
}

/// Parse a `provider/model` client specification without reading the environment.
///
/// The first `/` separates the provider from the model. Any additional `/`
/// characters remain part of the model identifier. Omitting the slash selects
/// the provider's default model.
///
/// Provider names are case-insensitive. Supported names are `openai`,
/// `anthropic`, `gemini`, and `grok`; `xai` is an alias for `grok`.
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
/// The provider's API key is read from its standard environment variable.
/// Unknown model identifiers are preserved and sent as custom model strings.
/// Leave off `/model` to use the provider's default model.
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
/// # let _ = ticket;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an error when the specification is invalid, the provider feature is
/// disabled, or the provider's API key environment variable is unavailable.
pub fn client(spec: &str) -> Result<AnyClient> {
    let route = parse_route(spec)?;
    let client = match route.key_policy {
        KeyPolicy::ProviderDefault => AnyClient::from_env_for(route.provider)?,
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
        for valid_name in ["openai", "anthropic", "gemini", "grok", "xai"] {
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
}

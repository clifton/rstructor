#![cfg(feature = "_client")]
//! Environment-driven construction of [`AnyClient`].
//!
//! Environment variables are process-global, and tests within a single binary
//! share that process. If these scenarios were split across multiple `#[test]`
//! functions, the default parallel test runner would interleave their mutations
//! of the same five keys and produce flaky, order-dependent failures. To keep the
//! behavior deterministic, **all** environment manipulation lives in this single
//! test. The original values of every key are saved on entry and restored on exit
//! (even on panic, via a drop guard).

use rstructor::{AnyClient, ApiErrorKind, LLMClient, Provider, RStructorError};

/// Provider API-key environment variables, in detection-precedence order.
const ENV_KEYS: [&str; 5] = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
];

/// Snapshot of the five keys captured at test start; restores them when dropped.
///
/// Using a drop guard guarantees the original environment is reinstated even if an
/// assertion panics partway through, so a failure here cannot poison other test
/// binaries or the developer's shell session.
struct EnvGuard {
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl EnvGuard {
    fn capture() -> Self {
        let saved = ENV_KEYS
            .iter()
            .map(|&key| (key, std::env::var_os(key)))
            .collect();
        EnvGuard { saved }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in &self.saved {
            // SAFETY: single-threaded restore at end of the only env-mutating test.
            unsafe {
                match value {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

/// Set exactly the named keys (to a dummy value) and remove every other tracked key.
///
/// API keys are never validated at construction time (the clients only read the
/// variable's presence), so a placeholder string is sufficient to drive detection.
fn set_only(present: &[&str]) {
    for &key in &ENV_KEYS {
        // SAFETY: env mutation is confined to this single test (see module docs).
        unsafe {
            if present.contains(&key) {
                std::env::set_var(key, "test-key-placeholder");
            } else {
                std::env::remove_var(key);
            }
        }
    }
}

fn assert_detected(provider: Provider, context: &str) {
    assert_eq!(
        AnyClient::from_env().unwrap().provider(),
        provider,
        "AnyClient::from_env: {context}"
    );
    assert_eq!(
        rstructor::client_from_env().unwrap().provider(),
        provider,
        "rstructor::client_from_env: {context}"
    );
}

#[test]
fn anyclient_from_env_detection_precedence_and_per_provider() {
    let _guard = EnvGuard::capture();

    // --- from_env precedence: OpenAI > Anthropic > Gemini/Google > Grok ---

    // OpenAI + Anthropic both set -> OpenAI wins (highest precedence).
    set_only(&["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]);
    assert_detected(
        Provider::OpenAI,
        "OpenAI must outrank Anthropic when both keys are present",
    );

    // All five set -> still OpenAI.
    set_only(&ENV_KEYS);
    assert_detected(
        Provider::OpenAI,
        "OpenAI must win when every key is present",
    );

    // Remove OpenAI -> Anthropic wins over Grok and Gemini.
    set_only(&["ANTHROPIC_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY"]);
    assert_detected(
        Provider::Anthropic,
        "Anthropic must outrank Grok and Gemini once OpenAI is absent",
    );

    // Only Grok + Gemini -> Gemini wins.
    set_only(&["XAI_API_KEY", "GEMINI_API_KEY"]);
    assert_detected(
        Provider::Gemini,
        "Gemini must outrank Grok when both keys are present",
    );

    // The GOOGLE_API_KEY alias has the same precedence as GEMINI_API_KEY.
    set_only(&["XAI_API_KEY", "GOOGLE_API_KEY"]);
    assert_detected(
        Provider::Gemini,
        "Google's Gemini key alias must outrank Grok",
    );

    // Only Grok (XAI) set -> Grok.
    set_only(&["XAI_API_KEY"]);
    assert_detected(Provider::Grok, "XAI_API_KEY selects Grok");

    // Either Gemini key works.
    set_only(&["GEMINI_API_KEY"]);
    assert_detected(Provider::Gemini, "GEMINI_API_KEY selects Gemini");

    set_only(&["GOOGLE_API_KEY"]);
    assert_detected(Provider::Gemini, "GOOGLE_API_KEY selects Gemini");

    // Only Anthropic set -> Anthropic.
    set_only(&["ANTHROPIC_API_KEY"]);
    assert_detected(Provider::Anthropic, "ANTHROPIC_API_KEY selects Anthropic");

    // Only OpenAI set -> OpenAI.
    set_only(&["OPENAI_API_KEY"]);
    assert_detected(Provider::OpenAI, "OPENAI_API_KEY selects OpenAI");

    // GEMINI_API_KEY remains the preferred alias when both Gemini keys are set.
    set_only(&["GEMINI_API_KEY", "GOOGLE_API_KEY"]);
    assert_detected(
        Provider::Gemini,
        "either Gemini key selects the same provider",
    );

    // --- from_env_for: deterministic per-provider construction ---
    // With its key present each provider builds; once its key is removed it errors
    // with AuthenticationFailed (precedence is irrelevant to from_env_for).

    let cases = [
        (Provider::OpenAI, "OPENAI_API_KEY"),
        (Provider::Anthropic, "ANTHROPIC_API_KEY"),
        (Provider::Grok, "XAI_API_KEY"),
        (Provider::Gemini, "GEMINI_API_KEY"),
    ];

    for (provider, key) in cases {
        // Only this provider's key is present.
        set_only(&[key]);
        let client = AnyClient::from_env_for(provider)
            .unwrap_or_else(|e| panic!("from_env_for({provider:?}) should succeed: {e}"));
        assert_eq!(
            client.provider(),
            provider,
            "from_env_for({provider:?}) must produce a client reporting that provider"
        );

        // Remove all keys -> from_env_for for this provider must fail with AuthenticationFailed.
        // AnyClient does not implement Debug, so match instead of `expect_err`.
        set_only(&[]);
        match AnyClient::from_env_for(provider) {
            Ok(_) => {
                panic!("from_env_for({provider:?}) must fail when the provider's key is absent")
            }
            Err(err) => assert!(
                matches!(
                    err.api_error_kind(),
                    Some(ApiErrorKind::AuthenticationFailed)
                ),
                "from_env_for({provider:?}) without a key should be AuthenticationFailed, got {err:?}"
            ),
        }
    }

    set_only(&["GOOGLE_API_KEY"]);
    assert_eq!(
        AnyClient::from_env_for(Provider::Gemini)
            .unwrap()
            .provider(),
        Provider::Gemini,
        "from_env_for(Gemini) must accept GOOGLE_API_KEY"
    );

    // --- from_env with NO key set -> AuthenticationFailed, provider label "AnyClient" ---

    set_only(&[]);
    // AnyClient does not implement Debug, so match instead of `expect_err`.
    let err = match AnyClient::from_env() {
        Ok(_) => panic!("from_env must fail when no provider key is set"),
        Err(err) => err,
    };
    assert!(
        matches!(
            err.api_error_kind(),
            Some(ApiErrorKind::AuthenticationFailed)
        ),
        "no-key from_env should yield AuthenticationFailed, got {err:?}"
    );
    match &err {
        RStructorError::ApiError { provider, kind, .. } => {
            assert_eq!(
                provider, "AnyClient",
                "no-key from_env must report the synthetic provider label \"AnyClient\""
            );
            assert!(matches!(kind, ApiErrorKind::AuthenticationFailed));
        }
        other => panic!("expected ApiError, got {other:?}"),
    }

    let routed_err = match rstructor::client_from_env() {
        Ok(_) => panic!("client_from_env must fail when no provider key is set"),
        Err(err) => err,
    };
    assert!(
        matches!(
            routed_err.api_error_kind(),
            Some(ApiErrorKind::AuthenticationFailed)
        ),
        "top-level no-key client_from_env should yield AuthenticationFailed, got {routed_err:?}"
    );

    // --- provider() reporting via From<ConcreteClient> ---
    // Build each concrete client (its key is present), convert with `.into()`, and
    // verify the resulting AnyClient reports the matching provider.

    set_only(&["OPENAI_API_KEY"]);
    let openai: AnyClient = rstructor::OpenAIClient::from_env().unwrap().into();
    assert_eq!(openai.provider(), Provider::OpenAI);

    set_only(&["ANTHROPIC_API_KEY"]);
    let anthropic: AnyClient = rstructor::AnthropicClient::from_env().unwrap().into();
    assert_eq!(anthropic.provider(), Provider::Anthropic);

    set_only(&["XAI_API_KEY"]);
    let grok: AnyClient = rstructor::GrokClient::from_env().unwrap().into();
    assert_eq!(grok.provider(), Provider::Grok);

    set_only(&["GEMINI_API_KEY"]);
    let gemini: AnyClient = rstructor::GeminiClient::from_env().unwrap().into();
    assert_eq!(gemini.provider(), Provider::Gemini);

    // `_guard` restores the original environment here.
}

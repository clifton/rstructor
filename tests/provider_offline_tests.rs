//! Offline unit tests for provider-facing pure logic.
//!
//! Covers two report rows (no network, no mockito):
//!   * `provider | ThinkingLevel mapping fns (4) zero direct tests`
//!   * `provider | Model enum from_str round-trip completeness`
//!
//! For the model enums the canonical string conversion is [`as_str`] (the macro
//! in `src/backend/model_macro.rs` generates `as_str`, `from_string`, `FromStr`,
//! `From<&str>` and `From<String>` — but *no* `Display` impl). The "Display
//! round-trips" phrasing in the coverage report is therefore exercised through
//! `as_str` (the real public canonical form), and the round-trip property
//! asserted is `from_string(m.as_str()) == m` plus `from_str`/`From<&str>`
//! agreement, looping over the full variant list read out of the source.

use rstructor::{AnthropicModel, GeminiModel, GrokModel, OpenAIModel, ThinkingLevel};
use std::str::FromStr;

// ---------------------------------------------------------------------------
// ThinkingLevel mapping functions
// ---------------------------------------------------------------------------

#[test]
fn thinking_level_default_is_low() {
    assert_eq!(ThinkingLevel::default(), ThinkingLevel::Low);
}

#[test]
fn thinking_level_claude_budget_tokens() {
    // Off=0, Minimal=1024, Low=2048, Medium=4096, High=8192
    assert_eq!(ThinkingLevel::Off.claude_budget_tokens(), 0);
    assert_eq!(ThinkingLevel::Minimal.claude_budget_tokens(), 1024);
    assert_eq!(ThinkingLevel::Low.claude_budget_tokens(), 2048);
    assert_eq!(ThinkingLevel::Medium.claude_budget_tokens(), 4096);
    assert_eq!(ThinkingLevel::High.claude_budget_tokens(), 8192);
}

#[test]
fn thinking_level_claude_budget_tokens_are_monotonic_above_off() {
    // Sanity: budgets strictly increase from Minimal up to High.
    let ordered = [
        ThinkingLevel::Minimal,
        ThinkingLevel::Low,
        ThinkingLevel::Medium,
        ThinkingLevel::High,
    ];
    for pair in ordered.windows(2) {
        assert!(
            pair[0].claude_budget_tokens() < pair[1].claude_budget_tokens(),
            "{:?} budget should be < {:?} budget",
            pair[0],
            pair[1]
        );
    }
    // Off is the only zero budget.
    assert_eq!(ThinkingLevel::Off.claude_budget_tokens(), 0);
}

#[test]
fn thinking_level_claude_thinking_enabled() {
    // Off => false, everything else => true.
    assert!(!ThinkingLevel::Off.claude_thinking_enabled());
    assert!(ThinkingLevel::Minimal.claude_thinking_enabled());
    assert!(ThinkingLevel::Low.claude_thinking_enabled());
    assert!(ThinkingLevel::Medium.claude_thinking_enabled());
    assert!(ThinkingLevel::High.claude_thinking_enabled());
}

#[test]
fn thinking_level_claude_budget_zero_iff_disabled() {
    // The budget is exactly 0 precisely when thinking is disabled.
    for level in [
        ThinkingLevel::Off,
        ThinkingLevel::Minimal,
        ThinkingLevel::Low,
        ThinkingLevel::Medium,
        ThinkingLevel::High,
    ] {
        assert_eq!(
            level.claude_budget_tokens() == 0,
            !level.claude_thinking_enabled(),
            "budget==0 should match !claude_thinking_enabled for {level:?}"
        );
    }
}

#[test]
fn thinking_level_openai_reasoning_effort() {
    // Off="none", Minimal="low", Low="low", Medium="medium", High="high".
    assert_eq!(ThinkingLevel::Off.openai_reasoning_effort(), Some("none"));
    assert_eq!(
        ThinkingLevel::Minimal.openai_reasoning_effort(),
        Some("low")
    );
    assert_eq!(ThinkingLevel::Low.openai_reasoning_effort(), Some("low"));
    assert_eq!(
        ThinkingLevel::Medium.openai_reasoning_effort(),
        Some("medium")
    );
    assert_eq!(ThinkingLevel::High.openai_reasoning_effort(), Some("high"));
}

#[test]
fn thinking_level_gemini_level() {
    // Off=None, Minimal="minimal", Low="low", Medium="medium", High="high".
    assert_eq!(ThinkingLevel::Off.gemini_level(), None);
    assert_eq!(ThinkingLevel::Minimal.gemini_level(), Some("minimal"));
    assert_eq!(ThinkingLevel::Low.gemini_level(), Some("low"));
    assert_eq!(ThinkingLevel::Medium.gemini_level(), Some("medium"));
    assert_eq!(ThinkingLevel::High.gemini_level(), Some("high"));
}

#[test]
fn thinking_level_off_is_only_none_for_gemini() {
    // gemini_level returns None only for Off (Gemini "disable thinking").
    for level in [
        ThinkingLevel::Minimal,
        ThinkingLevel::Low,
        ThinkingLevel::Medium,
        ThinkingLevel::High,
    ] {
        assert!(
            level.gemini_level().is_some(),
            "{level:?} should produce a Gemini level string"
        );
    }
    assert_eq!(ThinkingLevel::Off.gemini_level(), None);
}

// ---------------------------------------------------------------------------
// Model enum round-trip completeness
// ---------------------------------------------------------------------------
//
// The variant lists below are mirrored from the `define_model_enum!`
// declarations in src/backend/{openai,anthropic,gemini,grok}.rs. Each entry is
// (variant, exact API id). The loops assert the full round-trip surface for
// every named variant.

/// Asserts the round-trip surface for a single named model variant:
///   * `as_str()` returns the expected exact API id,
///   * `from_string(as_str())` recovers the same variant,
///   * `FromStr` and `From<&str>` agree with `from_string`.
macro_rules! assert_model_roundtrip {
    ($ty:ty, $variant:expr, $id:expr) => {{
        let variant: $ty = $variant;
        let id: &str = $id;

        // Named variant -> exact API id.
        assert_eq!(
            variant.as_str(),
            id,
            "{:?}.as_str() mismatch",
            variant.clone()
        );

        // id -> variant (the canonical "Display"/string round-trip).
        let from_str_ctor = <$ty>::from_string(id);
        assert_eq!(
            from_str_ctor, variant,
            "from_string({id:?}) did not recover the named variant"
        );

        // from_string(as_str(v)) == v for every named variant.
        assert_eq!(
            <$ty>::from_string(variant.as_str()),
            variant,
            "as_str -> from_string round-trip failed for {id:?}"
        );

        // FromStr is infallible and agrees with from_string.
        let via_from_str = <$ty>::from_str(id).expect("FromStr is Infallible");
        assert_eq!(via_from_str, variant, "FromStr disagreed for {id:?}");

        // From<&str> agrees too.
        let via_from_ref: $ty = id.into();
        assert_eq!(via_from_ref, variant, "From<&str> disagreed for {id:?}");

        // From<String> agrees too.
        let via_from_string: $ty = id.to_string().into();
        assert_eq!(
            via_from_string, variant,
            "From<String> disagreed for {id:?}"
        );
    }};
}

#[test]
fn openai_model_roundtrip_all_variants() {
    use OpenAIModel::*;
    let table: &[(OpenAIModel, &str)] = &[
        (Gpt55Pro, "gpt-5.5-pro"),
        (Gpt55, "gpt-5.5"),
        (Gpt54Pro, "gpt-5.4-pro"),
        (Gpt54, "gpt-5.4"),
        (Gpt54Mini, "gpt-5.4-mini"),
        (Gpt54Nano, "gpt-5.4-nano"),
        (Gpt53ChatLatest, "gpt-5.3-chat-latest"),
        (Gpt53Codex, "gpt-5.3-codex"),
        (Gpt52Pro, "gpt-5.2-pro"),
        (Gpt52, "gpt-5.2"),
        (Gpt52ChatLatest, "gpt-5.2-chat-latest"),
        (Gpt52Codex, "gpt-5.2-codex"),
        (Gpt51, "gpt-5.1"),
        (Gpt5ChatLatest, "gpt-5-chat-latest"),
        (Gpt5Pro, "gpt-5-pro"),
        (Gpt5, "gpt-5"),
        (Gpt5Nano, "gpt-5-nano"),
        (Gpt5Mini, "gpt-5-mini"),
        (Gpt41, "gpt-4.1"),
        (Gpt41Mini, "gpt-4.1-mini"),
        (Gpt41Nano, "gpt-4.1-nano"),
        (Gpt4O, "gpt-4o"),
        (Gpt4OMini, "gpt-4o-mini"),
        (Gpt4Turbo, "gpt-4-turbo"),
        (Gpt4, "gpt-4"),
        (Gpt35Turbo, "gpt-3.5-turbo"),
    ];
    assert_eq!(table.len(), 26, "OpenAI variant table drifted from source");
    for (variant, id) in table {
        assert_model_roundtrip!(OpenAIModel, variant.clone(), id);
    }
}

#[test]
fn anthropic_model_roundtrip_all_variants() {
    use AnthropicModel::*;
    let table: &[(AnthropicModel, &str)] = &[
        (ClaudeOpus48, "claude-opus-4-8"),
        (ClaudeOpus47, "claude-opus-4-7"),
        (ClaudeSonnet46, "claude-sonnet-4-6"),
        (ClaudeOpus46, "claude-opus-4-6"),
        (ClaudeOpus45, "claude-opus-4-5-20251101"),
        (ClaudeHaiku45, "claude-haiku-4-5-20251001"),
        (ClaudeSonnet45, "claude-sonnet-4-5-20250929"),
        (ClaudeOpus41, "claude-opus-4-1-20250805"),
        (ClaudeOpus4, "claude-opus-4-20250514"),
        (ClaudeSonnet4, "claude-sonnet-4-20250514"),
    ];
    assert_eq!(
        table.len(),
        10,
        "Anthropic variant table drifted from source"
    );
    for (variant, id) in table {
        assert_model_roundtrip!(AnthropicModel, variant.clone(), id);
    }
}

#[test]
fn gemini_model_roundtrip_all_variants() {
    use GeminiModel::*;
    let table: &[(GeminiModel, &str)] = &[
        (Gemini35Flash, "gemini-3.5-flash"),
        (Gemini31ProPreview, "gemini-3.1-pro-preview"),
        (
            Gemini31ProPreviewCustomTools,
            "gemini-3.1-pro-preview-customtools",
        ),
        (Gemini3FlashPreview, "gemini-3-flash-preview"),
        (Gemini31FlashLite, "gemini-3.1-flash-lite"),
        (Gemini31FlashLitePreview, "gemini-3.1-flash-lite-preview"),
        (Gemini3ProPreview, "gemini-3-pro-preview"),
        (Gemini25Pro, "gemini-2.5-pro"),
        (Gemini25Flash, "gemini-2.5-flash"),
        (Gemini25FlashLite, "gemini-2.5-flash-lite"),
        (Gemini25FlashImage, "gemini-2.5-flash-image"),
        (Gemini20Flash, "gemini-2.0-flash"),
        (Gemini20Flash001, "gemini-2.0-flash-001"),
        (Gemini20FlashLite, "gemini-2.0-flash-lite"),
        (Gemini20FlashLite001, "gemini-2.0-flash-lite-001"),
        (GeminiProLatest, "gemini-pro-latest"),
        (GeminiFlashLatest, "gemini-flash-latest"),
        (GeminiFlashLiteLatest, "gemini-flash-lite-latest"),
    ];
    assert_eq!(table.len(), 18, "Gemini variant table drifted from source");
    for (variant, id) in table {
        assert_model_roundtrip!(GeminiModel, variant.clone(), id);
    }
}

#[test]
fn grok_model_roundtrip_all_variants() {
    use GrokModel::*;
    let table: &[(GrokModel, &str)] = &[
        (Grok43, "grok-4.3"),
        (Grok420Reasoning, "grok-4.20-0309-reasoning"),
        (Grok420NonReasoning, "grok-4.20-0309-non-reasoning"),
        (Grok420MultiAgent, "grok-4.20-multi-agent-0309"),
        (GrokBuild01, "grok-build-0.1"),
    ];
    assert_eq!(table.len(), 5, "Grok variant table drifted from source");
    for (variant, id) in table {
        assert_model_roundtrip!(GrokModel, variant.clone(), id);
    }
}

// ---------------------------------------------------------------------------
// Custom-variant fallthrough: "" and unknown strings round-trip via Custom
// ---------------------------------------------------------------------------

#[test]
fn empty_string_maps_to_custom_for_every_provider() {
    // "" is not a known id for any provider -> Custom("") with empty as_str.
    assert_eq!(
        OpenAIModel::from_string(""),
        OpenAIModel::Custom(String::new())
    );
    assert_eq!(
        AnthropicModel::from_string(""),
        AnthropicModel::Custom(String::new())
    );
    assert_eq!(
        GeminiModel::from_string(""),
        GeminiModel::Custom(String::new())
    );
    assert_eq!(GrokModel::from_string(""), GrokModel::Custom(String::new()));

    // as_str of Custom("") is the empty string -> round-trips.
    assert_eq!(OpenAIModel::from_string("").as_str(), "");
    assert_eq!(AnthropicModel::from_string("").as_str(), "");
    assert_eq!(GeminiModel::from_string("").as_str(), "");
    assert_eq!(GrokModel::from_string("").as_str(), "");
}

#[test]
fn unknown_string_maps_to_custom_and_roundtrips() {
    // Arbitrary unknown ids become Custom and round-trip losslessly.
    let unknowns_openai = ["local-llama", "gpt-99-ultra", "GPT-4O"]; // case-sensitive
    for s in unknowns_openai {
        let m = OpenAIModel::from_string(s);
        assert_eq!(m, OpenAIModel::Custom(s.to_string()));
        assert_eq!(m.as_str(), s);
        assert_eq!(OpenAIModel::from_string(m.as_str()), m);
        let via: OpenAIModel = s.into();
        assert_eq!(via, m);
        assert_eq!(OpenAIModel::from_str(s).unwrap(), m);
    }

    // Case sensitivity: uppercased known id is NOT recognized.
    assert_eq!(
        OpenAIModel::from_string("GPT-4O"),
        OpenAIModel::Custom("GPT-4O".to_string())
    );
    assert_eq!(
        AnthropicModel::from_string("CLAUDE-OPUS-4-8"),
        AnthropicModel::Custom("CLAUDE-OPUS-4-8".to_string())
    );
    assert_eq!(
        GeminiModel::from_string("GEMINI-2.5-FLASH"),
        GeminiModel::Custom("GEMINI-2.5-FLASH".to_string())
    );
    assert_eq!(
        GrokModel::from_string("GROK-4.3"),
        GrokModel::Custom("GROK-4.3".to_string())
    );

    // Whitespace-padded known ids are not trimmed -> Custom.
    assert_eq!(
        OpenAIModel::from_string(" gpt-4o "),
        OpenAIModel::Custom(" gpt-4o ".to_string())
    );
}

#[test]
fn custom_variant_roundtrips_through_all_constructors() {
    let id = "some-self-hosted-model-1";
    let custom = GrokModel::Custom(id.to_string());
    assert_eq!(custom.as_str(), id);
    assert_eq!(GrokModel::from_string(id), custom);
    assert_eq!(GrokModel::from_str(id).unwrap(), custom);
    let via_ref: GrokModel = id.into();
    assert_eq!(via_ref, custom);
    let via_string: GrokModel = id.to_string().into();
    assert_eq!(via_string, custom);
}

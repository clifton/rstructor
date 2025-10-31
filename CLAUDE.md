# CLAUDE.md - Agent Guidelines for RStructor

## Rules
- Always make sure there are relevant tests created and that they pass before finishing each task.
- Ensure the generated rustdocs are high quality with examples.
- Doctests should not be ignored and should always be checked.
- Do not add logic into examples that should be library features. Make the library feature instead.
- Always assume that test environment has valid LLM provider api keys.

## Build & Test Commands
- Build: `cargo build`
- Run all tests: `cargo test`
- Run specific test: `cargo test test_name`
- Run examples: `cargo run --example example_name`
- Format code: `cargo fmt`
- Lint: `cargo clippy`

## Code Style Guidelines
- Use Rust 2024 edition features
- Follow Rust naming conventions: snake_case for functions/variables, PascalCase for types
- Use clear, descriptive variable and function names
- Document all public API items with rustdoc comments
- Organize imports in order: std, external crates, local modules
- Prefer returning Result over unwrap/expect in public APIs
- Use tracing for logging instead of println
- Use async/await for all network operations
- Follow the builder pattern for configuration objects
- Use thiserror for custom error types
- Test all new functionality with unit tests
- Format code with rustfmt and address clippy warnings before committing
- Use feature flags for optional dependencies (e.g., specific LLM backends)

## Macro Attribute Implementation Guidelines
- Always implement attribute parsing directly within the macro system, not with JSON strings
- Use native Rust syntax for attribute values and complex data structures
- For example, array literals should be parsed directly as `#[llm(examples = ["one", "two", "three"])]`
- Never use JSON serialized strings for attribute values (e.g., do not use `examples = r#"["value1"]"#`)
- Parse container attributes individually without relying on JSON parsing
- Support multiple attribute specification styles that feel natural in Rust
- For multi-value attributes, support both parentheses and array syntax

## Model List Maintenance Guidelines

**CRITICAL**: Model lists must be kept up-to-date with provider releases. When adding new providers or updating existing ones, always refer to the official documentation sources below:

### Official Model Documentation Sources

**OpenAI:**
- Model Documentation: https://platform.openai.com/docs/models
- Models API Endpoint: https://platform.openai.com/docs/api-reference/models/list

**Anthropic:**
- All Models Overview: https://docs.anthropic.com/en/docs/about-claude/models/all-models
- Models List API: https://docs.anthropic.com/en/api/models-list

**xAI (Grok):**
- Models Documentation: https://docs.x.ai/docs/models

### Update Process

**CRITICAL RULES:**
1. **ALWAYS use API endpoints to get current models** - Never guess or rely on web search. Use the official API endpoints:
   - **OpenAI**: `GET https://api.openai.com/v1/models` (requires `Authorization: Bearer $OPENAI_API_KEY`)
   - **Anthropic**: `GET https://api.anthropic.com/v1/models` (requires `x-api-key: $ANTHROPIC_API_KEY` and `anthropic-version: 2023-06-01`)
   - **xAI (Grok)**: Check `https://docs.x.ai/docs/models` or use their API if available
2. **NEVER guess model identifiers** - Always get exact model names from API responses or official documentation
3. **NEVER rely on web search results** - Web search often returns outdated, incorrect, or speculative information
4. **ALWAYS use exact API identifiers** - Model identifiers must match exactly what the API expects (e.g., `gpt-5-chat-latest`, `claude-sonnet-4-5-20250929`, `grok-4-0709`)
5. **If API key is not available**: Ask the user to provide the exact model identifiers from the API endpoint, rather than guessing
6. **Verify date stamps make sense** - If version X.Y is newer than X.Z, its date stamp should be later (e.g., Claude 3.7 date should be after Claude 3.5 date)
7. **Check for new major versions** - Don't assume only minor version updates; check for major version releases (e.g., Claude 4, GPT-5)
8. **Verify model name format** - Different providers may use different naming conventions (e.g., `claude-sonnet-4-20250514` vs `claude-4-sonnet-20250514`)
9. **Filter for chat completion models** - Only include models suitable for chat completions (exclude specialized models like search-api, codex, audio, etc. unless specifically needed)

**Steps:**
1. **Get current models from API**: Use `curl` or API calls to fetch the latest model list from the provider's `/v1/models` endpoint
2. **Parse API response**: Extract model `id` fields from the JSON response
3. **Filter appropriate models**: For chat completions, include main chat models and exclude specialized variants unless needed
4. **Verify model identifiers**: Ensure date stamps and version numbers are correct (e.g., Claude 3.7 should have a date later than Claude 3.5)
5. **When updating**: Add new models to the appropriate enum (`Model`, `AnthropicModel`, `GrokModel`) in the respective backend files, ordered newest to oldest
6. **Remove deprecated models**: Check API response for models that are no longer available and remove them
7. **Default models**: Update default model selection to use the latest recommended model when appropriate
8. **Documentation**: Update rustdoc comments to reference the official documentation links
9. **Testing**: Ensure new models work correctly with integration tests

### Periodic Review Schedule

- Review model lists quarterly or when new models are announced
- Check for deprecated models and remove or mark them as deprecated
- Update default model selections to use the latest recommended models
- Verify all model identifiers match current API documentation
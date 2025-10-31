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

1. **Before adding/updating models**: Check the official documentation links above for the latest model identifiers
2. **When updating**: Add new models to the appropriate enum (`Model`, `AnthropicModel`, `GrokModel`) in the respective backend files
3. **Model identifiers**: Use the exact API model identifiers from the official documentation (e.g., `gpt-4o`, `claude-3-5-sonnet-20240620`, `grok-2-1212`)
4. **Default models**: Update default model selection to use the latest recommended model when appropriate
5. **Documentation**: Update rustdoc comments to reference the official documentation links
6. **Testing**: Ensure new models work correctly with integration tests

### Periodic Review Schedule

- Review model lists quarterly or when new models are announced
- Check for deprecated models and remove or mark them as deprecated
- Update default model selections to use the latest recommended models
- Verify all model identifiers match current API documentation
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
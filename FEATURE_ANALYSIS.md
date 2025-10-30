# RStructor Feature Analysis: Missing Important Features

## Executive Summary

After analyzing the RStructor codebase and comparing it with the Python Instructor library and best practices from AI blogs, this document identifies potentially missing important features that would enhance the library's capabilities and align it more closely with modern LLM structured output patterns.

## Current Features ✅

1. ✅ Basic structured output generation (`generate_struct`)
2. ✅ Automatic retry with validation error feedback (`generate_struct_with_retry`)
3. ✅ Custom validation via `validate()` method
4. ✅ JSON Schema generation from Rust types
5. ✅ Support for nested structures, enums, arrays
6. ✅ Multiple LLM providers (OpenAI, Anthropic)
7. ✅ Custom types (dates, UUIDs) via `CustomTypeSchema`
8. ✅ Container and field-level attributes (description, examples)
9. ✅ Comprehensive logging via tracing
10. ✅ Error handling with detailed error types

## Missing Critical Features 🔴

### 1. **Streaming Responses / Partial Mode**

**Status**: Missing (mentioned in roadmap but not implemented)

**Impact**: High - Essential for production use cases where:
- Long responses need to be processed incrementally
- Real-time feedback is required
- Token streaming optimization is needed

**Python Instructor Equivalent**:
```python
# Instructor supports streaming with partial validation
response = client.chat.completions.create(
    model="gpt-4",
    response_format={"type": "json_schema", "json_schema": schema},
    stream=True,
    mode=instructor.Mode.PARTIAL
)
```

**Proposed Rust API**:
```rust
// Stream partial responses
let mut stream = client.generate_struct_stream::<Movie>(prompt).await?;
while let Some(partial) = stream.next().await {
    // Process partial data as it arrives
    match partial {
        PartialResponse::Incremental(data) => handle_partial(data),
        PartialResponse::Complete(data) => handle_complete(data),
        PartialResponse::Error(e) => handle_error(e),
    }
}
```

### 2. **Response Modes (Strict, Partial, Function Tools)**

**Status**: Missing

**Impact**: High - Different use cases require different validation strategies:
- **Strict Mode**: Fail fast on first validation error (current behavior)
- **Partial Mode**: Return partial data even if validation fails
- **Tools Mode**: Use function calling for complex workflows

**Python Instructor Equivalent**:
```python
# Different modes for different use cases
client = instructor.patch(openai_client, mode=instructor.Mode.STRICT)
client = instructor.patch(openai_client, mode=instructor.Mode.PARTIAL)
client = instructor.patch(openai_client, mode=instructor.Mode.FUNCTIONS)
```

**Proposed Rust API**:
```rust
#[derive(Debug, Clone, Copy)]
pub enum ResponseMode {
    Strict,   // Fail on validation errors (default)
    Partial,  // Return partial results even if invalid
    Retry,    // Auto-retry with error feedback (current behavior)
}

let client = OpenAIClient::new(api_key)?
    .mode(ResponseMode::Partial)
    .build();
```

### 3. **Conversation History / Multi-Turn Context**

**Status**: Missing

**Impact**: High - Critical for:
- Chat applications
- Multi-step reasoning
- Context-aware structured extraction
- Agent workflows

**Current Limitation**: Only single prompts supported, no message history

**Python Instructor Equivalent**:
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Extract movie info"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "Now extract actor info"}
]
response = client.chat.completions.create(messages=messages, ...)
```

**Proposed Rust API**:
```rust
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

pub enum MessageRole {
    System,
    User,
    Assistant,
}

let messages = vec![
    Message::system("You are a helpful assistant"),
    Message::user("Extract movie info"),
    Message::assistant("Here's the movie info..."),
    Message::user("Now extract actor info"),
];

let result: ActorInfo = client.generate_struct_with_messages::<ActorInfo>(&messages).await?;
```

### 4. **Response Metadata and Extraction Tracking**

**Status**: Missing

**Impact**: Medium - Useful for:
- Debugging why extraction failed
- Understanding token usage
- Tracking which parts of response were parsed
- Usage analytics

**Python Instructor Equivalent**:
```python
response = client.chat.completions.create(...)
# Access metadata: response.usage, response.raw_response, etc.
```

**Proposed Rust API**:
```rust
pub struct GenerationMetadata {
    pub tokens_used: Option<u32>,
    pub model_used: String,
    pub finish_reason: String,
    pub raw_response: Option<String>,
    pub validation_attempts: u32,
}

pub struct GenerationResult<T> {
    pub data: T,
    pub metadata: GenerationMetadata,
}

let result: GenerationResult<Movie> = client.generate_struct_with_metadata::<Movie>(prompt).await?;
println!("Tokens used: {:?}", result.metadata.tokens_used);
```

### 5. **Partial JSON Parsing / Recovery**

**Status**: Partial (basic error messages show partial JSON)

**Impact**: Medium - Would improve:
- Error recovery for malformed JSON
- Better error messages with context
- Ability to extract valid fields from partial responses

**Current Behavior**: Returns error if JSON is malformed
**Proposed**: Attempt to parse valid fields and report which fields failed

**Proposed Rust API**:
```rust
#[derive(Debug)]
pub struct PartialParseResult<T> {
    pub parsed_fields: T,  // Fields that parsed successfully
    pub failed_fields: Vec<String>,  // Fields that failed
    pub partial_json: String,  // Original partial JSON
}

let result: PartialParseResult<Movie> = client.generate_struct_partial::<Movie>(prompt).await?;
```

### 6. **Response Modification / Fixing Hooks**

**Status**: Missing

**Impact**: Medium - Enables:
- Post-processing of LLM responses
- Automatic fixing of common errors
- Custom transformation pipelines

**Python Instructor Equivalent**:
```python
def fix_response(response: str) -> str:
    # Custom logic to fix common issues
    return response.replace("null", '"null"')

client = instructor.patch(openai_client, response_format={"fix": fix_response})
```

**Proposed Rust API**:
```rust
type ResponseFixer = Box<dyn Fn(&str) -> Result<String> + Send + Sync>;

let client = OpenAIClient::new(api_key)?
    .response_fixer(|json| {
        // Fix common JSON issues
        Ok(json.replace("null", "\"null\""))
    })
    .build();
```

### 7. **Batch Processing**

**Status**: Missing

**Impact**: Medium - Important for:
- Processing multiple prompts efficiently
- Rate limit management
- Parallel structured extraction

**Proposed Rust API**:
```rust
let prompts = vec!["Extract movie 1", "Extract movie 2", "Extract movie 3"];
let results: Vec<Result<Movie>> = client.generate_struct_batch::<Movie>(&prompts).await?;
```

### 8. **Custom Validators with Context**

**Status**: Partial (basic validation exists, but limited)

**Impact**: Low-Medium - Would enhance:
- More sophisticated validation logic
- Cross-field validation
- Dependency validation

**Current**: Only `validate()` method with no context
**Proposed**: Validators with access to schema, field paths, etc.

**Proposed Rust API**:
```rust
pub trait Validator<T> {
    fn validate(&self, value: &T, context: &ValidationContext) -> Result<()>;
}

pub struct ValidationContext {
    pub schema: &Schema,
    pub field_path: Vec<String>,
    pub raw_json: Option<&str>,
}
```

### 9. **Schema Refinement / Prompt Enhancement**

**Status**: Partial (hardcoded prompt enhancements)

**Impact**: Low-Medium - Would allow:
- Custom prompt templates
- Better schema-to-prompt conversion
- Provider-specific optimizations

**Current**: Hardcoded prompt templates in each backend
**Proposed**: Configurable prompt builders

### 10. **Tool/Function Calling Beyond Structured Output**

**Status**: Missing

**Impact**: Low - Useful for:
- Multi-function agent workflows
- Complex reasoning chains
- Tool selection before structured output

**Note**: OpenAI backend uses function calling but only for structured output, not for general tool usage.

### 11. **Support for Additional Output Formats**

**Status**: Missing (only JSON/JSON Schema supported)

**Impact**: Low - Some providers support:
- XML output
- YAML output
- Custom formats

### 12. **Async Iterator / Stream Trait Support**

**Status**: Missing

**Impact**: Medium - Important for Rust ecosystem compatibility:
- `Stream` trait from `futures` crate
- Integration with async frameworks
- Composable async workflows

**Proposed Rust API**:
```rust
use futures::Stream;

impl LLMClient {
    fn generate_struct_stream<T>(&self, prompt: &str) -> impl Stream<Item = Result<PartialResponse<T>>>;
}
```

### 13. **Response Caching**

**Status**: Missing

**Impact**: Low-Medium - Useful for:
- Development/testing
- Cost optimization
- Deterministic testing

**Proposed Rust API**:
```rust
let client = OpenAIClient::new(api_key)?
    .cache(CacheStrategy::Memory)
    .build();
```

### 14. **Rate Limiting / Backoff Strategies**

**Status**: Missing

**Impact**: Medium - Critical for production:
- Automatic rate limit handling
- Exponential backoff
- Request queuing

**Proposed Rust API**:
```rust
let client = OpenAIClient::new(api_key)?
    .rate_limit(RateLimitConfig {
        requests_per_minute: 60,
        backoff_strategy: BackoffStrategy::Exponential,
    })
    .build();
```

### 15. **Better Error Context / Debugging**

**Status**: Partial (has error types but limited context)

**Impact**: Medium - Would improve:
- Development experience
- Production debugging
- Error recovery

**Proposed**: Enhanced error types with:
- Full request/response context
- Schema version
- Field-level error mapping
- Suggestions for fixing errors

## Priority Recommendations

### High Priority (Should implement soon)
1. **Streaming Responses** - Critical for production use
2. **Conversation History** - Essential for real-world applications
3. **Response Modes** - Different use cases need different behaviors
4. **Rate Limiting** - Critical for production reliability

### Medium Priority (Nice to have)
5. **Response Metadata** - Better observability
6. **Batch Processing** - Efficiency improvements
7. **Partial JSON Parsing** - Better error recovery
8. **Stream Trait Support** - Ecosystem compatibility

### Low Priority (Future enhancements)
9. **Response Modification Hooks** - Advanced use cases
10. **Custom Validators with Context** - Advanced validation
11. **Response Caching** - Optimization
12. **Additional Output Formats** - Edge cases

## Implementation Notes

### Streaming Implementation Strategy
- Start with OpenAI streaming (ChatCompletionStream)
- Use `futures::Stream` trait for compatibility
- Implement partial parsing that accumulates tokens
- Validate incrementally where possible

### Conversation History Strategy
- Add `messages` parameter to `generate_struct`
- Support system, user, assistant roles
- Maintain conversation state optionally
- Clear separation between single prompt and multi-turn

### Response Modes Strategy
- Make mode a client configuration option
- `Strict`: Current behavior (fail on error)
- `Partial`: Return partial results with errors
- `Retry`: Current `generate_struct_with_retry` behavior (make it default?)

## References

- Python Instructor Library: https://github.com/jxnl/instructor
- Pydantic Documentation: https://docs.pydantic.dev/
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Anthropic Structured Outputs: https://docs.anthropic.com/claude/docs/structured-outputs

## Conclusion

RStructor has a solid foundation with core features working well. The main gaps are around:
1. **Streaming** - Essential for modern LLM applications
2. **Multi-turn conversations** - Required for most real-world use cases
3. **Production features** - Rate limiting, better error handling, observability

Focusing on these areas would bring RStructor to feature parity with Python Instructor and make it production-ready for Rust applications.

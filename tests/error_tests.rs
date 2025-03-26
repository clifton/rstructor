#[cfg(test)]
mod error_tests {
    use rstructor::{RStructorError, Result};
    use serde_json::json;

    #[test]
    fn test_api_error() {
        let err = RStructorError::ApiError("API connection failed".to_string());
        let err_string = format!("{}", err);
        assert_eq!(err_string, "API error: API connection failed");
    }

    #[test]
    fn test_validation_error() {
        let err = RStructorError::ValidationError("Invalid data".to_string());
        let err_string = format!("{}", err);
        assert_eq!(err_string, "Validation error: Invalid data");
    }

    #[test]
    fn test_schema_error() {
        let err = RStructorError::SchemaError("Invalid schema".to_string());
        let err_string = format!("{}", err);
        assert_eq!(err_string, "Schema error: Invalid schema");
    }

    #[test]
    fn test_serialization_error() {
        let err = RStructorError::SerializationError("Failed to serialize".to_string());
        let err_string = format!("{}", err);
        assert_eq!(err_string, "Serialization error: Failed to serialize");
    }

    #[test]
    fn test_from_json_error() {
        // Create a JSON error
        let json_err = serde_json::from_value::<String>(json!(42)).unwrap_err();
        
        // Convert to RStructorError
        let err: RStructorError = json_err.into();
        
        // Check error type and message
        match err {
            RStructorError::JsonError(_) => {
                // Success - correct error type was created
            }
            other => {
                panic!("Expected JsonError, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_result_type() {
        // Test Ok case
        let ok_result: Result<i32> = Ok(42);
        assert_eq!(ok_result, Ok(42));
        
        // Test Error case
        let err_result: Result<i32> = Err(RStructorError::ValidationError("test error".to_string()));
        assert!(err_result.is_err());
    }
}
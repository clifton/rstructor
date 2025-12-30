#[cfg(test)]
mod error_tests {
    use rstructor::{ApiErrorKind, RStructorError, Result};
    use serde_json::json;
    use std::time::Duration;

    #[test]
    fn test_api_error_unexpected_response() {
        let err = RStructorError::api_error(
            "TestProvider",
            ApiErrorKind::UnexpectedResponse {
                details: "No content returned".to_string(),
            },
        );
        let err_string = format!("{}", err);
        assert!(err_string.contains("unexpected response"));
        assert!(err_string.contains("No content returned"));
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_api_error_rich() {
        let err = RStructorError::api_error("OpenAI", ApiErrorKind::AuthenticationFailed);
        let err_string = format!("{}", err);
        assert_eq!(
            err_string,
            "Authentication failed. Check your OPENAI_API_KEY environment variable."
        );
        assert!(!err.is_retryable());
        assert!(matches!(
            err.api_error_kind(),
            Some(ApiErrorKind::AuthenticationFailed)
        ));
    }

    #[test]
    fn test_api_error_rate_limited() {
        let err = RStructorError::api_error(
            "Anthropic",
            ApiErrorKind::RateLimited {
                retry_after: Some(Duration::from_secs(30)),
            },
        );
        assert!(err.is_retryable());
        assert_eq!(err.retry_delay(), Some(Duration::from_secs(30)));
        let err_string = format!("{}", err);
        assert!(err_string.contains("30 seconds"));
    }

    #[test]
    fn test_api_error_kind_is_retryable() {
        // Retryable errors
        assert!(ApiErrorKind::RateLimited { retry_after: None }.is_retryable());
        assert!(
            ApiErrorKind::RateLimited {
                retry_after: Some(Duration::from_secs(10))
            }
            .is_retryable()
        );
        assert!(ApiErrorKind::ServiceUnavailable.is_retryable());
        assert!(ApiErrorKind::GatewayError { code: 520 }.is_retryable());
        assert!(ApiErrorKind::GatewayError { code: 521 }.is_retryable());
        assert!(ApiErrorKind::GatewayError { code: 522 }.is_retryable());
        assert!(ApiErrorKind::GatewayError { code: 523 }.is_retryable());
        assert!(ApiErrorKind::GatewayError { code: 524 }.is_retryable());
        assert!(ApiErrorKind::ServerError { code: 500 }.is_retryable());
        assert!(ApiErrorKind::ServerError { code: 502 }.is_retryable());

        // Non-retryable errors
        assert!(!ApiErrorKind::AuthenticationFailed.is_retryable());
        assert!(!ApiErrorKind::PermissionDenied.is_retryable());
        assert!(!ApiErrorKind::RequestTooLarge.is_retryable());
        assert!(
            !ApiErrorKind::BadRequest {
                details: "test".into()
            }
            .is_retryable()
        );
        assert!(
            !ApiErrorKind::InvalidModel {
                model: "gpt-5".into(),
                suggestion: None
            }
            .is_retryable()
        );
        assert!(
            !ApiErrorKind::Other {
                code: 418,
                message: "teapot".into()
            }
            .is_retryable()
        );
        assert!(
            !ApiErrorKind::UnexpectedResponse {
                details: "test".into()
            }
            .is_retryable()
        );
    }

    #[test]
    fn test_all_error_kinds_have_retry_delay() {
        // Retryable errors have delays
        assert!(
            ApiErrorKind::RateLimited { retry_after: None }
                .retry_delay()
                .is_some()
        );
        assert_eq!(
            ApiErrorKind::RateLimited {
                retry_after: Some(Duration::from_secs(42))
            }
            .retry_delay(),
            Some(Duration::from_secs(42))
        );
        assert!(ApiErrorKind::ServiceUnavailable.retry_delay().is_some());
        assert!(
            ApiErrorKind::GatewayError { code: 520 }
                .retry_delay()
                .is_some()
        );
        assert!(
            ApiErrorKind::ServerError { code: 500 }
                .retry_delay()
                .is_some()
        );

        // Non-retryable errors have no delay
        assert!(ApiErrorKind::AuthenticationFailed.retry_delay().is_none());
        assert!(ApiErrorKind::PermissionDenied.retry_delay().is_none());
        assert!(ApiErrorKind::RequestTooLarge.retry_delay().is_none());
        assert!(
            ApiErrorKind::BadRequest {
                details: "test".into()
            }
            .retry_delay()
            .is_none()
        );
        assert!(
            ApiErrorKind::InvalidModel {
                model: "x".into(),
                suggestion: None
            }
            .retry_delay()
            .is_none()
        );
        assert!(
            ApiErrorKind::UnexpectedResponse {
                details: "x".into()
            }
            .retry_delay()
            .is_none()
        );
    }

    #[test]
    fn test_all_error_kinds_have_user_messages() {
        let test_cases = [
            (
                ApiErrorKind::RateLimited { retry_after: None },
                "Rate limit",
            ),
            (
                ApiErrorKind::RateLimited {
                    retry_after: Some(Duration::from_secs(5)),
                },
                "5 seconds",
            ),
            (
                ApiErrorKind::InvalidModel {
                    model: "test-model".into(),
                    suggestion: None,
                },
                "test-model",
            ),
            (
                ApiErrorKind::InvalidModel {
                    model: "old".into(),
                    suggestion: Some("new".into()),
                },
                "new",
            ),
            (ApiErrorKind::ServiceUnavailable, "temporarily unavailable"),
            (ApiErrorKind::GatewayError { code: 520 }, "520"),
            (ApiErrorKind::AuthenticationFailed, "API_KEY"),
            (ApiErrorKind::PermissionDenied, "Permission denied"),
            (ApiErrorKind::RequestTooLarge, "too large"),
            (
                ApiErrorKind::BadRequest {
                    details: "invalid param".into(),
                },
                "invalid param",
            ),
            (ApiErrorKind::ServerError { code: 500 }, "500"),
            (
                ApiErrorKind::Other {
                    code: 418,
                    message: "I'm a teapot".into(),
                },
                "teapot",
            ),
            (
                ApiErrorKind::UnexpectedResponse {
                    details: "empty array".into(),
                },
                "empty array",
            ),
        ];

        for (kind, expected_substr) in test_cases {
            let msg = kind.user_message("TestProvider");
            assert!(
                msg.contains(expected_substr),
                "Expected '{}' to contain '{}' for {:?}",
                msg,
                expected_substr,
                kind
            );
        }
    }

    #[test]
    fn test_error_display_implementations() {
        // All variants should have Display implementations that don't panic
        let variants = [
            ApiErrorKind::RateLimited { retry_after: None },
            ApiErrorKind::RateLimited {
                retry_after: Some(Duration::from_secs(10)),
            },
            ApiErrorKind::InvalidModel {
                model: "test".into(),
                suggestion: None,
            },
            ApiErrorKind::InvalidModel {
                model: "test".into(),
                suggestion: Some("alt".into()),
            },
            ApiErrorKind::ServiceUnavailable,
            ApiErrorKind::GatewayError { code: 520 },
            ApiErrorKind::AuthenticationFailed,
            ApiErrorKind::PermissionDenied,
            ApiErrorKind::RequestTooLarge,
            ApiErrorKind::BadRequest {
                details: "test".into(),
            },
            ApiErrorKind::ServerError { code: 500 },
            ApiErrorKind::Other {
                code: 999,
                message: "custom".into(),
            },
            ApiErrorKind::UnexpectedResponse {
                details: "test".into(),
            },
        ];

        for variant in variants {
            let display = format!("{}", variant);
            assert!(
                !display.is_empty(),
                "Display for {:?} should not be empty",
                variant
            );
        }
    }

    #[test]
    fn test_rstructor_error_is_retryable() {
        // API errors delegate to kind
        let retryable =
            RStructorError::api_error("Test", ApiErrorKind::RateLimited { retry_after: None });
        assert!(retryable.is_retryable());

        let not_retryable = RStructorError::api_error("Test", ApiErrorKind::AuthenticationFailed);
        assert!(!not_retryable.is_retryable());

        // Timeout is retryable
        assert!(RStructorError::Timeout.is_retryable());

        // Other errors are not retryable
        assert!(!RStructorError::ValidationError("test".into()).is_retryable());
        assert!(!RStructorError::SchemaError("test".into()).is_retryable());
        assert!(!RStructorError::SerializationError("test".into()).is_retryable());
    }

    #[test]
    fn test_timeout_has_retry_delay() {
        assert_eq!(
            RStructorError::Timeout.retry_delay(),
            Some(Duration::from_secs(1))
        );
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
        let err_result: Result<i32> =
            Err(RStructorError::ValidationError("test error".to_string()));
        assert!(err_result.is_err());
    }
}

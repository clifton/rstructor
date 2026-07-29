//! Put typed LLM extraction behind an axum JSON handler.
//!
//! The router is generic over `LLMClient`, so production can inject an
//! `AnyClient` while this runnable example uses `MockClient` and makes one
//! deterministic in-process request:
//!
//! ```text
//! cargo run --example axum_handler_example --features mock
//! ```

use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::{Method, Request, StatusCode, header};
use axum::routing::post;
use axum::{Json, Router};
use rstructor::{Instructor, LLMClient, MockClient};
use serde::{Deserialize, Serialize};
use tower::ServiceExt;

#[derive(Debug, Deserialize, Serialize)]
struct ExtractRequest {
    text: String,
}

#[derive(Debug, Deserialize, Instructor, PartialEq, Serialize)]
#[llm(description = "A reconciled discrepancy between expected and actual positions")]
struct PositionBreak {
    portfolio_id: String,
    symbol: String,
    expected_quantity: i64,
    actual_quantity: i64,
    as_of: String,
}

struct AppState<C> {
    client: C,
}

async fn extract_position_break<C>(
    State(state): State<Arc<AppState<C>>>,
    Json(request): Json<ExtractRequest>,
) -> Result<Json<PositionBreak>, (StatusCode, String)>
where
    C: LLMClient + Send + Sync + 'static,
{
    state
        .client
        .materialize(&request.text)
        .await
        .map(Json)
        .map_err(|error| (StatusCode::UNPROCESSABLE_ENTITY, error.to_string()))
}

fn app<C>(client: C) -> Router
where
    C: LLMClient + Send + Sync + 'static,
{
    Router::new()
        .route(
            "/position-breaks/extract",
            post(extract_position_break::<C>),
        )
        .with_state(Arc::new(AppState { client }))
}

async fn exercise_router() -> Result<PositionBreak, Box<dyn std::error::Error>> {
    let client = MockClient::new().with_response(
        r#"{
            "portfolio_id": "HF-ALPHA-001",
            "symbol": "ESU6",
            "expected_quantity": -240,
            "actual_quantity": -238,
            "as_of": "2026-07-29T14:31:00Z"
        }"#,
    );

    let request = ExtractRequest {
        text: "At 14:31 UTC the HF-ALPHA-001 book expected short 240 ESU6, \
               but the clearing file shows short 238."
            .to_string(),
    };
    let response = app(client)
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/position-breaks/extract")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&request)?))?,
        )
        .await?;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024).await?;
    Ok(serde_json::from_slice(&body)?)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let position_break = exercise_router().await?;
    assert_eq!(
        position_break,
        PositionBreak {
            portfolio_id: "HF-ALPHA-001".to_string(),
            symbol: "ESU6".to_string(),
            expected_quantity: -240,
            actual_quantity: -238,
            as_of: "2026-07-29T14:31:00Z".to_string(),
        }
    );

    println!("{}", serde_json::to_string_pretty(&position_break)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn handler_materializes_a_typed_response_in_process() {
        let position_break = exercise_router()
            .await
            .expect("the in-process request should succeed");

        assert_eq!(position_break.symbol, "ESU6");
        assert_eq!(
            position_break.actual_quantity - position_break.expected_quantity,
            2
        );
    }
}

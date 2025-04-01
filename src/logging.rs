//! Logging functionality for RStructor
//!
//! This module provides utilities for configuring and working with logging
//! through the `tracing` crate.

use tracing::Level;
use tracing_subscriber::{fmt, EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Log levels supported by RStructor.
///
/// These map to the tracing level hierarchy: ERROR, WARN, INFO, DEBUG, TRACE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Error logs only - highest priority messages for critical failures
    Error,
    /// Warning and error logs - indicate potential issues
    Warn,
    /// Info, warning, and error logs - normal operational messages
    Info,
    /// Debug, info, warning, and error logs - detailed information for troubleshooting
    Debug,
    /// Trace, debug, info, warning, and error logs - highly detailed diagnostics
    Trace,
}

impl LogLevel {
    /// Convert to the corresponding tracing level
    fn to_tracing_level(self) -> Level {
        match self {
            LogLevel::Error => Level::ERROR,
            LogLevel::Warn => Level::WARN,
            LogLevel::Info => Level::INFO,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Trace => Level::TRACE,
        }
    }
}

/// Initialize logging for RStructor with a specific log level.
///
/// This function sets up the tracing subscriber with the specified log level.
/// It's typically called once at the start of your application.
///
/// # Examples
///
/// ```no_run
/// use rstructor::logging::{init_logging, LogLevel};
///
/// // Initialize with info level logs
/// init_logging(LogLevel::Info);
///
/// // Now tracing macros will work
/// tracing::info!("Application starting");
/// tracing::debug!("This won't be shown with INFO level");
/// ```
///
/// # Environment Variable
///
/// You can override the logging level by setting the `RSTRUCTOR_LOG` environment variable:
///
/// ```bash
/// RSTRUCTOR_LOG=debug cargo run
/// ```
///
/// This will take precedence over the level passed to `init_logging()`.
pub fn init_logging(level: LogLevel) {
    // Check if RSTRUCTOR_LOG environment variable is set
    let env_filter = EnvFilter::try_from_env("RSTRUCTOR_LOG")
        .unwrap_or_else(|_| {
            // If not set, use the provided level
            EnvFilter::new(format!("rstructor={}", level.to_tracing_level()))
        });

    // Create a subscriber with a custom filter and formatter
    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(env_filter)
        .init();

    // Log initialization
    tracing::info!("RStructor logging initialized at level: {:?}", level);
}

/// Initialize logging with custom environment filter
///
/// This allows for more granular control over what gets logged.
///
/// # Examples
///
/// ```no_run
/// use rstructor::logging::init_logging_with_filter;
///
/// // Initialize with a custom filter string
/// init_logging_with_filter("rstructor=debug,rstructor::backend=trace");
/// ```
pub fn init_logging_with_filter(filter: &str) {
    let env_filter = EnvFilter::try_new(filter).unwrap_or_else(|_| {
        tracing::warn!("Invalid filter string: {}, using default (info)", filter);
        EnvFilter::new("rstructor=info")
    });

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(env_filter)
        .init();

    tracing::info!("RStructor logging initialized with custom filter: {}", filter);
}
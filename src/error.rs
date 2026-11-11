//! Typed error surface of the `llmrace` library.

use thiserror::Error;

/// Convenient result alias used across the library.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced while configuring or running provider benchmarks.
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid configuration (unknown provider, missing env var…).
    #[error("configuration error: {0}")]
    Config(String),

    /// The provider endpoint answered with a non-success HTTP status.
    #[error("{provider} returned HTTP {status}: {body}")]
    Http {
        /// Provider that produced the failure.
        provider: String,
        /// HTTP status code.
        status: u16,
        /// Truncated response body for diagnostics.
        body: String,
    },

    /// The response stream failed or timed out.
    #[error("stream error: {0}")]
    Stream(String),

    /// The provider violated the expected streaming protocol.
    #[error("protocol error: {0}")]
    Protocol(String),

    /// Local I/O failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

//! # llmrace
//!
//! Race LLM providers head-to-head: streaming latency, time-to-first-byte,
//! throughput and cost, measured from real streamed completions.
//!
//! The crate is organised as:
//! - [`provider`]: provider registry, wire protocols, streaming plumbing
//! - [`sse`]: incremental Server-Sent Events parsing
//! - [`stats`]: descriptive statistics with t-based confidence intervals
//! - [`bench`]: the benchmark engine and lane summaries
//! - [`config`]: named profiles from `llmrace.toml`
//! - [`cost`]: per-model pricing and cost estimation
//! - [`report`]: markdown / JSON / CSV exports
//! - [`regression`]: baseline comparison for CI quality gates
//! - [`tui`]: the live terminal race view
//!
//! ## Example
//!
//! Run a quick deterministic race against the built-in mock provider:
//!
//! ```
//! use llmrace::bench::{spawn_lane, BenchConfig};
//! use llmrace::provider::ProviderSpec;
//!
//! # fn main() {
//! # let rt = tokio::runtime::Runtime::new().unwrap();
//! # rt.block_on(async {
//! let config = BenchConfig {
//!     prompt: "hello".into(),
//!     runs: 1,
//!     warmup: 0,
//!     ..BenchConfig::default()
//! };
//! let (handle, _progress) = spawn_lane(
//!     reqwest::Client::new(),
//!     ProviderSpec::mock(),
//!     "mock-1".into(),
//!     config,
//! );
//! let summary = handle.await.unwrap();
//! assert!(summary.successful());
//! # });
//! # }
//! ```

#![deny(missing_docs)]
#![deny(clippy::unwrap_used)]

pub mod bench;
pub mod config;
pub mod cost;
pub mod error;
pub mod provider;
pub mod regression;
pub mod report;
pub mod sse;
pub mod stats;
pub mod tui;

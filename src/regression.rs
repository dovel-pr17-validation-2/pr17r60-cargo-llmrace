//! Regression detection against a previously recorded baseline report.
//!
//! Enables `llmrace` to act as a CI quality gate: race the providers, then
//! compare the fresh results against a JSON report from an earlier run and
//! fail (non-zero exit) when any metric degrades beyond a threshold.

use crate::bench::LaneSummary;
use serde::Deserialize;

/// Threshold-aware comparison result for one provider.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderComparison {
    /// Provider id.
    pub name: String,
    /// Model recorded in the baseline.
    pub baseline_model: String,
    /// Model benchmarked now.
    pub current_model: String,
    /// Per-metric comparisons.
    pub metrics: Vec<MetricComparison>,
}

/// Comparison of a single metric between baseline and current run.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricComparison {
    /// Human-readable metric label.
    pub label: &'static str,
    /// Baseline value.
    pub baseline: f64,
    /// Current value.
    pub current: f64,
    /// Relative change in percent (positive = worse for latency metrics).
    pub delta_pct: f64,
    /// Whether this metric regressed beyond the threshold.
    pub regressed: bool,
}

/// Outcome of a full baseline comparison.
#[derive(Debug, Clone, Default)]
pub struct RegressionReport {
    /// Per-provider comparisons for providers present in both runs.
    pub comparisons: Vec<ProviderComparison>,
    /// Providers benchmarked now that were absent from the baseline.
    pub new_providers: Vec<String>,
    /// Providers present in the baseline that were not benchmarked now.
    pub skipped_providers: Vec<String>,
}

impl RegressionReport {
    /// `true` when at least one metric regressed beyond the threshold.
    pub fn any_regressed(&self) -> bool {
        self.comparisons
            .iter()
            .any(|c| c.metrics.iter().any(|m| m.regressed))
    }

    /// Renders the comparison as a terminal table.
    pub fn terminal_table(&self, max_regression_pct: f64) -> String {
        let mut out = format!(
            "regression check vs baseline (threshold {max_regression_pct:.0}%)\n{:<14} {:<16} {:>18} {:>18} {:>9}\n",
            "Provider", "Metric", "Baseline", "Current", "Δ"
        );
        out.push_str(&"-".repeat(out.len()));
        out.push('\n');
        for c in &self.comparisons {
            for m in &c.metrics {
                let mark = if m.regressed { " ✗" } else { "" };
                out.push_str(&format!(
                    "{:<14} {:<16} {:>16.1} → {:<16.1} {:>+8.1}%{}\n",
                    c.name, m.label, m.baseline, m.current, m.delta_pct, mark
                ));
            }
        }
        for name in &self.new_providers {
            out.push_str(&format!("{name:<14} new provider (no baseline)\n"));
        }
        for name in &self.skipped_providers {
            out.push_str(&format!("{name:<14} in baseline but not benchmarked\n"));
        }
        if self.any_regressed() {
            out.push_str("\n✗ REGRESSION DETECTED: at least one metric exceeded the threshold\n");
        } else {
            out.push_str("\n✓ no regressions beyond threshold\n");
        }
        out
    }
}

/// The subset of a saved JSON report needed for regression checks.
#[derive(Debug, Clone, Deserialize)]
pub struct BaselineReport {
    /// Providers recorded in the baseline.
    pub providers: Vec<BaselineProvider>,
}

/// One provider entry of a baseline report.
#[derive(Debug, Clone, Deserialize)]
pub struct BaselineProvider {
    /// Provider id.
    pub provider: String,
    /// Model that was benchmarked.
    #[serde(default)]
    pub model: String,
    /// Mean total latency in milliseconds.
    #[serde(default)]
    pub mean_total_ms: f64,
    /// p95 total latency in milliseconds.
    #[serde(default)]
    pub p95_total_ms: f64,
    /// Mean time-to-first-byte in milliseconds.
    #[serde(default)]
    pub mean_ttfb_ms: f64,
    /// Aggregate throughput in tokens per second.
    #[serde(default)]
    pub mean_tokens_per_sec: f64,
}

impl BaselineReport {
    /// Parses a baseline from the JSON written by `llmrace --json`.
    pub fn from_json_str(raw: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(raw)
    }
}

/// Compares a fresh race against a baseline with a regression threshold in percent.
///
/// Latency metrics (mean, p95, TTFB) regress when they increase beyond the
/// threshold; throughput regresses when it decreases beyond the threshold.
/// Providers must have succeeded in the current run to be compared.
pub fn evaluate(
    current: &[LaneSummary],
    baseline: &BaselineReport,
    max_regression_pct: f64,
) -> RegressionReport {
    let mut report = RegressionReport::default();
    let mut compared: Vec<String> = Vec::new();

    for lane in current.iter().filter(|s| s.successful()) {
        let Some(base) = baseline.providers.iter().find(|p| p.provider == lane.name) else {
            report.new_providers.push(lane.name.clone());
            continue;
        };
        compared.push(lane.name.clone());
        let metrics = vec![
            latency_metric(
                "mean latency",
                base.mean_total_ms,
                lane.mean_total_ms(),
                max_regression_pct,
            ),
            latency_metric(
                "p95 latency",
                base.p95_total_ms,
                lane.p95_total_ms(),
                max_regression_pct,
            ),
            latency_metric(
                "ttfb",
                base.mean_ttfb_ms,
                lane.mean_ttfb_ms(),
                max_regression_pct,
            ),
            throughput_metric(
                "tokens/s",
                base.mean_tokens_per_sec,
                lane.mean_tokens_per_sec(),
                max_regression_pct,
            ),
        ];
        report.comparisons.push(ProviderComparison {
            name: lane.name.clone(),
            baseline_model: base.model.clone(),
            current_model: lane.model.clone(),
            metrics,
        });
    }

    report.skipped_providers = baseline
        .providers
        .iter()
        .map(|p| p.provider.clone())
        .filter(|name| !compared.contains(name))
        .collect();
    report
}

fn latency_metric(
    label: &'static str,
    baseline: f64,
    current: f64,
    threshold: f64,
) -> MetricComparison {
    let delta_pct = if baseline.abs() < f64::EPSILON {
        0.0
    } else {
        (current - baseline) / baseline * 100.0
    };
    MetricComparison {
        label,
        baseline,
        current,
        delta_pct,
        regressed: delta_pct > threshold,
    }
}

fn throughput_metric(
    label: &'static str,
    baseline: f64,
    current: f64,
    threshold: f64,
) -> MetricComparison {
    let delta_pct = if baseline.abs() < f64::EPSILON {
        0.0
    } else {
        (current - baseline) / baseline * 100.0
    };
    MetricComparison {
        label,
        baseline,
        current,
        delta_pct,
        regressed: delta_pct < -threshold,
    }
}

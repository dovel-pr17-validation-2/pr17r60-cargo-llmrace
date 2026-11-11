//! Descriptive statistics for benchmark samples.
//!
//! Provides mean, median, percentiles with linear interpolation, sample
//! variance, a 95% confidence interval for the mean via the Student t
//! distribution, and 3-sigma outlier detection.

/// A collection of `f64` observations with descriptive-statistics methods.
#[derive(Debug, Clone, Default)]
pub struct Sample {
    values: Vec<f64>,
}

impl Sample {
    /// Creates an empty sample.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a sample from an existing vector.
    pub fn from_values(values: Vec<f64>) -> Self {
        Sample { values }
    }

    /// Appends an observation.
    pub fn push(&mut self, v: f64) {
        self.values.push(v);
    }

    /// Returns the number of observations.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` when no observations have been recorded.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns a slice of the raw observations.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Arithmetic mean. Returns `0.0` for an empty sample.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Minimum observation, or `None` if empty.
    pub fn min(&self) -> Option<f64> {
        self.values.iter().copied().fold(None, |acc, v| match acc {
            None => Some(v),
            Some(m) => Some(m.min(v)),
        })
    }

    /// Maximum observation, or `None` if empty.
    pub fn max(&self) -> Option<f64> {
        self.values.iter().copied().fold(None, |acc, v| match acc {
            None => Some(v),
            Some(m) => Some(m.max(v)),
        })
    }

    /// Median (50th percentile). Returns `0.0` for an empty sample.
    pub fn median(&self) -> f64 {
        self.percentile(50.0)
    }

    /// Unbiased sample variance (denominator `n - 1`). Returns `0.0` for fewer than two observations.
    pub fn variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        let ss: f64 = self.values.iter().map(|v| (v - m) * (v - m)).sum();
        ss / (self.values.len() - 1) as f64
    }

    /// Sample standard deviation. Returns `0.0` for fewer than two observations.
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Percentile with linear interpolation between closest ranks.
    ///
    /// The `p` argument is a percentage in `0.0..=100.0`. For an empty sample
    /// this returns `0.0`.
    ///
    /// ```
    /// use llmrace::stats::Sample;
    /// let s = Sample::from_values(vec![10.0, 20.0, 30.0, 40.0, 100.0]);
    /// assert!((s.percentile(50.0) - 30.0).abs() < 1e-12);
    /// ```
    pub fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        if self.values.len() == 1 {
            return self.values[0];
        }
        let p = p.clamp(0.0, 100.0);
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = p / 100.0 * (sorted.len() - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let frac = idx - lo as f64;
            sorted[lo] + frac * (sorted[hi] - sorted[lo])
        }
    }

    /// 95% confidence interval for the mean, `(low, high)`.
    ///
    /// Uses the Student t distribution with interpolation over a hardcoded
    /// critical-value table for `df <= 30` and the normal approximation
    /// (`z = 1.96`) beyond. Returns `None` for fewer than two observations.
    pub fn t_ci95(&self) -> Option<(f64, f64)> {
        let n = self.values.len();
        if n < 2 {
            return None;
        }
        let df = n - 1;
        let t = t_critical_95(df);
        let m = self.mean();
        let half = t * self.stddev() / (n as f64).sqrt();
        Some((m - half, m + half))
    }

    /// Indices of observations more than three standard deviations from the mean.
    pub fn outliers_3sigma(&self) -> Vec<usize> {
        if self.values.len() < 2 {
            return Vec::new();
        }
        let m = self.mean();
        let sd = self.stddev();
        if sd == 0.0 {
            return Vec::new();
        }
        self.values
            .iter()
            .enumerate()
            .filter(|(_, v)| ((**v - m).abs() / sd) > 3.0)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Two-sided 95% critical values of the Student t distribution by degrees of freedom.
const T_TABLE_95: &[(usize, f64)] = &[
    (1, 12.706),
    (2, 4.303),
    (3, 3.182),
    (4, 2.776),
    (5, 2.571),
    (6, 2.447),
    (7, 2.365),
    (8, 2.306),
    (9, 2.262),
    (10, 2.228),
    (15, 2.131),
    (20, 2.086),
    (25, 2.060),
    (30, 2.042),
];

fn t_critical_95(df: usize) -> f64 {
    if df >= 30 {
        return 1.96;
    }
    if let Some((_, v)) = T_TABLE_95.iter().find(|(k, _)| *k == df) {
        return *v;
    }
    let Some((lo_df, lo_t)) = T_TABLE_95.iter().rev().find(|(k, _)| *k < df) else {
        return 1.96;
    };
    let Some((hi_df, hi_t)) = T_TABLE_95.iter().find(|(k, _)| *k > df) else {
        return 1.96;
    };
    let w = (df - lo_df) as f64 / (hi_df - lo_df) as f64;
    lo_t + w * (hi_t - lo_t)
}

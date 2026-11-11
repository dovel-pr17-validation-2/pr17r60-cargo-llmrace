use futures_util::StreamExt;
use llmrace::bench::{spawn_lane, BenchConfig, RunMetrics};
use llmrace::provider::{decode_payload, stream_completion, CompletionRequest, ProviderSpec, Wire};
use llmrace::regression::{evaluate, BaselineReport};
use llmrace::sse::{parse_data_events, SseParser};
use llmrace::stats::Sample;
use llmrace::{cost, report};

#[test]
fn sse_parses_events_split_at_every_boundary() {
    let payload = "data: {\"a\":1}\n\ndata: {\"b\":2}\n\ndata: [DONE]\n\n";
    let bytes = payload.as_bytes();
    for split in 1..bytes.len() {
        let mut parser = SseParser::new();
        let mut blocks = parser.feed(&bytes[..split]);
        blocks.extend(parser.feed(&bytes[split..]));
        assert_eq!(blocks.len(), 3, "split at {split}");
    }
}

#[test]
fn sse_handles_crlf_and_mixed_separators() {
    let mut parser = SseParser::new();
    let blocks = parser.feed(b"data: one\r\n\r\ndata: two\n\ndata: three\r\n\r\n");
    assert_eq!(blocks.len(), 3);
    assert_eq!(parse_data_events(&blocks[0]), vec!["one"]);
    assert_eq!(parse_data_events(&blocks[1]), vec!["two"]);
    assert_eq!(parse_data_events(&blocks[2]), vec!["three"]);
}

#[test]
fn sse_finish_flushes_trailing_block() {
    let mut parser = SseParser::new();
    assert!(parser.feed(b"data: complete\n\n").len() == 1);
    let flushed = parser.finish();
    assert!(flushed.is_empty());
    let mut parser = SseParser::new();
    parser.feed(b"data: complete\n\ndata: partial");
    assert_eq!(parser.finish().len(), 1);
}

#[test]
fn sse_ignores_non_data_lines() {
    let mut parser = SseParser::new();
    let blocks = parser.feed(b": comment\nevent: x\nid: 7\ndata: payload\n\n");
    assert_eq!(blocks.len(), 1);
    assert_eq!(parse_data_events(&blocks[0]), vec!["payload"]);
}

#[test]
fn percentile_linear_interpolation_is_exact() {
    let s = Sample::from_values(vec![10.0, 20.0, 30.0, 40.0, 100.0]);
    assert!((s.percentile(50.0) - 30.0).abs() < 1e-12);
    assert!((s.percentile(0.0) - 10.0).abs() < 1e-12);
    assert!((s.percentile(100.0) - 100.0).abs() < 1e-12);
    assert!((s.percentile(95.0) - 88.0).abs() < 1e-12);
    assert!((s.percentile(25.0) - 20.0).abs() < 1e-12);
}

#[test]
fn t_ci95_width_matches_table() {
    let s = Sample::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let (lo, hi) = s.t_ci95().unwrap();
    let mean = 3.0;
    let expected_half = 2.776 * s.stddev() / 5.0_f64.sqrt();
    assert!((mean - lo - expected_half).abs() < 1e-3);
    assert!((hi - mean - expected_half).abs() < 1e-3);
    assert!(Sample::new().t_ci95().is_none());
}

#[test]
fn outliers_flagged_beyond_three_sigma() {
    let mut values = vec![0.0; 20];
    values.push(100.0);
    let s = Sample::from_values(values);
    assert_eq!(s.outliers_3sigma(), vec![20]);
}

#[test]
fn openai_wire_decodes_deltas_and_usage() {
    let delta = decode_payload(
        r#"{"choices":[{"delta":{"content":"Hel"}}]}"#,
        Wire::OpenAiCompat,
    )
    .unwrap();
    assert_eq!(delta.token.as_deref(), Some("Hel"));

    let empty = decode_payload(
        r#"{"choices":[{"delta":{"content":""}}]}"#,
        Wire::OpenAiCompat,
    )
    .unwrap();
    assert!(empty.token.is_none());

    let usage = decode_payload(
        r#"{"choices":[{"delta":{}}],"usage":{"completion_tokens":2}}"#,
        Wire::OpenAiCompat,
    )
    .unwrap();
    assert_eq!(usage.completion_tokens, Some(2));

    assert!(decode_payload("[DONE]", Wire::OpenAiCompat).is_none());
}

#[test]
fn anthropic_wire_decodes_events() {
    let delta = decode_payload(
        r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}"#,
        Wire::Anthropic,
    )
    .unwrap();
    assert_eq!(delta.token.as_deref(), Some("Hi"));

    let usage = decode_payload(
        r#"{"type":"message_delta","usage":{"output_tokens":17}}"#,
        Wire::Anthropic,
    )
    .unwrap();
    assert_eq!(usage.completion_tokens, Some(17));

    assert!(decode_payload(r#"{"type":"ping"}"#, Wire::Anthropic).is_none());
}

#[test]
fn custom_provider_points_at_custom_endpoint() {
    let spec = ProviderSpec::custom(
        "vllm",
        "http://gpu-box:8000/v1/chat/completions",
        Some("VLLM_API_KEY"),
        "meta-llama/Llama-3.1-8B-Instruct",
    );
    assert_eq!(*spec.id, *"vllm");
    assert_eq!(
        spec.endpoint.as_ref(),
        "http://gpu-box:8000/v1/chat/completions"
    );
    assert_eq!(spec.wire, Wire::OpenAiCompat);
    assert_eq!(spec.api_key_env.as_deref(), Some("VLLM_API_KEY"));
}

#[tokio::test]
async fn mock_stream_emits_tokens_and_usage() {
    let client = reqwest::Client::new();
    let spec = ProviderSpec::mock();
    let req = CompletionRequest {
        model: "mock-1",
        prompt: "hello",
        max_tokens: 32,
    };
    let mut stream = stream_completion(&client, &spec, &req).await.unwrap();
    let mut tokens = 0u64;
    let mut usage = None;
    while let Some(ev) = stream.next().await {
        let ev = ev.unwrap();
        if ev.token.is_some() {
            tokens += 1;
        }
        if let Some(n) = ev.completion_tokens {
            usage = Some(n);
        }
    }
    assert_eq!(tokens, 24);
    assert_eq!(usage, Some(24));
}

#[tokio::test]
async fn full_mock_race_produces_reports() {
    let config = BenchConfig {
        prompt: "demo".into(),
        runs: 2,
        warmup: 0,
        ..BenchConfig::default()
    };
    let (handle, progress) = spawn_lane(
        reqwest::Client::new(),
        ProviderSpec::mock(),
        "mock-1".into(),
        config,
    );
    drop(progress);
    let summary = handle.await.unwrap();
    assert!(summary.successful(), "error: {:?}", summary.error);
    assert_eq!(summary.runs.len(), 2);
    assert!(summary.mean_tokens_per_sec() > 0.0);
    assert!(summary.ci95_total_ms().is_some());

    let md = report::markdown(std::slice::from_ref(&summary), "demo", 2, 32);
    assert!(md.contains("# 🏁 llmrace report"));
    assert!(md.contains("mock"));
    assert!(md.contains("±"));

    let csv = report::csv(std::slice::from_ref(&summary));
    assert!(csv.starts_with("provider,model,run,total_ms"));
    assert_eq!(csv.lines().count(), 3);

    let json = report::json_report(&[summary], "demo", 2, 32);
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["providers"][0]["provider"], "mock");
}

#[tokio::test]
async fn concurrency_two_measures_saturation_throughput() {
    let config = BenchConfig {
        prompt: "demo".into(),
        runs: 1,
        warmup: 0,
        concurrency: 2,
        ..BenchConfig::default()
    };
    let (handle, progress) = spawn_lane(
        reqwest::Client::new(),
        ProviderSpec::mock(),
        "mock-1".into(),
        config,
    );
    drop(progress);
    let summary = handle.await.unwrap();
    assert!(summary.successful());
    assert_eq!(summary.runs.len(), 1);
    assert_eq!(summary.concurrency, 2);
    assert_eq!(summary.runs[0].tokens, 48);
    let single_stream = 24.0 / (0.276);
    assert!(
        summary.mean_tokens_per_sec() > single_stream * 1.5,
        "aggregate {} should approximate saturation throughput, not the per-stream {}",
        summary.mean_tokens_per_sec(),
        single_stream
    );
}

#[test]
fn run_metrics_generation_math() {
    let m = RunMetrics {
        total_ms: 1000.0,
        ttfb_ms: 200.0,
        tokens: 160,
        chars: 640,
        prompt_tokens: 0,
    };
    assert!((m.gen_ms() - 800.0).abs() < 1e-9);
    assert!((m.tokens_per_sec() - 200.0).abs() < 1e-9);
}

#[test]
fn cost_estimates_and_pricing_sane() {
    let t = cost::estimate_prompt_tokens("Explain zero-knowledge proofs in two paragraphs.");
    assert!(t > 5.0 && t < 20.0, "estimate: {t}");
    assert_eq!(cost::pricing("openai", "gpt-4o-mini"), Some((0.15, 0.60)));
    assert_eq!(
        cost::pricing("anthropic", "claude-sonnet-4-5"),
        Some((3.0, 15.0))
    );
    assert_eq!(cost::pricing("custom-anything", "whatever"), None);
    assert_eq!(cost::format_usd(0.000234), "$0.000234");
}

fn regression_summary(
    name: &str,
    mean_ms: f64,
    p95_ms: f64,
    ttfb_ms: f64,
    tps: f64,
) -> llmrace::bench::LaneSummary {
    let gen_s = (mean_ms - ttfb_ms).max(0.0) / 1000.0;
    let run = RunMetrics {
        total_ms: mean_ms,
        ttfb_ms,
        tokens: (tps * gen_s).round() as u64,
        chars: 400,
        prompt_tokens: 0,
    };
    let _ = p95_ms;
    llmrace::bench::LaneSummary {
        name: name.into(),
        model: "m".into(),
        runs: vec![run],
        error: None,
        prompt: "p".into(),
        concurrency: 1,
    }
}

#[test]
fn regression_passes_within_threshold() {
    let baseline_raw = r#"{"providers":[{"provider":"mock","model":"mock-1","mean_total_ms":100.0,"p95_total_ms":120.0,"mean_ttfb_ms":20.0,"mean_tokens_per_sec":80.0}]}"#;
    let baseline = BaselineReport::from_json_str(baseline_raw).unwrap();
    let current = vec![regression_summary("mock", 110.0, 130.0, 22.0, 76.0)];
    let report = evaluate(&current, &baseline, 15.0);
    assert!(!report.any_regressed());
    assert!(report.comparisons.len() == 1);
    assert!(report.new_providers.is_empty());
    assert!(report.skipped_providers.is_empty());
}

#[test]
fn regression_fails_beyond_threshold() {
    let baseline_raw = r#"{"providers":[{"provider":"mock","model":"mock-1","mean_total_ms":100.0,"p95_total_ms":120.0,"mean_ttfb_ms":20.0,"mean_tokens_per_sec":80.0}]}"#;
    let baseline = BaselineReport::from_json_str(baseline_raw).unwrap();

    let slow = vec![regression_summary("mock", 130.0, 150.0, 20.0, 78.0)];
    let report = evaluate(&slow, &baseline, 15.0);
    assert!(report.any_regressed());
    let mean = report.comparisons[0]
        .metrics
        .iter()
        .find(|m| m.label == "mean latency")
        .unwrap();
    assert!(mean.regressed);
    assert!((mean.delta_pct - 30.0).abs() < 1e-9);

    let dropped_tps = vec![regression_summary("mock", 100.0, 120.0, 20.0, 60.0)];
    let report = evaluate(&dropped_tps, &baseline, 15.0);
    assert!(report.any_regressed());
}

#[test]
fn regression_reports_new_and_skipped_providers() {
    let baseline_raw = r#"{"providers":[{"provider":"groq","model":"m","mean_total_ms":100.0,"mean_tokens_per_sec":80.0}]}"#;
    let baseline = BaselineReport::from_json_str(baseline_raw).unwrap();
    let current = vec![regression_summary("mock", 100.0, 100.0, 10.0, 80.0)];
    let report = evaluate(&current, &baseline, 15.0);
    assert_eq!(report.new_providers, vec!["mock".to_string()]);
    assert_eq!(report.skipped_providers, vec!["groq".to_string()]);
    assert!(!report.any_regressed());
    let table = report.terminal_table(15.0);
    assert!(table.contains("new provider"));
    assert!(table.contains("not benchmarked"));
}

#[test]
fn gemini_wire_decodes_deltas_and_usage() {
    let delta = decode_payload(
        r#"{"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}"#,
        Wire::Gemini,
    )
    .unwrap();
    assert_eq!(delta.token.as_deref(), Some("Hel"));

    let usage = decode_payload(
        r#"{"candidates":[],"usageMetadata":{"candidatesTokenCount":42}}"#,
        Wire::Gemini,
    )
    .unwrap();
    assert_eq!(usage.completion_tokens, Some(42));

    assert!(
        decode_payload(
            r#"{"candidates":[{"content":{"parts":[{"text":""}]}}]}"#,
            Wire::Gemini
        )
        .is_none()
            || decode_payload(
                r#"{"candidates":[{"content":{"parts":[{"text":""}]}}]}"#,
                Wire::Gemini
            )
            .map(|e| e.token.is_none())
            .unwrap_or(true)
    );
}

#[test]
fn gemini_registry_entry() {
    let spec = llmrace::provider::ProviderSpec::find("gemini").unwrap();
    assert_eq!(spec.wire, Wire::Gemini);
    assert_eq!(*spec.default_model, *"gemini-2.0-flash");
    assert_eq!(spec.api_key_env.as_deref(), Some("GEMINI_API_KEY"));
    assert!(spec.endpoint.contains("{model}"));
    assert_eq!(
        llmrace::cost::pricing("gemini", "gemini-2.0-flash"),
        Some((0.10, 0.40))
    );
}

#[test]
fn openai_compat_provider_batch_specs() {
    let cases = [
        ("deepseek", "DEEPSEEK_API_KEY", "deepseek-chat"),
        ("mistral", "MISTRAL_API_KEY", "mistral-small-latest"),
        (
            "together",
            "TOGETHER_API_KEY",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ),
        (
            "fireworks",
            "FIREWORKS_API_KEY",
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
        ),
        ("cerebras", "CEREBRAS_API_KEY", "llama-3.3-70b"),
    ];
    for (id, key_env, model) in cases {
        let spec = llmrace::provider::ProviderSpec::find(id)
            .unwrap_or_else(|| panic!("provider {id} missing from registry"));
        assert_eq!(spec.wire, Wire::OpenAiCompat, "{id}");
        assert_eq!(spec.api_key_env.as_deref(), Some(key_env), "{id}");
        assert_eq!(*spec.default_model, *model, "{id}");
        assert!(spec.endpoint.starts_with("https://"), "{id}");
    }
}

#[test]
fn openai_compat_provider_batch_pricing() {
    assert_eq!(
        llmrace::cost::pricing("deepseek", "deepseek-chat"),
        Some((0.27, 1.10))
    );
    assert_eq!(
        llmrace::cost::pricing("mistral", "mistral-small-latest"),
        Some((0.20, 0.60))
    );
    assert_eq!(
        llmrace::cost::pricing("cerebras", "llama-3.3-70b"),
        Some((0.85, 1.20))
    );
}

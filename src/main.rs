use anyhow::{Context, Result};
use clap::Parser;
use llmrace::bench::{spawn_lane, BenchConfig, LaneSummary};
use llmrace::provider::ProviderSpec;
use llmrace::{report, tui};
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "llmrace",
    version,
    about = "🏁 Race LLM providers head-to-head in your terminal",
    long_about = "llmrace benchmarks streaming latency, throughput and cost across LLM providers (OpenAI, Anthropic, Groq, OpenRouter, local Ollama or any OpenAI-compatible endpoint) with a live TUI race and shareable markdown, JSON and CSV reports."
)]
struct Cli {
    /// Providers to race (see --list). Omit to auto-detect from env vars.
    providers: Vec<String>,

    /// Prompt sent to every provider.
    #[arg(
        short,
        long,
        default_value = "Write one vivid paragraph about a lighthouse at dawn."
    )]
    prompt: String,

    /// Measured rounds per provider.
    #[arg(short = 'n', long, default_value_t = 3)]
    runs: usize,

    /// Warmup rounds excluded from statistics.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Run rounds for this many seconds instead of a fixed count.
    #[arg(long)]
    duration: Option<u64>,

    /// Concurrent requests per provider per round.
    #[arg(short = 'c', long, default_value_t = 1)]
    concurrency: usize,

    /// Model for every provider (overrides defaults).
    #[arg(short, long)]
    model: Option<String>,

    /// Sampling cap passed as max_tokens.
    #[arg(long, default_value_t = 256)]
    max_tokens: u64,

    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 120)]
    timeout: u64,

    /// Register a custom OpenAI-compatible endpoint (e.g. vLLM, LiteLLM).
    #[arg(long, requires = "provider_name")]
    endpoint: Option<String>,

    /// Name for the custom endpoint provider.
    #[arg(long)]
    provider_name: Option<String>,

    /// Env var holding the API key for the custom endpoint.
    #[arg(long)]
    api_key_env: Option<String>,

    /// Disable the live TUI (CI-friendly).
    #[arg(long)]
    no_tui: bool,

    /// Also write a JSON report.
    #[arg(long)]
    json: bool,

    /// Also write a CSV report with per-run rows.
    #[arg(long)]
    csv: bool,

    /// Compare results against a baseline JSON report (CI quality gate).
    #[arg(long)]
    baseline: Option<String>,

    /// Maximum tolerated regression in percent when --baseline is set.
    #[arg(long, default_value_t = 15.0)]
    max_regression: f64,

    /// Output report path (markdown).
    #[arg(short, long, default_value = "llmrace-report.md")]
    out: String,

    /// Load a named profile from llmrace.toml (explicit flags win).
    #[arg(long)]
    profile: Option<String>,

    /// List built-in providers.
    #[arg(long)]
    list: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let (cli, _profile) = apply_profile(cli)?;

    if cli.list {
        let header = format!("{:<14} {:<30} {}\n", "PROVIDER", "DEFAULT MODEL", "NEEDS");
        print!("{header}");
        for p in ProviderSpec::registry() {
            let row = format!(
                "{:<14} {:<30} {}\n",
                p.id,
                p.default_model,
                p.api_key_env.as_deref().unwrap_or("nothing (local)")
            );
            print!("{row}");
        }
        return Ok(());
    }

    let selected = resolve_providers(&cli)?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(cli.timeout.max(1)))
        .build()
        .context("failed to build http client")?;

    let runs = match cli.duration {
        Some(secs) => duration_rounds(&selected, &cli, &client, secs).await,
        None => cli.runs,
    };

    let config = BenchConfig {
        prompt: cli.prompt.clone(),
        max_tokens: cli.max_tokens,
        runs: runs.max(1),
        warmup: cli.warmup,
        concurrency: cli.concurrency.max(1),
        timeout_secs: cli.timeout,
    };

    println!(
        "🏁 llmrace v{} — {} providers × {} rounds ({} warmup, c={})",
        env!("CARGO_PKG_VERSION"),
        selected.len(),
        config.runs,
        config.warmup,
        config.concurrency
    );
    println!("   prompt: \"{}\"\n", cli.prompt.replace('\n', " "));

    let lanes: Vec<(Arc<str>, String)> = selected
        .iter()
        .map(|p| {
            (
                Arc::clone(&p.id),
                cli.model
                    .clone()
                    .unwrap_or_else(|| p.default_model.to_string()),
            )
        })
        .collect();

    let mut handles = Vec::new();
    let mut receivers = Vec::new();
    for p in &selected {
        let model = cli
            .model
            .clone()
            .unwrap_or_else(|| p.default_model.to_string());
        let (handle, rx) = spawn_lane(
            client.clone(),
            p.clone(),
            model,
            BenchConfig { ..config.clone() },
        );
        handles.push(handle);
        receivers.push(rx);
    }

    let summaries: Vec<LaneSummary> = if cli.no_tui {
        drop(receivers);
        let mut all = Vec::new();
        for h in handles {
            match h.await {
                Ok(s) => {
                    print_lane_progress(&s);
                    all.push(s);
                }
                Err(e) => eprintln!("lane task failed: {e}"),
            }
        }
        all
    } else {
        let merged = std::sync::mpsc::channel::<llmrace::bench::LaneUpdate>();
        for rx in receivers {
            let tx = merged.0.clone();
            tokio::spawn(async move {
                while let Ok(u) = rx.recv() {
                    if tx.send(u).is_err() {
                        break;
                    }
                }
            });
        }
        drop(merged.0);
        let lanes_clone = lanes.clone();
        let prompt = cli.prompt.clone();
        let runs_ui = config.runs;
        let rx_final = merged.1;
        let tui = tui::Tui::new(lanes_clone, prompt, runs_ui);
        let tui_handle = tokio::task::spawn_blocking(move || tui.run(rx_final).map(|_| ()));

        let mut all = Vec::new();
        for h in handles {
            match h.await {
                Ok(s) => all.push(s),
                Err(e) => eprintln!("lane task failed: {e}"),
            }
        }
        match tui_handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(e) => return Err(e.into()),
        }
        all
    };

    let table = report::terminal_table(&summaries);
    println!("\n{table}");

    let md = report::markdown(&summaries, &cli.prompt, config.runs, cli.max_tokens);
    std::fs::write(&cli.out, &md).with_context(|| format!("writing {}", cli.out))?;
    println!("📄 markdown report saved to {}", cli.out);

    if cli.json {
        let j = report::json_report(&summaries, &cli.prompt, config.runs, cli.max_tokens);
        let path = report_sibling(&cli.out, "json");
        std::fs::write(&path, j).with_context(|| format!("writing {path}"))?;
        println!("📊 json report saved to {path}");
    }
    if cli.csv {
        let c = report::csv(&summaries);
        let path = report_sibling(&cli.out, "csv");
        std::fs::write(&path, c).with_context(|| format!("writing {path}"))?;
        println!("📈 csv report saved to {path}");
    }

    if let Some(baseline_path) = &cli.baseline {
        let raw = std::fs::read_to_string(baseline_path)
            .with_context(|| format!("reading baseline {baseline_path}"))?;
        let baseline = llmrace::regression::BaselineReport::from_json_str(&raw)
            .with_context(|| format!("parsing baseline {baseline_path}"))?;
        let regression = llmrace::regression::evaluate(&summaries, &baseline, cli.max_regression);
        println!("\n{}", regression.terminal_table(cli.max_regression));
        if regression.any_regressed() {
            std::process::exit(1);
        }
    }

    Ok(())
}

fn report_sibling(out: &str, ext: &str) -> String {
    let stem = out.strip_suffix(".md").unwrap_or(out.trim_end_matches('.'));
    format!("{stem}.{ext}")
}

fn apply_profile(mut cli: Cli) -> Result<(Cli, Option<llmrace::config::Profile>)> {
    let Some(name) = cli.profile.clone() else {
        return Ok((cli, None));
    };
    let config = llmrace::config::Config::discover().with_context(|| "loading llmrace.toml")?;
    let profile = config
        .profile(&name)
        .with_context(|| format!("profile '{name}'"))?
        .clone();

    if cli.providers.is_empty() && !profile.providers.is_empty() {
        cli.providers = profile.providers.clone();
    }
    if cli.model.is_none() {
        cli.model = profile.model.clone();
    }
    if arg_not_passed(&["-p", "--prompt"]) {
        if let Some(prompt) = &profile.prompt {
            cli.prompt = prompt.clone();
        }
    }
    if arg_not_passed(&["-n", "--runs"]) {
        if let Some(v) = profile.runs {
            cli.runs = v;
        }
    }
    if arg_not_passed(&["--warmup"]) {
        if let Some(v) = profile.warmup {
            cli.warmup = v;
        }
    }
    if arg_not_passed(&["-c", "--concurrency"]) {
        if let Some(v) = profile.concurrency {
            cli.concurrency = v;
        }
    }
    if arg_not_passed(&["--max-tokens"]) {
        if let Some(v) = profile.max_tokens {
            cli.max_tokens = v;
        }
    }
    if arg_not_passed(&["--timeout"]) {
        if let Some(v) = profile.timeout {
            cli.timeout = v;
        }
    }
    println!("⚙  profile '{name}' loaded");
    Ok((cli, Some(profile)))
}

fn arg_not_passed(flags: &[&str]) -> bool {
    let args: Vec<String> = std::env::args().collect();
    !args.iter().any(|a| {
        flags
            .iter()
            .any(|f| a == *f || a.starts_with(&format!("{f}=")) || a.starts_with(&format!("{f} ")))
    })
}

async fn duration_rounds(
    _selected: &[ProviderSpec],
    cli: &Cli,
    _client: &reqwest::Client,
    secs: u64,
) -> usize {
    let estimated_round_ms = 2000.0 * f64::from(cli.concurrency.max(1) as u32);
    let by_estimate = ((secs as f64 * 1000.0) / estimated_round_ms).round() as usize;
    by_estimate.max(1)
}

fn resolve_providers(cli: &Cli) -> Result<Vec<ProviderSpec>> {
    let mut out: Vec<ProviderSpec> = Vec::new();
    if let (Some(endpoint), Some(name)) = (&cli.endpoint, &cli.provider_name) {
        let model = cli.model.clone().unwrap_or_else(|| "default".to_string());
        out.push(ProviderSpec::custom(
            name,
            endpoint,
            cli.api_key_env.as_deref(),
            &model,
        ));
    }
    if !cli.providers.is_empty() {
        for name in &cli.providers {
            out.push(
                ProviderSpec::find(name)
                    .with_context(|| format!("unknown provider '{name}' (try --list)"))?,
            );
        }
        return Ok(out);
    }
    if out.is_empty() {
        let available: Vec<ProviderSpec> = ProviderSpec::registry()
            .into_iter()
            .filter(|p| *p.id != *"mock" && p.is_available())
            .collect();
        if available.is_empty() {
            anyhow::bail!(
                "no providers configured; set an API key (e.g. OPENAI_API_KEY), start Ollama, or name providers explicitly"
            );
        }
        return Ok(available);
    }
    Ok(out)
}

fn print_lane_progress(s: &LaneSummary) {
    if let Some(e) = &s.error {
        println!("  ✗ {:<14} {}", s.name, e);
    } else {
        let ci = match s.ci95_total_ms() {
            Some((lo, hi)) => format!("±{:.0}", (hi - lo) / 2.0),
            None => "—".to_string(),
        };
        println!(
            "  ✓ {:<14} {:>6.1} tok/s   {:>6.0} ms {}   TTFB {:>5.0} ms",
            s.name,
            s.mean_tokens_per_sec(),
            s.mean_total_ms(),
            ci,
            s.mean_ttfb_ms()
        );
    }
}

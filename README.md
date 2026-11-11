<div align="center">

# 🏁 llmrace

**Race LLM providers head-to-head in your terminal.**

Latency · Time-to-first-byte · Tokens/s · p95 · Cost — live, in one command.

[![CI](https://img.shields.io/github/actions/workflow/status/dfbustosus/llmrace/ci.yml?style=flat-square&logo=githubactions&label=CI)](https://github.com/dfbustosus/llmrace/actions)
[![Crates.io](https://img.shields.io/crates/v/llmrace?style=flat-square&logo=rust&color=3390FF)](https://crates.io/crates/llmrace)
[![License](https://img.shields.io/crates/l/llmrace?style=flat-square&color=8B949E)](#license)
[![Made with Rust](https://img.shields.io/badge/built%20with-Rust-DEA584?style=flat-square&logo=rust)](https://www.rust-lang.org)

*A single static binary. No telemetry. Works with cloud APIs and local models.*

</div>

---

## Why

Every LLM provider claims to be "fast". **TTFB, streaming throughput and p95 tail latency tell a different story** depending on your region, your model and your prompt length. `llmrace` measures it for real: it fires the *same streaming request* at every provider **simultaneously**, watches the tokens arrive live, and ranks them.

## Demo

```text
 🏁 llmrace   run 1/3 ·  4.2s · "Write one vivid paragraph about a lighthouse at dawn."
┌─ GROQ ────────────────────────────────────────────────────────────────┐
 │ 🏁 412 ms   184 tok  ·  TTFB 88 ms                                   │
 │ `llama-3.3-70b-versatile`                                            │
 │ ████████████████████████████████████████░░░░░░░░░░░░░░░              │
└───────────────────────────────────────────────────────────────────────┘
┌─ OPENAI ──────────────────────────────────────────────────────────────┐
 │ ▶ streaming…   61 tok  ·  TTFB 210 ms                                │
 │ `gpt-4o-mini`                                                        │
 │ ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░             │
└───────────────────────────────────────────────────────────────────────┘
┌─ OLLAMA (local) ──────────────────────────────────────────────────────┐
 │ ▶ streaming…   23 tok  ·  TTFB 41 ms                                 │
 │ `llama3.2`                                                           │
 │ █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░        │
└───────────────────────────────────────────────────────────────────────┘
q/Esc quit · report written on exit
```

## Install

**Prebuilt binary** (no Rust toolchain needed):

```bash
cargo binstall llmrace
```

Or grab a binary for your platform from the [releases page](https://github.com/dfbustosus/llmrace/releases) — Linux (x86_64), macOS (Apple Silicon & Intel), Windows (x86_64), each with SHA-256 checksums.

**From crates.io:**

```bash
cargo install llmrace
```

**From source:**

```bash
git clone https://github.com/dfbustosus/llmrace
cd llmrace && cargo install --path .
```

## Quick start

```bash
# Race every provider you have configured (auto-detected from env vars)
llmrace

# Race specific providers
llmrace openai anthropic groq

# Include your local Ollama models
llmrace ollama groq -m llama-3.3-70b-versatile

# Saturation throughput: 8 parallel requests per provider
llmrace openai groq -c 8 -n 5

# Race ANY OpenAI-compatible endpoint (vLLM, LiteLLM, Together…)
llmrace --endpoint http://localhost:8000/v1/chat/completions \
        --provider-name vllm --api-key-env VLLM_API_KEY \
        openai groq

# Custom prompt, JSON + CSV output for CI
llmrace openai groq -p "Explain transformer attention in 3 sentences." -n 5 --json --csv
```

Every provider with a matching env var races automatically — no config file:

| Provider | Env var | Default model |
|----------|---------|---------------|
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-latest` |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` |
| `ollama` | — (local) | `llama3.2` |
| custom | `--endpoint` + `--api-key-env` | any OpenAI-compatible server |

## What it measures

| Metric | Definition |
|--------|------------|
| **TTFB** | Time from request to first streamed token |
| **Tokens/s** | Completion tokens divided by the *generation window* (total − TTFB); with `--concurrency N`, total tokens of all N parallel requests over the round window — i.e. saturation throughput |
| **Latency p50 / p95** | Distribution of round time-to-last-token (linear-interpolation percentiles) |
| **Mean ± 95% CI** | Student-t confidence interval on the mean latency |
| **Cost/run** | Provider-reported token counts × public per-model prices (editable in `src/cost.rs`) |

Methodology notes:

- One **warmup round** per provider (excluded from stats) to absorb connection setup and TLS handshakes
- Rounds are **concurrent across providers** but sequential within a provider, so providers don't throttle each other
- Rounds with 3σ latency outliers are flagged with `†`
- Token counts prefer **provider-reported usage** (`include_usage` on OpenAI-compatible APIs, `message_delta` on Anthropic) and fall back to stream-event counting
- All requests use the same prompt, `max_tokens` and sampling defaults per run
- Per-request timeouts (`--timeout`) are enforced mid-stream; panics in a single request never abort the race

## Reports

```bash
llmrace openai groq --out race.md --json --csv
```

- `race.md` — ranked table with **mean ± 95% CI**, p50/p95, TTFB, tokens/s and cost, ready to paste into GitHub issues or blog posts
- `race.json` — full per-run data, confidence intervals and outliers for CI pipelines
- `race.csv` — per-run rows for spreadsheets and dashboards

## Using llmrace as a CI quality gate

Record a baseline once, then fail your pipeline when a provider regresses:

```bash
# 1. Record the baseline (commit llmrace-report.json to your repo)
llmrace openai groq --json --out llmrace-report.json

# 2. In CI: race again and compare — exits 1 on regression
llmrace openai groq --baseline llmrace-report.json --max-regression 15
```

The check compares mean latency, p95, TTFB (increases) and tokens/s (decreases) against the threshold, prints an old → new diff table, and flags providers that are new or missing from the baseline.

## Engineering

- `#![deny(missing_docs)]` and `#![deny(clippy::unwrap_used)]` on the library
- Typed errors via `thiserror`; `anyhow` only at the binary edge
- Zero-alignment-allocation SSE parser that survives arbitrary network chunk boundaries (exhaustively tested at every split point)
- Criterion benchmarks for the parsing and statistics hot paths
- Panic containment: a failing request becomes an error row, never a crash
- CI on Linux/macOS/Windows: fmt + clippy `-D warnings` + tests + end-to-end mock race

## Roadmap

Tracked in [GitHub milestones](https://github.com/dfbustosus/llmrace/milestones) — highlights:

- **[v0.3.0](https://github.com/dfbustosus/llmrace/milestone/1)** — regression mode as a CI quality gate, Gemini + DeepSeek/Mistral/Together/Fireworks/Cerebras providers, config profiles, prompt-variant sweeps, watch mode, real-HTTP integration tests
- **[v0.4.0](https://github.com/dfbustosus/llmrace/milestone/2)** — Homebrew tap, code coverage, cargo-deny + dependabot, SSE fuzzing, MSRV enforcement, changelog discipline
- **[v1.0.0](https://github.com/dfbustosus/llmrace/milestone/3)** — persistent history + `llmrace trend`, community health files, stability guarantees

Most provider additions are tagged [`good first issue`](https://github.com/dfbustosus/llmrace/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) — grab one.

## Contributing

PRs welcome — adding a provider is ~40 lines (see `src/provider.rs`). Fork it, add your favorite provider, ship it.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

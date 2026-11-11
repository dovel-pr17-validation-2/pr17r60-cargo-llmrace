# Contributing to llmrace

Thanks for considering a contribution!

## Adding a provider

Providers live in `src/provider.rs`. To add one:

1. Add a constructor (e.g. `Provider::myprovider()`) with the endpoint, env var name, default model and wire protocol. Any OpenAI-compatible API reuses `Wire::OpenAiCompat`.
2. Push it onto `Provider::registry()`.
3. If it has public per-token prices, add a row to the table in `src/cost.rs`.
4. Add or extend a test. Run `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`.

That's it — usually under 40 lines.

## Bug reports

Include the provider, model, OS, and the output of `llmrace --version` plus the error line from the lane.

## Style

- `cargo fmt` formatting, no unwraps in library paths, errors via `anyhow` at the binary edge.
- Keep the TUI dependency-light: `ratatui` + `crossterm` only.

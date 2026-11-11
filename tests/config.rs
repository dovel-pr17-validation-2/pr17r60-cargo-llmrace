use llmrace::config::Config;

const SAMPLE: &str = r#"
[profiles.fast]
providers = ["openai", "groq"]
prompt = "Reply with a single word."
runs = 1
warmup = 0
concurrency = 2

[profiles.overnight]
providers = ["openai", "anthropic", "groq", "ollama"]
runs = 20
warmup = 2
max_tokens = 1024
timeout = 300
"#;

#[test]
fn parses_profiles_and_fields() {
    let config = Config::from_toml_str(SAMPLE).expect("parse");
    assert_eq!(config.profiles.len(), 2);

    let fast = config.profile("fast").expect("fast");
    assert_eq!(fast.providers, vec!["openai", "groq"]);
    assert_eq!(fast.prompt.as_deref(), Some("Reply with a single word."));
    assert_eq!(fast.runs, Some(1));
    assert_eq!(fast.warmup, Some(0));
    assert_eq!(fast.concurrency, Some(2));
    assert_eq!(fast.model, None);

    let overnight = config.profile("overnight").expect("overnight");
    assert_eq!(overnight.runs, Some(20));
    assert_eq!(overnight.max_tokens, Some(1024));
    assert_eq!(overnight.timeout, Some(300));
    assert!(overnight.prompt.is_none());
}

#[test]
fn missing_profile_lists_available() {
    let config = Config::from_toml_str(SAMPLE).expect("parse");
    let err = config.profile("nope").expect_err("should fail");
    let msg = err.to_string();
    assert!(msg.contains("nope"));
    assert!(msg.contains("fast"));
    assert!(msg.contains("overnight"));
}

#[test]
fn malformed_toml_is_a_config_error() {
    let err = Config::from_toml_str("this is not = = toml").expect_err("should fail");
    assert!(err.to_string().contains("invalid llmrace.toml"));
}

#[test]
fn empty_config_has_no_profiles() {
    let config = Config::default();
    assert_eq!(config.available(), "");
}

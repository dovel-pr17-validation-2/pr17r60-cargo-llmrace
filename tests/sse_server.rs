use futures_util::StreamExt;
use llmrace::bench::{spawn_lane, BenchConfig};
use llmrace::provider::{stream_completion, CompletionRequest, ProviderSpec, Wire};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

enum Script {
    Sse {
        events: Vec<String>,
        chunk_size: usize,
    },
    Error {
        status: u16,
        body: String,
    },
}

async fn spawn_server(script: Script) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        loop {
            let n = socket.read(&mut tmp).await.expect("read request");
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(header_end) = find_header_end(&buf) {
                let headers = String::from_utf8_lossy(&buf[..header_end]).to_ascii_lowercase();
                if let Some(cl) = content_length(&headers) {
                    if buf.len() >= header_end + 4 + cl {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        match script {
            Script::Sse { events, chunk_size } => {
                socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
                    )
                    .await
                    .expect("write headers");
                let payload: String = events.join("\n\n") + "\n\n";
                let bytes = payload.as_bytes();
                for chunk in bytes.chunks(chunk_size.max(1)) {
                    socket.write_all(chunk).await.expect("write chunk");
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
            }
            Script::Error { status, body } => {
                let head = format!(
                    "HTTP/1.1 {status} Err\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                socket
                    .write_all(head.as_bytes())
                    .await
                    .expect("write error head");
                socket
                    .write_all(body.as_bytes())
                    .await
                    .expect("write error body");
            }
        }
        socket.shutdown().await.ok();
    });
    (format!("http://{addr}/v1/chat/completions"), handle)
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn content_length(headers: &str) -> Option<usize> {
    headers
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse().ok())
}

fn openai_events() -> Vec<String> {
    vec![
        r#"data: {"choices":[{"delta":{"content":"Hel"}}]}"#.into(),
        r#"data: {"choices":[{"delta":{"content":"lo "}}]}"#.into(),
        r#"data: {"choices":[{"delta":{"content":"world"}}]}"#.into(),
        r#"data: {"choices":[{"delta":{"content":"!"}}]}"#.into(),
        r#"data: {"choices":[{"delta":{}}],"usage":{"completion_tokens":4}}"#.into(),
        "data: [DONE]".into(),
    ]
}

#[tokio::test]
async fn full_http_path_streams_and_counts_tokens() {
    let events = openai_events();
    let (endpoint, server) = spawn_server(Script::Sse {
        events,
        chunk_size: 1024,
    })
    .await;
    let spec = ProviderSpec::custom("testup", &endpoint, None, "test-model");

    let (handle, _progress) = spawn_lane(
        reqwest::Client::new(),
        spec,
        "test-model".into(),
        BenchConfig {
            prompt: "hi".into(),
            runs: 1,
            warmup: 0,
            ..BenchConfig::default()
        },
    );
    let summary = handle.await.expect("lane task");
    server.await.expect("server task");
    assert!(summary.successful(), "error: {:?}", summary.error);
    assert_eq!(summary.runs.len(), 1);
    assert_eq!(summary.runs[0].tokens, 4);
    assert!(summary.runs[0].ttfb_ms > 0.0);
    assert!(summary.runs[0].total_ms >= summary.runs[0].ttfb_ms);
}

#[tokio::test]
async fn irregular_chunk_boundaries_survive_the_full_stack() {
    let events = openai_events();
    let (endpoint, server) = spawn_server(Script::Sse {
        events,
        chunk_size: 3,
    })
    .await;
    let spec = ProviderSpec::custom("testup", &endpoint, None, "test-model");

    let mut stream = stream_completion(
        &reqwest::Client::new(),
        &spec,
        &CompletionRequest {
            model: "test-model",
            prompt: "hi",
            max_tokens: 16,
        },
    )
    .await
    .expect("stream start");
    let mut tokens = 0u64;
    let mut usage = None;
    while let Some(ev) = stream.next().await {
        let ev = ev.expect("event");
        if ev.token.is_some() {
            tokens += 1;
        }
        if let Some(n) = ev.completion_tokens {
            usage = Some(n);
        }
    }
    server.await.expect("server task");
    assert_eq!(tokens, 4);
    assert_eq!(usage, Some(4));
}

#[tokio::test]
async fn http_error_surfaces_as_lane_error() {
    let body = r#"{"error":{"message":"rate limited","type":"rate_limit"}}"#.to_string();
    let (endpoint, server) = spawn_server(Script::Error { status: 429, body }).await;
    let spec = ProviderSpec::custom("testup", &endpoint, None, "test-model");

    let (handle, _progress) = spawn_lane(
        reqwest::Client::new(),
        spec,
        "test-model".into(),
        BenchConfig {
            prompt: "hi".into(),
            runs: 1,
            warmup: 0,
            ..BenchConfig::default()
        },
    );
    let summary = handle.await.expect("lane task");
    server.await.expect("server task");
    assert!(!summary.successful());
    let err = summary.error.expect("lane error");
    assert!(err.contains("HTTP 429"), "unexpected error: {err}");
    assert!(
        err.contains("rate limited"),
        "error should include body: {err}"
    );
}

#[test]
fn wire_of_custom_provider_is_openai_compat() {
    let spec = ProviderSpec::custom("x", "http://localhost:9/v1/chat/completions", None, "m");
    assert_eq!(spec.wire, Wire::OpenAiCompat);
}

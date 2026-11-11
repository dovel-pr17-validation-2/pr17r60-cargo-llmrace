use criterion::{black_box, criterion_group, criterion_main, Criterion};
use llmrace::sse::SseParser;
use llmrace::stats::Sample;

fn realistic_stream(n_events: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    for i in 0..n_events {
        bytes.extend_from_slice(
            format!(
                "event: chunk\ndata: {{\"id\":\"x\",\"choices\":[{{\"delta\":{{\"content\":\"token {i}\"}}}}]}}\n\n"
            )
            .as_bytes(),
        );
    }
    bytes
}

fn bench_sse_feed_7byte_chunks(c: &mut Criterion) {
    let stream = realistic_stream(200);
    c.bench_function("sse_feed_200_events_7b_chunks", |b| {
        b.iter(|| {
            let mut parser = SseParser::new();
            let mut blocks = 0usize;
            for chunk in stream.chunks(7) {
                blocks += black_box(parser.feed(black_box(chunk))).len();
            }
            blocks += parser.finish().len();
            black_box(blocks)
        })
    });
}

fn bench_percentile_10k(c: &mut Criterion) {
    let values: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.37) % 512.0).collect();
    let sample = Sample::from_values(values);
    c.bench_function("percentile_p95_10k", |b| {
        b.iter(|| black_box(sample.percentile(black_box(95.0))))
    });
}

criterion_group!(benches, bench_sse_feed_7byte_chunks, bench_percentile_10k);
criterion_main!(benches);

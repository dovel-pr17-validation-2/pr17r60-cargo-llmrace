//! Server-Sent Events (SSE) parsing for streaming LLM APIs.
//!
//! The [`SseParser`] accumulates raw bytes and yields complete event blocks as
//! they arrive, tolerating arbitrary chunk boundaries from the network layer.

/// An incremental SSE byte-stream parser.
///
/// Feed it raw response bytes as they arrive; it returns every complete
/// event block (the text between blank-line separators). Call [`SseParser::finish`]
/// once the stream ends to flush any trailing block that was not terminated
/// by a blank line.
#[derive(Debug, Default)]
pub struct SseParser {
    buf: Vec<u8>,
}

impl SseParser {
    /// Creates an empty parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds raw bytes and returns all event blocks completed by this write.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut blocks = Vec::new();
        while let Some(pos) = find_separator(&self.buf) {
            let sep_len = if self.buf[pos..].starts_with(b"\r\n\r\n") {
                4
            } else {
                2
            };
            let block: Vec<u8> = self.buf.drain(..pos + sep_len).collect();
            if let Some(text) = block_text(&block) {
                blocks.push(text);
            }
        }
        blocks
    }

    /// Flushes and returns a trailing unterminated block, if any.
    pub fn finish(&mut self) -> Vec<String> {
        if self.buf.is_empty() {
            return Vec::new();
        }
        let block = std::mem::take(&mut self.buf);
        match block_text(&block) {
            Some(text) => vec![text],
            None => Vec::new(),
        }
    }
}

fn block_text(block: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(block);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn find_separator(buf: &[u8]) -> Option<usize> {
    if buf.len() < 2 {
        return None;
    }
    let mut i = 0;
    while i + 1 < buf.len() {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some(i);
        }
        if buf[i] == b'\r' && buf[i..].starts_with(b"\r\n\r\n") {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Extracts the `data:` payload strings from a complete event block, zero-copy.
///
/// Lines that are not `data:` lines (event names, ids, comments) are ignored.
pub fn parse_data_events(block: &str) -> Vec<&str> {
    block
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            line.strip_prefix("data:").map(|rest| rest.trim_start())
        })
        .collect()
}

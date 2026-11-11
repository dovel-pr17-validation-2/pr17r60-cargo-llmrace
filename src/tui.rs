//! Live terminal race view.

use crate::bench::LaneUpdate;
use ratatui::{
    crossterm::event::{self, Event, KeyCode},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Sparkline},
    Frame,
};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

const REDRAW_TICK: Duration = Duration::from_millis(250);
const SPARK_POINTS: usize = 40;

static RESTORE_GUARD: OnceLock<()> = OnceLock::new();

fn restore_once() {
    if RESTORE_GUARD.set(()).is_err() {
        return;
    }
    ratatui::restore();
}

struct TerminalGuard;

impl TerminalGuard {
    fn install() -> Self {
        TerminalGuard
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_once();
    }
}

#[derive(Clone)]
struct Lane {
    name: Arc<str>,
    model: String,
    tokens: u64,
    elapsed_ms: f64,
    ttfb_ms: Option<f64>,
    finished: bool,
    error: Option<String>,
    best_ms: Option<f64>,
    rate_history: Vec<f64>,
    last_tokens: u64,
    last_sample: Instant,
}

impl Lane {
    fn new(name: Arc<str>, model: String) -> Self {
        Lane {
            name,
            model,
            tokens: 0,
            elapsed_ms: 0.0,
            ttfb_ms: None,
            finished: false,
            error: None,
            best_ms: None,
            rate_history: Vec::new(),
            last_tokens: 0,
            last_sample: Instant::now(),
        }
    }
}

#[derive(Clone)]
struct RaceState {
    lanes: Vec<Lane>,
    prompt: String,
    runs: usize,
    started: Instant,
    all_done: bool,
}

/// Terminal user interface driving the live race.
pub struct Tui {
    state: RaceState,
}

impl Tui {
    /// Creates a TUI for the given lanes.
    pub fn new(lanes: Vec<(Arc<str>, String)>, prompt: String, runs: usize) -> Self {
        Tui {
            state: RaceState {
                lanes: lanes
                    .into_iter()
                    .map(|(name, model)| Lane::new(name, model))
                    .collect(),
                prompt,
                runs,
                started: Instant::now(),
                all_done: false,
            },
        }
    }

    /// Applies one progress update.
    pub fn update(&mut self, u: LaneUpdate) {
        if let Some(lane) = self.state.lanes.iter_mut().find(|l| *l.name == *u.provider) {
            if u.finished {
                lane.finished = true;
                if let Some(e) = u.error {
                    lane.error = Some(e);
                } else if u.elapsed_ms > 0.0 {
                    lane.best_ms = Some(lane.best_ms.map_or(u.elapsed_ms, |b| b.min(u.elapsed_ms)));
                }
            } else {
                lane.tokens = u.tokens;
                lane.elapsed_ms = u.elapsed_ms;
                lane.ttfb_ms = u.ttfb_ms.or(lane.ttfb_ms);
            }
        }
        self.state.all_done = self.state.lanes.iter().all(|l| l.finished);
    }

    fn sample_rates(&mut self) {
        for lane in self.state.lanes.iter_mut() {
            let elapsed_s = lane.last_sample.elapsed().as_secs_f64();
            if elapsed_s < 0.1 {
                continue;
            }
            let delta = lane.tokens.saturating_sub(lane.last_tokens);
            let rate = delta as f64 / elapsed_s;
            lane.rate_history.push(rate);
            if lane.rate_history.len() > SPARK_POINTS {
                lane.rate_history.remove(0);
            }
            lane.last_tokens = lane.tokens;
            lane.last_sample = Instant::now();
        }
    }

    /// Runs the event loop until the user quits or all lanes finish.
    ///
    /// The screen is only redrawn when new updates arrived or a 250 ms tick
    /// elapsed, and terminal state is restored even when rendering fails.
    pub fn run(mut self, updates: std::sync::mpsc::Receiver<LaneUpdate>) -> anyhow::Result<Self> {
        let mut terminal = ratatui::init();
        let _guard = TerminalGuard::install();
        let mut last_draw = Instant::now() - REDRAW_TICK;
        loop {
            let mut received = false;
            while let Ok(u) = updates.try_recv() {
                self.update(u);
                received = true;
            }

            if received || last_draw.elapsed() >= REDRAW_TICK {
                self.sample_rates();
                terminal.draw(|f| self.render(f))?;
                last_draw = Instant::now();
            }

            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                        return Ok(self);
                    }
                }
            }
            if self.state.all_done {
                let deadline = Instant::now() + Duration::from_secs(30);
                loop {
                    if event::poll(Duration::from_millis(200))? {
                        if let Event::Key(key) = event::read()? {
                            if matches!(
                                key.code,
                                KeyCode::Char('q') | KeyCode::Esc | KeyCode::Enter
                            ) {
                                return Ok(self);
                            }
                        }
                    }
                    if Instant::now() > deadline {
                        return Ok(self);
                    }
                }
            }
        }
    }

    fn render(&self, f: &mut Frame) {
        use ratatui::layout::{Constraint, Layout};

        let chunks = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(f.area());

        let title = Paragraph::new(Line::from(vec![
            Span::styled(
                " 🏁 llmrace ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                format!(
                    "{} rounds · {:>4.1}s · {}",
                    self.state.runs,
                    self.state.started.elapsed().as_secs_f32(),
                    truncate_str(self.state.prompt.trim(), 48)
                ),
                Style::default().fg(Color::DarkGray),
            ),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        f.render_widget(title, chunks[0]);

        let lane_height = 5;
        let inner = Layout::vertical(
            self.state
                .lanes
                .iter()
                .map(|_| Constraint::Length(lane_height)),
        )
        .split(chunks[1]);

        for (i, lane) in self.state.lanes.iter().enumerate() {
            let accent = if lane.error.is_some() {
                Color::Red
            } else if lane.finished {
                Color::Green
            } else {
                Color::Cyan
            };
            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(accent))
                .title(Span::styled(
                    format!(" {} ", lane.name.to_uppercase()),
                    Style::default().fg(accent).add_modifier(Modifier::BOLD),
                ));
            let rows = Layout::vertical([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(2),
            ])
            .split(block.inner(inner[i]));

            let status = if let Some(e) = &lane.error {
                format!(" ✗ {}", truncate_str(e, 42))
            } else if lane.finished {
                match lane.best_ms {
                    Some(ms) => format!(" 🏁 {:.0} ms", ms),
                    None => " 🏁 done".to_string(),
                }
            } else {
                " ▶ streaming…".to_string()
            };

            let meta = Line::from(vec![
                Span::styled(
                    status,
                    Style::default().fg(if lane.error.is_some() {
                        Color::Red
                    } else {
                        Color::Green
                    }),
                ),
                Span::styled(
                    format!("  {} tok", lane.tokens),
                    Style::default().fg(Color::White),
                ),
                Span::styled(
                    format!(
                        "  ·  TTFB {} ms",
                        lane.ttfb_ms
                            .map(|t| format!("{t:.0}"))
                            .unwrap_or("…".into())
                    ),
                    Style::default().fg(Color::DarkGray),
                ),
            ]);
            f.render_widget(Paragraph::new(meta), rows[0]);
            f.render_widget(
                Paragraph::new(Span::styled(
                    format!("`{}`", lane.model),
                    Style::default().fg(Color::DarkGray),
                )),
                rows[1],
            );

            let bars: Vec<u64> = lane.rate_history.iter().map(|r| r.round() as u64).collect();
            let spark = Sparkline::default()
                .data(&bars)
                .style(Style::default().fg(Color::LightYellow));
            f.render_widget(spark, rows[2]);
            f.render_widget(block, inner[i]);
        }

        let footer = Paragraph::new(Line::from(vec![
            Span::styled("q/Esc", Style::default().fg(Color::Yellow)),
            Span::styled(
                " quit · report written on exit",
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        f.render_widget(footer, chunks[2]);
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}…")
    }
}

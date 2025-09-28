use std::{str::FromStr, time::Duration};

use color_eyre::{Result, eyre::eyre};
use flume::Sender;
use ratatui::crossterm::event::{self, KeyCode, KeyEvent, KeyEventKind};
use ratatui::{
    DefaultTerminal,
    prelude::*,
    widgets::{Paragraph, Tabs},
};
use tui_input::{Input, backend::crossterm::EventHandler};
use tui_logger::TuiLoggerSmartWidget;

pub struct State {
    logs_open: bool,
    event_tx: Sender<AppEvent>,
    error: Option<color_eyre::Report>,
    cursor_position: Option<Position>,
    mode: Mode,
    exit: bool,
}

impl State {
    pub fn with_mode(self, mode: Mode) -> Self {
        State { mode, ..self }
    }
}

#[derive(derive_more::Display)]
pub enum Mode {
    #[display("NORM")]
    Normal,
    #[display("CMD")]
    Command { input: Input },
}

pub enum AppEvent {
    None,
    Key(KeyEvent),
    MoveCursor(Position),
}

macro_rules! key {
    // Variant with no arguments, e.g. Enter, Esc
    ($code:ident, $kind:ident) => {
        KeyEvent {
            code: KeyCode::$code,
            kind: KeyEventKind::$kind,
            ..
        }
    };
    // Variant with arguments, e.g. Char('x'), Char(_)
    ($code:ident ( $($arg:tt)* ), $kind:ident) => {
        KeyEvent {
            code: KeyCode::$code($($arg)*),
            kind: KeyEventKind::$kind,
            ..
        }
    };
}

impl State {
    #[must_use]
    fn handle_event(self, event: AppEvent) -> State {
        match (&self.mode, event) {
            // pressing ':' enters command mode
            (Mode::Normal, AppEvent::Key(key!(Char(_), Press))) => State {
                mode: Mode::Command {
                    input: Input::new("".to_string()),
                },
                ..self
            },

            (Mode::Command { input }, AppEvent::Key(key!(Enter, Press))) => {
                let command = Command::from_str(input.value());
                match command {
                    Ok(command) => command.reduce(self).with_mode(Mode::Normal),
                    Err(err) => State {
                        mode: Mode::Normal,
                        error: Some(err),
                        ..self
                    },
                }
            }
            (Mode::Command { .. }, AppEvent::Key(key!(Esc, Press))) => self.with_mode(Mode::Normal),
            (Mode::Command { input }, AppEvent::Key(key_event)) => {
                let mut input = input.clone();
                input.handle_event(&event::Event::Key(key_event));
                self.with_mode(Mode::Command { input })
            }

            (_, AppEvent::MoveCursor(pos)) => Self {
                cursor_position: Some(pos),
                ..self
            },

            _ => self,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Command {
    Quit,
}

impl Command {
    pub fn reduce(self, state: State) -> State {
        match (&state, self) {
            (_, Command::Quit) => State {
                exit: true,
                ..state
            },
        }
    }
}

impl FromStr for Command {
    type Err = color_eyre::Report;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "q" | "quit" => Ok(Self::Quit),
            _ => Err(eyre!("not a command")),
        }
    }
}

pub fn run(mut terminal: DefaultTerminal) -> Result<()> {
    // for later
    let (tx, rx) = flume::unbounded::<AppEvent>();

    let mut state = State {
        logs_open: true,
        error: None,
        event_tx: tx.clone(),
        mode: Mode::Normal,
        exit: false,
        cursor_position: None,
    };

    loop {
        if let Ok(event) = input() {
            state = state.handle_event(event);
        }
        if let Ok(event) = rx.try_recv() {
            state = state.handle_event(event);
        }

        if state.exit {
            return Ok(());
        }

        terminal.draw(|frame| {
            if let Some(cursor_position) = state.cursor_position {
                frame.set_cursor_position(cursor_position);
                state.cursor_position = None;
            }
            render(frame, &state)
        })?;

        // 60fps (i think)
        std::thread::sleep(Duration::from_millis(16));
    }
}

pub fn render(frame: &mut Frame, state: &State) {
    frame.render_widget(MainWidget { app_state: state }, frame.area());
}

fn input() -> Result<AppEvent> {
    let true = event::poll(Duration::from_secs(0))? else {
        return Ok(AppEvent::None);
    };
    let event = event::read()?;

    match event {
        event::Event::FocusGained => Ok(AppEvent::None),
        event::Event::FocusLost => Ok(AppEvent::None),
        event::Event::Key(key_event) => Ok(AppEvent::Key(key_event)),
        event::Event::Mouse(_) => Ok(AppEvent::None),
        event::Event::Paste(_) => Ok(AppEvent::None),
        event::Event::Resize(_, _) => Ok(AppEvent::None),
    }
}

pub struct MainWidget<'a> {
    app_state: &'a State,
}

impl<'a> Widget for MainWidget<'a> {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        let bottom_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(2)])
            .spacing(1)
            .split(area);

        let layout = match self.app_state.logs_open {
            true => Layout::horizontal([Constraint::Min(100), Constraint::Fill(1)]),
            false => todo!(),
        };
        let layout = layout.split(bottom_layout[0]);

        TuiLoggerSmartWidget::default()
            .style_error(Style::default().fg(Color::Red))
            .style_debug(Style::default().fg(Color::Green))
            .style_warn(Style::default().fg(Color::Yellow))
            .style_trace(Style::default().fg(Color::Magenta))
            .style_info(Style::default().fg(Color::Cyan))
            .output_target(true)
            .output_file(true)
            .output_line(true)
            .render(layout[0], buf);
        Tabs::new(vec!["Mem"]);

        match &self.app_state.mode {
            Mode::Normal => {
                Line::from(vec![Span::from(format!(" {}", self.app_state.mode))])
                    .render(bottom_layout[1], buf);
            }
            Mode::Command { input } => {
                let text = format!(" CMD: {}", input.value());
                _ = self
                    .app_state
                    .event_tx
                    .send(AppEvent::MoveCursor(Position::new(
                        bottom_layout[1].x + text.len() as u16,
                        bottom_layout[1].y,
                    )))
                    .inspect_err(|err| tracing::error!(%err));

                let text_area =
                    Paragraph::new(text).style(Style::new().add_modifier(Modifier::RAPID_BLINK));
                text_area.render(bottom_layout[1], buf);
            }
        }
    }
}

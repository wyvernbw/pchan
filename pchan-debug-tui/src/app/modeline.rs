use std::str::FromStr;

use color_eyre::eyre::{Result, eyre};

use ratatui::crossterm::event;
use ratatui::prelude::*;
use ratatui::widgets::Paragraph;
use tui_input::Input;
use tui_input::backend::crossterm::EventHandler;

use crate::app::Component;
use crate::key;

pub struct Modeline;
pub enum Mode {
    Normal,
    Command(ModeCommandState),
}
pub struct ModeCommandState {
    pub input: Input,
}
pub struct ModelineState {
    pub mode: Mode,
}
pub enum Command {
    None,
    Quit,
}

impl FromStr for Command {
    type Err = color_eyre::Report;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "q" | "quit" => Ok(Command::Quit),
            _ => Err(eyre!("invalid command")),
        }
    }
}

impl StatefulWidget for Modeline {
    type State = ModelineState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        let [_, bottom] = Layout::vertical([Constraint::Min(1), Constraint::Max(1)]).areas(area);
        match &state.mode {
            Mode::Normal => {
                Paragraph::new("  NORM  ").render(bottom, buf);
            }
            Mode::Command(mode_command_state) => {
                let text = format!("  CMD: {}  ", mode_command_state.input.value());
                Paragraph::new(text).render(bottom, buf);
            }
        }
    }
}

impl Component for Modeline {
    type ComponentState = ModelineState;
    type ComponentSummary = Command;

    fn handle_input(&mut self, event: event::Event, state: &mut ModelineState) -> Result<Command> {
        if let Mode::Command(state) = &mut state.mode {
            state.input.handle_event(&event);
        }

        if let event::Event::Key(key_event) = event {
            match (&mut state.mode, key_event) {
                (Mode::Normal, key!(Char(':'), Press)) => {
                    state.mode = Mode::Command(ModeCommandState {
                        input: Input::default(),
                    });
                }
                (Mode::Command(cmd), key!(Enter, Press)) => {
                    let cmd = cmd.input.value();
                    let cmd = Command::from_str(cmd);
                    state.mode = Mode::Normal;
                    return cmd;
                }
                (Mode::Command(_), key!(Esc, Press)) => {
                    state.mode = Mode::Normal;
                }
                _ => {}
            }
        };

        Ok(Command::None)
    }
}

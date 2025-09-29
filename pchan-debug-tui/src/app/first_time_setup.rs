use std::path::PathBuf;

use color_eyre::eyre::Result;

use ratatui::crossterm::event;
use ratatui::layout::Flex;
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};
use tui_input::Input;
use tui_input::backend::crossterm::EventHandler;

use crate::app::{Component, TypingAction};
use crate::key;

pub struct FirstTimeSetup;
#[derive(Default)]
pub struct FirstTimeSetupState {
    pub bios_path_input: Input,
    pub bios_path_input_active: bool,
}

impl StatefulWidget for FirstTimeSetup {
    type State = FirstTimeSetupState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        let [area] = Layout::horizontal([Constraint::Ratio(1, 3)])
            .flex(Flex::Center)
            .areas(area);
        let [header, bios_path_input, bios_path_input_help] = Layout::vertical([
            Constraint::Max(4),
            Constraint::Length(3),
            Constraint::Max(3),
        ])
        .flex(Flex::Center)
        .areas(area);

        Paragraph::new(
            "Welcome to the P-chan debugger!\n\nPlease provide the information required to run the debugger.",
        )
        .wrap(Wrap { trim: false })
        .centered()
        .render(header, buf);

        Paragraph::new(state.bios_path_input.value())
            .block(
                Block::bordered()
                    .title("path to bios file:")
                    .border_type(BorderType::Rounded)
                    .border_style(match state.bios_path_input_active {
                        true => Style::new().green(),
                        false => Style::new(),
                    }),
            )
            .render(bios_path_input, buf);

        Paragraph::new("<Esc> cancel input\n<Enter> start typing")
            .centered()
            .render(bios_path_input_help, buf);
    }
}

impl Component for FirstTimeSetup {
    type ComponentState = FirstTimeSetupState;

    type ComponentSummary = TypingAction<PathBuf>;

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary> {
        state.bios_path_input_active = true;
        state.bios_path_input.handle_event(&event);
        if let event::Event::Key(key) = event {
            match key {
                key!(Enter, Press) if !state.bios_path_input.value().is_empty() => {
                    state.bios_path_input_active = false;
                    let path = state.bios_path_input.value().to_string();
                    return Ok(TypingAction::Submit(path.into()));
                }
                key!(Esc, Press) => {
                    state.bios_path_input_active = false;
                    return Ok(TypingAction::Escape);
                }
                _ => {}
            }
        }
        Ok(TypingAction::Pending)
    }
}

use std::path::PathBuf;

use color_eyre::eyre::Result;

use ratatui::crossterm::event;
use ratatui::layout::Flex;
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};
use tui_input::Input;
use tui_input::backend::crossterm::EventHandler;

use crate::app::focus::{Focus, FocusProp};
use crate::app::{Component, TypingAction};
use crate::key;

pub struct FirstTimeSetup(pub FocusProp<Self>);
#[derive(Default)]
pub struct FirstTimeSetupState {
    pub bios_path_input: Input,
}

impl Focus for FirstTimeSetup {}

impl Component for FirstTimeSetup {
    type ComponentState = FirstTimeSetupState;

    type ComponentSummary = TypingAction<PathBuf>;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()> {
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
                    .border_style(self.0.style()),
            )
            .render(bios_path_input, buf);

        Paragraph::new("<Esc> cancel input\n<Enter> start typing")
            .centered()
            .render(bios_path_input_help, buf);
        Ok(())
    }

    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
        _: Rect,
    ) -> Result<Self::ComponentSummary> {
        if self.0.is_focused() {
            state.bios_path_input.handle_event(&event);
            if let event::Event::Key(key) = event {
                match key {
                    key!(Enter, Press) if !state.bios_path_input.value().is_empty() => {
                        let path = state.bios_path_input.value().to_string();
                        return Ok(TypingAction::Submit(path.into()));
                    }
                    key!(Esc, Press) => {
                        self.0.unfocus();
                        return Ok(TypingAction::Escape);
                    }
                    _ => {}
                }
            }
        } else if let event::Event::Key(key!(Enter, Press)) = event {
            self.0.push_focus();
            return Ok(TypingAction::Enter);
        }
        Ok(TypingAction::Pending)
    }
}

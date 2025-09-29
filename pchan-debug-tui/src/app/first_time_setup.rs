use ratatui::widgets::{StatefulWidget, Widget};
use tui_input::Input;

use crate::app::State;

pub struct FirstTimeSetup<'a> {
    pub app_state: &'a State,
    bios_path: Input,
}

pub struct FirstTimeSetupState {}

impl<'a> Widget for FirstTimeSetup<'a> {
    fn render(self, area: ratatui::prelude::Rect, buf: &mut ratatui::prelude::Buffer) {
        todo!()
    }
}

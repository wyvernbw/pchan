use crossterm::event::MouseEvent;
use ratatui::layout::Rect;

pub mod button;
pub mod checkbox;

pub enum EventResponse {
    Next,
    None,
    GrabFocus,
}

pub trait MouseEventExt {
    fn is_inside(&self, area: Rect) -> bool;
}

impl MouseEventExt for MouseEvent {
    fn is_inside(&self, area: Rect) -> bool {
        area.contains(ratatui::layout::Position {
            x: self.column,
            y: self.row,
        })
    }
}

use std::{borrow::Cow, marker::PhantomData};

use crossterm::event::Event;
use crossterm_simple_event::CrosstermSimpleEvent;
use ratatui::{prelude::*, style::Styled};

use crate::widgets::EventResponse;

#[derive(Default)]
pub struct CheckboxState {
    toggled: bool,
    area:    Rect,
}

pub struct Checkbox<'a, S: Into<Cow<'a, str>>> {
    text:  S,
    style: Style,
    _s:    PhantomData<&'a ()>,
}

impl<'a, S: Into<Cow<'a, str>>> Checkbox<'a, S> {
    pub fn new(text: S) -> Self {
        Self {
            text,
            _s: PhantomData,
            style: Style::default(),
        }
    }
}

impl<'a, S: Into<Cow<'a, str>>> StatefulWidget for Checkbox<'a, S> {
    type State = CheckboxState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        state.area = area;
        let [check_area, _, text_area] = Layout::horizontal([
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Fill(1),
        ])
        .areas(area);
        let check = match state.toggled {
            true => "[✦]",
            false => "[ ]",
        };
        check.set_style(self.style).render(check_area, buf);
        let text = self.text.into();
        text.set_style(self.style).render(text_area, buf);
    }
}

impl CheckboxState {
    pub fn handle_event(&mut self, ev: &Event) -> EventResponse {
        match ev {
            Event::Key(_) => match ev.simple().as_str() {
                "enter" => {
                    self.toggled = !self.toggled;
                    EventResponse::None
                }
                "tab" => EventResponse::Next,
                _ => EventResponse::None,
            },
            Event::Mouse(mouse_event) => {
                if !self.area.contains(Position {
                    x: mouse_event.column,
                    y: mouse_event.row,
                }) {
                    return EventResponse::None;
                }
                match mouse_event.kind {
                    crossterm::event::MouseEventKind::Down(_) => {
                        self.toggled = !self.toggled;
                        EventResponse::GrabFocus
                    }
                    _ => EventResponse::None,
                }
            }
            _ => EventResponse::None,
        }
    }

    pub fn value(&self) -> bool {
        self.toggled
    }
}

impl<'a, T: Into<Cow<'a, str>>> Styled for Checkbox<'a, T> {
    type Item = Self;

    fn style(&self) -> Style {
        self.style
    }

    fn set_style<S: Into<Style>>(mut self, style: S) -> Self::Item {
        self.style = style.into();
        self
    }
}

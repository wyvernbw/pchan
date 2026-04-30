use crossterm::event::{Event, MouseButton, MouseEventKind};
use ratatui::{
    prelude::*,
    style::Styled,
    widgets::{Block, BorderType},
};
use std::borrow::Cow;

use crate::widgets::MouseEventExt;

pub struct Button<'a> {
    text:   Cow<'a, str>,
    block:  Block<'a>,
    styles: ButtonStyles,
}

#[derive(Default)]
pub struct ButtonStyles {
    pub normal:  Style,
    pub pressed: Style,
}

impl<'a> Default for Button<'a> {
    fn default() -> Self {
        Self {
            text:   Default::default(),
            block:  Block::bordered().border_type(BorderType::Rounded),
            styles: ButtonStyles::default(),
        }
    }
}

impl<'a> Button<'a> {
    pub fn new(text: impl Into<Cow<'a, str>>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    pub fn styles(&self) -> &ButtonStyles {
        &self.styles
    }

    pub fn set_styles(mut self, styles: ButtonStyles) -> Self {
        self.styles = styles;
        self
    }
}

#[derive(Default)]
pub struct ButtonState {
    area:       Rect,
    pressed:    bool,
    pressed_by: Option<MouseButton>,
}

impl ButtonState {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<'a> Styled for Button<'a> {
    type Item = Self;

    fn style(&self) -> Style {
        self.styles.normal
    }

    fn set_style<S: Into<Style>>(mut self, style: S) -> Self::Item {
        self.styles.normal = style.into();
        self
    }
}

impl<'a> StatefulWidget for Button<'a> {
    type State = ButtonState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        state.area = area;
        let inner_area = self.block.inner(area);
        let style = match state.pressed {
            true => self.styles.pressed,
            false => self.styles.normal,
        };

        self.block.style(style).render(area, buf);
        Line::raw(self.text)
            .centered()
            .style(style)
            .render(inner_area, buf);
    }
}

pub enum ButtonResponse {
    None,
    Clicked,
    Released,
}

impl ButtonState {
    pub fn press(&mut self) {
        self.pressed = true;
    }

    pub fn release(&mut self) {
        self.pressed = false;
    }

    pub fn handle_event(&mut self, ev: &Event) -> ButtonResponse {
        match ev {
            Event::Mouse(mouse_event) if mouse_event.is_inside(self.area) => match mouse_event.kind
            {
                MouseEventKind::Down(btn) if self.pressed_by == None => {
                    self.press();
                    self.pressed_by = Some(btn);
                    ButtonResponse::Clicked
                }
                MouseEventKind::Up(btn) if self.pressed_by == Some(btn) => {
                    self.release();
                    self.pressed_by = None;
                    ButtonResponse::Released
                }
                _ => ButtonResponse::None,
            },
            _ => ButtonResponse::None,
        }
    }
}

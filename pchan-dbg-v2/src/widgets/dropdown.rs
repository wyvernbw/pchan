use crossterm::event::{Event, MouseEventKind};
use crossterm_simple_event::CrosstermSimpleEvent;
use ratatui::{
    prelude::*,
    style::Styled,
    widgets::{Block, Clear, List, ListState, Paragraph},
};

use crate::widgets::MouseEventExt;

pub struct Dropdown<'a, Value> {
    style:       Style,
    focus_style: Style,
    values:      &'a [Value],
    label:       Option<&'a str>,
    block:       Block<'a>,
}

impl<'a, Value> Dropdown<'a, Value> {
    pub fn new(values: &'a [Value]) -> Self {
        Self {
            style: Style::new(),
            focus_style: Style::new(),
            label: None,
            values,
            block: Block::new(),
        }
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = block;
        self
    }

    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    pub fn focus_style(mut self, style: Style) -> Self {
        self.focus_style = style;
        self
    }

    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }
}

pub struct DropdownState<Value> {
    area:          Rect,
    open:          bool,
    focused:       bool,
    current:       usize,
    list_state:    ListState,
    count:         usize,
    current_value: Option<Value>,
}

impl<T> Default for DropdownState<T> {
    fn default() -> Self {
        Self {
            area:          Default::default(),
            open:          Default::default(),
            focused:       Default::default(),
            current:       Default::default(),
            list_state:    Default::default(),
            count:         Default::default(),
            current_value: Default::default(),
        }
    }
}

impl<'a, Value: Clone + std::fmt::Display> StatefulWidget for Dropdown<'a, Value> {
    type State = DropdownState<Value>;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        state.area = area;
        state.count = self.values.len();
        state.current_value = self.values.get(state.current).cloned();
        let style = match state.focused {
            true => self.focus_style,
            false => self.style,
        };
        match state.open {
            false => {
                let label = self.label.unwrap_or("");
                let label = format!(" ▼ {}", label,);
                let current = self.values[state.current].clone().to_string();
                let [label_area, value_area] = area.layout(
                    &Layout::horizontal([
                        Constraint::Length(label.len() as u16),
                        Constraint::Length(current.len() as u16 + 1),
                    ])
                    .flex(layout::Flex::SpaceBetween),
                );
                self.block.style(style).render(area, buf);
                label.set_style(style).render(label_area, buf);
                current.set_style(style).render(value_area, buf);
            }
            true => {
                let values = List::from_iter(self.values.iter().map(|value| format!(" {value} ")))
                    .block(self.block)
                    .style(self.style)
                    .highlight_style(self.focus_style);
                let list_area = Rect::new(area.x, area.y, area.width, self.values.len() as u16);
                Clear.render(list_area, buf);
                StatefulWidget::render(values, list_area, buf, &mut state.list_state);
            }
        }
    }
}

impl<Value> DropdownState<Value> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn open(&mut self) -> &mut Self {
        self.open = true;
        self
    }

    pub fn close(&mut self) -> &mut Self {
        self.open = false;
        self
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn set_focus(&mut self, focus: bool) -> &mut Self {
        self.focused = focus;
        self
    }

    pub fn focus(&mut self) -> &mut Self {
        self.set_focus(true)
    }

    pub fn blur(&mut self) -> &mut Self {
        self.set_focus(false)
    }

    pub fn current(&self) -> Option<&Value> {
        self.current_value.as_ref()
    }

    pub fn handle_event(&mut self, ev: &Event) {
        match ev {
            Event::Mouse(mouse_event) => {
                if let MouseEventKind::Down(_) = mouse_event.kind {
                    match self.open {
                        true => {
                            let area = Rect::new(
                                self.area.x,
                                self.area.y,
                                self.area.width,
                                self.count as u16,
                            );
                            if mouse_event.is_inside(area) {
                                let idx = mouse_event.row.saturating_sub(area.y);
                                self.close();
                                self.current = idx as usize;
                                self.list_state.select(Some(idx.into()));
                                self.blur();
                            }
                        }
                        false => {
                            if mouse_event.is_inside(self.area) {
                                self.set_focus(!self.focused);
                                self.open();
                                self.list_state.select(Some(self.current));
                            }
                        }
                    }
                }
            }
            ev => {
                if !self.focused {
                    return;
                }
                match (self.open, ev.simple().as_str()) {
                    (false, "enter") => {
                        self.open();
                        self.list_state.select(Some(self.current));
                    }
                    (true, "j" | "down") => {
                        self.list_state.select_next();
                    }
                    (true, "k" | "up") => {
                        self.list_state.select_previous();
                    }
                    (true, "enter") => {
                        self.current = self.list_state.selected().unwrap_or(0);
                        self.close();
                    }
                    _ => {}
                };
            }
        }
    }
}

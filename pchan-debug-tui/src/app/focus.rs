use std::{
    any::{Any, TypeId},
    marker::PhantomData,
};

use flume::{Receiver, Sender};
use ratatui::style::{Style, Stylize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FocusId(TypeId);

pub trait Focus: Any {
    fn as_focus() -> FocusId {
        FocusId(TypeId::of::<Self>())
    }
    fn focus_next() -> Option<FocusId> {
        None
    }
    fn focus_prev() -> Option<FocusId> {
        None
    }

    fn focus_down() -> Option<FocusId> {
        None
    }
    fn focus_up() -> Option<FocusId> {
        None
    }
    fn focus_left() -> Option<FocusId> {
        None
    }
    fn focus_right() -> Option<FocusId> {
        None
    }
}

pub struct FocusProvider {
    receiver: Receiver<FocusEvent>,
    sender: Sender<FocusEvent>,
    stack: Vec<FocusId>,
}

impl FocusProvider {
    pub fn new(current: Option<FocusId>) -> FocusProvider {
        let (sender, receiver) = flume::unbounded();
        let mut stack = Vec::with_capacity(32);
        if let Some(current) = current {
            stack.push(current);
        }
        FocusProvider {
            sender,
            receiver,
            stack,
        }
    }
    fn focus(&mut self, on: FocusId) {
        match &mut self.stack[..] {
            [] => self.stack.push(on),
            [.., last] => {
                *last = on;
            }
        }
    }
    fn push_focus(&mut self, on: FocusId) {
        self.stack.push(on);
    }
    fn unfocus(&mut self) {
        self.stack.pop();
    }
    /// call in a loop
    pub fn process(&mut self) {
        while let Ok(event) = self.receiver.try_recv() {
            match event {
                FocusEvent::Focus(focus_id) => self.focus(focus_id),
                FocusEvent::PushFocus(focus_id) => self.push_focus(focus_id),
                FocusEvent::Unfocus => self.unfocus(),
            }
        }
    }
    pub fn props<T>(&self) -> FocusProp<T> {
        FocusProp {
            sender: self.sender.clone(),
            current: self.stack.last().cloned(),
            _self: PhantomData::<T>,
        }
    }
}

impl Default for FocusProvider {
    fn default() -> Self {
        Self::new(None)
    }
}

pub enum FocusEvent {
    /// use for focusing on elements within the same logical focus group
    Focus(FocusId),
    /// use to push a new focus group on the stack, (eg. a popup or the modeline)
    PushFocus(FocusId),
    /// inverse of the `PushFocus` event
    Unfocus,
}

#[derive(Debug, Clone)]
pub struct FocusProp<T> {
    sender: Sender<FocusEvent>,
    current: Option<FocusId>,
    _self: PhantomData<T>,
}

impl<T: Focus> FocusProp<T> {
    pub fn is_focused(&self) -> bool {
        self.current == Some(T::as_focus())
    }
    pub fn unfocus(&self) {
        _ = self
            .sender
            .send(FocusEvent::Unfocus)
            .inspect_err(|err| tracing::warn!(%err));
    }
    fn send_focus_event(&self, id: FocusId) {
        _ = self
            .sender
            .send(FocusEvent::Focus(id))
            .inspect_err(|err| tracing::warn!(%err));
    }
    pub fn focus(&self) {
        self.send_focus_event(T::as_focus());
    }
    pub fn focus_next(&self) {
        if let Some(next) = T::focus_next() {
            self.send_focus_event(next);
        }
    }
    pub fn focus_prev(&self) {
        if let Some(prev) = T::focus_prev() {
            self.send_focus_event(prev);
        }
    }
    pub fn focus_down(&self) {
        if let Some(down) = T::focus_down() {
            self.send_focus_event(down);
        }
    }
    pub fn focus_up(&self) {
        if let Some(up) = T::focus_up() {
            self.send_focus_event(up);
        }
    }
    pub fn focus_left(&self) {
        if let Some(left) = T::focus_left() {
            self.send_focus_event(left);
        }
    }
    pub fn focus_right(&self) {
        if let Some(right) = T::focus_right() {
            self.send_focus_event(right);
        }
    }
    pub fn push_focus(&self) {
        _ = self
            .sender
            .send(FocusEvent::PushFocus(T::as_focus()))
            .inspect_err(|err| tracing::warn!(%err));
    }
    pub fn style(&self) -> Style {
        match self.is_focused() {
            true => Style::new().green(),
            false => Style::new(),
        }
    }
}

impl<T> FocusProp<T> {
    pub fn prop<W>(&self) -> FocusProp<W> {
        FocusProp {
            sender: self.sender.clone(),
            current: self.current.clone(),
            _self: PhantomData::<W>,
        }
    }
}

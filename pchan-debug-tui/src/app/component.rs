use std::marker::PhantomData;

use color_eyre::eyre::Result;
use ratatui::crossterm::event;
use ratatui::prelude::*;

pub trait Component {
    type ComponentState;
    type ComponentSummary = ();
    fn handle_input(
        &mut self,
        event: event::Event,
        state: &mut Self::ComponentState,
    ) -> Result<Self::ComponentSummary>;
    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::ComponentState) -> Result<()>;
}

pub struct ComponentWidget<T> {
    inner: T,
}

impl<T> ComponentWidget<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T> StatefulWidget for ComponentWidget<T>
where
    T: Component,
{
    type State = T::ComponentState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        match self.inner.render(area, buf, state) {
            Ok(()) => {}
            Err(err) => tracing::error!(%err),
        }
    }
}

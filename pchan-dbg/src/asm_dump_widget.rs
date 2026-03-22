use std::sync::Arc;

use manatui::ratatui::crossterm::event::Event;
use manatui::tea::focus::Focus;
use manatui::{layout::layout::SharedText, prelude::*};
use manatui_tea_ui::{
    common::FocusItemState,
    components::pager::{Pager, PagerView},
};

use crate::border_style_focus;

#[derive(Debug, Default)]
pub struct AsmDump {
    pager: Pager,
    focus: FocusItemState,
}

impl Focus for AsmDump {
    fn set_focus(&self, value: bool) {
        self.focus.set_focus(value);
    }

    fn focus(&self) -> bool {
        self.focus.focus()
    }

    fn rect(&self) -> Option<manatui::ratatui::prelude::Rect> {
        self.focus.rect()
    }

    fn keymaps(&self) -> &'static [manatui::tea::focus::KeyMap] {
        self.pager.keymaps()
    }

    fn hit_test(&self) -> manatui::tea::observe::HitEvent {
        self.focus.hit_test()
    }
}

impl AsmDump {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn update(mut self, event: &Event) -> AsmDump {
        self.pager.set_focus(self.focus());
        if !self.focus() {
            return self;
        }

        self.pager = self.pager.update(event);
        self
    }
}

#[subview]
pub(crate) fn asm_dump_view(state: &AsmDump, asm: Arc<str>) -> View {
    let focused = state.focus();
    let border_style = border_style_focus(focused);
    let asm = SharedText::<Arc<str>, Paragraph>::new(asm);
    ui! {
        <Block>
            <Text>"Objdump"</Text>
            <Block .rounded .border_style={border_style}
                {state.focus.hit_test.clone()}
                {state.focus.area_ref.clone()}
                Width::grow() Height::grow()
            >
                <PagerView .content={asm.into_view()} .state={&state.pager}
                    Width::grow() Height::grow()
                />
            </Block>
        </Block>
    }
}

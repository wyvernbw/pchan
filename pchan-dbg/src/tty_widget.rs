use manatui_tea_ui::components::button::{Button, ButtonEvent, ButtonView};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use manatui::{
    prelude::*,
    ratatui::crossterm::event::{Event, MouseEventKind},
    tea::{
        focus::{CTRL_KEYMAP, EventOutcome, Focus, FocusGroup, VIM_CTRL_KEYMAP},
        observe::{AreaRef, HitTest},
    },
};

use crate::border_style_focus;

#[derive(Debug, Default)]
pub(crate) struct TtyViewState {
    focused:         AtomicBool,
    fg:              FocusGroup,
    pop_out_button:  Button,
    minimize_button: Button,
    area_ref:        AreaRef,
    hit_test:        HitTest,
    popped_out:      Option<PoppedOut>,
    minimized:       bool,
}

#[derive(Debug, Default, Clone, Copy)]
struct PoppedOut {
    click_offset: (u16, u16),
    final_offset: (u16, u16),
}

impl Focus for TtyViewState {
    fn set_focus(&self, value: bool) {
        self.focused.store(value, Ordering::Relaxed);
    }

    fn focus(&self) -> bool {
        self.focused.load(Ordering::Relaxed)
    }

    fn rect(&self) -> Option<ratatui::prelude::Rect> {
        self.area_ref.get()
    }

    fn keymaps(&self) -> &'static [manatui::tea::focus::KeyMap] {
        &[VIM_CTRL_KEYMAP, CTRL_KEYMAP]
    }

    fn hit_test(&self) -> manatui::tea::observe::HitEvent {
        self.hit_test.get()
    }
}

impl TtyViewState {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn build_focus(&self) {
        self.fg
            .items()
            .next_untagged(&self.pop_out_button)
            .next_untagged(&self.minimize_button)
            .commit();
    }

    pub(crate) fn update(mut self, event: &Event) -> Self {
        if !self.focus() {
            self.pop_out_button.set_focus(false);
            return self;
        }
        self.build_focus();

        match self.fg.set_wrap_around(true).update(event) {
            EventOutcome::Consumed(fg) => {
                self.fg = fg;
                self.build_focus();
                return self;
            }
            EventOutcome::Unhandled(fg) => {
                self.fg = fg;
            }
        }

        let (btn, btn_event) = self.pop_out_button.update(event);
        self.pop_out_button = btn;
        match btn_event {
            ButtonEvent::Clicked => {
                if self.popped_out.is_none() {
                    self.popped_out = Some(PoppedOut::default());
                } else {
                    self.popped_out = None;
                    self.pop_out_button.set_focus(false);
                }
            }
            ButtonEvent::None => {}
        };

        let (min_btn, min_event) = self.minimize_button.update(event);
        self.minimize_button = min_btn;
        match min_event {
            ButtonEvent::Clicked => {
                self.minimized = !self.minimized;
            }
            ButtonEvent::None => {}
        }

        if let Event::Mouse(mouse) = event {
            match mouse.kind {
                MouseEventKind::Down(btn) if self.popped_out.is_some() => {
                    if let Some(rect) = self.rect() {
                        self.popped_out = Some(PoppedOut {
                            click_offset: (
                                mouse.column.saturating_sub(rect.x),
                                mouse.row.saturating_sub(rect.y),
                            ),
                            final_offset: (rect.x, rect.y),
                        })
                    }
                }
                MouseEventKind::Drag(btn) => {
                    if let Some(popped_out) = self.popped_out {
                        self.popped_out = Some(PoppedOut {
                            final_offset: (
                                mouse.column.saturating_sub(popped_out.click_offset.0),
                                mouse.row.saturating_sub(popped_out.click_offset.1),
                            ),
                            ..popped_out
                        });
                    }
                }
                _ => {}
            }
        }

        self
    }
}

#[subview]
pub(crate) fn tty_view(tty: &[Arc<str>], state: &TtyViewState) -> View {
    let tty_iter = tty.iter().map(|line| ui! {"{line}"});
    let focused = state.focus();
    let border_style = border_style_focus(focused);

    let position = match (state.popped_out, state.rect()) {
        (Some(_), None) => Position::Auto,
        (Some(popped), Some(rect)) => Position::Absolute(
            Value::Cells(popped.final_offset.0),
            Value::Cells(popped.final_offset.1),
        ),
        (None, None) => Position::Auto,
        (None, Some(_)) => Position::Auto,
    };
    let width = match (state.popped_out.is_some(), state.rect()) {
        (true, None) | (false, None) | (false, Some(_)) => Width::grow(),
        (true, Some(rect)) => Width::fixed(rect.width),
    };
    let height = match (state.popped_out.is_some(), state.rect()) {
        (true, None) | (false, None) | (false, Some(_)) => Height::grow(),
        (true, Some(rect)) => Height::fixed(rect.height),
    };

    if state.minimized {
        return ui! {
        <Block>
            <Text>"TTY"</Text>
            <Block
                .rounded .border_style={border_style} .title="+ tty +"
                {width} Height::fixed(3)
                Padding::new(1, 1, 0, 1)
                {state.area_ref.clone()}
                {state.hit_test.clone()}
                {position}
                Clear
            >
                <Block Width::grow() Direction::Horizontal MainJustify::End Gap(2)>
                    <ButtonView .state={&state.pop_out_button} .clicked_style={Style::new().reversed()}>
                        <Block>
                        "+ ⇱ +"
                        </Block>
                    </ButtonView>
                    <ButtonView .state={&state.minimize_button} .clicked_style={Style::new().reversed()}>
                        <Block>
                        "+ X +"
                        </Block>
                    </ButtonView>
                </Block>
            </Block>
        </Block>
        };
    }

    ui! {
        <Block>
            <Text>"TTY"</Text>
            <Block
                .rounded .border_style={border_style} .title="+ tty +"
                {width} {height}
                Padding::new(1, 1, 0, 1)
                {state.area_ref.clone()}
                {state.hit_test.clone()}
                {position}
                Clear
            >
                <Block Width::grow() Direction::Horizontal MainJustify::End Gap(2)>
                    <ButtonView .state={&state.pop_out_button} .clicked_style={Style::new().reversed()}>
                        <Block>
                        "+ ⇱ +"
                        </Block>
                    </ButtonView>
                    <ButtonView .state={&state.minimize_button} .clicked_style={Style::new().reversed()}>
                        <Block>
                        "+ X +"
                        </Block>
                    </ButtonView>
                </Block>
                <Block Width::grow() Height::grow()>
                {tty_iter}
                </Block>
            </Block>
        </Block>
    }
}

use std::sync::atomic::{AtomicBool, Ordering};

use manatui::{
    prelude::*,
    ratatui::{
        crossterm::event::{Event, MouseEventKind},
        text::Span,
    },
    tea::{
        focus::{CTRL_KEYMAP, Focus, VIM_CTRL_KEYMAP},
        observe::{AreaRef, HitTest},
    },
    utils::keyv2,
};
use manatui_tea_ui::components::text_input::{TextInput, TextInputEvent, TextInputView};
use pchan_emu::{dynarec_v2::emitters::DecodedOp, io::IO};
use pchan_utils::hex;

use crate::{Hseparator, border_style_focus, emu_task::DebugView, lipgloss_colors::LipglossStyle};

pub(crate) struct MemViewState {
    focused:       AtomicBool,
    area_ref:      AreaRef,
    hit_test:      HitTest,
    page_address:  u32,
    address:       u32,
    go_to_input:   TextInput,
    go_to_visible: bool,
}

impl Default for MemViewState {
    fn default() -> Self {
        Self {
            focused:       Default::default(),
            area_ref:      Default::default(),
            hit_test:      Default::default(),
            page_address:  0xbfc0_0000,
            address:       0xbfc0_0000,
            go_to_input:   TextInput::new(),
            go_to_visible: false,
        }
    }
}

impl Focus for MemViewState {
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

impl MemViewState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn offset(mut self, offset: i32) -> Self {
        self.address = self.address.saturating_add_signed(offset);
        self
    }

    pub(crate) fn update(mut self, event: &Event) -> Self {
        if !self.focus() {
            return self;
        };
        match self.go_to_visible {
            false => {
                self = match event {
                    keyv2!('g') => {
                        self.go_to_visible = true;
                        self
                    }
                    keyv2!(left) | keyv2!('l') => self.offset(4),
                    keyv2!(right) | keyv2!('h') => self.offset(-4),
                    keyv2!(up) | keyv2!('k') => self.offset(-16),
                    keyv2!(down) | keyv2!('j') => self.offset(16),
                    _ => self,
                };
                let page_size = self.rect().unwrap_or_default().height.saturating_sub(7) * 4 * 4;
                let page_size = page_size as u32;
                if self.address < self.page_address {
                    self.page_address -= page_size;
                } else if self.address >= self.page_address + page_size {
                    self.page_address += page_size;
                }
                self
            }
            true => {
                match event {
                    keyv2!(esc) => {
                        self.go_to_input.set_focus(false);
                        self.go_to_visible = false;
                        return self;
                    }
                    _ => {}
                }

                self.go_to_input.set_focus(true);
                let (input, event) = self.go_to_input.update(event);
                self.go_to_input = input;
                match event {
                    TextInputEvent::None => {}
                    TextInputEvent::Confirm => {
                        self.go_to_input.set_focus(false);
                        self.go_to_visible = false;
                        let address = self
                            .go_to_input
                            .value()
                            .trim_start()
                            .trim_start_matches("0x");
                        let Ok(address) = u32::from_str_radix(address, 16) else {
                            return self;
                        };
                        self.page_address = address;
                        self.address = address;
                    }
                }
                self
            }
        }
    }
}

#[subview]
pub(crate) fn mem_view(state: &MemViewState, view: &DebugView) -> View {
    let focused = state.focus();
    let border_style = border_style_focus(focused);
    let mem_start = state.page_address;
    let mem_range = mem_start..(mem_start + 1024);
    let mem_content = mem_range
        .step_by(0x4)
        .map(|addr| view.emu().try_read_pure::<u32>(addr).unwrap_or(0x0))
        .enumerate()
        .array_chunks::<4>()
        .map(|words| {
            let idx = words[0].0;
            let addr = idx as u32 * 4 * 4 * 4 + mem_start;
            let base_style = match (idx >> 2).is_multiple_of(2) {
                true => Style::new(),
                false => Style::new().fg(Color::from_u32(0xeeeeee)),
            };
            let addr = format!("{}", hex(addr)).set_style(base_style.dim());
            let words = words.map(|(word_address, word)| {
                let word_address = word_address as u32 * 4 + mem_start;
                let style = match word_address == state.address && focused {
                    true => base_style.black().on_c0700(),
                    false => base_style,
                };
                format!("{:08x}", word).set_style(style)
            });
            Row::new([[addr].as_slice(), words.as_slice()].concat())
        });
    let mem_title = format!("+ mem: {} +", hex(state.address));
    let current_word = view
        .emu()
        .try_read_pure::<u32>(state.address)
        .unwrap_or_default();
    let current_i8 = current_word
        .to_le_bytes()
        .map(|byte| byte as i8)
        .map(|byte| ui! {"{byte:04}"});
    let current_u8 = current_word
        .to_le_bytes()
        .map(|byte| byte as u8)
        .map(|byte| ui! {"{byte:04}"});
    let current_char = current_word
        .to_le_bytes()
        .map(|byte| byte as char)
        .map(|byte| ui! {"{byte}"});
    let current_i32 = current_word as i32;
    let [current_mips] = DecodedOp::decode([current_word]);
    ui! {
        <Block Width::grow() Height::grow() MaxWidth::fixed(2 + 8 + 1 + 4*8 + 8 )
            {state.hit_test.clone()}
            {state.area_ref.clone()}
        >
            "Memory"
            <Block .rounded .border_style={border_style} .title={mem_title}
                Width::grow() Height::grow() Padding::new(2, 2, 1, 1)
            >
                <Table .widths={[10, 8, 8, 8, 8]} .column_spacing={1} .rows={mem_content}  Height::grow() Width::grow()/>
                <Hseparator .border_type={BorderType::LightDoubleDashed} .style={Style::new().dim()}/>
                <Block Width::grow()>
                {if state.go_to_visible {
                    ui! {
                        <Block .rounded .title="jump to" Width::grow()>
                            <TextInputView
                                .state={&state.go_to_input}
                                .cursor_style={Style::new().black().on_c0700()}
                                .select_style={Style::new().reversed()}
                                Width::grow()
                            />
                        </Block>
                    }
                } else {
                        ui! {<Block/>}
                }}
                </Block>
                <Block Direction::Horizontal Gap(1) Width::grow()>
                    <Block Width::grow() Direction::Horizontal Gap(1)>
                        <Block Width::grow() Direction::Horizontal>
                            <Span .style={Style::new().dim()}>"ascii: "</Span>
                            <Block Direction::Horizontal Width::grow()>
                                {current_char}
                            </Block>
                        </Block>
                        <Block Width::grow() Direction::Horizontal>
                            <Span .style={Style::new().dim()}>"asm: "</Span>
                            <Block Direction::Horizontal >
                                "{current_mips}"
                            </Block>
                        </Block>
                    </Block>
                </Block>
                <Block Direction::Horizontal Gap(1) Width::grow()>
                    <Block Width::grow() Direction::Horizontal>
                        <Span .style={Style::new().dim()}>"i32: "</Span>
                        "{current_i32}"
                    </Block>
                    <Block Width::grow() Direction::Horizontal>
                        <Span .style={Style::new().dim()}>"u32: "</Span>
                        <Block Direction::Horizontal Gap(1) Width::grow()>
                            "{current_word}"
                        </Block>
                    </Block>
                </Block>
                <Block Direction::Horizontal Gap(1)>
                    <Block Direction::Horizontal Width::percentage(50)>
                        <Span .style={Style::new().dim()}>"i8: "</Span>
                        <Block Direction::Horizontal Gap(1)>
                            {current_i8}
                        </Block>
                    </Block>
                    <Block Direction::Horizontal Width::percentage(50)>
                        <Span .style={Style::new().dim()}>"u8: "</Span>
                        <Block Direction::Horizontal Gap(1)>
                            {current_u8}
                        </Block>
                    </Block>
                </Block>
            </Block>
        </Block>
    }
}

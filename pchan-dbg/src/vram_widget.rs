use arbitrary_int::prelude::*;
use bitbybit::bitfield;
use manatui::prelude::canvas::{Canvas, Rectangle};
use manatui::prelude::*;
use manatui::ratatui::prelude::*;
use manatui::tea::focus::{DEFAULT_KEYMAP, Focus, VIM_KEYMAP};
use manatui_tea_ui::common::FocusItemState;
use pchan_emu::Bus;

use crate::emu_task::DebugView;

#[derive(Debug, Default, Clone)]
pub struct VramCanvasWidget {
    style:    Style,
    dbg_view: DebugView,
}

impl Widget for VramCanvasWidget {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        #[bitfield(u16)]
        struct Pixel {
            #[bits(0..=4, rw)]
            r: u5,
            #[bits(5..=9, rw)]
            g: u5,
            #[bits(10..=14, rw)]
            b: u5,
        }
        let canvas = Canvas::default()
            .x_bounds([0.0, 1024.0])
            .y_bounds([0.0, 512.0])
            .marker(symbols::Marker::Quadrant)
            .paint(|ctx| {
                for y in 0..512 {
                    for x in 0..1024 {
                        let vram = &self.dbg_view.emu().gpu().vram;
                        let vram_addr = 1024 * y + x;
                        let pixel = vram[vram_addr];
                        // ctx.draw(&Rectangle {
                        //     x:      x as f64,
                        //     y:      y as f64,
                        //     width:  1.0,
                        //     height: 1.0,
                        //     color:  Color::Rgb((x * 255 / 1024) as u8, (y * 255 / 512) as u8, 255),
                        // });
                        // continue;
                        let pixel = Pixel::new_with_raw_value(pixel);

                        ctx.draw(&Rectangle {
                            x:      x as f64,
                            y:      y as f64,
                            width:  1.0,
                            height: 1.0,
                            color:  Color::Rgb(pixel.r().as_(), pixel.g().as_(), pixel.b().as_()),
                        });
                    }
                }
            });
        canvas.render(area, buf);
    }
}

impl Styled for VramCanvasWidget {
    type Item = Self;

    fn style(&self) -> Style {
        self.style
    }

    fn set_style<S: Into<Style>>(mut self, style: S) -> Self::Item {
        self.style = style.into();
        self
    }
}

impl VramCanvasWidget {
    pub fn debug_view(mut self, dbg_view: DebugView) -> Self {
        self.dbg_view = dbg_view;
        self
    }
}

pub struct VramCanvas {
    focus: FocusItemState,
}

impl VramCanvas {
    pub fn new() -> Self {
        Self {
            focus: FocusItemState::default(),
        }
    }
}

impl Focus for VramCanvas {
    fn set_focus(&self, value: bool) {
        self.focus.set_focus(value);
    }

    fn focus(&self) -> bool {
        self.focus.focus()
    }

    fn rect(&self) -> Option<Rect> {
        self.focus.rect()
    }

    fn keymaps(&self) -> &'static [manatui::tea::focus::KeyMap] {
        &[DEFAULT_KEYMAP, VIM_KEYMAP]
    }

    fn hit_test(&self) -> manatui::tea::observe::HitEvent {
        self.focus.hit_test()
    }
}

#[subview]
pub fn vram_canvas_view(state: &VramCanvas, dbg_view: DebugView) -> View {
    ui! {
        <Block
            {state.focus.area_ref.clone()}
            {state.focus.hit_test.clone()}
            Width::grow() Height::grow()
        >
            <VramCanvasWidget .debug_view={dbg_view} Width::grow() Height::grow() />
        </Block>
    }
}

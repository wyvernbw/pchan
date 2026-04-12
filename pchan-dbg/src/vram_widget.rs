use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, RwLock};

use arbitrary_int::prelude::*;
use bitbybit::bitfield;
use image::{DynamicImage, ImageBuffer, Rgb};
use manatui::prelude::canvas::{Canvas, Rectangle};
use manatui::prelude::*;
use manatui::ratatui::prelude::*;
use manatui::tea::HotLoop;
use manatui::tea::focus::{DEFAULT_KEYMAP, Focus, VIM_KEYMAP};
use manatui_tea_ui::common::FocusItemState;
use pchan_emu::Bus;
use pchan_emu::gpu::VBLANK_COUNT;
use ratatui_image::picker::Picker;
use ratatui_image::protocol::Protocol;
use ratatui_image::protocol::halfblocks::Halfblocks;

use crate::emu_task::DebugView;

#[derive(Debug, Default, Clone)]
pub struct VramCanvasWidget {
    style:    Style,
    dbg_view: DebugView,
}

thread_local! {
    static PICKER: Picker = Picker::from_query_stdio().unwrap();
}
static VRAM_HASH: AtomicU64 = AtomicU64::new(0);
static VRAM_IMAGE: LazyLock<RwLock<Protocol>> = LazyLock::new(|| {
    RwLock::new(Protocol::Halfblocks(
        Halfblocks::new(
            DynamicImage::ImageRgb8(ImageBuffer::default()),
            Rect::default(),
        )
        .unwrap(),
    ))
});

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
        let vram = &self.dbg_view.emu().gpu().vram;
        let mut hasher = DefaultHasher::new();
        format!("vblank #{}", VBLANK_COUNT.load(Ordering::Relaxed)).render(area, buf);
        let area = area.offset(layout::Offset { x: 0, y: 1 });
        vram.hash(&mut hasher);
        let hash = hasher.finish();
        let old_hash = VRAM_HASH.load(Ordering::Acquire);
        if old_hash != hash {
            VRAM_HASH.store(hash, Ordering::Release);
            let mut output = ImageBuffer::new(1024, 512);
            for y in 0..512 {
                for x in 0..1024 {
                    let vram_addr = x + y * 1024;
                    let pixel = Pixel::new_with_raw_value(vram[vram_addr]);
                    output.put_pixel(
                        x as u32,
                        y as u32,
                        Rgb([
                            (pixel.r().as_::<u16>() << 3) as u8,
                            (pixel.g().as_::<u16>() << 3) as u8,
                            (pixel.b().as_::<u16>() << 3) as u8,
                        ]),
                    );
                }
            }
            let output = DynamicImage::ImageRgb8(output);
            let img = PICKER
                .with(|picker| {
                    picker.new_protocol(
                        output,
                        area,
                        ratatui_image::Resize::Scale(Some(ratatui_image::FilterType::Nearest)),
                    )
                })
                .unwrap();
            ratatui_image::Image::new(&img).render(area, buf);
            *VRAM_IMAGE.write().unwrap() = img;
        } else {
            let img = VRAM_IMAGE.read().unwrap();
            ratatui_image::Image::new(&img).render(area, buf);
        }
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

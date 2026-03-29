use std::mem::transmute;

use crate::gpu::DrawPixels;
use crate::gpu::GpuStatReg;
use crate::gpu::IVramCoord;
use crate::gpu::VramCoord;
use crate::io::CastIOInto;
use arbitrary_int::prelude::*;
use bitbybit::bitenum;
use bitbybit::bitfield;
use bon::Builder;
use glam::U8Vec2;
use tracing::Level;

#[derive(Debug, Clone)]
pub struct DrawCall {
    pub gpustat: GpuStatReg,
    pub inner:   DrawCallKind,
}

#[derive(Debug, Clone)]
pub enum DrawCallKind {
    Rect(DrawRect),
}

#[derive(Debug, Clone, Copy)]
pub enum DrawRectDecoder {
    Color {
        color: DrawRectColor,
    },
    /// entered when textured=true
    Vertex1Textured {
        color:   DrawRectColor,
        vertex1: VramCoord,
    },
    /// entered when textured=false, var_size=true
    Vertex1VarSize {
        color:   DrawRectColor,
        vertex1: VramCoord,
    },
    Uv {
        color:   DrawRectColor,
        vertex1: VramCoord,
        uv:      Uv,
    },
}

#[derive(Debug, Clone, Builder)]
#[derive_const(Default)]
pub struct DrawRect {
    pub color:    DrawRectColor,
    pub vertex1:  VramCoord,
    pub uv:       Option<Uv>,
    pub var_size: Option<VramCoord>,
}

#[derive(Debug, Clone, Copy, Default, Builder)]
pub struct Uv {
    pub clut: U8Vec2,
    pub uv:   U8Vec2,
}

impl Uv {
    pub fn from_u32(value: u32) -> Self {
        unsafe { transmute(value) }
    }
}

#[bitfield(u32)]
pub struct Gp0SetDrawAreaCmd {
    #[bits(0..=9, rw)]
    x_coord: u10,

    #[bits(10..=18, rw)]
    y_coord_v1: u9,
    #[bits(10..=19, rw)]
    y_coord_v2: u10,
}

#[bitfield(u32)]
pub struct Gp0SetDrawOffsetCmd {
    #[bits(0..=10, rw)]
    x_offset: i11,
    #[bits(11..=21, rw)]
    y_offset: i11,
}

/// GP0(E6h) - Mask Bit Setting
///
/// ```md
///  0     Set mask while drawing (0=TextureBit15, 1=ForceBit15=1)   ;GPUSTAT.11
///  1     Check mask before draw (0=Draw Always, 1=Draw if Bit15=0) ;GPUSTAT.12
///  2-23  Not used (zero)
///  24-31 Command  (E6h)
/// ```
#[bitfield(u32)]
pub struct Gp0SetMaskBitCmd {
    #[bit(0, rw)]
    draw_mask:   bool,
    #[bit(1, rw)]
    draw_pixels: DrawPixels,
}

#[derive(Debug, Clone, Default)]
pub struct DrawOptsRegister {
    pub draw_area_top_left:     VramCoord,
    pub draw_area_bottom_right: VramCoord,
    pub draw_offset:            IVramCoord,
}

/// ```md
///  31-29        011    rectangle render
///  28-27        sss    rectangle size
///    26         1/0    textured / untextured
///    25         1/0    semi-transparent / opaque
///    24         1/0    raw texture / modulation
///   23-0        rgb    first color value.
/// ```
///
/// Credits to PSX-SPX by Martin Korth [Gpu Status Register](https://problemkaputt.de/psx-spx.htm#gpustatusregister)
#[bitfield(u32, debug)]
#[derive_const(Default)]
pub struct DrawRectColor {
    #[bits(0..=23, rw)]
    rgb: u24,
    #[bit(24, rw)]
    raw: bool,

    #[bit(25, rw)]
    semi_transparent: bool,

    #[bit(26, rw)]
    textured: bool,

    #[bits(27..=28, rw)]
    size: RectSize,
}

#[bitenum(u2, exhaustive = true)]
#[derive(Debug)]
pub enum RectSize {
    VarSize     = 0x0,
    SinglePixel = 0x1,
    Sprite8x8   = 0x2,
    Sprite16x16 = 0x3,
}

impl DrawRectDecoder {
    pub fn new(value: impl Copy) -> Self {
        let color = DrawRectColor::new_with_raw_value(value.io_into_u32());
        Self::Color { color }
    }
}

pub trait DrawCallDecoder: Sized {
    type Output;
    fn advance<T: Copy>(self, value: T) -> Result<Self, Self::Output>;
}

impl DrawCallDecoder for DrawRectDecoder {
    type Output = DrawRect;

    #[pchan_macros::instrument(level = Level::TRACE, skip_all, ret)]
    fn advance<T: Copy>(self, value: T) -> Result<Self, DrawRect> {
        match self {
            DrawRectDecoder::Color { color } => {
                let vertex1 = VramCoord::from(value.io_into_u32());
                match (color.textured(), color.size()) {
                    (_, RectSize::VarSize) => Ok(Self::Vertex1VarSize { color, vertex1 }),
                    (true, _) => Ok(Self::Vertex1Textured { color, vertex1 }),
                    (false, _) => Err(DrawRect {
                        color,
                        vertex1,
                        ..Default::default()
                    }),
                }
            }
            DrawRectDecoder::Vertex1Textured { color, vertex1 } => match color.size() {
                RectSize::VarSize => Ok(Self::Uv {
                    color,
                    vertex1,
                    uv: Uv::from_u32(value.io_into_u32()),
                }),
                _ => Err(DrawRect {
                    color,
                    vertex1,
                    uv: Some(Uv::from_u32(value.io_into_u32())),
                    var_size: None,
                }),
            },
            DrawRectDecoder::Vertex1VarSize { color, vertex1 } => match color.textured() {
                true => Ok(Self::Uv {
                    color,
                    vertex1,
                    uv: Uv::from_u32(value.io_into_u32()),
                }),
                false => Err(DrawRect {
                    color,
                    vertex1,
                    uv: None,
                    var_size: Some(VramCoord::from(value.io_into_u32())),
                }),
            },
            DrawRectDecoder::Uv { color, vertex1, uv } => Err(DrawRect {
                color,
                vertex1,
                uv: Some(uv),
                var_size: Some(VramCoord::from(value.io_into_u32())),
            }),
        }
    }
}

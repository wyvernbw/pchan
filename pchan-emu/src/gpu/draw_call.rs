use crate::gpu::DrawPixels;
use crate::gpu::IVramCoord;
use crate::gpu::VramCoord;
use arbitrary_int::prelude::*;
use bitbybit::bitfield;

pub enum DrawCall {
    Rect(DrawRect),
}

pub enum DrawRectDecoder {
    Color {
        color: DrawRectColor,
    },
    Vertex1 {
        color:   DrawRectColor,
        vertex1: VramCoord,
    },
    Uv {
        color:   DrawRectColor,
        vertex1: VramCoord,
        uv:      u32,
    },
    Size {
        color:   DrawRectColor,
        vertex1: VramCoord,
        uv:      u32,
        size:    VramCoord,
    },
}

pub struct DrawRect {
    color:    DrawRectColor,
    vertex1:  VramCoord,
    // TODO: uv attribute
    uv:       Option<u32>,
    var_size: Option<VramCoord>,
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
///
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

///  31-29        011    rectangle render
///  28-27        sss    rectangle size
///    26         1/0    textured / untextured
///    25         1/0    semi-transparent / opaque
///    24         1/0    raw texture / modulation
///   23-0        rgb    first color value.
///
/// Credits to PSX-SPX by Martin Korth [Gpu Status Register](https://problemkaputt.de/psx-spx.htm#gpustatusregister)
#[bitfield(u32)]
pub struct DrawRectColor {
    #[bits(0..=23, rw)]
    rgb: u24,
}

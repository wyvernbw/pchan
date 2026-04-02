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
use glam::I16Vec2;
use glam::U8Vec2;
use glam::U8Vec3;
use smallvec::SmallVec;
use smallvec::smallvec;
use tracing::Level;

#[derive(Debug, Clone)]
pub struct DrawCall {
    pub gpustat: GpuStatReg,
    pub inner:   DrawCallKind,
}

#[derive(Debug, Clone)]
pub enum DrawCallKind {
    Rect(DrawRect),
    Polygon(DrawPolygon),
    Line(DrawLine),
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

#[derive(Debug, Clone)]
pub struct DrawPolygon {
    pub header: DrawPolygonHeader,
    pub attrs:  heapless::Vec<DrawPolygonAttribute, 4>,
}

#[derive(Debug, Clone)]
pub struct DrawPolygonAttribute {
    pub color:  Option<U8Vec3>,
    pub vertex: I16Vec2,
    pub uv:     Option<Uv>,
}

/// ```md
///  bit number   value   meaning
///  31-29        001    polygon render
///    28         1/0    gouraud / flat shading
///    27         1/0    4 / 3 vertices
///    26         1/0    textured / untextured
///    25         1/0    semi-transparent / opaque
///    24         1/0    raw texture / modulation
///   23-0        rgb    first color value.
/// ```
#[bitfield(u32, debug)]
#[derive(Default)]
pub struct DrawPolygonHeader {
    #[bits(0..=23, rw)]
    color: u24,
    #[bit(24)]
    raw:   bool,

    #[bit(25, rw)]
    semi_transparent: bool,

    #[bit(26, rw)]
    textured:     bool,
    #[bit(27, rw)]
    vertex_count: DrawPolygonVertexCount,
    #[bit(28, rw)]
    shading:      Shading,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
enum DrawPolygonVertexCount {
    Three = 0x0,
    Four  = 0x1,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
pub enum Shading {
    Flat    = 0x0,
    Gouraud = 0x1,
}

#[derive(Debug, Clone, Default)]
pub struct DrawPolygonDecoder {
    header:       DrawPolygonHeader,
    attrs:        heapless::Vec<DrawPolygonAttribute, 4>,
    current_attr: DrawPolygonAttributeDecoder,
}

#[derive(Debug, Clone, Default)]
enum DrawPolygonAttributeDecoder {
    #[default]
    IdleGoraudUntextured,
    IdleFlatUntextured,
    IdleGoraudTextured,
    IdleFlatTextured,

    ColorUntextured {
        color: u32,
    },
    ColorTextured {
        color: u32,
    },

    // these are both textured
    VertexFlat {
        vertex: I16Vec2,
    },
    VertexGoraud {
        color:  u32,
        vertex: I16Vec2,
    },
}

impl DrawCallDecoder for DrawPolygonAttributeDecoder {
    type Output = DrawPolygonAttribute;

    fn advance<T: Copy>(self, value: T) -> Result<Self, Self::Output> {
        let value = value.io_into_u32();
        let vertex_value = unsafe { transmute::<u32, I16Vec2>(value) };
        let uv_value = unsafe { transmute::<u32, Uv>(value) };
        match self {
            DrawPolygonAttributeDecoder::IdleGoraudUntextured => {
                Ok(Self::ColorUntextured { color: value })
            }
            DrawPolygonAttributeDecoder::IdleFlatUntextured => Err(DrawPolygonAttribute {
                color:  None,
                vertex: vertex_value,
                uv:     None,
            }),
            DrawPolygonAttributeDecoder::IdleFlatTextured => Ok(Self::VertexFlat {
                vertex: vertex_value,
            }),
            DrawPolygonAttributeDecoder::IdleGoraudTextured => {
                Ok(Self::ColorTextured { color: value })
            }
            DrawPolygonAttributeDecoder::ColorUntextured { color } => Err(DrawPolygonAttribute {
                color:  Some(u8vec3_from_u32(color)),
                vertex: vertex_value,
                uv:     None,
            }),
            DrawPolygonAttributeDecoder::ColorTextured { color } => Ok(Self::VertexGoraud {
                color,
                vertex: vertex_value,
            }),
            DrawPolygonAttributeDecoder::VertexFlat { vertex } => Err(DrawPolygonAttribute {
                color: None,
                vertex,
                uv: Some(uv_value),
            }),
            DrawPolygonAttributeDecoder::VertexGoraud { color, vertex } => {
                Err(DrawPolygonAttribute {
                    color: Some(u8vec3_from_u32(color)),
                    vertex,
                    uv: Some(uv_value),
                })
            }
        }
    }
}

fn u8vec3_from_u32(value: u32) -> U8Vec3 {
    U8Vec3::from_slice(&value.to_le_bytes()[0..3])
}

impl DrawCallDecoder for DrawPolygonDecoder {
    type Output = DrawPolygon;

    fn advance<T: Copy>(mut self, value: T) -> Result<Self, Self::Output> {
        match self.current_attr.advance(value) {
            Ok(decoder) => {
                self.current_attr = decoder;
                Ok(self)
            }
            Err(attribute) => {
                // SAFETY: no point in checking
                unsafe {
                    self.attrs.push_unchecked(attribute);
                }
                let expected = match self.header.vertex_count() {
                    DrawPolygonVertexCount::Three => 3,
                    DrawPolygonVertexCount::Four => 4,
                };
                if self.attrs.len() >= expected {
                    Err(DrawPolygon {
                        header: self.header,
                        attrs:  self.attrs,
                    })
                } else {
                    Ok(Self {
                        current_attr: DrawPolygonAttributeDecoder::from_header(self.header, false),
                        ..self
                    })
                }
            }
        }
    }
}

impl DrawPolygonAttributeDecoder {
    pub const fn from_header(header: DrawPolygonHeader, is_first: bool) -> Self {
        // first attribute is essentially flat shaded since its color is present
        // in the header.
        if is_first {
            return match header.textured() {
                true => Self::IdleFlatTextured,
                false => Self::IdleFlatUntextured,
            };
        }
        match (header.shading(), header.textured()) {
            (Shading::Flat, true) => Self::IdleFlatTextured,
            (Shading::Flat, false) => Self::IdleFlatUntextured,
            (Shading::Gouraud, true) => Self::IdleGoraudTextured,
            (Shading::Gouraud, false) => Self::IdleGoraudUntextured,
        }
    }
}

impl DrawPolygonDecoder {
    pub fn new(value: u32) -> Self {
        let header = DrawPolygonHeader::new_with_raw_value(value);
        let current_attr = DrawPolygonAttributeDecoder::from_header(header, true);
        Self {
            header,
            attrs: Default::default(),
            current_attr,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DrawLine {
    header: DrawLineHeader,
    // most lines are 2 vertices, for poly-lines
    // we do a heap allocation
    attrs:  SmallVec<[DrawLineAttribute; 2]>,
}

/// ```md
///  bit number   value   meaning
///  31-29        010    line render
///    28         1/0    gouraud / flat shading
///    27         1/0    polyline / single line
///    25         1/0    semi-transparent / opaque
///   23-0        rgb    first color value.
/// ```
#[bitfield(u32, debug)]
#[derive(Default)]
pub struct DrawLineHeader {
    #[bits(0..=23, rw)]
    rgb:              u24,
    #[bit(25, rw)]
    semi_transparent: bool,
    #[bit(27, rw)]
    poly:             bool,
    #[bit(28, rw)]
    shading:          Shading,
}

#[derive(Debug, Clone)]
pub struct DrawLineAttribute {
    color:  Option<u32>,
    vertex: I16Vec2,
}

#[derive(Debug, Clone, Default)]
pub enum DrawLineAttributeDecoder {
    #[default]
    Flat,
    Goraud,
    GoraudColor {
        color: u32,
    },
}

impl DrawCallDecoder for DrawLineAttributeDecoder {
    type Output = DrawLineAttribute;

    fn advance<T: Copy>(self, value: T) -> Result<Self, Self::Output> {
        let value32 = value.io_into_u32();

        let vertex = unsafe { transmute::<u32, I16Vec2>(value32) };
        match self {
            DrawLineAttributeDecoder::Flat => Err(DrawLineAttribute {
                color: None,
                vertex,
            }),
            DrawLineAttributeDecoder::Goraud => Ok(Self::GoraudColor { color: value32 }),
            DrawLineAttributeDecoder::GoraudColor { color } => Err(DrawLineAttribute {
                color: Some(color),
                vertex,
            }),
        }
    }
}

impl DrawLineAttributeDecoder {
    pub const fn from_header(header: DrawLineHeader) -> Self {
        match header.shading() {
            Shading::Flat => Self::Flat,
            Shading::Gouraud => Self::Goraud,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DrawLineDecoder {
    header:       DrawLineHeader,
    attrs:        SmallVec<[DrawLineAttribute; 2]>,
    current_attr: DrawLineAttributeDecoder,
}

impl DrawCallDecoder for DrawLineDecoder {
    type Output = DrawLine;

    #[tracing::instrument(skip_all)]
    fn advance<T: Copy>(mut self, value: T) -> Result<Self, Self::Output> {
        tracing::info!(attrs = self.attrs.len());
        let value32 = value.io_into_u32();
        if self.header.poly() {
            if value32 & 0xf000f000 == 0x50005000 {
                return Err(DrawLine {
                    header: self.header,
                    attrs:  self.attrs,
                });
            }
        }
        match self.current_attr.advance(value) {
            Ok(decoder) => {
                self.current_attr = decoder;
                Ok(self)
            }
            Err(attribute) => match (self.header.poly(), self.attrs.len()) {
                (false, 0) => {
                    self.attrs.push(attribute);
                    self.current_attr = DrawLineAttributeDecoder::from_header(self.header);
                    Ok(self)
                }
                (false, 1) => {
                    self.attrs.push(attribute);
                    Err(DrawLine {
                        header: self.header,
                        attrs:  self.attrs,
                    })
                }
                (false, _) => unreachable!(),
                (true, _) => {
                    self.attrs.push(attribute);
                    self.current_attr = DrawLineAttributeDecoder::from_header(self.header);
                    Ok(self)
                }
            },
        }
    }
}

impl DrawLineDecoder {
    pub const fn new(header: u32) -> Self {
        let header = DrawLineHeader::new_with_raw_value(header);
        Self {
            header,
            attrs: SmallVec::new_const(),
            current_attr: DrawLineAttributeDecoder::from_header(header),
        }
    }
}

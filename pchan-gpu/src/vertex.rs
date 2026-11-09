use arbitrary_int::prelude::*;
use core::mem::offset_of;

use glam::prelude::*;
use pchan_emu::gpu::{
    Gp0TexWindowCmd, GpuStatReg, TextureColorMode,
    vram::{IVramCoord, VramCoord},
};
use wgpu::*;

// FIXME: pack this better
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Vertex {
    pub pos: I16Vec2,

    // these 2 must be packed together
    color: U8Vec3,
    color_mode: TextureColorMode,

    uv: U16Vec2,
    clut: U16Vec2,
    texpage_base: U8Vec2,
    flags: Flags,
    draw_area_top_left: VramCoord,
    draw_area_bottom_right: VramCoord,
    draw_offset: IVramCoord,
    gpustat: GpuStatReg,
    tex_window: Gp0TexWindowCmd,
}

pub struct VertexSpec {
    pub pos: I16Vec2,
    pub color: U8Vec3,
    pub color_mode: TextureColorMode,
    pub uv: U16Vec2,
    pub clut: U16Vec2,
    pub texpage_base: U8Vec2,
    pub flags: Flags,
    pub draw_area_top_left: VramCoord,
    pub draw_area_bottom_right: VramCoord,
    pub draw_offset: IVramCoord,
    pub gpustat: GpuStatReg,
    pub tex_window: Gp0TexWindowCmd,
}

impl Vertex {
    pub fn new(spec: VertexSpec) -> Self {
        Self {
            pos: spec.pos,
            color: spec.color,
            color_mode: spec.color_mode,
            uv: spec.uv,
            clut: spec.clut,
            texpage_base: spec.texpage_base,
            flags: spec.flags,
            draw_area_top_left: spec.draw_area_top_left,
            draw_area_bottom_right: spec.draw_area_bottom_right,
            draw_offset: spec.draw_offset,
            gpustat: spec.gpustat,
            tex_window: spec.tex_window,
        }
    }
    pub fn desc() -> &'static [VertexAttribute] {
        &[
            // position @location(0)
            VertexAttribute {
                format: VertexFormat::Sint16x2,
                offset: 0x0,
                shader_location: 0,
            },
            // color_and_mode @location(1)
            VertexAttribute {
                format: VertexFormat::Uint32,
                offset: offset_of!(Vertex, color) as _,
                shader_location: 1,
            },
            // clut @location(2)
            VertexAttribute {
                format: VertexFormat::Uint16x2,
                offset: offset_of!(Vertex, clut) as _,
                shader_location: 2,
            },
            // uv @location(3)
            VertexAttribute {
                format: VertexFormat::Uint16x2,
                offset: offset_of!(Vertex, uv) as _,
                shader_location: 3,
            },
            // texpage_base @location(4)
            VertexAttribute {
                format: VertexFormat::Uint8x2,
                offset: offset_of!(Vertex, texpage_base) as _,
                shader_location: 4,
            },
            // flags @location(5)
            VertexAttribute {
                format: VertexFormat::Uint8,
                offset: offset_of!(Vertex, flags) as _,
                shader_location: 5,
            },
            // draw_area_top_left @location(6)
            VertexAttribute {
                format: VertexFormat::Uint16x2,
                offset: offset_of!(Vertex, draw_area_top_left) as _,
                shader_location: 6,
            },
            // draw_area_bottom_right @location(7)
            VertexAttribute {
                format: VertexFormat::Uint16x2,
                offset: offset_of!(Vertex, draw_area_bottom_right) as _,
                shader_location: 7,
            },
            // draw_offset @location(8)
            VertexAttribute {
                format: VertexFormat::Sint16x2,
                offset: offset_of!(Vertex, draw_offset) as _,
                shader_location: 8,
            },
            // gpustat @location(9)
            VertexAttribute {
                format: VertexFormat::Uint32,
                offset: offset_of!(Vertex, gpustat) as _,
                shader_location: 9,
            },
            // tex_window @location(10)
            VertexAttribute {
                format: VertexFormat::Uint32,
                offset: offset_of!(Vertex, tex_window) as _,
                shader_location: 10,
            },
        ]
    }
}

#[bitbybit::bitfield(u8, debug)]
#[derive(Default)]
pub struct Flags {
    #[bit(0, rw)]
    dither: bool,
    #[bit(1, rw)]
    set_mask: bool,
    #[bit(2, rw)]
    quick_fill: bool,
    #[bit(3, rw)]
    textured: bool,
    #[bit(4, rw)]
    modulation: bool,
    #[bits(5..=6, rw)]
    semi_transparency: u2,
    #[bit(7, rw)]
    stp_on: bool,
}

impl Flags {
    pub const fn from_gpustat(gpustat: GpuStatReg) -> Self {
        Self::ZERO
            .with_set_mask(gpustat.set_mask())
            .with_dither(gpustat.dither())
            .with_semi_transparency(gpustat.semi_transparency())
    }

    pub const fn with_optional_semi_transparency(self, stp: Option<u2>) -> Self {
        match stp {
            Some(stp) => self.with_semi_transparency(stp),
            None => self,
        }
    }
}
impl From<VramCoord> for Vertex {
    fn from(value: VramCoord) -> Self {
        Self {
            pos: i16vec2(value.x as i16, value.y as i16),
            ..Default::default()
        }
    }
}

impl Vertex {
    const SIZE: u64 = size_of::<Self>() as u64;

    pub fn with_pos(mut self, pos: I16Vec2) -> Self {
        self.pos = pos;
        self
    }
    pub fn with_color(mut self, color: U8Vec3) -> Self {
        self.color = color;
        self
    }
    pub fn with_uv(mut self, uv: U16Vec2) -> Self {
        self.uv = uv;
        self
    }
    pub fn with_clut(mut self, clut: U16Vec2) -> Self {
        self.clut = clut;
        self
    }
    pub fn repeat_tex_window(size: U16Vec2) -> U8Vec2 {
        size.as_u8vec2()
    }
    pub fn with_color_mode(mut self, mode: TextureColorMode) -> Self {
        self.color_mode = mode;
        self
    }
}

#[derive(Debug)]
pub struct Quad {
    top_left: Vertex,
    top_right: Vertex,
    bottom_left: Vertex,
    bottom_right: Vertex,
}

impl Quad {
    pub fn new(top_left: Vertex, size: U16Vec2) -> Self {
        let tex_size = size;
        Quad {
            top_left,
            top_right: top_left
                .with_pos(top_left.pos + i16vec2(size.x as i16, 0))
                .with_uv(top_left.uv.wrapping_add(u16vec2(tex_size.x, 0))),
            bottom_left: top_left
                .with_pos(top_left.pos + i16vec2(0, size.y as i16))
                .with_uv(top_left.uv.wrapping_add(u16vec2(0, tex_size.y))),
            bottom_right: top_left
                .with_pos(top_left.pos + i16vec2(size.x as i16, size.y as i16))
                .with_uv(top_left.uv.wrapping_add(u16vec2(tex_size.x, tex_size.y))),
        }
    }
    pub fn vertices(self) -> [Vertex; 4] {
        [
            self.top_left,
            self.top_right,
            self.bottom_left,
            self.bottom_right,
        ]
    }
    fn triangulate(self) -> [Vertex; 6] {
        [
            self.top_left,
            self.top_right,
            self.bottom_left,
            self.bottom_left,
            self.top_right,
            self.bottom_right,
        ]
    }
}

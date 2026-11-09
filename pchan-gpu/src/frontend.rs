use arbitrary_int::prelude::*;
use pchan_emu::gpu::vram::VramCoord;
use pchan_emu::memory::mb;
use pchan_utils::default;
use thiserror::Error;

use core::alloc::Allocator;
use core::num::NonZeroU64;
use core::ops::Range;
use core::{mem, slice};
use std::sync::MutexGuard;
use wgpu::util::StagingBelt;

use glam::prelude::*;
use pchan_emu::gpu::draw_call::{
    DrawCall, DrawCallKind, DrawPolygon, DrawRect, Gp0HostBlit, Gp0QuickFill, Gp0VramBlit,
    GpuInternalDrawReg, RectSize, Shading,
};
use pchan_emu::gpu::{Gp0TexWindowCmd, GpuStatReg, TextureColorMode};
pub use wgpu;
use wgpu::*;

use crate::utils::{Aabb, ensure_vertex_order, triangulate_quad};
use crate::vertex::{Flags, Quad, Vertex, VertexSpec};
use crate::{BlitUniforms, HWDisplayUniforms};

#[derive(derive_more::Debug, Default)]
pub enum Command<A: Allocator> {
    DrawGeometry(GeometryDraw<A>),
    VramToVramBlit {
        blit: Gp0VramBlit,
        gpustat: GpuStatReg,
    },
    CpuToVram(CpuToVramBlitBatch<A>),
    #[default]
    Nop,
}

#[derive(derive_more::Debug, Clone)]
pub struct Scene<A: Allocator> {
    vertex_buf: Vec<Vertex, A>,
    pub vertex_buf_start: u32,
    pub aabb: Aabb,
}

#[derive(Error, Debug)]
enum DrawRectError {
    #[error("draw rect: malformed call, missing draw size")]
    MissingVarSize,
}

impl<A: Allocator> Scene<A> {
    pub fn vertex_bytes(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                self.vertex_buf.as_ptr().cast::<u8>(),
                mem::size_of_val(self.vertex_buf.as_slice()),
            )
        }
    }

    fn add_draw_rect_draw_call(
        &mut self,
        draw_rect: &DrawRect,
        gpustat: GpuStatReg,
        draw_reg: &GpuInternalDrawReg,
        tex_window: Gp0TexWindowCmd,
    ) -> Result<usize, DrawRectError> {
        let rgb = draw_rect.header.rgb().to_ne_bytes();
        let color_mode = match draw_rect.header.textured() {
            true => gpustat.texpage_colors(),
            false => TextureColorMode::C24BitDirect,
        };
        let color = U8Vec3::from_array(rgb);
        let uv = draw_rect.uv.unwrap_or_default();
        let flags = Flags::from_gpustat(gpustat)
            .with_textured(draw_rect.header.textured())
            .with_modulation(!draw_rect.header.raw())
            .with_stp_on(draw_rect.header.semi_transparent());
        let top_left = Vertex::new(VertexSpec {
            pos: i16vec2(draw_rect.vertex1.x as i16, draw_rect.vertex1.y as i16),
            color,
            color_mode,
            uv: uv.uv.as_u16vec2(),
            clut: uv.extra_as_clut(),
            texpage_base: gpustat.texpage_base(),
            flags,
            draw_area_top_left: draw_reg.draw_area_top_left,
            draw_area_bottom_right: draw_reg.draw_area_bottom_right,
            draw_offset: draw_reg.draw_offset,
            gpustat,
            tex_window,
        });

        let quad: Quad = match (draw_rect.header.size(), draw_rect.var_size) {
            (RectSize::VarSize, None) => return Err(DrawRectError::MissingVarSize),
            (RectSize::VarSize, Some(size)) => Quad::new(top_left, u16vec2(size.x, size.y)),
            (RectSize::SinglePixel, _) => Quad::new(top_left, u16vec2(1, 1)),
            (RectSize::Sprite8x8, _) => Quad::new(top_left, u16vec2(8, 8)),
            (RectSize::Sprite16x16, _) => Quad::new(top_left, u16vec2(16, 16)),
        };
        let vertices = triangulate_quad(&quad.vertices());
        for vertex in &vertices {
            self.aabb = self.aabb.grow_to_include(vertex.pos).clamp(Aabb::VRAM);
        }
        self.vertex_buf.extend(vertices);
        Ok(self.vertex_bytes().len())
    }

    fn add_draw_polygon_draw_call(
        &mut self,
        draw_polygon: &DrawPolygon,
        gpustat: GpuStatReg,
        draw_reg: &GpuInternalDrawReg,
        tex_window: Gp0TexWindowCmd,
    ) -> usize {
        let header = draw_polygon.header;
        let clut = draw_polygon.clut;
        let texpage = draw_polygon.texpage;
        let texpage_base = texpage.unwrap_or_default().to_u8vec2();

        let color = header.color();
        let shading = header.shading();
        let color_mode = match (header.textured(), texpage) {
            (true, Some(texpage)) => texpage.texpage_colors(),
            _ => TextureColorMode::C15BitDirect,
        };
        let flags = Flags::from_gpustat(gpustat)
            .with_optional_semi_transparency(draw_polygon.semi_transparency())
            .with_modulation(header.modulation())
            .with_stp_on(header.semi_transparent())
            .with_dither(gpustat.dither() && (header.modulation() || header.goraud()))
            .with_textured(header.textured());
        let mut vertices = draw_polygon
            .attrs
            .iter()
            .map(|attr| {
                let mut spec = VertexSpec {
                    pos: attr.vertex,
                    color: U8Vec3::from_array(color.to_ne_bytes()),
                    color_mode,
                    uv: attr.uv.unwrap_or_default().uv.as_u16vec2(),
                    texpage_base,
                    clut,
                    flags,
                    draw_area_top_left: draw_reg.draw_area_top_left,
                    draw_area_bottom_right: draw_reg.draw_area_bottom_right,
                    draw_offset: draw_reg.draw_offset,
                    gpustat,
                    tex_window,
                };
                match shading {
                    Shading::Flat => {}
                    Shading::Gouraud => {
                        spec.color = attr
                            .color
                            .unwrap_or(U8Vec3::from_array(color.to_ne_bytes()));
                    }
                }
                Vertex::new(spec)
            })
            .collect::<heapless::Vec<_, 4>>();
        match header.vertex_count() {
            pchan_emu::gpu::draw_call::DrawPolygonVertexCount::Three => {
                assert_eq!(vertices.len(), 3);
                ensure_vertex_order(&mut vertices, [0, 1, 2]);
                for vertex in &vertices {
                    self.aabb = self.aabb.grow_to_include(vertex.pos).clamp(Aabb::VRAM);
                }
                self.vertex_buf.extend(vertices);
            }
            pchan_emu::gpu::draw_call::DrawPolygonVertexCount::Four => {
                assert_eq!(vertices.len(), 4);
                let vertices = triangulate_quad(&vertices);

                for vertex in &vertices {
                    self.aabb = self.aabb.grow_to_include(vertex.pos).clamp(Aabb::VRAM);
                }

                self.vertex_buf.extend_from_slice(&vertices);
            }
        };
        self.vertex_bytes().len()
    }

    fn add_quick_fill_draw_call(
        &mut self,
        fill: &Gp0QuickFill,
        gpustat: GpuStatReg,
        _: &GpuInternalDrawReg,
        tex_window: Gp0TexWindowCmd,
    ) -> usize {
        let flags = Flags::from_gpustat(gpustat)
            .with_modulation(false)
            .with_stp_on(false)
            .with_dither(false)
            .with_textured(false)
            .with_quick_fill(true);

        let topleft = Vertex::new(VertexSpec {
            pos: fill.pos.fill_cmd_pos_mask().as_i16vec2(),
            color: fill.color,
            color_mode: TextureColorMode::C15BitDirect,
            uv: default(),
            clut: default(),
            texpage_base: default(),
            flags,
            draw_area_top_left: default(),
            draw_area_bottom_right: VramCoord::new(1024, 512),
            draw_offset: default(),
            gpustat,
            tex_window,
        });

        let quad = Quad::new(topleft, fill.size.fill_cmd_size_mask().as_u16vec2());
        let mut vertices = triangulate_quad(&quad.vertices());

        ensure_vertex_order(&mut vertices, [0, 1, 2]);
        for vertex in &vertices {
            self.aabb = self.aabb.grow_to_include(vertex.pos).clamp(Aabb::VRAM);
        }
        self.vertex_buf.extend(vertices);

        self.vertex_bytes().len()
    }
}

pub struct GeometryDraw<A: Allocator> {
    pub batches: Vec<BatchedGeometryDraw, A>,
    pub scene: Scene<A>,
    dp: HWDisplayUniforms,
}

#[derive(derive_more::Debug)]
pub struct BatchedGeometryDraw {
    pub kind: BatchedGeometryDrawKind,
    pub span: Range<u32>,
}

#[derive(derive_more::Debug, PartialEq, Clone, Copy)]
pub enum BatchedGeometryDrawKind {
    OpaqueOnly,
    Transparent(Transparency),
}

#[derive(derive_more::Debug, PartialEq, Clone, Copy)]
pub enum Transparency {
    Add(f32),
    Average,
    Subtract,
}

pub struct CpuToVramBlit {
    blit: Gp0HostBlit,
    gpustat: GpuStatReg,
    pub cursor: u32,
    pub index: u32,
}

struct CpuToVramBlitBatchBuilder<A: Allocator> {
    cursor: u32,
    staging_buf: Buffer,
    staging_offset: BufferAddress,
    blits: Vec<CpuToVramBlit, A>,
    uniforms: Vec<BlitUniforms, A>,
}

pub struct CpuToVramBlitBatch<A: Allocator> {
    pub staging_buf: Buffer,
    pub staging_offset: BufferAddress,
    pub data_len: u64,
    pub blits: Vec<CpuToVramBlit, A>,
    pub uniforms: Vec<BlitUniforms, A>,
}

#[derive(derive_more::Debug)]
pub(crate) struct Pass<'b, A: Allocator> {
    pub cmds: Vec<Command<A>, A>,
    builder: PassCmdBuilder<A>,
    vertex_buf_cursor: u32,
    pub belt: MutexGuard<'b, StagingBelt>,
    #[debug(skip)]
    pub alloc: A,
}

enum PassCmdBuilder<A: Allocator> {
    None,
    Geometry(GeometryBuilder<A>),
    CpuToVramBlit(CpuToVramBlitBatchBuilder<A>),
}

#[derive(derive_more::Debug)]
struct GeometryBuilder<A: Allocator> {
    builder: BatchedGeometryDraw,
    batches: Vec<BatchedGeometryDraw, A>,
    scene: Scene<A>,
    aabbs: Vec<Aabb, A>,
    dp: HWDisplayUniforms,
}

impl<A: Allocator + Copy> Scene<A> {
    fn empty(at: u32, alloc: A) -> Self {
        Self {
            vertex_buf: Vec::with_capacity_in(512, alloc),
            vertex_buf_start: at,
            aabb: Aabb::void(),
        }
    }
    fn from_draw_call(at: u32, draw_call: DrawCall, alloc: A) -> Option<Self> {
        let mut scene = Scene {
            vertex_buf: Vec::new_in(alloc),
            vertex_buf_start: at,
            aabb: Aabb::void(),
        };
        match draw_call.inner {
            DrawCallKind::Rect(draw_rect) => {
                scene
                    .add_draw_rect_draw_call(
                        &draw_rect,
                        draw_call.gpustat,
                        &draw_call.draw_reg,
                        draw_call.tex_window,
                    )
                    .unwrap();
                Some(scene)
            }
            DrawCallKind::Polygon(draw_polygon) => {
                scene.add_draw_polygon_draw_call(
                    &draw_polygon,
                    draw_call.gpustat,
                    &draw_call.draw_reg,
                    draw_call.tex_window,
                );
                Some(scene)
            }
            DrawCallKind::Line(_draw_line) => {
                tracing::warn!("todo: DrawCall Line");
                Some(scene)
            }
            _ => None,
        }
    }
}

impl From<u2> for Transparency {
    fn from(value: u2) -> Self {
        match value.value() {
            0 => Transparency::Average,
            1 => Transparency::Add(1.0),
            2 => Transparency::Subtract,
            3 => Transparency::Add(0.25),
            _ => unreachable!("u2 cannot be greater than 3"),
        }
    }
}

fn draw_call_transparency(draw: &DrawCall) -> Option<Transparency> {
    if draw.inner.is_transparent() {
        let gpustat_mode = Transparency::from(draw.gpustat.semi_transparency());
        match &draw.inner {
            DrawCallKind::Polygon(poly) => Some(
                poly.semi_transparency()
                    .map_or(gpustat_mode, Transparency::from),
            ),
            _ => Some(gpustat_mode),
        }
    } else {
        None
    }
}

impl From<&DrawCall> for BatchedGeometryDrawKind {
    fn from(value: &DrawCall) -> Self {
        match draw_call_transparency(value) {
            Some(mode) => BatchedGeometryDrawKind::Transparent(mode),
            None => BatchedGeometryDrawKind::OpaqueOnly,
        }
    }
}

impl BatchedGeometryDraw {
    fn new(start: u32, kind: BatchedGeometryDrawKind) -> Self {
        BatchedGeometryDraw {
            kind,
            span: start..start,
        }
    }
}

impl<A: Allocator + Copy> GeometryBuilder<A> {
    fn new(start: u32, dp: HWDisplayUniforms, alloc: A) -> Self {
        Self {
            builder: BatchedGeometryDraw::new(start, BatchedGeometryDrawKind::OpaqueOnly),
            batches: Vec::with_capacity_in(16, alloc),
            scene: Scene::empty(start, alloc),
            aabbs: Vec::with_capacity_in(512, alloc),
            dp,
        }
    }
    fn add_draw_call(&mut self, draw_call: DrawCall) {
        match &mut self.builder.kind {
            BatchedGeometryDrawKind::OpaqueOnly => {
                if let Some(tp) = draw_call_transparency(&draw_call) {
                    self.begin_transparent(tp);
                    return self.add_draw_call(draw_call);
                }
            }
            BatchedGeometryDrawKind::Transparent(tp) => match draw_call_transparency(&draw_call) {
                Some(new_tp) if new_tp == *tp => {}
                Some(new_tp) => {
                    self.begin_transparent(new_tp);
                    return self.add_draw_call(draw_call);
                }
                None => {
                    self.begin_opaque();
                    return self.add_draw_call(draw_call);
                }
            },
        };
        match draw_call.inner {
            DrawCallKind::Rect(draw_rect) => {
                let len = self
                    .scene
                    .add_draw_rect_draw_call(
                        &draw_rect,
                        draw_call.gpustat,
                        &draw_call.draw_reg,
                        draw_call.tex_window,
                    )
                    .unwrap();
                let end = len as u32 + self.scene.vertex_buf_start;
                self.builder.span.end = end;
            }
            DrawCallKind::Polygon(draw_polygon) => {
                let len = self.scene.add_draw_polygon_draw_call(
                    &draw_polygon,
                    draw_call.gpustat,
                    &draw_call.draw_reg,
                    draw_call.tex_window,
                );
                let end = len as u32 + self.scene.vertex_buf_start;
                self.builder.span.end = end;
            }
            DrawCallKind::Line(_) => {
                tracing::warn!("todo: DrawCallKind::Line");
            }
            DrawCallKind::QuickFill(quick_fill) => {
                let len = self.scene.add_quick_fill_draw_call(
                    &quick_fill,
                    draw_call.gpustat,
                    &draw_call.draw_reg,
                    draw_call.tex_window,
                );
                let end = len as u32 + self.scene.vertex_buf_start;
                self.builder.span.end = end;
            }
            DrawCallKind::BlitVram2Vram(_)
            | DrawCallKind::BlitCpu2Vram(_, _)
            | DrawCallKind::BlitVram2Cpu(_)
            | DrawCallKind::Nop => {}
        }
    }

    fn begin_transparent(&mut self, tp: Transparency) {
        self.flush_batch();
        self.builder.kind = BatchedGeometryDrawKind::Transparent(tp);
    }

    fn begin_opaque(&mut self) {
        self.flush_batch();
        self.builder.kind = BatchedGeometryDrawKind::OpaqueOnly;
    }

    fn flush_batch(&mut self) {
        if self.builder.span.is_empty() {
            self.builder.kind = BatchedGeometryDrawKind::OpaqueOnly;
            self.aabbs.clear();
            return;
        }
        let end = self.builder.span.end;
        let draw = mem::replace(
            &mut self.builder,
            BatchedGeometryDraw::new(end, BatchedGeometryDrawKind::OpaqueOnly),
        );
        self.batches.push(draw);
        self.aabbs.clear();
    }
}

pub const CPU_TO_VRAM_BLIT_BUF_SIZE: usize = mb(2);
pub const CPU_TO_VRAM_UNIFORM_BUF_SIZE: usize = mb(2);
pub const CPU_TO_VRAM_MAX_BLITS: usize = CPU_TO_VRAM_UNIFORM_BUF_SIZE / size_of::<BlitUniforms>();

impl<A: Allocator + Copy> CpuToVramBlitBatchBuilder<A> {
    fn new(pass: &mut Pass<A>) -> Self {
        let buf = pass.belt.allocate(
            NonZeroU64::new(CPU_TO_VRAM_BLIT_BUF_SIZE as u64).unwrap(),
            NonZeroU64::new(4).unwrap(),
        );
        let offset = buf.offset();
        Self {
            cursor: 0,
            blits: Vec::with_capacity_in(64, pass.alloc),
            uniforms: Vec::with_capacity_in(64, pass.alloc),
            staging_buf: buf.buffer().clone(),
            staging_offset: offset,
        }
    }

    fn add_draw_call(
        &mut self,
        blit: Gp0HostBlit,
        data: Vec<u8>,
        gpustat: GpuStatReg,
    ) -> Result<(), Vec<u8>> {
        if self.uniforms.len() > CPU_TO_VRAM_MAX_BLITS {
            return Err(data);
        }

        let cursor = self.cursor;
        let end = cursor + data.len() as u32;

        self.cursor = end;

        let buf_start = u64::from(cursor) + self.staging_offset;
        let buf_end = u64::from(end) + self.staging_offset;
        // FIXME: fix buffer alignment issues
        self.staging_buf
            .get_mapped_range_mut(buf_start..buf_end)
            .unwrap()
            .copy_from_slice(&data);

        self.blits.push(CpuToVramBlit {
            blit,
            gpustat,
            cursor,
            index: self.blits.len() as u32,
        });
        self.uniforms
            .push(BlitUniforms::from_host_blit(blit, gpustat));

        Ok(())
    }

    fn flush(self) -> Option<CpuToVramBlitBatch<A>> {
        if self.uniforms.is_empty() || self.blits.is_empty() {
            return None;
        }
        Some(CpuToVramBlitBatch {
            staging_buf: self.staging_buf,
            staging_offset: self.staging_offset,
            blits: self.blits,
            data_len: u64::from(self.cursor),
            uniforms: self.uniforms,
        })
    }
}

impl<'b, A: Allocator + Copy> Pass<'b, A> {
    pub fn empty(belt: MutexGuard<'b, StagingBelt>, alloc: A) -> Self {
        let cmds = Vec::new_in(alloc);
        Pass {
            belt,
            cmds,
            builder: PassCmdBuilder::None,
            alloc,
            vertex_buf_cursor: 0,
        }
    }

    pub fn new_in(belt: MutexGuard<'b, StagingBelt>, alloc: A) -> Self {
        let cmds = Vec::new_in(alloc);
        Pass {
            cmds,
            builder: PassCmdBuilder::None,
            alloc,
            vertex_buf_cursor: 0,
            belt,
        }
    }

    pub fn add_draw_call(&mut self, mut draw_call: DrawCall) {
        match &mut self.builder {
            PassCmdBuilder::Geometry(builder) => match draw_call.inner {
                DrawCallKind::Nop => {}
                DrawCallKind::Rect(_)
                | DrawCallKind::Polygon(_)
                | DrawCallKind::Line(_)
                | DrawCallKind::QuickFill(_) => {
                    builder.add_draw_call(draw_call);
                }
                _ => {
                    self.flush_geometry();
                    self.add_draw_call(draw_call);
                }
            },
            PassCmdBuilder::CpuToVramBlit(builder) => match draw_call.inner {
                DrawCallKind::Nop => {}
                DrawCallKind::BlitCpu2Vram(gp0_host_blit, data) => {
                    match builder.add_draw_call(gp0_host_blit, data, draw_call.gpustat) {
                        Ok(_) => {}
                        Err(data) => {
                            tracing::warn!(
                                "ran out of space in cpu to vram blit staging buffer. maybe increase the capacity"
                            );
                            draw_call.inner = DrawCallKind::BlitCpu2Vram(gp0_host_blit, data);
                            self.flush_cpu_to_vram();
                            self.add_draw_call(draw_call);
                        }
                    }
                }
                _ => {
                    self.flush_cpu_to_vram();
                    self.add_draw_call(draw_call);
                }
            },
            PassCmdBuilder::None => match draw_call.inner {
                DrawCallKind::Nop => {}
                DrawCallKind::Rect(_)
                | DrawCallKind::Polygon(_)
                | DrawCallKind::Line(_)
                | DrawCallKind::QuickFill(_) => {
                    let dp = HWDisplayUniforms::from_hw(&draw_call.display, draw_call.gpustat);
                    let geometry_builder =
                        GeometryBuilder::new(self.vertex_buf_cursor, dp, self.alloc);
                    self.builder = PassCmdBuilder::Geometry(geometry_builder);
                    self.add_draw_call(draw_call);
                }
                DrawCallKind::BlitVram2Vram(blit) => {
                    self.cmds.push(Command::VramToVramBlit {
                        blit,
                        gpustat: draw_call.gpustat,
                    });
                }
                DrawCallKind::BlitCpu2Vram(_, _) => {
                    self.builder =
                        PassCmdBuilder::CpuToVramBlit(CpuToVramBlitBatchBuilder::new(self));
                    self.add_draw_call(draw_call)
                }
                DrawCallKind::BlitVram2Cpu(_blit) => todo!(),
            },
        }
    }

    pub fn flush_geometry(&mut self) {
        match self.builder.take() {
            PassCmdBuilder::Geometry(mut geom) => {
                geom.flush_batch();
                self.vertex_buf_cursor += geom.scene.vertex_bytes().len() as u32;
                self.cmds.push(Command::DrawGeometry(GeometryDraw {
                    batches: geom.batches,
                    scene: geom.scene,
                    dp: geom.dp,
                }));
            }
            other => {
                self.builder = other;
            }
        }
    }

    pub fn flush_cpu_to_vram(&mut self) {
        match self.builder.take() {
            PassCmdBuilder::CpuToVramBlit(builder) => {
                if let Some(cmd) = builder.flush() {
                    self.cmds.push(Command::CpuToVram(cmd));
                }
            }
            other => {
                self.builder = other;
            }
        }
    }
}

impl<A: Allocator> PassCmdBuilder<A> {
    fn take(&mut self) -> Self {
        mem::replace(self, PassCmdBuilder::None)
    }
}

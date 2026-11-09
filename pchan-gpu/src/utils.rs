use core::mem;
use glam::prelude::*;
use pchan_emu::gpu::draw_call::{DrawCall, DrawCallKind, RectSize};

use crate::Vertex;

/// PSX vertex order according to PSX SPX by Martin Korth
///
/// <https://psx-spx.consoledev.net/graphicsprocessingunitgpu/#notes>
pub fn triangulate_quad(vertices: &[Vertex]) -> [Vertex; 6] {
    let v0 = vertices[0];
    let v1 = vertices[1];
    let v2 = vertices[2];
    let v3 = vertices[3];
    [v0, v1, v2, v1, v2, v3]
}

/// Credits to jsgroth at <https://jsgroth.dev/blog/posts/ps1-diamond/#preparing-the-scene>
pub fn ensure_vertex_order(vertex_buf: &mut [Vertex], indices: [usize; 3]) {
    let [v0, v1, v2] = vertex_buf
        .get_disjoint_mut(indices)
        .expect("indices must be disjoint");
    let v0_pos = v0.pos.as_i64vec2();
    let v1_pos = v1.pos.as_i64vec2();
    let v2_pos = v2.pos.as_i64vec2();
    let cross_product_z = (v1_pos.x - v0_pos.x) * (v2_pos.y - v0_pos.y)
        - (v1_pos.y - v0_pos.y) * (v2_pos.x - v0_pos.x);
    if cross_product_z < 0 {
        mem::swap(v0, v1);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aabb {
    pub min: I16Vec2,
    pub max: I16Vec2,
}

impl Aabb {
    pub const VRAM: Self = Aabb {
        min: i16vec2(0, 0),
        max: i16vec2(1024, 512),
    };
    pub const fn void() -> Self {
        Self {
            min: I16Vec2::MAX,
            max: I16Vec2::MIN,
        }
    }
    pub const fn zero() -> Self {
        Self {
            min: I16Vec2::ZERO,
            max: I16Vec2::ZERO,
        }
    }
    pub fn new(origin: I16Vec2, size: I16Vec2) -> Self {
        debug_assert!(size.cmpge(I16Vec2::ZERO).all(), "negative size");
        let max = origin + size;
        Self { min: origin, max }
    }
    #[must_use]
    pub fn grow_to_include(self, pos: I16Vec2) -> Self {
        let min = self.min.min(pos);
        let max = self.max.max(pos);
        Self { min, max }
    }
    pub fn intersects(self, o: Aabb) -> bool {
        self.min.cmplt(o.max).all() && o.min.cmplt(self.max).all()
    }
    pub fn union(self, o: Aabb) -> Aabb {
        Aabb {
            min: self.min.min(o.min),
            max: self.max.max(o.max),
        }
    }
    pub fn clamp(self, o: Aabb) -> Aabb {
        Aabb {
            min: self.min.clamp(o.min, o.max),
            max: self.max.clamp(o.min, o.max),
        }
    }
    pub fn bounding_box_of(draw_call: &DrawCall) -> Option<Self> {
        match &draw_call.inner {
            DrawCallKind::Rect(draw_rect) => match draw_rect.header.size() {
                RectSize::VarSize => Some(Aabb::new(
                    draw_rect.vertex1.as_i16vec2(),
                    draw_rect.var_size.map_or(I16Vec2::ZERO, |c| c.as_i16vec2()),
                )),
                RectSize::SinglePixel => {
                    Some(Aabb::new(draw_rect.vertex1.as_i16vec2(), i16vec2(1, 1)))
                }
                RectSize::Sprite8x8 => {
                    Some(Aabb::new(draw_rect.vertex1.as_i16vec2(), i16vec2(8, 8)))
                }
                RectSize::Sprite16x16 => {
                    Some(Aabb::new(draw_rect.vertex1.as_i16vec2(), i16vec2(16, 16)))
                }
            },
            DrawCallKind::Polygon(draw_polygon) => Some(
                draw_polygon
                    .attrs
                    .iter()
                    .fold(Aabb::void(), |aabb, attr| aabb.grow_to_include(attr.vertex)),
            ),
            DrawCallKind::Line(draw_line) => {
                let aabb = draw_line
                    .attrs
                    .iter()
                    .fold(Aabb::void(), |aabb, attr| aabb.grow_to_include(attr.vertex));
                Some(aabb)
            }
            _ => None,
        }
    }

    pub fn to_origin_size(self) -> (I16Vec2, I16Vec2) {
        (self.min, self.max - self.min)
    }

    pub fn to_vram_origin_size(self) -> (UVec2, UVec2) {
        let (origin, size) = self.to_origin_size();
        let cast = |x: i16| u32::from(x.max(0).cast_unsigned());

        let ox = cast(origin.x);
        let oy = cast(origin.y);
        let max_w = 1024u32.saturating_sub(ox);
        let max_h = 512u32.saturating_sub(oy);

        let width = cast(size.x).next_multiple_of(4).min(max_w);
        let height = cast(size.y).next_multiple_of(4).min(max_h);

        (uvec2(ox, oy), uvec2(width, height))
    }
}

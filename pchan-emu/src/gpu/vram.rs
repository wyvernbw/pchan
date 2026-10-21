use arbitrary_int::prelude::*;
use bitbybit::*;
use core::alloc::Allocator;
use core::iter;
use core::mem::transmute;
use derive_more as d;
use glam::prelude::U16Vec2;
use pchan_encode::derive::Endecode;

use crate::gpu::{DrawPixels, GpuState, Vram, mask_bit, set_mask_bit};
use crate::memory::kb;

#[derive(Copy, PartialEq, PartialOrd, Ord, Eq, d::Mul, d::MulAssign, Endecode)]
#[derive_const(Clone, d::Add, d::AddAssign, Default, d::Debug)]
#[repr(C)]
#[mul(forward)]
#[endecode(alloc(default))]
#[debug("({x},{y})")]
pub struct VramCoord {
    pub x: u16,
    pub y: u16,
}

#[bitfield(u16)]
#[must_use]
pub struct Rgb5 {
    #[bits(0..=4, rw)]
    r:    u5,
    #[bits(5..=9, rw)]
    g:    u5,
    #[bits(10..=14, rw)]
    b:    u5,
    #[bit(15, rw)]
    mask: bool,
}

impl VramCoord {
    #[must_use]
    pub fn new(xpos: u16, ypos: u16) -> Self {
        Self { x: xpos, y: ypos }
    }
    #[must_use]
    pub fn wrap(mut self) -> Self {
        self.x %= 1024;
        self.y %= 512;
        self
    }

    #[must_use]
    pub fn copy_cmd_pos_mask(mut self) -> Self {
        self.x &= 0x3ff;
        self.y &= 0x1ff;
        self
    }

    #[must_use]
    pub fn fill_cmd_pos_mask(mut self) -> Self {
        self.x &= 0x3f0;
        self.y &= 0x1ff;
        self
    }

    #[must_use]
    pub fn copy_cmd_size_mask(mut self) -> Self {
        if self.x == 0 {
            self.x = 0x400;
        }
        if self.y == 0 {
            self.y = 0x200;
        }
        self.x = ((self.x - 1) & 0x3ff) + 1;
        self.y = ((self.y - 1) & 0x1ff) + 1;
        self
    }

    #[must_use]
    pub fn fill_cmd_size_mask(mut self) -> Self {
        self.x = ((self.x & 0x3ff) + 0x0f) & (!0x0f);
        self.y &= 0x1ff;
        self
    }

    #[must_use]
    pub fn area(&self) -> usize {
        self.x as usize * self.y as usize
    }

    #[must_use]
    pub fn as_u16vec2(&self) -> U16Vec2 {
        U16Vec2::new(self.x, self.y)
    }
}

const impl From<u32> for VramCoord {
    fn from(value: u32) -> Self {
        unsafe { transmute(value) }
    }
}

#[derive(
    Debug, Clone, Copy, d::Add, d::AddAssign, PartialEq, PartialOrd, Ord, Eq, Default, Endecode,
)]
#[repr(C)]
pub struct IVramCoord {
    pub x: i16,
    pub y: i16,
}

impl IVramCoord {
    #[must_use]
    pub fn new(xpos: i16, ypos: i16) -> Self {
        Self { x: xpos, y: ypos }
    }
}

#[derive(
    Debug, Clone, Copy, d::Add, d::AddAssign, PartialEq, PartialOrd, Ord, Eq, Endecode, Default,
)]
pub struct VramCursor {
    start:  VramCoord,
    curr:   VramCoord,
    border: VramCoord,
}

impl VramCursor {
    #[must_use]
    pub fn new(start: VramCoord, border: VramCoord) -> Self {
        Self {
            border,
            curr: start,
            start,
        }
    }
    fn next(&mut self) -> Option<VramCoord> {
        if self.done() {
            return None;
        }

        let curr = self.curr;

        self.curr.x += 1;
        if self.curr.x >= self.border.x {
            self.curr.x = self.start.x;
            self.curr.y += 1;
        }

        Some(curr.wrap())
    }

    #[must_use]
    pub fn done(&self) -> bool {
        self.curr.y >= self.border.y
    }

    pub fn iter(&mut self) -> impl Iterator<Item = VramCoord> {
        iter::from_fn(|| self.next())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        let rect = self
            .border
            .as_u16vec2()
            .saturating_sub(self.start.as_u16vec2());
        usize::from(rect.x * rect.y)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<A: Allocator> GpuState<A> {
    pub(crate) fn vram_write(&mut self, coord: VramCoord, value: u16) {
        let coord = coord.wrap();
        let addr = coord.x as usize + coord.y as usize * kb(1);
        self.vram_mutation_signal = true;
        match &mut self.vram {
            Vram::Mapped(vram) => {
                vram[addr] = value;
            }
            Vram::Lent => {
                tracing::error!("vram is lent.");
            }
        }
    }

    pub(crate) fn vram_read_direct(&mut self, coord: VramCoord) -> u16 {
        let coord = coord.wrap();
        let addr = coord.x as usize + coord.y as usize * kb(1);
        match &self.vram {
            Vram::Mapped(vram) => vram[addr],
            Vram::Lent => {
                tracing::error!("vram is lent.");
                0
            }
        }
    }

    /// returns value through `self.gp0read`
    pub(crate) fn vram_read(&mut self, coord: VramCoord, idx: usize) {
        debug_assert!((0..=1).contains(&idx));
        let value = self.vram_read_direct(coord);
        self.gp0read[idx] = value;
    }

    pub fn vram_draw(&mut self, coord: VramCoord, mut value: u16) {
        match self.gpustat.draw_pixels() {
            DrawPixels::Always => {}
            DrawPixels::NotToMaskedAreas => {
                let current = self.vram_read_direct(coord);

                if mask_bit(current) {
                    return;
                }
                if self.gpustat.set_mask() {
                    value = set_mask_bit(value, true);
                }
            }
        }
        self.vram_write(coord, value);
    }
}

impl<A: Allocator + Copy> GpuState<A> {
    pub async fn lock_vram(&mut self) -> &[u16] {
        self.task_gpu_flush_draw_calls().await;
        self.wait_for_render_result().await;
        self.vram.as_slice()
    }
}

impl<A: Allocator> Vram<A> {
    fn as_slice(&self) -> &[u16] {
        match &self {
            Vram::Mapped(items) => items,
            Vram::Lent => &[],
        }
    }
}

#[cfg(test)]
#[test]
fn test_vram_cursor() {
    let mut cursor = VramCursor::new(VramCoord::new(0, 0), VramCoord::new(2, 2));
    assert_eq!(cursor.next(), Some(VramCoord::new(0, 0)));
    assert_eq!(cursor.next(), Some(VramCoord::new(1, 0)));
    assert_eq!(cursor.next(), Some(VramCoord::new(0, 1)));
    assert_eq!(cursor.next(), Some(VramCoord::new(1, 1)));
    assert_eq!(cursor.next(), None);

    let mut cursor = VramCursor::new(VramCoord::new(0, 511), VramCoord::new(2, 512));
    assert_eq!(cursor.next(), Some(VramCoord::new(0, 511)));
    assert_eq!(cursor.next(), Some(VramCoord::new(1, 511)));
    assert_eq!(cursor.next(), None);
    // assert_eq!(cursor.next(), None);
}

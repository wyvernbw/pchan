use arbitrary_int::prelude::*;
use bitbybit::*;
use core::alloc::Allocator;
use core::iter;
use core::mem::transmute;
use derive_more as d;
use pchan_encode::derive::Endecode;
use pchan_utils::hex;

use crate::gpu::{
    DrawPixels, GpuCmd, GpuInfoCmd, GpuModel, GpuStatReg, GpuState, Vram, mask_bit, set_mask_bit,
};
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
        if self.x >= 1024 {
            self.x = 0;
        }
        if self.y >= 512 {
            self.y = 0;
        }
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

    pub fn area(&self) -> usize {
        self.x as usize * self.y as usize
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
        self.curr = self.curr.wrap();
        let curr = self.curr;

        self.curr.x += 1;
        if self.curr.x >= self.border.x {
            self.curr.x = self.start.x;
            self.curr.y += 1;
        }

        Some(curr)
    }

    #[must_use]
    pub fn done(&self) -> bool {
        self.curr.y >= self.border.y
    }

    pub fn iter(&mut self) -> impl Iterator<Item = VramCoord> {
        iter::from_fn(|| self.next())
    }
}

pub struct Read;
pub struct ReadWrite;

pub trait VramAccessType {
    type VramRef<'a>;
    type SignalRef<'a>;
}

impl VramAccessType for Read {
    type VramRef<'a> = &'a [u16];
    type SignalRef<'a> = &'a bool;
}
impl VramAccessType for ReadWrite {
    type VramRef<'a> = &'a mut [u16];
    type SignalRef<'a> = &'a mut bool;
}

#[derive(derive_more::Deref)]
pub struct VramGuard<'a, T: VramAccessType> {
    #[deref]
    vram:   T::VramRef<'a>,
    signal: T::SignalRef<'a>,
}

impl VramGuard<'_, ReadWrite> {
    fn as_readonly(&self) -> VramGuard<'_, Read> {
        VramGuard {
            vram:   self.vram,
            signal: self.signal,
        }
    }
    pub fn vram_write(&mut self, coord: VramCoord, value: u16) {
        let coord = coord.wrap();
        let addr = coord.x as usize + coord.y as usize * kb(1);
        *self.signal = true;
        self.vram[addr] = value;
    }

    /// like `vram_write`, but takes mask bit into account
    pub fn vram_draw(&mut self, coord: VramCoord, mut value: u16, gpustat: GpuStatReg) {
        match gpustat.draw_pixels() {
            DrawPixels::Always => {}
            DrawPixels::NotToMaskedAreas => {
                let current = self.vram_read(coord);

                if mask_bit(current) {
                    return;
                }
                if gpustat.set_mask() {
                    value = set_mask_bit(value, true);
                }
            }
        }
        self.vram_write(coord, value);
    }

    pub fn vram_read(&mut self, coord: VramCoord) -> u16 {
        self.as_readonly().vram_read(coord)
    }
}
impl VramGuard<'_, Read> {
    fn vram_read(&mut self, coord: VramCoord) -> u16 {
        let coord = coord.wrap();
        let addr = coord.x as usize + coord.y as usize * kb(1);
        self.vram[addr]
    }
}

impl<A: Allocator + Copy> GpuState<A> {
    pub fn get_gpu_info_cmd(&self, cmd: GpuCmd) -> Option<GpuInfoCmd> {
        let value = cmd.raw_value();
        let value = match self.model {
            GpuModel::Gpu160Pin => return None,
            GpuModel::Gpu180Pin => value & 0x7,
            GpuModel::Gpu208Pin => value & 0xf,
        };
        Some(
            GpuInfoCmd::from_repr(value as _)
                .unwrap_or_else(|| todo!("gpu get info {}", hex(value))),
        )
    }

    pub fn lock_vram(&mut self) -> VramGuard<'_, Read> {
        self.flush_draw_calls();
        self.wait_for_render_result();
        let Vram::Mapped(vram) = &self.vram else {
            panic!("unmapped");
        };
        VramGuard {
            vram,
            signal: &self.vram_mutation_signal,
        }
    }

    pub fn lock_vram_mut(&mut self) -> VramGuard<'_, ReadWrite> {
        self.flush_draw_calls();
        self.wait_for_render_result();
        let Vram::Mapped(vram) = &mut self.vram else {
            panic!("unmapped");
        };
        VramGuard {
            vram,
            signal: &mut self.vram_mutation_signal,
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

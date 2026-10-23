use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use core::alloc::Allocator;
use core::mem::offset_of;
use core::simd::prelude::*;
use core::{ptr, slice};
use glam::prelude::*;
use pchan_encode::derive::Endecode;
use std::simd::SimdElement;

use crate::Emu;
use crate::bindings::ExternFnTable;
use crate::gpu::vram::Rgb5;

#[repr(C)]
#[derive(Default, Debug, Endecode, Clone)]
pub struct Gte {
    pub data: GteDataRegs,
    pub ctrl: GteCtrlRegs,
}

/// # GTE Data Register Summary (cop2r0-31)
///
/// ```md
///   cop2r0-1   3xS16 VXY0,VZ0              Vector 0 (X,Y,Z)
///   cop2r2-3   3xS16 VXY1,VZ1              Vector 1 (X,Y,Z)
///   cop2r4-5   3xS16 VXY2,VZ2              Vector 2 (X,Y,Z)
///   cop2r6     4xU8  RGBC                  Color/code value
///   cop2r7     1xU16 OTZ                   Average Z value (for Ordering Table)
///   cop2r8     1xS16 IR0                   16bit Accumulator (Interpolate)
///   cop2r9-11  3xS16 IR1,IR2,IR3           16bit Accumulator (Vector)
///   cop2r12-15 6xS16 SXY0,SXY1,SXY2,SXYP   Screen XY-coordinate FIFO  (3 stages)
///   cop2r16-19 4xU16 SZ0,SZ1,SZ2,SZ3       Screen Z-coordinate FIFO   (4 stages)
///   cop2r20-22 12xU8 RGB0,RGB1,RGB2        Color CRGB-code/color FIFO (3 stages)
///   cop2r23    4xU8  (RES1)                Prohibited
///   cop2r24    1xS32 MAC0                  32bit Maths Accumulators (Value)
///   cop2r25-27 3xS32 MAC1,MAC2,MAC3        32bit Maths Accumulators (Vector)
///   cop2r28-29 1xU15 IRGB,ORGB             Convert RGB Color (48bit vs 15bit)
///   cop2r30-31 2xS32 LZCS,LZCR             Count Leading-Zeroes/Ones (sign bits)
/// ```
#[repr(C)]
#[derive(Default, Debug, Endecode, Clone)]
pub struct GteDataRegs {
    vxyz0:     i16x3,
    _vz0_pad:  i16,
    vxyz1:     i16x3,
    _vz1_pad:  i16,
    vxyz2:     i16x3,
    _vz2_pad:  i16,
    rgbc:      u8x4,
    otz:       u16,
    _otz_pad:  u16,
    ///   cop2r8     1xS16 IR0                   16bit Accumulator (Interpolate)
    ir0:       i32,
    ///   cop2r9-11  3xS16 IR1,IR2,IR3           16bit Accumulator (Vector)
    irv:       i32x3,
    sxy_fifo:  [i16x2; 4],
    sz_fifo:   [u32; 4],
    crgb_fifo: [u8x4; 3],
    res1:      U8Vec4,
    ///   cop2r24    1xS32 MAC0                  32bit Maths Accumulators (Value)
    mac0:      i32,
    ///   cop2r25-27 3xS32 MAC1,MAC2,MAC3        32bit Maths Accumulators (Vector)
    macv:      i32x3,
    irgb_orgb: [u32; 2],
    lzcs_lzcr: [i32; 2],
}

const _: () = const {
    assert!(offset_of!(GteDataRegs, vxyz0) == 0);
    assert!(offset_of!(GteDataRegs, vxyz1) == 2 * 4);
    assert!(offset_of!(GteDataRegs, vxyz2) == 4 * 4);
    assert!(offset_of!(GteDataRegs, rgbc) == 6 * 4);
    assert!(offset_of!(GteDataRegs, otz) == 7 * 4);
    assert!(offset_of!(GteDataRegs, ir0) == 8 * 4);
    assert!(offset_of!(GteDataRegs, irv) == 9 * 4);
    assert!(offset_of!(GteDataRegs, sxy_fifo) == 12 * 4);
    assert!(offset_of!(GteDataRegs, sz_fifo) == 16 * 4);
    assert!(offset_of!(GteDataRegs, crgb_fifo) == 20 * 4);
    assert!(offset_of!(GteDataRegs, mac0) == 24 * 4);
    assert!(offset_of!(GteDataRegs, macv) == 25 * 4);
    assert!(offset_of!(GteDataRegs, irgb_orgb) == 28 * 4);
    assert!(offset_of!(GteDataRegs, lzcs_lzcr) == 30 * 4);
    assert!(size_of::<GteDataRegs>() == 32 * 4);
};

impl GteDataRegs {
    #[must_use]
    pub fn as_slice_mut(&mut self) -> &mut [u32] {
        unsafe {
            slice::from_raw_parts_mut(
                ptr::from_mut::<Self>(self).cast::<u32>(),
                size_of::<Self>() / size_of::<u32>(),
            )
        }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        unsafe {
            slice::from_raw_parts(
                ptr::from_ref::<Self>(self).cast::<u32>(),
                size_of::<Self>() / size_of::<u32>(),
            )
        }
    }
}

#[allow(non_camel_case_types)]
pub mod types {
    use super::*;

    pub type i64x3 = Simd<i64, 3>;
    pub type i32x3 = Simd<i32, 3>;
    pub type i16x3 = Simd<i16, 3>;
}

use types::*;

/// # GTE Control Register Summary (cop2r32-63)
///
/// ```md
///  cop2r32-36 9xS16 RT11RT12,..,RT33 Rotation matrix     (3x3)        ;cnt0-4
///  cop2r37-39 3x 32 TRX,TRY,TRZ      Translation vector  (X,Y,Z)      ;cnt5-7
///  cop2r40-44 9xS16 L11L12,..,L33    Light source matrix (3x3)        ;cnt8-12
///  cop2r45-47 3x 32 RBK,GBK,BBK      Background color    (R,G,B)      ;cnt13-15
///  cop2r48-52 9xS16 LR1LR2,..,LB3    Light color matrix source (3x3)  ;cnt16-20
///  cop2r53-55 3x 32 RFC,GFC,BFC      Far color           (R,G,B)      ;cnt21-23
///  cop2r56-57 2x 32 OFX,OFY          Screen offset       (X,Y)        ;cnt24-25
///  cop2r58 BuggyU16 H                Projection plane distance.       ;cnt26
///  cop2r59      S16 DQA              Depth queing parameter A (coeff) ;cnt27
///  cop2r60       32 DQB              Depth queing parameter B (offset);cnt28
///  cop2r61-62 2xS16 ZSF3,ZSF4        Average Z scale factors          ;cnt29-30
///  cop2r63      U20 FLAG             Returns any calculation errors   ;cnt31
/// ```
#[repr(C)]
#[derive(Default, Debug, Endecode, Clone)]
pub struct GteCtrlRegs {
    rt_mat:  [i16x3; 3],
    _rt_pad: i16,
    tr_vec:  i32x3,
    ls_mat:  [i16x3; 3],
    bg_col:  i32x3,
    lc_mat:  [i16x3; 3],
    _lc_pad: i16,
    far_col: i32x3,
    of:      i32x2,

    /// cop2r58 `BuggyU16` H
    /// Projection plane distance.
    /// ;cnt26
    h:      u16,
    _h_pad: u16,

    dqa:      i16,
    _dqa_pad: i16,
    dqb:      i32,

    zsf3:      i16,
    _zsf3_pad: i16,
    zsf4:      i16,
    _zsf4_pad: i16,

    flag: Flag,
}

const _: () = const {
    assert!(offset_of!(GteCtrlRegs, rt_mat) == 0);
    assert!(offset_of!(GteCtrlRegs, tr_vec) == 5 * 4);
    assert!(offset_of!(GteCtrlRegs, ls_mat) == 8 * 4);
    assert!(offset_of!(GteCtrlRegs, bg_col) == 13 * 4);
    assert!(offset_of!(GteCtrlRegs, lc_mat) == 16 * 4);
    assert!(offset_of!(GteCtrlRegs, far_col) == 21 * 4);
    assert!(offset_of!(GteCtrlRegs, of) == 24 * 4);
    assert!(offset_of!(GteCtrlRegs, h) == 26 * 4);
    assert!(offset_of!(GteCtrlRegs, dqa) == 27 * 4);
    assert!(offset_of!(GteCtrlRegs, dqb) == 28 * 4);
    assert!(offset_of!(GteCtrlRegs, zsf3) == 29 * 4);
    assert!(offset_of!(GteCtrlRegs, zsf4) == 30 * 4);
    assert!(offset_of!(GteCtrlRegs, flag) == 31 * 4);
    assert!(size_of::<GteCtrlRegs>() == 32 * 4);
};

/// # cop2r63 (cnt31) - FLAG - Returns any calculation errors.
///
/// ```md
///  31   Error Flag (Bit30..23, and 18..13 ORed together) (Read only)
///  30   MAC1 Result larger than 43 bits and positive
///  29   MAC2 Result larger than 43 bits and positive
///  28   MAC3 Result larger than 43 bits and positive
///  27   MAC1 Result larger than 43 bits and negative
///  26   MAC2 Result larger than 43 bits and negative
///  25   MAC3 Result larger than 43 bits and negative
///  24   IR1 saturated to +0000h..+7FFFh (lm=1) or to -8000h..+7FFFh (lm=0)
///  23   IR2 saturated to +0000h..+7FFFh (lm=1) or to -8000h..+7FFFh (lm=0)
///  22   IR3 saturated to +0000h..+7FFFh (lm=1) or to -8000h..+7FFFh (lm=0)
///  21   Color-FIFO-R saturated to +00h..+FFh
///  20   Color-FIFO-G saturated to +00h..+FFh
///  19   Color-FIFO-B saturated to +00h..+FFh
///  18   SZ3 or OTZ saturated to +0000h..+FFFFh
///  17   Divide overflow. RTPS/RTPT division result saturated to max=1FFFFh
///  16   MAC0 Result larger than 31 bits and positive
///  15   MAC0 Result larger than 31 bits and negative
///  14   SX2 saturated to -0400h..+03FFh
///  13   SY2 saturated to -0400h..+03FFh
///  12   IR0 saturated to +0000h..+1000h
///  0-11 Not used (always zero) (Read only)
/// ```
#[repr(C)]
#[bitfield(u32)]
#[derive(Endecode, Debug, Default)]
struct Flag {
    #[bits(28..=30, rw)]
    macv_ovf_pos:    u3,
    #[bits(25..=27, rw)]
    macv_ovf_neg:    u3,
    #[bits(22..=24, rw)]
    irv_sat:         u3,
    #[bit(17, rw)]
    divide_overflow: bool,
    #[bit(16, rw)]
    mac0_ovf_pos:    bool,
    #[bit(15, rw)]
    mac0_ovf_neg:    bool,
    #[bit(12, rw)]
    ir0_sat:         bool,
}

impl GteCtrlRegs {
    #[must_use]
    pub fn as_slice_mut(&mut self) -> &mut [u32] {
        unsafe {
            slice::from_raw_parts_mut(
                ptr::from_mut::<Self>(self).cast::<u32>(),
                size_of::<Self>() / size_of::<u32>(),
            )
        }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        unsafe {
            slice::from_raw_parts(
                ptr::from_ref::<Self>(self).cast::<u32>(),
                size_of::<Self>() / size_of::<u32>(),
            )
        }
    }
}

/// # GTE Command Encoding (COP2 imm25 opcodes)
///
/// ```md
///  31-25  Must be 0100101b for "COP2 imm25" instructions
///  20-24  Fake GTE Command Number (00h..1Fh) (ignored by hardware)
///  19     sf - Shift Fraction in IR registers (0=No fraction, 1=12bit fraction)
///  17-18  MVMVA Multiply Matrix    (0=Rotation. 1=Light, 2=Color, 3=Reserved)
///  15-16  MVMVA Multiply Vector    (0=V0, 1=V1, 2=V2, 3=IR/long)
///  13-14  MVMVA Translation Vector (0=TR, 1=BK, 2=FC/Bugged, 3=None)
///  11-12  Always zero                        (ignored by hardware)
///  10     lm - Saturate IR1,IR2,IR3 result (0=To -8000h..+7FFFh, 1=To 0..+7FFFh)
///  6-9    Always zero                        (ignored by hardware)
///  0-5    Real GTE Command Number (00h..3Fh) (used by hardware)
/// ```
#[bitfield(u32)]
#[must_use]
pub struct GteCommandField {
    #[bits(0..=5, r)]
    cmd:           Option<GteCmd>,
    #[bit(10, r)]
    lm:            SaturateMode,
    #[bits(13..=14, r)]
    mvmva_tr_vec:  MVMVATranslationVec,
    #[bits(15..=16, r)]
    mvmva_mul_vec: MVMVAMultiplyVec,
    #[bits(17..=18, r)]
    mvmva_mul_mat: MVMVAMultiplyMat,
    #[bit(19, r)]
    sf:            bool,
}

#[bitenum(u1, exhaustive = true)]
enum SaturateMode {
    Signed,
    Unsigned,
}

///  13-14  MVMVA Translation Vector (0=TR, 1=BK, 2=FC/Bugged, 3=None)
#[bitenum(u2, exhaustive = true)]
enum MVMVATranslationVec {
    TR,
    BK,
    FCBugged,
    None,
}

///  15-16  MVMVA Multiply Vector    (0=V0, 1=V1, 2=V2, 3=IR/long)
#[bitenum(u2, exhaustive = true)]
enum MVMVAMultiplyVec {
    V0,
    V1,
    V2,
    IRLong,
}

///  17-18  MVMVA Multiply Matrix    (0=Rotation. 1=Light, 2=Color, 3=Reserved)
#[bitenum(u2, exhaustive = true)]
enum MVMVAMultiplyMat {
    Rotation,
    Light,
    Color,
    Reserved,
}

const fn test() {
    const C: GteCommandField = GteCommandField::new_with_raw_value(0x03400006);
    const N: Result<GteCmd, UInt<u8, 6>> = C.cmd();
}

/// # GTE Command Summary (sorted by Real Opcode bits) (bit0-5)
///
/// ```md
///   Opc  Name   Clk Expl.
///   00h  -          N/A (modifies similar registers than RTPS...)
///   01h  RTPS   15  Perspective Transformation single
///   0xh  -          N/A
///   06h  NCLIP  8   Normal clipping
///   0xh  -          N/A
///   0Ch  OP(sf) 6   Cross product of 2 vectors
///   0xh  -          N/A
///   10h  DPCS   8   Depth Cueing single
///   11h  INTPL  8   Interpolation of a vector and far color vector
///   12h  MVMVA  8   Multiply vector by matrix and add vector (see below)
///   13h  NCDS   19  Normal color depth cue single vector
///   14h  CDP    13  Color Depth Que
///   15h  -          N/A
///   16h  NCDT   44  Normal color depth cue triple vectors
///   1xh  -          N/A
///   1Bh  NCCS   17  Normal Color Color single vector
///   1Ch  CC     11  Color Color
///   1Dh  -          N/A
///   1Eh  NCS    14  Normal color single
///   1Fh  -          N/A
///   20h  NCT    30  Normal color triple
///   2xh  -          N/A
///   28h  SQR(sf)5   Square of vector IR
///   29h  DCPL   8   Depth Cue Color light
///   2Ah  DPCT   17  Depth Cueing triple (should be fake=08h, but isn't)
///   2xh  -          N/A
///   2Dh  AVSZ3  5   Average of three Z values
///   2Eh  AVSZ4  6   Average of four Z values
///   2Fh  -          N/A
///   30h  RTPT   23  Perspective Transformation triple
///   3xh  -          N/A
///   3Dh  GPF(sf)5   General purpose interpolation
///   3Eh  GPL(sf)5   General purpose interpolation with base
///   3Fh  NCCT   39  Normal Color Color triple vector
/// ```
#[bitenum(u6)]
#[derive(PartialEq, Eq, Debug)]
enum GteCmd {
    Rtps  = 0x01,
    Nclip = 0x06,
    Op    = 0x0c,
    Dpcs  = 0x10,
    Intpl = 0x11,
    Mvmva = 0x12,
    Ncds  = 0x13,
    Cdp   = 0x14,
    Ncdt  = 0x16,
    Nccs  = 0x1b,
    Cc    = 0x1c,
    Ncs   = 0x1e,
    Nct   = 0x20,
    Sqr   = 0x28,
    Dcpl  = 0x29,
    Dpct  = 0x2a,
    Avsz3 = 0x2d,
    Avsz4 = 0x2e,
    Rtpt  = 0x30,
    Gpf   = 0x3d,
    Gpl   = 0x3e,
    Ncct  = 0x3f,
}

type GteFn<A> = fn(&mut Emu<A>, GteCommandField);

#[repr(C)]
#[derive(Debug, Clone)]
pub struct GteBindings<A: Allocator> {
    rtpt:  GteFn<A>,
    rtps:  GteFn<A>,
    nclip: GteFn<A>,
    ncs:   GteFn<A>,
    nct:   GteFn<A>,
    nccs:  GteFn<A>,
    ncct:  GteFn<A>,
    ncds:  GteFn<A>,
    ncdt:  GteFn<A>,
    avsz3: GteFn<A>,
    avsz4: GteFn<A>,
    mvmva: GteFn<A>,

    write_sxyp: fn(&mut Emu<A>),
    write_irgb: fn(&mut Emu<A>, u32),
    write_lzcs: fn(&mut Emu<A>, i32),
}

impl<A: Allocator + Copy> GteBindings<A> {
    pub const OFFSET: u32 = ExternFnTable::<A>::OFFSET + offset_of!(ExternFnTable<A>, gte) as u32;
    pub const RTPT: u32 = offset_of!(Self, rtpt) as u32 + Self::OFFSET;
    pub const RTPS: u32 = offset_of!(Self, rtps) as u32 + Self::OFFSET;
    pub const NCLIP: u32 = offset_of!(Self, nclip) as u32 + Self::OFFSET;
    pub const NCS: u32 = offset_of!(Self, ncs) as u32 + Self::OFFSET;
    pub const NCT: u32 = offset_of!(Self, nct) as u32 + Self::OFFSET;
    pub const NCCS: u32 = offset_of!(Self, nccs) as u32 + Self::OFFSET;
    pub const NCCT: u32 = offset_of!(Self, ncct) as u32 + Self::OFFSET;
    pub const NCDS: u32 = offset_of!(Self, ncds) as u32 + Self::OFFSET;
    pub const NCDT: u32 = offset_of!(Self, ncdt) as u32 + Self::OFFSET;
    pub const AVSZ3: u32 = offset_of!(Self, avsz3) as u32 + Self::OFFSET;
    pub const AVSZ4: u32 = offset_of!(Self, avsz4) as u32 + Self::OFFSET;
    pub const MVMVA: u32 = offset_of!(Self, mvmva) as u32 + Self::OFFSET;
    pub const WRITE_SXYP: u32 = offset_of!(Self, write_sxyp) as u32 + Self::OFFSET;
    pub const WRITE_IRGB: u32 = offset_of!(Self, write_irgb) as u32 + Self::OFFSET;
    pub const WRITE_LZCS: u32 = offset_of!(Self, write_lzcs) as u32 + Self::OFFSET;

    pub(crate) const fn new() -> Self {
        Self {
            rtpt:  Emu::gte_rtpt,
            rtps:  Emu::gte_rtps,
            nclip: Emu::gte_nclip,
            ncs:   Emu::gte_ncs,
            nct:   Emu::gte_nct,
            nccs:  Emu::gte_nccs,
            ncct:  Emu::gte_ncct,
            ncds:  Emu::gte_ncds,
            ncdt:  Emu::gte_ncdt,
            avsz3: Emu::gte_avsz3,
            avsz4: Emu::gte_avsz4,
            mvmva: Emu::gte_mvmva,

            write_sxyp: Emu::gte_write_sxyp,
            write_irgb: Emu::gte_write_irgb,
            write_lzcs: Emu::gte_write_lzcs,
        }
    }
}

impl<A: Allocator + Copy> Default for GteBindings<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl Gte {
    #[must_use]
    pub fn as_slice_mut(&mut self) -> &mut [u32] {
        unsafe {
            slice::from_raw_parts_mut(
                ptr::from_mut::<Self>(self).cast::<u32>(),
                (size_of::<GteDataRegs>() + size_of::<GteCtrlRegs>()) / size_of::<u32>(),
            )
        }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        unsafe {
            slice::from_raw_parts(
                ptr::from_ref::<Self>(self).cast::<u32>(),
                (size_of::<GteDataRegs>() + size_of::<GteCtrlRegs>()) / size_of::<u32>(),
            )
        }
    }

    fn advance_sz_fifo(&mut self) {
        self.data.sz_fifo.rotate_left(1);
    }

    fn advance_sxy_fifo(&mut self) {
        self.data.sxy_fifo.rotate_left(1);
        self.data.sxy_fifo[3] = self.data.sxy_fifo[2];
    }

    fn advance_crgb_fifo(&mut self) {
        self.data.crgb_fifo.rotate_left(1);
    }

    fn clear_flag(&mut self) {
        self.ctrl.flag = Flag::default();
    }

    fn shift_macv(&mut self, macv: i64x3, sf: bool) -> i32x3 {
        self.cast_macv(macv);
        (macv >> (i64::from(sf) * 12)).cast()
    }

    fn cast_macv(&mut self, macv: i64x3) -> i32x3 {
        let ovf_pos = macv.simd_gt(i64x3::splat(1 << 43));
        let ovf_neg = macv.simd_lt(i64x3::splat(-1 << 43));
        let old_ovf_pos = self.ctrl.flag.macv_ovf_pos();
        let old_ovf_neg = self.ctrl.flag.macv_ovf_neg();
        self.ctrl
            .flag
            .set_macv_ovf_pos(old_ovf_pos | ovf_pos.to_bitmask().as_::<u3>().reverse_bits());
        self.ctrl
            .flag
            .set_macv_ovf_neg(old_ovf_neg | ovf_neg.to_bitmask().as_::<u3>().reverse_bits());
        macv.cast()
    }

    fn cast_mac0(&mut self, mac0: i64) -> i32 {
        let ovf_pos = mac0 > i64::from(i32::MAX);
        let ovf_neg = mac0 < i64::from(i32::MIN);
        let old_ovf_pos = self.ctrl.flag.mac0_ovf_pos();
        let old_ovf_neg = self.ctrl.flag.mac0_ovf_neg();
        self.ctrl.flag.set_mac0_ovf_pos(old_ovf_pos | ovf_pos);
        self.ctrl.flag.set_mac0_ovf_neg(old_ovf_neg | ovf_neg);
        mac0 as i32
    }

    fn sat_ir0(&mut self, ir0: i64) -> i32 {
        let new_ir0 = ir0.clamp(0, 0x1000);
        let flag = new_ir0 != ir0;
        let old_flag = self.ctrl.flag.ir0_sat();
        self.ctrl.flag.set_ir0_sat(flag | old_flag);
        new_ir0 as i32
    }

    fn sat_irv<S: SimdInt<Cast<i64> = Simd<i64, 3>>>(
        &mut self,
        irv: S,
        cmd: GteCommandField,
    ) -> i16x3 {
        let irv = irv.cast::<i64>();
        let new_irv = match (cmd.cmd(), cmd.lm()) {
            (Ok(GteCmd::Rtps | GteCmd::Rtpt), lm) => {
                let (min, max) = match lm {
                    SaturateMode::Signed => (-0x8000, 0x7fff),
                    SaturateMode::Unsigned => (0x0, 0x7fff),
                };
                let min = i64x3::from_array([min, min, -0x8000]);
                let max = i64x3::splat(max);
                irv.simd_clamp(min, max)
            }
            (_, SaturateMode::Signed) => irv.simd_clamp(Simd::splat(-0x8000), Simd::splat(0x7fff)),
            (_, SaturateMode::Unsigned) => irv.simd_clamp(Simd::splat(-0x0), Simd::splat(0x7fff)),
        };
        let sat_flags = new_irv.simd_ne(irv).to_bitmask().as_::<u3>().reverse_bits();
        let old_sat_flags = self.ctrl.flag.irv_sat();
        self.ctrl.flag.set_irv_sat(old_sat_flags | sat_flags);
        new_irv.cast()
    }
}

impl GteCommandField {
    fn saturate(self, value: impl Into<i64>) -> i16 {
        match self.lm() {
            SaturateMode::Signed => value.into().clamp(-0x8000, 0x7fff).as_i16(),
            SaturateMode::Unsigned => value.into().clamp(0x0, 0x7fff).as_i16(),
        }
    }

    fn saturate_vec<const N: usize, S: SimdInt<Cast<i64> = Simd<i64, N>>>(
        self,
        value: S,
    ) -> Simd<i16, N>
    where
        S::Cast<i64>: Ord,
    {
        match self.lm() {
            SaturateMode::Signed => value
                .cast::<i64>()
                .simd_clamp(Simd::splat(-0x8000), Simd::splat(0x7fff))
                .cast(),
            SaturateMode::Unsigned => value
                .cast::<i64>()
                .simd_clamp(Simd::splat(-0x0), Simd::splat(0x7fff))
                .cast(),
        }
    }
}

impl<A: Allocator> Emu<A> {
    /// ```plain
    /// IR1 = MAC1 = (TRX*1000h + RT11*VX0 + RT12*VY0 + RT13*VZ0) SAR (sf*12)
    /// IR2 = MAC2 = (TRY*1000h + RT21*VX0 + RT22*VY0 + RT23*VZ0) SAR (sf*12)
    /// IR3 = MAC3 = (TRZ*1000h + RT31*VX0 + RT32*VY0 + RT33*VZ0) SAR (sf*12)
    /// SZ3 = MAC3 SAR ((1-sf)*12)                           ;ScreenZ FIFO 0..+FFFFh
    /// MAC0=(((H*20000h/SZ3)+1)/2)*IR1+OFX, SX2=MAC0/10000h ;ScrX FIFO -400h..+3FFh
    /// MAC0=(((H*20000h/SZ3)+1)/2)*IR2+OFY, SY2=MAC0/10000h ;ScrY FIFO -400h..+3FFh
    /// MAC0=(((H*20000h/SZ3)+1)/2)*DQA+DQB, IR0=MAC0/1000h  ;Depth cueing 0..+1000h
    /// ```
    fn rtps(&mut self, cmd: GteCommandField, v: i16x3) {
        let gte = &mut self.cpu.gte;

        let mac = {
            let rt = gte.ctrl.rt_mat;
            let tr = gte.ctrl.tr_vec;
            let mac = tr.cast::<i64>() * i64x3::splat(0x1000) + matmul3x3v(rt, v);
            gte.shift_macv(mac, cmd.sf())
        };

        gte.data.macv = mac;
        gte.data.irv = gte.sat_irv(mac, cmd).cast();

        let sz3 = mac[2] >> (u8::from(!cmd.sf()) * 12);
        let sz3 = sz3.clamp(0, 0xffff).as_u32();
        gte.advance_sz_fifo();
        gte.data.sz_fifo[3] = sz3;

        let div = if u32::from(gte.ctrl.h) >= sz3 * 2 {
            gte.ctrl.flag.set_divide_overflow(true);
            0x1ffff
        } else {
            ((i64::from(gte.ctrl.h) * 0x20000 / i64::from(sz3)) + 1) as i32 / 2
        };
        let div = div.as_i32();

        let mac0v = i64x3::splat(div.as_i64())
            * i64x3::from_array([
                gte.data.irv[0].as_i64(),
                gte.data.irv[1].as_i64(),
                gte.ctrl.dqa.as_i64(),
            ])
            + i64x3::from_array([
                gte.ctrl.of[0].as_i64(),
                gte.ctrl.of[1].as_i64(),
                gte.ctrl.dqb.as_i64(),
            ]);

        let sxy2 = mac0v.resize::<2>(0) >> 16;
        let sxy2 = sxy2.simd_clamp(i64x2::splat(-0x400), i64x2::splat(0x3ff));
        gte.advance_sxy_fifo();
        gte.data.sxy_fifo[2] = sxy2.cast();
        gte.data.sxy_fifo[3] = gte.data.sxy_fifo[2];

        let mac0 = mac0v[2];
        gte.data.mac0 = gte.cast_mac0(mac0);
        gte.data.ir0 = gte.sat_ir0(mac0 >> 12);
    }

    fn gte_rtps(&mut self, cmd: GteCommandField) {
        self.cpu.gte.clear_flag();
        let v0 = self.cpu.gte.data.vxyz0;
        self.rtps(cmd, v0);
    }

    fn gte_rtpt(&mut self, cmd: GteCommandField) {
        self.cpu.gte.clear_flag();
        self.rtps(cmd, self.cpu.gte.data.vxyz0);
        self.rtps(cmd, self.cpu.gte.data.vxyz1);
        self.rtps(cmd, self.cpu.gte.data.vxyz2);
    }

    /// ```plain
    /// MAC0 =   SX0*SY1 + SX1*SY2 + SX2*SY0 - SX0*SY2 - SX1*SY0 - SX2*SY1
    /// ```
    fn gte_nclip(&mut self, _: GteCommandField) {
        let gte = &mut self.cpu.gte;
        gte.clear_flag();

        let sxy = gte.data.sxy_fifo.map(|sxy| sxy.cast::<i64>());
        let v0 = sxy[0];
        let v1 = sxy[1];
        let v2 = sxy[2];
        let cross = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]);
        gte.data.mac0 = gte.cast_mac0(cross);
    }

    /// ```plain
    /// [IR1,IR2,IR3] = [MAC1,MAC2,MAC3] = (LLM*V0) SAR (sf*12)
    /// [IR1,IR2,IR3] = [MAC1,MAC2,MAC3] = (BK*1000h + LCM*IR) SAR (sf*12)
    /// [MAC1,MAC2,MAC3] = [R*IR1,G*IR2,B*IR3] SHL 4          ;<--- for NCDx/NCCx
    /// [MAC1,MAC2,MAC3] = MAC+(FC-MAC)*IR0                   ;<--- for NCDx only
    /// [MAC1,MAC2,MAC3] = [MAC1,MAC2,MAC3] SAR (sf*12)       ;<--- for NCDx/NCCx
    /// Color FIFO = [MAC1/16,MAC2/16,MAC3/16,CODE], [IR1,IR2,IR3] = [MAC1,MAC2,MAC3]
    /// ````
    fn ncxx<const C: u8>(&mut self, cmd: GteCommandField, v: i16x3) {
        let gte = &mut self.cpu.gte;
        let shift = i64::from(cmd.sf()) * 12;
        let macv = gte.shift_macv(matmul3x3v(gte.ctrl.ls_mat, v), cmd.sf());
        gte.data.macv = macv;
        gte.data.irv = gte.sat_irv(macv, cmd).cast();

        let lcmxir = matmul3x3v(gte.ctrl.lc_mat, gte.data.irv.cast::<i16>());
        let macv = gte.shift_macv(
            gte.ctrl.bg_col.cast::<i64>() * i64x3::splat(0x1000) + lcmxir,
            cmd.sf(),
        );
        gte.data.macv = macv;
        gte.data.irv = gte.sat_irv(macv, cmd).cast();

        let cmd_kind: GteCmd = const {
            match GteCmd::new_with_raw_value(u6::new(C)) {
                Ok(cmd_n) => cmd_n,
                Err(_) => panic!("not a proper gtecmd"),
            }
        };

        if matches!(
            cmd_kind,
            GteCmd::Ncds | GteCmd::Ncdt | GteCmd::Nccs | GteCmd::Ncct
        ) {
            let rgb = gte.data.rgbc.resize::<3>(0);
            let macv = (rgb.cast::<i32>() * gte.data.irv) << 4;
            gte.data.macv = macv;
        }

        if matches!(cmd_kind, GteCmd::Ncds | GteCmd::Ncdt) {
            let macv = gte.data.macv.cast::<i64>();
            let macv = macv
                + (gte.ctrl.far_col.cast::<i64>() - macv) * i64x3::splat(i64::from(gte.data.ir0));
            gte.data.macv = gte.cast_macv(macv);
        }

        if matches!(
            cmd_kind,
            GteCmd::Ncds | GteCmd::Ncdt | GteCmd::Nccs | GteCmd::Ncct
        ) {
            gte.data.macv >>= i32x3::splat(shift as i32);
        }

        let color = (gte.data.macv >> 4)
            .simd_clamp(i32x3::splat(0), i32x3::splat(255))
            .cast::<u8>()
            .resize::<4>(gte.data.rgbc[3]);
        gte.advance_crgb_fifo();
        gte.data.crgb_fifo[2] = color;
        gte.data.irv = cmd.saturate_vec(gte.data.macv).cast();
    }

    fn gte_ncs(&mut self, cmd: GteCommandField) {
        const NCS: u8 = GteCmd::Ncs.raw_value().value();
        self.cpu.gte.clear_flag();
        self.ncxx::<NCS>(cmd, self.cpu.gte.data.vxyz0);
    }
    fn gte_nct(&mut self, cmd: GteCommandField) {
        const NCT: u8 = GteCmd::Nct.raw_value().value();
        self.cpu.gte.clear_flag();
        self.ncxx::<NCT>(cmd, self.cpu.gte.data.vxyz0);
        self.ncxx::<NCT>(cmd, self.cpu.gte.data.vxyz1);
        self.ncxx::<NCT>(cmd, self.cpu.gte.data.vxyz2);
    }
    fn gte_nccs(&mut self, cmd: GteCommandField) {
        const NCCS: u8 = GteCmd::Nccs.raw_value().value();
        self.cpu.gte.clear_flag();
        self.ncxx::<NCCS>(cmd, self.cpu.gte.data.vxyz0);
    }
    fn gte_ncct(&mut self, cmd: GteCommandField) {
        const NCCT: u8 = GteCmd::Ncct.raw_value().value();
        self.cpu.gte.clear_flag();
        self.ncxx::<NCCT>(cmd, self.cpu.gte.data.vxyz0);
        self.ncxx::<NCCT>(cmd, self.cpu.gte.data.vxyz1);
        self.ncxx::<NCCT>(cmd, self.cpu.gte.data.vxyz2);
    }
    fn gte_ncds(&mut self, cmd: GteCommandField) {
        const NCDS: u8 = GteCmd::Ncds.raw_value().value();
        self.cpu.gte.clear_flag();
        self.ncxx::<NCDS>(cmd, self.cpu.gte.data.vxyz0);
    }
    fn gte_ncdt(&mut self, cmd: GteCommandField) {
        const NCDT: u8 = GteCmd::Ncdt.raw_value().value();
        self.cpu.gte.clear_flag();
        self.ncxx::<NCDT>(cmd, self.cpu.gte.data.vxyz0);
        self.ncxx::<NCDT>(cmd, self.cpu.gte.data.vxyz1);
        self.ncxx::<NCDT>(cmd, self.cpu.gte.data.vxyz2);
    }
    /// ```plain
    /// MAC0 =  ZSF3*(SZ1+SZ2+SZ3)       ;for AVSZ3
    /// OTZ  =  MAC0/1000h               ;for both (saturated to 0..FFFFh)
    /// ```
    fn gte_avsz3(&mut self, _: GteCommandField) {
        let gte = &mut self.cpu.gte;
        gte.clear_flag();
        let sz = gte.data.sz_fifo;
        let mac0 = gte.ctrl.zsf3.as_i64() * (sz[1] + sz[2] + sz[3]).as_i64();
        // TODO: flags
        gte.data.mac0 = gte.cast_mac0(mac0);
        gte.data.otz = (mac0 >> 12).clamp(0, 0xffff).as_u16();
    }
    /// ```plain
    /// MAC0 =  ZSF4*(SZ0+SZ1+SZ2+SZ3)   ;for AVSZ4
    /// OTZ  =  MAC0/1000h               ;for both (saturated to 0..FFFFh)
    /// ```
    fn gte_avsz4(&mut self, _: GteCommandField) {
        let gte = &mut self.cpu.gte;
        gte.clear_flag();
        let sz = gte.data.sz_fifo;
        let mac0 = gte.ctrl.zsf4.as_i64() * (sz[0] + sz[1] + sz[2] + sz[3]).as_i64();
        // TODO: flags
        gte.data.mac0 = gte.cast_mac0(mac0);
        gte.data.otz = (mac0 >> 12).clamp(0, 0xffff).as_u16();
    }

    /// Multiply vector by matrix and vector addition.
    /// ```plain
    ///   Mx = matrix specified by mx  ;RT/LLM/LCM - Rotation, light or color matrix
    ///   Vx = vector specified by v   ;V0, V1, V2, or [IR1,IR2,IR3]
    ///   Tx = translation vector specified by cv  ;TR or BK or Bugged/FC, or None
    /// ```
    /// Calculation:
    /// ```plain
    ///   MAC1 = (Tx1*1000h + Mx11*Vx1 + Mx12*Vx2 + Mx13*Vx3) SAR (sf*12)
    ///   MAC2 = (Tx2*1000h + Mx21*Vx1 + Mx22*Vx2 + Mx23*Vx3) SAR (sf*12)
    ///   MAC3 = (Tx3*1000h + Mx31*Vx1 + Mx32*Vx2 + Mx33*Vx3) SAR (sf*12)
    ///   [IR1,IR2,IR3] = [MAC1,MAC2,MAC3]
    /// ```
    fn gte_mvmva(&mut self, cmd: GteCommandField) {
        let gte = &mut self.cpu.gte;
        gte.clear_flag();
        let shift = i64::from(cmd.sf()) * 12;
        let mx = match cmd.mvmva_mul_mat() {
            MVMVAMultiplyMat::Rotation => gte.ctrl.rt_mat,
            MVMVAMultiplyMat::Light => gte.ctrl.ls_mat,
            MVMVAMultiplyMat::Color => gte.ctrl.lc_mat,
            MVMVAMultiplyMat::Reserved => return,
        };
        let vx = match cmd.mvmva_mul_vec() {
            MVMVAMultiplyVec::V0 => gte.data.vxyz0,
            MVMVAMultiplyVec::V1 => gte.data.vxyz1,
            MVMVAMultiplyVec::V2 => gte.data.vxyz2,
            MVMVAMultiplyVec::IRLong => gte.data.irv.cast(),
        };
        let tx = match cmd.mvmva_tr_vec() {
            MVMVATranslationVec::TR => gte.ctrl.tr_vec,
            MVMVATranslationVec::BK => gte.ctrl.bg_col,
            MVMVATranslationVec::FCBugged => gte.ctrl.far_col,
            MVMVATranslationVec::None => i32x3::splat(0),
        };

        let macv = (tx.cast::<i64>() * i64x3::splat(0x1000) + matmul3x3v(mx, vx)) >> shift;
        // TODO: flags
        gte.data.macv = gte.cast_macv(macv);
        gte.data.irv = gte.sat_irv(macv, cmd).cast();
    }

    fn gte_write_sxyp(&mut self) {
        self.cpu.gte.advance_sxy_fifo();
    }

    fn gte_write_irgb(&mut self, value: u32) {
        let value = value.as_u16();
        self.cpu.gte.data.irgb_orgb[0] = u32::from(value);
        let rgb = Rgb5::new_with_raw_value(value.as_u16());
        self.cpu.gte.data.irgb_orgb[1] =
            u32::from_ne_bytes([rgb.r().value(), rgb.g().value(), rgb.b().value(), 0]);
    }

    fn gte_write_lzcs(&mut self, value: i32) {
        let gte = &mut self.cpu.gte;
        let cnt = match value {
            ..0 => value.leading_ones(),
            0.. => value.leading_zeros(),
        };
        gte.data.lzcs_lzcr[0] = value;
        gte.data.lzcs_lzcr[1] = cnt as i32;
    }
}

/// mat is row major, v is a column vector, returns a column vector
fn matmul3x3v<S>(mat: [S; 3], v: S) -> i64x3
where
    S: SimdInt<Cast<i64> = Simd<i64, 3>>,
{
    let mat = mat.map(|row| row.cast::<i64>());
    let v = v.cast::<i64>();
    Simd::from_array([
        (mat[0] * v).reduce_sum(),
        (mat[1] * v).reduce_sum(),
        (mat[2] * v).reduce_sum(),
    ])
}

fn transpose3x3<T: SimdElement>(mat: [Simd<T, 3>; 3]) -> [Simd<T, 3>; 3] {
    [
        Simd::from_array([mat[0][0], mat[1][0], mat[2][0]]),
        Simd::from_array([mat[0][1], mat[1][1], mat[2][1]]),
        Simd::from_array([mat[0][2], mat[1][2], mat[2][2]]),
    ]
}

impl<A: Allocator + Copy> Emu<A> {
    pub(crate) fn gte_function(cmd: GteCommandField) -> Option<u32> {
        let cmd = cmd.cmd().ok()?;
        macro_rules! trace_todo {
            () => {{
                tracing::warn!("not yet implemented: gte2 cmd {cmd:?}");
                None
            }};
        }
        match cmd {
            GteCmd::Rtps => Some(GteBindings::<A>::RTPS),
            GteCmd::Nclip => Some(GteBindings::<A>::NCLIP),
            GteCmd::Op => trace_todo!(),
            GteCmd::Dpcs => trace_todo!(),
            GteCmd::Intpl => trace_todo!(),
            GteCmd::Mvmva => Some(GteBindings::<A>::MVMVA),
            GteCmd::Ncds => Some(GteBindings::<A>::NCDS),
            GteCmd::Cdp => trace_todo!(),
            GteCmd::Ncdt => Some(GteBindings::<A>::NCDT),
            GteCmd::Nccs => Some(GteBindings::<A>::NCCS),
            GteCmd::Cc => trace_todo!(),
            GteCmd::Ncs => Some(GteBindings::<A>::NCS),
            GteCmd::Nct => Some(GteBindings::<A>::NCT),
            GteCmd::Sqr => trace_todo!(),
            GteCmd::Dcpl => trace_todo!(),
            GteCmd::Dpct => trace_todo!(),
            GteCmd::Avsz3 => Some(GteBindings::<A>::AVSZ3),
            GteCmd::Avsz4 => Some(GteBindings::<A>::AVSZ4),
            GteCmd::Rtpt => Some(GteBindings::<A>::RTPT),
            GteCmd::Gpf => trace_todo!(),
            GteCmd::Gpl => trace_todo!(),
            GteCmd::Ncct => Some(GteBindings::<A>::NCCT),
        }
    }
}

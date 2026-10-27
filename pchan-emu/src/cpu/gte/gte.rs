use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use core::alloc::Allocator;
use core::mem::offset_of;
use core::simd::prelude::*;
use core::{cmp, ptr, slice};
use glam::prelude::*;
use pchan_encode::derive::Endecode;
use std::simd::SimdElement;

use crate::Emu;
use crate::bindings::ExternFnTable;
use crate::cpu::Cpu;
use crate::gpu::vram::Rgb5;
use crate::memory::ext;

#[repr(C)]
#[derive(Default, Debug, Endecode, Clone)]
pub struct Gte {
    pub data: GteDataRegs,
    pub ctrl: GteCtrlRegs,
    pub rgb:  u32,
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
    #[bit(31, rw)]
    err:      bool,
    #[bits([23..=30, 13..=18], rw)]
    err_cond: u14,

    #[bits(28..=30, rw)]
    macv_ovf_pos:    u3,
    #[bits(25..=27, rw)]
    macv_ovf_neg:    u3,
    #[bits(22..=24, rw)]
    irv_sat:         u3,
    #[bit(18, rw)]
    sz3_otz_sat:     bool,
    #[bit(17, rw)]
    divide_overflow: bool,
    #[bit(16, rw)]
    mac0_ovf_pos:    bool,
    #[bit(15, rw)]
    mac0_ovf_neg:    bool,
    #[bits(13..=14, rw)]
    sxy2_sat:        u2,
    #[bit(12, rw)]
    ir0_sat:         bool,
}

impl Flag {
    fn update_err(&mut self) {
        self.set_err(self.err_cond().value() != 0);
    }
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

    write_sxyp: fn(&mut Emu<A>, u32),
    write_irgb: fn(&mut Emu<A>, u32),
    write_lzcs: fn(&mut Emu<A>, i32),
    write_flag: fn(&mut Emu<A>, u32),

    read_irgb_orgb: fn(&mut Emu<A>),
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
    pub const WRITE_FLAG: u32 = offset_of!(Self, write_flag) as u32 + Self::OFFSET;

    pub const READ_IRGB_ORGB: u32 = offset_of!(Self, read_irgb_orgb) as u32 + Self::OFFSET;

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
            write_flag: Emu::gte_write_flag,

            read_irgb_orgb: Emu::gte_read_irgb_orgb,
        }
    }
}

impl<A: Allocator + Copy> Default for GteBindings<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl Gte {
    pub const RGB_U32: u32 =
        (offset_of!(Emu, cpu) + offset_of!(Cpu, gte) + offset_of!(Self, rgb)) as u32;

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
        self.check_macv(macv);
        let macv = macv >> (i64::from(sf) * 12);
        macv.cast()
    }

    fn cast_macv(&mut self, macv: i64x3) -> i32x3 {
        self.check_macv(macv).cast()
    }

    fn check_macv(&mut self, macv: i64x3) -> i64x3 {
        let ovf_pos = macv.simd_gt(i64x3::splat((1 << 43) - 1));
        let ovf_neg = macv.simd_lt(i64x3::splat(-1 << 43));
        let old_ovf_pos = self.ctrl.flag.macv_ovf_pos();
        let old_ovf_neg = self.ctrl.flag.macv_ovf_neg();
        self.ctrl
            .flag
            .set_macv_ovf_pos(old_ovf_pos | ovf_pos.to_bitmask().as_::<u3>().reverse_bits());
        self.ctrl
            .flag
            .set_macv_ovf_neg(old_ovf_neg | ovf_neg.to_bitmask().as_::<u3>().reverse_bits());
        (macv << 20) >> 20
    }

    fn cast_mac0(&mut self, mac0: i64) -> i32 {
        self.check_mac0(mac0) as i32
    }

    fn check_mac0(&mut self, mac0: i64) -> i64 {
        let ovf_pos = mac0 > i64::from(i32::MAX);
        let ovf_neg = mac0 < i64::from(i32::MIN);
        let old_ovf_pos = self.ctrl.flag.mac0_ovf_pos();
        let old_ovf_neg = self.ctrl.flag.mac0_ovf_neg();
        self.ctrl.flag.set_mac0_ovf_pos(old_ovf_pos | ovf_pos);
        self.ctrl.flag.set_mac0_ovf_neg(old_ovf_neg | ovf_neg);
        mac0
    }

    fn check_mac0v(&mut self, mac0v: i64x3) -> i64x3 {
        let ovf_pos = mac0v.simd_gt(i64x3::splat(i64::from(i32::MAX))).any();
        let ovf_neg = mac0v.simd_lt(i64x3::splat(i64::from(i32::MIN))).any();
        let old_ovf_pos = self.ctrl.flag.mac0_ovf_pos();
        let old_ovf_neg = self.ctrl.flag.mac0_ovf_neg();
        self.ctrl.flag.set_mac0_ovf_pos(old_ovf_pos | ovf_pos);
        self.ctrl.flag.set_mac0_ovf_neg(old_ovf_neg | ovf_neg);

        mac0v
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
        let (new_irv, check_irv) = match (cmd.cmd(), cmd.lm()) {
            (_, SaturateMode::Signed) => {
                let irv = irv.simd_clamp(Simd::splat(-0x8000), Simd::splat(0x7fff));
                (irv, irv)
            }
            (_, SaturateMode::Unsigned) => {
                let irv = irv.simd_clamp(Simd::splat(-0x0), Simd::splat(0x7fff));
                (irv, irv)
            }
        };
        let sat_flags = check_irv
            .simd_ne(irv)
            .to_bitmask()
            .as_::<u3>()
            .reverse_bits();
        let old_sat_flags = self.ctrl.flag.irv_sat();
        self.ctrl.flag.set_irv_sat(old_sat_flags | sat_flags);
        new_irv.cast()
    }

    /// force sf=1 and lm=0 for ir3 check
    fn sat_irv_rtps<S: SimdInt<Cast<i64> = Simd<i64, 3>>>(
        &mut self,
        macv_unshifted: S,
        cmd: GteCommandField,
    ) -> i16x3 {
        let macv_unshifted = macv_unshifted.cast::<i64>();
        let irv = self.shift_macv(macv_unshifted, cmd.sf());
        let check_irv = {
            let mut irv = irv;
            irv[2] = (macv_unshifted[2] >> 12).truncate();
            irv
        };

        let (min, max) = match cmd.lm() {
            SaturateMode::Signed => (-0x8000, 0x7fff),
            SaturateMode::Unsigned => (0x0, 0x7fff),
        };
        let minv = i32x3::from_array([min, min, -0x8000]);
        let maxv = i32x3::splat(max);

        let clamped_check_irv = check_irv.simd_clamp(minv, maxv);
        let clamped_irv = irv.simd_clamp(i32x3::splat(min), maxv);

        let sat_flags = clamped_check_irv
            .simd_ne(check_irv)
            .to_bitmask()
            .as_::<u3>()
            .reverse_bits();
        let old_sat_flags = self.ctrl.flag.irv_sat();
        self.ctrl.flag.set_irv_sat(old_sat_flags | sat_flags);
        clamped_irv.cast()
    }

    fn sat_sxy(&mut self, sxy: i64x2) -> i16x2 {
        let new_sxy = sxy.simd_clamp(i64x2::splat(-0x400), i64x2::splat(0x3ff));
        let old_sat_flags = self.ctrl.flag.sxy2_sat();
        let new_sat_flags = new_sxy.simd_ne(sxy).to_bitmask().as_::<u2>().reverse_bits();
        self.ctrl.flag.set_sxy2_sat(old_sat_flags | new_sat_flags);
        new_sxy.cast()
    }

    fn push_sxy(&mut self, value: i16x2) {
        self.data.sxy_fifo[3] = value;
        self.advance_sxy_fifo();
    }

    fn sat_sz3_otz(&mut self, x: impl Into<i64>) -> u16 {
        let x = x.into();
        let new_x = x.clamp(0, 0xffff);
        let new_flag = x != new_x;
        let old_flag = self.ctrl.flag.sz3_otz_sat();
        self.ctrl.flag.set_sz3_otz_sat(old_flag | new_flag);
        new_x.cast_unsigned().truncate()
    }

    /// ```
    /// if (H < SZ3*2) then                            ;check if overflow
    ///   z = count_leading_zeroes(SZ3)                ;z=0..0Fh (for 16bit SZ3)
    ///   n = (H SHL z)                                ;n=0..7FFF8000h
    ///   d = (SZ3 SHL z)                              ;d=8000h..FFFFh
    ///   u = unr_table[(d-7FC0h) SHR 7] + 101h        ;u=200h..101h
    ///   d = ((2000080h - (d * u)) SHR 8)             ;d=10000h..0FF01h
    ///   d = ((0000080h + (d * u)) SHR 8)             ;d=20000h..10000h
    ///   n = min(1FFFFh, (((n*d) + 8000h) SHR 16))    ;n=0..1FFFFh
    /// else n = 1FFFFh, FLAG.Bit17=1, FLAG.Bit31=1    ;n=1FFFFh plus overflow flag    
    /// ```
    #[expect(clippy::many_single_char_names)]
    fn unr_div(&mut self, h: u32, sz3: u32) -> u64 {
        if h >= sz3 * 2 {
            self.ctrl.flag.set_divide_overflow(true);
            return 0x1ffff;
        }

        let z = (sz3 as u16).leading_zeros();
        let n = h << z;
        let d = sz3 << z;
        let u: i32 = (u32::from(UNR_TABLE[((d - 0x7fc0) >> 7) as usize]) + 0x101) as i32;
        let d: i32 = (0x2000080 - (d as i32 * u)) >> 8;
        let d: i32 = (0x0000080 + (d * u)) >> 8;

        cmp::min(0x1ffff, (ext::zero(n) * ext::zero(d) + 0x8000) >> 16)
    }

    /// mat is row major, v is a column vector, returns a column vector
    fn macv_matmul3x3v<const FAST: bool>(
        &mut self,
        mat: [impl SimdInt<Cast<i64> = Simd<i64, 3>>; 3],
        v: impl SimdInt<Cast<i64> = Simd<i64, 3>>,
    ) -> i64x3 {
        let mat = mat.map(|row| row.cast::<i64>());
        let v = v.cast::<i64>();
        if FAST {
            Simd::from_array([
                (mat[0] * v).reduce_sum(),
                (mat[1] * v).reduce_sum(),
                (mat[2] * v).reduce_sum(),
            ])
        } else {
            (0..3)
                .map(|c| {
                    (
                        i64x3::from_array([mat[0][c], mat[1][c], mat[2][c]]),
                        i64x3::splat(v[c]),
                    )
                })
                .map(|(m, v)| self.check_macv(m * v))
                .sum()
        }
    }
}

static UNR_TABLE: [u8; 0x100 + 1] = [
    0xFF, 0xFD, 0xFB, 0xF9, 0xF7, 0xF5, 0xF3, 0xF1, 0xEF, 0xEE, 0xEC, 0xEA, 0xE8, 0xE6, 0xE4, 0xE3,
    0xE1, 0xDF, 0xDD, 0xDC, 0xDA, 0xD8, 0xD6, 0xD5, 0xD3, 0xD1, 0xD0, 0xCE, 0xCD, 0xCB, 0xC9, 0xC8,
    0xC6, 0xC5, 0xC3, 0xC1, 0xC0, 0xBE, 0xBD, 0xBB, 0xBA, 0xB8, 0xB7, 0xB5, 0xB4, 0xB2, 0xB1, 0xB0,
    0xAE, 0xAD, 0xAB, 0xAA, 0xA9, 0xA7, 0xA6, 0xA4, 0xA3, 0xA2, 0xA0, 0x9F, 0x9E, 0x9C, 0x9B, 0x9A,
    0x99, 0x97, 0x96, 0x95, 0x94, 0x92, 0x91, 0x90, 0x8F, 0x8D, 0x8C, 0x8B, 0x8A, 0x89, 0x87, 0x86,
    0x85, 0x84, 0x83, 0x82, 0x81, 0x7F, 0x7E, 0x7D, 0x7C, 0x7B, 0x7A, 0x79, 0x78, 0x77, 0x75, 0x74,
    0x73, 0x72, 0x71, 0x70, 0x6F, 0x6E, 0x6D, 0x6C, 0x6B, 0x6A, 0x69, 0x68, 0x67, 0x66, 0x65, 0x64,
    0x63, 0x62, 0x61, 0x60, 0x5F, 0x5E, 0x5D, 0x5D, 0x5C, 0x5B, 0x5A, 0x59, 0x58, 0x57, 0x56, 0x55,
    0x54, 0x53, 0x53, 0x52, 0x51, 0x50, 0x4F, 0x4E, 0x4D, 0x4D, 0x4C, 0x4B, 0x4A, 0x49, 0x48, 0x48,
    0x47, 0x46, 0x45, 0x44, 0x43, 0x43, 0x42, 0x41, 0x40, 0x3F, 0x3F, 0x3E, 0x3D, 0x3C, 0x3C, 0x3B,
    0x3A, 0x39, 0x39, 0x38, 0x37, 0x36, 0x36, 0x35, 0x34, 0x33, 0x33, 0x32, 0x31, 0x31, 0x30, 0x2F,
    0x2E, 0x2E, 0x2D, 0x2C, 0x2C, 0x2B, 0x2A, 0x2A, 0x29, 0x28, 0x28, 0x27, 0x26, 0x26, 0x25, 0x24,
    0x24, 0x23, 0x22, 0x22, 0x21, 0x20, 0x20, 0x1F, 0x1E, 0x1E, 0x1D, 0x1D, 0x1C, 0x1B, 0x1B, 0x1A,
    0x19, 0x19, 0x18, 0x18, 0x17, 0x16, 0x16, 0x15, 0x15, 0x14, 0x14, 0x13, 0x12, 0x12, 0x11, 0x11,
    0x10, 0x0F, 0x0F, 0x0E, 0x0E, 0x0D, 0x0D, 0x0C, 0x0C, 0x0B, 0x0A, 0x0A, 0x09, 0x09, 0x08, 0x08,
    0x07, 0x07, 0x06, 0x06, 0x05, 0x05, 0x04, 0x04, 0x03, 0x03, 0x02, 0x02, 0x01, 0x01, 0x00, 0x00,
    0x00,
];

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

        let (mac, macv_unshifted) = {
            let rt = gte.ctrl.rt_mat;
            let tr = gte.ctrl.tr_vec;
            let mac = gte.check_macv(tr.cast::<i64>() << 12);
            let mac = mac + gte.macv_matmul3x3v::<false>(rt, v);
            let mac = gte.check_macv(mac);
            (gte.shift_macv(mac, cmd.sf()), mac)
        };

        gte.data.macv = mac;
        gte.data.irv = gte.sat_irv_rtps(macv_unshifted, cmd).cast();

        let sz3 = macv_unshifted[2] >> 12;
        let sz3 = gte.sat_sz3_otz(sz3);
        gte.advance_sz_fifo();
        gte.data.sz_fifo[3] = u32::from(sz3);

        let div = gte.unr_div(u32::from(gte.ctrl.h), u32::from(sz3));

        let mac0v = i64x3::splat(div.as_i64());
        let mac0v = gte.check_mac0v(
            mac0v
                * i64x3::from_array([
                    gte.data.irv[0].as_i64(),
                    gte.data.irv[1].as_i64(),
                    gte.ctrl.dqa.as_i64(),
                ])
                + i64x3::from_array([
                    gte.ctrl.of[0].as_i64(),
                    gte.ctrl.of[1].as_i64(),
                    gte.ctrl.dqb.as_i64(),
                ]),
        );

        let sxy2 = mac0v.resize::<2>(0) >> 16;
        let sxy2 = gte.sat_sxy(sxy2);
        gte.push_sxy(sxy2);

        let mac0 = mac0v[2];
        gte.data.mac0 = gte.cast_mac0(mac0);
        gte.data.ir0 = gte.sat_ir0(mac0 >> 12);

        gte.ctrl.flag.update_err();
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
        gte.ctrl.flag.update_err();
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
        let macv = gte.macv_matmul3x3v::<false>(gte.ctrl.ls_mat, v);
        let macv = gte.shift_macv(macv, cmd.sf());
        gte.data.macv = macv;
        gte.data.irv = gte.sat_irv(macv, cmd).cast();

        let lcmxir = gte.macv_matmul3x3v::<false>(gte.ctrl.lc_mat, gte.data.irv.cast::<i16>());
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
        gte.ctrl.flag.update_err();
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
        gte.ctrl.flag.update_err();
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
        gte.ctrl.flag.update_err();
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

        let mx = match cmd.mvmva_mul_mat() {
            MVMVAMultiplyMat::Rotation => gte.ctrl.rt_mat,
            MVMVAMultiplyMat::Light => gte.ctrl.ls_mat,
            MVMVAMultiplyMat::Color => gte.ctrl.lc_mat,
            MVMVAMultiplyMat::Reserved => todo!("mvmva: mx reserved"),
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

        let macv = (tx.cast::<i64>() << 12) + gte.macv_matmul3x3v::<false>(mx, vx);
        let macv = gte.shift_macv(macv, cmd.sf());
        gte.data.macv = macv;
        gte.data.irv = gte.sat_irv(macv, cmd).cast();
        gte.ctrl.flag.update_err();
    }

    fn gte_write_sxyp(&mut self, value: u32) {
        self.cpu.gte.data.sxy_fifo[3] = i16x2::from_ne_bytes(u8x4::from_array(value.to_ne_bytes()));
        self.cpu.gte.advance_sxy_fifo();
    }

    fn gte_write_irgb(&mut self, value: u32) {
        let value = value.as_u16();
        let rgb = Rgb5::new_with_raw_value(value.as_u16());

        self.cpu.gte.data.irgb_orgb[0] = u32::from(rgb.with_mask(false).raw_value());

        let irv = i16x3::from_array([
            i16::from(rgb.r().value()),
            i16::from(rgb.g().value()),
            i16::from(rgb.b().value()),
        ]);
        let irv = irv << 7;
        self.cpu.gte.data.irv = irv.cast();
        self.gte_read_irgb_orgb();
    }

    fn gte_read_irgb_orgb(&mut self) {
        let irv = self.cpu.gte.data.irv.cast::<i16>();
        let rgb = (irv >> 7).simd_clamp(i16x3::splat(0x0), i16x3::splat(0x1f));
        let rgb = rgb.resize::<4>(0).cast::<u8>().cast::<u32>();
        let rgb = rgb[0] | (rgb[1] << 5) | (rgb[2] << 10);
        self.cpu.gte.rgb = rgb;
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

    fn gte_write_flag(&mut self, value: u32) {
        let mut flag = Flag::new_with_raw_value(value);
        flag.update_err();
        self.cpu.gte.ctrl.flag = flag;
    }
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

use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use core::alloc::Allocator;
use core::mem::offset_of;
use core::ops::Not;
use core::simd::prelude::*;
use core::{ptr, slice};
use glam::prelude::*;
use pchan_encode::derive::Endecode;

use crate::Emu;
use crate::bindings::ExternFnTable;

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
    vxyz0:    i16x3,
    _vz0_pad: i16,
    vxyz1:    i16x3,
    _vz1_pad: i16,
    vxyz2:    i16x3,
    _vz2_pad: i16,
    rgbc:     u8x4,
    otz:      u16,
    _otz_pad: u16,
    ir0:      i16,
    _ir0_pad: i16,

    ir123:      i16x3,
    _ir123_pad: i16,

    sxy_fifo:  [i16x2; 4],
    sz_fifo:   [u16x2; 4],
    crgb_fifo: [u8x4; 3],
    res1:      U8Vec4,
    mac0:      i32,
    macv:      i32x3,
    irgb_orgb: [u32; 2],
    lzcs_lzcr: [i32; 2],
}

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
    sof:     i32x2,

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

    flag: u32,
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
struct GteCommandField {
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

const C: GteCommandField = GteCommandField::new_with_raw_value(0x02280030);
const fn test() {
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

#[repr(C)]
#[derive(Debug, Clone)]
pub struct GteBindings<A: Allocator> {
    rtpt: fn(&mut Emu<A>, cmd: GteCommandField),
}

impl<A: Allocator + Copy> GteBindings<A> {
    pub const OFFSET: u32 = ExternFnTable::<A>::OFFSET + offset_of!(ExternFnTable<A>, gte) as u32;
    pub const RTPT: u32 = offset_of!(Self, rtpt) as u32 + Self::OFFSET;

    pub(crate) const fn new() -> Self {
        Self {
            rtpt: Emu::gte_rtpt,
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
}

impl<A: Allocator> Emu<A> {
    fn gte_rtps(&mut self, cmd: GteCommandField) {
        let gte = &mut self.cpu.gte;
        let rt = gte.ctrl.rt_mat;
        let tr = gte.ctrl.tr_vec;
        let v0 = gte.data.vxyz0;
        let mac = tr * i32x3::splat(0x1000)
            + rt[0].cast::<i32>() * i32x3::splat(i32::from(v0[0]))
            + rt[1].cast::<i32>() * i32x3::splat(i32::from(v0[1]))
            + rt[2].cast::<i32>() * i32x3::splat(i32::from(v0[2]));
        let mac = mac >> (i32::from(cmd.sf()) * 12);
        gte.data.ir123 = mac.cast();

        let sz3 = mac[2] >> (u8::from(cmd.sf()).not() * 12);
    }
    fn gte_rtpt(&mut self, cmd: GteCommandField) {}
}

use arbitrary_int::prelude::*;
use bitbybit::bitenum;
use bitbybit::bitfield;
use pchan_utils::hex;
use tracing::instrument;

use crate::Bus;
use crate::Emu;
use crate::io::IOResult;
use crate::io::UnhandledIO;
use crate::io::cast_io;

#[derive(derive_more::Debug, Clone)]
pub struct GpuState {
    gpustat: GpuStatReg,
}

impl GpuState {}

impl Default for GpuState {
    fn default() -> Self {
        let mut gpustat = GpuStatReg::default();
        gpustat.mock_ready();
        Self { gpustat }
    }
}

pub trait Gpu: Bus {
    #[instrument(skip(self), "r")]
    fn read<T: Copy>(&self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        match address {
            0x1f80_1814 => Ok(cast_io(self.gpu().gpustat)),
            _ => Err(UnhandledIO(address)),
        }
    }
    #[instrument(skip(self, value), "w")]
    fn write<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f80_1810 => todo!("gp0 command"),
            0x1f80_1814 => todo!("gp1 command"),
            _ => Err(UnhandledIO(address)),
        }
    }
}

impl Gpu for Emu {}

///
/// # 1F801814h - GPUSTAT - GPU Status Register (R)
///
/// ```plaintext
/// 0-3   Texture page X Base   (N*64)                              ;GP0(E1h).0-3
/// 4     Texture page Y Base   (N*256) (ie. 0 or 256)              ;GP0(E1h).4
/// 5-6   Semi Transparency     (0=B/2+F/2, 1=B+F, 2=B-F, 3=B+F/4)  ;GP0(E1h).5-6
/// 7-8   Texture page colors   (0=4bit, 1=8bit, 2=15bit, 3=Reserved)GP0(E1h).7-8
/// 9     Dither 24bit to 15bit (0=Off/strip LSBs, 1=Dither Enabled);GP0(E1h).9
/// 10    Drawing to display area (0=Prohibited, 1=Allowed)         ;GP0(E1h).10
/// 11    Set Mask-bit when drawing pixels (0=No, 1=Yes/Mask)       ;GP0(E6h).0
/// 12    Draw Pixels           (0=Always, 1=Not to Masked areas)   ;GP0(E6h).1
/// 13    Interlace Field       (or, always 1 when GP1(08h).5=0)
/// 14    "Reverseflag"         (0=Normal, 1=Distorted)             ;GP1(08h).7
/// 15    Texture Disable       (0=Normal, 1=Disable Textures)      ;GP0(E1h).11
/// 16    Horizontal Resolution 2     (0=256/320/512/640, 1=368)    ;GP1(08h).6
/// 17-18 Horizontal Resolution 1     (0=256, 1=320, 2=512, 3=640)  ;GP1(08h).0-1
/// 19    Vertical Resolution         (0=240, 1=480, when Bit22=1)  ;GP1(08h).2
/// 20    Video Mode                  (0=NTSC/60Hz, 1=PAL/50Hz)     ;GP1(08h).3
/// 21    Display Area Color Depth    (0=15bit, 1=24bit)            ;GP1(08h).4
/// 22    Vertical Interlace          (0=Off, 1=On)                 ;GP1(08h).5
/// 23    Display Enable              (0=Enabled, 1=Disabled)       ;GP1(03h).0
/// 24    Interrupt Request (IRQ1)    (0=Off, 1=IRQ)       ;GP0(1Fh)/GP1(02h)
/// 25    DMA / Data Request, meaning depends on GP1(04h) DMA Direction:
///         When GP1(04h)=0 ---> Always zero (0)
///         When GP1(04h)=1 ---> FIFO State  (0=Full, 1=Not Full)
///         When GP1(04h)=2 ---> Same as GPUSTAT.28
///         When GP1(04h)=3 ---> Same as GPUSTAT.27
/// 26    Ready to receive Cmd Word   (0=No, 1=Ready)  ;GP0(...) ;via GP0
/// 27    Ready to send VRAM to CPU   (0=No, 1=Ready)  ;GP0(C0h) ;via GPUREAD
/// 28    Ready to receive DMA Block  (0=No, 1=Ready)  ;GP0(...) ;via GP0
/// 29-30 DMA Direction (0=Off, 1=?, 2=CPUtoGP0, 3=GPUREADtoCPU)    ;GP1(04h).0-1
/// 31    Drawing even/odd lines in interlace mode (0=Even or Vblank, 1=Odd)
/// ```
///
/// Credits to PSX-SPX by Martin Korth [Gpu Status Register](https://problemkaputt.de/psx-spx.htm#gpustatusregister)
///
#[bitfield(u32)]
#[derive(derive_more::Debug, Default, derive_more::Into)]
#[debug("{}", hex(self.raw_value))]
pub struct GpuStatReg {
    #[bits(0..=3, rw)]
    texpage_x_base:       u4,
    #[bit(4, rw)]
    texpage_y_base:       bool,
    #[bits(5..=6, rw)]
    semi_transparency:    u2,
    #[bits(7..=8, rw)]
    texpage_colors:       u2,
    #[bit(9, rw)]
    dither:               bool,
    #[bit(10, rw)]
    draw_to_display:      bool,
    #[bit(11, rw)]
    draw_mask:            bool,
    #[bit(12, rw)]
    draw_pixels:          DrawPixels,
    #[bit(13, rw)]
    interlace_field:      bool,
    #[bit(14, rw)]
    reverse_flag:         ReverseFlag,
    #[bit(15, rw)]
    texture_disable:      bool,
    #[bit(16, rw)]
    h_resolution_2:       bool,
    #[bits(17..=18, rw)]
    h_resolution_1:       u2,
    #[bit(19, rw)]
    v_resolution:         bool,
    #[bit(20, rw)]
    video_mode:           VideoMode,
    #[bit(21, rw)]
    display_color_depth:  bool,
    #[bit(22, rw)]
    v_interlace:          bool,
    #[bit(23, rw)]
    display_enable:       bool,
    #[bit(24, rw)]
    irq:                  bool,
    #[bit(25, rw)]
    dma_request:          bool,
    #[bit(26, rw)]
    ready_recv_cmd:       bool,
    #[bit(27, rw)]
    ready_send_vram:      bool,
    #[bit(28, rw)]
    ready_recv_dma_block: bool,
    #[bits(29..=30, rw)]
    dma_direction:        DmaDirection,
    #[bit(31, rw)]
    even_odd_in_vblank:   DrawEvenOdd,
}

impl GpuStatReg {
    pub fn mock_ready(&mut self) {
        self.set_ready_recv_cmd(true);
        self.set_ready_recv_dma_block(true);
    }
}

#[derive(Debug)]
#[bitenum(u1, exhaustive = true)]
enum DrawPixels {
    Always           = 0x0,
    NotToMaskedAreas = 0x1,
}

#[derive(Debug)]
#[bitenum(u1, exhaustive = true)]
enum ReverseFlag {
    Normal    = 0x0,
    Distorted = 0x1,
}

#[derive(Debug)]
#[bitenum(u1, exhaustive = true)]
enum VideoMode {
    Ntsc = 0x0,
    Pal  = 0x1,
}

#[derive(Debug)]
#[bitenum(u2, exhaustive = true)]
enum DmaDirection {
    Off          = 0x0,
    Unknown      = 0x1,
    CpuToGp0     = 0x2,
    CpuReadToCpu = 0x3,
}

#[derive(Debug)]
#[bitenum(u1, exhaustive = true)]
enum DrawEvenOdd {
    EvenOrVBlank = 0x0,
    Odd          = 0x1,
}

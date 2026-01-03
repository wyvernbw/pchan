use std::mem::transmute;

use arbitrary_int::prelude::*;
use bitbybit::bitenum;
use bitbybit::bitfield;
use derive_more as d;
use heapless::Deque;
use pchan_utils::hex;
use tracing::instrument;

use crate::Bus;
use crate::Emu;
use crate::io::CastIOFrom;
use crate::io::CastIOInto;
use crate::io::IOResult;
use crate::io::UnhandledIO;
use crate::memory::kb;
use crate::memory::mb;

#[derive(derive_more::Debug, Clone)]
pub struct GpuState {
    #[debug(skip)]
    vram:          Box<[u16]>,
    #[debug(skip)]
    cmd_queue:     Deque<u32, 16>,
    gpustat:       GpuStatReg,
    gp0:           Gp0,
    gp0read:       [u16; 2],
    gp0read_queue: Deque<u32, 32>,
    /// GP0(0xe1) - Draw Mode setting (aka "Texpage")
    tex_attr:      TexpageCmd,
    /// GP0(0xe2) - Texture Window setting
    tex_window:    TexWindowCmd,
    model:         GpuModel,
}

#[derive(derive_more::Debug, Clone, Default)]
pub enum GpuModel {
    #[default]
    Gpu160Pin,
    Gpu180Pin,
    Gpu208Pin,
}

impl GpuState {}

impl Default for GpuState {
    fn default() -> Self {
        let mut gpustat = GpuStatReg::default();
        gpustat.mock_ready();
        Self {
            gpustat,
            vram: vec![0; mb(1)].into_boxed_slice(),
            gp0: Gp0::WaitingForCmd,
            gp0read: Default::default(),
            gp0read_queue: Deque::new(),
            cmd_queue: Deque::new(),
            model: GpuModel::default(),
            tex_attr: Default::default(),
            tex_window: Default::default(),
        }
    }
}

pub trait Gpu: Bus {
    #[instrument(skip(self), "gpu:r")]
    fn read<T: Copy>(&mut self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801810 => {
                #[allow(clippy::single_match)]
                match &self.gpu().gp0 {
                    Gp0::CpRectVramToCpu(Gp0CpRect::RecvData(cursor)) => {
                        let mut cursor = *cursor;
                        for (idx, at) in cursor.iter().take(2).enumerate() {
                            self.gpu_mut().vram_read(at, idx);
                        }
                        let gp0 = match cursor.done() {
                            true => {
                                self.gpu_mut().gpustat.set_ready_send_vram(false);
                                Gp0::WaitingForCmd
                            }
                            false => Gp0::CpRectVramToCpu(Gp0CpRect::RecvData(cursor)),
                        };
                        self.gpu_mut().gp0 = gp0;
                    }
                    _ => {}
                }
                tracing::info!(gp0 = ?self.gpu().gp0);
                Ok(self.gpu().gp0read.io_from_u32())
            }
            0x1f80_1814 => Ok(self.gpu().gpustat.io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
    }
    #[instrument(skip(self, value), "gpu:w")]
    fn write<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f80_1810 => {
                self.gp0_command(value.io_into_u32());
                Ok(())
            }
            0x1f80_1814 => {
                self.gp1_command(value.io_into_u32());
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }

    fn reduce(&mut self, cmd: GpuCmd) -> Gp0 {
        match cmd.cmd() {
            // nop
            0xc0..=0xdf => Gp0::CpRectVramToCpu(Gp0CpRect::RecvDest),
            0x00 | 0x04..0x1e | 0xe0 | 0xe7..0xef => Gp0::WaitingForCmd,
            0xa0 => Gp0::CpRectCpuToVram(Gp0CpRect::RecvDest),
            0xe1 => {
                let texpage = TexpageCmd::new_with_raw_value(cmd.raw_value());

                self.gpu_mut().tex_attr = texpage;
                let gpustat = &mut self.gpu_mut().gpustat;

                gpustat.set_texpage_x_base(texpage.texpage_x_base());
                gpustat.set_texpage_y_base(texpage.texpage_y_base());
                gpustat.set_semi_transparency(texpage.semi_transparency());
                gpustat.set_texpage_colors(texpage.texpage_colors());
                gpustat.set_dither(texpage.dither());
                gpustat.set_draw_to_display(texpage.draw_to_display());
                gpustat.set_texture_disable(texpage.texture_disable());

                Gp0::WaitingForCmd
            }
            value => todo!("gp0 command: {}", hex(value)),
        }
    }

    #[instrument(skip_all)]
    fn gp0_command<T: Copy>(&mut self, value: T) {
        let value = value.io_into_u32();
        let cmd = GpuCmd::new_with_raw_value(value);
        let gp0 = match &self.gpu().gp0 {
            Gp0::WaitingForCmd => self.reduce(cmd),
            Gp0::CpRectCpuToVram(Gp0CpRect::RecvDest) => {
                let dest: VramCoord = unsafe { transmute(value) };
                Gp0::CpRectCpuToVram(Gp0CpRect::RecvSize { dest })
            }
            Gp0::CpRectCpuToVram(Gp0CpRect::RecvSize { dest }) => {
                let size: VramCoord = unsafe { transmute(value) };
                Gp0::CpRectCpuToVram(Gp0CpRect::RecvData(VramCursor::new(*dest, *dest + size)))
            }
            Gp0::CpRectCpuToVram(Gp0CpRect::RecvData(cursor)) => {
                let mut cursor = *cursor;
                for (at, halfword) in cursor.iter().take(2).zip(halfwords(value)) {
                    self.gpu_mut().vram_write(at, halfword);
                }
                match cursor.done() {
                    true => Gp0::WaitingForCmd,
                    false => Gp0::CpRectCpuToVram(Gp0CpRect::RecvData(cursor)),
                }
            }

            Gp0::CpRectVramToCpu(Gp0CpRect::RecvDest) => {
                let dest: VramCoord = unsafe { transmute(value) };
                Gp0::CpRectVramToCpu(Gp0CpRect::RecvSize { dest })
            }

            Gp0::CpRectVramToCpu(Gp0CpRect::RecvSize { dest }) => {
                let dest = *dest;
                let size: VramCoord = unsafe { transmute(value) };

                self.gpu_mut().gpustat.set_ready_send_vram(true);
                Gp0::CpRectVramToCpu(Gp0CpRect::RecvData(VramCursor::new(dest, dest + size)))
            }
            Gp0::CpRectVramToCpu(Gp0CpRect::RecvData(_)) => {
                self.gpu_mut().gpustat.set_ready_send_vram(false);

                self.reduce(cmd)
            }
        };

        tracing::info!(?gp0);

        self.gpu_mut().gp0 = gp0;
    }

    fn gp1_command<T: Copy>(&mut self, value: T) {
        let value = GpuCmd::new_with_raw_value(value.io_into_u32());
        match value.cmd() {
            0x00 => {
                self.gpu_mut().cmd_queue.clear();
                self.gpu_mut().gpustat = GpuStatReg::new_with_raw_value(0x14802000);
                self.gpu_mut().gpustat.mock_ready();
            }
            0x01 => {
                self.gpu_mut().cmd_queue.clear();
                self.gpu_mut().gpustat.mock_ready();
            }
            0x08 => {
                let cmd = DisplayModeCmd::new_with_raw_value(value.raw_value);
                let gpustat = &mut self.gpu_mut().gpustat;
                gpustat.set_h_resolution_1(cmd.hres_1());
                gpustat.set_v_resolution(cmd.vres());
                gpustat.set_video_mode(cmd.video_mode());
                gpustat.set_display_color_depth(cmd.display_color_depth());
                gpustat.set_v_interlace(cmd.v_interlace());
                gpustat.set_h_resolution_2(cmd.hres_2());
            }
            // get gpu info
            0x10 => {
                let Some(cmd) = self.gpu().get_gpu_info_cmd(value) else {
                    return;
                };
                match cmd {
                    GpuInfoCmd::Unused00 | GpuInfoCmd::Unused01 => {}
                    GpuInfoCmd::TexWindow => {
                        self.gpu_mut().gp0read = unsafe {
                            transmute::<u32, [u16; 2]>(self.gpu().tex_window.raw_value())
                        };
                    }
                }
            }
            value => todo!("gp1 command: {}", hex(value)),
        }
    }
}

impl GpuState {
    fn get_gpu_info_cmd(&self, cmd: GpuCmd) -> Option<GpuInfoCmd> {
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
    fn vram_write(&mut self, coord: VramCoord, value: u16) {
        let addr = coord.x as usize + coord.y as usize * kb(1);
        self.vram[addr] = value;
    }

    /// returns value through `self.gp0read`
    fn vram_read(&mut self, coord: VramCoord, idx: usize) {
        let addr = coord.x as usize + coord.y as usize * kb(1);
        self.gp0read[idx] = self.vram[addr];
    }
}

pub fn halfwords(word: u32) -> [u16; 2] {
    [word as u16, (word >> 16) as u16]
}

#[derive(Debug, Clone)]
pub enum Gp0 {
    WaitingForCmd,
    CpRectCpuToVram(Gp0CpRect),
    CpRectVramToCpu(Gp0CpRect),
}

#[derive(Debug, Clone, Copy, d::Add, d::AddAssign, PartialEq, PartialOrd, Ord, Eq)]
#[repr(C)]
pub struct VramCoord {
    x: u16,
    y: u16,
}

impl VramCoord {
    pub fn new(xpos: u16, ypos: u16) -> Self {
        Self { x: xpos, y: ypos }
    }
}

#[derive(Debug, Clone, Copy, d::Add, d::AddAssign, PartialEq, PartialOrd, Ord, Eq)]
pub struct VramCursor {
    start:  VramCoord,
    curr:   VramCoord,
    border: VramCoord,
}

impl VramCursor {
    fn new(start: VramCoord, border: VramCoord) -> Self {
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
        if self.curr.x == self.border.x {
            self.curr.x = self.start.x;
            self.curr.y += 1;
        }

        Some(curr)
    }

    fn done(&self) -> bool {
        self.curr.y == self.border.y
    }

    fn iter(&mut self) -> impl Iterator<Item = VramCoord> {
        std::iter::from_fn(|| self.next())
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

#[derive(Debug, Clone)]
pub enum Gp0CpRect {
    RecvDest,
    RecvSize { dest: VramCoord },
    RecvData(VramCursor),
}

#[bitfield(u32)]
pub struct GpuCmd {
    #[bits(0..=23, r)]
    fields: u24,
    #[bits(24..=31, r)]
    cmd:    u8,
}

impl Gpu for Emu {}

///
/// # 1F801814h - GPUSTAT - GPU Status Register (R)
///
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
    h_resolution_2:       HRes2,
    #[bits(17..=18, rw)]
    h_resolution_1:       HRes1,
    #[bit(19, rw)]
    v_resolution:         VRes,
    #[bit(20, rw)]
    video_mode:           VideoMode,
    #[bit(21, rw)]
    display_color_depth:  DisplayColorDepth,
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
        // self.set_ready_send_vram(true);
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

/// #  GP0(E1h) - Draw Mode setting (aka "Texpage")
///
/// Likely sets the relevant bits in the GpuStat register.
///
/// PSX-SPX summary:
///
/// 0-3   Texture page X Base   (N*64) (ie. in 64-halfword steps)    ;GPUSTAT.0-3
/// 4     Texture page Y Base   (N*256) (ie. 0 or 256)               ;GPUSTAT.4
/// 5-6   Semi Transparency     (0=B/2+F/2, 1=B+F, 2=B-F, 3=B+F/4)   ;GPUSTAT.5-6
/// 7-8   Texture page colors   (0=4bit, 1=8bit, 2=15bit, 3=Reserved);GPUSTAT.7-8
/// 9     Dither 24bit to 15bit (0=Off/strip LSBs, 1=Dither Enabled) ;GPUSTAT.9
/// 10    Drawing to display area (0=Prohibited, 1=Allowed)          ;GPUSTAT.10
/// 11    Texture Disable (0=Normal, 1=Disable if GP1(09h).Bit0=1)   ;GPUSTAT.15
///         (Above might be chipselect for (absent) second VRAM chip?)
/// 12    Textured Rectangle X-Flip   (BIOS does set this bit on power-up...?)
/// 13    Textured Rectangle Y-Flip   (BIOS does set it equal to GPUSTAT.13...?)
/// 14-23 Not used (should be 0)
/// 24-31 Command  (E1h)
///
/// [Link to PSX-SPX](https://problemkaputt.de/psx-spx.htm#gpuioportsdmachannelscommandsvram:~:text=GP0%28E1h%29%20%2D%20Draw%20Mode%20setting%20%28aka%20%22Texpage%22%29)
///
#[bitfield(u32)]
#[derive(derive_more::Debug, Default, derive_more::Into)]
#[debug("{}", hex(self.raw_value))]
pub struct TexpageCmd {
    #[bits(0..=3, rw)]
    texpage_x_base:    u4,
    #[bit(4, rw)]
    texpage_y_base:    bool,
    #[bits(5..=6, rw)]
    semi_transparency: u2,
    #[bits(7..=8, rw)]
    texpage_colors:    u2,
    #[bit(9, rw)]
    dither:            bool,
    #[bit(10, rw)]
    draw_to_display:   bool,
    #[bit(11, rw)]
    texture_disable:   bool,
    #[bit(12, rw)]
    tex_rect_x_flip:   bool,
    #[bit(13, rw)]
    tex_rect_y_flip:   bool,
}

/// # GP1(10h) - Get GPU Info
/// GP1(11h..1Fh) - Mirrors of GP1(10h), Get GPU Info
///
/// After sending the command, the result can be immediately read from GPUREAD register (there's no NOP or other delay required) (namely GPUSTAT.Bit27 is used only for VRAM-Reads, but NOT for GPU-Info-Reads, so do not try to wait for that flag).
///
///   0-23  Select Information which is to be retrieved (via following GPUREAD)
///
/// On Old 180pin GPUs, following values can be selected:
///
///   00h-01h = Returns Nothing (old value in GPUREAD remains unchanged)
///   02h     = Read Texture Window setting  ;GP0(E2h) ;20bit/MSBs=Nothing
///   03h     = Read Draw area top left      ;GP0(E3h) ;19bit/MSBs=Nothing
///   04h     = Read Draw area bottom right  ;GP0(E4h) ;19bit/MSBs=Nothing
///   05h     = Read Draw offset             ;GP0(E5h) ;22bit
///   06h-07h = Returns Nothing (old value in GPUREAD remains unchanged)
///   08h-FFFFFFh = Mirrors of 00h..07h
///
/// On New 208pin GPUs, following values can be selected:
///
///   00h-01h = Returns Nothing (old value in GPUREAD remains unchanged)
///   02h     = Read Texture Window setting  ;GP0(E2h) ;20bit/MSBs=Nothing
///   03h     = Read Draw area top left      ;GP0(E3h) ;20bit/MSBs=Nothing
///   04h     = Read Draw area bottom right  ;GP0(E4h) ;20bit/MSBs=Nothing
///   05h     = Read Draw offset             ;GP0(E5h) ;22bit
///   06h     = Returns Nothing (old value in GPUREAD remains unchanged)
///   07h     = Read GPU Type (usually 2)    ;see "GPU Versions" chapter
///   08h     = Unknown (Returns 00000000h) (lightgun on some GPUs?)
///   09h-0Fh = Returns Nothing (old value in GPUREAD remains unchanged)
///   10h-FFFFFFh = Mirrors of 00h..0Fh
#[derive(Debug, strum::FromRepr)]
#[repr(u8)]
pub enum GpuInfoCmd {
    Unused00  = 0x00,
    Unused01  = 0x01,
    TexWindow = 0x02,
    // TODO
}

/// # GP0(E2h) - Texture Window setting
///
///   0-4    Texture window Mask X   (in 8 pixel steps)
///   5-9    Texture window Mask Y   (in 8 pixel steps)
///   10-14  Texture window Offset X (in 8 pixel steps)
///   15-19  Texture window Offset Y (in 8 pixel steps)
///   20-23  Not used (zero)
///   24-31  Command  (E2h)
#[bitfield(u32)]
#[derive(Debug, Default)]
pub struct TexWindowCmd {
    #[bits(0..=4)]
    mask_x:   u5,
    #[bits(5..=9)]
    mask_y:   u5,
    #[bits(10..=14)]
    offset_x: u5,
    #[bits(15..=19)]
    offset_y: u5,
    #[bits(20..=23)]
    _pad:     u4,
    #[bits(24..=31)]
    _cmd:     u8,
}

/// # GP1(08h) - Display mode
/// 0-1   Horizontal Resolution 1     (0=256, 1=320, 2=512, 3=640) ;GPUSTAT.17-18
/// 2     Vertical Resolution         (0=240, 1=480, when Bit5=1)  ;GPUSTAT.19
/// 3     Video Mode                  (0=NTSC/60Hz, 1=PAL/50Hz)    ;GPUSTAT.20
/// 4     Display Area Color Depth    (0=15bit, 1=24bit)           ;GPUSTAT.21
/// 5     Vertical Interlace          (0=Off, 1=On)                ;GPUSTAT.22
/// 6     Horizontal Resolution 2     (0=256/320/512/640, 1=368)   ;GPUSTAT.16
/// 7     Flip screen horizontally    (0=Off, 1=On, v1 only)       ;GPUSTAT.14
/// 8-23  Not used (zero)
///
#[bitfield(u32)]
pub struct DisplayModeCmd {
    #[bits(0..=1, rw)]
    hres_1:              HRes1,
    #[bit(2, rw)]
    vres:                VRes,
    #[bit(3, rw)]
    video_mode:          VideoMode,
    #[bit(4, rw)]
    display_color_depth: DisplayColorDepth,
    #[bit(5, rw)]
    v_interlace:         bool,
    #[bit(6, rw)]
    hres_2:              HRes2,
    #[bit(7, rw)]
    screen_hflip:        bool,
}

#[bitenum(u2, exhaustive = true)]
pub enum HRes1 {
    Res256,
    Res320,
    Res512,
    Res640,
}

#[bitenum(u1, exhaustive = true)]
pub enum HRes2 {
    Standard,
    Res368,
}

#[bitenum(u1, exhaustive = true)]
pub enum VRes {
    Res240,
    Res480,
}

#[bitenum(u1, exhaustive = true)]
pub enum DisplayColorDepth {
    Depth15Bit,
    Depth24Bit,
}

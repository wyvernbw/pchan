pub mod draw_call;
pub mod vram;

use core::alloc::Allocator;
use core::mem::{self, transmute};
use core::ops;
use core::sync::atomic::{AtomicBool, AtomicU64};
use core::time::Duration;
use std::time::Instant;

use alloc::alloc::Global;
use alloc::collections::VecDeque;
use arbitrary_int::prelude::*;
use bitbybit::{bitenum, bitfield};
use glam::{U8Vec2, U8Vec3, U16Vec2, U64Vec2, u64vec2};
use pchan_encode::derive::Endecode;
use pchan_utils::{AsyncChan, hex};
use smol::lock::Mutex;
use tracing::Level;

use crate::Emu;
use crate::gpu::draw_call::{
    DrawCall, DrawCallDecoder, DrawCallKind, DrawLineDecoder, DrawPolygonDecoder, DrawRectDecoder,
    Gp0HostBlit, Gp0HostBlitDecoder, Gp0SetDrawAreaCmd, Gp0SetDrawOffsetCmd, Gp0SetMaskBitCmd,
    Gp0VramBlitDecoder, GpuInternalDrawReg, QuickFillDecoder,
};
use crate::gpu::vram::{VramCoord, VramCursor};
use crate::io::evque::Task;
use crate::io::irq::Irq;
use crate::io::{CastIOFrom, CastIOInto, IOResult, UnhandledIO};
use crate::memory::{kb, mb};

pub static VBLANK_COUNT: AtomicU64 = AtomicU64::new(0);

extern crate alloc;

#[derive(Debug, Clone)]
pub struct Conn {
    pub draw_call_chan: AsyncChan<DrawCallMsg>,
    pub vram_out_chan:  AsyncChan<VramRes>,
}

#[derive(Debug, Clone)]
pub enum DrawCallMsg {
    DrawCall(DrawCall),
    Flush(FlushMsg),
    UpdateDisplay(crate::gpu::Display, GpuStatReg),
    Reset(kanal::AsyncSender<()>),
    RequestVram(&'static VramSink, kanal::AsyncSender<()>),
}

#[derive(Debug, Clone)]
pub struct FlushMsg {
    pub display:  crate::gpu::Display,
    /// gpustat at vblank
    pub gpustat:  GpuStatReg,
    pub draw_idx: usize,
}

#[derive(derive_more::Debug, Clone)]
pub struct VramMsg<A: Allocator = Global> {
    #[debug(skip)]
    pub vram:  Box<[u16], A>,
    pub dirty: bool,
}

pub enum VramRes<A: Allocator = Global> {
    Ok(VramPayload<A>),
    Error,
}

pub struct VramPayload<A: Allocator = Global> {
    pub vram:     Box<[u16], A>,
    pub draw_idx: usize,
}

#[derive(derive_more::Debug, Clone, Endecode)]
#[endecode(alloc(with = create_vram))]
pub enum Vram<A: Allocator> {
    Mapped(#[endecode(alloc(with = create_vram))] Box<[u16], A>),
    Lent,
}

const fn vram_size<T: Sized>() -> usize {
    mb(2) / size_of::<T>()
}
pub type VramSink = Mutex<[u16; vram_size::<u16>()]>;
pub static VRAM: VramSink = Mutex::new([0u16; vram_size::<u16>()]);
pub static VRAM_REF: &VramSink = &VRAM;

#[derive(derive_more::Debug, Clone, Endecode)]
pub struct GpuState<A: Allocator> {
    #[debug(skip)]
    pub gpustat:      GpuStatReg,
    pub gp0:          Gp0,
    pub gp0read:      [u16; 2],
    pub gp0read_fifo: VecDeque<u16, A>,
    pub gp0write:     VecDeque<u32, A>,
    pub dp:           Display,
    /// GP0(0xe2) - Texture Window setting
    pub tex_window:   Gp0TexWindowCmd,
    pub draw_reg:     GpuInternalDrawReg,
    pub model:        GpuModel,

    pending_write:        Option<Gp0HostBlit>,
    pending_read:         Option<Gp0HostBlit>,
    pending_draw_calls:   VecDeque<DrawCall, A>,
    ongoing_read:         Option<Gp0HostBlit>,
    draw_calls_in_flight: bool,
    #[endecode(skip)]
    pub(crate) draw_idx:  usize,

    #[debug(skip)]
    #[endecode(skip)]
    pub conn:          Conn,
    #[endecode(skip)]
    pub last_vblank:   Instant,
    pub vblank_signal: bool,

    #[endecode(skip)]
    pub vram_mutation_signal: bool,
    #[debug(skip)]
    #[endecode(skip)]
    alloc:                    A,
}

#[derive(derive_more::Debug, Clone, Default, Endecode)]
pub enum GpuModel {
    #[default]
    Gpu160Pin,
    Gpu180Pin,
    Gpu208Pin,
}

impl<A: Allocator + Copy> GpuState<A> {
    pub fn new(alloc: A) -> Self {
        let mut gpustat = GpuStatReg::default();
        gpustat.mock_ready();
        Self {
            gpustat,
            gp0: Gp0::WaitingForCmd,
            gp0read: Default::default(),
            model: GpuModel::default(),
            tex_window: Gp0TexWindowCmd::default(),
            draw_reg: GpuInternalDrawReg::default(),
            conn: Conn {
                draw_call_chan: kanal::bounded_async(128),
                vram_out_chan:  kanal::bounded_async(128),
            },
            draw_calls_in_flight: false,
            dp: Display::default(),
            last_vblank: Instant::now(),
            vblank_signal: false,
            vram_mutation_signal: false,
            alloc,
            gp0write: VecDeque::with_capacity_in(256 * 256 * 2, alloc),
            pending_read: None,
            ongoing_read: None,
            pending_write: None,
            gp0read_fifo: VecDeque::new_in(alloc),
            draw_idx: 0,
            pending_draw_calls: VecDeque::new_in(alloc),
        }
    }
}

pub static RESET_FLAG: AtomicBool = AtomicBool::new(false);

#[must_use]
pub fn create_vram<A: Allocator>(alloc: A) -> Box<[u16], A> {
    unsafe { Box::new_zeroed_slice_in(mb(1) / 2, alloc).assume_init() }
}

fn mask_bit(value: u16) -> bool {
    value & (1 << 15) != 0
}
fn set_mask_bit(value: u16, set: bool) -> u16 {
    match set {
        true => value | (1 << 15),
        false => value & !(1 << 15),
    }
}

impl<A: Allocator + Copy> Emu<A> {
    #[pchan_macros::instrument(level = "trace", skip(self), "gpu:r")]
    pub fn gpu_read<T: Copy>(&mut self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801810 => {
                #[allow(clippy::single_match)]
                match &self.gpu.gp0 {
                    Gp0::CpuReadingVram => {
                        assert!(self.gpu.gpustat.gp0_rffl());
                        for nibble in 0..2 {
                            match self.gpu.gp0read_fifo.pop_front() {
                                None => {
                                    self.gpu.gpustat.set_gp0_rffl(false);
                                    self.gpu.gpustat.set_ready_recv_cmd(true);
                                    self.gpu.gp0 = Gp0::WaitingForCmd;
                                    break;
                                }
                                Some(value) => {
                                    self.gpu.gp0read[nibble] = value;
                                }
                            };
                        }
                    }
                    _ => {}
                }
                Ok(self.gpu.gp0read.io_from_u32())
            }
            0x1f80_1814 => Ok(self.gpu.gpustat.io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
    }
    pub fn gpu_read_pure<T: Copy>(&self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        match address {
            0x1f80_1814 => Ok(self.gpu.gpustat.io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
    }
    #[pchan_macros::instrument(level = "trace", skip(self, value), "gpu:w")]
    pub fn gpu_write<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        // if size_of::<T>() != 4 {
        //     panic!(
        //         "write to io gpu register is not word sized. {} has size of {}",
        //         std::any::type_name::<T>(),
        //         size_of::<T>()
        //     )
        // }

        match address {
            0x1f80_1810 => {
                self.gpu_gp0_cmd(GpuCmd::new_with_raw_value(value.io_into_u32_overwrite(0x0)));
                Ok(())
            }
            0x1f80_1814 => {
                self.gpu_gp1_cmd(GpuCmd::new_with_raw_value(value.io_into_u32_overwrite(0x0)));
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }

    pub fn gpu_gp0reduce(&mut self, cmd: GpuCmd) -> Gp0 {
        tracing::debug!(cmd = %hex(cmd.cmd()));
        match cmd.cmd() {
            // nop
            0x0 | 0x4..=0x1e | 0xe0 | 0xe7..=0xef => Gp0::WaitingForCmd,

            0x01 => {
                // clear cache?
                Gp0::WaitingForCmd
            }

            0x02 => Gp0::QuickFillDecode(QuickFillDecoder::Color {
                color: U8Vec3::from_array(cmd.fields().to_le_bytes()),
            }),

            0xc0..=0xdf => Gp0::BlitVramToCpuDecoding(Gp0HostBlitDecoder::RecvDest),
            0x1f => {
                self.gpu.gpustat.set_irq(true);
                self.irq_trigger(Irq::Irq1Gpu);
                Gp0::WaitingForCmd
            }
            0xa0..=0xbf => {
                tracing::debug!("start cpu to vram copy");
                Gp0::BlitCpuToVramDecoding(Gp0HostBlitDecoder::RecvDest)
            }
            0xe1 => {
                let texpage = TexpageCmd::new_with_raw_value(cmd.raw_value());

                let gpustat = &mut self.gpu.gpustat;

                gpustat.set_texpage_x_base(texpage.texpage_x_base());
                gpustat.set_texpage_y_base(texpage.texpage_y_base());
                gpustat.set_semi_transparency(texpage.semi_transparency());
                gpustat.set_texpage_colors(texpage.texpage_colors());
                gpustat.set_dither(texpage.dither());
                gpustat.set_draw_to_display(texpage.draw_to_display());
                gpustat.set_texture_disable(texpage.texture_disable());

                Gp0::WaitingForCmd
            }
            // GP0(E2h) - Texture Window setting
            0xe2 => {
                let cmd = Gp0TexWindowCmd::new_with_raw_value(cmd.raw_value());
                self.gpu.tex_window = cmd;

                Gp0::WaitingForCmd
            }
            // GP0(E3h) - Set Drawing Area top left (X1,Y1)
            0xe3 => {
                let opts = &mut self.gpu.draw_reg;
                let cmd = Gp0SetDrawAreaCmd::new_with_raw_value(cmd.raw_value());
                opts.draw_area_top_left.x = cmd.x_coord().as_();
                opts.draw_area_top_left.y = cmd.y_coord_v2().as_();

                Gp0::WaitingForCmd
            }
            // GP0(E4h) - Set Drawing Area bottom right (X2,Y2)
            0xe4 => {
                let opts = &mut self.gpu.draw_reg;
                let cmd = Gp0SetDrawAreaCmd::new_with_raw_value(cmd.raw_value());
                opts.draw_area_bottom_right.x = cmd.x_coord().as_();
                opts.draw_area_bottom_right.y = cmd.y_coord_v2().as_();

                Gp0::WaitingForCmd
            }
            // GP0(E5h) - Set Drawing Offset (X,Y)
            0xe5 => {
                let opts = &mut self.gpu.draw_reg;
                let cmd = Gp0SetDrawOffsetCmd::new_with_raw_value(cmd.raw_value());
                opts.draw_offset.x = cmd.x_offset().as_();
                opts.draw_offset.y = cmd.y_offset().as_();

                Gp0::WaitingForCmd
            }
            // GP0(E6h) - Mask Bit Setting
            0xe6 => {
                let cmd = Gp0SetMaskBitCmd::new_with_raw_value(cmd.raw_value());
                let gpustat = &mut self.gpu.gpustat;
                gpustat.set_set_mask(cmd.draw_mask());
                gpustat.set_draw_pixels(cmd.draw_pixels());

                Gp0::WaitingForCmd
            }
            // Draw polygon
            0x20..=0x3f => Gp0::DrawPolygonDecode(DrawPolygonDecoder::new(cmd.raw_value())),
            // Draw line
            0x40..=0x5f => Gp0::DrawLineDecode(DrawLineDecoder::new(cmd.raw_value())),
            // Draw Rect
            0x60..=0x7f => Gp0::DrawRectDecode(DrawRectDecoder::new(cmd.raw_value())),
            0x80..=0x9f => {
                self.gpu.gpustat.set_ready_recv_cmd(false);
                Gp0::BlitVramToVramDecoding(Gp0VramBlitDecoder::RecvSrc)
            }
            value => todo!("gp0 command: {}", hex(value)),
        }
    }

    pub(crate) fn gpu_gp0_drain_write_queue(&mut self) {
        let mut data = mem::replace(&mut self.gpu.gp0write, VecDeque::new_in(self.alloc));
        while let Some(word) = data.pop_front() {
            self.gpu_gp0_cmd(GpuCmd::new_with_raw_value(word));
        }
        self.gpu.gp0write = data;
    }

    pub(crate) async fn task_gpu_start_blit_vram_to_cpu(&mut self) {
        if let Some(blit) = self.gpu.pending_read.take() {
            self.gpu.task_gpu_flush_draw_calls().await;
            self.gpu.request_vram().await;
            self.gpu.gp0read_fifo.clear();
            self.gpu.ongoing_read = Some(blit);
            self.gpu.gp0 = Gp0::CpuReadingVram;
            self.gpu.gpustat.set_ready_recv_cmd(false);
        }
    }

    #[pchan_macros::instrument(level = "trace", skip_all)]
    pub fn gpu_gp0_cmd(&mut self, cmd: GpuCmd) {
        let value = cmd.raw_value();
        tracing::trace!(gp0=?self.gpu.gp0, gp0cmd = %hex(cmd));
        match mem::take(&mut self.gpu.gp0) {
            Gp0::Busy => {
                self.gpu.gp0write.push_back(value);
            }
            Gp0::WaitingForCmd => {
                self.gpu.gp0 = self.gpu_gp0reduce(cmd);
            }
            Gp0::BlitCpuToVramDecoding(decoder) => match decoder.advance(value) {
                Ok(decoder) => self.gpu.gp0 = Gp0::BlitCpuToVramDecoding(decoder),
                Err(draw_call) => {
                    if tracing::enabled!(Level::TRACE) {
                        tracing::trace!(?draw_call, "decoded");
                        tracing::info!("blit:cpu-to-vram:{:?}", draw_call);
                    }

                    self.gpu.gp0 = Gp0::OffloadBlitCpuToVram {
                        cmd:  draw_call,
                        data: Vec::with_capacity(draw_call.size.area()),
                    }
                }
            },
            Gp0::OffloadBlitCpuToVram { cmd, mut data } => {
                let bytes = value.to_le_bytes();
                let buf_size = cmd.size.area() * size_of::<u16>();
                data.extend(bytes);
                if data.len() >= buf_size {
                    self.gpu_spawn_issue_draw_call(DrawCallKind::BlitCpu2Vram(cmd, data));
                    self.gpu.gp0 = Gp0::WaitingForCmd;
                    self.gpu.gpustat.set_ready_recv_cmd(true);
                } else {
                    self.gpu.gp0 = Gp0::OffloadBlitCpuToVram { cmd, data };
                }
            }

            Gp0::BlitVramToCpuDecoding(decoder) => match decoder.advance(value) {
                Ok(decoder) => {
                    self.gpu.gp0 = Gp0::BlitVramToCpuDecoding(decoder);
                }
                Err(blit) => {
                    assert!(
                        self.gpu.pending_read.is_none(),
                        "multiple pending read, a queue is required"
                    );
                    self.gpu.pending_read = Some(blit);
                    self.evque.spawn(Task::GpuStartBlitVramToCpu);
                }
            },

            // vram to vram copy
            Gp0::BlitVramToVramDecoding(decoder) => match decoder.advance(value) {
                Ok(decoder) => {
                    self.gpu.gp0 = Gp0::BlitVramToVramDecoding(decoder);
                }
                Err(vram_blit) => {
                    if tracing::enabled!(Level::TRACE) {
                        tracing::trace!(?vram_blit, "decoded");
                        tracing::info!("blit:cpu-to-vram:{:?}", vram_blit);
                    }
                    self.gpu.gpustat.set_ready_recv_cmd(true);
                    self.gpu_spawn_issue_draw_call(DrawCallKind::BlitVram2Vram(vram_blit));
                    self.gpu.gp0 = Gp0::WaitingForCmd;
                }
            },

            Gp0::QuickFillDecode(decoder) => match decoder.advance(value) {
                Ok(decoder) => {
                    self.gpu.gp0 = Gp0::QuickFillDecode(decoder);
                }
                Err(draw_call) => {
                    self.gpu.gpustat.set_ready_recv_cmd(true);
                    self.gpu_spawn_issue_draw_call(DrawCallKind::QuickFill(draw_call));
                    self.gpu.gp0 = Gp0::WaitingForCmd;
                }
            },

            Gp0::DrawRectDecode(decoder) => {
                let decoder = decoder.advance(value);
                match decoder {
                    Ok(decoder) => {
                        self.gpu.gp0 = Gp0::DrawRectDecode(decoder);
                    }
                    Err(draw_call) => {
                        if tracing::enabled!(Level::TRACE) {
                            tracing::trace!(?draw_call, "decoded");
                        }
                        self.gpu.gpustat.set_ready_recv_cmd(true);
                        self.gpu_spawn_issue_draw_call(DrawCallKind::Rect(draw_call));
                        self.gpu.gp0 = Gp0::WaitingForCmd;
                    }
                }
            }
            Gp0::DrawPolygonDecode(decoder) => {
                let decoder = decoder.advance(value);
                match decoder {
                    Ok(decoder) => {
                        self.gpu.gp0 = Gp0::DrawPolygonDecode(decoder);
                    }
                    Err(draw_call) => {
                        if tracing::enabled!(Level::TRACE) {
                            tracing::trace!(?draw_call, "decoded");
                        }

                        if let Some(texpage) = draw_call.texpage
                            && draw_call.header.textured()
                        {
                            self.gpu
                                .gpustat
                                .set_texpage_x_base(texpage.texpage_x_base());
                            self.gpu
                                .gpustat
                                .set_texpage_y_base(texpage.texpage_y_base());
                            self.gpu
                                .gpustat
                                .set_texpage_colors(texpage.texpage_colors());
                        }

                        self.gpu.gpustat.set_ready_recv_cmd(true);
                        self.gpu_spawn_issue_draw_call(DrawCallKind::Polygon(draw_call));
                        self.gpu.gp0 = Gp0::WaitingForCmd;
                    }
                }
            }
            Gp0::DrawLineDecode(decoder) => {
                let decoder = decoder.advance(value);
                match decoder {
                    Ok(decoder) => {
                        self.gpu.gp0 = Gp0::DrawLineDecode(decoder);
                    }
                    Err(draw_call) => {
                        if tracing::enabled!(Level::TRACE) {
                            tracing::trace!(?draw_call, "decoded");
                        }
                        self.gpu.gpustat.set_ready_recv_cmd(true);
                        self.gpu_spawn_issue_draw_call(DrawCallKind::Line(draw_call));
                        self.gpu.gp0 = Gp0::WaitingForCmd;
                    }
                }
            }
            Gp0::CpuReadingVram => {
                self.gpu.gpustat.set_ready_recv_cmd(true);
                tracing::warn!("vram->cpu blit interrupted by command")
            }
        };
    }

    pub fn gpu_gp1_cmd(&mut self, value: GpuCmd) {
        tracing::trace!(cmd = ?value.cmd());
        match value.cmd() {
            0x00 => {
                // self.gpu.gp0cmd_queue.clear();
                self.gpu.gpustat = GpuStatReg::new_with_raw_value(0x14802000);
                self.gpu.gpustat.mock_ready();
            }
            0x01 => {
                // self.gpu.gp0cmd_queue.clear();
            }
            0x02 => {
                self.gpu.gpustat.set_irq(false);
            }
            // GP1(03h) - Display Enable
            0x03 => {
                let cmd = Gp1DisplayEnableCmd::new_with_raw_value(value.raw_value());
                self.gpu.gpustat.set_display_enable(cmd.on_off());
            }
            0x04 => {
                let dir = DmaDirection::new_with_raw_value(value.fields().as_());
                self.gpu.gpustat.set_dma_direction(dir);
                self.gpu.compute_dma_request();
            }
            // GP1(05h) - Start of Display area (in VRAM)
            0x05 => {
                let cmd = Gp1StartOfDisplayArea::new_with_raw_value(value.raw_value());
                self.gpu.dp.display_vram_start = cmd.to_u16vec2();
                tracing::debug!("set framebuffer coords");
            }
            0x06 => {
                ///  0-11   X1 (260h+0)       ;12bit       ;\counted in video clock units,
                ///  12-23  X2 (260h+320*8)   ;12bit       ;/relative to HSYNC
                #[bitfield(u32)]
                struct HDisplayRange {
                    #[bits(0..=11, r)]
                    start: u12,
                    #[bits(12..=23, r)]
                    end:   u12,
                }

                let value = HDisplayRange::new_with_raw_value(value.raw_value());
                self.gpu.dp.range_start.x = value.start().as_();
                self.gpu.dp.range_end.x = value.end().as_();
                tracing::debug!("set h framebuffer range");
            }
            0x07 => {
                /// 0-9   Y1 (NTSC=88h-(240/2), (PAL=A3h-(288/2))  ;\scanline numbers on screen,
                /// 10-19 Y2 (NTSC=88h+(240/2), (PAL=A3h+(288/2))  ;/relative to VSYNC
                /// 20-23 Not used (zero)
                #[bitfield(u32)]
                struct VDisplayRange {
                    #[bits(0..=9, r)]
                    start: u10,
                    #[bits(10..=19,r)]
                    end:   u10,
                }
                let value = VDisplayRange::new_with_raw_value(value.raw_value());
                self.gpu.dp.range_start.y = value.start().as_();
                self.gpu.dp.range_end.y = value.end().as_();
                tracing::debug!("set v framebuffer range");
            }
            0x08 => {
                let cmd = DisplayModeCmd::new_with_raw_value(value.raw_value);
                tracing::debug!(pc = %hex(self.cpu.pc), cmd=%hex(cmd), "GP1(08h) display_mode = {cmd:#?}");
                let gpustat = &mut self.gpu.gpustat;
                gpustat.set_h_resolution_1(cmd.hres_1());
                gpustat.set_v_resolution(cmd.vres());
                gpustat.set_video_mode(cmd.video_mode());
                gpustat.set_display_color_depth(cmd.display_color_depth());
                gpustat.set_v_interlace(cmd.v_interlace());
                gpustat.set_h_resolution_2(cmd.hres_2());
                gpustat.set_reverse_flag(cmd.screen_hflip());
            }
            // get gpu info
            0x10 => {
                let Some(cmd) = self.gpu.get_gpu_info_cmd(value) else {
                    return;
                };
                match cmd {
                    GpuInfoCmd::Unused00 | GpuInfoCmd::Unused01 => {}
                    GpuInfoCmd::TexWindow => {
                        self.gpu.gp0read =
                            unsafe { transmute::<u32, [u16; 2]>(self.gpu.tex_window.raw_value()) };
                    }
                }
            }
            value => todo!("gp1 command: {}", hex(value)),
        }
    }

    pub(crate) async fn task_gpu_issue_draw_call(&mut self) {
        let mut draw_call_kinds = mem::replace(
            &mut self.gpu.pending_draw_calls,
            VecDeque::new_in(self.alloc),
        );
        for kind in draw_call_kinds.drain(..) {
            self.gpu.issue_draw_call(kind).await
        }
        self.gpu.pending_draw_calls = draw_call_kinds;
    }

    pub fn gpu_spawn_issue_draw_call(&mut self, kind: DrawCallKind) {
        self.gpu
            .pending_draw_calls
            .push_back(self.gpu.create_draw_call(kind));
        self.evque.spawn(Task::GpuIssueDrawCall);
    }
}

impl<A: Allocator> Emu<A> {
    pub(crate) async fn run_gpu(&mut self) {
        if let Some(blit) = &self.gpu.ongoing_read {
            let vram = VRAM.lock().await;
            self.gpu.gpustat.set_gp0_rffl(true);

            let mut cursor = VramCursor::new(blit.dest, blit.dest + blit.size);
            for at in cursor.iter() {
                let value = vram_get(&*vram, at);
                self.gpu.gp0read_fifo.push_back(value);
            }
            self.gpu.ongoing_read = None;
            self.gpu.gpustat.set_ready_recv_cmd(true);
        }
    }

    pub fn gpu_flush_draw_calls(&mut self) {
        self.evque.spawn(Task::GpuFlushDrawCalls);
    }
}

impl<A: Allocator + Clone> Emu<A> {
    pub fn gpu_reconnect(&mut self, other: &Emu<A>) {
        self.gpu.conn = other.gpu.conn.clone();
    }
}

impl<A: Allocator> GpuState<A> {
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
}

impl<A: Allocator + Copy> GpuState<A> {
    pub async fn task_gpu_flush_draw_calls(&mut self) {
        self.draw_idx += 1;
        self.conn
            .draw_call_chan
            .0
            .send(DrawCallMsg::Flush(FlushMsg {
                display:  self.dp.clone(),
                gpustat:  self.gpustat,
                draw_idx: self.draw_idx,
            }))
            .await
            .unwrap();
    }

    pub async fn request_vram(&mut self) {
        let (tx, rx) = kanal::bounded_async(0);
        self.conn
            .draw_call_chan
            .0
            .send(DrawCallMsg::RequestVram(&VRAM, tx))
            .await
            .unwrap();
        rx.recv().await.expect("ack channel closed");
    }

    pub fn compute_dma_request(&mut self) {
        let direction = self.gpustat.dma_direction();
        match direction {
            DmaDirection::Off => {
                self.gpustat.set_dma_request(false);
            }
            DmaDirection::Unknown => {
                // let fifo_available = !self.gp0cmd_queue.is_full();
                self.gpustat.set_dma_request(true);
            }
            DmaDirection::CpuToGp0 => {
                self.gpustat.set_dma_request(false);
            }
            DmaDirection::CpuReadToCpu => {
                self.gpustat.set_dma_request(self.gpustat.gp0_rffl());
            }
        }
    }
}

fn vram_get(vram: &[u16], coord: VramCoord) -> u16 {
    let coord = coord.wrap();
    let addr = coord.x as usize + coord.y as usize * kb(1);

    vram[addr]
}

impl<A: Allocator> GpuState<A> {
    pub fn flip_even_odd(&mut self, even_odd: Option<DrawEvenOdd>) {
        if self.gpustat.v_interlace() {
            let even_odd = even_odd.unwrap_or_else(|| self.gpustat.even_odd_in_vblank());
            self.gpustat.set_even_odd_in_vblank(!even_odd);
        }
    }

    pub fn reset_renderer(&mut self) {
        self.draw_idx = 0;
        let (tx, rx) = kanal::bounded_async(0);
        self.conn
            .draw_call_chan
            .0
            .as_sync()
            .send(DrawCallMsg::Reset(tx))
            .unwrap();
        rx.as_sync().recv_timeout(Duration::from_secs(2)).unwrap();
        self.drain_messages();
    }

    pub fn drain_messages(&self) {
        while let Ok(Some(_)) = self.conn.draw_call_chan.1.as_sync().try_recv() {}
        while let Ok(Some(_)) = self.conn.vram_out_chan.1.as_sync().try_recv() {}
    }
}

#[must_use]
pub fn halfwords(word: u32) -> [u16; 2] {
    [word as u16, (word >> 16) as u16]
}

#[derive(Debug, Clone, Endecode, Default)]
pub enum Gp0 {
    #[default]
    WaitingForCmd,
    Busy,

    QuickFillDecode(QuickFillDecoder),

    DrawPolygonDecode(DrawPolygonDecoder),
    DrawLineDecode(DrawLineDecoder),
    DrawRectDecode(DrawRectDecoder),
    // copy commands
    BlitCpuToVramDecoding(Gp0HostBlitDecoder),
    OffloadBlitCpuToVram {
        cmd:  Gp0HostBlit,
        #[endecode(alloc(global))]
        data: Vec<u8>,
    },
    BlitVramToCpuDecoding(Gp0HostBlitDecoder),
    BlitVramToVramDecoding(Gp0VramBlitDecoder),
    CpuReadingVram,
}

#[bitfield(u32)]
#[must_use]
pub struct GpuCmd {
    #[bits(0..=23, r)]
    fields: u24,
    #[bits(24..=31, r)]
    cmd:    u8,
}

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
/// 26    IDLE  Ready to receive Cmd Word   (0=No, 1=Ready)  ;GP0(...) ;via GP0 write
/// 27    RFFL  GP0 read FIFO full          (0=No, 1=Full)   ;GP0(C0h) ;via GP0 read
/// 28    WFEP  GP0 write FIFO empty        (0=No, 1=Empty)  ;GP0(...) ;via GP0 write
/// 29-30 DMA Direction (0=Off, 1=?, 2=CPUtoGP0, 3=GPUREADtoCPU)    ;GP1(04h).0-1
/// 31    Drawing even/odd lines in interlace mode (0=Even or Vblank, 1=Odd)
///
/// Credits to PSX-SPX by Martin Korth [Gpu Status Register](https://problemkaputt.de/psx-spx.htm#gpustatusregister)
///
#[bitfield(u32, debug)]
#[derive(Default, derive_more::Into, Endecode)]
#[must_use]
pub struct GpuStatReg {
    #[bits(0..=3, rw)]
    texpage_x_base:      u4,
    #[bit(4, rw)]
    texpage_y_base:      u1,
    #[bits(5..=6, rw)]
    semi_transparency:   u2,
    #[bits(7..=8, rw)]
    texpage_colors:      TextureColorMode,
    #[bit(9, rw)]
    dither:              bool,
    #[bit(10, rw)]
    draw_to_display:     bool,
    /// `GPU_STAT.11`
    #[bit(11, rw)]
    set_mask:            bool,
    /// `GPU_STAT.12`
    #[bit(12, rw)]
    draw_pixels:         DrawPixels,
    #[bit(13, rw)]
    interlace_field:     bool,
    #[bit(14, rw)]
    reverse_flag:        ReverseFlag,
    #[bit(15, rw)]
    texture_disable:     bool,
    #[bit(16, rw)]
    h_resolution_2:      HRes2,
    #[bits(17..=18, rw)]
    h_resolution_1:      HRes1,
    #[bit(19, rw)]
    v_resolution:        VRes,
    #[bit(20, rw)]
    video_mode:          VideoMode,
    #[bit(21, rw)]
    display_color_depth: DisplayColorDepth,
    #[bit(22, rw)]
    v_interlace:         bool,
    #[bit(23, rw)]
    display_enable:      bool,
    #[bit(24, rw)]
    irq:                 bool,
    #[bit(25, rw)]
    dma_request:         bool,
    #[bit(26, rw)]
    ready_recv_cmd:      bool,
    #[bit(27, rw)]
    gp0_rffl:            bool,
    #[bit(28, rw)]
    gp0_wfep:            bool,
    #[bits(29..=30, rw)]
    dma_direction:       DmaDirection,
    #[bit(31, rw)]
    even_odd_in_vblank:  DrawEvenOdd,
}

impl GpuStatReg {
    pub fn mock_ready(&mut self) {
        self.set_ready_recv_cmd(true);
        self.set_gp0_wfep(true);
        self.set_gp0_rffl(false);
    }

    #[must_use]
    pub fn resolution(self) -> U16Vec2 {
        let vertical = match (self.v_resolution(), self.v_interlace()) {
            (VRes::Res480, true) => 480,
            _ => 240,
        };
        let horizontal = match self.h_resolution_2() {
            HRes2::Standard => match self.h_resolution_1() {
                HRes1::Res256 => 256,
                HRes1::Res320 => 320,
                HRes1::Res512 => 512,
                HRes1::Res640 => 640,
            },
            HRes2::Res368 => 368,
        };
        U16Vec2::new(horizontal, vertical)
    }

    #[must_use]
    pub fn texpage_base(self) -> U8Vec2 {
        U8Vec2::new(self.texpage_x_base().as_u8(), self.texpage_y_base().as_u8())
    }
}

#[derive(Debug)]
#[bitenum(u1, exhaustive = true)]
pub enum DrawPixels {
    Always           = 0x0,
    NotToMaskedAreas = 0x1,
}

#[derive(Debug)]
#[bitenum(u1, exhaustive = true)]
enum ReverseFlag {
    Normal    = 0x0,
    Distorted = 0x1,
}

#[derive(Debug, derive_more::Display)]
#[bitenum(u1, exhaustive = true)]
pub enum VideoMode {
    #[display("NTSC")]
    Ntsc = 0x0,
    #[display("PAL")]
    Pal  = 0x1,
}

#[derive(Debug)]
#[bitenum(u2, exhaustive = true)]
pub enum DmaDirection {
    Off          = 0x0,
    Unknown      = 0x1,
    CpuToGp0     = 0x2,
    CpuReadToCpu = 0x3,
}

#[derive(Debug)]
#[bitenum(u1, exhaustive = true)]
pub enum DrawEvenOdd {
    EvenOrVBlank = 0x0,
    Odd          = 0x1,
}

impl ops::Not for DrawEvenOdd {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            DrawEvenOdd::EvenOrVBlank => Self::Odd,
            DrawEvenOdd::Odd => Self::EvenOrVBlank,
        }
    }
}

/// #  GP0(E1h) - Draw Mode setting (aka "Texpage")
///
/// Likely sets the relevant bits in the `GpuStat` register.
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
#[derive(derive_more::Debug, Default, derive_more::Into, Endecode)]
#[endecode(alloc(default))]
#[debug("{}", hex(self.raw_value))]
#[must_use]
pub struct TexpageCmd {
    #[bits(0..=3, rw)]
    texpage_x_base:    u4,
    #[bit(4, rw)]
    texpage_y_base:    u1,
    #[bits(5..=6, rw)]
    semi_transparency: u2,
    #[bits(7..=8, rw)]
    texpage_colors:    TextureColorMode,
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

impl TexpageCmd {
    #[must_use]
    pub fn to_u8vec2(self) -> U8Vec2 {
        U8Vec2::new(self.texpage_x_base().as_u8(), self.texpage_y_base().as_u8())
    }
}

#[bitenum(u2, exhaustive = true)]
#[derive(Debug, Default)]
pub enum TextureColorMode {
    C4Bit        = 0x0,
    C7Bit        = 0x1,
    #[default]
    C15BitDirect = 0x2,
    // this is usually reserved but we can use it to
    // mark 24bit direct mode (for untextured rectangles etc)
    C24BitDirect = 0x3,
}

/// # GP1(03h) - Display Enable
/// ```md
///   0     Display On/Off   (0=On, 1=Off)                         ;GPUSTAT.23
///   1-23  Not used (zero)
/// ```
#[bitfield(u32)]
#[must_use]
pub struct Gp1DisplayEnableCmd {
    #[bit(0, rw)]
    on_off: bool,
}

/// # GP1(05h) - Start of Display area (in VRAM)
/// ```md
///   0-9   X (0-1023)    (halfword address in VRAM)  (relative to begin of VRAM)
///   10-18 Y (0-511)     (scanline number in VRAM)   (relative to begin of VRAM)
///   19-23 Not used (zero)
/// ```
/// Upper/left Display source address in VRAM. The size and target position on screen is set via Display Range registers; target=X1,Y2; size=(X2-X1/cycles_per_pix), (Y2-Y1).
/// Unknown if using Y values in 512-1023 range is supported (with 2 MB VRAM).
#[bitfield(u32)]
struct Gp1StartOfDisplayArea {
    #[bits(0..=9, rw)]
    x: u10,
    #[bits(10..=18, rw)]
    y: u9,
}

impl Gp1StartOfDisplayArea {
    fn to_u16vec2(self) -> U16Vec2 {
        U16Vec2::new(self.x().as_u16(), self.y().as_u16())
    }
}

/// # GP1(10h) - Get GPU Info
///
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
#[bitfield(u32, debug)]
#[derive(Default, Endecode)]
#[must_use]
pub struct Gp0TexWindowCmd {
    #[bits(0..=4, r)]
    mask_x:   u5,
    #[bits(5..=9, r)]
    mask_y:   u5,
    #[bits(10..=14, r)]
    offset_x: u5,
    #[bits(15..=19, r)]
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
#[bitfield(u32, debug)]
#[must_use]
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
    screen_hflip:        ReverseFlag,
}

#[bitenum(u2, exhaustive = true)]
#[derive(Debug)]
pub enum HRes1 {
    Res256,
    Res320,
    Res512,
    Res640,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
pub enum HRes2 {
    Standard,
    Res368,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Debug)]
pub enum VRes {
    Res240,
    Res480,
}

#[bitenum(u1, exhaustive = true)]
#[derive(Default, Debug)]
pub enum DisplayColorDepth {
    #[default]
    Depth15Bit,
    Depth24Bit,
}

/// # GP1(05h) - Start of Display area (in VRAM)
///
/// ```md
///   0-9   X (0-1023)    (halfword address in VRAM)  (relative to begin of VRAM)
///   10-18 Y (0-511)     (scanline number in VRAM)   (relative to begin of VRAM)
///   19-23 Not used (zero)
/// ```
///
/// Upper/left Display source address in VRAM. The size and target
/// position on screen is set via Display Range registers; target=X1,Y2;
/// size=(X2-X1/cycles_per_pix), (Y2-Y1). Unknown if using Y values in 512-1023
/// range is supported (with 2 MB VRAM).
#[bitfield(u19)]
#[must_use]
pub struct StartOfDisplayCmd {
    #[bits(0..=9, rw)]
    address:  u10,
    #[bits(10..=18, rw)]
    scanline: u9,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoEventKind {
    Hblank,
    Vblank,
}

impl<A: Allocator> GpuState<A> {
    #[must_use]
    pub fn video_cycles_per_scanline(&self) -> u64 {
        match self.gpustat.video_mode() {
            VideoMode::Ntsc => 3413,
            VideoMode::Pal => 3406,
        }
    }

    #[must_use]
    pub fn cycles_vblank_pending(&self) -> u64 {
        let scanlines = if self.dp.current_scanline > self.dp.v_end() {
            Display::NTSC_TOTAL_LINES - self.dp.current_scanline + self.dp.v_end()
        } else {
            self.dp.v_end() - self.dp.current_scanline
        };
        let cycles_per_scanline = self.video_cycles_per_scanline();
        scanlines * cycles_per_scanline
            + (Display::NTSC_TOTAL_VCYCLES_PER_LINE - self.dp.video_cycle_in_scanline)
    }

    #[must_use]
    pub fn cycles_hblank_pending(&self) -> u64 {
        if self.dp.video_cycle_in_scanline > self.dp.h_end() {
            Display::NTSC_TOTAL_VCYCLES_PER_LINE - self.dp.video_cycle_in_scanline + self.dp.h_end()
        } else {
            self.dp.h_end() - self.dp.video_cycle_in_scanline
        }
    }

    #[must_use]
    pub fn pending_event(&self) -> (VideoEventKind, u64) {
        let _hblank = self.cycles_hblank_pending();
        let vblank = self.cycles_vblank_pending();

        // if hblank < vblank {
        //     (VideoEventKind::Hblank, hblank)
        // } else {
        //     (VideoEventKind::Vblank, vblank)
        // }
        (VideoEventKind::Vblank, vblank)
    }

    pub async fn issue_draw_call(&mut self, draw_call: DrawCall) {
        self.draw_calls_in_flight = true;
        self.conn
            .draw_call_chan
            .0
            .send(DrawCallMsg::DrawCall(draw_call))
            .await
            .unwrap();
    }

    pub fn create_draw_call(&self, kind: DrawCallKind) -> DrawCall {
        DrawCall {
            gpustat:    self.gpustat,
            tex_window: self.tex_window,
            inner:      kind,
            draw_reg:   self.draw_reg.clone(),
            display:    self.dp.clone(),
        }
    }
}

#[derive(derive_more::Debug, Clone, Default, Endecode)]
pub struct Display {
    /// video event queue
    pub video_cycle:             u64,
    pub video_cycle_in_scanline: u64,

    range_start: U64Vec2,
    range_end:   U64Vec2,

    current_scanline: u64,

    /// used for storing fractional part when converting
    /// from cpu to video cycles
    fract_01: u64,

    pub display_vram_start: U16Vec2,
}

/// The functionality in this trait largely deals with video cycles (or video clock units).
/// These cycles are *not* relative to the resolution, the way dot clocks are, meaning they
/// are not absolute dot positions, but relative timings tied to HSYNC.
///
/// see <https://psx-spx.consoledev.net/graphicsprocessingunitgpu/#gp106h-horizontal-display-range-on-screen>
impl<A: Allocator + Copy> Emu<A> {
    const fn cpu_cycles_to_video_cycles(&mut self, cycles: u64) -> u64 {
        // this might be based on the actual console hardware not on the
        // video mode you set in the gpu
        let factor = match self.gpu.gpustat.video_mode() {
            VideoMode::Ntsc => 715909,
            VideoMode::Pal => 709379,
        };
        let cycles = cycles * factor;
        let cycles = cycles + self.gpu.dp.fract_01;
        self.gpu.dp.fract_01 = cycles % 451584;

        cycles / 451584
    }

    fn video_cycles_to_cpu_cycles_approx(&self, cycles: u64) -> u64 {
        let factor = match self.gpu.gpustat.video_mode() {
            VideoMode::Ntsc => 715909,
            VideoMode::Pal => 709379,
        };
        cycles * 451584 / factor
    }

    /// returns start (topleft) and end (bottomright)
    pub fn dp_coords(&self) -> (U16Vec2, U16Vec2) {
        let width = match self.gpu.gpustat.h_resolution_2() {
            HRes2::Standard => match self.gpu.gpustat.h_resolution_1() {
                HRes1::Res256 => 256,
                HRes1::Res320 => 320,
                HRes1::Res512 => 512,
                HRes1::Res640 => 640,
            },
            HRes2::Res368 => 368,
        };
        let height = match self.gpu.gpustat.v_resolution() {
            VRes::Res240 => 240,
            VRes::Res480 => 480,
        };
        let start = self.gpu.dp.display_vram_start;
        (start, start + U16Vec2::new(width, height))
    }

    pub async fn run_video_io(&mut self, by_cpu_cycles: u64) {
        let cycles = self.cpu_cycles_to_video_cycles(by_cpu_cycles);
        self.gpu.dp.update_ranges();
        self.run_video_events(cycles).await;
    }

    async fn run_video_events(&mut self, advance_by: u64) {
        let cycles_per_scanline = self.gpu.video_cycles_per_scanline();
        self.gpu.dp.video_cycle_in_scanline += advance_by;
        if self.gpu.dp.video_cycle_in_scanline < cycles_per_scanline {
            self.timers.trigger_hblank_end();
            return;
        }
        let scanlines_to_run = self.gpu.dp.video_cycle_in_scanline / cycles_per_scanline;
        self.gpu.dp.video_cycle_in_scanline %= cycles_per_scanline;

        for _ in 0..scanlines_to_run {
            let dp = &self.gpu.dp;
            let new_vblank = !dp.in_vblank() && dp.in_vblank_with(dp.current_scanline + 1);

            if new_vblank {
                self.run_vblank().await;
            }

            self.timers.trigger_hblank_start();
            self.timers.trigger_hblank_end();

            self.gpu.dp.current_scanline += 1;

            let dp = &mut self.gpu.dp;
            if dp.current_scanline >= Display::NTSC_TOTAL_LINES {
                dp.current_scanline = 0;
                self.gpu.flip_even_odd(None);
            }
        }

        if self.gpu.dp.in_hblank() {
            self.timers.trigger_hblank_start();
        }
    }
}

impl Display {
    pub const NTSC_TOTAL_VCYCLES_PER_LINE: u64 = 3413;
    const NTSC_TOTAL_LINES: u64 = 263;
    const NTSC_ACTIVE_H_START: u64 = 488;
    const NTSC_ACTIVE_H_END: u64 = 3288;
    const NTSC_ACTIVE_V_START: u64 = 16;
    const NTSC_ACTIVE_V_END: u64 = 256;

    fn update_ranges(&mut self) {
        self.range_start = u64vec2(Self::NTSC_ACTIVE_H_START, Self::NTSC_ACTIVE_V_START);
        self.range_end = u64vec2(Self::NTSC_ACTIVE_H_END, Self::NTSC_ACTIVE_V_END);
    }

    const fn h_start(&self) -> u64 {
        self.range_start.x
    }

    const fn h_end(&self) -> u64 {
        self.range_end.x
    }

    const fn v_start(&self) -> u64 {
        self.range_start.y
    }

    const fn v_end(&self) -> u64 {
        self.range_end.y
    }

    pub(crate) fn in_hblank(&self) -> bool {
        !(self.h_start()..self.h_end()).contains(&self.video_cycle_in_scanline)
    }

    pub(crate) fn in_vblank(&self) -> bool {
        !(self.v_start()..self.v_end()).contains(&self.current_scanline)
    }

    pub(crate) fn in_vblank_with(&self, line: u64) -> bool {
        !(self.v_start()..self.v_end()).contains(&line)
    }
}

pub mod draw_call;

use std::mem::transmute;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;

use arbitrary_int::prelude::*;
use bitbybit::bitenum;
use bitbybit::bitfield;
use derive_more as d;
use glam::U8Vec2;
use glam::U64Vec2;
use glam::u64vec2;
use heapless::Deque;
use heapless::binary_heap::Min;
use pchan_utils::AsyncChan;
use pchan_utils::hex;
use tracing::Level;
use tracing::instrument;

use crate::Bus;
use crate::Emu;
use crate::gpu::draw_call::DrawCall;
use crate::gpu::draw_call::DrawCallDecoder;
use crate::gpu::draw_call::DrawCallKind;
use crate::gpu::draw_call::DrawLineDecoder;
use crate::gpu::draw_call::DrawOptsRegister;
use crate::gpu::draw_call::DrawPolygonDecoder;
use crate::gpu::draw_call::DrawPolygonHeader;
use crate::gpu::draw_call::DrawRectDecoder;
use crate::gpu::draw_call::Gp0SetDrawAreaCmd;
use crate::gpu::draw_call::Gp0SetDrawOffsetCmd;
use crate::gpu::draw_call::Gp0SetMaskBitCmd;
use crate::io::CastIOFrom;
use crate::io::CastIOInto;
use crate::io::IOResult;
use crate::io::UnhandledIO;
use crate::io::irq::Interrupts;
use crate::io::irq::Irq;
use crate::io::vblank::VBlank;
use crate::memory::kb;
use crate::memory::mb;

pub static VBLANK_COUNT: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct Conn<D> {
    pub draw_call_chan: AsyncChan<Vec<D>>,
    pub vram_in_chan:   AsyncChan<Box<[u16]>>,
    pub vram_out_chan:  AsyncChan<Box<[u16]>>,
}

#[derive(derive_more::Debug, Clone)]
pub struct GpuState {
    #[debug(skip)]
    pub vram:            Box<[u16]>,
    #[debug(skip)]
    pub gpustat:         GpuStatReg,
    pub gp0:             Gp0,
    pub gp0read:         [u16; 2],
    pub gp0cmd_queue:    Deque<u32, 16>,
    pub gp0read_queue:   Deque<u32, 32>,
    pub gp1cmd_queue:    Deque<u32, 16>,
    pub dp:              Display,
    /// GP0(0xe2) - Texture Window setting
    pub tex_window:      Gp0TexWindowCmd,
    // TODO: remove this
    pub draw_opts_reg:   DrawOptsRegister,
    pub draw_call_queue: Vec<DrawCall>,
    pub model:           GpuModel,

    #[debug(skip)]
    pub conn:          Conn<DrawCall>,
    waiting_on_render: bool,
    pub last_vblank:   Instant,
    pub vblank_signal: bool,
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
            vram: create_vram(),
            gp0: Gp0::WaitingForCmd,
            gp0read: Default::default(),
            gp0read_queue: Deque::new(),
            gp0cmd_queue: Deque::new(),
            gp1cmd_queue: Deque::default(),
            model: GpuModel::default(),
            tex_window: Default::default(),
            draw_opts_reg: Default::default(),
            draw_call_queue: vec![],
            conn: Conn {
                draw_call_chan: kanal::bounded_async(0),
                vram_in_chan:   kanal::bounded_async(0),
                vram_out_chan:  kanal::bounded_async(1),
            },
            dp: Display::default(),
            waiting_on_render: false,
            last_vblank: Instant::now(),
            vblank_signal: false,
        }
    }
}

pub fn create_vram() -> Box<[u16]> {
    vec![0; mb(1)].into_boxed_slice()
}

pub trait Gpu: Bus + Interrupts {
    #[cfg_attr(debug_assertions, instrument(skip(self), "gpu:r"))]
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
                                self.gpu_mut().gpustat.set_ready_recv_cmd(true);
                                Gp0::WaitingForCmd
                            }
                            false => Gp0::CpRectVramToCpu(Gp0CpRect::RecvData(cursor)),
                        };
                        self.gpu_mut().gp0 = gp0;
                    }
                    Gp0::CpRectCpuToVram(_) => {}
                    _ => {
                        self.gpu_mut().gpustat.set_ready_recv_cmd(true);
                    }
                }
                tracing::trace!(gp0read = ?self.gpu().gp0read);
                Ok(self.gpu().gp0read.io_from_u32())
            }
            0x1f80_1814 => Ok(self.gpu().gpustat.io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
    }
    fn read_pure<T: Copy>(&self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        match address {
            0x1f80_1814 => Ok(self.gpu().gpustat.io_from_u32()),
            _ => Err(UnhandledIO(address)),
        }
    }
    #[cfg_attr(debug_assertions, instrument(skip(self, value), "gpu:w"))]
    fn write<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
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
                self.gpu_mut()
                    .gp0cmd_queue
                    .push_back(value.io_into_u32_overwrite(0x0))
                    .unwrap();
                Ok(())
            }
            0x1f80_1814 => {
                self.gpu_mut()
                    .gp1cmd_queue
                    .push_back(value.io_into_u32())
                    .unwrap();
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }

    fn gp0reduce(&mut self, cmd: GpuCmd) -> Gp0 {
        match cmd.cmd() {
            // nop
            0x0 | 0x4..=0x1e | 0xe0 | 0xe7..=0xef => Gp0::WaitingForCmd,

            0x01 => {
                // clear cache?
                Gp0::WaitingForCmd
            }

            0xc0..=0xdf => {
                self.gpu_mut().gpustat.set_ready_recv_cmd(false);
                Gp0::CpRectVramToCpu(Gp0CpRect::RecvDest)
            }
            0x1f => {
                self.gpu_mut().gpustat.set_irq(true);
                self.trigger_irq(Irq::Irq1Gpu);
                Gp0::WaitingForCmd
            }
            0xa0..=0xbf => {
                tracing::debug!("start cpu to vram copy");
                self.gpu_mut().gpustat.set_ready_recv_cmd(false);
                Gp0::CpRectCpuToVram(Gp0CpRect::RecvDest)
            }
            0xe1 => {
                let texpage = TexpageCmd::new_with_raw_value(cmd.raw_value());

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
            // GP0(E2h) - Texture Window setting
            0xe2 => {
                let cmd = Gp0TexWindowCmd::new_with_raw_value(cmd.raw_value());
                self.gpu_mut().tex_window = cmd;

                Gp0::WaitingForCmd
            }
            // GP0(E3h) - Set Drawing Area top left (X1,Y1)
            0xe3 => {
                let opts = &mut self.gpu_mut().draw_opts_reg;
                let cmd = Gp0SetDrawAreaCmd::new_with_raw_value(cmd.raw_value());
                opts.draw_area_top_left.x = cmd.x_coord().as_();
                opts.draw_area_top_left.y = cmd.y_coord_v2().as_();

                Gp0::WaitingForCmd
            }
            // GP0(E4h) - Set Drawing Area bottom right (X2,Y2)
            0xe4 => {
                let opts = &mut self.gpu_mut().draw_opts_reg;
                let cmd = Gp0SetDrawAreaCmd::new_with_raw_value(cmd.raw_value());
                opts.draw_area_bottom_right.x = cmd.x_coord().as_();
                opts.draw_area_bottom_right.y = cmd.y_coord_v2().as_();

                Gp0::WaitingForCmd
            }
            // GP0(E5h) - Set Drawing Offset (X,Y)
            0xe5 => {
                let opts = &mut self.gpu_mut().draw_opts_reg;
                let cmd = Gp0SetDrawOffsetCmd::new_with_raw_value(cmd.raw_value());
                opts.draw_offset.x = cmd.x_offset().as_();
                opts.draw_offset.y = cmd.y_offset().as_();

                Gp0::WaitingForCmd
            }
            // GP0(E6h) - Mask Bit Setting
            0xe6 => {
                let cmd = Gp0SetMaskBitCmd::new_with_raw_value(cmd.raw_value());
                let gpustat = &mut self.gpu_mut().gpustat;
                gpustat.set_set_mask(cmd.draw_mask());
                gpustat.set_draw_pixels(cmd.draw_pixels());

                Gp0::WaitingForCmd
            }
            // Draw polygon
            0x20..=0x3f => {
                self.gpu_mut().gpustat.set_ready_recv_cmd(false);
                Gp0::DrawPolygonDecode(DrawPolygonDecoder::new(cmd.raw_value()))
            }
            // Draw line
            0x40..=0x5f => {
                self.gpu_mut().gpustat.set_ready_recv_cmd(false);
                Gp0::DrawLineDecode(DrawLineDecoder::new(cmd.raw_value()))
            }
            // Draw Rect
            0x60..=0x7f => {
                self.gpu_mut().gpustat.set_ready_recv_cmd(false);
                Gp0::DrawRectDecode(DrawRectDecoder::new(cmd.raw_value()))
            }
            0x80..=0x9f => {
                self.gpu_mut().gpustat.set_ready_recv_cmd(false);
                Gp0::CpRectVramToVram(Gp0VramCpRect::RecvSrc)
            }
            value => todo!("gp0 command: {}", hex(value)),
        }
    }

    fn gp0_cmd_queue_push(&mut self, value: u32) -> Result<(), u32> {
        self.gpu_mut().gp0cmd_queue.push_back(value)
    }

    fn gp0_cmd_queue_push_or_flush(&mut self, value: u32) {
        while self.gp0_cmd_queue_push(value).is_err() {
            self.gp0_cmd_queue_flush();
        }
    }

    #[cfg_attr(debug_assertions, instrument(skip_all))]
    fn gp0_cmd_queue_flush(&mut self) {
        while let Some(value) = self.gpu_mut().gp0cmd_queue.pop_front() {
            tracing::trace!(gp0cmd = %hex(value));
            let cmd = GpuCmd::new_with_raw_value(value);
            let gp0 = match &mut self.gpu_mut().gp0 {
                Gp0::WaitingForCmd => self.gp0reduce(cmd),
                Gp0::CpRectCpuToVram(Gp0CpRect::RecvDest) => {
                    let dest: VramCoord = unsafe { transmute(value) };
                    let dest = dest.copy_cmd_pos_mask();
                    tracing::debug!("cpu to vram copy destination: {dest:?}");
                    Gp0::CpRectCpuToVram(Gp0CpRect::RecvSize { dest })
                }
                Gp0::CpRectCpuToVram(Gp0CpRect::RecvSize { dest }) => {
                    let size: VramCoord = unsafe { transmute(value) };
                    let size = size.copy_cmd_size_mask();
                    tracing::debug!("cpu to vram copy size: {size:?}");
                    Gp0::CpRectCpuToVram(Gp0CpRect::RecvData(VramCursor::new(*dest, *dest + size)))
                }
                Gp0::CpRectCpuToVram(Gp0CpRect::RecvData(cursor)) => {
                    let mut cursor = *cursor;
                    for (at, halfword) in cursor.iter().take(2).zip(halfwords(value)) {
                        self.gpu_mut().vram_write(at, halfword);
                    }
                    match cursor.done() {
                        true => {
                            self.gpu_mut().gpustat.set_ready_recv_cmd(true);

                            Gp0::WaitingForCmd
                        }
                        false => Gp0::CpRectCpuToVram(Gp0CpRect::RecvData(cursor)),
                    }
                }

                Gp0::CpRectVramToCpu(Gp0CpRect::RecvDest) => {
                    let dest: VramCoord = unsafe { transmute(value) };
                    let dest = dest.copy_cmd_pos_mask();
                    Gp0::CpRectVramToCpu(Gp0CpRect::RecvSize { dest })
                }

                Gp0::CpRectVramToCpu(Gp0CpRect::RecvSize { dest }) => {
                    let dest = *dest;
                    let size: VramCoord = unsafe { transmute(value) };
                    let size = size.copy_cmd_size_mask();

                    self.gpu_mut().gpustat.set_ready_send_vram(true);
                    Gp0::CpRectVramToCpu(Gp0CpRect::RecvData(VramCursor::new(dest, dest + size)))
                }
                // cancel vram to cpu?
                Gp0::CpRectVramToCpu(Gp0CpRect::RecvData(_)) => self.gpu().gp0.clone(),

                // vram to vram copy
                Gp0::CpRectVramToVram(Gp0VramCpRect::RecvSrc) => {
                    let src: VramCoord = unsafe { transmute(value) };
                    let src = src.copy_cmd_pos_mask();
                    Gp0::CpRectVramToVram(Gp0VramCpRect::RecvDest { src })
                }
                Gp0::CpRectVramToVram(Gp0VramCpRect::RecvDest { src }) => {
                    let dest: VramCoord = unsafe { transmute(value) };
                    let dest = dest.copy_cmd_pos_mask();
                    Gp0::CpRectVramToVram(Gp0VramCpRect::RecvSize { src: *src, dest })
                }
                Gp0::CpRectVramToVram(Gp0VramCpRect::RecvSize { src, dest }) => {
                    let size: VramCoord = unsafe { transmute(value) };
                    let size = size.copy_cmd_size_mask();
                    let mut src_cursor = VramCursor::new(*src, size);
                    let mut dest_cursor = VramCursor::new(*dest, size);

                    for (src, dest) in src_cursor.iter().zip(dest_cursor.iter()) {
                        let value = self.gpu_mut().vram_read_direct(src);
                        self.gpu_mut().vram_write(dest, value);
                    }

                    self.gpu_mut().gpustat.set_ready_recv_cmd(true);
                    Gp0::WaitingForCmd
                }

                Gp0::DrawRectDecode(decoder) => {
                    let decoder = decoder.advance(value);
                    match decoder {
                        Ok(decoder) => Gp0::DrawRectDecode(decoder),
                        Err(draw_call) => {
                            tracing::trace!(?draw_call, "decoded");
                            self.gpu_mut().gpustat.set_ready_recv_cmd(true);
                            self.issue_draw_call(DrawCallKind::Rect(draw_call));
                            Gp0::WaitingForCmd
                        }
                    }
                }
                Gp0::DrawPolygonDecode(decoder) => {
                    // we put this back in at the end of the function
                    let decoder = std::mem::take(decoder);

                    let decoder = decoder.advance(value);
                    match decoder {
                        Ok(decoder) => Gp0::DrawPolygonDecode(decoder),
                        Err(draw_call) => {
                            tracing::trace!(?draw_call, "decoded");
                            self.gpu_mut().gpustat.set_ready_recv_cmd(true);
                            self.issue_draw_call(DrawCallKind::Polygon(draw_call));
                            Gp0::WaitingForCmd
                        }
                    }
                }
                Gp0::DrawLineDecode(decoder) => {
                    let decoder = std::mem::take(decoder);
                    let decoder = decoder.advance(value);
                    match decoder {
                        Ok(decoder) => Gp0::DrawLineDecode(decoder),
                        Err(draw_call) => {
                            tracing::trace!(?draw_call, "decoded");
                            self.gpu_mut().gpustat.set_ready_recv_cmd(true);
                            self.issue_draw_call(DrawCallKind::Line(draw_call));
                            Gp0::WaitingForCmd
                        }
                    }
                }
            };

            self.gpu_mut().gp0 = gp0;
        }
    }

    fn gp1_cmd_queue_flush(&mut self) {
        while let Some(value) = self.gpu_mut().gp1cmd_queue.pop_front() {
            let value = GpuCmd::new_with_raw_value(value.io_into_u32());
            tracing::trace!(cmd = ?value.cmd());
            match value.cmd() {
                0x00 => {
                    self.gpu_mut().gp0cmd_queue.clear();
                    self.gpu_mut().gpustat = GpuStatReg::new_with_raw_value(0x14802000);
                    self.gpu_mut().gpustat.mock_ready();
                }
                0x01 => {
                    self.gpu_mut().gp0cmd_queue.clear();
                }
                0x02 => {
                    self.gpu_mut().gpustat.set_irq(false);
                }
                // GP1(03h) - Display Enable
                0x03 => {
                    let cmd = Gp1DisplayEnableCmd::new_with_raw_value(value.raw_value());
                    self.gpu_mut().gpustat.set_display_enable(cmd.on_off());
                }
                0x04 => {
                    let dir = DmaDirection::new_with_raw_value(value.fields().as_());
                    self.gpu_mut().gpustat.set_dma_direction(dir);
                    self.gpu_mut().compute_dma_request();
                }
                0x05 => {
                    // TODO
                    tracing::debug!("set framebuffer coords");
                }
                0x06 => {
                    // TODO
                    tracing::debug!("set h framebuffer range");
                }
                0x07 => {
                    // TODO
                    tracing::debug!("set v framebuffer range");
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
                    gpustat.set_reverse_flag(cmd.screen_hflip());
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

    fn create_draw_call(&self, kind: DrawCallKind) -> DrawCall {
        DrawCall {
            gpustat: self.gpu().gpustat,
            inner:   kind,
        }
    }

    fn issue_draw_call(&mut self, kind: DrawCallKind) {
        let draw_call = self.create_draw_call(kind);
        self.gpu_mut().draw_call_queue.push(draw_call);
    }

    fn flush_draw_calls(&mut self) {
        if self.gpu().draw_call_queue.is_empty() {
            return;
        }

        tracing::info!("flushing {} draw calls", self.gpu().draw_call_queue.len());
        let queue = std::mem::take(&mut self.gpu_mut().draw_call_queue);
        let vram = self.gpu().vram.clone();
        self.gpu()
            .conn
            .draw_call_chan
            .0
            .as_sync()
            .send(queue)
            .unwrap();
        self.gpu().conn.vram_in_chan.0.as_sync().send(vram).unwrap();
        self.gpu_mut().waiting_on_render = true;
    }

    fn gpu_reconnect(&mut self, other: &impl Gpu) {
        self.gpu_mut().conn = other.gpu().conn.clone();
    }

    fn run_gpu_commands(&mut self) {
        self.gp0_cmd_queue_flush();
        self.gp1_cmd_queue_flush();
        self.gpu_mut().compute_dma_request();
        self.poll_draw_result();
    }

    fn poll_draw_result(&mut self) {
        if let Ok(Some(vram)) = self.gpu().conn.vram_out_chan.1.try_recv() {
            tracing::info!("received gpu result (vram)");
            self.gpu_mut().vram = vram;
            self.gpu_mut().waiting_on_render = false;
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

    fn vram_flush_render(&mut self) {
        if self.waiting_on_render {
            let vram = self
                .conn
                .vram_out_chan
                .1
                .as_sync()
                .recv()
                .expect("channel dropped: failed to receive queued render");
            self.vram = vram;
            self.waiting_on_render = false;
        }
    }

    fn vram_write(&mut self, coord: VramCoord, value: u16) {
        self.vram_flush_render();
        let coord = coord.wrap();
        let addr = coord.x as usize + coord.y as usize * kb(1);
        self.vram[addr] = value;
    }

    fn vram_read_direct(&mut self, coord: VramCoord) -> u16 {
        self.vram_flush_render();
        let coord = coord.wrap();
        let addr = coord.x as usize + coord.y as usize * kb(1);
        self.vram[addr]
    }

    /// returns value through `self.gp0read`
    fn vram_read(&mut self, coord: VramCoord, idx: usize) {
        let value = self.vram_read_direct(coord);
        self.gp0read[idx] = value;
    }

    pub fn compute_dma_request(&mut self) {
        let direction = self.gpustat.dma_direction();
        match direction {
            DmaDirection::Off => {
                self.gpustat.set_dma_request(false);
            }
            DmaDirection::Unknown => {
                let fifo_available = !self.gp0cmd_queue.is_full();
                self.gpustat.set_dma_request(fifo_available);
            }
            DmaDirection::CpuToGp0 => {
                self.gpustat
                    .set_dma_request(self.gpustat.ready_recv_dma_block());
            }
            DmaDirection::CpuReadToCpu => {
                self.gpustat.set_dma_request(self.gpustat.ready_send_vram());
            }
        }
    }

    pub fn flip_even_odd(&mut self, even_odd: Option<DrawEvenOdd>) {
        if self.gpustat.v_interlace() {
            let even_odd = even_odd.unwrap_or_else(|| self.gpustat.even_odd_in_vblank());
            self.gpustat.set_even_odd_in_vblank(!even_odd);
        }
    }
}

pub fn halfwords(word: u32) -> [u16; 2] {
    [word as u16, (word >> 16) as u16]
}

#[derive(Debug, Clone)]
pub enum Gp0 {
    WaitingForCmd,

    // copy commands
    CpRectCpuToVram(Gp0CpRect),
    CpRectVramToCpu(Gp0CpRect),
    CpRectVramToVram(Gp0VramCpRect),

    DrawPolygonDecode(DrawPolygonDecoder),
    DrawLineDecode(DrawLineDecoder),
    DrawRectDecode(DrawRectDecoder),
}

#[derive(Debug, Copy, PartialEq, PartialOrd, Ord, Eq)]
#[derive_const(Clone, d::Add, d::AddAssign, Default)]
#[repr(C)]
pub struct VramCoord {
    pub x: u16,
    pub y: u16,
}

impl VramCoord {
    pub fn new(xpos: u16, ypos: u16) -> Self {
        Self { x: xpos, y: ypos }
    }
    pub fn wrap(mut self) -> Self {
        if self.x >= 1024 {
            self.x = 0;
        }
        if self.y >= 512 {
            self.y = 0;
        }
        self
    }

    pub fn copy_cmd_pos_mask(mut self) -> Self {
        self.x &= 0x3ff;
        self.y &= 0x1ff;
        self
    }

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
}

impl const From<u32> for VramCoord {
    fn from(value: u32) -> Self {
        unsafe { transmute(value) }
    }
}

#[derive(Debug, Clone, Copy, d::Add, d::AddAssign, PartialEq, PartialOrd, Ord, Eq, Default)]
#[repr(C)]
pub struct IVramCoord {
    x: i16,
    y: i16,
}

impl IVramCoord {
    pub fn new(xpos: i16, ypos: i16) -> Self {
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
        self.curr = self.curr.wrap();
        let curr = self.curr;

        self.curr.x += 1;
        if self.curr.x == self.border.x {
            self.curr.x = self.start.x;
            self.curr.y += 1;
        }

        Some(curr)
    }

    fn done(&self) -> bool {
        self.curr.y >= self.border.y
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

#[derive(Debug, Clone)]
pub enum Gp0VramCpRect {
    RecvSrc,
    RecvDest { src: VramCoord },
    RecvSize { src: VramCoord, dest: VramCoord },
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
    texpage_y_base:       u1,
    #[bits(5..=6, rw)]
    semi_transparency:    u2,
    #[bits(7..=8, rw)]
    texpage_colors:       TextureColorMode,
    #[bit(9, rw)]
    dither:               bool,
    #[bit(10, rw)]
    draw_to_display:      bool,
    #[bit(11, rw)]
    set_mask:             bool,
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
        self.set_ready_send_vram(false);
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

impl std::ops::Not for DrawEvenOdd {
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
pub struct Gp1DisplayEnableCmd {
    #[bit(0, rw)]
    on_off: bool,
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
#[bitfield(u32)]
#[derive(Debug, Default)]
pub struct Gp0TexWindowCmd {
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
    screen_hflip:        ReverseFlag,
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

impl GpuState {
    pub fn video_cycles_per_scanline(&self) -> u64 {
        match self.gpustat.video_mode() {
            VideoMode::Ntsc => 3413,
            VideoMode::Pal => 3406,
        }
    }

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

    pub fn cycles_hblank_pending(&self) -> u64 {
        if self.dp.video_cycle_in_scanline > self.dp.h_end() {
            Display::NTSC_TOTAL_VCYCLES_PER_LINE - self.dp.video_cycle_in_scanline + self.dp.h_end()
        } else {
            self.dp.h_end() - self.dp.video_cycle_in_scanline
        }
    }

    pub fn pending_event(&self) -> (VideoEventKind, u64) {
        let hblank = self.cycles_hblank_pending();
        let vblank = self.cycles_vblank_pending();

        // if hblank < vblank {
        //     (VideoEventKind::Hblank, hblank)
        // } else {
        //     (VideoEventKind::Vblank, vblank)
        // }
        (VideoEventKind::Vblank, vblank)
    }
}

#[derive(derive_more::Debug, Clone, Default)]
pub struct Display {
    /// video event queue
    pub video_cycle:             u64,
    pub video_cycle_in_scanline: u64,

    display_range_start: U64Vec2,
    display_range_end:   U64Vec2,

    current_scanline: u64,

    /// used for storing fractional part when converting
    /// from cpu to video cycles
    fract_01: u64,
}

/// The functionality in this trait largely deals with video cycles (or video clock units).
/// These cycles are *not* relative to the resolution, the way dot clocks are, meaning they
/// are not absolute dot positions, but relative timings tied to HSYNC.
///
/// see [https://psx-spx.consoledev.net/graphicsprocessingunitgpu/#gp106h-horizontal-display-range-on-screen]
pub trait VideoEvents: Gpu + VBlank {
    fn cpu_cycles_to_video_cycles(&mut self, cycles: u64) -> u64 {
        // this might be based on the actual console hardware not on the
        // video mode you set in the gpu
        let factor = match self.gpu().gpustat.video_mode() {
            VideoMode::Ntsc => 715909,
            VideoMode::Pal => 709379,
        };
        let cycles = cycles * factor;
        let cycles = cycles + self.gpu().dp.fract_01;
        self.gpu_mut().dp.fract_01 = cycles % 451584;

        cycles / 451584
    }

    fn video_cycles_to_cpu_cycles_approx(&self, cycles: u64) -> u64 {
        let factor = match self.gpu().gpustat.video_mode() {
            VideoMode::Ntsc => 715909,
            VideoMode::Pal => 709379,
        };
        cycles * 451584 / factor
    }

    fn run_video_io(&mut self, by_cpu_cycles: u64) {
        let cycles = self.cpu_cycles_to_video_cycles(by_cpu_cycles);
        self.gpu_mut().dp.update_ranges();
        self.run_video_events(cycles);
    }

    fn run_video_events(&mut self, advance_by: u64) {
        let cycles_per_scanline = self.gpu().video_cycles_per_scanline();
        self.gpu_mut().dp.video_cycle_in_scanline += advance_by;
        if self.gpu_mut().dp.video_cycle_in_scanline < cycles_per_scanline {
            // DONE: Timer 1 update
            self.timers_mut().trigger_hblank();
            return;
        }
        let scanlines_to_run = self.gpu().dp.video_cycle_in_scanline / cycles_per_scanline;
        self.gpu_mut().dp.video_cycle_in_scanline =
            self.gpu().dp.video_cycle_in_scanline % cycles_per_scanline;

        for _ in 0..scanlines_to_run {
            let dp = &self.gpu().dp;
            let new_vblank = !dp.in_vblank() && dp.in_vblank_with(dp.current_scanline + 1);

            // DONE: timer 1 update
            self.timers_mut().trigger_hblank();

            if new_vblank {
                self.run_vblank();
            }

            self.gpu_mut().dp.current_scanline += 1;

            let dp = &mut self.gpu_mut().dp;
            if dp.current_scanline >= Display::NTSC_TOTAL_LINES {
                dp.current_scanline = 0;
                self.gpu_mut().flip_even_odd(None);
            }
        }
    }
}

impl VideoEvents for Emu {}

impl Display {
    const NTSC_TOTAL_VCYCLES_PER_LINE: u64 = 3413;
    const NTSC_TOTAL_LINES: u64 = 263;
    const NTSC_ACTIVE_H_START: u64 = 488;
    const NTSC_ACTIVE_H_END: u64 = 3288;
    const NTSC_ACTIVE_V_START: u64 = 16;
    const NTSC_ACTIVE_V_END: u64 = 256;

    fn update_ranges(&mut self) {
        self.display_range_start = u64vec2(Self::NTSC_ACTIVE_H_START, Self::NTSC_ACTIVE_V_START);
        self.display_range_end = u64vec2(Self::NTSC_ACTIVE_H_END, Self::NTSC_ACTIVE_V_END);
    }

    const fn h_start(&self) -> u64 {
        self.display_range_start.x
    }

    const fn h_end(&self) -> u64 {
        self.display_range_end.x
    }

    const fn v_start(&self) -> u64 {
        self.display_range_start.y
    }

    const fn v_end(&self) -> u64 {
        self.display_range_end.y
    }

    fn in_hblank(&self) -> bool {
        !(self.h_start()..self.h_end()).contains(&self.video_cycle_in_scanline)
    }

    fn in_vblank(&self) -> bool {
        !(self.v_start()..self.v_end()).contains(&self.current_scanline)
    }

    fn in_vblank_with(&self, line: u64) -> bool {
        !(self.v_start()..self.v_end()).contains(&line)
    }
}

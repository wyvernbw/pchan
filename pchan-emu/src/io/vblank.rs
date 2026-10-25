use core::alloc::Allocator;

use crate::Emu;
use crate::gpu::{DrawEvenOdd, VBLANK_COUNT};
use crate::io::irq::Irq;

pub const CPU_FREQ: u32 = 33_868_800;
pub const NTSC_CYCLES: u32 = CPU_FREQ / 60;

impl<A: Allocator + Copy> Emu<A> {
    pub fn run_vblank(&mut self) {
        let even_odd = self.gpu.gpustat.even_odd_in_vblank();
        self.gpu
            .gpustat
            .set_even_odd_in_vblank(DrawEvenOdd::EvenOrVBlank);
        match self.gpu.vram {
            crate::gpu::Vram::Mapped(_) => {}
            crate::gpu::Vram::Lent => {
                if let Some(crate::gpu::VramRes::Ok(vram)) =
                    self.gpu.conn.vram_out_chan.1.as_sync().try_recv().unwrap()
                {
                    self.gpu.submit_vram(vram);
                }
            }
        }
        self.gpu_flush_draw_calls();
        self.irq_trigger(Irq::Irq0Vblank);
        self.gpu.vblank_signal = true;

        self.gpu.flip_even_odd(Some(even_odd));

        VBLANK_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn consume_vblank_signal(&mut self) -> bool {
        let signal = self.gpu.vblank_signal;
        if signal {
            self.tracy.frame_mark();
        }
        self.gpu.vblank_signal = false;
        signal
    }
}

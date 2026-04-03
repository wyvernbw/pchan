use std::time::Instant;

use crate::{
    Emu,
    gpu::{DrawEvenOdd, Gpu, VBLANK_COUNT},
    io::{Interrupts, irq::Irq},
};

pub const CPU_FREQ: u32 = 33_868_800;
pub const NTSC_CYCLES: u32 = CPU_FREQ / 60;

pub trait VBlank: Interrupts + Gpu {
    #[deprecated]
    fn run_poll_vblank(&mut self) {
        let even_odd = self.gpu().gpustat.even_odd_in_vblank();
        let mut cycles = &mut self.cpu_mut().vblank_timer;
        while *cycles >= NTSC_CYCLES {
            *cycles -= NTSC_CYCLES;

            // gpustat.31 is 0x0 *during* vblank
            self.gpu_mut()
                .gpustat
                .set_even_odd_in_vblank(DrawEvenOdd::EvenOrVBlank);
            self.flush_draw_calls();
            self.trigger_irq(Irq::Irq0Vblank);

            self.gpu_mut().flip_even_odd(Some(even_odd));

            cycles = &mut self.cpu_mut().vblank_timer;
        }
    }

    fn run_vblank(&mut self) {
        let even_odd = self.gpu().gpustat.even_odd_in_vblank();
        self.gpu_mut()
            .gpustat
            .set_even_odd_in_vblank(DrawEvenOdd::EvenOrVBlank);
        self.flush_draw_calls();
        self.trigger_irq(Irq::Irq0Vblank);
        self.gpu_mut().vblank_signal = true;

        self.gpu_mut().flip_even_odd(Some(even_odd));

        VBLANK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn consume_vblank_signal(&mut self) -> bool {
        let signal = self.gpu().vblank_signal;
        self.gpu_mut().vblank_signal = false;
        signal
    }
}

impl VBlank for Emu {}

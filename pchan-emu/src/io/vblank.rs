use crate::Emu;
use crate::gpu::{DrawEvenOdd, VBLANK_COUNT};
use crate::io::irq::Irq;

pub const CPU_FREQ: u32 = 33_868_800;
pub const NTSC_CYCLES: u32 = CPU_FREQ / 60;

impl Emu {
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
            self.gpu_mut().flush_draw_calls();
            self.irq_trigger(Irq::Irq0Vblank);

            self.gpu_mut().flip_even_odd(Some(even_odd));

            cycles = &mut self.cpu_mut().vblank_timer;
        }
    }

    pub fn run_vblank(&mut self) {
        let even_odd = self.gpu().gpustat.even_odd_in_vblank();
        self.gpu_mut()
            .gpustat
            .set_even_odd_in_vblank(DrawEvenOdd::EvenOrVBlank);
        self.gpu_mut().flush_draw_calls();
        self.irq_trigger(Irq::Irq0Vblank);
        self.gpu_mut().vblank_signal = true;

        self.gpu_mut().flip_even_odd(Some(even_odd));

        VBLANK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn consume_vblank_signal(&mut self) -> bool {
        let signal = self.gpu().vblank_signal;
        if signal {
            self.tracy.frame_mark();
        }
        self.gpu_mut().vblank_signal = false;
        signal
    }
}

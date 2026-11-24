use crate::Emu;
use crate::cpu::irq::Irq;
use crate::gpu::{DrawEvenOdd, VBLANK_COUNT};

impl Emu {
    pub async fn run_vblank(&mut self) {
        let even_odd = self.gpu.gpustat.even_odd_in_vblank();
        self.gpu
            .gpustat
            .set_even_odd_in_vblank(DrawEvenOdd::EvenOrVBlank);
        self.gpu.task_gpu_flush_draw_calls().await;
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

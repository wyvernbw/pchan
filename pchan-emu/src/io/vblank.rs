use crate::{Emu, io::Interrupts};

use super::Irq;

const CPU_FREQ: u64 = 33_868_800;
const NTSC_CYCLES: u64 = CPU_FREQ / 60;

pub trait VBlank: Interrupts {
    fn run_vblank(&mut self) {
        let cycles = &mut self.cpu_mut().cycles;
        if *cycles >= NTSC_CYCLES {
            *cycles = 0;
            self.trigger_irq(Irq::Irq0Vblank);
        }
    }
}

impl VBlank for Emu {}

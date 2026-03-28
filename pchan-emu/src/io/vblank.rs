use crate::{
    Emu,
    io::{Interrupts, irq::Irq},
};

pub const CPU_FREQ: u32 = 33_868_800;
pub const NTSC_CYCLES: u32 = CPU_FREQ / 60;

pub trait VBlank: Interrupts {
    fn run_vblank(&mut self) {
        let cycles = &mut self.cpu_mut().vblank_timer;
        if *cycles >= NTSC_CYCLES {
            *cycles -= NTSC_CYCLES;
            self.trigger_irq(Irq::Irq0Vblank);
        }
    }
}

impl VBlank for Emu {}

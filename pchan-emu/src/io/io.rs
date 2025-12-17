use crate::Emu;

pub mod timers;
pub mod tty;

pub struct IO;

impl Emu {
    pub fn run_io(&mut self) {
        IO::advance_timers(&self.cpu, &self.mem);
        self.io_kernel_functions();
    }

    pub fn io_kernel_functions(&mut self) {
        let pc = self.cpu.pc & 0x1fff_ffff;
        match (pc, self.cpu.gpr[9]) {
            (0xa0, 0x3c) | (0xb0, 0x3d) => {
                self.tty.putchar(self.cpu.gpr[4] as u8 as _);
            }
            _ => {}
        }
    }
}

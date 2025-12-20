use pchan_utils::hex;

use crate::io::timers::Timers;
use crate::memory::Extend;
use crate::{
    Bus, Emu,
    io::cdrom::CDRom,
    memory::{ext, fastmem::Fastmem},
};

pub mod cdrom;
pub mod timers;
pub mod tty;

impl Emu {
    pub fn run_io(&mut self) {
        self.run_timer_pipeline();
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

pub trait IO: Bus {
    fn read<T: Copy>(&self, address: u32) -> T;
    fn write<T: Copy>(&mut self, address: u32, value: T);
    fn write_many<T: Copy>(&mut self, mut address: u32, values: &[T]) {
        for value in values.iter().copied() {
            self.write(address, value);
            address += 0x4;
        }
    }
    fn read_ext<T: Copy + Extend<E>, E>(&self, address: u32) -> T::Out {
        let value = self.read::<T>(address);
        Extend::<E>::ext(value)
    }
    fn write_ext<T, E>(&mut self, address: u32, value: T)
    where
        T::Out: Copy,
        T: Extend<E>,
    {
        let value = Extend::<E>::ext(value);
        self.write(address, value);
    }
}

#[derive(thiserror::Error, Debug, Clone, Copy)]
#[error("unhandled io at address {}", hex(self.0))]
pub struct UnhandledIO(pub u32);

impl IO for Emu {
    fn read<T: Copy>(&self, address: u32) -> T {
        let result = Fastmem::read::<T>(self, address)
            .or_else(|_| Timers::read_timers(self, address))
            .or_else(|_| CDRom::read::<T>(self, address));
        match result {
            Ok(value) => value,
            Err(err) => panic!("{}", err),
        }
    }

    fn write<T: Copy>(&mut self, address: u32, value: T) {
        let result = Fastmem::write::<T>(self, address, value)
            .or_else(|_| Timers::write_timers(self, address, value))
            .or_else(|_| CDRom::write::<T>(self, address, value));
        if let Err(err) = result {
            panic!("{}", err);
        }
    }
}

use pchan_utils::hex;
use tracing::instrument;

use crate::io::timers::Timers;
use crate::memory::{Extend, GUEST_MEM_MAP, MEM_MAP};
use crate::{Bus, Emu, io::cdrom::CDRom, memory::fastmem::Fastmem};

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
    fn try_read<T: Copy>(&self, address: u32) -> IOResult<T>;
    fn try_write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()>;
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

pub type IOResult<T> = Result<T, UnhandledIO>;

trait GenericIOFallback: Bus {
    #[instrument(skip(self))]
    fn read<T: Copy>(&self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801000..0x1fa00000 => {
                tracing::trace!("fallback to generic io read");
                Ok(self
                    .mem()
                    .read_region(MEM_MAP.io, GUEST_MEM_MAP.io, address))
            }
            _ => Err(UnhandledIO(address)),
        }
    }
    fn write<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801000..0x1fa00000 => {
                tracing::trace!("fallback to generic io write");
                self.mem_mut()
                    .write_region(MEM_MAP.io, GUEST_MEM_MAP.io, address, value);
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }
}

impl GenericIOFallback for Emu {}

trait CacheControl: Bus {
    #[instrument(skip(self))]
    fn read<T: Copy>(&self, address: u32) -> IOResult<T> {
        match address {
            0xfffe0130 => Ok(self.mem().read_region(
                MEM_MAP.cache_control,
                GUEST_MEM_MAP.cache_control,
                address,
            )),
            _ => Err(UnhandledIO(address)),
        }
    }
    fn write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()> {
        match address {
            0xfffe0130 => {
                self.mem_mut().write_region(
                    MEM_MAP.cache_control,
                    GUEST_MEM_MAP.cache_control,
                    address,
                    value,
                );
                Ok(())
            }
            _ => Err(UnhandledIO(address)),
        }
    }
}

impl CacheControl for Emu {}

#[derive(thiserror::Error, Debug, Clone, Copy)]
#[error("unhandled io at address {}", hex(self.0))]
pub struct UnhandledIO(pub u32);

impl IO for Emu {
    fn read<T: Copy>(&self, address: u32) -> T {
        match self.try_read(address) {
            Ok(value) => value,
            Err(err) => self.panic(&format!("{}", err)),
        }
    }

    fn write<T: Copy>(&mut self, address: u32, value: T) {
        if let Err(err) = self.try_write(address, value) {
            self.panic(&format!("{}", err));
        }
    }

    fn try_read<T: Copy>(&self, address: u32) -> IOResult<T> {
        Fastmem::read::<T>(self, address)
            .or_else(|_| Timers::read_timers(self, address))
            .or_else(|_| CDRom::read::<T>(self, address))
            .or_else(|_| GenericIOFallback::read::<T>(self, address))
            .or_else(|_| CacheControl::read::<T>(self, address))
    }

    fn try_write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()> {
        Fastmem::write::<T>(self, address, value)
            .or_else(|_| Timers::write_timers(self, address, value))
            .or_else(|_| CDRom::write::<T>(self, address, value))
            .or_else(|_| GenericIOFallback::write::<T>(self, address, value))
            .or_else(|_| CacheControl::write::<T>(self, address, value))
    }
}

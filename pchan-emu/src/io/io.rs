use bitbybit::{bitenum, bitfield};
use pchan_utils::hex;
use tracing::instrument;

use crate::bootloader::Bootloader;
use crate::cpu::exceptions::Exceptions;
use crate::gpu::Gpu;
use crate::io::timers::Timers;
use crate::io::vblank::VBlank;
use crate::memory::{Extend, GUEST_MEM_MAP, MEM_MAP, ScratchpadMem};
use crate::{Bus, Emu, io::cdrom::CDRom, memory::fastmem::Fastmem};
use derive_more as d;

pub mod cdrom;
pub mod timers;
pub mod tty;
pub mod vblank;

impl Emu {
    pub fn run_io(&mut self) {
        self.cpu_mut().cycles = self.cpu().cycles.wrapping_add(self.cpu().d_clock as u64);
        self.run_timer_pipeline();
        self.run_io_kernel_functions();
        self.run_vblank();
        #[cfg(feature = "amidog-tests")]
        {
            use crate::bootloader::AMIDOG_TESTS;
            self.run_sideloading(AMIDOG_TESTS).unwrap();
        }
    }

    pub fn run_io_kernel_functions(&mut self) {
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
    fn try_read<T: Copy>(&mut self, address: u32) -> IOResult<T>;
    fn try_write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()>;
    fn read<T: Copy>(&mut self, address: u32) -> T;
    fn write<T: Copy>(&mut self, address: u32, value: T);
    fn try_read_pure<T: Copy>(&self, address: u32) -> IOResult<T>;
    fn read_pure<T: Copy>(&self, address: u32) -> T;
    fn write_many<T: Copy>(&mut self, mut address: u32, values: &[T]) {
        for value in values.iter().copied() {
            self.write(address, value);
            address += 0x4;
        }
    }
    fn read_ext<T: Copy + Extend<E>, E>(&mut self, address: u32) -> T::Out {
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
    #[instrument(skip(self), "r")]
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
    #[instrument(skip(self, value), "w")]
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
    fn read<T: Copy>(&mut self, address: u32) -> T {
        match self.try_read(address) {
            Ok(value) => value,
            Err(err) => self.panic(&format!("{}", err)),
        }
    }

    fn read_pure<T: Copy>(&self, address: u32) -> T {
        match self.try_read_pure(address) {
            Ok(value) => value,
            Err(err) => self.panic(&format!("{}", err)),
        }
    }

    fn write<T: Copy>(&mut self, address: u32, value: T) {
        if let Err(err) = self.try_write(address, value) {
            self.panic(&format!("{}", err));
        }
    }

    fn try_read<T: Copy>(&mut self, address: u32) -> IOResult<T> {
        Fastmem::read::<T>(self, address)
            .or_else(|_| ScratchpadMem::read(self, address))
            .or_else(|_| Timers::read_timers(self, address))
            .or_else(|_| Gpu::read(self, address))
            .or_else(|_| CDRom::read::<T>(self, address))
            .or_else(|_| GenericIOFallback::read::<T>(self, address))
            .or_else(|_| CacheControl::read::<T>(self, address))
    }

    fn try_read_pure<T: Copy>(&self, address: u32) -> IOResult<T> {
        Fastmem::read::<T>(self, address)
            .or_else(|_| ScratchpadMem::read(self, address))
            .or_else(|_| Timers::read_timers(self, address))
            .or_else(|_| CDRom::read::<T>(self, address))
            .or_else(|_| GenericIOFallback::read::<T>(self, address))
            .or_else(|_| CacheControl::read::<T>(self, address))
    }

    fn try_write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()> {
        Fastmem::write::<T>(self, address, value)
            .or_else(|_| ScratchpadMem::write(self, address, value))
            .or_else(|_| Timers::write_timers(self, address, value))
            .or_else(|_| Gpu::write(self, address, value))
            .or_else(|_| CDRom::write::<T>(self, address, value))
            .or_else(|_| GenericIOFallback::write::<T>(self, address, value))
            .or_else(|_| CacheControl::write::<T>(self, address, value))
    }
}

pub trait CastIOInto: Copy {
    fn io_into_u32(&self) -> u32 {
        assert!(
            size_of::<Self>() <= 4,
            "invalid cast of IO channel value to T. T has size {} >= 4",
            size_of::<Self>()
        );
        unsafe { std::mem::transmute_copy::<Self, u32>(self) }
    }
}

impl<T: Copy> CastIOInto for T {}

pub trait CastIOFrom: Copy {
    fn io_from_u32<T>(self) -> T {
        assert!(
            size_of::<T>() <= 4,
            "invalid cast of IO channel value to T. T has size {} >= 4",
            size_of::<T>()
        );
        unsafe { std::mem::transmute_copy::<Self, T>(&self) }
    }
}

impl<T: Copy> CastIOFrom for T {}

#[cfg(test)]
#[test]
fn test_io_from_u32() {
    assert_eq!(0xdeadbeefu32.io_from_u32::<u32>(), 0xdeadbeef);
    assert_eq!(0xdeadbeefu32.io_from_u32::<u16>(), 0xbeef);
    assert_eq!(0xdeadbeefu32.io_from_u32::<u8>(), 0xef);
    assert_eq!(0xdeadbeefu32.io_from_u32::<i32>(), 0xdeadbeefu32 as i32);
    assert_eq!(0xdeadbeefu32.io_from_u32::<i16>(), 0xbeefu32 as i16);
    assert_eq!(0xdeadbeefu32.io_from_u32::<i8>(), 0xefu32 as i8);
}

#[bitfield(u32)]
#[derive(d::Deref)]
pub struct IrqField {
    #[bit(0)]
    irq0_vblank: bool,
    #[bit(4)]
    irq4_timer0: bool,
    #[bit(5)]
    irq5_timer1: bool,
    #[bit(6)]
    irq6_timer2: bool,
}

#[bitenum(u8)]
pub enum Irq {
    Irq0Vblank = 0x0,
    Irq4Timer0 = 0x4,
    Irq5Timer1 = 0x5,
    Irq6Timer2 = 0x6,
}

pub trait Interrupts: Bus + IO + Exceptions {
    fn trigger_irq(&mut self, irq: Irq) {
        let stat = self.read::<IrqField>(0x1f801070);
        let mask = self.read::<IrqField>(0x1f801074);
        let stat = *stat & *mask & (1 << irq as u8);
        self.write(0x1f801070, stat);
        self.handle_exception(crate::cpu::exceptions::Exception::Interrupt);
    }
}

impl Interrupts for Emu {}

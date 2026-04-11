use arbitrary_int::prelude::*;
use pchan_utils::hex;
use tracing::instrument;

use crate::bootloader::Bootloader;
use crate::cpu::exceptions::Exceptions;
use crate::gpu::{Gpu, VideoEvents};
use crate::io::dma::Dma;
use crate::io::irq::Interrupts;
use crate::io::timers::Timers;
use crate::memory::{Extend, GUEST_MEM_MAP, MEM_MAP, ScratchpadMem};
use crate::{Bus, Emu, io::cdrom::CDRom, memory::fastmem::Fastmem};

pub mod cdrom;
pub mod dma;
pub mod irq;
pub mod timers;
pub mod tty;
pub mod vblank;

impl Emu {
    #[instrument("io", skip_all, fields(pc = %hex(self.cpu.pc)))]
    pub fn run_io(&mut self) {
        self.cpu_mut().vblank_timer = self.cpu().vblank_timer.wrapping_add(self.cpu().d_clock);
        self.cpu_mut().cycles = self.cpu().cycles.wrapping_add(self.cpu().d_clock as u64);

        // gpu commands must run before dma to ensure gp0 fifo is cleared
        self.run_gpu_commands();
        self.run_video_io(self.cpu.d_clock as u64);

        let mut d_clock = self.cpu.d_clock;
        while d_clock > 0 {
            self.timers_advance_by_cpu(d_clock.min(u16::MAX as u32) as u16);
            self.run_timer_pipeline();
            d_clock = d_clock.saturating_sub(u16::MAX as u32);
        }
        self.run_io_kernel_functions();

        self.run_dma_transfers();
        self.run_irq_io();
        self.run_exceptions_io();
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

    pub fn pending_event(&self) -> u64 {
        let video_event = self.gpu().pending_event().1;
        let video_event = self.video_cycles_to_cpu_cycles_approx(video_event);
        let dma_event = self
            .dma()
            .pending_event()
            .unwrap_or(u64::MAX)
            .saturating_sub(self.cpu.cycles);
        video_event.min(dma_event)
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
    #[pchan_macros::instrument(ret, skip_all)]
    fn try_write32_unaligned_l(&mut self, address: u32, value: u32) -> IOResult<()> {
        let spill = address % size_of::<u32>() as u32;
        let aligned_address = address - spill;
        let read_value = self.try_read::<u32>(aligned_address)?;

        let spill = 4 - spill - 1;
        let spill = spill << 3; // bytes to bits
        let mask = 0xffff_ffff >> spill;
        let read_value = read_value & (!mask);
        let value = value >> spill;
        let value = value | read_value;

        #[cfg(debug_assertions)]
        {
            tracing::trace!(value = %hex(value), mask = %hex(mask));
        }

        self.try_write(aligned_address, value)
    }
    #[pchan_macros::instrument(ret, skip_all)]
    fn try_write32_unaligned_r(&mut self, address: u32, value: u32) -> IOResult<()> {
        let shift = (address & 3) << 3;
        let aligned = address & !3;
        let read_value = self.try_read::<u32>(aligned)?;

        let spill_n = 32 - shift;
        let mask = 0xffff_ffffu32.unbounded_shr(spill_n);
        let read_value = read_value & mask;
        let value = value << shift;
        tracing::trace!(value = %hex(value), mask = %hex(mask));
        let value = value | read_value;

        self.try_write(aligned, value)
    }

    #[pchan_macros::instrument(skip_all)]
    fn try_read32_unaligned_l(&mut self, address: u32, overwrite: u32) -> IOResult<u32> {
        let shift = (address & 3) << 3;
        let aligned = address & !3;
        let read_value = self.try_read::<u32>(aligned)?;

        let mask = 0xffff_ffffu32.unbounded_shr(shift + 8);
        let overwrite = overwrite & mask;
        let shift = 24 - shift;
        let value = read_value << shift;

        #[cfg(debug_assertions)]
        {
            tracing::info!(ret=%hex(value));
        }

        Ok(overwrite | value)
    }

    #[pchan_macros::instrument(skip_all)]
    fn try_read32_unaligned_r(&mut self, address: u32, overwrite: u32) -> IOResult<u32> {
        let shift = (address & 3) << 3;
        let aligned = address & !3;
        let read_value = self.try_read::<u32>(aligned)?;

        let mask = 0xffff_ffffu32.unbounded_shl(32 - shift);
        let overwrite = overwrite & mask;
        let value = read_value >> shift;

        #[cfg(debug_assertions)]
        {
            tracing::info!(ret=%hex(value));
        }

        Ok(overwrite | value)
    }

    fn write32_unaligned_l(&mut self, address: u32, value: u32);
    fn write32_unaligned_r(&mut self, address: u32, value: u32);

    fn read32_unaligned_l(&mut self, address: u32, overwrite: u32) -> u32;
    fn read32_unaligned_r(&mut self, address: u32, overwrite: u32) -> u32;
}

pub type IOResult<T> = Result<T, UnhandledIO>;

trait GenericIOFallback: Bus {
    #[cfg_attr(debug_assertions, instrument(skip(self), "r"))]
    fn read<T: Copy>(&self, address: u32) -> IOResult<T> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801000..0x1fa00000 => {
                Ok(self
                    .mem()
                    .read_region(MEM_MAP.io, GUEST_MEM_MAP.io, address))
            }
            _ => Err(UnhandledIO(address)),
        }
    }
    #[cfg_attr(debug_assertions, instrument(skip(self, value), "w"))]
    fn write<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
        let address = address & 0x1fffffff;
        match address {
            0x1f801000..0x1fa00000 => {
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
    #[cfg_attr(debug_assertions, instrument(skip(self)))]
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
            .or_else(|_| Interrupts::read(self, address))
            .or_else(|_| Gpu::read(self, address))
            .or_else(|_| Dma::read(self, address))
            .or_else(|_| Timers::read_timers(self, address))
            .or_else(|_| CDRom::read::<T>(self, address))
            .or_else(|_| GenericIOFallback::read::<T>(self, address))
            .or_else(|_| CacheControl::read::<T>(self, address))
    }

    fn try_read_pure<T: Copy>(&self, address: u32) -> IOResult<T> {
        Fastmem::read::<T>(self, address)
            .or_else(|_| ScratchpadMem::read(self, address))
            .or_else(|_| Interrupts::read(self, address))
            .or_else(|_| Gpu::read_pure(self, address))
            .or_else(|_| Dma::read(self, address))
            .or_else(|_| Timers::read_timers(self, address))
            .or_else(|_| CDRom::read::<T>(self, address))
            .or_else(|_| GenericIOFallback::read::<T>(self, address))
            .or_else(|_| CacheControl::read::<T>(self, address))
    }

    fn try_write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()> {
        Fastmem::write::<T>(self, address, value)
            .or_else(|_| ScratchpadMem::write(self, address, value))
            .or_else(|_| Timers::write_timers(self, address, value))
            .or_else(|_| Interrupts::write(self, address, value))
            .or_else(|_| Gpu::write(self, address, value))
            .or_else(|_| Dma::write(self, address, value))
            .or_else(|_| CDRom::write::<T>(self, address, value))
            .or_else(|_| GenericIOFallback::write::<T>(self, address, value))
            .or_else(|_| CacheControl::write::<T>(self, address, value))
    }

    #[pchan_macros::instrument(ret, skip_all)]
    fn write32_unaligned_l(&mut self, address: u32, value: u32) {
        if let Err(err) = self.try_write32_unaligned_l(address, value) {
            self.panic(&format!("{}", err));
        }
    }

    #[pchan_macros::instrument(ret, skip_all)]
    fn write32_unaligned_r(&mut self, address: u32, value: u32) {
        if let Err(err) = self.try_write32_unaligned_r(address, value) {
            self.panic(&format!("{}", err));
        }
    }

    fn read32_unaligned_l(&mut self, address: u32, overwrite: u32) -> u32 {
        match self.try_read32_unaligned_l(address, overwrite) {
            Err(err) => {
                self.panic(&format!("{err}"));
            }
            Ok(value) => value,
        }
    }

    fn read32_unaligned_r(&mut self, address: u32, overwrite: u32) -> u32 {
        match self.try_read32_unaligned_r(address, overwrite) {
            Err(err) => {
                self.panic(&format!("{err}"));
            }
            Ok(value) => value,
        }
    }
}

pub trait CastIOInto: Copy {
    fn io_into_u32(&self) -> u32 {
        self.io_into_u32_overwrite(0x0)
    }

    fn io_into_u32_overwrite(&self, original: u32) -> u32 {
        assert!(
            size_of::<Self>() <= 4,
            "invalid cast of IO channel value to T. T has size {} >= 4",
            size_of::<Self>()
        );
        let mut buf = original.to_ne_bytes();
        unsafe {
            std::ptr::copy_nonoverlapping(
                self as *const Self as *const u8,
                buf.as_mut_ptr(),
                size_of::<Self>(),
            );
        }
        u32::from_ne_bytes(buf)
    }

    fn io_as(&self) -> UInt<u32, { size_of::<Self>() }> {
        self.io_into_u32().into()
    }
}

impl<T: Copy> CastIOInto for T {}

pub trait CastIOFrom: Copy {
    fn io_from_u32<T>(self) -> T {
        let typename = std::any::type_name::<T>();
        assert!(
            size_of::<T>() <= 4,
            "invalid cast of IO channel value to {typename}. {typename} has size {} >= 4",
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

#[cfg(test)]
#[test]
fn test_io_into_u32_overwrite() {
    let original = 0xff00a0u32;
    let new = 0x0000ffu8;
    let result = new.io_into_u32_overwrite(original);
    assert_eq!(result, 0xff00ff);
}

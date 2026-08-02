use arbitrary_int::prelude::*;
use pchan_utils::hex;

use crate::Emu;
use crate::memory::{Extend, GUEST_MEM_MAP, MEM_MAP};

#[path = "./cdrom/cdrom.rs"]
pub mod cdrom;
pub mod dma;
pub mod evque;
pub mod irq;
#[path = "./sio/sio.rs"]
pub mod sio;
pub mod timers;
pub mod tty;
pub mod vblank;

#[macro_export]
macro_rules! trace_todo {
    ($args: tt) => {{
        tracing::warn!($args);
        Ok(())
    }};
    ($value: expr, $args: tt) => {{
        tracing::warn!($args);
        Ok($value.io_from_u32())
    }};
}

impl Emu {
    #[pchan_macros::instrument(level = "trace", "io", skip_all, fields(pc = %hex(self.cpu.pc)))]
    pub fn run_io(&mut self) {
        #[cfg(feature = "amidog-tests")]
        {
            use crate::bootloader::AMIDOG_TESTS;
            self.sideload_exe(AMIDOG_TESTS).unwrap();
        }

        let d_clock = self.cpu().d_clock as u64;
        self.cpu_mut().vblank_timer = self.cpu().vblank_timer.wrapping_add(d_clock as u32);
        self.cpu_mut().cycles = self.cpu().cycles.wrapping_add(d_clock);
        self.evque_advance(d_clock);

        self.run_video_io(d_clock);
        self.run_sio_bdtimers(d_clock);

        let mut d_clock = d_clock;
        while d_clock > 0 {
            self.timers_advance_by_cpu(d_clock.min(u16::MAX as u64) as u16);
            self.run_timer_pipeline();
            d_clock = d_clock.saturating_sub(u16::MAX as u64);
        }

        self.run_io_kernel_functions();
        self.run_exceptions_io();
        _ = self.run_sideloading();
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

impl Emu {
    pub fn write_many<T: Copy>(&mut self, mut address: u32, values: &[T]) {
        for value in values.iter().copied() {
            self.write(address, value);
            address += 0x4;
        }
    }
    pub fn read_ext<T: Copy + Extend<E>, E>(&mut self, address: u32) -> T::Out {
        let value = self.read::<T>(address);
        Extend::<E>::ext(value)
    }
    pub fn write_ext<T, E>(&mut self, address: u32, value: T)
    where
        T::Out: Copy,
        T: Extend<E>,
    {
        let value = Extend::<E>::ext(value);
        self.write(address, value);
    }
    #[pchan_macros::instrument(level = "trace", skip_all)]
    pub fn try_write32_unaligned_l(&mut self, address: u32, value: u32) -> IOResult<()> {
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
    #[pchan_macros::instrument(level = "trace", skip_all)]
    pub fn try_write32_unaligned_r(&mut self, address: u32, value: u32) -> IOResult<()> {
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

    #[pchan_macros::instrument(level = "trace", skip_all)]
    pub fn try_read32_unaligned_l(&mut self, address: u32, overwrite: u32) -> IOResult<u32> {
        let shift = (address & 3) << 3;
        let aligned = address & !3;
        let read_value = self.try_read::<u32>(aligned)?;

        let mask = 0xffff_ffffu32.unbounded_shr(shift + 8);
        let overwrite = overwrite & mask;
        let shift = 24 - shift;
        let value = read_value << shift;

        Ok(overwrite | value)
    }

    #[pchan_macros::instrument(level = "trace", skip_all)]
    pub fn try_read32_unaligned_r(&mut self, address: u32, overwrite: u32) -> IOResult<u32> {
        let shift = (address & 3) << 3;
        let aligned = address & !3;
        let read_value = self.try_read::<u32>(aligned)?;

        let mask = 0xffff_ffffu32.unbounded_shl(32 - shift);
        let overwrite = overwrite & mask;
        let value = read_value >> shift;

        Ok(overwrite | value)
    }
}

pub type IOResult<T> = Result<T, UnhandledIO>;

impl Emu {
    #[pchan_macros::instrument(
        level = "trace",
        skip_all,
        fields(address = %hex(address))
        "generic:r"
    )]
    fn generic_read<T: Copy>(&self, address: u32) -> IOResult<T> {
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
    #[pchan_macros::instrument(
        level = "trace",
        skip_all,
        fields(address = %hex(address), value = %hex(value.io_into_u32()))
        "generic:w"
    )]
    fn generic_write<T: Copy>(&mut self, address: u32, value: T) -> Result<(), UnhandledIO> {
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

impl Emu {
    #[pchan_macros::instrument(
        level = "trace",
        skip_all,
        fields(address = %hex(address))
        "cache_ctrl:r"
    )]
    fn cache_ctrl_read<T: Copy>(&self, address: u32) -> IOResult<T> {
        match address {
            0xfffe0130 => Ok(self.mem().read_region(
                MEM_MAP.cache_control,
                GUEST_MEM_MAP.cache_control,
                address,
            )),
            _ => Err(UnhandledIO(address)),
        }
    }
    #[pchan_macros::instrument(
        level = "trace",
        skip_all,
        fields(address = %hex(address), value = %hex(value.io_into_u32()))
        "cache_ctrl:w"
    )]
    fn cache_ctrl_write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()> {
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

#[derive(thiserror::Error, derive_more::Debug, Clone, Copy)]
#[error("unhandled io at address {}", hex(self.0))]
pub struct UnhandledIO(#[debug("{}", hex(self.0))] pub u32);

impl Emu {
    pub fn read<T: Copy>(&mut self, address: u32) -> T {
        match self.try_read(address) {
            Ok(value) => value,
            Err(err) => self.panic(&format!("{}", err)),
        }
    }

    pub fn read_pure<T: Copy>(&self, address: u32) -> T {
        match self.try_read_pure(address) {
            Ok(value) => value,
            Err(err) => self.panic(&format!("{}", err)),
        }
    }

    pub fn write<T: Copy>(&mut self, address: u32, value: T) {
        if let Err(err) = self.try_write(address, value) {
            self.panic(&format!("{}", err));
        }
    }

    // #[pchan_macros::instrument(skip_all, fields(pc = %hex(self.cpu.pc)))]
    pub fn try_read<T: Copy>(&mut self, address: u32) -> IOResult<T> {
        #[cfg(feature = "debugger-ext")]
        {
            use crate::debug::BreakpointKind;

            self.dbg.break_on(address, BreakpointKind::READ);
        }

        let inspect_read = |msg: &str, res: IOResult<T>| {
            #[cfg(feature = "trace")]
            {
                res.inspect(|_| tracing::info!("{msg}: {}", hex(address)))
            }
            #[cfg(not(feature = "trace"))]
            {
                res
            }
        };

        self.fastmem_read::<T>(address)
            .or_else(|_| inspect_read("r(scratch)", self.scratch_read(address)))
            .or_else(|_| inspect_read("r(irq)", self.irq_read(address)))
            .or_else(|_| inspect_read("r(gpu)", self.gpu_read(address)))
            .or_else(|_| inspect_read("r(spu)", self.spu_read(address)))
            .or_else(|_| inspect_read("r(dma)", self.dma_read(address)))
            .or_else(|_| inspect_read("r(timers)", self.timers_read(address)))
            .or_else(|_| inspect_read("r(sio)", self.sio_read::<T>(address)))
            .or_else(|_| inspect_read("r(cdrom)", self.cdrom_read::<T>(address)))
            .or_else(|_| inspect_read("r(cache_ctrl)", self.cache_ctrl_read::<T>(address)))
            .or_else(|_| inspect_read("r(unknown)", self.generic_read::<T>(address)))
    }

    pub fn try_read_pure<T: Copy>(&self, address: u32) -> IOResult<T> {
        self.fastmem_read::<T>(address)
            .or_else(|_| self.scratch_read(address))
            .or_else(|_| self.irq_read(address))
            .or_else(|_| self.gpu_read_pure(address))
            .or_else(|_| self.sio_read_pure::<T>(address))
            // TODO: spu_read_pure
            .or_else(|_| self.dma_read(address))
            .or_else(|_| self.timers_read(address))
            .or_else(|_| self.cache_ctrl_read::<T>(address))
            .or_else(|_| self.generic_read::<T>(address))
    }

    pub fn try_write<T: Copy>(&mut self, address: u32, value: T) -> IOResult<()> {
        #[cfg(feature = "debugger-ext")]
        {
            use crate::debug::BreakpointKind;

            self.dbg.break_on(address, BreakpointKind::WRITE);
        }

        self.fastmem_write::<T>(address, value)
            .or_else(|_| self.scratch_write(address, value))
            .or_else(|_| self.timers_write(address, value))
            .or_else(|_| self.irq_write(address, value))
            .or_else(|_| self.gpu_write(address, value))
            .or_else(|_| self.spu_write(address, value))
            .or_else(|_| self.dma_write(address, value))
            .or_else(|_| self.sio_write::<T>(address, value))
            .or_else(|_| self.cdrom_write::<T>(address, value))
            .or_else(|_| self.cache_ctrl_write::<T>(address, value))
            .or_else(|_| self.generic_write::<T>(address, value))
    }

    #[pchan_macros::instrument(skip_all)]
    pub fn write32_unaligned_l(&mut self, address: u32, value: u32) {
        if let Err(err) = self.try_write32_unaligned_l(address, value) {
            self.panic(&format!("{}", err));
        }
    }

    #[pchan_macros::instrument(skip_all)]
    pub fn write32_unaligned_r(&mut self, address: u32, value: u32) {
        if let Err(err) = self.try_write32_unaligned_r(address, value) {
            self.panic(&format!("{}", err));
        }
    }

    pub fn read32_unaligned_l(&mut self, address: u32, overwrite: u32) -> u32 {
        match self.try_read32_unaligned_l(address, overwrite) {
            Err(err) => {
                self.panic(&format!("{err}"));
            }
            Ok(value) => value,
        }
    }

    pub fn read32_unaligned_r(&mut self, address: u32, overwrite: u32) -> u32 {
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

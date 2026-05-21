// allow for dev
#![allow(dead_code)]
#![allow(long_running_const_eval)]
#![allow(incomplete_features)]
#![allow(clippy::collapsible_if)]
#![feature(arbitrary_self_types_pointers)]
#![cfg_attr(test, feature(random))]
#![feature(read_array)]
#![feature(stmt_expr_attributes)]
#![feature(const_clone)]
#![feature(const_default)]
#![feature(derive_const)]
#![feature(const_convert)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(const_trait_impl)]
#![feature(iter_intersperse)]
#![feature(generic_const_exprs)]
#![feature(const_array)]
#![feature(portable_simd)]
#![feature(const_try)]
#![feature(explicit_tail_calls)]
#![feature(const_destruct)]
// allow unused variables in tests to supress the setup tracing warnings
#![cfg_attr(test, allow(unused_variables))]

use std::mem::offset_of;

#[cfg(feature = "debugger-ext")]
use crate::debug::DebuggerState;

use crate::bootloader::BootloaderState;
use crate::cpu::Cpu;
use crate::dynarec_v2::DynarecCache;
use crate::gpu::GpuState;
use crate::io::cdrom::CDRomState;
use crate::io::dma::DmaState;
use crate::io::evque::Evque;
use crate::io::irq::IrqState;
use crate::io::sio::SioState;
use crate::io::timers::TimerState;
use crate::io::tty::Tty;
use crate::memory::MemoryState;
use crate::spu::{Spu, SpuState};

pub mod bindings;
pub mod bootloader;
pub mod cpu;
// #[path = "./dynarec/dynarec.rs"]
// pub mod dynarec;
#[cfg(feature = "debugger-ext")]
pub mod debug;
#[path = "./dynarec-v2/dynarec-v2.rs"]
pub mod dynarec_v2;
#[path = "./gpu/gpu.rs"]
pub mod gpu;
#[path = "./io/io.rs"]
pub mod io;
pub mod memory;
#[path = "./spu/spu.rs"]
pub mod spu;

pub const fn max_simd_width_bytes() -> usize {
    if cfg!(target_feature = "avx512f") {
        return 64;
    } // 512 bits

    if cfg!(target_feature = "neon") {
        return 16;
    }

    if cfg!(target_feature = "avx2") {
        return 32;
    } // 256 bits

    if cfg!(target_feature = "sse2") {
        return 16;
    } // 128 bits

    1
}

pub const MAX_SIMD_WIDTH: usize = max_simd_width_bytes();

pub const fn max_simd_elements<T>() -> usize {
    max_simd_width_bytes() / size_of::<T>()
}

#[derive(Default, derive_more::Debug, Clone)]
#[repr(C)]
pub struct Emu {
    pub cpu:           Cpu,
    #[debug(skip)]
    pub dynarec_cache: DynarecCache,
    pub mem:           MemoryState,
    pub boot:          BootloaderState,
    pub tty:           Tty,
    pub gpu:           GpuState,
    pub dma:           DmaState,
    pub timers:        TimerState,
    #[debug(skip)]
    pub spu:           SpuState,
    #[cfg(feature = "debugger-ext")]
    pub dbg:           DebuggerState,
    pub cdrom:         CDRomState,
    pub sio:           SioState,
    pub irq:           IrqState,
    pub evque:         Evque<Self>,
}

impl Emu {
    const PC_OFFSET: usize = offset_of!(Emu, cpu) + Cpu::PC_OFFSET;
    const D_CLOCK_OFFSET: usize = offset_of!(Emu, cpu) + Cpu::D_CLOCK_OFFSET;
    const HILO_OFFSET: usize = offset_of!(Emu, cpu) + Cpu::HILO_OFFSET;

    pub fn reg_offset(reg: u8) -> usize {
        offset_of!(Self, cpu) + Cpu::reg_offset(reg)
    }

    pub fn panic(&self, panic_msg: &str) -> ! {
        panic!(
            "emulator panicked at pc={} with:\n{panic_msg}\n\nstate = {:#?}",
            hex(self.cpu.pc),
            self
        )
    }

    pub fn new() -> Self {
        let mut emu = Emu::default();
        emu.handle_ev_spu_clock(io::evque::EvCtx::ZERO);
        emu
    }
}

use pchan_utils::hex;

pub trait Bus {
    fn mem_mut(&mut self) -> &mut MemoryState;
    fn mem(&self) -> &MemoryState;
    fn cpu_mut(&mut self) -> &mut Cpu;
    fn cpu(&self) -> &Cpu;
    fn bootloader_mut(&mut self) -> &mut BootloaderState;
    fn bootloader(&mut self) -> &BootloaderState;
    fn gpu(&self) -> &GpuState;
    fn gpu_mut(&mut self) -> &mut GpuState;
    fn timers(&self) -> &TimerState;
    fn timers_mut(&mut self) -> &mut TimerState;
    fn dma(&self) -> &DmaState;
    fn dma_mut(&mut self) -> &mut DmaState;
    fn spu(&self) -> &SpuState;
    fn spu_mut(&mut self) -> &mut SpuState;
    fn cdrom(&self) -> &CDRomState;
    fn cdrom_mut(&mut self) -> &mut CDRomState;
    fn sio(&self) -> &SioState;
    fn sio_mut(&mut self) -> &mut SioState;
    fn irq_mut(&mut self) -> &mut IrqState;
    fn irq(&self) -> &IrqState;
    fn evque(&self) -> &Evque<Self>;
    fn evque_mut(&mut self) -> &mut Evque<Self>;
}

impl Bus for Emu {
    #[inline(always)]
    fn mem_mut(&mut self) -> &mut MemoryState {
        &mut self.mem
    }
    #[inline(always)]
    fn cpu(&self) -> &Cpu {
        &self.cpu
    }
    #[inline(always)]
    fn mem(&self) -> &MemoryState {
        &self.mem
    }
    #[inline(always)]
    fn cpu_mut(&mut self) -> &mut Cpu {
        &mut self.cpu
    }
    #[inline(always)]
    fn bootloader_mut(&mut self) -> &mut BootloaderState {
        &mut self.boot
    }
    #[inline(always)]
    fn bootloader(&mut self) -> &BootloaderState {
        &self.boot
    }
    #[inline(always)]
    fn gpu_mut(&mut self) -> &mut GpuState {
        &mut self.gpu
    }
    #[inline(always)]
    fn gpu(&self) -> &GpuState {
        &self.gpu
    }
    #[inline(always)]
    fn timers(&self) -> &TimerState {
        &self.timers
    }
    #[inline(always)]
    fn timers_mut(&mut self) -> &mut TimerState {
        &mut self.timers
    }
    #[inline(always)]
    fn dma(&self) -> &DmaState {
        &self.dma
    }
    #[inline(always)]
    fn dma_mut(&mut self) -> &mut DmaState {
        &mut self.dma
    }

    #[inline(always)]
    fn spu(&self) -> &SpuState {
        &self.spu
    }

    #[inline(always)]
    fn spu_mut(&mut self) -> &mut SpuState {
        &mut self.spu
    }

    #[inline(always)]
    fn cdrom(&self) -> &CDRomState {
        &self.cdrom
    }

    #[inline(always)]
    fn cdrom_mut(&mut self) -> &mut CDRomState {
        &mut self.cdrom
    }

    #[inline(always)]
    fn sio(&self) -> &SioState {
        &self.sio
    }

    #[inline(always)]
    fn sio_mut(&mut self) -> &mut SioState {
        &mut self.sio
    }
    fn irq_mut(&mut self) -> &mut IrqState {
        &mut self.irq
    }
    fn irq(&self) -> &IrqState {
        &self.irq
    }

    fn evque(&self) -> &Evque<Self> {
        &self.evque
    }

    fn evque_mut(&mut self) -> &mut Evque<Self> {
        &mut self.evque
    }
}

#[cfg(test)]
pub mod test_utils {

    use crate::Emu;
    use rstest::fixture;

    #[fixture]
    pub fn emulator() -> Emu {
        Emu::default()
    }
}

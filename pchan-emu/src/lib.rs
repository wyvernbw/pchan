// allow for dev
#![allow(dead_code)]
#![allow(long_running_const_eval)]
#![allow(incomplete_features)]
#![allow(clippy::collapsible_if)]
// feature flags
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
#![feature(arbitrary_self_types)]
#![feature(alloc_slice_into_array)]
#![feature(allocator_ext)]
// allow unused variables in tests to supress the setup tracing warnings
#![cfg_attr(test, allow(unused_variables))]
use std::alloc::Global;
//
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
use crate::spu::SpuState;

pub mod bindings;
pub mod bootloader;
pub mod cpu;
#[cfg(feature = "debugger-ext")]
pub mod debug;
#[path = "./dynarec-v2/dynarec-v2.rs"]
pub mod dynarec_v2;
#[path = "./gpu/gpu.rs"]
pub mod gpu;
#[path = "./io/io.rs"]
pub mod io;
pub mod memory;
pub mod run;
#[path = "./spu/spu.rs"]
pub mod spu;

#[derive(derive_more::Debug, Clone)]
#[repr(C)]
pub struct Emu<A: Allocator> {
    pub cpu:           Cpu,
    #[debug(skip)]
    pub dynarec_cache: DynarecCache<A>,
    pub mem:           MemoryState<A>,
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
    pub tracy:         TracyClient,
    pub stats:         Stats,
}

#[derive(Default, derive_more::Debug, Clone)]
pub struct Stats {
    pub blocks_compiled: u64,
    pub blocks_ran:      u64,
}

use core::alloc::Allocator;

impl<A: Allocator> Emu<A> {
    const PC_OFFSET: usize = offset_of!(Emu<A>, cpu) + Cpu::PC_OFFSET;
    const D_CLOCK_OFFSET: usize = offset_of!(Emu<A>, cpu) + Cpu::D_CLOCK_OFFSET;
    const HILO_OFFSET: usize = offset_of!(Emu<A>, cpu) + Cpu::HILO_OFFSET;

    pub fn reg_offset(reg: u8) -> usize {
        offset_of!(Self, cpu) + Cpu::reg_offset(reg)
    }

    pub fn panic(&self, panic_msg: &str) -> ! {
        self.dma.dump_cdrom_data();
        panic!(
            "emulator panicked at pc={} with:\n{panic_msg}\n\nstate = {:#?}",
            hex(self.cpu.pc),
            self
        )
    }

    pub fn new_in(alloc: &dyn Allocator) -> Self {
        let mut emu = Self {
            cpu:           Cpu::new(),
            dynarec_cache: DynarecCache::new(alloc),
            mem:           MemoryState::new(alloc),
            boot:          BootloaderState::default(),
            tty:           Tty::default(),
            gpu:           Default::default(),
            dma:           Default::default(),
            timers:        Default::default(),
            spu:           Default::default(),
            dbg:           Default::default(),
            cdrom:         Default::default(),
            sio:           Default::default(),
            irq:           Default::default(),
            evque:         Default::default(),
            tracy:         Default::default(),
            stats:         Default::default(),
        };
        emu.handle_ev_spu_clock(io::evque::EvCtx::ZERO);
        emu
    }
}

impl Emu<std::alloc::Global> {
    pub fn new() -> Self {
        let emu = Emu::new_in(&std::alloc::Global);
        emu
    }
}

impl Stats {
    pub fn pop_frame_blocks_compiled(&mut self) -> u64 {
        let res = self.blocks_compiled;
        self.blocks_compiled = 0;
        res
    }
    pub fn pop_frame_blocks_ran(&mut self) -> u64 {
        let res = self.blocks_ran;
        self.blocks_ran = 0;
        res
    }
}

use pchan_utils::hex;
use pchan_utils::tracy::TracyClient;

impl<A: Allocator> Emu<A> {
    #[inline(always)]
    pub fn mem_mut(&mut self) -> &mut MemoryState<A> {
        &mut self.mem
    }
    #[inline(always)]
    pub fn cpu(&self) -> &Cpu {
        &self.cpu
    }
    #[inline(always)]
    pub fn mem(&self) -> &MemoryState<A> {
        &self.mem
    }
    #[inline(always)]
    pub fn cpu_mut(&mut self) -> &mut Cpu {
        &mut self.cpu
    }
    #[inline(always)]
    pub fn bootloader_mut(&mut self) -> &mut BootloaderState {
        &mut self.boot
    }
    #[inline(always)]
    pub fn bootloader(&mut self) -> &BootloaderState {
        &self.boot
    }
    #[inline(always)]
    pub fn gpu_mut(&mut self) -> &mut GpuState {
        &mut self.gpu
    }
    #[inline(always)]
    pub fn gpu(&self) -> &GpuState {
        &self.gpu
    }
    #[inline(always)]
    pub fn timers(&self) -> &TimerState {
        &self.timers
    }
    #[inline(always)]
    pub fn timers_mut(&mut self) -> &mut TimerState {
        &mut self.timers
    }
    #[inline(always)]
    pub fn dma(&self) -> &DmaState {
        &self.dma
    }
    #[inline(always)]
    pub fn dma_mut(&mut self) -> &mut DmaState {
        &mut self.dma
    }
    #[inline(always)]
    pub fn spu(&self) -> &SpuState {
        &self.spu
    }
    #[inline(always)]
    pub fn spu_mut(&mut self) -> &mut SpuState {
        &mut self.spu
    }
    #[inline(always)]
    pub fn cdrom(&self) -> &CDRomState {
        &self.cdrom
    }
    #[inline(always)]
    pub fn cdrom_mut(&mut self) -> &mut CDRomState {
        &mut self.cdrom
    }
    #[inline(always)]
    pub fn sio(&self) -> &SioState {
        &self.sio
    }
    #[inline(always)]
    pub fn sio_mut(&mut self) -> &mut SioState {
        &mut self.sio
    }
    #[inline(always)]
    pub fn irq_mut(&mut self) -> &mut IrqState {
        &mut self.irq
    }
    #[inline(always)]
    pub fn irq(&self) -> &IrqState {
        &self.irq
    }
    #[inline(always)]
    pub fn evque(&self) -> &Evque<Self> {
        &self.evque
    }
    #[inline(always)]
    pub fn evque_mut(&mut self) -> &mut Evque<Self> {
        &mut self.evque
    }
}

impl Default for Emu<Global> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
pub mod test_utils {

    use std::alloc::Global;

    use crate::Emu;
    use rstest::fixture;

    #[fixture]
    pub fn emulator() -> Emu<Global> {
        Emu::new()
    }
}

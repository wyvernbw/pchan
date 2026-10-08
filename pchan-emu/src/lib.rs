// allow for dev
#![allow(dead_code)]
#![allow(long_running_const_eval)]
#![allow(incomplete_features)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::inline_always)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
// required for dynasm macro
#![allow(clippy::semicolon_if_nothing_returned)]
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
#![feature(integer_widen_truncate)]
#![feature(allocator_ext)]
#![feature(default_field_values)]
#![feature(iter_array_chunks)]
// allow unused variables in tests to supress the setup tracing warnings
#![cfg_attr(test, allow(unused_variables))]
use core::alloc::Allocator;
use std::mem::offset_of;

extern crate alloc;

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
pub struct Emu<A: Allocator = Global> {
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
    pub spu:           SpuState<A>,
    #[cfg(feature = "debugger-ext")]
    pub dbg:           DebuggerState<A>,
    pub cdrom:         CDRomState,
    pub sio:           SioState,
    pub irq:           IrqState,
    pub evque:         Evque<Self>,
    pub tracy:         TracyClient,
    pub stats:         Stats,
    #[debug(skip)]
    pub alloc:         A,
}

#[derive(Default, derive_more::Debug, Clone)]
pub struct Stats {
    pub blocks_compiled: u64,
    pub blocks_ran:      u64,
}
impl<A: Allocator + Copy> Emu<A> {
    pub fn new_in(alloc: A) -> Self {
        let mut emu = Self {
            cpu: Cpu::new(),
            dynarec_cache: DynarecCache::new(alloc),
            mem: MemoryState::new(alloc),
            boot: BootloaderState::default(),
            tty: Tty::default(),
            gpu: GpuState::default(),
            dma: DmaState::default(),
            timers: TimerState::default(),
            spu: SpuState::new(alloc),
            #[cfg(feature = "debugger-ext")]
            dbg: DebuggerState::new(alloc),
            cdrom: CDRomState::default(),
            sio: SioState::default(),
            irq: IrqState::default(),
            evque: Evque::default(),
            tracy: TracyClient::default(),
            stats: Stats::default(),
            alloc,
        };
        emu.handle_ev_spu_clock(io::evque::EvCtx::default());
        emu
    }
}

impl<A: Allocator> Emu<A> {
    const PC_OFFSET: usize = offset_of!(Emu<A>, cpu) + Cpu::PC_OFFSET;
    const D_CLOCK_OFFSET: usize = offset_of!(Emu<A>, cpu) + Cpu::D_CLOCK_OFFSET;
    const HILO_OFFSET: usize = offset_of!(Emu<A>, cpu) + Cpu::HILO_OFFSET;

    #[must_use]
    pub fn reg_offset(reg: u8) -> usize {
        offset_of!(Self, cpu) + Cpu::reg_offset(reg)
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn panic(&self, panic_msg: &str) -> ! {
        self.dma.dump_cdrom_data();
        tracing::error!(
            "emulator panicked at pc={} with:\n{panic_msg}\n\nstate = {:#?}",
            hex(self.cpu.pc),
            self
        );
        panic!(
            "emulator panicked at pc={} with:\n{panic_msg}. state dumped to trace.",
            hex(self.cpu.pc),
        );
    }
}

impl Emu<Global> {
    #[must_use]
    pub fn new() -> Self {
        Emu::new_in(std::alloc::Global)
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

use alloc::alloc::Global;
use pchan_utils::hex;
use pchan_utils::tracy::TracyClient;

impl Default for Emu<Global> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
pub mod test_utils {

    use crate::Emu;
    use alloc::alloc::Global;
    use rstest::fixture;

    #[fixture]
    pub fn emulator() -> Emu<Global> {
        Emu::new()
    }
}

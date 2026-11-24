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
#![allow(clippy::unused_async_trait_impl)] // worst lint of all time?
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
#![feature(const_array)]
#![feature(portable_simd)]
#![feature(const_destruct)]
#![feature(arbitrary_self_types)]
#![feature(integer_widen_truncate)]
#![feature(allocator_ext)]
#![feature(default_field_values)]
#![feature(vec_push_within_capacity)]
#![feature(specialization)]
#![feature(alloc_io)]
#![feature(core_io)]
#![feature(pattern)]
// allow unused variables in tests to supress the setup tracing warnings
#![cfg_attr(test, allow(unused_variables))]
use std::mem::offset_of;

extern crate alloc;

use crate::bindings::ExternFnTable;
use crate::cpu::irq::Irqs;
use crate::cpu::timers::TimerState;
#[cfg(feature = "debugger-ext")]
use crate::debug::DebuggerState;

use crate::bootloader::BootloaderState;
use crate::cdrom::CDRom;
use crate::cpu::Cpu;
use crate::dma::DmaController;
use crate::dynarec_v2::DynCache;
use crate::evque::Evque;
use crate::gpu::Gpu;
use crate::mdec::MDec;
use crate::memory::Memory;
use crate::sio::Sio;
use crate::spu::Spu;
use crate::tty::Tty;

pub mod bindings;
pub mod bootloader;
#[path = "./cdrom/cdrom.rs"]
pub mod cdrom;
#[path = "./cpu/cpu.rs"]
pub mod cpu;
#[cfg(feature = "debugger-ext")]
pub mod debug;
pub mod dma;
#[path = "./dynarec-v2/dynarec-v2.rs"]
pub mod dynarec_v2;
pub mod evque;
#[path = "./gpu/gpu.rs"]
pub mod gpu;
pub mod io;
pub mod mdec;
#[path = "./memory/memory.rs"]
pub mod memory;
pub mod run;
#[path = "./sio/sio.rs"]
pub mod sio;
#[path = "./spu/spu.rs"]
pub mod spu;
pub mod tty;

#[derive(derive_more::Debug, Clone)]
#[repr(C)]
pub struct Emu {
    pub cpu:             Cpu,
    #[debug(skip)]
    pub(crate) fn_table: ExternFnTable,
    #[debug(skip)]
    pub dynarec_cache:   DynCache,
    pub mem:             Memory,
    pub boot:            BootloaderState,
    pub tty:             Tty,
    pub gpu:             Gpu,
    pub dma:             DmaController,
    pub timers:          TimerState,
    #[debug(skip)]
    pub spu:             Spu,
    #[cfg(feature = "debugger-ext")]
    pub dbg:             DebuggerState,
    pub cdrom:           CDRom,
    pub sio:             Sio,
    pub irq:             Irqs,
    pub mdec:            MDec,
    pub evque:           Evque,
    pub tracy:           TracyClient,
    pub stats:           Stats,
}

#[derive(Default, derive_more::Debug, Clone)]
pub struct Stats {
    pub blocks_compiled: u64,
    pub blocks_ran:      u64,

    pub dma_transfers_per_frame: u64,
}
impl Emu {
    #[must_use]
    pub fn new() -> Self {
        let mut emu = Self {
            cpu: Cpu::new(),
            fn_table: ExternFnTable::new(),
            dynarec_cache: DynCache::new(),
            mem: Memory::new(),
            boot: BootloaderState::new(),
            tty: Tty::new(),
            gpu: Gpu::new(),
            dma: DmaController::new(),
            timers: TimerState::default(),
            spu: Spu::new(),
            #[cfg(feature = "debugger-ext")]
            dbg: DebuggerState::new(),
            cdrom: CDRom::new(),
            sio: Sio::default(),
            irq: Irqs::default(),
            evque: Evque::new(),
            tracy: TracyClient::default(),
            stats: Stats::default(),
            mdec: MDec::default(),
        };
        emu.handle_ev_spu_clock(evque::EvCtx::default());
        emu
    }
}

impl Emu {
    const PC_OFFSET: usize = offset_of!(Emu, cpu) + Cpu::PC_OFFSET;
    const D_CLOCK_OFFSET: usize = offset_of!(Emu, cpu) + Cpu::D_CLOCK_OFFSET;
    const HILO_OFFSET: usize = offset_of!(Emu, cpu) + Cpu::HILO_OFFSET;

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

impl Default for Emu {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
pub mod test_utils {

    use crate::Emu;
    use rstest::fixture;

    #[fixture]
    pub fn emulator() -> Emu {
        Emu::new()
    }
}

// allow for dev
#![allow(dead_code)]
#![allow(long_running_const_eval)]
#![allow(incomplete_features)]
#![allow(clippy::collapsible_if)]
#![feature(test)]
#![feature(mut_ref)]
#![feature(arbitrary_self_types_pointers)]
#![feature(slice_swap_unchecked)]
#![feature(random)]
#![feature(read_array)]
#![feature(ascii_char)]
#![feature(bstr)]
#![feature(iter_array_chunks)]
#![feature(pointer_is_aligned_to)]
#![feature(stmt_expr_attributes)]
#![feature(const_clone)]
#![feature(const_default)]
#![feature(derive_const)]
#![feature(const_convert)]
#![feature(associated_type_defaults)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(const_ops)]
#![feature(const_trait_impl)]
#![feature(debug_closure_helpers)]
#![feature(iter_intersperse)]
#![feature(generic_const_exprs)]
#![feature(portable_simd)]
#![feature(try_blocks)]
// allow unused variables in tests to supress the setup tracing warnings
#![cfg_attr(test, allow(unused_variables))]

use std::{collections::HashMap, mem::offset_of};

use crate::{
    bootloader::BootloaderState, cpu::Cpu, dynarec_v2::DynarecBlock, gpu::GpuState, io::tty::Tty,
    memory::MemoryState,
};

pub mod bindings;
pub mod bootloader;
pub mod cpu;
// #[path = "./dynarec/dynarec.rs"]
// pub mod dynarec;
#[path = "./dynarec-v2/dynarec-v2.rs"]
pub mod dynarec_v2;
#[path = "./gpu/gpu.rs"]
pub mod gpu;
#[path = "./io/io.rs"]
pub mod io;
pub mod memory;

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
    pub dynarec_cache: HashMap<u32, DynarecBlock>,
    pub mem:           MemoryState,
    pub boot:          BootloaderState,
    pub tty:           Tty,
    pub gpu:           GpuState,
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

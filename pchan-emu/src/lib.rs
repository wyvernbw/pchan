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
    bootloader::BootloaderState,
    cpu::Cpu,
    dynarec::{FetchSummary, prelude::PureInstBuilder},
    dynarec_v2::DynarecBlock,
    io::tty::Tty,
    jit::{JitCache, LUTMap},
    memory::MemoryState,
};

pub mod cranelift_bs {
    pub use cranelift::codegen::ir::*;
    #[allow(ambiguous_glob_reexports)]
    pub use cranelift::jit::*;
    pub use cranelift::module::*;
    pub use cranelift::prelude::isa::*;
    pub use cranelift::prelude::*;
}

pub mod bindings;
pub mod bootloader;
pub mod cpu;
#[path = "./dynarec/dynarec.rs"]
pub mod dynarec;
#[path = "./dynarec-v2/dynarec-v2.rs"]
pub mod dynarec_v2;
#[path = "./io/io.rs"]
pub mod io;
pub mod jit;
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
    pub cpu: Cpu,
    #[debug(skip)]
    pub dynarec_cache: HashMap<u32, DynarecBlock>,
    pub mem: MemoryState,
    pub boot: BootloaderState,
    #[debug(skip)]
    pub jit_cache: JitCache,
    #[debug(skip)]
    pub inst_cache: LUTMap<FetchSummary>,
    pub tty: Tty,
}

impl Emu {
    const PC_OFFSET: usize = offset_of!(Emu, cpu) + Cpu::PC_OFFSET;
    const D_CLOCK_OFFSET: usize = offset_of!(Emu, cpu) + Cpu::D_CLOCK_OFFSET;

    pub fn reg_offset(reg: u8) -> usize {
        offset_of!(Self, cpu) + Cpu::reg_offset(reg)
    }

    pub fn panic(&self, panic_msg: &str) -> ! {
        panic!(
            r#"
            emulator panicked at pc={} with:
            {panic_msg}

            state = {:?}
            "#,
            hex(self.cpu.pc),
            self
        )
    }
}

use cranelift::{
    codegen::ir::{Inst, Opcode},
    prelude::*,
};
use pchan_utils::hex;

pub trait IntoInst {
    fn into_inst(self) -> Inst;
}

impl IntoInst for Inst {
    fn into_inst(self) -> Inst {
        self
    }
}

impl<T> IntoInst for (Inst, T) {
    fn into_inst(self) -> Inst {
        self.0
    }
}

pub trait FnBuilderExt<'a> {
    fn type_of(&self, value: Value) -> Type;
    fn single_result(&self, inst: Inst) -> Value;

    #[allow(non_snake_case)]
    fn Nop(&mut self) -> Inst;
    #[allow(non_snake_case)]
    fn PtrCast(&mut self, value: Value, ptr_type: Type) -> (Value, Inst);
    #[allow(non_snake_case)]
    fn IConst(&mut self, imm: impl Into<i64>) -> (Value, Inst);
    fn inst<R: IntoInst>(&mut self, f: impl Fn(&mut Self) -> R) -> (Value, Inst);
    fn pure<'short>(&'short mut self) -> PureInstBuilder<'short, 'a>;
}

impl<'a> FnBuilderExt<'a> for FunctionBuilder<'a> {
    fn type_of(&self, value: Value) -> Type {
        self.func.dfg.value_type(value)
    }
    fn single_result(&self, inst: Inst) -> Value {
        self.inst_results(inst)[0]
    }
    fn inst<R: IntoInst>(&mut self, f: impl Fn(&mut Self) -> R) -> (Value, Inst) {
        let inst = f(self).into_inst();
        let value = self.single_result(inst);
        (value, inst)
    }
    fn Nop(&mut self) -> Inst {
        let (inst, _) = self.pure().NullAry(Opcode::Nop, types::INVALID);
        inst
    }

    fn PtrCast(&mut self, value: Value, ptr_type: Type) -> (Value, Inst) {
        let value_type = self.type_of(value);
        if value_type == ptr_type {
            (value, self.Nop())
        } else {
            let extend_or_reduce = match (ptr_type, value_type) {
                (types::I64, types::I32 | types::I16 | types::I8)
                | (types::I32, types::I16 | types::I8) => {
                    self.pure().Unary(Opcode::Uextend, ptr_type, value).0
                }
                (types::I32, types::I64) => self.pure().Unary(Opcode::Ireduce, ptr_type, value).0,
                _ => panic!(
                    "invalid cast from {} to pointer type {}",
                    value_type, ptr_type
                ),
            };
            let value = self.single_result(extend_or_reduce);
            (value, extend_or_reduce)
        }
    }
    fn IConst(&mut self, imm: impl Into<i64>) -> (Value, Inst) {
        let imm = imm.into();
        self.inst(|f| {
            f.pure()
                .UnaryImm(Opcode::Iconst, types::I32, Imm64::new(imm))
                .0
        })
    }

    fn pure<'short>(&'short mut self) -> PureInstBuilder<'short, 'a> {
        let current_block = self.current_block().unwrap();
        PureInstBuilder {
            builder: self,
            block: current_block,
        }
    }
}

pub trait Bus {
    fn mem_mut(&mut self) -> &mut MemoryState;
    fn mem(&self) -> &MemoryState;
    fn cpu_mut(&mut self) -> &mut Cpu;
    fn cpu(&self) -> &Cpu;
    fn bootloader_mut(&mut self) -> &mut BootloaderState;
    fn bootloader(&mut self) -> &BootloaderState;
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
}

#[cfg(test)]
pub mod test_utils {

    use crate::{Emu, jit::JIT};
    use rstest::fixture;

    #[fixture]
    pub fn emulator() -> Emu {
        Emu::default()
    }

    #[fixture]
    pub fn jit() -> JIT {
        JIT::default()
    }
}

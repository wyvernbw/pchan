// allow for dev
#![allow(dead_code)]
#![allow(long_running_const_eval)]
#![allow(incomplete_features)]
#![allow(clippy::collapsible_if)]
#![feature(test)]
#![feature(arbitrary_self_types_pointers)]
#![feature(slice_swap_unchecked)]
#![feature(if_let_guard)]
#![feature(random)]
#![feature(stmt_expr_attributes)]
#![feature(ptr_as_ref_unchecked)]
#![feature(const_for)]
#![feature(assert_matches)]
#![feature(slice_concat_trait)]
#![feature(slice_as_array)]
#![feature(const_clone)]
#![feature(const_default)]
#![feature(derive_const)]
#![feature(hash_map_macro)]
#![feature(iter_map_windows)]
#![feature(iterator_try_collect)]
#![feature(const_convert)]
#![feature(explicit_tail_calls)]
#![feature(associated_type_defaults)]
#![feature(trait_alias)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(const_ops)]
#![feature(impl_trait_in_assoc_type)]
#![feature(const_trait_impl)]
#![feature(debug_closure_helpers)]
#![feature(iter_intersperse)]
#![feature(generic_const_exprs)]
#![feature(portable_simd)]
#![feature(iter_collect_into)]
#![feature(custom_inner_attributes)]
#![feature(iter_array_chunks)]
#![feature(try_blocks)]
#![feature(box_as_ptr)]
// allow unused variables in tests to supress the setup tracing warnings
#![cfg_attr(test, allow(unused_variables))]

use std::{
    collections::HashMap,
    mem::offset_of,
    simd::{LaneCount, SimdElement, SupportedLaneCount},
};

use crate::{
    bootloader::Bootloader,
    cpu::Cpu,
    dynarec::{FetchSummary, prelude::PureInstBuilder},
    dynarec_v2::DynarecBlock,
    jit::{JitCache, LUTMap},
    memory::{Chunk, Memory},
};

pub mod cranelift_bs {
    pub use cranelift::codegen::ir::*;
    #[allow(ambiguous_glob_reexports)]
    pub use cranelift::jit::*;
    pub use cranelift::module::*;
    pub use cranelift::prelude::isa::*;
    pub use cranelift::prelude::*;
}

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
    pub mem: Memory,
    pub boot: Bootloader,
    pub jit_cache: JitCache,
    pub inst_cache: InstCache,
}

pub type InstCache = LUTMap<FetchSummary>;

use memory::Extend;

impl Emu {
    const PC_OFFSET: usize = offset_of!(Emu, cpu) + Cpu::PC_OFFSET;
    const D_CLOCK_OFFSET: usize = offset_of!(Emu, cpu) + Cpu::D_CLOCK_OFFSET;

    pub fn read<T, E>(&self, address: u32) -> T::Out
    where
        T: Extend<E> + Copy,
    {
        self.mem.read::<T, E>(&self.cpu, address)
    }

    pub fn write<T: Copy>(&mut self, address: u32, value: T) {
        self.mem.write(&self.cpu, address, value);
    }

    pub fn write_many<T: SimdElement>(&mut self, address: u32, values: &[T])
    where
        LaneCount<{ Chunk::<T>::LANE_COUNT }>: SupportedLaneCount,
    {
        self.mem.write_many(&self.cpu, address, values);
    }
    pub fn reg_offset(reg: u8) -> usize {
        offset_of!(Self, cpu) + Cpu::reg_offset(reg)
    }
}

use cranelift::{
    codegen::ir::{Inst, Opcode},
    prelude::*,
};

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

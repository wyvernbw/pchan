// allow for dev
#![allow(dead_code)]
#![allow(long_running_const_eval)]
#![allow(incomplete_features)]
#![feature(slice_as_array)]
#![feature(const_for)]
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
#![feature(try_blocks)]
#![feature(impl_trait_in_assoc_type)]
#![feature(const_trait_impl)]
#![feature(debug_closure_helpers)]
#![feature(iter_intersperse)]
#![feature(generic_const_exprs)]
#![feature(portable_simd)]
#![feature(iter_collect_into)]
// allow unused variables in tests to supress the setup tracing warnings
#![cfg_attr(test, allow(unused_variables))]

use crate::{bootloader::Bootloader, cpu::Cpu, jit::JIT, memory::Memory};

pub mod cranelift_bs {
    pub use crate::dynarec::*;
    pub use cranelift::codegen::ir::*;
    #[allow(ambiguous_glob_reexports)]
    pub use cranelift::jit::*;
    pub use cranelift::module::*;
    pub use cranelift::prelude::isa::*;
    pub use cranelift::prelude::*;
}
pub mod bootloader;
pub mod cpu;
pub mod dynarec;
pub mod jit;
pub mod memory;

#[derive(Default, derive_more::Debug)]
pub struct Emu {
    #[debug(skip)]
    pub mem: Memory,
    pub cpu: Cpu,
    pub jit: JIT,
    pub boot: Bootloader,
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

pub trait FnBuilderExt {
    fn type_of(&self, value: Value) -> Type;
    fn single_result(&self, inst: Inst) -> Value;

    #[allow(non_snake_case)]
    fn Nop(&mut self) -> Inst;
    #[allow(non_snake_case)]
    fn PtrCast(&mut self, value: Value, ptr_type: Type) -> (Value, Inst);
    fn inst<R: IntoInst>(&mut self, f: impl Fn(&mut Self) -> R) -> (Value, Inst);
}

impl FnBuilderExt for FunctionBuilder<'_> {
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
        let (inst, _) = self.ins().NullAry(Opcode::Nop, types::INVALID);
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
                    self.ins().Unary(Opcode::Uextend, ptr_type, value).0
                }
                (types::I32, types::I64) => self.ins().Unary(Opcode::Ireduce, ptr_type, value).0,
                _ => panic!(
                    "invalid cast from {} to pointer type {}",
                    value_type, ptr_type
                ),
            };
            let value = self.single_result(extend_or_reduce);
            (value, extend_or_reduce)
        }
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

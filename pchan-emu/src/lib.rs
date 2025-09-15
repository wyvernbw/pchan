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

use cranelift::prelude::*;

pub trait FnBuilderExt {
    fn type_of(&self, value: Value) -> Type;
}

impl FnBuilderExt for FunctionBuilder<'_> {
    fn type_of(&self, value: Value) -> Type {
        self.func.dfg.value_type(value)
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

use std::ptr;

use cranelift::codegen::ir;

use crate::{cranelift_bs::*, memory::Memory};

#[cfg(test)]
mod cranelift_tests;
mod ops;

#[derive(Default)]
#[repr(C)]
pub(crate) struct Cpu {
    gpr: [u32; 32],
    pc: u32,
}

type Reg = usize;

const RA: Reg = 31;

pub(crate) struct JIT {
    /// The function builder context, which is reused across multiple
    /// FunctionBuilder instances.
    fn_builder_ctx: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    ctx: codegen::Context,

    /// The data description, which is to data objects what `ctx` is to functions.
    data_description: DataDescription,

    /// The module, with the jit backend, which manages the JIT'd
    /// functions.
    module: JITModule,

    basic_sig: Signature,
}

impl Default for JIT {
    fn default() -> Self {
        // Set up JIT
        let mut flags = settings::builder();
        flags.set("opt_level", "speed").unwrap();
        let isa = cranelift::native::builder()
            .unwrap()
            .finish(settings::Flags::new(flags))
            .unwrap();
        let jit_builder = JITBuilder::with_isa(isa, default_libcall_names());
        let module = JITModule::new(jit_builder);
        let fn_builder_ctx = FunctionBuilderContext::new();
        let ctx = module.make_context();
        let data_description = DataDescription::new();

        let ptr = module.target_config().pointer_type();
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(ptr));

        Self {
            module,
            fn_builder_ctx,
            data_description,
            ctx,
            basic_sig: sig,
        }
    }
}

#[bon::bon]
impl JIT {
    #[inline]
    pub(crate) fn pointer_type(&self) -> ir::Type {
        self.module.target_config().pointer_type()
    }

    pub(crate) fn get_func(&self, id: FuncId) -> BlockFn {
        let code_ptr = self.module.get_finalized_function(id);
        unsafe { std::mem::transmute::<*const u8, BlockFn>(code_ptr) }
    }

    pub(crate) fn create_signature(&self) -> Signature {
        self.basic_sig.clone()
    }

    #[builder]
    pub(crate) fn emit_load_reg(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        idx: usize,
    ) -> Value {
        let block_state = builder.block_params(block)[0];
        let offset = core::mem::offset_of!(Cpu, gpr);
        let offset = i32::try_from(offset + idx * size_of::<u32>()).expect("offset overflow");

        builder
            .ins()
            .load(types::I32, MemFlags::new(), block_state, offset)
    }

    #[builder]
    pub(crate) fn emit_store_reg(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        idx: usize,
        value: Value,
    ) {
        let block_state = builder.block_params(block)[0];
        let offset = core::mem::offset_of!(Cpu, gpr);
        let offset = i32::try_from(offset + idx * size_of::<u32>()).expect("offset overflow");
        builder
            .ins()
            .store(MemFlags::new(), value, block_state, offset);
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct BlockFn(fn(*const Cpu, *const Memory));

impl FnMut<(&Cpu, &Memory)> for BlockFn {
    extern "rust-call" fn call_mut(&mut self, args: (&Cpu, &Memory)) -> Self::Output {
        self.0(ptr::from_ref(args.0), ptr::from_ref(args.1))
    }
}

impl FnOnce<(&Cpu, &Memory)> for BlockFn {
    type Output = ();
    extern "rust-call" fn call_once(mut self, args: (&Cpu, &Memory)) -> Self::Output {
        self.call_mut(args)
    }
}

impl Fn<(&Cpu, &Memory)> for BlockFn {
    extern "rust-call" fn call(&self, args: (&Cpu, &Memory)) -> Self::Output {
        self.0(ptr::from_ref(args.0), ptr::from_ref(args.1))
    }
}

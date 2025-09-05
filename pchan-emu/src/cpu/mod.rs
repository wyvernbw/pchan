use std::{mem::offset_of, ptr};

use cranelift::codegen::ir;

use crate::cranelift_bs::*;

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

        Self {
            module,
            fn_builder_ctx,
            data_description,
            ctx,
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
pub(crate) struct BlockFn(fn(*const Cpu));

impl FnMut<(&Cpu,)> for BlockFn {
    extern "rust-call" fn call_mut(&mut self, args: (&Cpu,)) -> Self::Output {
        self.0(ptr::from_ref(args.0))
    }
}

impl FnOnce<(&Cpu,)> for BlockFn {
    type Output = ();
    extern "rust-call" fn call_once(mut self, args: (&Cpu,)) -> Self::Output {
        self.call_mut(args)
    }
}

impl Fn<(&Cpu,)> for BlockFn {
    extern "rust-call" fn call(&self, args: (&Cpu,)) -> Self::Output {
        self.0(ptr::from_ref(args.0))
    }
}

#[cfg(test)]
mod cranelift_tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::{fixture, rstest};

    use crate::cpu::{Cpu, JIT};

    #[fixture]
    fn cpu() -> Cpu {
        Cpu::default()
    }
    #[fixture]
    fn jit() -> JIT {
        JIT::default()
    }

    #[rstest]
    fn basic_jit(setup_tracing: (), mut cpu: Cpu, mut jit: JIT) -> color_eyre::Result<()> {
        use crate::cranelift_bs::*;

        // Function signature: i32 fn()
        let mut sig = Signature::new(CallConv::SystemV);
        sig.returns.push(AbiParam::new(types::I32));

        let func_id = jit
            .module
            .declare_function("const_42", Linkage::Export, &sig)?;
        let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);

        // Build function: return 42
        {
            let mut builder = FunctionBuilder::new(&mut func, &mut jit.fn_builder_ctx);
            let block = builder.create_block();
            builder.switch_to_block(block);
            builder.seal_block(block);
            let value = builder.ins().iconst(types::I32, 42);
            builder.ins().return_(&[value]);
            builder.finalize();
        }
        jit.ctx.func = func;
        // Create function
        jit.module.define_function(func_id, &mut jit.ctx)?;
        jit.module.clear_context(&mut jit.ctx);
        jit.module.finalize_definitions()?;

        let code_ptr = jit.module.get_finalized_function(func_id);
        let const_fn = unsafe { std::mem::transmute::<*const u8, fn() -> i32>(code_ptr) };
        tracing::info!("Result: {}", const_fn()); // Prints: Result: 42
        assert_eq!(const_fn(), 42);
        Ok(())
    }

    #[rstest]
    fn basic_adder(setup_tracing: (), mut cpu: Cpu, mut jit: JIT) -> color_eyre::Result<()> {
        use crate::cranelift_bs::*;

        let ptr = jit.pointer_type();
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(ptr));

        let func_id = jit
            .module
            .declare_function("adder", Linkage::Export, &sig)?;
        let mut func = Function::with_name_signature(UserFuncName::user(0, 1), sig);

        {
            let mut builder = FunctionBuilder::new(&mut func, &mut jit.fn_builder_ctx);
            let block = builder.create_block();
            builder.append_block_params_for_function_params(block);
            builder.switch_to_block(block);
            builder.seal_block(block);

            let a = JIT::emit_load_reg()
                .builder(&mut builder)
                .block(block)
                .idx(9)
                .call();
            let b = JIT::emit_load_reg()
                .builder(&mut builder)
                .block(block)
                .idx(10)
                .call();

            // x + y
            let sum = builder.ins().iadd(a, b);
            JIT::emit_store_reg()
                .builder(&mut builder)
                .block(block)
                .idx(11)
                .value(sum)
                .call();
            builder.ins().return_(&[]);

            builder.finalize();
        }

        jit.ctx.func = func;
        jit.module.define_function(func_id, &mut jit.ctx)?;

        jit.module.clear_context(&mut jit.ctx);
        jit.module.finalize_definitions()?;

        let adder = jit.get_func(func_id);

        cpu.gpr[9] = 69;
        cpu.gpr[10] = 42;

        adder(&cpu);

        assert_eq!(cpu.gpr[11], cpu.gpr[9] + cpu.gpr[10]);

        Ok(())
    }
}

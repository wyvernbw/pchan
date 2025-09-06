use std::{collections::HashMap, fmt::Display, ptr};

use cranelift::codegen::ir;
use tracing::instrument;

use crate::{cpu::ops::EmitSummary, cranelift_bs::*, memory::Memory};

#[cfg(test)]
mod cranelift_tests;
pub mod ops;

#[derive(Default)]
#[repr(C)]
pub(crate) struct Cpu {
    pub(crate) gpr: [u64; 32],
    pub(crate) pc: u64,
}

impl Display for Cpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gpr = self
            .gpr
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .map(|(idx, value)| format!("{idx}={value}"))
            .intersperse(",".to_string())
            .collect::<String>();
        let gpr = if gpr.is_empty() {
            "None".to_string()
        } else {
            gpr
        };
        write!(f, "cpu:gpr[{gpr}]")
    }
}

type Reg = usize;

const RA: Reg = 31;

pub(crate) struct JIT {
    /// The function builder context, which is reused across multiple
    /// FunctionBuilder instances.
    pub(crate) fn_builder_ctx: FunctionBuilderContext,
    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    pub(crate) ctx: codegen::Context,
    /// The data description, which is to data objects what `ctx` is to functions.
    pub(crate) data_description: DataDescription,
    /// The module, with the jit backend, which manages the JIT'd
    /// functions.
    pub(crate) module: JITModule,
    pub(crate) basic_sig: Signature,
    pub(crate) block_map: HashMap<u64, BlockFn>,
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
        sig.params.push(AbiParam::new(ptr));

        Self {
            module,
            fn_builder_ctx,
            data_description,
            ctx,
            basic_sig: sig,
            block_map: HashMap::default(),
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

    pub(crate) fn init_block(builder: &mut FunctionBuilder) -> Block {
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);
        block
    }

    #[builder]
    #[instrument(skip(builder, block))]
    pub(crate) fn emit_load_reg(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        idx: usize,
    ) -> Value {
        if idx == 0 {
            return builder.ins().iconst(types::I64, 0);
        }
        let block_state = builder.block_params(block)[0];
        let offset = core::mem::offset_of!(Cpu, gpr);
        let offset = i32::try_from(offset + idx * size_of::<u64>()).expect("offset overflow");

        builder
            .ins()
            .load(types::I64, MemFlags::new(), block_state, offset)
    }

    #[builder]
    #[instrument(skip(builder, block))]
    pub(crate) fn emit_store_reg(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        idx: usize,
        value: Value,
    ) {
        const GPR: usize = const { core::mem::offset_of!(Cpu, gpr) };
        tracing::debug!(?GPR);
        let block_state = builder.block_params(block)[0];
        let offset = i32::try_from(GPR + idx * size_of::<u64>()).expect("offset overflow");
        tracing::debug!(?offset);
        builder
            .ins()
            .store(MemFlags::new(), value, block_state, offset);
    }

    #[builder]
    #[instrument(skip(builder, block))]
    pub(crate) fn emit_updates(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        updates: Box<[(usize, Value)]>,
        mut cache: Option<&mut [Option<Value>; 32]>,
    ) {
        for (id, value) in updates.iter() {
            JIT::emit_store_reg()
                .block(block)
                .builder(builder)
                .idx(*id)
                .value(*value)
                .call();
            if let Some(cache) = cache.as_deref_mut() {
                cache[*id] = Some(*value);
            }
        }
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct BlockFn(fn(*mut Cpu, *mut [u8]));

impl FnMut<(&mut Cpu, &mut Memory)> for BlockFn {
    extern "rust-call" fn call_mut(&mut self, args: (&mut Cpu, &mut Memory)) -> Self::Output {
        self.0(ptr::from_mut(args.0), ptr::from_mut(args.1.as_mut()))
    }
}

impl FnOnce<(&mut Cpu, &mut Memory)> for BlockFn {
    type Output = ();
    extern "rust-call" fn call_once(mut self, args: (&mut Cpu, &mut Memory)) -> Self::Output {
        self.call_mut(args)
    }
}

impl Fn<(&mut Cpu, &mut Memory)> for BlockFn {
    extern "rust-call" fn call(&self, args: (&mut Cpu, &mut Memory)) -> Self::Output {
        self.0(ptr::from_mut(args.0), ptr::from_mut(args.1.as_mut()))
    }
}

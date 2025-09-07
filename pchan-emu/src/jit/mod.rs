use std::{collections::HashMap, ptr};

use cranelift::codegen::ir;
use tracing::instrument;

use crate::{
    cpu::{
        Cpu,
        ops::{DecodedOp, EmitParams, Op},
    },
    cranelift_bs::*,
    memory::Memory,
};

pub struct JIT {
    /// The function builder context, which is reused across multiple
    /// FunctionBuilder instances.
    pub fn_builder_ctx: FunctionBuilderContext,
    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    pub ctx: codegen::Context,
    /// The data description, which is to data objects what `ctx` is to functions.
    pub data_description: DataDescription,
    /// The module, with the jit backend, which manages the JIT'd
    /// functions.
    pub module: JITModule,
    pub basic_sig: Signature,
    pub block_map: HashMap<u64, BlockFn>,
    pub func_idx: usize,
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
            func_idx: 1,
        }
    }
}

#[bon::bon]
impl JIT {
    #[inline]
    pub fn pointer_type(&self) -> ir::Type {
        self.module.target_config().pointer_type()
    }

    pub fn get_func(&self, id: FuncId) -> BlockFn {
        let code_ptr = self.module.get_finalized_function(id);
        unsafe { std::mem::transmute::<*const u8, BlockFn>(code_ptr) }
    }

    pub fn create_signature(&self) -> Signature {
        self.basic_sig.clone()
    }

    #[inline]
    pub fn create_function(&mut self, address: u64) -> Result<(FuncId, Function), ModuleError> {
        let sig = self.create_signature();
        let func_id = self.module.declare_function(
            &format!("pc_0x{:08X}", address),
            Linkage::Hidden,
            &sig,
        )?;
        let func = Function::with_name_signature(
            UserFuncName::user(self.func_idx as u32, self.func_idx as u32),
            self.create_signature(),
        );
        self.func_idx += 1;
        Ok((func_id, func))
    }

    #[inline]
    pub fn create_fn_builder<'a>(&'a mut self, func: &'a mut Function) -> FunctionBuilder<'a> {
        FunctionBuilder::new(func, &mut self.fn_builder_ctx)
    }

    pub fn use_cached_function(&self, address: u64, cpu: &mut Cpu, mem: &mut Memory) -> bool {
        if let Some(function) = self.block_map.get(&address) {
            tracing::trace!("using cached function: 0x{:08X?}", function.0 as usize);
            function(cpu, mem);
            true
        } else {
            false
        }
    }

    pub fn finish_function(&mut self, func_id: FuncId, func: Function) -> ModuleResult<()> {
        self.ctx.func = func;
        self.module.define_function(func_id, &mut self.ctx)?;

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;
        Ok(())
    }

    pub fn init_block(builder: &mut FunctionBuilder) -> Block {
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);
        block
    }

    #[builder]
    #[instrument(skip(builder, block))]
    pub fn emit_load_reg(builder: &mut FunctionBuilder<'_>, block: Block, idx: usize) -> Value {
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
    pub fn emit_store_reg(
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
    #[instrument(skip(builder, block, cache))]
    pub fn emit_updates(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        updates: Option<&[(usize, Value)]>,
        mut cache: Option<&mut [Option<Value>; 32]>,
    ) {
        let Some(updates) = updates else {
            return;
        };
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
pub struct BlockFn(pub fn(*mut Cpu, *mut [u8]));

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

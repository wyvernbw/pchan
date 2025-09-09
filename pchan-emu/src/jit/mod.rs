use std::{
    collections::{HashMap, HashSet},
    ptr,
};

use cranelift::codegen::ir;
use tracing::{Instrument, Level, instrument};

use crate::{
    cpu::{Cpu, REG_STR},
    cranelift_bs::*,
    memory::Memory,
};

#[derive(derive_more::Debug)]
pub struct JIT {
    /// The function builder context, which is reused across multiple
    /// FunctionBuilder instances.
    #[debug(skip)]
    pub fn_builder_ctx: FunctionBuilderContext,
    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    #[debug(skip)]
    pub ctx: codegen::Context,
    /// The data description, which is to data objects what `ctx` is to functions.
    #[debug(skip)]
    pub data_description: DataDescription,
    /// The module, with the jit backend, which manages the JIT'd
    /// functions.
    #[debug(skip)]
    pub module: JITModule,
    #[debug(skip)]
    pub basic_sig: Signature,
    pub block_map: BlockMap,
    pub dirty_pages: HashSet<BlockPage>,
    pub func_idx: usize,
}

#[derive(derive_more::Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
#[debug("Page#{}", self.0)]
pub struct BlockPage(u64);

impl BlockPage {
    const SHIFT: u64 = 8;

    pub fn new(address: u64) -> Self {
        Self(address >> Self::SHIFT)
    }
}

#[derive(Debug, Clone, Default)]
pub struct BlockMap(HashMap<BlockPage, HashMap<u64, BlockFn>>);

impl BlockMap {
    pub fn insert(&mut self, address: u64, func: BlockFn) -> Option<BlockFn> {
        let page = BlockPage::new(address);
        self.0.entry(page).or_default().insert(address, func)
    }
    pub fn get(&self, address: u64) -> Option<&BlockFn> {
        let page = BlockPage::new(address);
        self.0.get(&page).and_then(|map| map.get(&address))
    }
}

impl Default for JIT {
    fn default() -> Self {
        // Set up JIT
        let mut flags = settings::builder();
        flags.set("opt_level", "none").unwrap();
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
            block_map: BlockMap::default(),
            func_idx: 1,
            dirty_pages: HashSet::default(),
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
    pub fn create_function(
        &mut self,
        address: u64,
    ) -> Result<(FuncId, Function), Box<ModuleError>> {
        let sig = self.create_signature();
        let func_id = self.module.declare_function(
            &format!("pc_0x{:08X}:{}", address, self.func_idx),
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

    pub fn clear_cache(&mut self) {
        self.dirty_pages.clear();
        self.block_map.0.clear();
    }

    pub fn apply_dirty_pages(&mut self, address: u64) {
        let page = BlockPage::new(address);
        if self.dirty_pages.remove(&page)
            && let Some(page) = self.block_map.0.get_mut(&page)
        {
            page.clear();
        }
    }

    pub fn use_cached_function(&self, address: u64, cpu: &mut Cpu, mem: &mut Memory) -> bool {
        if let Some(function) = self.block_map.get(address) {
            tracing::trace!("invoking cached function {function:?}");
            function(cpu, mem, false);
            true
        } else {
            false
        }
    }

    pub fn finish_function(
        &mut self,
        func_id: FuncId,
        func: Function,
    ) -> Result<(), Box<ModuleError>> {
        self.ctx.func = func;
        if let Err(err) = self.ctx.replace_redundant_loads() {
            tracing::warn!(%err);
        };
        self.module.define_function(func_id, &mut self.ctx)?;

        self.module.finalize_definitions()?;
        self.module.clear_context(&mut self.ctx);
        self.ctx.clear();
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
    #[instrument(skip(builder, block), fields(reg=REG_STR[idx], value))]
    pub fn emit_store_reg(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        idx: usize,
        value: Value,
    ) {
        if idx == 0 {
            return;
        }
        const GPR: usize = const { core::mem::offset_of!(Cpu, gpr) };
        let block_state = builder.block_params(block)[0];
        let offset = i32::try_from(GPR + idx * size_of::<u64>()).expect("offset overflow");
        tracing::trace!("stored");
        builder
            .ins()
            .store(MemFlags::new(), value, block_state, offset);
    }

    #[builder]
    #[instrument(skip_all)]
    pub fn apply_cache_updates(
        updates: Option<&[(usize, Value)]>,
        cache: &mut [Option<Value>; 32],
    ) {
        let Some(updates) = updates else {
            return;
        };
        for (id, value) in updates.iter() {
            cache[*id] = Some(*value);
        }
        tracing::trace!("applied cache");
    }

    #[builder]
    #[instrument(
        level = Level::DEBUG,
        skip(builder, block, cache ),
    )]
    pub fn emit_updates(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        cache: Option<&mut [Option<Value>; 32]>,
    ) {
        let Some(cache) = cache else {
            return;
        };
        cache
            .iter()
            .enumerate()
            .flat_map(|(idx, value)| value.map(|value| (idx, value)))
            .for_each(|(id, value)| {
                JIT::emit_store_reg()
                    .block(block)
                    .builder(builder)
                    .idx(id)
                    .value(value)
                    .call();
            });
        tracing::trace!("done");
    }

    pub fn cache_usage(&self) -> (usize, usize) {
        (
            self.block_map.0.len(),
            self.block_map.0.values().map(|page| page.len()).sum(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct BlockFn(pub fn(*mut Cpu, *mut [u8]));

type BlockFnArgs<'a> = (&'a mut Cpu, &'a mut Memory, bool);

impl BlockFn {
    fn call_block(&self, args: BlockFnArgs) {
        if args.2 {
            self.0
                .instrument(tracing::info_span!("fn", addr = ?self.0))
                .inner()(ptr::from_mut(args.0), ptr::from_mut(args.1.as_mut()))
        } else {
            self.0(ptr::from_mut(args.0), ptr::from_mut(args.1.as_mut()))
        }
    }
}

impl FnMut<BlockFnArgs<'_>> for BlockFn {
    extern "rust-call" fn call_mut(&mut self, args: BlockFnArgs) -> Self::Output {
        self.call_block(args)
    }
}

impl FnOnce<BlockFnArgs<'_>> for BlockFn {
    type Output = ();
    extern "rust-call" fn call_once(mut self, args: BlockFnArgs) -> Self::Output {
        self.call_mut(args)
    }
}

impl Fn<BlockFnArgs<'_>> for BlockFn {
    extern "rust-call" fn call(&self, args: BlockFnArgs) -> Self::Output {
        self.call_block(args);
    }
}

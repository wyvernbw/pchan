use std::{
    collections::{HashMap, HashSet},
    mem::offset_of,
    ptr,
};

use cranelift::codegen::{
    gimli::SectionBaseAddresses,
    ir::{self, immediates::Offset32},
};
use tracing::{Instrument, Level, instrument};

use crate::{
    FnBuilderExt,
    cpu::{Cop0, Cpu, REG_STR},
    cranelift_bs::*,
    dynarec::{CachedValue, EmitSummary, EntryCache},
    memory::{Memory, MemoryRegion},
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
pub struct BlockPage(u32);

impl BlockPage {
    const SHIFT: u32 = 8;

    pub fn new(address: u32) -> Self {
        Self(address >> Self::SHIFT)
    }
}

/// TODO: make it store a module function ref as well
#[derive(Debug, Clone, Default)]
pub struct BlockMap(HashMap<BlockPage, HashMap<u32, BlockFn>>);

impl BlockMap {
    pub fn insert(&mut self, address: u32, func: BlockFn) -> Option<BlockFn> {
        let page = BlockPage::new(address);
        self.0.entry(page).or_default().insert(address, func)
    }
    pub fn get(&self, address: u32) -> Option<&BlockFn> {
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

#[derive(Debug, Clone, Copy)]
pub struct CacheUpdates<'a> {
    registers: &'a [(usize, CachedValue)],
    hi: &'a Option<CachedValue>,
    lo: &'a Option<CachedValue>,
}

impl<'a> CacheUpdates<'a> {
    pub fn new(emit_summary: &'a EmitSummary) -> CacheUpdates<'a> {
        CacheUpdates {
            registers: &emit_summary.register_updates,
            hi: &emit_summary.hi,
            lo: &emit_summary.lo,
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
        address: u32,
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

    pub fn apply_dirty_pages(&mut self, address: u32) {
        let page = BlockPage::new(address);
        if self.dirty_pages.remove(&page)
            && let Some(page) = self.block_map.0.get_mut(&page)
        {
            page.clear();
        }
    }

    pub fn use_cached_function(
        &self,
        address: u32,
        cpu: &mut Cpu,
        mem: &mut Memory,
        mem_map: &[MemoryRegion],
    ) -> bool {
        if let Some(function) = self.block_map.get(address) {
            tracing::trace!("invoking cached function {function:?}");
            function(cpu, mem, false, mem_map);
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
    pub fn emit_load_reg(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        idx: usize,
    ) -> (Value, Inst) {
        if idx == 0 {
            let inst = builder
                .pure()
                .UnaryConst(Opcode::Iconst, types::I32, Constant::from_u32(0))
                .0;
            let const0 = builder.single_result(inst);
            return (const0, inst);
        }
        let cpu_value = builder.block_params(block)[0];
        let offset = core::mem::offset_of!(Cpu, gpr);
        let offset = i32::try_from(offset + idx * size_of::<u32>()).expect("offset overflow");
        let load = builder
            .pure()
            .Load(
                Opcode::Load,
                types::I32,
                MemFlags::new(),
                Offset32::new(offset),
                cpu_value,
            )
            .0;
        let reg = builder.single_result(load);
        (reg, load)
    }

    #[inline]
    pub fn emit_load_from_cpu(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        offset: i32,
    ) -> (Value, Inst) {
        let cpu_ptr = builder.block_params(block)[0];
        let load = builder
            .pure()
            .Load(
                Opcode::Load,
                types::I32,
                MemFlags::new(),
                Offset32::new(offset),
                cpu_ptr,
            )
            .0;
        let value = builder.single_result(load);

        (value, load)
    }

    #[builder]
    #[instrument(skip(builder, block))]
    #[inline]
    pub fn emit_load_cop_reg(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        cop: u8,
        idx: usize,
    ) -> (Value, Inst) {
        const COP0: usize = const { core::mem::offset_of!(Cpu, cop0) };
        const COP_SIZE: usize = size_of::<Cop0>();

        let offset = i32::try_from(COP0 + COP_SIZE * cop as usize + idx * size_of::<u32>())
            .expect("offset overflow");

        JIT::emit_load_from_cpu(builder, block, offset)
    }

    #[builder]
    #[instrument(skip(builder, block))]
    #[inline]
    pub fn emit_load_hi(builder: &mut FunctionBuilder<'_>, block: Block) -> (Value, Inst) {
        const HI: usize = core::mem::offset_of!(Cpu, hilo) + size_of::<u32>();

        JIT::emit_load_from_cpu(builder, block, HI as i32)
    }

    #[builder]
    #[instrument(skip(builder, block))]
    #[inline]
    pub fn emit_load_lo(builder: &mut FunctionBuilder<'_>, block: Block) -> (Value, Inst) {
        const LO: usize = core::mem::offset_of!(Cpu, hilo);

        JIT::emit_load_from_cpu(builder, block, LO as i32)
    }

    #[builder]
    #[instrument(skip(builder, block), fields(reg=REG_STR[idx], value))]
    pub fn emit_store_reg(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        idx: usize,
        value: Value,
    ) -> Inst {
        let value_type = builder.type_of(value);
        debug_assert_ne!(value_type, types::I64, "got 64bit write to register");

        if idx == 0 {
            return builder.Nop();
        }
        const GPR: usize = const { core::mem::offset_of!(Cpu, gpr) };
        let cpu_ptr = builder.block_params(block)[0];
        let offset = i32::try_from(GPR + idx * size_of::<u32>()).expect("offset overflow");

        builder
            .pure()
            .Store(
                Opcode::Store,
                value_type,
                MemFlags::new(),
                Offset32::new(offset),
                value,
                cpu_ptr,
            )
            .0
    }

    pub fn emit_store_to_cpu(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        value: Value,
        offset: i32,
    ) -> Inst {
        let cpu_ptr = builder.block_params(block)[0];
        let vtype = builder.type_of(value);
        builder
            .pure()
            .Store(
                Opcode::Store,
                vtype,
                MemFlags::new(),
                Offset32::new(offset),
                value,
                cpu_ptr,
            )
            .0
    }

    #[builder]
    #[instrument(skip(builder, block), fields(reg=idx, value))]
    pub fn emit_store_cop_reg(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        cop: u8,
        idx: usize,
        value: Value,
    ) -> Inst {
        debug_assert!((0..4).contains(&cop));
        debug_assert_eq!(
            builder.type_of(value),
            types::I32,
            "coprocessor register value must be i32"
        );

        const COP0: usize = const { core::mem::offset_of!(Cpu, cop0) };
        const COP_SIZE: usize = size_of::<Cop0>();
        let offset = i32::try_from(COP0 + COP_SIZE * cop as usize + idx * size_of::<u32>())
            .expect("offset overflow");

        JIT::emit_store_to_cpu(builder, block, value, offset)
    }

    #[instrument(skip(builder, block))]
    pub fn emit_store_hi(builder: &mut FunctionBuilder<'_>, block: Block, value: Value) -> Inst {
        debug_assert_eq!(
            builder.type_of(value),
            types::I32,
            "hi register value must be i32"
        );
        const HI: usize = const { core::mem::offset_of!(Cpu, hilo) + size_of::<u32>() };

        JIT::emit_store_to_cpu(builder, block, value, HI as i32)
    }

    #[instrument(skip(builder, block))]
    pub fn emit_store_lo(builder: &mut FunctionBuilder<'_>, block: Block, value: Value) -> Inst {
        debug_assert_eq!(
            builder.type_of(value),
            types::I32,
            "lo register value must be i32"
        );

        const LO: usize = const { core::mem::offset_of!(Cpu, hilo) };

        JIT::emit_store_to_cpu(builder, block, value, LO as i32)
    }

    #[instrument(skip(builder, block))]
    pub fn emit_store_hilo(builder: &mut FunctionBuilder<'_>, block: Block, value: Value) -> Inst {
        debug_assert_eq!(
            builder.func.dfg.value_type(value),
            types::I64,
            "expected an i64 value!"
        );
        JIT::emit_store_hi(builder, block, value)
    }

    #[builder]
    #[instrument(skip_all)]
    pub fn apply_cache_updates<'a, 'b>(updates: CacheUpdates<'a>, cache: &'b mut EntryCache) {
        for (id, value) in updates.registers.iter() {
            cache.registers[*id] = Some(*value);
        }
        // DONE: apply updates to hi and lo
        if let Some(hi) = updates.hi {
            cache.hi = Some(*hi);
        }
        if let Some(lo) = updates.lo {
            cache.lo = Some(*lo);
        }
        // tracing::trace!("applied cache");
    }

    #[builder]
    #[instrument(
        level = Level::DEBUG,
        skip(builder, block, cache ),
    )]
    pub fn emit_updates(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        cache: Option<&mut EntryCache>,
    ) -> Vec<Inst> {
        let Some(cache) = cache else {
            return vec![];
        };
        let mut instructions = cache
            .registers
            .iter()
            .enumerate()
            // skip $zero
            .skip(1)
            .flat_map(|(idx, value)| value.map(|value| (idx, value)))
            .filter(|(_, value)| value.dirty)
            .map(|(id, value)| {
                debug_assert_eq!(
                    builder.type_of(value.value),
                    types::I32,
                    "cached register must be an i32 value"
                );
                JIT::emit_store_reg()
                    .block(block)
                    .builder(builder)
                    .idx(id)
                    .value(value.value)
                    .call()
            })
            .collect::<Vec<_>>();

        // DONE: emit store hi and lo
        if let Some(hi) = cache.hi
            && hi.dirty
        {
            instructions.push(JIT::emit_store_hi(builder, block, hi.value));
        }
        if let Some(lo) = cache.lo
            && lo.dirty
        {
            instructions.push(JIT::emit_store_lo(builder, block, lo.value));
        }
        instructions
    }

    #[builder]
    pub fn emit_map_address_to_physical(
        fn_builder: &mut FunctionBuilder<'_>,
        address: Value,
    ) -> (Value, Inst) {
        debug_assert_eq!(
            fn_builder.func.dfg.value_type(address),
            types::I32,
            "expected 32bit virtual address!"
        );

        let band = fn_builder
            .pure()
            .BinaryImm64(
                Opcode::BandImm,
                types::I32,
                Imm64::new(0x1FFF_FFFF),
                address,
            )
            .0;
        let address = fn_builder.single_result(band);

        (address, band)
    }

    #[builder]
    pub fn emit_map_address_to_host(
        fn_builder: &mut FunctionBuilder<'_>,
        ptr_type: Type,
        mem_map_ptr: Value,
        address: Value,
    ) -> (Value, [Inst; 12]) {
        debug_assert_eq!(
            fn_builder.func.dfg.value_type(address),
            types::I32,
            "expected 32bit virtual address!"
        );

        // map virutal address to psx physical
        let (address, band) = JIT::emit_map_address_to_physical()
            .fn_builder(fn_builder)
            .address(address)
            .call();

        // cast address to host pointer type
        let (address, ptrcast0) = fn_builder.PtrCast(address, ptr_type);

        // convert address into memory region table index
        let ushr_imm0 = fn_builder
            .pure()
            .BinaryImm64(Opcode::UshrImm, ptr_type, Imm64::new(16), address)
            .0;
        let index = fn_builder.single_result(ushr_imm0);

        // convert index into table offset
        let imul_imm0 = fn_builder
            .pure()
            .BinaryImm64(
                Opcode::ImulImm,
                ptr_type,
                Imm64::new(size_of::<MemoryRegion>() as i64),
                index,
            )
            .0;
        let lookup_offset = fn_builder.single_result(imul_imm0);

        // add table offset to table pointer to get lookup address
        let iadd0 = fn_builder
            .pure()
            .Binary(Opcode::Iadd, ptr_type, mem_map_ptr, lookup_offset)
            .0;
        let lookup = fn_builder.single_result(iadd0);

        // load region descriptor at lookup address
        let load0 = fn_builder
            .pure()
            .LoadNoOffset(Opcode::Load, types::I64, MemFlags::new(), lookup)
            .0;
        let region_descriptor = fn_builder.single_result(load0);

        // get MemoryRegion.phys_start (high 32 bits)
        let ushr_imm1 = fn_builder
            .pure()
            .BinaryImm64(
                Opcode::UshrImm,
                types::I64,
                Imm64::new(offset_of!(MemoryRegion, phys_start) as i64 * 8),
                region_descriptor,
            )
            .0;
        let phys_start = fn_builder.single_result(ushr_imm1);

        // get MemoryRegion.host_start (low 32 bits)
        // first, reduce to I32
        let ireduce0 = fn_builder
            .pure()
            .Unary(Opcode::Ireduce, types::I32, region_descriptor)
            .0;
        let host_start = fn_builder.single_result(ireduce0);
        // then, extend back to I64, this effectively clears the high 32 bits
        let uextend0 = fn_builder
            .pure()
            .Unary(Opcode::Uextend, types::I64, host_start)
            .0;
        let host_start = fn_builder.single_result(uextend0);

        // subtract the psx physical start of the region from the physical address,
        // obtaining an offset into the region
        let isub0 = fn_builder
            .pure()
            .Binary(Opcode::Isub, types::I64, address, phys_start)
            .0;
        let offset_in_region = fn_builder.single_result(isub0);

        // calculate the host address by adding the region offset to the
        // start of the host region
        let (host_address, iadd1) = fn_builder.inst(|f| {
            f.pure()
                .Binary(Opcode::Iadd, types::I64, host_start, offset_in_region)
                .0
        });

        // cast host address from I64 to host pointer type
        let (host_address, ptrcast1) = fn_builder.PtrCast(host_address, ptr_type);

        (
            host_address,
            [
                band, ptrcast0, ushr_imm0, imul_imm0, iadd0, load0, ushr_imm1, ireduce0, uextend0,
                isub0, iadd1, ptrcast1,
            ],
        )
    }

    pub fn emit_store_pc(fn_builder: &mut FunctionBuilder<'_>, block: Block, pc: Value) -> Inst {
        tracing::info!("write to pc");

        JIT::emit_store_to_cpu(fn_builder, block, pc, offset_of!(Cpu, pc) as i32)
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
pub struct BlockFn(pub fn(*mut Cpu, *mut u8, *const MemoryRegion));

type BlockFnArgs<'a> = (&'a mut Cpu, &'a mut Memory, bool, &'a [MemoryRegion]);

impl BlockFn {
    fn call_block(&self, args: BlockFnArgs) {
        if args.2 {
            self.0
                .instrument(tracing::info_span!("fn", addr = ?self.0))
                .inner()(
                ptr::from_mut(args.0),
                args.1.as_mut().as_mut_ptr(),
                args.3.as_ptr(),
            )
        } else {
            self.0(
                ptr::from_mut(args.0),
                args.1.as_mut().as_mut_ptr(),
                args.3.as_ptr(),
            )
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

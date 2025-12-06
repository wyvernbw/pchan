use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    hash::Hash,
    mem::offset_of,
    ptr,
    sync::Arc,
};

use color_eyre::owo_colors::OwoColorize;
use cranelift::codegen::ir::{self, immediates::Offset32};
use cranelift_codegen::{
    gimli::{self, RunTimeEndian, write::FrameTable},
    isa::unwind::UnwindInfo,
};
use rapidhash::fast::RapidHasher;
use std::hash::Hasher;
use tracing::{Instrument, Level, enabled, instrument};

use crate::{
    Emu, FnBuilderExt,
    cpu::{Cop0, Cpu, REG_STR, Reg, ops::DecodedOp},
    cranelift_bs::*,
    dynarec::{CacheUpdates, EntryCache},
    io::IO,
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
    pub func_idx: usize,
    pub func_table: FunctionTable,
}

#[derive(derive_more::Debug, Default)]
pub struct JitCache {
    pub fn_map: FunctionMap,
    pub dirty_pages: HashSet<FunctionPage>,
}

impl JitCache {
    pub fn clear_cache(&mut self) {
        self.dirty_pages.clear();
        self.fn_map.0.clear();
    }

    pub fn apply_dirty_pages(&mut self, address: u32) {
        let page = FunctionPage::new(address);
        if self.dirty_pages.remove(&page)
            && let Some(page) = self.fn_map.0.get_mut(&page)
        {
            page.clear();
        }
    }

    pub fn cache_usage(&self) -> (usize, usize) {
        (
            self.fn_map.0.len(),
            self.fn_map.0.values().map(|page| page.len()).sum(),
        )
    }

    pub fn use_cached_function(
        &self,
        address: u32,
        fetched_ops: &[DecodedOp],
    ) -> Result<&BlockFn, u64> {
        let hash_ops = hash_ops(fetched_ops);
        self.fn_map
            .get(address)
            .and_then(|blockfn| {
                if hash_ops == blockfn.hash {
                    Some(blockfn)
                } else {
                    None
                }
            })
            .ok_or(hash_ops)
    }
}

#[derive(derive_more::Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
#[debug("Page#{}", self.0)]
pub struct FunctionPage(u32);

impl FunctionPage {
    const SHIFT: u32 = 8;

    pub fn new(address: u32) -> Self {
        Self(address >> Self::SHIFT)
    }
}

/// TODO: make it store a module function ref as well
#[derive(Debug, Clone, Default)]
pub struct FunctionMap(HashMap<FunctionPage, HashMap<u32, BlockFn>>);

impl FunctionMap {
    pub fn insert(&mut self, address: u32, func: BlockFn) -> Option<BlockFn> {
        let page = FunctionPage::new(address);
        self.0.entry(page).or_default().insert(address, func)
    }
    pub fn get(&self, address: u32) -> Option<&BlockFn> {
        let page = FunctionPage::new(address);
        self.0.get(&page).and_then(|map| map.get(&address))
    }
    pub fn functions(&self) -> impl Iterator<Item = (&u32, &BlockFn)> {
        self.0.values().flat_map(|page| page.iter())
    }
}

#[derive(derive_more::Debug)]
pub struct FunctionTable {
    read32: FuncId,
    readi16: FuncId,
    readi8: FuncId,
    readu16: FuncId,
    readu8: FuncId,

    write32: FuncId,
    write16: FuncId,
    write8: FuncId,

    handle_rfe: FuncId,
    handle_break: FuncId,
}

#[derive(derive_more::Debug)]
pub struct FuncRefTable {
    pub read32: FuncRef,
    pub readi16: FuncRef,
    pub readi8: FuncRef,
    pub readu16: FuncRef,
    pub readu8: FuncRef,

    pub write32: FuncRef,
    pub write16: FuncRef,
    pub write8: FuncRef,

    pub handle_rfe: FuncRef,
    pub handle_break: FuncRef,
}

impl Default for JIT {
    fn default() -> Self {
        // Set up JIT
        let mut flags = settings::builder();
        flags.set("opt_level", "none").unwrap();
        flags.set("unwind_info", "true").unwrap();

        let isa = cranelift::native::builder()
            .unwrap()
            .finish(settings::Flags::new(flags))
            .unwrap();
        let mut jit_builder = JITBuilder::with_isa(isa, default_libcall_names());
        jit_builder
            .symbol("read32", Memory::read32 as *const u8)
            .symbol("readi16", Memory::readi16 as *const u8)
            .symbol("readi8", Memory::readi8 as *const u8)
            .symbol("readu16", Memory::readu16 as *const u8)
            .symbol("readu8", Memory::readu8 as *const u8)
            .symbol("write32", Memory::write32 as *const u8)
            .symbol("write16", Memory::write16 as *const u8)
            .symbol("write8", Memory::write8 as *const u8)
            .symbol("handle_rfe", Cpu::handle_rfe as *const u8)
            .symbol("handle_break", Cpu::handle_break as *const u8);

        let mut module = JITModule::new(jit_builder);
        let ptr = module.target_config().pointer_type();

        let read32 = {
            let mut read32_sig = Signature::new(CallConv::SystemV);
            read32_sig.params.push(AbiParam::new(ptr));
            read32_sig.params.push(AbiParam::new(ptr));
            read32_sig.params.push(AbiParam::new(types::I32));
            read32_sig.returns.push(AbiParam::new(types::I32));

            module
                .declare_function("read32", Linkage::Import, &read32_sig)
                .expect("failed to link read32 function")
        };

        let readi16 = {
            let mut readi16_sig = Signature::new(CallConv::SystemV);
            readi16_sig.params.push(AbiParam::new(ptr));
            readi16_sig.params.push(AbiParam::new(ptr));
            readi16_sig.params.push(AbiParam::new(types::I32));
            readi16_sig.returns.push(AbiParam::new(types::I32));

            module
                .declare_function("readi16", Linkage::Import, &readi16_sig)
                .expect("failed to link readi16 function")
        };

        let readi8 = {
            let mut readi8_sig = Signature::new(CallConv::SystemV);
            readi8_sig.params.push(AbiParam::new(ptr));
            readi8_sig.params.push(AbiParam::new(ptr));
            readi8_sig.params.push(AbiParam::new(types::I32));
            readi8_sig.returns.push(AbiParam::new(types::I32));

            module
                .declare_function("readi8", Linkage::Import, &readi8_sig)
                .expect("failed to link readi8 function")
        };

        let readu16 = {
            let mut readu16_sig = Signature::new(CallConv::SystemV);
            readu16_sig.params.push(AbiParam::new(ptr));
            readu16_sig.params.push(AbiParam::new(ptr));
            readu16_sig.params.push(AbiParam::new(types::I32));
            readu16_sig.returns.push(AbiParam::new(types::I32));

            module
                .declare_function("readu16", Linkage::Import, &readu16_sig)
                .expect("failed to link readu16 function")
        };

        let readu8 = {
            let mut readu8_sig = Signature::new(CallConv::SystemV);
            readu8_sig.params.push(AbiParam::new(ptr));
            readu8_sig.params.push(AbiParam::new(ptr));
            readu8_sig.params.push(AbiParam::new(types::I32));
            readu8_sig.returns.push(AbiParam::new(types::I32));

            module
                .declare_function("readu8", Linkage::Import, &readu8_sig)
                .expect("failed to link readu8 function")
        };

        let write32 = {
            let mut write32_sig = Signature::new(CallConv::SystemV);
            write32_sig.params.push(AbiParam::new(ptr));
            write32_sig.params.push(AbiParam::new(ptr));
            write32_sig.params.push(AbiParam::new(types::I32));
            write32_sig.params.push(AbiParam::new(types::I32));

            module
                .declare_function("write32", Linkage::Import, &write32_sig)
                .expect("failed to link write32 function")
        };

        let write16 = {
            let mut write16_sig = Signature::new(CallConv::SystemV);
            write16_sig.params.push(AbiParam::new(ptr));
            write16_sig.params.push(AbiParam::new(ptr));
            write16_sig.params.push(AbiParam::new(types::I32));
            write16_sig.params.push(AbiParam::new(types::I32));

            module
                .declare_function("write16", Linkage::Import, &write16_sig)
                .expect("failed to link write16 function")
        };

        let write8 = {
            let mut write8_sig = Signature::new(CallConv::SystemV);
            write8_sig.params.push(AbiParam::new(ptr));
            write8_sig.params.push(AbiParam::new(ptr));
            write8_sig.params.push(AbiParam::new(types::I32));
            write8_sig.params.push(AbiParam::new(types::I32));

            module
                .declare_function("write8", Linkage::Import, &write8_sig)
                .expect("failed to link write8 function")
        };

        let handle_rfe = {
            let mut handle_rfe_sig = Signature::new(CallConv::SystemV);
            handle_rfe_sig.params.push(AbiParam::new(ptr));

            module
                .declare_function("handle_rfe", Linkage::Import, &handle_rfe_sig)
                .expect("failed to link handle_rfe function")
        };

        let handle_break = {
            let mut handle_break_sig = Signature::new(CallConv::SystemV);
            handle_break_sig.params.push(AbiParam::new(ptr));

            module
                .declare_function("handle_break", Linkage::Import, &handle_break_sig)
                .expect("failed to link handle_break function")
        };

        let fn_builder_ctx = FunctionBuilderContext::new();
        let ctx = module.make_context();
        let data_description = DataDescription::new();

        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(ptr));
        // sig.params.push(AbiParam::new(ptr));

        Self {
            module,
            fn_builder_ctx,
            data_description,
            ctx,
            basic_sig: sig,
            func_idx: 1,
            func_table: FunctionTable {
                read32,
                readi16,
                readi8,
                readu16,
                readu8,

                write32,
                write16,
                write8,

                handle_rfe,
                handle_break,
            },
        }
    }
}

pub fn hash_ops(ops: &[DecodedOp]) -> u64 {
    let mut hasher = RapidHasher::default();
    ops.hash(&mut hasher);
    hasher.finish()
}

#[bon::bon]
impl JIT {
    pub const FN_PARAMS: usize = 2;

    #[inline]
    pub fn pointer_type(&self) -> ir::Type {
        self.module.target_config().pointer_type()
    }

    pub fn get_func(&self, id: FuncId, func: Function, hash: u64) -> BlockFn {
        let code_ptr = self.module.get_finalized_function(id);
        let ptr = unsafe { std::mem::transmute::<*const u8, BlockFnPtr>(code_ptr) };
        BlockFn {
            fn_ptr: ptr,
            func: Arc::new(func),
            hash,
        }
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

    pub fn create_func_ref_table(&mut self, func: &mut Function) -> FuncRefTable {
        let read32 = self
            .module
            .declare_func_in_func(self.func_table.read32, func);
        let readi16 = self
            .module
            .declare_func_in_func(self.func_table.readi16, func);
        let readi8 = self
            .module
            .declare_func_in_func(self.func_table.readi8, func);
        let readu16 = self
            .module
            .declare_func_in_func(self.func_table.readu16, func);
        let readu8 = self
            .module
            .declare_func_in_func(self.func_table.readu8, func);
        let write32 = self
            .module
            .declare_func_in_func(self.func_table.write32, func);
        let write16 = self
            .module
            .declare_func_in_func(self.func_table.write16, func);
        let write8 = self
            .module
            .declare_func_in_func(self.func_table.write8, func);
        let handle_rfe = self
            .module
            .declare_func_in_func(self.func_table.handle_rfe, func);
        let handle_break = self
            .module
            .declare_func_in_func(self.func_table.handle_break, func);
        FuncRefTable {
            read32,
            readi16,
            readi8,
            readu16,
            readu8,
            write32,
            write16,
            write8,
            handle_rfe,
            handle_break,
        }
    }

    #[inline]
    pub fn create_fn_builder<'a>(&'a mut self, func: &'a mut Function) -> FunctionBuilder<'a> {
        FunctionBuilder::new(func, &mut self.fn_builder_ctx)
    }

    pub fn finish_function(
        &mut self,
        func_id: FuncId,
        func: Function,
        hash: u64,
    ) -> Result<BlockFn, Box<ModuleError>> {
        self.ctx.func = func.clone();
        if let Err(err) = self.ctx.replace_redundant_loads() {
            tracing::warn!(%err);
        };

        self.module.define_function(func_id, &mut self.ctx)?;

        self.module.finalize_definitions()?;

        unsafe extern "C" {
            // libunwind import
            fn __register_frame(fde: *const u8);
        }

        let compiled_code = &self.ctx.compiled_code().unwrap();
        let unwind_info = compiled_code.create_unwind_info(self.module.isa()).unwrap();
        if let Some(info) = unwind_info {
            match info {
                UnwindInfo::SystemV(info) => {
                    let endian = match self.module.isa().endianness() {
                        cranelift_codegen::ir::Endianness::Little => RunTimeEndian::Little,
                        cranelift_codegen::ir::Endianness::Big => RunTimeEndian::Big,
                    };
                    let mut frame_table = FrameTable::default();
                    let func_start = self.module.get_finalized_function(func_id);
                    let fde =
                        info.to_fde(gimli::write::Address::Constant(func_start as usize as u64)); // Create FDE.
                    let cie = self.module.isa().create_systemv_cie().unwrap();
                    let cie_id = frame_table.add_cie(cie);
                    frame_table.add_fde(cie_id, fde); // Add shared CIE.

                    // Write EH frame bytes.
                    let mut eh_frame = gimli::write::EhFrame(gimli::write::EndianVec::new(endian));
                    frame_table.write_eh_frame(&mut eh_frame).unwrap();
                    let mut eh_frame = eh_frame.0.into_vec();

                    // GCC expects a terminating "empty" length, so write a 0 length at the end of the table.
                    eh_frame.extend(&[0, 0, 0, 0]);

                    let eh_frame_bytes: &'static [u8] = Box::leak(eh_frame.into_boxed_slice());

                    #[cfg(target_os = "macos")]
                    unsafe {
                        // On macOS, `__register_frame` takes a pointer to a single FDE
                        let start = eh_frame_bytes.as_ptr();
                        let end = start.add(eh_frame_bytes.len());
                        let mut current = start;

                        // Walk all of the entries in the frame table and register them
                        while current < end {
                            let len = std::ptr::read::<u32>(current as *const u32) as usize;

                            // Skip over the CIE
                            if current != start {
                                __register_frame(current);
                            }

                            // Move to the next table entry (+4 because the length itself is not inclusive)
                            current = current.add(len + 4);
                        }
                        tracing::info!("macos: added unwind fde entries");
                    }

                    #[cfg(not(target_os = "macos"))]
                    unsafe {
                        __register_frame(eh_frame_bytes.as_ptr());
                    }
                }
                UnwindInfo::WindowsX64(_) => {
                    // FIXME implement this
                }
                unwind_info => unimplemented!("{:?}", unwind_info),
            };
        } else {
            tracing::warn!("unable to attach unwind info");
        }

        self.module.clear_context(&mut self.ctx);
        self.ctx.clear();

        let res = self.get_func(func_id, func, hash);
        Ok(res)
    }

    pub fn init_block(builder: &mut FunctionBuilder) -> Block {
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);
        block
    }

    #[builder]
    #[instrument(skip(builder))]
    pub fn emit_load_reg(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        idx: u8,
    ) -> (Value, Inst) {
        if idx == 0 {
            let inst = builder
                .pure()
                .UnaryImm(Opcode::Iconst, types::I32, Imm64::new(0))
                .0;
            let const0 = builder.single_result(inst);
            return (const0, inst);
        }
        let cpu_value = builder.block_params(block)[0];
        let offset = core::mem::offset_of!(Cpu, gpr);
        let offset =
            i32::try_from(offset + idx as usize * size_of::<u32>()).expect("offset overflow");
        let load = builder
            .pure()
            .Load(
                Opcode::Load,
                types::I32,
                MemFlags::new().with_aligned().with_notrap(),
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
                MemFlags::new().with_notrap().with_aligned(),
                Offset32::new(offset),
                cpu_ptr,
            )
            .0;
        let value = builder.single_result(load);

        (value, load)
    }

    #[inline]
    pub fn emit_any_load_from_cpu(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        type_: types::Type,
        offset: i32,
    ) -> (Value, Inst) {
        let cpu_ptr = builder.block_params(block)[0];
        let load = builder
            .pure()
            .Load(
                Opcode::Load,
                type_,
                MemFlags::new().with_notrap().with_aligned(),
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
    #[instrument(skip(builder, block), fields(reg=REG_STR[idx as usize], value))]
    pub fn emit_store_reg(
        builder: &mut FunctionBuilder<'_>,
        block: Block,
        idx: Reg,
        value: Value,
    ) -> Inst {
        let value_type = builder.type_of(value);
        debug_assert_ne!(value_type, types::I64, "got 64bit write to register");

        if idx == 0 {
            return builder.Nop();
        }
        const GPR: usize = const { core::mem::offset_of!(Cpu, gpr) };
        let cpu_ptr = builder.block_params(block)[0];
        let offset = i32::try_from(GPR + idx as usize * size_of::<u32>()).expect("offset overflow");

        builder
            .pure()
            .Store(
                Opcode::Store,
                value_type,
                MemFlags::new().with_notrap().with_aligned(),
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
                MemFlags::new().with_aligned().with_notrap(),
                Offset32::new(offset),
                value,
                cpu_ptr,
            )
            .0
    }

    #[builder]
    #[instrument(skip(builder, block, idx), fields(reg=idx, value))]
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

        let offset = match cop {
            0 => offset_of!(Cpu, cop0),
            1 => offset_of!(Cpu, cop1),
            2 => offset_of!(Cpu, _pad_cop2_gte),
            _ => panic!("invalid coprocessor"),
        };
        let offset = offset + idx * size_of::<u32>();

        tracing::info!(?offset, "write to coprocessor");
        JIT::emit_store_to_cpu(builder, block, value, offset as i32)
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
    pub fn apply_cache_updates(updates: &CacheUpdates, cache: &mut EntryCache) {
        for (id, update) in updates.registers.iter() {
            if let Some(value) = update.try_value() {
                cache.registers[*id as usize] = Some(value);
                tracing::trace!("updated ${}={:?}", REG_STR[*id as usize], value.value);
            }
        }
        // DONE: apply updates to hi and lo
        if let Some(hi) = &updates.hi {
            cache.hi = hi.try_value();
        }
        if let Some(lo) = &updates.lo {
            cache.lo = lo.try_value();
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
        cache: Option<&EntryCache>,
        delta_cycles: u64,
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
                let id = id as u8;
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

        const D_CLOCK_OFFSET: i32 = offset_of!(Cpu, d_clock) as i32;
        let (dcycles, loadcycles) =
            JIT::emit_any_load_from_cpu(builder, block, types::I64, D_CLOCK_OFFSET);
        let (dcycles, addcycles) = builder.inst(|f| {
            f.pure()
                .BinaryImm64(
                    Opcode::IaddImm,
                    types::I64,
                    Imm64::new(delta_cycles as i64),
                    dcycles,
                )
                .0
        });
        let store_clock = JIT::emit_store_to_cpu(builder, block, dcycles, D_CLOCK_OFFSET);
        instructions.push(loadcycles);
        instructions.push(addcycles);
        instructions.push(store_clock);

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
    #[deprecated]
    pub fn emit_map_address_to_host(
        _fn_builder: &mut FunctionBuilder<'_>,
        _ptr_type: Type,
        _mem_map_ptr: Value,
        _address: Value,
    ) -> (Value, [Inst; 12]) {
        unreachable!("called deprecated function `emit_map_address_to_host`")
    }

    pub fn emit_store_pc(fn_builder: &mut FunctionBuilder<'_>, block: Block, pc: Value) -> Inst {
        // tracing::info!("write to pc");

        JIT::emit_store_to_cpu(fn_builder, block, pc, offset_of!(Cpu, pc) as i32)
    }
}

type BlockFnPtr = fn(*mut Cpu, *mut Memory);

/// # BlockFn
///
/// compiled basic block of emulation.
///
/// side effects:
/// - will update pc
/// - will reset cpu's delta clock before running
/// - will update cpu's delta clock
/// - may trigger interrupts
#[derive(derive_more::Debug, Clone)]
#[debug("{:?}:{}", self.fn_ptr, self.hash)]
pub struct BlockFn {
    pub fn_ptr: BlockFnPtr,
    #[debug(skip)]
    pub func: Arc<Function>,
    pub hash: u64,
}

type BlockFnArgs<'a> = (&'a mut Emu, bool);

impl BlockFn {
    fn call_block(&self, (emu, instrument): BlockFnArgs) {
        // reset delta clock before running
        emu.cpu.d_clock = 0;

        if instrument {
            (self.fn_ptr)
                .instrument(tracing::info_span!("fn", addr = ?self.fn_ptr))
                .inner()(ptr::from_mut(&mut emu.cpu), ptr::from_mut(&mut emu.mem))
        } else {
            (self.fn_ptr)(ptr::from_mut(&mut emu.cpu), ptr::from_mut(&mut emu.mem))
        };

        IO::run_timer_pipeline(&mut emu.cpu, &mut emu.mem);
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

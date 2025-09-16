use crate::{IntoInst, cranelift_bs::*};
use std::{
    borrow::Cow,
    collections::HashMap,
    ops::{Index, IndexMut},
};

use bon::{Builder, bon, builder};
use color_eyre::eyre::{Context, eyre};

use cranelift::codegen::{bitset::ScalarBitSet, cursor::Cursor, entity::EntityList};
use petgraph::{
    algo::is_cyclic_directed,
    dot::{Config, Dot},
    prelude::*,
    visit::{DfsEvent, depth_first_search},
};
use tracing::{Level, info_span, instrument, span};

use crate::{
    Emu, FnBuilderExt,
    cpu::{Cpu, ops::prelude::*},
    jit::{BlockPage, JIT},
    memory::{MEM_MAP, Memory, PhysAddr},
};

#[derive(Debug, Builder, Clone)]
pub struct JitSummary {
    pub function: Option<Function>,
}

#[derive(Debug, Builder, Clone)]
pub struct SummarizeDeps<'a> {
    function: Option<&'a Function>,
}

pub trait SummarizeJit {
    fn summarize(deps: SummarizeDeps) -> Self;
}

impl SummarizeJit for () {
    fn summarize(_: SummarizeDeps) -> Self {}
}

impl SummarizeJit for JitSummary {
    fn summarize(deps: SummarizeDeps) -> Self {
        Self::builder()
            .maybe_function(deps.function.cloned())
            .build()
    }
}

pub struct EmitBlockSummary;

#[derive(Default, Clone, Debug, Hash)]
pub struct EntryCache {
    pub registers: [Option<CachedValue>; 32],
    pub const_one: Option<Value>,
    pub const_zero_i64: Option<Value>,
    pub hi: Option<CachedValue>,
    pub lo: Option<CachedValue>,
}

#[derive(Default, Clone, Debug)]
pub struct CacheDependency {
    pub registers: ScalarBitSet<u32>,
    pub hi: bool,
    pub lo: bool,
    pub const_one: bool,
    pub const_zero_i64: bool,
}

impl CacheDependency {
    fn new() -> Self {
        CacheDependency {
            registers: ScalarBitSet::default(),
            hi: false,
            lo: false,
            const_one: false,
            const_zero_i64: false,
        }
    }
    fn from_cache_entry(entry: &EntryCache) -> Self {
        let mut set = ScalarBitSet::default();
        for (idx, value) in entry.registers.iter().enumerate() {
            if value.is_some() {
                set.insert(idx as u8);
            }
        }
        CacheDependency {
            registers: set,
            hi: entry.hi.is_some(),
            lo: entry.lo.is_some(),
            const_one: entry.const_one.is_some(),
            const_zero_i64: entry.const_zero_i64.is_some(),
        }
    }
}

#[bon]
impl Emu {
    pub fn load_bios(&mut self) -> color_eyre::Result<()> {
        self.boot.load_bios(&mut self.mem)?;
        Ok(())
    }
    pub fn jump_to_bios(&mut self) {
        self.cpu.jump_to_bios();
    }
    pub fn step_jit(&mut self) -> color_eyre::Result<()> {
        self.step_jit_summarize()
    }
    pub fn step_jit_summarize<T: SummarizeJit>(&mut self) -> color_eyre::Result<T> {
        let initial_address = self.cpu.pc;

        let ptr_type = self.jit.pointer_type();

        self.jit.apply_dirty_pages(initial_address);

        // try cache first
        let cached =
            self.jit
                .use_cached_function(initial_address, &mut self.cpu, &mut self.mem, &MEM_MAP);
        if cached {
            return Ok(T::summarize(SummarizeDeps::builder().build()));
        }

        // collect blocks in function
        let mut cfg = Graph::with_capacity(20, 0);
        let entry_idx = cfg.add_node(BasicBlock::new(self.cpu.pc, 0));
        let mut blocks = fetch(
            FetchParams::builder()
                .pc(self.cpu.pc)
                .mem(&self.mem)
                .cfg(cfg)
                .current_node_index(entry_idx)
                .build(),
        )?;

        tracing::trace!(cfg.cycle = is_cyclic_directed(&blocks.cfg));
        tracing::info!(
            "\n{:?}",
            Dot::with_config(&blocks.cfg, &[Config::EdgeNoLabel, Config::NodeIndexLabel])
        );

        for node in blocks.cfg.node_indices() {
            let block = &blocks.cfg[node];
            for op in blocks.ops_for(block).iter() {
                if let Some(address) = op.invalidates_cache_at() {
                    let page = BlockPage::new(address);
                    self.jit.dirty_pages.insert(page);
                }
            }
        }

        let (func_id, mut func) = self.jit.create_function(initial_address)?;
        let mut fn_builder = self.jit.create_fn_builder(&mut func);

        let mut dfs = Dfs::new(&blocks.cfg, entry_idx);
        while let Some(node) = dfs.next(&blocks.cfg) {
            blocks.cfg[node].clif_block = Some(fn_builder.create_block());

            // if tracing::enabled!(Level::TRACE) {
            //     let _span = trace_span!("trace_cfg").entered();
            //     tracing::trace!(cfg.node = ?blocks.cfg[node].clif_block);
            //     for op in blocks.cfg[node].ops.iter() {
            //         tracing::trace!("    {op}");
            //     }

            //     tracing::trace!(
            //         to = ?&blocks.cfg
            //             .neighbors_directed(node, Direction::Outgoing)
            //             .map(|n| blocks.cfg[n].clif_block)
            //             .collect::<Vec<_>>(),
            //         "    branch"
            //     );
            // };
        }

        let mut state_map = BlockStateMap::default();
        state_map.insert(
            entry_idx,
            BlockState {
                cache: EntryCache::default(),
                deps: None,
                basic_block: blocks.cfg[entry_idx].clone(),
                node: entry_idx,
                ptr_type,
            },
        );
        let mut prev_node = entry_idx;

        depth_first_search(&blocks.cfg, Some(entry_idx), |event| match event {
            DfsEvent::TreeEdge(a, _) => {
                prev_node = a;
            }
            DfsEvent::BackEdge(_, b) => {
                prev_node = b;
            }
            DfsEvent::Discover(node, _) => {
                let basic_block = &blocks.cfg[node];
                let cranelift_block = basic_block.clif_block();

                let prev_state = state_map.get(prev_node).unwrap();
                let state = state_map.get(node).unwrap_or(prev_state).clone();
                let mut state = BlockState {
                    basic_block: basic_block.clone(),
                    deps: state
                        .deps
                        .or(Some(CacheDependency::from_cache_entry(&prev_state.cache))),
                    node,
                    ..state
                };

                tracing::info!(?node, ?cranelift_block, "compiling...");

                let _ = Self::emit_block()
                    .state_map(&mut state_map)
                    .fn_builder(&mut fn_builder)
                    .cpu(&mut self.cpu)
                    .ptr_type(ptr_type)
                    .node(node)
                    .cfg(&blocks.cfg)
                    .ops(blocks.ops_for(basic_block))
                    .call();

                state_map.insert(node, state);
                // prev_register_cache = node;
            }
            _ => {}
        });

        let _span = info_span!("jit_comp", pc = %format!("0x{:08X}", initial_address)).entered();
        fn_builder.seal_all_blocks();

        Self::close_function(fn_builder);
        let summary = T::summarize(SummarizeDeps::builder().function(&func).build());
        self.jit.finish_function(func_id, func)?;

        let function = self.jit.get_func(func_id);
        tracing::info!("compiled function: {:?}", function.0);
        function(&mut self.cpu, &mut self.mem, true, &MEM_MAP);
        tracing::info!("{:#?}", self.cpu);
        self.jit.block_map.insert(initial_address, function);

        Ok(summary)
    }
    pub fn close_function(fn_builder: FunctionBuilder) {
        tracing::trace!("closing function");
        fn_builder.finalize();
    }
    #[builder]
    pub fn emit_block<'a, 'b>(
        ops: &[DecodedOp],
        state_map: &'a mut BlockStateMap,
        fn_builder: &'a mut FunctionBuilder<'b>,
        ptr_type: types::Type,
        cfg: &'a Graph<BasicBlock, ()>,
        node: NodeIndex,
        cpu: &mut Cpu,
    ) -> EmitBlockSummary {
        // fn_builder.seal_block(cranelift_block);
        let block_state = &mut state_map[node];
        let cranelift_block = block_state.basic_block.clif_block();
        fn_builder.switch_to_block(cranelift_block);

        fn_builder.append_block_param(cranelift_block, ptr_type);
        fn_builder.append_block_param(cranelift_block, ptr_type);
        fn_builder.append_block_param(cranelift_block, ptr_type);
        for reg in block_state.deps.as_ref().unwrap().registers.iter() {
            let reg = reg as usize;
            let dirty = block_state.cache.registers[reg]
                .map(|reg| reg.dirty)
                .unwrap_or(false);
            block_state.cache.registers[reg] = Some(CachedValue {
                value: fn_builder.append_block_param(cranelift_block, types::I32),
                dirty,
            });
        }

        collect_instructions(node, fn_builder, state_map, cfg, ops);

        EmitBlockSummary
    }

    pub fn run(&mut self) -> color_eyre::Result<()> {
        loop {
            self.step_jit()?;
            tracing::info!("step: pc=0x{:08X}", self.cpu.pc);
        }
    }
}

#[derive(Debug, Clone)]
struct FetchSummary {
    cfg: Graph<BasicBlock, ()>,
    decoded_ops: Vec<DecodedOp>,
}

impl FetchSummary {
    pub fn ops_for(&self, basic_block: &BasicBlock) -> &[DecodedOp] {
        &self.decoded_ops[basic_block.start..(basic_block.start + basic_block.len)]
    }
}

#[derive(Builder, Debug)]
struct FetchParams<'a> {
    pc: u32,
    mem: &'a Memory,
    #[builder(default)]
    cfg: Graph<BasicBlock, ()>,
    current_node_index: Option<NodeIndex<u32>>,
    #[builder(default)]
    mapped: Cow<'a, HashMap<u32, NodeIndex>>,
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    // pub ops: Vec<DecodedOp>,
    pub start: usize,
    pub len: usize,
    pub address: u32,
    pub clif_block: Option<Block>,
}

impl BasicBlock {
    pub fn new(address: u32, start: usize) -> Self {
        Self {
            address,
            clif_block: None,
            start,
            len: 0,
        }
    }
    pub fn set_block(self, clif_block: Block) -> Self {
        Self {
            clif_block: Some(clif_block),
            ..self
        }
    }
    pub fn clif_block(&self) -> Block {
        self.clif_block
            .expect("basic block has no attached cranelift block!")
    }
}

fn find_first_value(than: u32, mapped: &HashMap<u32, NodeIndex>) -> Option<u32> {
    mapped.keys().filter(|addr| addr < &&than).max().cloned()
}

fn fetch(params: FetchParams<'_>) -> color_eyre::Result<FetchSummary> {
    let FetchParams {
        pc: initial_pc,
        mem,
        mut cfg,
        current_node_index,
        mut mapped,
    } = params;

    let entry_node =
        current_node_index.unwrap_or_else(|| cfg.add_node(BasicBlock::new(initial_pc, 0)));

    mapped.to_mut().insert(initial_pc, entry_node);

    struct State {
        current_node: NodeIndex,
        pc: u32,
        from_jump: bool,
    }

    let mut stack = vec![State {
        current_node: entry_node,
        pc: initial_pc,
        from_jump: true,
    }];
    stack.reserve(256);

    let mut ops = Vec::with_capacity(512);

    while let Some(state) = stack.pop() {
        let _span = span!(Level::DEBUG, "fetch", pc = %format!("0x{:04X}", state.pc), node = ?state.current_node.index()).entered();

        if state.from_jump {
            tracing::trace!("=== block({:?}) ===", state.current_node);
        }

        let op = mem.read::<u32>(PhysAddr(state.pc));
        let op = OpCode(op);
        let op = DecodedOp::try_from(op).wrap_err(eyre!("failed at pc 0x{:08X}", state.pc))?;
        cfg[state.current_node].len += 1;
        ops.push(op);

        tracing::trace!(op = format!("{op}"));

        match op.is_block_boundary() {
            Some(BoundaryType::Function { .. }) => {
                continue;
            }
            None => {
                const NOP_TOLERANCE: usize = 64;
                let len = cfg[state.current_node].len;
                if len > NOP_TOLERANCE {
                    let slice = &ops[(len - NOP_TOLERANCE)..];
                    if slice.iter().all(|op| matches!(op, DecodedOp::NOP(_))) {
                        return Err(eyre!("reading empty program. jit canceled."));
                    }
                }
                stack.push(State {
                    pc: state.pc + 4,
                    from_jump: false,
                    ..state
                });
            }
            Some(BoundaryType::Block { offset }) => {
                let new_address = offset.calculate_address(state.pc);

                if let Some(cached_node) = mapped.get(&new_address) {
                    cfg.add_edge(state.current_node, *cached_node, ());
                    continue;
                }

                let next_node = cfg.add_node(BasicBlock::new(new_address, ops.len()));
                cfg.add_edge(state.current_node, next_node, ());
                mapped.to_mut().insert(new_address, next_node);

                tracing::trace!("cfg.link {:?} -> {:?}", state.current_node, next_node);
                // tracing::debug!(offset, "jump in block");
                stack.push(State {
                    current_node: next_node,
                    pc: new_address,
                    from_jump: true,
                });
            }
            Some(BoundaryType::BlockSplit { lhs, rhs }) => {
                tracing::trace!(offsets = ?[lhs, rhs], delay_hazard = %op, "potential split in block");

                let offsets = [rhs, lhs];
                for offset in offsets.into_iter() {
                    let new_address = offset.calculate_address(state.pc);

                    if let Some(cached_node) = mapped.get(&new_address) {
                        cfg.add_edge(state.current_node, *cached_node, ());
                        continue;
                    }

                    let next_node = cfg.add_node(BasicBlock::new(new_address, ops.len()));
                    cfg.add_edge(state.current_node, next_node, ());
                    mapped.to_mut().insert(new_address, next_node);

                    tracing::trace!(
                        ?offset,
                        "cfg.link {:?} -> {:?}",
                        state.current_node,
                        next_node
                    );

                    stack.push(State {
                        current_node: next_node,
                        pc: new_address,
                        from_jump: true,
                    })
                }
            }
        };
    }

    Ok(FetchSummary {
        cfg,
        decoded_ops: ops,
    })
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct CachedValue {
    pub dirty: bool,
    pub value: Value,
}

#[derive(Builder)]
pub struct EmitCtx<'a, 'b> {
    pub fn_builder: &'a mut FunctionBuilder<'b>,
    pub state_map: &'a mut BlockStateMap,
    pub ptr_type: types::Type,
    pub node: NodeIndex,
    pub pc: u32,
    pub cfg: &'a Graph<BasicBlock, ()>,
}

impl<'a, 'b> EmitCtx<'a, 'b> {
    pub fn block(&self) -> &BasicBlock {
        &self.cfg[self.node]
    }
    #[instrument(skip(self))]
    pub fn next_at(&self, idx: usize) -> &BasicBlock {
        let idx = self
            .cfg
            .neighbors_directed(self.node, Direction::Outgoing)
            .nth(idx)
            .unwrap();
        &self.cfg[idx]
    }

    pub fn state(&self) -> &BlockState {
        &self.state_map[self.node]
    }

    pub fn state_mut(&mut self) -> &mut BlockState {
        &mut self.state_map[self.node]
    }

    pub fn cache(&self) -> &EntryCache {
        &self.state().cache
    }

    pub fn cache_mut(&mut self) -> &mut EntryCache {
        &mut self.state_mut().cache
    }

    #[instrument(skip(fn_builder, self))]
    fn out_params(&self, to: NodeIndex, fn_builder: &mut FunctionBuilder) -> Vec<BlockArg> {
        // tracing::trace!("{:#?}", self.deps_map);
        let next_block_deps: Option<_> =
            self.state_map.get(to).and_then(|state| state.deps.as_ref());

        let mut args = self.emulator_params();
        if let Some(next_block_deps) = next_block_deps {
            let iter = next_block_deps
                .registers
                .iter()
                .flat_map(|register| self.cache().registers[register as usize])
                .map(|value| value.value)
                .map(BlockArg::Value);
            args.extend(iter);
        } else {
            args.extend(
                self.cache()
                    .registers
                    .iter()
                    .flatten()
                    .cloned()
                    .map(|value| value.value)
                    .map(BlockArg::Value),
            );
        }
        args
    }
    pub fn neighbour_count(&self) -> usize {
        self.cfg
            .neighbors_directed(self.node, Direction::Outgoing)
            .count()
    }
    pub fn emulator_params(&self) -> Vec<BlockArg> {
        vec![
            BlockArg::Value(self.cpu()),
            BlockArg::Value(self.memory()),
            BlockArg::Value(self.mem_map()),
        ]
    }

    pub fn cpu(&self) -> Value {
        let block = self.block().clif_block();
        self.fn_builder.block_params(block)[0]
    }
    pub fn memory(&self) -> Value {
        let block = self.cfg[self.node].clif_block();
        self.fn_builder.block_params(block)[1]
    }
    pub fn mem_map(&self) -> Value {
        let block = self.block().clif_block();
        self.fn_builder.block_params(block)[2]
    }
    pub fn inst<R: IntoInst>(&mut self, f: impl Fn(&mut FunctionBuilder) -> R) -> (Value, Inst) {
        self.fn_builder.inst(f)
    }
    pub fn emit_get_one(&mut self) -> (Value, Inst) {
        match self.cache().const_one {
            Some(one) => (one, self.fn_builder.Nop()),
            None => {
                let (one, iconsti32) = self.fn_builder.inst(|f| {
                    f.ins()
                        .UnaryImm(Opcode::Iconst, types::I32, Imm64::new(1))
                        .0
                });
                self.cache_mut().const_one = Some(one);
                (one, iconsti32)
            }
        }
    }
    pub fn emit_get_zero_i64(&mut self) -> (Value, Inst) {
        match self.cache().const_zero_i64 {
            Some(zero_i64) => (zero_i64, self.fn_builder.Nop()),
            None => {
                let (zero_i64, iconsti32) = self.fn_builder.inst(|f| {
                    f.ins()
                        .UnaryImm(Opcode::Iconst, types::I32, Imm64::new(1))
                        .0
                });
                self.cache_mut().const_zero_i64 = Some(zero_i64);
                (zero_i64, iconsti32)
            }
        }
    }
    pub fn emit_get_zero(&mut self) -> (Value, Inst) {
        self.emit_get_register(0)
    }
    pub fn emit_get_hi(&mut self) -> (Value, Inst) {
        let block = self.block().clif_block();
        match self.cache().hi {
            Some(hi) => (hi.value, self.fn_builder.Nop()),
            None => {
                let (hi, loadhi) = JIT::emit_load_hi()
                    .builder(self.fn_builder)
                    .block(block)
                    .call();
                self.cache_mut().hi = Some(CachedValue {
                    dirty: false,
                    value: hi,
                });
                (hi, loadhi)
            }
        }
    }
    pub fn emit_get_lo(&mut self) -> (Value, Inst) {
        let block = self.block().clif_block();
        match self.cache_mut().lo {
            Some(lo) => (lo.value, self.fn_builder.Nop()),
            None => {
                let (lo, loadlo) = JIT::emit_load_lo()
                    .builder(self.fn_builder)
                    .block(block)
                    .call();
                self.cache_mut().lo = Some(CachedValue {
                    dirty: false,
                    value: lo,
                });
                (lo, loadlo)
            }
        }
    }
    pub fn emit_get_register(&mut self, id: usize) -> (Value, Inst) {
        let block = self.block().clif_block();
        match self.cache_mut().registers[id] {
            Some(value) => (value.value, self.fn_builder.Nop()),
            None => {
                let (value, loadreg) = JIT::emit_load_reg()
                    .builder(self.fn_builder)
                    .block(block)
                    .idx(id)
                    .call();
                self.cache_mut().registers[id] = Some(CachedValue {
                    dirty: false,
                    value,
                });
                (value, loadreg)
            }
        }
    }
    pub fn emit_get_cop_register(
        &mut self,
        fn_builder: &mut FunctionBuilder,
        cop: u8,
        reg: usize,
    ) -> (Value, Inst) {
        JIT::emit_load_cop_reg()
            .builder(fn_builder)
            .block(self.block().clif_block())
            .idx(reg)
            .cop(cop)
            .call()
    }
    pub fn update_cache_immediate(&mut self, id: usize, value: Value) {
        self.cache_mut().registers[id] = Some(CachedValue {
            dirty: false,
            value,
        });
    }
    pub fn emit_map_address_to_host(&mut self, address: Value) -> (Value, [Inst; 12]) {
        let mem_map = self.mem_map();
        JIT::emit_map_address_to_host()
            .fn_builder(self.fn_builder)
            .ptr_type(self.ptr_type)
            .address(address)
            .mem_map_ptr(mem_map)
            .call()
    }
    pub fn emit_map_address_to_physical(&mut self, address: Value) -> (Value, Inst) {
        JIT::emit_map_address_to_physical()
            .fn_builder(self.fn_builder)
            .address(address)
            .call()
    }
    pub fn emit_store_pc(&mut self, value: Value) -> Inst {
        JIT::emit_store_pc(self.fn_builder, self.block().clif_block(), value)
    }
    pub fn emit_store_register(&mut self, reg: usize, value: Value) {
        let block = self.block().clif_block();
        JIT::emit_store_reg()
            .builder(self.fn_builder)
            .block(block)
            .idx(reg)
            .value(value)
            .call();
    }
    pub fn emit_store_cop_register(&mut self, cop: u8, reg: usize, value: Value) {
        let block = self.block().clif_block();
        JIT::emit_store_cop_reg()
            .builder(self.fn_builder)
            .block(block)
            .idx(reg)
            .value(value)
            .cop(cop)
            .call();
    }
}

#[derive(Builder, derive_more::Debug, Default)]
#[builder(finish_fn(vis = "", name = build_internal))]
pub struct EmitSummary {
    #[builder(field = Vec::new())]
    pub register_updates: Vec<(usize, CachedValue)>,

    #[debug(skip)]
    #[builder(into)]
    pub instructions: Vec<ClifInstruction>,

    pub pc_update: Option<u32>,

    #[builder(with = |value: Value| CachedValue {
        dirty: true,
        value
    })]
    pub hi: Option<CachedValue>,

    #[builder(with = |value: Value| CachedValue {
        dirty: true,
        value
    })]
    pub lo: Option<CachedValue>,
}

impl<S: emit_summary_builder::State> EmitSummaryBuilder<S> {
    pub fn register_updates(mut self, values: impl IntoIterator<Item = (usize, Value)>) -> Self {
        self.register_updates.extend(
            values
                .into_iter()
                .map(|(reg, value)| (reg, CachedValue { dirty: true, value })),
        );
        self
    }
}

impl<S: emit_summary_builder::IsComplete> EmitSummaryBuilder<S> {
    pub fn build(self, fn_builder: &FunctionBuilder) -> EmitSummary {
        if cfg!(debug_assertions) {
            for (_, value) in &self.register_updates {
                assert_eq!(
                    fn_builder.type_of(value.value),
                    types::I32,
                    "emit summary invalid type for register value"
                );
            }
        }
        self.build_internal()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ClifInstruction {
    pub queue_type: ClifInstructionQueueType,
    pub instruction: Inst,
}

#[derive(Debug, Clone, Copy)]
pub enum ClifInstructionQueueType {
    Now,
    Delayed(u32),
    Bottom,
}

impl ClifInstructionQueueType {
    pub fn tick(&mut self) -> bool {
        match self {
            ClifInstructionQueueType::Now => true,
            ClifInstructionQueueType::Delayed(by) => {
                *by = by.saturating_sub(1);
                *by == 0
            }
            ClifInstructionQueueType::Bottom => false,
        }
    }
}

#[derive(derive_more::Debug, Clone)]
pub struct BlockState {
    cache: EntryCache,
    deps: Option<CacheDependency>,
    #[debug(skip)]
    basic_block: BasicBlock,
    ptr_type: types::Type,
    node: NodeIndex,
}

fn collect_instructions<'a>(
    node: NodeIndex,
    fn_builder: &mut FunctionBuilder,
    state_map: &mut BlockStateMap,
    cfg: &Graph<BasicBlock, ()>,
    ops: &[DecodedOp],
) {
    let mut queue = Vec::with_capacity(32);
    let mut final_instruction = None;

    let ptr_type = state_map[node].ptr_type;
    let address = state_map[node].basic_block.address;

    for (idx, op) in ops.iter().enumerate() {
        let pc = address + idx as u32 * 4;
        let summary = op.emit_ir(EmitCtx {
            fn_builder,
            ptr_type,
            state_map,
            pc,
            cfg,
            node,
        });
        let block_state = state_map.get_mut(node).unwrap();
        for inst in summary.instructions {
            match inst.queue_type {
                ClifInstructionQueueType::Now => {
                    fn_builder
                        .cursor()
                        .at_last_inst(block_state.basic_block.clif_block())
                        .insert_inst(inst.instruction);
                }
                ClifInstructionQueueType::Bottom => {
                    final_instruction = Some(inst.instruction);
                }
                ClifInstructionQueueType::Delayed(_) => {
                    queue.push(inst);
                }
            }
            queue.retain_mut(|inst| {
                let ready = inst.queue_type.tick();
                if ready {
                    fn_builder
                        .cursor()
                        .at_last_inst(block_state.basic_block.clif_block())
                        .insert_inst(inst.instruction);
                    false
                } else {
                    true
                }
            });
        }
    }
    let final_instruction = final_instruction.unwrap_or_else(|| {
        fn_builder
            .ins()
            .MultiAry(Opcode::Return, types::INVALID, EntityList::new())
            .0
    });

    let block_state = state_map.get(node).unwrap();
    fn_builder
        .cursor()
        .at_bottom(block_state.basic_block.clif_block())
        .insert_inst(final_instruction);
}

#[derive(Debug, Default, Clone)]
pub struct BlockStateMap {
    pub inner: Vec<Option<BlockState>>,
}

impl BlockStateMap {
    pub fn get(&self, index: NodeIndex) -> Option<&BlockState> {
        self.inner[index.index()].as_ref()
    }
    pub fn get_mut(&mut self, index: NodeIndex) -> Option<&mut BlockState> {
        self.inner[index.index()].as_mut()
    }
    pub fn insert(&mut self, index: NodeIndex, value: BlockState) {
        self.inner[index.index()] = Some(value);
    }
}

impl Index<NodeIndex> for BlockStateMap {
    type Output = BlockState;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl IndexMut<NodeIndex> for BlockStateMap {
    fn index_mut(&mut self, index: NodeIndex) -> &mut BlockState {
        self.get_mut(index).unwrap()
    }
}

#[inline]
pub const fn now(inst: Inst) -> ClifInstruction {
    ClifInstruction {
        instruction: inst,
        queue_type: ClifInstructionQueueType::Now,
    }
}

#[inline]
pub const fn delayed(by: u32, inst: Inst) -> ClifInstruction {
    ClifInstruction {
        instruction: inst,
        queue_type: ClifInstructionQueueType::Delayed(by),
    }
}

#[inline]
pub const fn bottom(inst: Inst) -> ClifInstruction {
    ClifInstruction {
        queue_type: ClifInstructionQueueType::Bottom,
        instruction: inst,
    }
}

trait BasicBlockIndexOps {
    fn for_block(&self, block: &BasicBlock) -> &[DecodedOp];
}

impl BasicBlockIndexOps for &[DecodedOp] {
    fn for_block(&self, block: &BasicBlock) -> &[DecodedOp] {
        &self[block.start..(block.start + block.len)]
    }
}

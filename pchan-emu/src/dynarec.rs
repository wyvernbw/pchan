use cranelift::{
    codegen::{
        bitset::ScalarBitSet,
        ir::{BlockArg, Function},
    },
    frontend::FuncInstBuilder,
    prelude::*,
};
use std::{borrow::Cow, collections::HashMap};

use bon::{Builder, bon, builder};
use color_eyre::eyre::{Context, eyre};

use petgraph::{
    algo::is_cyclic_directed,
    dot::{Config, Dot},
    prelude::*,
    visit::{DfsEvent, depth_first_search},
};
use tracing::{Level, info_span, instrument, span};

use crate::{
    Emu, FnBuilderExt,
    cpu::{
        Cpu,
        ops::{Hazard, prelude::*},
    },
    jit::{BlockPage, CacheUpdates, JIT},
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

#[derive(Clone, Debug)]
pub struct HazardEntry<'a> {
    hazard: Hazard<'a>,
    cache: EntryCache,
    node: NodeIndex,
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
        let entry_idx = cfg.add_node(BasicBlock::new(self.cpu.pc));
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
            for op in blocks.cfg[node].ops.iter() {
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

        let entry_block = blocks.cfg[entry_idx].clif_block();

        let mut deps_map = hash_map! {
            entry_block => CacheDependency::new()
        };

        let mut cache_map = hash_map! {
            entry_idx => EntryCache::default()
        };
        let mut hazards_map = hash_map! {
            entry_idx => Vec::<HazardEntry<'_>>::with_capacity(32)
        };
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

                let prev_register_cache = cache_map.get(&prev_node).unwrap();
                let prev_hazards = hazards_map.get(&prev_node).unwrap();
                let mut reg_cache = cache_map.get(&node).unwrap_or(prev_register_cache).clone();
                let mut hazards = hazards_map.get(&node).unwrap_or(prev_hazards).clone();

                tracing::info!(?node, ?cranelift_block, "compiling...");

                let deps = deps_map
                    .entry(blocks.cfg[node].clif_block())
                    .or_insert(CacheDependency::from_cache_entry(prev_register_cache))
                    .clone();

                let _ = Self::emit_block()
                    .fn_builder(&mut fn_builder)
                    .cranelift_block(cranelift_block)
                    .cpu(&mut self.cpu)
                    .ptr_type(ptr_type)
                    .node(node)
                    .cfg(&blocks.cfg)
                    .deps(&deps)
                    .register_cache(&mut reg_cache)
                    .deps_map(&deps_map)
                    .hazards(&mut hazards)
                    .call();

                cache_map.insert(node, reg_cache);
                hazards_map.insert(node, hazards);
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
    pub fn emit_block<'a>(
        fn_builder: &mut FunctionBuilder<'_>,
        cranelift_block: Block,
        ptr_type: types::Type,
        cfg: &'a Graph<BasicBlock, ()>,
        node: NodeIndex,
        register_cache: &mut EntryCache,
        deps: &CacheDependency,
        deps_map: &HashMap<Block, CacheDependency>,
        hazards: &mut Vec<HazardEntry<'a>>,
        cpu: &mut Cpu,
    ) -> EmitBlockSummary {
        // fn_builder.seal_block(cranelift_block);
        fn_builder.switch_to_block(cranelift_block);
        fn_builder.append_block_param(cranelift_block, ptr_type);
        fn_builder.append_block_param(cranelift_block, ptr_type);
        fn_builder.append_block_param(cranelift_block, ptr_type);
        for reg in deps.registers.iter() {
            let reg = reg as usize;
            let dirty = register_cache.registers[reg]
                .map(|reg| reg.dirty)
                .unwrap_or(false);
            register_cache.registers[reg] = Some(CachedValue {
                value: fn_builder.append_block_param(cranelift_block, types::I32),
                dirty,
            });
        }
        let mut summary_queue: Option<EmitSummary> = None;
        let basic_block = &cfg[node];

        for (idx, op) in basic_block.ops.iter().enumerate() {
            // tracing::info!(op = %op);

            let hazard = Self::emit_op()
                .fn_builder(fn_builder)
                .ptr_type(ptr_type)
                .node(node)
                .deps_map(deps_map)
                .cfg(cfg)
                .register_cache(register_cache)
                .op(op)
                .idx(idx)
                .summary_queue(&mut summary_queue)
                .basic_block(basic_block)
                .hazards(hazards)
                .call();

            if let Some(hazard) = hazard {
                hazards.push(HazardEntry {
                    hazard,
                    cache: register_cache.clone(),
                    node,
                });
            }
        }

        EmitBlockSummary
    }

    #[builder]
    #[instrument(name = "emit", skip_all, fields(%op, ))]
    pub fn emit_op<'a>(
        fn_builder: &mut FunctionBuilder<'_>,
        ptr_type: types::Type,
        cfg: &Graph<BasicBlock, ()>,
        node: NodeIndex,
        register_cache: &mut EntryCache,
        deps_map: &HashMap<Block, CacheDependency>,
        op: &'a DecodedOp,
        basic_block: &BasicBlock,
        idx: usize,
        summary_queue: &mut Option<EmitSummary>,
        hazards: &mut Vec<HazardEntry<'_>>,
    ) -> Option<Hazard<'a>> {
        // op instructions
        let summary = op.emit_ir(
            EmitCtx::builder()
                .ptr_type(ptr_type)
                .cache(register_cache)
                .pc(basic_block.address + idx as u32 * 4)
                .node(node)
                .cfg(cfg)
                .deps_map(deps_map)
                .fn_builder(fn_builder)
                .build(),
        );

        // apply previous op's delayed cache updates
        if let Some(summary) = summary_queue.take() {
            // flush_updates();
            JIT::apply_cache_updates()
                .updates(CacheUpdates::new(&summary))
                .cache(register_cache)
                .call();
        }

        // apply this op's immediate updates
        if let Some(summary) = summary {
            JIT::apply_cache_updates()
                .updates(CacheUpdates::new(&summary))
                .cache(register_cache)
                .call();
            *summary_queue = Some(summary);
        }

        // emit potential hazards from past instructions
        let pc = basic_block.address + idx as u32 * 4;
        let hazard = hazards
            .iter_mut()
            .position(|hazard| hazard.hazard.trigger == pc);
        if let Some(hazard_idx) = hazard {
            let hazard = &mut hazards[hazard_idx];
            let summary = (hazard.hazard.emit)(
                hazard.hazard.op,
                EmitCtx::builder()
                    .ptr_type(ptr_type)
                    .cache(&mut hazard.cache)
                    .pc(pc)
                    .node(hazard.node)
                    .cfg(cfg)
                    .deps_map(deps_map)
                    .fn_builder(fn_builder)
                    .build(),
            );
            hazards.swap_remove(hazard_idx);
            JIT::apply_cache_updates()
                .updates(CacheUpdates::new(&summary))
                .cache(register_cache)
                .call();
            *summary_queue = Some(summary);
        }

        // if op is boundary (last op) emit the op's delayed updates as well
        if op.is_block_boundary().is_some()
            && let Some(summary) = summary_queue.take()
        {
            // flush_updates();
            JIT::apply_cache_updates()
                .updates(CacheUpdates::new(&summary))
                .cache(register_cache)
                .call();
        }

        // if op is function boundary, emit updates to all cached values
        if op.is_function_boundary() {
            tracing::info!(%op, "emitting updates...");
            JIT::emit_updates()
                .builder(fn_builder)
                .block(basic_block.clif_block())
                .cache(register_cache)
                .call();
            if op.is_auto_pc() {
                let pc = fn_builder
                    .ins()
                    .iconst(types::I32, basic_block.address as i64 + idx as i64 * 4 + 4);
                let block = basic_block.clif_block();
                let cpu = fn_builder.block_params(block)[0];
                JIT::emit_store_pc(fn_builder, pc, cpu);
            }
        }

        // finally, emit op post update instructions (jumps, returns etc)
        op.post_update_emit_ir(
            EmitCtx::builder()
                .ptr_type(ptr_type)
                .cache(register_cache)
                .pc(basic_block.address + idx as u32 * 4)
                .node(node)
                .cfg(cfg)
                .deps_map(deps_map)
                .fn_builder(fn_builder)
                .build(),
        );

        op.get_hazard(
            EmitCtx::builder()
                .ptr_type(ptr_type)
                .cache(register_cache)
                .pc(basic_block.address + idx as u32 * 4)
                .node(node)
                .cfg(cfg)
                .deps_map(deps_map)
                .fn_builder(fn_builder)
                .build(),
        )
    }

    pub fn run(&mut self) -> color_eyre::Result<()> {
        loop {
            self.step_jit()?;
            tracing::info!("step: pc=0x{:08X}", self.cpu.pc);
        }
    }
}

#[derive(Debug, Clone)]
struct WalkFnSummary {
    cfg: Graph<BasicBlock, ()>,
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
    pub ops: Vec<DecodedOp>,
    pub address: u32,
    pub clif_block: Option<Block>,
}

impl BasicBlock {
    pub fn new(address: u32) -> Self {
        Self {
            address,
            clif_block: None,
            ops: Vec::with_capacity(32),
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

fn fetch(params: FetchParams<'_>) -> color_eyre::Result<WalkFnSummary> {
    fn emit_block_padding(
        state: &State,
        cfg: &mut Graph<BasicBlock, ()>,
        mem: &Memory,
    ) -> color_eyre::Result<()> {
        let remaining = state.max_padding_address.saturating_sub(state.pc);
        for i in (0..=remaining).step_by(4) {
            let addr = state.pc + i + 4;
            let op = mem.read::<u32>(PhysAddr(addr));
            let op = OpCode(op);
            let op = DecodedOp::try_from(op).wrap_err(eyre!("failed at pc 0x{:08X}", state.pc))?;
            cfg[state.current_node].ops.push(op);
            tracing::trace!(op = format!("{op}"));
        }
        Ok(())
    }

    let FetchParams {
        pc: initial_pc,
        mem,
        mut cfg,
        current_node_index,
        mut mapped,
    } = params;

    let entry_node =
        current_node_index.unwrap_or_else(|| cfg.add_node(BasicBlock::new(initial_pc)));

    mapped.to_mut().insert(initial_pc, entry_node);

    struct State {
        current_node: NodeIndex,
        pc: u32,
        from_jump: bool,
        max_padding: u32,
        max_padding_address: u32,
    }

    let mut stack = vec![State {
        current_node: entry_node,
        pc: initial_pc,
        from_jump: true,
        max_padding: 0,
        max_padding_address: initial_pc,
    }];
    stack.reserve(256);

    while let Some(mut state) = stack.pop() {
        let _span = span!(Level::DEBUG, "fetch", pc = %format!("0x{:04X}", state.pc), node = ?state.current_node.index()).entered();

        if state.from_jump {
            tracing::trace!("=== block({:?}) ===", state.current_node);
        }

        let op = mem.read::<u32>(PhysAddr(state.pc));
        let op = OpCode(op);
        let op = DecodedOp::try_from(op).wrap_err(eyre!("failed at pc 0x{:08X}", state.pc))?;
        cfg[state.current_node].ops.push(op);
        let trigger = op.hazard_trigger(state.pc).unwrap_or(0);
        let pad = trigger.saturating_sub(state.pc);
        if pad > state.max_padding {
            state.max_padding = pad;
            state.max_padding_address = state.pc;
        }

        tracing::trace!(op = format!("{op}"));
        tracing::trace!(?pad, ?state.max_padding);

        match op.is_block_boundary() {
            Some(BoundaryType::Function { .. }) => {
                emit_block_padding(&state, &mut cfg, mem)?;
                continue;
            }
            None => {
                const NOP_TOLERANCE: usize = 64;
                let len = cfg[state.current_node].ops.len();
                if len > NOP_TOLERANCE {
                    let slice = &cfg[state.current_node].ops[(len - NOP_TOLERANCE)..];
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

                let next_node = cfg.add_node(BasicBlock::new(new_address));
                cfg.add_edge(state.current_node, next_node, ());
                mapped.to_mut().insert(new_address, next_node);

                emit_block_padding(&state, &mut cfg, mem)?;

                tracing::trace!("cfg.link {:?} -> {:?}", state.current_node, next_node);
                // tracing::debug!(offset, "jump in block");
                stack.push(State {
                    current_node: next_node,
                    pc: new_address,
                    from_jump: true,
                    max_padding: 0,
                    max_padding_address: new_address,
                });
            }
            Some(BoundaryType::BlockSplit { lhs, rhs }) => {
                tracing::trace!(offsets = ?[lhs, rhs], delay_hazard = %op, "potential split in block");

                emit_block_padding(&state, &mut cfg, mem)?;

                let offsets = [rhs, lhs];
                for offset in offsets.into_iter() {
                    let new_address = offset.calculate_address(state.pc);

                    if let Some(cached_node) = mapped.get(&new_address) {
                        cfg.add_edge(state.current_node, *cached_node, ());
                        continue;
                    }

                    let next_node = cfg.add_node(BasicBlock::new(new_address));
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
                        max_padding: 0,
                        max_padding_address: new_address,
                    })
                }
            }
        };
    }

    Ok(WalkFnSummary { cfg })
}

/// Merge `src` into `dst` in-place. Ignores old node indices completely.
fn merge_into<N: Clone, E: Clone>(dst: &mut Graph<N, E, Directed>, src: &Graph<N, E, Directed>) {
    // Map old nodes -> new NodeIndex in dst
    let mut new_indices = Vec::with_capacity(src.node_count());

    // Add all nodes from src into dst
    for node in src.node_indices() {
        let weight = src[node].clone();
        let new_node = dst.add_node(weight);
        new_indices.push(new_node);
    }

    // Add all edges from src into dst, using new node indices
    for edge in src.edge_indices() {
        let (src_idx, dst_idx) = src.edge_endpoints(edge).unwrap();
        let weight = src.edge_weight(edge).unwrap().clone();
        dst.add_edge(
            new_indices[src_idx.index()],
            new_indices[dst_idx.index()],
            weight,
        );
    }
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct CachedValue {
    pub dirty: bool,
    pub value: Value,
}

#[derive(Builder)]
pub struct EmitCtx<'a, 'b> {
    pub fn_builder: &'a mut FunctionBuilder<'b>,
    pub ptr_type: types::Type,
    pub cache: &'a mut EntryCache,
    pub node: NodeIndex,
    pub pc: u32,
    pub cfg: &'a Graph<BasicBlock, ()>,
    pub deps_map: &'a HashMap<Block, CacheDependency>,
}

impl<'a, 'b> EmitCtx<'a, 'b> {
    pub fn block(&self) -> &BasicBlock {
        &self.cfg[self.node]
    }
    pub fn ins<'short>(&'short mut self) -> FuncInstBuilder<'short, 'b> {
        self.fn_builder.ins()
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
    #[instrument(skip(self))]
    pub fn out_params(&self, to: Block) -> Vec<BlockArg> {
        // tracing::trace!("{:#?}", self.deps_map);
        let next_block_deps: Option<_> = self.deps_map.get(&to).map(|dep| dep.registers);
        let mut args = self.emulator_params();
        if let Some(next_block_deps) = next_block_deps {
            let iter = next_block_deps
                .iter()
                .flat_map(|register| self.cache.registers[register as usize])
                .map(|value| value.value)
                .map(BlockArg::Value);
            args.extend(iter);
        } else {
            args.extend(
                self.cache
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
    pub fn emit_get_one(&mut self) -> Value {
        match self.cache.const_one {
            Some(one) => one,
            None => {
                let one = self.ins().iconst(types::I32, 1);
                self.cache.const_one = Some(one);
                one
            }
        }
    }
    pub fn emit_get_zero_i64(&mut self, fn_builder: &mut FunctionBuilder) -> Value {
        match self.cache.const_zero_i64 {
            Some(zero) => zero,
            None => {
                let zero = fn_builder.ins().iconst(types::I64, 0);
                self.cache.const_zero_i64 = Some(zero);
                zero
            }
        }
    }
    pub fn emit_get_zero(&mut self) -> Value {
        self.emit_get_register(0)
    }
    pub fn emit_get_hi(&mut self) -> Value {
        let block = self.block().clif_block();
        match self.cache.hi {
            Some(hi) => hi.value,
            None => {
                let hi = JIT::emit_load_hi()
                    .builder(self.fn_builder)
                    .block(block)
                    .call();
                self.cache.hi = Some(CachedValue {
                    dirty: false,
                    value: hi,
                });
                hi
            }
        }
    }
    pub fn emit_get_lo(&mut self) -> Value {
        let block = self.block().clif_block();
        match self.cache.lo {
            Some(lo) => lo.value,
            None => {
                let lo = JIT::emit_load_lo()
                    .builder(self.fn_builder)
                    .block(block)
                    .call();
                self.cache.lo = Some(CachedValue {
                    dirty: false,
                    value: lo,
                });
                lo
            }
        }
    }
    pub fn emit_get_register(&mut self, id: usize) -> Value {
        let block = self.block().clif_block();
        match self.cache.registers[id] {
            Some(value) => value.value,
            None => {
                let value = JIT::emit_load_reg()
                    .builder(self.fn_builder)
                    .block(block)
                    .idx(id)
                    .call();
                self.cache.registers[id] = Some(CachedValue {
                    dirty: false,
                    value,
                });
                value
            }
        }
    }
    pub fn emit_get_cop_register(
        &mut self,
        fn_builder: &mut FunctionBuilder,
        cop: u8,
        reg: usize,
    ) -> Value {
        JIT::emit_load_cop_reg()
            .builder(fn_builder)
            .block(self.block().clif_block())
            .idx(reg)
            .cop(cop)
            .call()
    }
    pub fn update_cache_immediate(&mut self, id: usize, value: Value) {
        self.cache.registers[id] = Some(CachedValue {
            dirty: false,
            value,
        });
    }
    pub fn emit_map_address_to_host(&mut self, address: Value) -> Value {
        let mem_map = self.mem_map();
        JIT::emit_map_address_to_host()
            .fn_builder(self.fn_builder)
            .ptr_type(self.ptr_type)
            .address(address)
            .mem_map_ptr(mem_map)
            .call()
    }
    pub fn emit_map_address_to_physical(&mut self, address: Value) -> Value {
        JIT::emit_map_address_to_physical()
            .fn_builder(self.fn_builder)
            .address(address)
            .call()
    }
    pub fn emit_store_pc(&mut self, value: Value) {
        let cpu = self.cpu();
        JIT::emit_store_pc(self.fn_builder, value, cpu);
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

#[derive(Builder, Debug, Default)]
#[builder(finish_fn(vis = "", name = build_internal))]
pub struct EmitSummary {
    #[builder(field = Vec::with_capacity(32))]
    pub register_updates: Vec<(usize, CachedValue)>,
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

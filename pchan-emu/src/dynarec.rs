use cranelift::{
    codegen::{bitset::ScalarBitSet, ir::Function},
    prelude::*,
};
use std::{borrow::Cow, collections::HashMap, mem::offset_of};

use bon::{Builder, bon, builder};
use color_eyre::eyre::{Context, eyre};

use petgraph::{
    algo::is_cyclic_directed,
    prelude::*,
    visit::{DfsEvent, depth_first_search},
};
use tracing::{Level, info_span, instrument, span};

use crate::{
    Emu,
    cpu::{
        Cpu,
        ops::{CachedValue, prelude::*},
    },
    jit::{BlockPage, CacheUpdates, CacheUpdatesRegisters, JIT},
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

#[derive(Default, Clone, Debug)]
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
        let mut prev_register_cache = entry_idx;

        depth_first_search(&blocks.cfg, Some(entry_idx), |event| match event {
            DfsEvent::TreeEdge(a, _) => {
                prev_register_cache = a;
            }
            DfsEvent::BackEdge(_, b) => {
                prev_register_cache = b;
            }
            DfsEvent::Discover(node, _) => {
                let basic_block = &blocks.cfg[node];
                let cranelift_block = basic_block.clif_block();

                let prev_register_cache = cache_map.get(&prev_register_cache).unwrap();
                let mut reg_cache = cache_map.get(&node).unwrap_or(prev_register_cache).clone();

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
                    .call();

                cache_map.insert(node, reg_cache);
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
    pub fn emit_block(
        fn_builder: &mut FunctionBuilder<'_>,
        cranelift_block: Block,
        ptr_type: types::Type,
        cfg: &Graph<BasicBlock, ()>,
        node: NodeIndex,
        register_cache: &mut EntryCache,
        deps: &CacheDependency,
        deps_map: &HashMap<Block, CacheDependency>,
        cpu: &mut Cpu,
    ) -> EmitBlockSummary {
        // fn_builder.seal_block(cranelift_block);
        fn_builder.switch_to_block(cranelift_block);
        fn_builder.append_block_param(cranelift_block, ptr_type);
        fn_builder.append_block_param(cranelift_block, ptr_type);
        fn_builder.append_block_param(cranelift_block, ptr_type);
        for reg in deps.registers.iter() {
            let reg = reg as usize;
            register_cache.registers[reg] = Some(CachedValue {
                value: fn_builder.append_block_param(cranelift_block, types::I32),
                dirty: false,
            });
        }
        let mut summary_queue: Option<EmitSummary> = None;
        let basic_block = &cfg[node];

        for (idx, op) in basic_block.ops.iter().enumerate() {
            // tracing::info!(op = %op);
            if op.is_block_boundary().is_some() {
                let summary = EmitBlockSummary;

                if idx + 1 < basic_block.ops.len() {
                    Self::emit_op()
                        .fn_builder(fn_builder)
                        .ptr_type(ptr_type)
                        .node(node)
                        .deps_map(deps_map)
                        .cfg(cfg)
                        .register_cache(register_cache)
                        .op(&basic_block.ops[idx + 1])
                        .idx(idx + 1)
                        .summary_queue(&mut summary_queue)
                        .basic_block(basic_block)
                        .call();
                }

                Self::emit_block_boundary()
                    .op(op)
                    .register_cache(register_cache)
                    .fn_builder(fn_builder)
                    .maybe_summary_queue(summary_queue.as_ref())
                    .cpu(cpu)
                    .ptr_type(ptr_type)
                    .idx(idx)
                    .node(node)
                    .cfg(cfg)
                    .deps_map(deps_map)
                    .call();
                return summary;
            }
            Self::emit_op()
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
                .call();
        }

        fn_builder.ins().return_(&[]);

        EmitBlockSummary
    }

    #[builder]
    #[instrument(name = "emit", skip_all, fields(%op, ))]
    pub fn emit_op(
        fn_builder: &mut FunctionBuilder<'_>,
        ptr_type: types::Type,
        cfg: &Graph<BasicBlock, ()>,
        node: NodeIndex,
        register_cache: &mut EntryCache,
        deps_map: &HashMap<Block, CacheDependency>,
        op: &DecodedOp,
        basic_block: &BasicBlock,
        idx: usize,
        summary_queue: &mut Option<EmitSummary>,
    ) {
        let summary = op.emit_ir(
            EmitParams::builder()
                .ptr_type(ptr_type)
                .cache(register_cache)
                .pc(basic_block.address + idx as u32 * 4)
                .node(node)
                .cfg(cfg)
                .deps_map(deps_map)
                .fn_builder(fn_builder)
                .build(),
        );

        if let Some(summary) = summary_queue.take() {
            // flush_updates();
            JIT::apply_cache_updates()
                .updates(CacheUpdates::new(&summary, CacheUpdatesRegisters::Delayed))
                .cache(register_cache)
                .call();
        }

        if let Some(summary) = summary {
            JIT::apply_cache_updates()
                .updates(CacheUpdates::new(
                    &summary,
                    CacheUpdatesRegisters::Immediate,
                ))
                .cache(register_cache)
                .call();
            *summary_queue = Some(summary);
        }
    }

    #[builder]
    #[instrument(name = "emitb", skip_all, fields(%op))]
    pub fn emit_block_boundary(
        op: &DecodedOp,
        mut summary_queue: Option<&EmitSummary>,
        register_cache: &mut EntryCache,
        fn_builder: &mut FunctionBuilder<'_>,
        cpu: &mut Cpu,
        ptr_type: types::Type,
        node: NodeIndex,
        cfg: &Graph<BasicBlock, ()>,
        deps_map: &HashMap<Block, CacheDependency>,
        idx: usize,
    ) {
        let basic_block = &cfg[node];

        if let Some(summary) = summary_queue.take() {
            JIT::apply_cache_updates()
                .updates(CacheUpdates::new(summary, CacheUpdatesRegisters::Delayed))
                .cache(register_cache)
                .call();
        }
        if let Some(BoundaryType::Function { auto_set_pc }) = op.is_block_boundary() {
            tracing::info!(%op, "function boundary");
            JIT::emit_updates()
                .builder(fn_builder)
                .block(basic_block.clif_block())
                .cache(register_cache)
                .call();
            if auto_set_pc {
                let cpu_value = fn_builder.block_params(basic_block.clif_block())[0];
                let pc_value = fn_builder.ins().iconst(
                    types::I32,
                    (basic_block.address as i32 + idx as i32 * 4) as i64,
                );
                fn_builder.ins().store(
                    MemFlags::new(),
                    pc_value,
                    cpu_value,
                    offset_of!(Cpu, pc) as i32,
                );
            }
        };
        if let Some(summary) = op.emit_ir(
            EmitParams::builder()
                .ptr_type(ptr_type)
                .fn_builder(fn_builder)
                .cache(register_cache)
                .pc(basic_block.address + idx as u32 * 4)
                .node(node)
                .cfg(cfg)
                .deps_map(deps_map)
                .build(),
        ) {
            JIT::apply_cache_updates()
                .updates(CacheUpdates::new(
                    &summary,
                    CacheUpdatesRegisters::Immediate,
                ))
                .cache(register_cache)
                .call();
            if let Some(pc) = summary.pc_update {
                cpu.pc = pc;
                tracing::debug!(?cpu.pc, "finished block", );
            }
            if !summary.finished_block {
                JIT::emit_updates()
                    .builder(fn_builder)
                    .block(basic_block.clif_block())
                    .cache(register_cache)
                    .call();
                fn_builder.ins().return_(&[]);
            }
        }
        tracing::debug!("{:?} compiled {} instructions", basic_block.clif_block, idx);
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
    }

    let mut stack = vec![State {
        current_node: entry_node,
        pc: initial_pc,
        from_jump: true,
    }];
    stack.reserve(256);

    while let Some(state) = stack.pop() {
        let _span = span!(Level::DEBUG, "fetch", pc = %format!("0x{:04X}", state.pc), node = ?state.current_node.index()).entered();

        if state.from_jump {
            tracing::trace!("=== block({:?}) ===", state.current_node);
        }

        let op = mem.read::<u32>(PhysAddr(state.pc));
        let op = OpCode(op);
        let op = DecodedOp::try_from(op).wrap_err(eyre!("failed at pc 0x{:08X}", state.pc))?;
        tracing::trace!(op = format!("{op}"));

        match op.is_block_boundary() {
            Some(BoundaryType::Function { .. }) => {
                cfg[state.current_node].ops.push(op);
                let op = mem.read::<u32>(PhysAddr(state.pc + 4));
                let op = OpCode(op);
                let op = DecodedOp::try_from(op)?;
                cfg[state.current_node].ops.push(op);

                continue;
            }
            None => {
                cfg[state.current_node].ops.push(op);
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

                cfg[state.current_node].ops.push(op);

                // execute instruction after branch
                let op = mem.read::<u32>(PhysAddr(state.pc + 4));
                let op = OpCode(op);
                let op = DecodedOp::try_from(op)?;

                // create next node
                // let next_node = splice_block_if_possible(
                //     mapped.to_mut(),
                //     &mut cfg,
                //     new_address,
                //     &state,
                //     recv_block,
                //     op,
                // )?;

                // if next_node.is_some() {
                //     tracing::debug!("quick link {:?} -> {:?}", state.current_node, next_node);
                //     continue;
                // }

                let next_node = cfg.add_node(BasicBlock::new(new_address));
                cfg.add_edge(state.current_node, next_node, ());
                cfg[next_node].ops.push(op);
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
                cfg[state.current_node].ops.push(op);

                // execute instruction after branch
                let op = mem.read::<u32>(PhysAddr(state.pc + 4));
                let op = OpCode(op);
                let op = DecodedOp::try_from(op)?;

                tracing::trace!(offsets = ?[lhs, rhs], delay_hazard = %op, "potential split in block");
                let offsets = [rhs, lhs];
                for offset in offsets.into_iter() {
                    let new_address = offset.calculate_address(state.pc);

                    if let Some(cached_node) = mapped.get(&new_address) {
                        cfg.add_edge(state.current_node, *cached_node, ());
                        continue;
                    }

                    // create the next node
                    // let next_node = splice_block_if_possible(
                    //     mapped.to_mut(),
                    //     &mut cfg,
                    //     new_address,
                    //     &state,
                    //     recv_block,
                    //     op,
                    // )?;

                    // if next_node.is_some() {
                    //     tracing::debug!("quick link {:?} -> {:?}", state.current_node, next_node);
                    //     continue;
                    // }

                    let next_node = cfg.add_node(BasicBlock::new(new_address));
                    cfg.add_edge(state.current_node, next_node, ());
                    cfg[next_node].ops.push(op);
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

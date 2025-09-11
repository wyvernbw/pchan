// allow for dev
#![allow(dead_code)]
#![allow(incomplete_features)]
#![feature(hash_map_macro)]
#![feature(iter_map_windows)]
#![feature(iterator_try_collect)]
#![feature(const_convert)]
#![feature(explicit_tail_calls)]
#![feature(associated_type_defaults)]
#![feature(trait_alias)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(const_ops)]
#![feature(try_blocks)]
#![feature(impl_trait_in_assoc_type)]
#![feature(const_trait_impl)]
#![feature(debug_closure_helpers)]
#![feature(iter_intersperse)]
#![feature(generic_const_exprs)]
#![feature(slice_as_array)]
#![feature(portable_simd)]
#![feature(iter_collect_into)]
// allow unused variables in tests to supress the setup tracing warnings
#![cfg_attr(test, allow(unused_variables))]

use std::{borrow::Cow, collections::HashMap, mem::offset_of};

use bon::{Builder, bon, builder};
use crossbeam::queue::ArrayQueue;
use petgraph::{
    algo::is_cyclic_directed,
    prelude::*,
    visit::{DfsEvent, depth_first_search},
};
use tracing::{Level, enabled, info_span, instrument, span, trace_span};

use crate::{
    bootloader::Bootloader,
    cpu::{
        Cpu, REG_STR,
        ops::{BoundaryType, DecodedOp, EmitParams, Op},
    },
    jit::{BlockPage, JIT},
    memory::{Memory, PhysAddr},
};

pub mod cranelift_bs {
    pub use cranelift::codegen::ir::*;
    #[allow(ambiguous_glob_reexports)]
    pub use cranelift::jit::*;
    pub use cranelift::module::*;
    pub use cranelift::prelude::isa::*;
    pub use cranelift::prelude::*;
}
pub mod bootloader;
pub mod cpu;
pub mod jit;
pub mod memory;

#[derive(Default, derive_more::Debug)]
pub struct Emu {
    #[debug(skip)]
    pub mem: Memory,
    pub cpu: Cpu,
    pub jit: JIT,
    pub boot: Bootloader,
}

use cranelift::{codegen::ir::Function, prelude::*};

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

#[bon]
impl Emu {
    pub fn load_bios(&mut self) -> color_eyre::Result<()> {
        self.boot.load_bios(&mut self.mem)?;
        Ok(())
    }
    fn create_block_queue(fn_builder: &mut FunctionBuilder) -> ArrayQueue<Block> {
        let clif_blocks = ArrayQueue::new(16);
        for _ in 0..16 {
            clif_blocks.push(fn_builder.create_block()).unwrap();
        }
        clif_blocks
    }
    pub fn step_jit(&mut self) -> color_eyre::Result<()> {
        self.step_jit_summarize()
    }
    pub fn step_jit_summarize<T: SummarizeJit>(&mut self) -> color_eyre::Result<T> {
        let initial_address = self.cpu.pc;

        let ptr_type = self.jit.pointer_type();

        self.jit.apply_dirty_pages(initial_address);

        // try cache first
        let cached = self
            .jit
            .use_cached_function(initial_address, &mut self.cpu, &mut self.mem);
        if cached {
            return Ok(T::summarize(SummarizeDeps::builder().build()));
        }

        // collect blocks in function
        let mut cfg = Graph::with_capacity(20, 64);
        let entry_idx = cfg.add_node(BasicBlock::new(self.cpu.pc as u32));
        let mut blocks = walk_fn(
            WalkFnParams::builder()
                .pc(self.cpu.pc as u32)
                .mem(&self.mem)
                .cfg(cfg)
                .current_node_index(entry_idx)
                .build(),
        )?;

        tracing::trace!(cfg.cycle = is_cyclic_directed(&blocks.cfg));

        let mut dfs = Dfs::new(&blocks.cfg, entry_idx);

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

        while let Some(node) = dfs.next(&blocks.cfg) {
            blocks.cfg[node].clif_block = Some(fn_builder.create_block());

            if tracing::enabled!(Level::TRACE) {
                let _span = trace_span!("trace_cfg").entered();
                tracing::trace!(cfg.node = ?blocks.cfg[node].clif_block);
                for op in blocks.cfg[node].ops.iter() {
                    tracing::trace!("    {op}");
                }

                tracing::trace!(
                    to = ?&blocks.cfg
                        .neighbors_directed(node, Direction::Outgoing)
                        .map(|n| blocks.cfg[n].clif_block)
                        .collect::<Vec<_>>(),
                    "    branch"
                );
            };
        }

        let entry_block = blocks.cfg[entry_idx].clif_block();

        let mut deps_map = hash_map! {
            entry_block => Vec::with_capacity(blocks.cfg.node_count())
        };
        let mut cache_map = hash_map! {
            entry_idx => [None; 32]
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
                let _span = info_span!(
                    "jit_comp",
                    node = node.index(),
                    b = ?cranelift_block,
                    ops.len = basic_block.ops.len()
                )
                .entered();
                tracing::trace!("=== compling {:?} === ", cranelift_block);

                let prev_register_cache = cache_map.get(&prev_register_cache).unwrap();
                let mut reg_cache = *cache_map.get(&node).unwrap_or(prev_register_cache);
                tracing::trace!(
                    "beginning with {} values in cache",
                    reg_cache.iter().flatten().count()
                );

                // TODO: combine in a single cache struct
                let prev_reg = prev_register_cache
                    .iter()
                    .enumerate()
                    .flat_map(|(i, value)| value.map(|_| i))
                    .collect();
                let deps = deps_map
                    .entry(blocks.cfg[node].clif_block())
                    .or_insert(prev_reg)
                    .clone();

                if enabled!(Level::TRACE) {
                    tracing::trace!("deps: {{");
                    for (block, deps) in deps_map.iter() {
                        let deps = deps
                            .iter()
                            .map(|&reg| format!("${}", REG_STR[reg]))
                            .collect::<Vec<_>>();
                        tracing::trace!("    {:?} => {:?}", block, deps);
                    }
                    tracing::trace!("}}");
                }

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

                tracing::trace!(
                    "ended with {} values in cache",
                    reg_cache.iter().flatten().count()
                );
                cache_map.insert(node, reg_cache);
                // prev_register_cache = node;
            }
            _ => {}
        });

        let _span = info_span!("jit_comp", pc = initial_address).entered();
        fn_builder.seal_all_blocks();

        Self::close_function(fn_builder);
        let summary = T::summarize(SummarizeDeps::builder().function(&func).build());
        self.jit.finish_function(func_id, func)?;

        let function = self.jit.get_func(func_id);
        tracing::info!("compiled function: {:?}", function.0);
        function(&mut self.cpu, &mut self.mem, true);
        self.jit.block_map.insert(initial_address, function);

        Ok(summary)
    }
    pub fn close_function(fn_builder: FunctionBuilder) {
        tracing::trace!("closing function");
        fn_builder.finalize();
    }
    #[builder]
    #[instrument(name = "emit", skip_all)]
    pub fn emit_block(
        fn_builder: &mut FunctionBuilder<'_>,
        cranelift_block: Block,
        ptr_type: types::Type,
        cfg: &Graph<BasicBlock, ()>,
        node: NodeIndex,
        register_cache: &mut [Option<Value>; 32],
        deps: &[usize],
        deps_map: &HashMap<Block, Vec<usize>>,
        cpu: &mut Cpu,
    ) -> EmitBlockSummary {
        // fn_builder.seal_block(cranelift_block);
        fn_builder.switch_to_block(cranelift_block);
        fn_builder.append_block_param(cranelift_block, ptr_type);
        fn_builder.append_block_param(cranelift_block, ptr_type);
        for reg in deps {
            register_cache[*reg] = Some(fn_builder.append_block_param(cranelift_block, ptr_type));
        }
        let mut updates_queue: Option<Box<[_]>> = None;
        let basic_block = &cfg[node];

        for (idx, op) in basic_block.ops.iter().enumerate() {
            // tracing::info!(op = %op);
            if op.is_block_boundary().is_some() {
                tracing::trace!(%op, "block boundary");
                let summary = EmitBlockSummary;
                Self::emit_block_boundary()
                    .op(op)
                    .register_cache(register_cache)
                    .fn_builder(fn_builder)
                    .maybe_updates_queue(updates_queue.as_deref())
                    .cpu(cpu)
                    .ptr_type(ptr_type)
                    .idx(idx)
                    .node(node)
                    .cfg(cfg)
                    .deps_map(deps_map)
                    .call();
                return summary;
            }
            let summary = op.emit_ir(
                EmitParams::builder()
                    .ptr_type(ptr_type)
                    .registers(register_cache)
                    .pc(basic_block.address + idx as u32 * 4)
                    .node(node)
                    .cfg(cfg)
                    .deps_map(deps_map)
                    .build(),
                fn_builder,
            );

            let mut flush_updates = |updates| {
                JIT::apply_cache_updates()
                    .maybe_updates(updates)
                    .cache(register_cache)
                    .call();
            };

            let updates = updates_queue.take();
            flush_updates(updates.as_deref());

            if let Some(summary) = summary {
                flush_updates(Some(&summary.register_updates));
                updates_queue = Some(summary.delayed_register_updates);
            }
        }

        JIT::emit_updates()
            .builder(fn_builder)
            .block(cranelift_block)
            .cache(register_cache)
            .call();
        let cpu_value = fn_builder.block_params(basic_block.clif_block())[0];
        let pc_value = fn_builder.ins().iconst(
            types::I64,
            basic_block.address as i64 + (basic_block.ops.len().saturating_sub(1)) as i64 * 4,
        );
        fn_builder.ins().store(
            MemFlags::new(),
            pc_value,
            cpu_value,
            offset_of!(Cpu, pc) as i32,
        );

        fn_builder.ins().return_(&[]);

        EmitBlockSummary
    }

    #[builder]
    pub fn emit_block_boundary(
        op: &DecodedOp,
        mut updates_queue: Option<&[(usize, Value)]>,
        register_cache: &mut [Option<Value>; 32],
        fn_builder: &mut FunctionBuilder<'_>,
        cpu: &mut Cpu,
        ptr_type: types::Type,
        node: NodeIndex,
        cfg: &Graph<BasicBlock, ()>,
        deps_map: &HashMap<Block, Vec<usize>>,
        idx: usize,
    ) {
        let updates = updates_queue.take();
        let basic_block = &cfg[node];

        JIT::apply_cache_updates()
            .maybe_updates(updates)
            .cache(register_cache)
            .call();
        if matches!(op.is_block_boundary(), Some(BoundaryType::Function)) {
            tracing::info!(%op, "function boundary");
            JIT::emit_updates()
                .builder(fn_builder)
                .block(basic_block.clif_block())
                .cache(register_cache)
                .call();
            let cpu_value = fn_builder.block_params(basic_block.clif_block())[0];
            let pc_value = fn_builder
                .ins()
                .iconst(types::I64, basic_block.address as i64 + idx as i64 * 4);
            fn_builder.ins().store(
                MemFlags::new(),
                pc_value,
                cpu_value,
                offset_of!(Cpu, pc) as i32,
            );
        }
        {}
        let summary = op.emit_ir(
            EmitParams::builder()
                .ptr_type(ptr_type)
                .registers(register_cache)
                .pc(basic_block.address + idx as u32 * 4)
                .node(node)
                .cfg(cfg)
                .deps_map(deps_map)
                .build(),
            fn_builder,
        );
        if let Some(summary) = summary
            && let Some(pc) = summary.pc_update
        {
            cpu.pc = pc as u64;
            tracing::debug!(cpu.pc);
        }
        tracing::debug!("{:?} compiled {} instructions", basic_block.clif_block, idx);
    }
    pub fn run(&mut self) -> color_eyre::Result<()> {
        loop {
            self.step_jit()?;
        }
    }
}

#[derive(Debug, Clone)]
struct WalkFnSummary {
    cfg: Graph<BasicBlock, ()>,
}

#[derive(Builder, Debug)]
struct WalkFnParams<'a> {
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
    ops: Vec<DecodedOp>,
    address: u32,
    clif_block: Option<Block>,
}

impl BasicBlock {
    fn new(address: u32) -> Self {
        Self {
            address,
            clif_block: None,
            ops: Vec::with_capacity(32),
        }
    }
    fn set_block(self, clif_block: Block) -> Self {
        Self {
            clif_block: Some(clif_block),
            ..self
        }
    }
    fn clif_block(&self) -> Block {
        self.clif_block
            .expect("basic block has no attached cranelift block!")
    }
}

fn find_first_value(than: u32, mapped: &HashMap<u32, NodeIndex>) -> Option<u32> {
    mapped.keys().filter(|addr| addr < &&than).max().cloned()
}

fn walk_fn(params: WalkFnParams<'_>) -> color_eyre::Result<WalkFnSummary> {
    let WalkFnParams {
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
        let _span = span!(Level::DEBUG, "walk_fn", pc = %format!("0x{:04X}", state.pc), node = ?state.current_node.index(), stack = stack.len()).entered();

        if state.from_jump {
            tracing::trace!("=== block({:?}) ===", state.current_node);
        }

        let op = mem.read::<u32>(PhysAddr(state.pc));
        let op = cpu::ops::OpCode(op);
        let op = DecodedOp::try_from(op)?;
        tracing::trace!(pc = state.pc, op = format!("{op}"), "read at");

        match op.is_block_boundary() {
            Some(BoundaryType::Function) => {
                continue;
            }
            None => {
                cfg[state.current_node].ops.push(op);
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
                let op = cpu::ops::OpCode(op);
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
                let op = cpu::ops::OpCode(op);
                let op = DecodedOp::try_from(op)?;

                tracing::trace!(offsets = ?[lhs, rhs], "potential split in block");
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

#[cfg(test)]
pub mod test_utils {

    use crate::Emu;
    use rstest::fixture;

    #[fixture]
    pub fn emulator() -> Emu {
        Emu::default()
    }
}

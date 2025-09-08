// allow for dev
#![allow(dead_code)]
#![allow(incomplete_features)]
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

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use bon::{Builder, bon, builder};
use color_eyre::eyre::eyre;
use crossbeam::queue::ArrayQueue;
use petgraph::{algo::is_cyclic_directed, prelude::*, visit::Topo};
use tracing::{Level, Span, debug_span, info_span, instrument, span, trace_span};

use crate::{
    bootloader::Bootloader,
    cpu::{
        Cpu,
        ops::{BoundaryType, DecodedOp, EmitParams, MipsOffset, Op},
    },
    jit::JIT,
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

#[derive(Default)]
pub struct Emu {
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

        // try cache first
        let cached = self
            .jit
            .use_cached_function(initial_address, &mut self.cpu, &mut self.mem);
        if cached {
            return Ok(T::summarize(SummarizeDeps::builder().build()));
        }

        let (func_id, mut func) = self.jit.create_function(initial_address)?;
        let mut fn_builder = self.jit.create_fn_builder(&mut func);

        // collect blocks in function
        let entry_block = fn_builder.create_block();
        let block_queue = Self::create_block_queue(&mut fn_builder);
        let mut cfg = Graph::default();
        let entry_idx = cfg.add_node(BasicBlock::new(self.cpu.pc as u32, entry_block));
        let blocks = walk_fn(
            WalkFnParams::builder()
                .pc(self.cpu.pc as u32)
                .mem(&self.mem)
                .cfg(cfg)
                .current_node_index(entry_idx)
                .cranelift_block_pool(&block_queue)
                .current_cranelift_block(entry_block)
                .build(),
        )?;
        tracing::info!(cfg.cycle = is_cyclic_directed(&blocks.cfg));

        let mut dfs = Dfs::new(&blocks.cfg, entry_idx);

        if tracing::enabled!(Level::TRACE) {
            let _span = trace_span!("trace_cfg").entered();
            while let Some(node) = dfs.next(&blocks.cfg) {
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
            }
        };

        let mut dfs = Dfs::new(&blocks.cfg, entry_idx);
        while let Some(node) = dfs.next(&blocks.cfg) {
            let basic_block = &blocks.cfg[node];
            let cranelift_block = basic_block.clif_block;
            let _span = info_span!(
                "jit_comp",
                node = node.index(),
                b = ?cranelift_block,
                ops.len = basic_block.ops.len()
            )
            .entered();
            Self::emit_block()
                .fn_builder(&mut fn_builder)
                .cranelift_block(cranelift_block)
                .cpu(&mut self.cpu)
                .ptr_type(ptr_type)
                .node(node)
                .cfg(&blocks.cfg)
                .call();
        }

        let _span = info_span!("jit_comp", pc = initial_address).entered();
        fn_builder.seal_all_blocks();

        Self::close_function(fn_builder);
        let summary = T::summarize(SummarizeDeps::builder().function(&func).build());
        self.jit.finish_function(func_id, func)?;

        let function = self.jit.get_func(func_id);
        function(&mut self.cpu, &mut self.mem);
        self.jit.block_map.insert(initial_address, function);

        Ok(summary)
    }
    pub fn close_function(fn_builder: FunctionBuilder) {
        tracing::debug!("closing function");
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
        cpu: &mut Cpu,
    ) {
        // fn_builder.seal_block(cranelift_block);
        fn_builder.switch_to_block(cranelift_block);
        fn_builder.append_block_param(cranelift_block, ptr_type);
        fn_builder.append_block_param(cranelift_block, ptr_type);
        let mut updates_queue: Option<Box<[_]>> = None;
        let mut register_cache: [Option<Value>; _] = [None; 32];
        let basic_block = &cfg[node];

        if basic_block.ops.is_empty() {
            tracing::debug!("empty block");
            fn_builder.ins().return_(&[]);
        }

        let mut count = 0;
        for (idx, op) in basic_block.ops.iter().enumerate() {
            // tracing::info!(op = %op);
            if op.is_block_boundary().is_some() {
                tracing::debug!("block boundary");
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
                    .call();
                return;
            }

            let summary = op.emit_ir(
                EmitParams::builder()
                    .ptr_type(ptr_type)
                    .registers(&mut register_cache)
                    .pc(basic_block.address + idx as u32 * 4)
                    .node(node)
                    .cfg(cfg)
                    .build(),
                fn_builder,
            );

            let mut flush_updates = |updates| {
                JIT::apply_cache_updates()
                    .maybe_updates(updates)
                    .cache(&mut register_cache)
                    .call();
            };

            let updates = updates_queue.take();
            flush_updates(updates.as_deref());

            if let Some(summary) = summary {
                flush_updates(Some(&summary.register_updates));
                updates_queue = Some(summary.delayed_register_updates);
            }

            count = idx;
        }

        let updates = register_cache
            .into_iter()
            .enumerate()
            .flat_map(|(idx, value)| Some(idx).zip(value))
            .collect::<Vec<_>>();

        JIT::emit_updates()
            .builder(fn_builder)
            .block(cranelift_block)
            .cache(&mut register_cache)
            .updates(&updates)
            .call();

        fn_builder.ins().return_(&[]);

        tracing::info!("{:?} compiled {} instructions", cranelift_block, count);
    }

    #[builder]
    pub fn emit_block_boundary(
        op: &DecodedOp,
        mut updates_queue: Option<&[(usize, Value)]>,
        mut register_cache: [Option<Value>; 32],
        fn_builder: &mut FunctionBuilder<'_>,
        cpu: &mut Cpu,
        ptr_type: types::Type,
        node: NodeIndex,
        cfg: &Graph<BasicBlock, ()>,
        idx: usize,
    ) {
        let updates = updates_queue.take();
        let basic_block = &cfg[node];

        JIT::apply_cache_updates()
            .maybe_updates(updates)
            .cache(&mut register_cache)
            .call();

        let updates = register_cache
            .into_iter()
            .enumerate()
            .flat_map(|(idx, value)| Some(idx).zip(value))
            .collect::<Vec<_>>();
        JIT::emit_updates()
            .builder(fn_builder)
            .block(basic_block.clif_block)
            .cache(&mut register_cache)
            .updates(&updates)
            .call();
        let summary = op.emit_ir(
            EmitParams::builder()
                .ptr_type(ptr_type)
                .registers(&mut register_cache)
                .pc(basic_block.address + idx as u32 * 4)
                .node(node)
                .cfg(cfg)
                .build(),
            fn_builder,
        );
        if let Some(summary) = summary
            && let Some(pc) = summary.pc_update
        {
            cpu.pc = pc as u64;
            tracing::info!(cpu.pc);
        }
        tracing::info!("{:?} compiled {} instructions", basic_block.clif_block, idx);
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
    cranelift_block_pool: &'a ArrayQueue<Block>,
    current_cranelift_block: Block,
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    ops: Vec<DecodedOp>,
    address: u32,
    clif_block: Block,
}

impl BasicBlock {
    fn new(address: u32, clif_block: Block) -> Self {
        Self {
            address,
            clif_block,
            ops: Vec::default(),
        }
    }
}

fn find_first_value(than: u32, mapped: &HashMap<u32, NodeIndex>) -> Option<u32> {
    mapped.keys().filter(|addr| addr < &&than).max().cloned()
}

fn walk_fn(params: WalkFnParams<'_>) -> color_eyre::Result<WalkFnSummary> {
    fn splice_block_if_possible(
        mapped: &mut HashMap<u32, NodeIndex>,
        cfg: &mut Graph<BasicBlock, ()>,
        new_address: u32,
        state: &State,
        recv_block: impl Fn() -> Result<Block, color_eyre::eyre::Error>,
        op: DecodedOp,
    ) -> color_eyre::Result<Option<NodeIndex>> {
        if let Some(first_before_new) = find_first_value(new_address, mapped) {
            let before_next = mapped[&first_before_new];
            let block_before_next = &cfg[before_next];
            let space = block_before_next.address
                ..(block_before_next.address + block_before_next.ops.len() as u32 * 4);

            if space.contains(&new_address) {
                tracing::debug!(node = before_next.index(), "found splice");
                // splice block
                let splice_idx = new_address - block_before_next.address;
                let splice_idx = splice_idx >> 2;
                let splice_idx = splice_idx as usize;

                let next_1 = before_next;
                let next_2 = cfg.add_node(BasicBlock::new(new_address, recv_block()?));

                tracing::debug!("splitting ops...");
                cfg[next_2].ops = cfg[next_1].ops.split_off(splice_idx);
                cfg[next_2].ops.push(op);
                tracing::debug!("done splitting");
                tracing::debug!("inserting jump instr from {:?} to {:?}...", next_1, next_2);
                cfg[next_1]
                    .ops
                    .push(DecodedOp::J(crate::cpu::ops::j::J { imm: new_address }));
                cfg[next_1].ops.push(DecodedOp::NOP(crate::cpu::ops::NOP));
                for node in [next_1, next_2] {
                    tracing::debug!("{:?} ops after split", node);
                    for op in cfg[node].ops.iter() {
                        tracing::debug!("    {op}");
                    }
                }

                let edges = cfg
                    .edges_directed(next_1, Direction::Outgoing)
                    .map(|e| e.id())
                    .collect::<Vec<_>>();
                for edge in edges {
                    let (_, to) = cfg.edge_endpoints(edge).unwrap();
                    cfg.remove_edge(edge);
                    cfg.add_edge(next_2, to, ());
                }
                cfg.add_edge(next_1, next_2, ());
                cfg.add_edge(state.current_node, next_2, ());

                mapped.insert(first_before_new, next_1);
                mapped.insert(new_address, next_2);

                return Ok(Some(next_2));
            }
        };
        Ok(None)
    }

    let WalkFnParams {
        pc: initial_pc,
        mem,
        mut cfg,
        current_node_index,
        mut mapped,
        cranelift_block_pool,
        current_cranelift_block,
    } = params;

    let entry_node = current_node_index
        .unwrap_or_else(|| cfg.add_node(BasicBlock::new(initial_pc, current_cranelift_block)));

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

    while let Some(state) = stack.pop() {
        let _span = span!(Level::INFO, "walk_fn", pc = %format!("0x{:04X}", state.pc), node = ?state.current_node.index()).entered();

        if state.from_jump {
            tracing::debug!(
                "=== block({:?}, {:?}) ===",
                cfg[state.current_node].clif_block,
                state.current_node
            );
        }

        let op = mem.read::<u32>(PhysAddr(state.pc));
        let op = cpu::ops::OpCode(op);
        let op = DecodedOp::try_new(op)?;
        tracing::debug!(pc = state.pc, op = format!("{op}"), "read at");

        let recv_block = || {
            let Some(next_block) = cranelift_block_pool.pop() else {
                tracing::warn!("queue ran out of blocks, consider increasing max queue size");
                return Err(eyre!("queue capacity exceeded"));
            };
            Ok(next_block)
        };

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
                let op = DecodedOp::try_new(op)?;

                // get block from pool
                let next_block = recv_block()?;

                // create next node
                let next_node = splice_block_if_possible(
                    mapped.to_mut(),
                    &mut cfg,
                    new_address,
                    &state,
                    recv_block,
                    op,
                )?;

                if next_node.is_some() {
                    tracing::debug!("quick link {:?} -> {:?}", state.current_node, next_node);
                    continue;
                }

                let next_node = cfg.add_node(BasicBlock::new(new_address, next_block));
                cfg.add_edge(state.current_node, next_node, ());
                cfg[next_node].ops.push(op);
                mapped.to_mut().insert(new_address, next_node);

                tracing::debug!("cfg.link {:?} -> {:?}", state.current_node, next_node);
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
                let op = DecodedOp::try_new(op)?;

                tracing::debug!(offsets = ?[lhs, rhs], "potential split in block");
                let offsets = [lhs, rhs];
                for offset in offsets.into_iter() {
                    let new_address = offset.calculate_address(state.pc);

                    if let Some(cached_node) = mapped.get(&new_address) {
                        cfg.add_edge(state.current_node, *cached_node, ());
                        continue;
                    }

                    let next_block = recv_block()?;

                    // create the next node
                    let next_node = splice_block_if_possible(
                        mapped.to_mut(),
                        &mut cfg,
                        new_address,
                        &state,
                        recv_block,
                        op,
                    )?;

                    if next_node.is_some() {
                        tracing::debug!("quick link {:?} -> {:?}", state.current_node, next_node);
                        continue;
                    }

                    let next_node = cfg.add_node(BasicBlock::new(new_address, next_block));
                    cfg.add_edge(state.current_node, next_node, ());
                    cfg[next_node].ops.push(op);
                    mapped.to_mut().insert(new_address, next_node);

                    tracing::debug!(
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

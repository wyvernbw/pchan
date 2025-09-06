// allow for dev
#![allow(dead_code)]
#![allow(incomplete_features)]
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

use std::{borrow::Cow, collections::HashSet};

use bon::{Builder, builder};
use cranelift::prelude::Block;
use crossbeam::queue::ArrayQueue;
use petgraph::{prelude::*, visit::Topo};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tracing::instrument;

use crate::{
    bootloader::Bootloader,
    cpu::{
        Cpu,
        ops::{BoundaryType, DecodedOp, EmitParams, Op},
    },
    jit::JIT,
    memory::{Memory, PhysAddr},
};

pub mod cranelift_bs {
    pub use cranelift::codegen::ir::*;
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

impl Emu {
    pub fn load_bios(&mut self) -> color_eyre::Result<()> {
        self.boot.load_bios(&mut self.mem)?;
        Ok(())
    }
    #[instrument(skip(self), fields(pc = %format!("0x{:08X}", self.cpu.pc)))]
    pub fn advance_jit(&mut self) -> color_eyre::Result<()> {
        use cranelift_bs::*;
        let initial_address = self.cpu.pc;

        let sig = self.jit.create_signature();
        let ptr_type = self.jit.pointer_type();

        // try cache first
        if let Some(function) = self.jit.block_map.get(&self.cpu.pc) {
            tracing::trace!("using cached function: 0x{:08X?}", function.0 as usize);
            function(&mut self.cpu, &mut self.mem);
            return Ok(());
        };

        let func_id = self.jit.module.declare_function(
            &format!("pc_0x{:08X}", initial_address),
            Linkage::Hidden,
            &sig,
        )?;
        let mut func = self.jit.create_function();
        let mut fn_builder = FunctionBuilder::new(&mut func, &mut self.jit.fn_builder_ctx);

        // collect blocks in function
        let clif_blocks = ArrayQueue::new(16);
        for _ in 0..16 {
            clif_blocks.push(fn_builder.create_block()).unwrap();
        }
        let entry_block = fn_builder.create_block();
        let blocks = find_block(
            FindBlockParams::builder()
                .pc(self.cpu.pc as u32)
                .mem(&self.mem)
                .depth(0)
                .max_depth(4)
                .cranelift_block_pool(&clif_blocks)
                .current_cranelift_block(entry_block)
                .build(),
        )?;
        let mut register_cache: [Option<Value>; _] = [None; 32];

        let mut topo = Topo::new(&blocks.cfg);
        let mut updates_queue: Option<Box<[_]>> = None;
        while let Some(node) = topo.next(&blocks.cfg) {
            let basic_block = &blocks.cfg[node];
            let cranelift_block = basic_block.clif_block;
            fn_builder.switch_to_block(cranelift_block);
            fn_builder.seal_block(cranelift_block);
            fn_builder.append_block_param(cranelift_block, ptr_type);
            fn_builder.append_block_param(cranelift_block, ptr_type);
            tracing::info!(block = ?cranelift_block, ops.len = basic_block.ops.len());

            for (idx, op) in basic_block.ops.iter().enumerate() {
                tracing::info!(block = ?cranelift_block, "emitting ir for {:?}", op);
                let summary = op.emit_ir(
                    EmitParams::builder()
                        .fn_builder(&mut fn_builder)
                        .ptr_type(ptr_type)
                        .registers(&register_cache)
                        .block(cranelift_block)
                        .pc(basic_block.address + idx as u32 * 4)
                        .next_blocks(&basic_block.next_block)
                        .build(),
                );
                if let Some(delayed_register_updates) = updates_queue.take() {
                    JIT::emit_updates()
                        .builder(&mut fn_builder)
                        .block(cranelift_block)
                        .cache(&mut register_cache)
                        .updates(&delayed_register_updates)
                        .call();
                }

                if let Some(summary) = summary {
                    updates_queue = Some(summary.delayed_register_updates);
                    JIT::emit_updates()
                        .builder(&mut fn_builder)
                        .block(cranelift_block)
                        .cache(&mut register_cache)
                        .updates(&summary.register_updates)
                        .call();
                    if let Some(pc) = summary.pc_update {
                        self.cpu.pc = pc as u64;
                        if let Some(delayed_register_updates) = updates_queue.take() {
                            JIT::emit_updates()
                                .builder(&mut fn_builder)
                                .block(cranelift_block)
                                .cache(&mut register_cache)
                                .updates(&delayed_register_updates)
                                .call();
                        }
                    }
                }

                let _ = op.post_emit_ir(
                    EmitParams::builder()
                        .fn_builder(&mut fn_builder)
                        .ptr_type(ptr_type)
                        .registers(&register_cache)
                        .block(cranelift_block)
                        .pc(basic_block.address + idx as u32 * 4)
                        .next_blocks(&basic_block.next_block)
                        .build(),
                );
            }
        }

        fn_builder.ins().return_(&[]);
        fn_builder.finalize();

        self.jit.ctx.func = func;
        self.jit
            .module
            .define_function(func_id, &mut self.jit.ctx)?;

        self.jit.module.clear_context(&mut self.jit.ctx);
        self.jit.module.finalize_definitions()?;

        let function = self.jit.get_func(func_id);
        function(&mut self.cpu, &mut self.mem);

        self.jit.block_map.insert(initial_address, function);

        Ok(())
    }
    pub fn run(&mut self) -> color_eyre::Result<()> {
        loop {
            self.advance_jit()?;
        }
    }
}

#[derive(Debug, Clone)]
struct FindBlockSummary {
    cfg: Graph<BasicBlock, i32>,
}

#[derive(Builder, Debug)]
struct FindBlockParams<'a> {
    pc: u32,
    mem: &'a Memory,
    depth: usize,
    max_depth: usize,
    #[builder(default)]
    cfg: Graph<BasicBlock, i32>,
    current_node_index: Option<NodeIndex<u32>>,
    #[builder(default)]
    mapped: Cow<'a, HashSet<u32>>,
    cranelift_block_pool: &'a ArrayQueue<Block>,
    current_cranelift_block: Block,
}

#[derive(Debug, Clone)]
struct BasicBlock {
    ops: Vec<DecodedOp>,
    address: u32,
    clif_block: Block,
    next_block: [Option<Block>; 2],
}

impl BasicBlock {
    fn new(address: u32, clif_block: Block) -> Self {
        Self {
            address,
            clif_block,
            ops: Vec::default(),
            next_block: [None; 2],
        }
    }
}

#[instrument(skip_all)]
fn find_block(params: FindBlockParams<'_>) -> color_eyre::Result<FindBlockSummary> {
    let FindBlockParams {
        pc,
        mem,
        depth,
        max_depth,
        mut cfg,
        current_node_index,
        mut mapped,
        cranelift_block_pool,
        current_cranelift_block,
    } = params;

    if depth > max_depth {
        return Ok(FindBlockSummary { cfg });
    }

    let op = mem.read::<u32>(PhysAddr(pc));
    let op = cpu::ops::OpCode(op);
    let op = DecodedOp::try_new(op)?;
    tracing::debug!(?op, "block reading at pc={}", pc,);

    let current_node = current_node_index.unwrap_or_else(|| {
        mapped.to_mut().insert(pc);
        cfg.add_node(BasicBlock::new(pc, current_cranelift_block))
    });

    match op.is_block_boundary() {
        Some(BoundaryType::Block { offset }) => {
            let new_address = (pc & 0xFF00_0000).wrapping_add_signed(offset);
            if !mapped.to_mut().insert(new_address) {
                return Ok(FindBlockSummary { cfg });
            };
            cfg[current_node].ops.push(op);

            // execute instruction after branch
            let op = mem.read::<u32>(PhysAddr(pc + 4));
            let op = cpu::ops::OpCode(op);
            let op = DecodedOp::try_new(op)?;

            // get block from pool
            let Some(next_block) = cranelift_block_pool.pop() else {
                tracing::warn!("queue ran out of blocks, consider increasing max queue size");
                return Ok(FindBlockSummary { cfg });
            };

            // create next node
            let next_node = cfg.add_node(BasicBlock::new(new_address, next_block));
            cfg.add_edge(current_node, next_node, offset);
            cfg[current_node].next_block[0] = Some(next_block);
            cfg[next_node].ops.push(op);
            tracing::debug!(offset, "jump in block");

            find_block(
                FindBlockParams::builder()
                    .pc(new_address)
                    .mem(mem)
                    .depth(depth + 1)
                    .cfg(cfg)
                    .max_depth(max_depth)
                    .current_node_index(next_node)
                    .mapped(mapped)
                    .cranelift_block_pool(cranelift_block_pool)
                    .current_cranelift_block(next_block)
                    .build(),
            )
        }
        Some(BoundaryType::BlockSplit { lhs, rhs }) => {
            cfg[current_node].ops.push(op);

            // execute instruction after branch
            let op = mem.read::<u32>(PhysAddr(pc + 4));
            let op = cpu::ops::OpCode(op);
            let op = DecodedOp::try_new(op)?;

            tracing::debug!(offsets = ?[lhs, rhs], "split in block");
            let offsets = [lhs, rhs];
            let new_cfgs = offsets
                .iter()
                .cloned()
                .enumerate()
                .map(|(idx, offset)| {
                    let new_address = (pc & 0xFF00_0000).wrapping_add_signed(offset);

                    // get independent state for each thread
                    let mut cfg = cfg.clone();

                    if !mapped.to_mut().insert(new_address) {
                        return Ok(FindBlockSummary {
                            cfg: Graph::default(),
                        });
                    };

                    let Some(next_block) = cranelift_block_pool.pop() else {
                        tracing::warn!(
                            "queue ran out of blocks, consider increasing max queue size"
                        );
                        return Ok(FindBlockSummary { cfg });
                    };

                    // create the next node
                    let next_node = cfg.add_node(BasicBlock::new(new_address, next_block));
                    cfg.add_edge(current_node, next_node, offset);
                    cfg[current_node].next_block[idx] = Some(next_block);
                    cfg[next_node].ops.push(op);

                    find_block(
                        FindBlockParams::builder()
                            .pc(new_address)
                            .mem(mem)
                            .depth(depth + 1)
                            // clone makes it easier to parallelize later
                            .cfg(cfg)
                            .max_depth(max_depth)
                            .current_node_index(next_node)
                            .mapped(Cow::Borrowed(mapped.to_mut()))
                            .cranelift_block_pool(cranelift_block_pool)
                            .current_cranelift_block(next_block)
                            .build(),
                    )
                })
                .collect::<color_eyre::Result<Box<[_]>>>()?;
            for FindBlockSummary { cfg: subgraph } in new_cfgs.iter() {
                merge_into(&mut cfg, subgraph);
            }
            Ok(FindBlockSummary { cfg })
        }
        Some(BoundaryType::Function) => Ok(FindBlockSummary { cfg }),
        None => {
            cfg[current_node].ops.push(op);
            find_block(
                FindBlockParams::builder()
                    .pc(pc + 4)
                    .mem(mem)
                    .depth(depth)
                    .cfg(cfg)
                    .max_depth(max_depth)
                    .current_node_index(current_node)
                    .mapped(mapped)
                    .cranelift_block_pool(cranelift_block_pool)
                    .current_cranelift_block(current_cranelift_block)
                    .build(),
            )
        }
    }
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

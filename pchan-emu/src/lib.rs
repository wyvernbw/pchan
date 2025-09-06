// allow for dev
#![allow(dead_code)]
#![allow(incomplete_features)]
#![feature(iterator_try_collect)]
#![feature(explicit_tail_calls)]
#![feature(associated_type_defaults)]
#![feature(trait_alias)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(const_ops)]
#![feature(try_blocks)]
#![feature(impl_trait_in_assoc_type)]
#![feature(const_from)]
#![feature(const_trait_impl)]
#![feature(debug_closure_helpers)]
#![feature(iter_intersperse)]
#![feature(generic_const_exprs)]
#![feature(slice_as_array)]
#![feature(portable_simd)]
#![feature(iter_collect_into)]
// allow unused variables in tests to supress the setup tracing warnings
#![cfg_attr(test, allow(unused_variables))]

use std::collections::HashSet;

use bon::{Builder, builder};
use petgraph::{prelude::*, visit::Topo};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::instrument;

use crate::{
    bootloader::Bootloader,
    cpu::{
        Cpu, JIT,
        ops::{BoundaryType, DecodedOp, EmitParams, Op},
    },
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
pub mod memory;

#[derive(Default)]
pub struct Emu {
    mem: Memory,
    cpu: Cpu,
    jit: JIT,
    boot: Bootloader,
}

impl Emu {
    fn load_bios(&mut self) -> color_eyre::Result<()> {
        self.boot.load_bios(&mut self.mem)?;
        Ok(())
    }
    #[instrument(skip(self), fields(pc = %format!("0x{:08X}", self.cpu.pc)))]
    fn advance_jit(&mut self) -> color_eyre::Result<()> {
        use cranelift_bs::*;

        let sig = self.jit.create_signature();
        let ptr_type = self.jit.pointer_type();

        // try cache first
        if let Some(function) = self.jit.block_map.get(&self.cpu.pc) {
            function(&mut self.cpu, &mut self.mem);
            return Ok(());
        };

        // collect blocks in function
        let blocks = find_block(
            FindBlockParams::builder()
                .pc(self.cpu.pc as u32)
                .mem(&self.mem)
                .depth(0)
                .max_depth(4)
                .build(),
        )?;
        let func_id =
            self.jit
                .module
                .declare_function(&self.cpu.pc.to_string(), Linkage::Hidden, &sig)?;
        let mut func = Function::with_name_signature(UserFuncName::user(0, 1), sig.clone());

        let mut fn_builder = FunctionBuilder::new(&mut func, &mut self.jit.fn_builder_ctx);
        let mut register_cache: [Option<Value>; _] = [None; 32];

        let mut topo = Topo::new(&blocks.cfg);
        let mut updates_queue = None;
        while let Some(node) = topo.next(&blocks.cfg) {
            let basic_block = &blocks.cfg[node];
            let cranelift_block = JIT::init_block(&mut fn_builder);
            for (idx, op) in basic_block.ops.iter().enumerate() {
                let summary = op.emit_ir(
                    EmitParams::builder()
                        .fn_builder(&mut fn_builder)
                        .ptr_type(ptr_type)
                        .registers(&register_cache)
                        .block(cranelift_block)
                        .pc(basic_block.address + idx as u32 * 4)
                        .build(),
                );
                if let Some(delayed_register_updates) = updates_queue.take() {
                    JIT::emit_updates()
                        .builder(&mut fn_builder)
                        .block(cranelift_block)
                        .cache(&mut register_cache)
                        .updates(delayed_register_updates)
                        .call();
                }

                if let Some(summary) = summary {
                    updates_queue = Some(summary.delayed_register_updates);
                    JIT::emit_updates()
                        .builder(&mut fn_builder)
                        .block(cranelift_block)
                        .cache(&mut register_cache)
                        .updates(summary.register_updates)
                        .call();
                }
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
        Ok(())
    }
    fn run(&mut self) -> color_eyre::Result<()> {
        loop {
            self.advance_jit()?;
        }
    }
}

#[derive(Debug, Clone)]
struct FindBlockSummary {
    cfg: Graph<BasicBlock, u32>,
}

#[derive(Builder, Debug, Clone)]
struct FindBlockParams<'a> {
    pc: u32,
    mem: &'a Memory,
    depth: usize,
    max_depth: usize,
    #[builder(default)]
    cfg: Graph<BasicBlock, u32>,
    current_node_index: Option<NodeIndex<u32>>,
    #[builder(default)]
    mapped: HashSet<u32>,
}

#[derive(Debug, Clone, Default)]
struct BasicBlock {
    ops: Vec<DecodedOp>,
    address: u32,
}

impl BasicBlock {
    fn new(address: u32) -> Self {
        Self {
            address,
            ..Default::default()
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
    } = params;
    if depth > max_depth {
        return Ok(FindBlockSummary { cfg });
    }
    tracing::debug!("block reading at pc={}", pc);
    let op = mem.read::<u32>(PhysAddr(pc));
    let op = cpu::ops::OpCode(op);
    let op = DecodedOp::try_new(op)?;

    let current_node = current_node_index.unwrap_or_else(|| {
        mapped.insert(pc);
        cfg.add_node(BasicBlock::new(pc))
    });

    match op.is_block_boundary() {
        Some(BoundaryType::Block { offset }) => {
            let new_address = pc + offset;
            if !mapped.insert(new_address) {
                return Ok(FindBlockSummary { cfg });
            };
            cfg[current_node].ops.push(op);
            let next_node = cfg.add_node(BasicBlock::new(new_address));
            cfg.add_edge(current_node, next_node, offset);
            tracing::debug!(offset, "jump in block");
            find_block(
                FindBlockParams::builder()
                    .pc(pc + offset)
                    .mem(mem)
                    .depth(depth + 1)
                    .cfg(cfg)
                    .max_depth(max_depth)
                    .current_node_index(next_node)
                    .mapped(mapped)
                    .build(),
            )
        }
        Some(BoundaryType::BlockSplit { offsets }) => {
            cfg[current_node].ops.push(op);
            tracing::debug!(?offsets, "split in block");
            let new_cfgs = offsets
                .par_iter()
                .flatten()
                .cloned()
                .map(|offset| {
                    let new_address = pc + offset;

                    let mut mapped = mapped.clone();
                    let mut cfg = cfg.clone();

                    if !mapped.insert(new_address) {
                        return Ok(FindBlockSummary {
                            cfg: Graph::default(),
                        });
                    };

                    let next_node = cfg.add_node(BasicBlock::new(new_address));
                    cfg.add_edge(current_node, next_node, offset);

                    find_block(
                        FindBlockParams::builder()
                            .pc(pc + offset)
                            .mem(mem)
                            .depth(depth + 1)
                            // clone makes it easier to parallelize later
                            .cfg(cfg)
                            .max_depth(max_depth)
                            .current_node_index(next_node)
                            .mapped(mapped.clone())
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
    use super::*;
    use rstest::fixture;

    #[fixture]
    pub fn emulator() -> Emu {
        Emu::default()
    }
}

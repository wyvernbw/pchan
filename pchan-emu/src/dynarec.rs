use crate::{
    IntoInst,
    cpu::Reg,
    cranelift_bs::*,
    dynarec::sparse_queue::SparseQueue,
    jit::{FuncRefTable, FunctionMap, JitCache},
    memory::ext,
};
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::Write,
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    rc::Rc,
};

use bon::{Builder, bon, builder};

use pchan_utils::hex;
use petgraph::{
    algo::is_cyclic_directed,
    dot::{Config, Dot},
    prelude::*,
    visit::{DfsEvent, depth_first_search},
};
use tracing::{Level, enabled, instrument};
use tracing::{info_span, trace_span};

pub mod prelude {
    pub use super::builder::*;
    pub use super::*;
    pub use crate::FnBuilderExt;
    pub use crate::cpu::REG_STR;
    pub use crate::cpu::ops::prelude::*;
    pub use crate::cpu::reg_str;
    pub use crate::cranelift_bs::*;
    pub use crate::dynarec::JitSummary;
    pub use crate::icmp;
    pub use crate::icmpimm;
    pub use crate::load;
    pub use crate::memory::ext;
    pub use crate::mult;
    pub use crate::shift;
    pub use crate::shiftimm;
    pub use crate::store;
    pub use pchan_utils::hex;
}
pub mod builder;
#[path = "sparse-queue.rs"]
pub mod sparse_queue;

use crate::{
    Emu, FnBuilderExt,
    cpu::{Cpu, ops::prelude::*},
    jit::{FunctionPage, JIT},
    memory::Memory,
};

#[derive(derive_more::Debug, Builder, Clone, derive_more::Display)]
#[display("{:#?}", self)]
pub struct JitSummary {
    #[debug("{}", self.decoded_ops)]
    pub decoded_ops: String,
    pub function: Option<Function>,
    pub panicked: bool,
    #[debug("{}", self.function_panic)]
    pub function_panic: String,
    #[debug("{}", self.cpu_state)]
    pub cpu_state: String,
}

#[derive(Debug, Builder, Clone)]
pub struct SummarizeDeps<'a> {
    function: Option<&'a Function>,
    function_panic: Option<Result<(), String>>,
    cpu: &'a Cpu,
    fetch_summary: Option<&'a FetchSummary>,
}

pub trait SummarizeJit {
    fn summarize(deps: SummarizeDeps) -> Self;
}

impl SummarizeJit for () {
    fn summarize(_: SummarizeDeps) -> Self {}
}

impl SummarizeJit for JitSummary {
    fn summarize(deps: SummarizeDeps) -> Self {
        let cpu_state = format!("{:#?}", deps.cpu);
        let blocks = if let Some(fetch) = deps.fetch_summary {
            let mut dfs = Dfs::new(&fetch.cfg, fetch.entry);
            let mut buf = String::with_capacity(320);
            writeln!(&mut buf, "{{").unwrap();
            while let Some(node) = dfs.next(&fetch.cfg) {
                // tracing::trace!(cfg.node = ?fetch.cfg[node].clif_block);
                writeln!(&mut buf, "  {:?}:", fetch.cfg[node].clif_block()).unwrap();
                let ops = fetch.ops_for(&fetch.cfg[node]);
                let mut address = fetch.cfg[node].address;
                for op in ops {
                    writeln!(&mut buf, "    {}:    {op}", hex(address)).unwrap();
                    address += 4;
                }
                if ops.is_empty() {
                    writeln!(&mut buf, "    (empty)").unwrap();
                }
                _ = write!(&mut buf, "    => jumps to: ");
                let mut jumped = false;
                for n in fetch.cfg.neighbors_directed(node, Direction::Outgoing) {
                    _ = write!(&mut buf, "{:?}, ", fetch.cfg[n].clif_block());
                    jumped = true;
                }
                if !jumped {
                    _ = write!(&mut buf, "(none)");
                }
                _ = writeln!(&mut buf);
                writeln!(&mut buf).unwrap();
            }
            writeln!(&mut buf, "}}").unwrap();
            buf
        } else {
            "N/A (ops not passed to summarize)".to_string()
        };
        let panicked = deps
            .function_panic
            .as_ref()
            .map(|info| info.is_err())
            .unwrap_or(false);
        let panic = match deps.function_panic {
            Some(result) => match result {
                Ok(_) => "no panic. 👍".to_string(),
                Err(err) => format!("panic: {:?}", err),
            },
            None => "N/A (level `trace` required for this information.)".to_string(),
        };
        Self::builder()
            .maybe_function(deps.function.cloned())
            .function_panic(panic)
            .cpu_state(cpu_state)
            .decoded_ops(blocks)
            .panicked(panicked)
            .build()
    }
}

pub struct EmitBlockSummary;

#[derive(Default, Clone, derive_more::Debug, Hash)]
pub struct EntryCache {
    #[debug("{:?}", self.registers.iter().filter(|reg| reg.is_some()).collect::<Vec<_>>())]
    pub registers: [Option<CachedValue>; 32],
    pub const_one: Option<Value>,
    pub const_zero_i64: Option<Value>,
    pub hi: Option<CachedValue>,
    pub lo: Option<CachedValue>,
}

impl EntryCache {
    fn iter(&self) -> impl Iterator<Item = CachedValue> {
        let EntryCache {
            registers,
            const_one: _,
            const_zero_i64: _,
            hi,
            lo,
        } = self;

        registers
            .iter()
            .flatten()
            .cloned()
            .chain([hi, lo].into_iter().flatten().cloned())
    }
}

#[bon]
impl Emu {
    pub fn load_bios(&mut self) -> color_eyre::Result<()> {
        self.boot.load_bios(&mut self.mem, &self.cpu)?;
        Ok(())
    }
    pub fn jump_to_bios(&mut self) {
        self.cpu.jump_to_bios();
    }
    pub fn step_jit(&mut self) -> color_eyre::Result<()> {
        self.step_jit_summarize()
    }
    #[instrument(name = "dynarec", skip_all, fields(pc = %hex(self.cpu.pc)))]
    pub fn step_jit_summarize<T: SummarizeJit>(&mut self) -> color_eyre::Result<T> {
        let initial_address = self.cpu.pc;

        let ptr_type = self.jit.pointer_type();

        self.jit_cache.apply_dirty_pages(initial_address);

        // try cache first
        let cached = self.jit_cache.use_cached_function(initial_address);
        if let Some(cached) = cached {
            cached(&mut self.cpu, &mut self.mem, false);
            tracing::info!("{:?} fn invoked", cached.fn_ptr);
            return Ok(T::summarize(
                SummarizeDeps::builder()
                    .cpu(&self.cpu)
                    .function(&cached.func)
                    .build(),
            ));
        }

        // collect blocks in function
        let cfg = Graph::with_capacity(256, 0);
        let mut blocks = fetch(
            FetchParams::builder()
                .pc(initial_address)
                .mem(&self.mem)
                .cpu(&self.cpu)
                .cfg(cfg)
                .build(),
        )?;

        if enabled!(Level::TRACE) {
            tracing::trace!(cfg.cycle = is_cyclic_directed(&blocks.cfg));
            tracing::trace!(
                "\n{:?}",
                Dot::with_config(&blocks.cfg, &[Config::EdgeNoLabel, Config::NodeIndexLabel])
            );
        }

        for node in blocks.cfg.node_indices() {
            let block = &blocks.cfg[node];
            for op in blocks.ops_for(block).iter() {
                if let Some(address) = op.invalidates_cache_at() {
                    let page = FunctionPage::new(address);
                    self.jit_cache.dirty_pages.insert(page);
                }
            }
        }

        let (func_id, mut func) = self.jit.create_function(initial_address)?;
        let func_ref_table = self.jit.create_func_ref_table(&mut func);
        let mut fn_builder = self.jit.create_fn_builder(&mut func);

        let mut dfs = Dfs::new(&blocks.cfg, blocks.entry);
        while let Some(node) = dfs.next(&blocks.cfg) {
            blocks.cfg[node].clif_block = Some(fn_builder.create_block());
        }

        let entry_idx = blocks.entry;

        depth_first_search(&blocks.cfg, Some(entry_idx), |event| {
            if let DfsEvent::Discover(node, _) = event {
                let basic_block = &blocks.cfg[node];
                let cranelift_block = basic_block.clif_block();

                if enabled!(Level::TRACE) {
                    tracing::trace!(
                        ?node,
                        ?cranelift_block,
                        "compiling {} psx ops...",
                        blocks.ops_for(basic_block).len()
                    );
                }

                let mut state = BlockState {
                    cache: EntryCache::default(),
                    basic_block: basic_block.clone(),
                    ptr_type,
                    node,
                };

                let _ = Self::emit_block()
                    .state(&mut state)
                    .fn_builder(&mut fn_builder)
                    .cpu(&mut self.cpu)
                    .jit_cache(&self.jit_cache)
                    .ptr_type(ptr_type)
                    .node(node)
                    .cfg(&blocks.cfg)
                    .ops(blocks.ops_for(basic_block))
                    .func_ref_table(&func_ref_table)
                    .call();
            }
        });

        // let _span = info_span!("jit_comp", pc = %format!("0x{:08X}", initial_address)).entered();
        fn_builder.seal_all_blocks();

        Self::close_function(fn_builder);
        self.jit.finish_function(func_id, func.clone())?;

        let function = self.jit.get_func(func_id, func);
        tracing::info!("{:?} fn compiled", function.fn_ptr);
        let potential_panic = {
            let _span = info_span!("fn", addr = ?function.fn_ptr).entered();
            if std::env::var("PCHAN_UNWIND").is_ok() {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    function(&mut self.cpu, &mut self.mem, true);
                }))
                .map_err(|err| panic_message::panic_message(&err).to_string());
                Some(result)
            } else {
                function(&mut self.cpu, &mut self.mem, true);
                None
            }
        };

        tracing::trace!("{:#?}", self.cpu);
        let summary_deps = SummarizeDeps::builder().function(&function.func);
        let summary_deps = summary_deps
            .maybe_function_panic(potential_panic)
            .fetch_summary(&blocks)
            .cpu(&self.cpu);
        let summary = T::summarize(summary_deps.build());

        self.jit_cache.fn_map.insert(initial_address, function);

        Ok(summary)
    }
    pub fn close_function(fn_builder: FunctionBuilder) {
        tracing::trace!("closing function");
        fn_builder.finalize();
    }

    #[instrument(level = Level::TRACE, skip(fn_builder, ptr_type))]
    pub fn init_block(fn_builder: &mut FunctionBuilder, block: Block, ptr_type: types::Type) {
        // insert base function parameters (ptr to cpu, mem, etc.)
        for _ in 0..JIT::FN_PARAMS {
            fn_builder.append_block_param(block, ptr_type);
        }
    }

    #[builder]
    pub fn emit_block<'a, 'b>(
        ops: &[DecodedOp],
        state: &'a mut BlockState,
        fn_builder: &'a mut FunctionBuilder<'b>,
        func_ref_table: &'a FuncRefTable,
        ptr_type: types::Type,
        cfg: &'a Graph<BasicBlock, ()>,
        node: NodeIndex,
        cpu: &mut Cpu,
        jit_cache: &JitCache,
    ) -> EmitBlockSummary {
        let cranelift_block = state.basic_block.clif_block();
        fn_builder.switch_to_block(cranelift_block);

        Self::init_block(fn_builder, cranelift_block, ptr_type);

        collect_instructions()
            .node(node)
            .state(state)
            .fn_builder(fn_builder)
            .func_ref_table(func_ref_table)
            .cfg(cfg)
            .ops(ops)
            .cpu(cpu)
            .function_map(&jit_cache.fn_map)
            .call();

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
    entry: NodeIndex,
    decoded_ops: Vec<DecodedOp>,
}

impl FetchSummary {
    fn ops_for<'a>(&'a self, block: &BasicBlock) -> &'a [DecodedOp] {
        let end = (block.ops.0.end as usize).min(self.decoded_ops.len());
        &self.decoded_ops[(block.ops.0.start as usize)..(end)]
    }
}

#[derive(Builder, Debug)]
struct FetchParams<'a> {
    pc: u32,
    mem: &'a Memory,
    cpu: &'a Cpu,
    #[builder(default)]
    cfg: Graph<BasicBlock, ()>,
    #[builder(default)]
    mapped: Cow<'a, HashMap<u32, NodeIndex>>,
}

#[derive(Debug, Clone, Default)]
pub struct BasicBlock {
    pub address: u32,
    pub clif_block: Option<Block>,
    pub ops: OpsHandle,
}

#[derive(Debug, Clone, Default)]
pub struct OpsHandle(pub Range<u32>);

impl<I> From<Range<I>> for OpsHandle
where
    I: Into<u32>,
{
    fn from(value: Range<I>) -> Self {
        let value = (value.start.into())..(value.end.into());
        OpsHandle(value)
    }
}

impl OpsHandle {
    fn from_usize(value: Range<usize>) -> Self {
        let value = (value.start as u32)..(value.end as u32);
        OpsHandle(value)
    }
}

impl BasicBlock {
    pub fn new(address: u32, ops: OpsHandle) -> Self {
        Self {
            address,
            clif_block: None,
            ops,
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

#[instrument(skip_all)]
fn fetch(params: FetchParams<'_>) -> color_eyre::Result<FetchSummary> {
    let FetchParams {
        pc: initial_pc,
        mem,
        cpu,
        mut cfg,
        mut mapped,
    } = params;

    struct State {
        start_pc: u32,
        node: NodeIndex,
    }

    let entry = cfg.add_node(BasicBlock::default());

    let mut stack = vec![State {
        start_pc: initial_pc,
        node: entry,
    }];

    stack.reserve(256);

    let mut ops = Vec::<DecodedOp>::with_capacity(512);

    while let Some(state) = stack.pop() {
        // if mapped.contains_key(&state.start_pc) {
        //     continue;
        // }
        mapped.to_mut().insert(state.start_pc, state.node);
        cfg[state.node].address = state.start_pc;

        if enabled!(Level::TRACE) {
            tracing::trace!(" ---- begin block ---- ");
            tracing::trace!(" • address: {}", hex(state.start_pc));
            tracing::trace!(" • node: {:?}", state.node);
        }
        let mut lifetime: Option<u32> = None;
        let mut boundary: Option<(DecodedOp, u32)> = None;

        let ops_start = ops.len();
        let mut ops_end = ops.len();

        let mut pc = state.start_pc;
        let mut nop_sequence = 0;

        while pc < u32::MAX {
            let opcode = mem.read::<OpCode, ext::NoExt>(cpu, pc);
            let op = DecodedOp::new(opcode);
            let block_boundary = op.is_block_boundary();
            if enabled!(Level::TRACE) {
                if op.is_illegal() {
                    tracing::trace!("0x{:08X?} illegal {:?}", pc, opcode);
                } else {
                    tracing::trace!(
                        "0x{:08X?}  {}",
                        pc,
                        if block_boundary.is_none() {
                            format!("{}", op)
                        } else {
                            format!("> {} <", op)
                        }
                    );
                }
            }
            if op.is_nop() {
                nop_sequence += 1;
            } else {
                nop_sequence = 0;
            }

            if !(op.is_nop() && nop_sequence > 4) {
                ops.push(op);
                ops_end = ops.len();
            }

            if let Some(lifetime) = lifetime.as_mut() {
                *lifetime = lifetime.saturating_sub(1);
                if *lifetime == 0 {
                    break;
                }
            }

            if block_boundary.is_some() && lifetime.is_none() {
                lifetime = Some(2);
                boundary = Some((op, pc));
            }

            pc += 4;
        }

        let (boundary, pc) = boundary.unwrap();

        cfg[state.node].ops = OpsHandle::from_usize(ops_start..ops_end);

        match boundary.is_block_boundary() {
            None => {}
            Some(BoundaryType::Function) => {}
            Some(BoundaryType::Block { offset }) => {
                let new_address = offset.calculate_address(pc);

                if new_address != initial_pc {
                    if let Some(next) = mapped.get(&new_address) {
                        tracing::trace!(" ! link: {:?} -> {:?}", state.node, next);
                        cfg.add_edge(state.node, *next, ());
                        continue;
                    }
                }

                let new_node = cfg.add_node(BasicBlock::default());
                cfg.add_edge(state.node, new_node, ());

                stack.push(State {
                    start_pc: new_address,
                    node: new_node,
                });
            }
            Some(BoundaryType::BlockSplit { lhs, rhs }) => {
                tracing::trace!(" ! branch: {:?} / {:?}", lhs, rhs);
                for offset in [rhs, lhs] {
                    tracing::trace!(?offset);
                    let new_address = offset.calculate_address(pc);

                    if new_address != initial_pc {
                        if let Some(next) = mapped.get(&new_address) {
                            tracing::trace!(" ! link: {:?} -> {:?}", state.node, next);
                            cfg.add_edge(state.node, *next, ());
                            continue;
                        }
                    }
                    let new_node = cfg.add_node(BasicBlock::default());
                    cfg.add_edge(state.node, new_node, ());

                    stack.push(State {
                        start_pc: new_address,
                        node: new_node,
                    });
                }
            }
        }
    }

    Ok(FetchSummary {
        cfg,
        entry,
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
    pub func_ref_table: &'a FuncRefTable,
    pub state: &'a mut BlockState,
    pub ptr_type: types::Type,
    pub node: NodeIndex,
    pub pc: u32,
    pub cfg: &'a Graph<BasicBlock, ()>,
    pub function_map: &'a FunctionMap,
    pub function_sig_ref: Option<SigRef>,
}

impl<'a, 'b> EmitCtx<'a, 'b> {
    pub fn block(&self) -> &BasicBlock {
        &self.cfg[self.node]
    }
    #[instrument(skip(self))]
    pub fn next_at(&self, idx: usize) -> NodeIndex {
        self.cfg
            .neighbors_directed(self.node, Direction::Outgoing)
            .nth(idx)
            .unwrap()
    }

    pub fn state(&self) -> &BlockState {
        self.state
    }

    pub fn state_mut(&mut self) -> &mut BlockState {
        self.state
    }

    pub fn cache(&self) -> &EntryCache {
        &self.state().cache
    }

    pub fn cache_mut(&mut self) -> &mut EntryCache {
        &mut self.state_mut().cache
    }

    #[instrument(skip(self))]
    pub fn out_params(&self, to: NodeIndex) -> Vec<BlockArg> {
        self.emulator_params()
    }

    pub fn block_call(&mut self, node: NodeIndex) -> (Vec<BlockArg>, BlockCall) {
        let then_block_label = self.cfg[node].clif_block();
        let then_params = self.out_params(node);
        let then_block_call = self
            .fn_builder
            .pure()
            .data_flow_graph_mut()
            .block_call(then_block_label, &then_params);
        (then_params, then_block_call)
    }
    pub fn try_fn_call(&mut self, address: u32) -> Option<[Inst; 2]> {
        match self.function_map.get(address).cloned() {
            Some(func) => {
                let sig_ref = self.function_sig_ref.unwrap_or_else(|| {
                    let mut sig = Signature::new(CallConv::SystemV);
                    sig.params.push(AbiParam::new(self.ptr_type));
                    sig.params.push(AbiParam::new(self.ptr_type));
                    let sig_ref = self.fn_builder.import_signature(sig);
                    self.function_sig_ref = Some(sig_ref);
                    sig_ref
                });
                let ptr_type = self.ptr_type;
                let (callee, iconst) = self.inst(|f| {
                    f.pure()
                        .UnaryImm(
                            Opcode::Iconst,
                            ptr_type,
                            Imm64::new(func.fn_ptr as usize as i64),
                        )
                        .0
                });
                let args = self.emulator_params_raw();
                let call = self.fn_builder.pure().call_indirect(sig_ref, callee, &args);
                Some([iconst, call])
            }
            None => None,
        }
    }
    pub fn neighbour_count(&self) -> usize {
        self.cfg
            .neighbors_directed(self.node, Direction::Outgoing)
            .count()
    }
    pub fn emulator_params_raw(&self) -> [Value; 2] {
        [self.cpu(), self.memory()]
    }
    pub fn emulator_params(&self) -> Vec<BlockArg> {
        self.emulator_params_raw().map(BlockArg::Value).to_vec()
    }

    pub fn cpu(&self) -> Value {
        let block = self.block().clif_block();
        self.fn_builder.block_params(block)[0]
    }
    pub fn memory(&self) -> Value {
        let block = self.cfg[self.node].clif_block();
        self.fn_builder.block_params(block)[1]
    }
    pub fn inst<R: IntoInst>(&mut self, f: impl Fn(&mut FunctionBuilder) -> R) -> (Value, Inst) {
        self.fn_builder.inst(f)
    }
    pub fn emit_get_one(&mut self) -> (Value, Inst) {
        match self.cache().const_one {
            Some(one) => (one, self.fn_builder.Nop()),
            None => {
                let (one, iconsti32) = self.fn_builder.inst(|f| {
                    f.pure()
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
                    f.pure()
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
    pub fn emit_get_register(&mut self, id: u8) -> (Value, Inst) {
        let block = self.block().clif_block();
        match self.cache_mut().registers[id as usize] {
            Some(value) => (value.value, self.fn_builder.Nop()),
            None => {
                let (value, loadreg) = JIT::emit_load_reg()
                    .builder(self.fn_builder)
                    .block(block)
                    .idx(id)
                    .call();
                self.cache_mut().registers[id as usize] = Some(CachedValue {
                    dirty: false,
                    value,
                });
                (value, loadreg)
            }
        }
    }
    pub fn emit_get_cop_register(&mut self, cop: u8, reg: usize) -> (Value, Inst) {
        let block = self.block().clif_block();
        JIT::emit_load_cop_reg()
            .builder(self.fn_builder)
            .block(block)
            .idx(reg)
            .cop(cop)
            .call()
    }
    pub fn update_cache_immediate(&mut self, id: Reg, value: Value) {
        self.cache_mut().registers[id as usize] = Some(CachedValue { dirty: true, value });
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
    pub fn emit_store_pc_imm(&mut self, imm: u32) -> [Inst; 2] {
        let (pc, createpc) = self.inst(|f| {
            f.pure()
                .UnaryImm(Opcode::Iconst, types::I32, Imm64::new(imm as i64))
                .0
        });
        let storepc = JIT::emit_store_pc(self.fn_builder, self.block().clif_block(), pc);
        [createpc, storepc]
    }
    pub fn emit_store_register(&mut self, reg: Reg, value: Value) -> Inst {
        let block = self.block().clif_block();
        JIT::emit_store_reg()
            .builder(self.fn_builder)
            .block(block)
            .idx(reg)
            .value(value)
            .call()
    }
    pub fn emit_store_cop_register(&mut self, cop: u8, reg: usize, value: Value) -> Inst {
        let block = self.block().clif_block();
        JIT::emit_store_cop_reg()
            .builder(self.fn_builder)
            .block(block)
            .idx(reg)
            .value(value)
            .cop(cop)
            .call()
    }
}

#[derive(Builder, derive_more::Debug, Default)]
#[builder(finish_fn(vis = "", name = build_internal))]
pub struct EmitSummary {
    #[builder(field)]
    updates: CacheUpdates,

    #[debug(skip)]
    #[builder(into)]
    pub instructions: Vec<ClifInstruction>,
}

impl<S: emit_summary_builder::State> EmitSummaryBuilder<S> {
    pub fn register_updates<U: Into<Update>>(
        mut self,
        values: impl IntoIterator<Item = (u8, U)>,
    ) -> Self {
        self.updates
            .registers
            .extend(values.into_iter().map(|(reg, value)| (reg, value.into())));
        self
    }

    pub fn hi(mut self, updt: impl Into<Update>) -> Self {
        self.updates.hi = Some(updt.into());
        self
    }

    pub fn lo(mut self, updt: impl Into<Update>) -> Self {
        self.updates.lo = Some(updt.into());
        self
    }
}

impl<S: emit_summary_builder::IsComplete> EmitSummaryBuilder<S> {
    pub fn build(self, fn_builder: &FunctionBuilder) -> EmitSummary {
        if cfg!(debug_assertions) {
            for (_, update) in &self.updates.registers {
                assert_eq!(
                    fn_builder.type_of(update.value.value),
                    types::I32,
                    "emit summary invalid type for register value"
                );
            }
        }
        self.build_internal()
    }
}

#[derive(Builder, derive_more::Debug)]
pub struct Update {
    pub value: CachedValue,
    pub remaining_time: Option<u32>,
}

impl Update {
    pub fn ready(&self) -> bool {
        self.remaining_time.is_none()
    }
    pub fn tick(&mut self) -> bool {
        if let Some(rem) = &mut self.remaining_time {
            *rem = rem.saturating_sub(1);
        }
        match &mut self.remaining_time {
            Some(0) => {
                self.remaining_time = None;
                true
            }
            Some(_) => false,
            None => true,
        }
    }
    pub fn try_value(&self) -> Option<CachedValue> {
        if self.remaining_time.is_some() {
            return None;
        }
        Some(self.value)
    }
    pub fn try_dirty(&self) -> Option<bool> {
        self.try_value().map(|value| value.dirty)
    }
}

impl From<Value> for Update {
    fn from(value: Value) -> Self {
        Update {
            value: CachedValue { dirty: true, value },
            remaining_time: None,
        }
    }
}

#[inline]
pub const fn updtdelay(by: u32, value: Value) -> Update {
    Update {
        value: CachedValue { dirty: true, value },
        remaining_time: Some(by + 1),
    }
}

#[derive(Builder, derive_more::Debug, Default)]
pub struct CacheUpdates {
    pub registers: Vec<(u8, Update)>,
    pub hi: Option<Update>,
    pub lo: Option<Update>,
}

impl CacheUpdates {
    pub fn ready(&self) -> bool {
        let mut ready = true;
        for (_, update) in &self.registers {
            ready &= update.ready();
        }
        if let Some(hi) = &self.hi {
            ready &= hi.ready();
        }
        if let Some(lo) = &self.lo {
            ready &= lo.ready();
        }
        ready
    }
    pub fn tick(&mut self) -> bool {
        let mut ready = true;
        for (_, update) in &mut self.registers {
            ready &= update.tick();
        }
        if let Some(hi) = &mut self.hi {
            ready &= hi.tick();
        }
        if let Some(lo) = &mut self.lo {
            ready &= lo.tick();
        }
        ready
    }
}

#[derive(derive_more::Debug, Clone)]
#[debug("Instr({:?}, {:?}{})", self.queue_type, self.instruction, if self.terminator { ", terminator" } else { "" })]
pub struct ClifInstruction {
    pub queue_type: ClifInstructionQueueType,
    pub instruction: ValidInst,
    pub terminator: bool,
    pub pc: u32,
}

impl ClifInstruction {
    pub fn instruction(&self, ctx: EmitCtx) -> Inst {
        match &self.instruction {
            ValidInst::Value(inst) => *inst,
            ValidInst::Lazy(cb) => cb(ctx),
            ValidInst::LazyBoxed(cb) => cb(ctx),
        }
    }
    pub fn with_pc(self, pc: u32) -> Self {
        Self { pc, ..self }
    }
}

#[derive(derive_more::Debug, Clone)]
pub enum ValidInst {
    #[debug("{:?}", 0)]
    Value(Inst),
    #[debug("lazy")]
    Lazy(fn(EmitCtx) -> Inst),
    #[debug("lazy(boxed)")]
    LazyBoxed(Rc<dyn Fn(EmitCtx) -> Inst + 'static>),
}

impl const From<Inst> for ValidInst {
    fn from(value: Inst) -> Self {
        ValidInst::Value(value)
    }
}

impl const From<fn(EmitCtx) -> Inst> for ValidInst {
    fn from(value: fn(EmitCtx) -> Inst) -> Self {
        ValidInst::Lazy(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ClifInstructionQueueType {
    Now,
    Delayed(u32),
    Bomb(u32),
    Seal,
    Bottom,
}

#[derive(Debug, Clone, Copy)]
pub struct TickSummary {
    ready: bool,
    bomb_went_off: bool,
}

impl ClifInstructionQueueType {
    pub fn is_bomb(&self) -> bool {
        matches!(self, ClifInstructionQueueType::Bomb(_))
    }
    pub fn remaining(&self) -> Option<u32> {
        match self {
            ClifInstructionQueueType::Now => None,
            ClifInstructionQueueType::Delayed(rem) => Some(*rem),
            ClifInstructionQueueType::Bomb(rem) => Some(*rem),
            ClifInstructionQueueType::Bottom => None,
            ClifInstructionQueueType::Seal => None,
        }
    }
    pub fn tick(&mut self) -> TickSummary {
        match self {
            ClifInstructionQueueType::Now => TickSummary {
                ready: true,
                bomb_went_off: false,
            },
            ClifInstructionQueueType::Delayed(by) => {
                *by = by.saturating_sub(1);
                TickSummary {
                    ready: *by == 0,
                    bomb_went_off: false,
                }
            }
            ClifInstructionQueueType::Bomb(by) => {
                *by = by.saturating_sub(1);
                TickSummary {
                    ready: *by == 0,
                    bomb_went_off: *by == 0,
                }
            }
            ClifInstructionQueueType::Seal | ClifInstructionQueueType::Bottom => TickSummary {
                ready: false,
                bomb_went_off: false,
            },
        }
    }
}

#[derive(derive_more::Debug, Clone)]
pub struct BlockState {
    cache: EntryCache,
    #[debug(skip)]
    basic_block: BasicBlock,
    ptr_type: types::Type,
    node: NodeIndex,
}

#[builder]
fn collect_instructions(
    node: NodeIndex,
    fn_builder: &mut FunctionBuilder<'_>,
    func_ref_table: &FuncRefTable,
    state: &mut BlockState,
    function_map: &FunctionMap,
    cfg: &Graph<BasicBlock, ()>,
    ops: &[DecodedOp],
    _cpu: &mut Cpu,
) {
    // TODO: replace vecs with an ordered arena
    // avoid reordering elements on remove
    let mut instruction_queue = SparseQueue::with_capacity(64);
    let mut updates_queue = SparseQueue::with_capacity(64);
    let mut final_instruction = None;

    let ptr_type = state.ptr_type;
    let address = state.basic_block.address;
    let block = state.basic_block.clif_block();

    let mut bomb_signal = None;
    let mut queued_bomb = None;
    let mut cycles_used = 0;
    for (idx, op) in ops.iter().enumerate() {
        let len = ops.len();
        let _span_1 = trace_span!("emit", "%" = %format!("[{}/{len}]", idx+1),  %op, ).entered();
        let pc = address + idx as u32 * 4;
        cycles_used += op.cycles();

        let summary = op.emit_ir(EmitCtx {
            function_sig_ref: None,
            function_map,
            fn_builder,
            func_ref_table,
            ptr_type,
            state,
            pc,
            cfg,
            node,
        });

        if summary.instructions.is_empty() && !op.is_nop() {
            tracing::warn!("emitted no instructions")
        }

        for inst in summary.instructions.into_iter() {
            let inst = inst.with_pc(pc);
            let _span = trace_span!("(inst)",).entered();

            if cfg!(debug_assertions) {
                if matches!(
                    inst.queue_type,
                    ClifInstructionQueueType::Bomb(2..) | ClifInstructionQueueType::Delayed(2..)
                ) && op.hazard().is_none()
                {
                    tracing::warn!(
                        "{} reports no hazard, but emits delayed instruction {:?}. this is likely a bug",
                        op,
                        inst.instruction
                    );
                }
            }

            match inst.queue_type {
                ClifInstructionQueueType::Now => {
                    let i = inst.instruction(EmitCtx {
                        function_map,
                        function_sig_ref: None,
                        fn_builder,
                        func_ref_table,
                        ptr_type,
                        state,
                        pc,
                        cfg,
                        node,
                    });
                    fn_builder.append_inst(i, inst.terminator);
                }
                ClifInstructionQueueType::Seal => {
                    let i = inst.instruction(EmitCtx {
                        fn_builder,
                        function_sig_ref: None,
                        function_map,
                        func_ref_table,
                        ptr_type,
                        state,
                        pc,
                        cfg,
                        node,
                    });
                    fn_builder.append_inst(i, inst.terminator);
                    final_instruction = Some(inst);
                }
                ClifInstructionQueueType::Bottom => {
                    instruction_queue.push(inst);
                }
                ClifInstructionQueueType::Delayed(_) => {
                    instruction_queue.push(inst);
                }
                ClifInstructionQueueType::Bomb(_) => {
                    queued_bomb = Some(inst.clone());
                    instruction_queue.push(inst.clone());
                }
            }
        }

        let _span = trace_span!("(post)",).entered();

        // immediate updates take precedence over scheduled ones
        if summary.updates.ready() {
            JIT::apply_cache_updates()
                .updates(&summary.updates)
                .cache(&mut state.cache)
                .call();
        } else {
            updates_queue.push(summary.updates);
        }

        updates_queue.retain_mut(|updates: &mut CacheUpdates| {
            let res = !updates.tick();
            JIT::apply_cache_updates()
                .updates(updates)
                .cache(&mut state.cache)
                .call();
            res
        });

        instruction_queue.retain_mut(|inst: &mut ClifInstruction| {
            let tick = inst.queue_type.tick();
            if tick.bomb_went_off {
                bomb_signal = Some(inst.clone());
                return false;
            }
            if tick.ready {
                let i = inst.instruction(EmitCtx {
                    fn_builder,
                    function_sig_ref: None,
                    function_map,
                    func_ref_table,
                    ptr_type,
                    state,
                    pc,
                    cfg,
                    node,
                });
                fn_builder.append_inst(i, inst.terminator);
                false
            } else {
                true
            }
        });

        if bomb_signal.is_some() {
            break;
        }
    }

    let _span = trace_span!("emit(closing)").entered();

    updates_queue.decay(|updates: &mut CacheUpdates| {
        let res = !updates.tick();
        JIT::apply_cache_updates()
            .updates(updates)
            .cache(&mut state.cache)
            .call();
        res
    });

    let updates = JIT::emit_updates()
        .cache(&state.cache)
        .builder(fn_builder)
        .block(block)
        .delta_cycles(cycles_used)
        .call();
    for inst in updates {
        let inst = bottom(inst);
        let _span = trace_span!("emit(updates)", ?inst).entered();
        instruction_queue.push(inst);
    }

    if !instruction_queue.is_empty() {
        instruction_queue.as_vec_mut().sort_by(|a, b| {
            a.as_ref()
                .map(|a| a.queue_type.remaining())
                .cmp(&b.as_ref().map(|b| b.queue_type.remaining()))
        });
        // emit remaining delayed instructions
        instruction_queue
            .iter()
            .filter(|inst| matches!(inst.queue_type, ClifInstructionQueueType::Delayed(_)))
            .for_each(|inst| {
                let i = inst.instruction(EmitCtx {
                    fn_builder,
                    function_sig_ref: None,
                    function_map,
                    func_ref_table,
                    ptr_type,
                    state,
                    pc: inst.pc,
                    cfg,
                    node,
                });
                fn_builder.append_inst(i, inst.terminator);
            });
        // emit all bottom instructions
        instruction_queue
            .iter()
            .filter(|inst| matches!(inst.queue_type, ClifInstructionQueueType::Bottom))
            .for_each(|inst| {
                let i = inst.instruction(EmitCtx {
                    fn_builder,
                    function_sig_ref: None,
                    function_map,
                    func_ref_table,
                    ptr_type,
                    state,
                    pc: inst.pc,
                    cfg,
                    node,
                });
                fn_builder.append_inst(i, inst.terminator);
            });
    }

    if let Some(bomb) = bomb_signal.or(queued_bomb) {
        let i = bomb.instruction(EmitCtx {
            fn_builder,
            function_sig_ref: None,
            function_map,
            func_ref_table,
            ptr_type,
            state,
            pc: bomb.pc,
            cfg,
            node,
        });
        fn_builder.append_inst(i, bomb.terminator);
    } else if let Some(final_instruction) = final_instruction {
        let i = final_instruction.instruction(EmitCtx {
            fn_builder,
            function_sig_ref: None,
            func_ref_table,
            function_map,
            ptr_type,
            state,
            pc: final_instruction.pc,
            cfg,
            node,
        });
        fn_builder.append_inst(i, final_instruction.terminator);
    } else {
        // fn_builder.ins().return_(&[]);
        tracing::error!("block did not emit a terminator!")
    }

    let i = fn_builder.func.layout.block_insts(block).count();
    tracing::trace!("compiled into {i} instructions")
}

#[inline]
pub const fn now(inst: impl const Into<ValidInst>) -> ClifInstruction {
    ClifInstruction {
        instruction: inst.into(),
        queue_type: ClifInstructionQueueType::Now,
        terminator: false,
        pc: 0,
    }
}

#[inline]
pub const fn delayed(by: u32, inst: impl const Into<ValidInst>) -> ClifInstruction {
    ClifInstruction {
        instruction: inst.into(),
        queue_type: ClifInstructionQueueType::Delayed(by + 1),
        terminator: false,
        pc: 0,
    }
}

#[inline]
pub const fn delayed_maybe(by: Option<u32>, inst: impl const Into<ValidInst>) -> ClifInstruction {
    match by {
        Some(by) => ClifInstruction {
            instruction: inst.into(),
            queue_type: ClifInstructionQueueType::Delayed(by + 1),
            terminator: false,
            pc: 0,
        },
        None => ClifInstruction {
            instruction: inst.into(),
            queue_type: ClifInstructionQueueType::Now,
            terminator: false,
            pc: 0,
        },
    }
}

#[inline]
pub const fn bomb(by: u32, inst: impl const Into<ValidInst>) -> ClifInstruction {
    ClifInstruction {
        instruction: inst.into(),
        queue_type: ClifInstructionQueueType::Bomb(by + 1),
        terminator: false,
        pc: 0,
    }
}

#[inline]
pub const fn seal(inst: impl const Into<ValidInst>) -> ClifInstruction {
    ClifInstruction {
        queue_type: ClifInstructionQueueType::Seal,
        instruction: inst.into(),
        terminator: false,
        pc: 0,
    }
}

#[inline]
pub const fn bottom(inst: impl const Into<ValidInst>) -> ClifInstruction {
    ClifInstruction {
        queue_type: ClifInstructionQueueType::Bottom,
        instruction: inst.into(),
        terminator: false,
        pc: 0,
    }
}

pub const fn lazy(cb: fn(EmitCtx) -> Inst) -> ValidInst {
    ValidInst::Lazy(cb)
}

// TODO: replace with closure struct enum
pub fn lazy_boxed<F: Fn(EmitCtx) -> Inst + 'static>(cb: F) -> ValidInst {
    ValidInst::LazyBoxed(Rc::new(cb))
}

#[inline]
pub fn terminator(inst: ClifInstruction) -> ClifInstruction {
    ClifInstruction {
        terminator: true,
        ..inst
    }
}

trait BasicBlockIndexOps {
    fn for_block(&self, block: &BasicBlock) -> &[DecodedOp];
}

impl BasicBlockIndexOps for &[DecodedOp] {
    fn for_block(&self, block: &BasicBlock) -> &[DecodedOp] {
        let end = (block.ops.0.end as usize).min(self.len());
        &self[(block.ops.0.start as usize)..(end)]
    }
}

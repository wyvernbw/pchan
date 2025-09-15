use crate::{
    FnBuilderExt, cranelift_bs::*, dynarec::BasicBlock, dynarec::CacheDependency,
    dynarec::EntryCache, jit::JIT,
};
use bon::Builder;
use cranelift::frontend::FuncInstBuilder;
use enum_dispatch::enum_dispatch;
use pchan_macros::OpCode;
use petgraph::prelude::*;
use std::{collections::HashMap, fmt::Display, ops::Range};
use thiserror::Error;
use tracing::instrument;

// alu
pub mod addiu;
pub mod addu;
pub mod and;
pub mod andi;
pub mod lui;
pub mod mfhi;
pub mod mflo;
pub mod mthi;
pub mod mtlo;
pub mod mult;
pub mod multu;
pub mod nor;
pub mod or;
pub mod ori;
pub mod sll;
pub mod sllv;
pub mod slt;
pub mod slti;
pub mod sltiu;
pub mod sltu;
pub mod sra;
pub mod srav;
pub mod srl;
pub mod srlv;
pub mod subu;
pub mod xor;
pub mod xori;

// jumps
pub mod beq;
pub mod bne;
pub mod j;
pub mod jal;
pub mod jalr;
pub mod jr;

// loads
pub mod lb;
pub mod lbu;
pub mod lh;
pub mod lhu;
pub mod lw;

// stores
pub mod sb;
pub mod sh;
pub mod sw;

// cop
pub mod mtc;

pub mod prelude {
    pub use super::OpCode;
    pub use super::addiu::*;
    pub use super::addu::*;
    pub use super::and::*;
    pub use super::andi::*;
    pub use super::beq::*;
    pub use super::bne::*;
    pub use super::j::*;
    pub use super::jal::*;
    pub use super::jalr::*;
    pub use super::jr::*;
    pub use super::lb::*;
    pub use super::lbu::*;
    pub use super::lh::*;
    pub use super::lhu::*;
    pub use super::lui::*;
    pub use super::lw::*;
    pub use super::mfhi::*;
    pub use super::mflo::*;
    pub use super::mtc::*;
    pub use super::mthi::*;
    pub use super::mtlo::*;
    pub use super::mult::*;
    pub use super::multu::*;
    pub use super::nop;
    pub use super::nor::*;
    pub use super::or::*;
    pub use super::ori::*;
    pub use super::sb::*;
    pub use super::sh::*;
    pub use super::sll::*;
    pub use super::sllv::*;
    pub use super::slt::*;
    pub use super::slti::*;
    pub use super::sltiu::*;
    pub use super::sltu::*;
    pub use super::sra::*;
    pub use super::srav::*;
    pub use super::srl::*;
    pub use super::srlv::*;
    pub use super::subu::*;
    pub use super::sw::*;
    pub use super::xor::*;
    pub use super::xori::*;
    pub use super::{
        BoundaryType, CachedValue, CopOp, DecodedOp, EmitParams, EmitSummary, MipsOffset, Op,
        PrimeOp, SecOp, TryFromOpcodeErr,
    };
}

use prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct OpCode(pub u32);

impl core::fmt::Debug for OpCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Op")
            .field_with(|f| write!(f, "0x{:08X}", &self.0))
            .finish()
    }
}

pub const fn nop() -> OpCode {
    OpCode::NOP
}

impl OpCode {
    pub const NOP: OpCode = OpCode(0x00000000);

    #[inline]
    pub const fn primary(&self) -> PrimeOp {
        let code = self.0 >> 26;
        PrimeOp::MAP[code as usize]
    }
    #[inline]
    pub const fn secondary(&self) -> SecOp {
        let code = self.0 & 0x3F;
        SecOp::MAP[code as usize]
    }
    #[inline]
    pub const fn cop(&self) -> CopOp {
        let code = self.bits(21..26);
        CopOp::MAP[code as usize]
    }
    #[inline]
    pub const fn bits(&self, range: Range<u8>) -> u32 {
        let mask = (0xFFFFFFFFu32.unbounded_shl(range.start as u32))
            ^ ((0xFFFFFFFFu32).unbounded_shl(range.end as u32));
        (self.0 & mask).unbounded_shr(range.start as u32)
    }
    #[inline]
    pub const fn set_bits(&self, range: Range<u8>, value: u32) -> Self {
        let mask = (0xFFFFFFFFu32.unbounded_shl(range.start as u32))
            ^ (0xFFFFFFFFu32.unbounded_shl(range.end as u32));

        let cleared = self.0 & !mask;

        let shifted = (value << range.start) & mask;

        Self(cleared | shifted)
    }

    pub const fn with_primary(self, primary: PrimeOp) -> Self {
        OpCode((self.0 & 0x03FF_FFFF) | ((primary as u32) << 26))
    }
    pub const fn with_secondary(self, secondary: SecOp) -> Self {
        OpCode((self.0 & 0xFFFF_FFE0) | (secondary as u32))
    }
    pub fn as_primary(self, primary: PrimeOp) -> Result<Self, TryFromOpcodeErr> {
        if self.primary() == primary {
            Ok(self)
        } else {
            Err(TryFromOpcodeErr::InvalidHeader)
        }
    }
    pub fn as_secondary(self, secondary: SecOp) -> Result<Self, TryFromOpcodeErr> {
        if self.secondary() == secondary {
            Ok(self)
        } else {
            Err(TryFromOpcodeErr::InvalidHeader)
        }
    }
    pub fn as_primary_cop(self) -> Result<(Self, u8), TryFromOpcodeErr> {
        match self.primary() {
            PrimeOp::COP0 => Ok((self, 0)),
            PrimeOp::COP1 => Err(TryFromOpcodeErr::InvalidCoprocessor(PrimeOp::COP1)),
            PrimeOp::COP2 => Ok((self, 2)),
            PrimeOp::COP3 => Err(TryFromOpcodeErr::InvalidCoprocessor(PrimeOp::COP3)),
            _ => Err(TryFromOpcodeErr::InvalidHeader),
        }
    }
    pub fn as_cop(self, cop: CopOp) -> Result<Self, TryFromOpcodeErr> {
        if self.cop() == cop {
            Ok(self)
        } else {
            Err(TryFromOpcodeErr::InvalidHeader)
        }
    }
    pub fn with_primary_cop(self, cop: u8) -> Self {
        match cop {
            0 => self.with_primary(PrimeOp::COP0),
            1 => self.with_primary(PrimeOp::COP1),
            2 => self.with_primary(PrimeOp::COP2),
            3 => self.with_primary(PrimeOp::COP3),
            _ => {
                tracing::error!("tried to construct op for inexistent cop{}", cop);
                self
            }
        }
    }
    pub fn with_cop(self, cop: CopOp) -> Self {
        self.set_bits(21..26, cop as u32)
    }
}

#[repr(u8)]
#[derive(OpCode, Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PrimeOp {
    SPECIAL = 0x00,
    BCONDZ = 0x01,
    J = 0x02,
    JAL = 0x03,
    BEQ = 0x04,
    BNE = 0x05,
    BLEZ = 0x06,
    BGTZ = 0x07,

    ADDI = 0x08,
    ADDIU = 0x09,
    SLTI = 0x0A,
    SLTIU = 0x0B,
    ANDI = 0x0C,
    ORI = 0x0D,
    XORI = 0x0E,
    LUI = 0x0F,

    COP0 = 0x10,
    COP1 = 0x11,
    COP2 = 0x12,
    COP3 = 0x13,

    LB = 0x20,
    LH = 0x21,
    LWL = 0x22,
    LW = 0x23,
    LBU = 0x24,
    LHU = 0x25,
    LWR = 0x26,

    SB = 0x28,
    SH = 0x29,
    SWL = 0x2A,
    SW = 0x2B,
    SWR = 0x2E,

    LWC0 = 0x30,
    LWC1 = 0x31,
    LWC2 = 0x32,
    LWC3 = 0x33,

    SWC0 = 0x38,
    SWC1 = 0x39,
    SWC2 = 0x3A,
    SWC3 = 0x3B,

    #[opcode(default)]
    ILLEGAL,
}

#[repr(u8)]
#[derive(OpCode, Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum SecOp {
    // Shift instructions
    SLL = 0x00,
    SRL = 0x02,
    SRA = 0x03,
    SLLV = 0x04,
    SRLV = 0x06,
    SRAV = 0x07,

    // Jump instructions
    JR = 0x08,
    JALR = 0x09,

    // Move from/to special registers
    MFHI = 0x10,
    MTHI = 0x11,
    MFLO = 0x12,
    MTLO = 0x13,

    // Multiply/Divide
    MULT = 0x18,
    MULTU = 0x19,
    DIV = 0x1A,
    DIVU = 0x1B,

    // Arithmetic
    ADD = 0x20,
    ADDU = 0x21,
    SUB = 0x22,
    SUBU = 0x23,
    AND = 0x24,
    OR = 0x25,
    XOR = 0x26,
    NOR = 0x27,
    SLT = 0x2A,
    SLTU = 0x2B,

    // System
    SYSCALL = 0x0C,
    BREAK = 0x0D,

    #[opcode(default)]
    ILLEGAL,
}

#[repr(u8)]
#[derive(OpCode, Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum CopOp {
    MTCn = 0b00100,

    #[opcode(default)]
    ILLEGAL,
}

#[derive(Debug, Clone, Copy)]
pub struct CachedValue {
    pub dirty: bool,
    pub value: Value,
}

#[derive(Builder)]
pub struct EmitParams<'a, 'b> {
    pub fn_builder: &'a mut FunctionBuilder<'b>,
    pub ptr_type: types::Type,
    pub cache: &'a mut EntryCache,
    pub node: NodeIndex,
    pub pc: u32,
    pub cfg: &'a Graph<BasicBlock, ()>,
    pub deps_map: &'a HashMap<Block, CacheDependency>,
}

impl<'a, 'b> EmitParams<'a, 'b> {
    fn block(&self) -> &BasicBlock {
        &self.cfg[self.node]
    }
    fn ins<'short>(&'short mut self) -> FuncInstBuilder<'short, 'b> {
        self.fn_builder.ins()
    }
    fn next_at(&self, idx: usize) -> &BasicBlock {
        let idx = self
            .cfg
            .neighbors_directed(self.node, Direction::Outgoing)
            .nth(idx)
            .unwrap();
        &self.cfg[idx]
    }
    fn neighbour_count(&self) -> usize {
        self.cfg
            .neighbors_directed(self.node, Direction::Outgoing)
            .count()
    }
    fn emulator_params(&self) -> Vec<BlockArg> {
        vec![
            BlockArg::Value(self.cpu()),
            BlockArg::Value(self.memory()),
            BlockArg::Value(self.mem_map()),
        ]
    }
    #[instrument(skip(self))]
    fn out_params(&self, to: Block) -> Vec<BlockArg> {
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
    fn cpu(&self) -> Value {
        let block = self.block().clif_block();
        self.fn_builder.block_params(block)[0]
    }
    fn memory(&self) -> Value {
        let block = self.cfg[self.node].clif_block();
        self.fn_builder.block_params(block)[1]
    }
    fn mem_map(&self) -> Value {
        let block = self.block().clif_block();
        self.fn_builder.block_params(block)[2]
    }
    fn emit_get_one(&mut self) -> Value {
        match self.cache.const_one {
            Some(one) => one,
            None => {
                let one = self.ins().iconst(types::I32, 1);
                self.cache.const_one = Some(one);
                one
            }
        }
    }
    fn emit_get_zero_i64(&mut self, fn_builder: &mut FunctionBuilder) -> Value {
        match self.cache.const_zero_i64 {
            Some(zero) => zero,
            None => {
                let zero = fn_builder.ins().iconst(types::I64, 0);
                self.cache.const_zero_i64 = Some(zero);
                zero
            }
        }
    }
    fn emit_get_zero(&mut self) -> Value {
        self.emit_get_register(0)
    }
    fn emit_get_hi(&mut self) -> Value {
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
    fn emit_get_lo(&mut self) -> Value {
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
    fn emit_get_register(&mut self, id: usize) -> Value {
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
    fn emit_get_cop_register(
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
    fn update_cache_immediate(&mut self, id: usize, value: Value) {
        self.cache.registers[id] = Some(CachedValue {
            dirty: false,
            value,
        });
    }
    fn emit_map_address_to_host(&mut self, address: Value) -> Value {
        let mem_map = self.mem_map();
        JIT::emit_map_address_to_host()
            .fn_builder(self.fn_builder)
            .ptr_type(self.ptr_type)
            .address(address)
            .mem_map_ptr(mem_map)
            .call()
    }
    fn emit_map_address_to_physical(&mut self, address: Value) -> Value {
        JIT::emit_map_address_to_physical()
            .fn_builder(self.fn_builder)
            .address(address)
            .call()
    }
    fn emit_store_pc(&mut self, value: Value) {
        let cpu = self.cpu();
        JIT::emit_store_pc(self.fn_builder, value, cpu);
    }
    fn emit_store_register(&mut self, reg: usize, value: Value) {
        let block = self.block().clif_block();
        JIT::emit_store_reg()
            .builder(self.fn_builder)
            .block(block)
            .idx(reg)
            .value(value)
            .call();
    }
    fn emit_store_cop_register(&mut self, cop: u8, reg: usize, value: Value) {
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
    #[builder(field = Vec::with_capacity(32))]
    pub delayed_register_updates: Vec<(usize, CachedValue)>,
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
    #[builder(default)]
    pub finished_block: bool,
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
    pub fn delayed_register_updates(
        mut self,
        values: impl IntoIterator<Item = (usize, Value)>,
    ) -> Self {
        self.delayed_register_updates.extend(
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
            for (_, value) in &self.delayed_register_updates {
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

#[derive(Debug, Error)]
pub enum TryFromOpcodeErr {
    #[error("invalid header")]
    InvalidHeader,
    #[error("unknown instruction")]
    UnknownInstruction,
    #[error("found opcode for invalid coprocessor {0:?}")]
    InvalidCoprocessor(PrimeOp),
}

#[derive(Debug, Clone, Copy)]
pub enum MipsOffset {
    RegionJump(u32),
    Relative(i32),
}

impl MipsOffset {
    pub fn calculate_address(self, base: u32) -> u32 {
        match self {
            MipsOffset::RegionJump(addr) => (base & 0xF000_0000) + addr,
            MipsOffset::Relative(offset) => base.wrapping_add_signed(offset),
        }
    }
}

#[derive(Debug)]
pub enum BoundaryType {
    Block { offset: MipsOffset },
    BlockSplit { lhs: MipsOffset, rhs: MipsOffset },
    Function { auto_set_pc: bool },
}

#[enum_dispatch(DecodedOp)]
pub trait Op: Sized + Display + TryFrom<OpCode> {
    fn invalidates_cache_at(&self) -> Option<u32> {
        None
    }
    fn is_block_boundary(&self) -> Option<BoundaryType>;
    fn into_opcode(self) -> crate::cpu::ops::OpCode;
    fn emit_ir(&self, state: EmitParams) -> Option<EmitSummary>;
}

#[derive(Debug, Clone, Copy)]
pub struct NOP;

impl Op for NOP {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::NOP
    }

    fn emit_ir(&self, _state: EmitParams) -> Option<EmitSummary> {
        None
    }
}

impl TryFrom<OpCode> for NOP {
    type Error = String;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        if value.0 == 0 {
            return Ok(NOP);
        }
        Err("Not nop".to_string())
    }
}

impl Display for NOP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "nop")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HaltBlock;

impl Display for HaltBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "haltblock")
    }
}

impl Op for HaltBlock {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::Function { auto_set_pc: true })
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode(69420)
    }

    fn emit_ir(&self, state: EmitParams) -> Option<EmitSummary> {
        state.fn_builder.ins().return_(&[]);
        None
    }
}

impl TryFrom<OpCode> for HaltBlock {
    type Error = String;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        if value.0 == 69420 {
            return Ok(HaltBlock);
        }
        Err("not halt".to_string())
    }
}

#[derive(Debug, Clone, Copy, strum::Display)]
#[enum_dispatch]
#[allow(clippy::upper_case_acronyms)]
pub enum DecodedOp {
    #[strum(transparent)]
    NOP(NOP),
    #[strum(transparent)]
    HaltBlock(HaltBlock),
    #[strum(transparent)]
    LB(LB),
    #[strum(transparent)]
    LBU(LBU),
    #[strum(transparent)]
    LH(LH),
    #[strum(transparent)]
    LHU(LHU),
    #[strum(transparent)]
    LW(LW),
    #[strum(transparent)]
    SB(SB),
    #[strum(transparent)]
    SH(SH),
    #[strum(transparent)]
    SW(SW),
    #[strum(transparent)]
    ADDU(ADDU),
    #[strum(transparent)]
    ADDIU(ADDIU),
    #[strum(transparent)]
    SUBU(SUBU),
    #[strum(transparent)]
    J(J),
    #[strum(transparent)]
    BEQ(BEQ),
    #[strum(transparent)]
    BNE(BNE),
    #[strum(transparent)]
    SLT(SLT),
    #[strum(transparent)]
    SLTU(SLTU),
    #[strum(transparent)]
    SLTI(SLTI),
    #[strum(transparent)]
    SLTIU(SLTIU),
    #[strum(transparent)]
    AND(AND),
    #[strum(transparent)]
    OR(OR),
    #[strum(transparent)]
    XOR(XOR),
    #[strum(transparent)]
    NOR(NOR),
    #[strum(transparent)]
    ANDI(ANDI),
    #[strum(transparent)]
    ORI(ORI),
    #[strum(transparent)]
    XORI(XORI),
    #[strum(transparent)]
    SLLV(SLLV),
    #[strum(transparent)]
    SRLV(SRLV),
    #[strum(transparent)]
    SRAV(SRAV),
    #[strum(transparent)]
    SLL(SLL),
    #[strum(transparent)]
    SRL(SRL),
    #[strum(transparent)]
    SRA(SRA),
    #[strum(transparent)]
    LUI(LUI),
    #[strum(transparent)]
    MULT(MULT),
    #[strum(transparent)]
    MULTU(MULTU),
    #[strum(transparent)]
    MFLO(MFLO),
    #[strum(transparent)]
    MFHI(MFHI),
    #[strum(transparent)]
    MTHI(MTHI),
    #[strum(transparent)]
    MTLO(MTLO),
    #[strum(transparent)]
    JAL(JAL),
    #[strum(transparent)]
    JR(JR),
    #[strum(transparent)]
    JALR(JALR),

    // cop
    #[strum(transparent)]
    MTCn(MTCn),
}

impl TryFrom<OpCode> for DecodedOp {
    type Error = TryFromOpcodeErr;

    #[instrument(err)]
    fn try_from(opcode: OpCode) -> Result<Self, Self::Error> {
        if opcode.0 == 69420 {
            return Ok(DecodedOp::HaltBlock(HaltBlock));
        }
        if opcode == OpCode::NOP {
            return Ok(DecodedOp::NOP(NOP));
        }

        macro_rules! copn {
            () => {
                (
                    PrimeOp::COP0 | PrimeOp::COP1 | PrimeOp::COP2 | PrimeOp::COP3,
                    _,
                    CopOp::MTCn,
                )
            };
        }

        match (opcode.primary(), opcode.secondary(), opcode.cop()) {
            copn!() => MTCn::try_from(opcode).map(Self::MTCn),
            (PrimeOp::SPECIAL, SecOp::JALR, _) => JALR::try_from(opcode).map(Self::JALR),
            (PrimeOp::SPECIAL, SecOp::JR, _) => JR::try_from(opcode).map(Self::JR),
            (PrimeOp::JAL, _, _) => JAL::try_from(opcode).map(Self::JAL),
            (PrimeOp::SPECIAL, SecOp::MTLO, _) => MTLO::try_from(opcode).map(Self::MTLO),
            (PrimeOp::SPECIAL, SecOp::MTHI, _) => MTHI::try_from(opcode).map(Self::MTHI),
            (PrimeOp::SPECIAL, SecOp::MFHI, _) => MFHI::try_from(opcode).map(Self::MFHI),
            (PrimeOp::SPECIAL, SecOp::MFLO, _) => MFLO::try_from(opcode).map(Self::MFLO),
            (PrimeOp::SPECIAL, SecOp::MULTU, _) => MULTU::try_from(opcode).map(Self::MULTU),
            (PrimeOp::SPECIAL, SecOp::MULT, _) => MULT::try_from(opcode).map(Self::MULT),
            (PrimeOp::LUI, _, _) => LUI::try_from(opcode).map(Self::LUI),
            (PrimeOp::SPECIAL, SecOp::SRA, _) => SRA::try_from(opcode).map(Self::SRA),
            (PrimeOp::SPECIAL, SecOp::SRL, _) => SRL::try_from(opcode).map(Self::SRL),
            (PrimeOp::SPECIAL, SecOp::SLL, _) => SLL::try_from(opcode).map(Self::SLL),
            (PrimeOp::SPECIAL, SecOp::SRAV, _) => SRAV::try_from(opcode).map(Self::SRAV),
            (PrimeOp::SPECIAL, SecOp::SRLV, _) => SRLV::try_from(opcode).map(Self::SRLV),
            (PrimeOp::SPECIAL, SecOp::SLLV, _) => SLLV::try_from(opcode).map(Self::SLLV),
            (PrimeOp::XORI, _, _) => XORI::try_from(opcode).map(Self::XORI),
            (PrimeOp::ORI, _, _) => ORI::try_from(opcode).map(Self::ORI),
            (PrimeOp::ANDI, _, _) => ANDI::try_from(opcode).map(Self::ANDI),
            (PrimeOp::SPECIAL, SecOp::NOR, _) => NOR::try_from(opcode).map(Self::NOR),
            (PrimeOp::SPECIAL, SecOp::XOR, _) => XOR::try_from(opcode).map(Self::XOR),
            (PrimeOp::SPECIAL, SecOp::OR, _) => OR::try_from(opcode).map(Self::OR),
            (PrimeOp::SPECIAL, SecOp::AND, _) => AND::try_from(opcode).map(Self::AND),
            (PrimeOp::SLTIU, _, _) => SLTIU::try_from(opcode).map(Self::SLTIU),
            (PrimeOp::SLTI, _, _) => SLTI::try_from(opcode).map(Self::SLTI),
            (PrimeOp::SPECIAL, SecOp::SLTU, _) => SLTU::try_from(opcode).map(Self::SLTU),
            (PrimeOp::SPECIAL, SecOp::SLT, _) => SLT::try_from(opcode).map(Self::SLT),
            (PrimeOp::BEQ, _, _) => BEQ::try_from(opcode).map(Self::BEQ),
            (PrimeOp::BNE, _, _) => BNE::try_from(opcode).map(Self::BNE),
            (PrimeOp::J, _, _) => J::try_from(opcode).map(Self::J),
            (PrimeOp::ADDIU | PrimeOp::ADDI, _, _) => ADDIU::try_from(opcode).map(Self::ADDIU),
            (PrimeOp::SPECIAL, SecOp::SUBU | SecOp::SUB, _) => {
                // TODO: implement SUB separately from SUBU
                SUBU::try_from(opcode).map(Self::SUBU)
            }
            (PrimeOp::SPECIAL, SecOp::ADDU | SecOp::ADD, _) => {
                // TODO: implement ADD separately from ADDU
                ADDU::try_from(opcode).map(Self::ADDU)
            }
            (PrimeOp::SW, _, _) => SW::try_from(opcode).map(Self::SW),
            (PrimeOp::SH, _, _) => SH::try_from(opcode).map(Self::SH),
            (PrimeOp::SB, _, _) => SB::try_from(opcode).map(Self::SB),
            (PrimeOp::LW, _, _) => LW::try_from(opcode).map(Self::LW),
            (PrimeOp::LHU, _, _) => LHU::try_from(opcode).map(Self::LHU),
            (PrimeOp::LH, _, _) => LH::try_from(opcode).map(Self::LH),
            (PrimeOp::LB, _, _) => LB::try_from(opcode).map(Self::LB),
            (PrimeOp::LBU, _, _) => LBU::try_from(opcode).map(Self::LBU),
            _ => Err(TryFromOpcodeErr::UnknownInstruction),
        }
    }
}

impl DecodedOp {
    pub fn new(opcode: OpCode) -> Self {
        Self::try_from(opcode).unwrap()
    }
}

#[derive(Debug, Clone, Copy, derive_more::Display)]
pub struct OpForceBoundary(pub DecodedOp);

impl Op for OpForceBoundary {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        let auto_set_pc =
            if let Some(BoundaryType::Function { auto_set_pc }) = self.0.is_block_boundary() {
                auto_set_pc
            } else {
                false
            };
        Some(BoundaryType::Function { auto_set_pc })
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        self.0.into_opcode()
    }

    fn emit_ir(&self, state: EmitParams) -> Option<EmitSummary> {
        self.0.emit_ir(state)
    }
}

impl TryFrom<OpCode> for OpForceBoundary {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        DecodedOp::try_from(value).map(Self)
    }
}

#[cfg(test)]
mod decode_display_tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::cpu::ops::DecodedOp;
    use crate::cpu::ops::prelude::*;

    #[rstest]
    #[case::nop(DecodedOp::new(nop()), "nop")]
    #[case::lb(DecodedOp::new(lb(8, 9, 4)), "lb $t0 $t1 4")]
    #[case::lbu(DecodedOp::new(lbu(8, 9, 4)), "lbu $t0 $t1 4")]
    #[case::lh(DecodedOp::new(lh(8, 9, 4)), "lh $t0 $t1 4")]
    #[case::lhu(DecodedOp::new(lhu(8, 9, 4)), "lhu $t0 $t1 4")]
    #[case::lw(DecodedOp::new(lw(8, 9, 4)), "lw $t0 $t1 4")]
    #[case::sb(DecodedOp::new(sb(8, 9, 4)), "sb $t0 $t1 4")]
    #[case::sh(DecodedOp::new(sh(8, 9, 4)), "sh $t0 $t1 4")]
    #[case::sw(DecodedOp::new(sw(8, 9, 4)), "sw $t0 $t1 4")]
    #[case::addu(DecodedOp::new(addu(8, 9, 10)), "addu $t0 $t1 $t2")]
    #[case::addiu(DecodedOp::new(addiu(8, 9, 123)), "addiu $t0 $t1 123")]
    #[case::subu(DecodedOp::new(subu(8, 9, 10)), "subu $t0 $t1 $t2")]
    #[case::j(DecodedOp::new(j(0x0040_0000)), "j 0x00400000")]
    #[case::beq(DecodedOp::new(beq(8, 9, 16)), "beq $t0 $t1 0x00000010")]
    #[case::bne(DecodedOp::new(bne(8, 9, 16)), "bne $t0 $t1 0x00000010")]
    #[case::slt(DecodedOp::new(slt(8, 9, 10)), "slt $t0 $t1 $t2")]
    #[case::sltu(DecodedOp::new(sltu(8, 9, 10)), "sltu $t0 $t1 $t2")]
    #[case::slti(DecodedOp::new(slti(8, 9, 32)), "slti $t0 $t1 32")]
    #[case::sltiu(DecodedOp::new(sltiu(8, 9, 32)), "sltiu $t0 $t1 32")]
    #[case::and(DecodedOp::new(and(8, 9, 10)), "and $t0 $t1 $t2")]
    #[case::or(DecodedOp::new(or(8, 9, 10)), "or $t0 $t1 $t2")]
    #[case::xor(DecodedOp::new(xor(8, 9, 10)), "xor $t0 $t1 $t2")]
    #[case::nor(DecodedOp::new(nor(8, 9, 10)), "nor $t0 $t1 $t2")]
    #[case::andi(DecodedOp::new(andi(8, 9, 4)), "andi $t0 $t1 4")]
    #[case::ori(DecodedOp::new(ori(8, 9, 4)), "ori $t0 $t1 4")]
    #[case::xori(DecodedOp::new(xori(8, 9, 4)), "xori $t0 $t1 4")]
    #[case::sllv(DecodedOp::new(sllv(8, 9, 10)), "sllv $t0 $t1 $t2")]
    #[case::srlv(DecodedOp::new(srlv(8, 9, 10)), "srlv $t0 $t1 $t2")]
    #[case::srav(DecodedOp::new(srav(8, 9, 10)), "srav $t0 $t1 $t2")]
    #[case::sll(DecodedOp::new(sll(8, 9, 4)), "sll $t0 $t1 4")]
    #[case::srl(DecodedOp::new(srl(8, 9, 4)), "srl $t0 $t1 4")]
    #[case::sra(DecodedOp::new(sra(8, 9, 4)), "sra $t0 $t1 4")]
    #[case::lui(DecodedOp::new(lui(8, 32)), "lui $t0 32")]
    #[case::mult(DecodedOp::new(mult(8, 9)), "mult $t0 $t1")]
    #[case::jal(DecodedOp::new(jal(0x0040_0000)), "jal 0x00400000")]
    #[case::multu(DecodedOp::new(multu(8, 9)), "multu $t0 $t1")]
    #[case::mflo(DecodedOp::new(mflo(8)), "mflo $t0")]
    #[case::mfhi(DecodedOp::new(mfhi(8)), "mfhi $t0")]
    #[case::mthi(DecodedOp::new(mthi(8)), "mthi $t0")]
    #[case::mtlo(DecodedOp::new(mtlo(8)), "mtlo $t0")]
    #[case::jr(DecodedOp::new(jr(8)), "jr $t0")]
    #[case::jalr(DecodedOp::new(jalr(8, 9)), "jalr $t0 $t1")]
    #[case::mtc(DecodedOp::new(mtc0(8, 16)), "mtc0 $t0, $r16")]
    fn test_display(setup_tracing: (), #[case] op: DecodedOp, #[case] expected: &str) {
        assert_eq!(op.to_string(), expected);
    }
}

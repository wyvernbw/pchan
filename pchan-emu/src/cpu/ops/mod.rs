use crate::{BasicBlock, CacheDependency, EntryCache, cranelift_bs::*, jit::JIT};
use bon::Builder;
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
pub mod mult;
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
pub mod xor;
pub mod xori;

// jumps
pub mod beq;
pub mod j;
pub mod jal;

// loads
pub mod lb;
pub mod lbu;
pub mod lh;
pub mod lhu;
pub mod lw;

// stores
pub mod sb;
pub mod sh;
pub mod subu;
pub mod sw;

pub mod prelude {
    pub use super::OpCode;
    pub use super::addiu::*;
    pub use super::addu::*;
    pub use super::and::*;
    pub use super::andi::*;
    pub use super::beq::*;
    pub use super::j::*;
    pub use super::jal::*;
    pub use super::lb::*;
    pub use super::lbu::*;
    pub use super::lh::*;
    pub use super::lhu::*;
    pub use super::lui::*;
    pub use super::lw::*;
    pub use super::mult::*;
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
        BoundaryType, EmitParams, EmitSummary, MipsOffset, Op, PrimeOp, SecOp, TryFromOpcodeErr,
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

#[derive(Debug, Clone, Copy)]
pub struct CachedValue {
    pub dirty: bool,
    pub value: Value,
}

#[derive(Builder)]
pub struct EmitParams<'a> {
    ptr_type: types::Type,
    cache: &'a mut EntryCache,
    node: NodeIndex,
    pc: u32,
    cfg: &'a Graph<BasicBlock, ()>,
    deps_map: &'a HashMap<Block, CacheDependency>,
}

impl<'a> EmitParams<'a> {
    fn block(&self) -> &BasicBlock {
        &self.cfg[self.node]
    }
    fn next_at(&self, idx: usize) -> &BasicBlock {
        let idx = self
            .cfg
            .neighbors_directed(self.node, Direction::Outgoing)
            .nth(idx)
            .unwrap();
        &self.cfg[idx]
    }
    fn emulator_params(&self, fn_builder: &mut FunctionBuilder) -> Vec<BlockArg> {
        vec![
            BlockArg::Value(self.cpu(fn_builder)),
            BlockArg::Value(self.memory(fn_builder)),
        ]
    }
    #[instrument(skip(fn_builder, self))]
    fn out_params(&self, to: Block, fn_builder: &mut FunctionBuilder) -> Vec<BlockArg> {
        tracing::trace!("{:#?}", self.deps_map);
        let next_block_deps: Option<_> = self.deps_map.get(&to).map(|dep| dep.registers);
        let mut args = self.emulator_params(fn_builder);
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
    fn cpu(&self, fn_builder: &mut FunctionBuilder) -> Value {
        let block = self.block().clif_block();
        fn_builder.block_params(block)[0]
    }
    fn memory(&self, fn_builder: &mut FunctionBuilder) -> Value {
        let block = self.cfg[self.node].clif_block();
        fn_builder.block_params(block)[1]
    }
    fn emit_get_one(&mut self, fn_builder: &mut FunctionBuilder) -> Value {
        match self.cache.const_one {
            Some(one) => one,
            None => {
                let one = fn_builder.ins().iconst(types::I32, 1);
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
    fn emit_get_zero(&mut self, fn_builder: &mut FunctionBuilder) -> Value {
        self.emit_get_register(fn_builder, 0)
    }
    fn emit_get_hi(&mut self, fn_builder: &mut FunctionBuilder) -> Value {
        let block = self.block().clif_block();
        match self.cache.hi {
            Some(hi) => hi.value,
            None => {
                let hi = JIT::emit_load_hi().builder(fn_builder).block(block).call();
                self.cache.hi = Some(CachedValue {
                    dirty: false,
                    value: hi,
                });
                hi
            }
        }
    }
    fn emit_get_lo(&mut self, fn_builder: &mut FunctionBuilder) -> Value {
        let block = self.block().clif_block();
        match self.cache.lo {
            Some(lo) => lo.value,
            None => {
                let lo = JIT::emit_load_lo().builder(fn_builder).block(block).call();
                self.cache.lo = Some(CachedValue {
                    dirty: false,
                    value: lo,
                });
                lo
            }
        }
    }
    fn emit_get_register(&mut self, fn_builder: &mut FunctionBuilder, id: usize) -> Value {
        let block = self.block().clif_block();
        match self.cache.registers[id] {
            Some(value) => value.value,
            None => {
                let value = JIT::emit_load_reg()
                    .builder(fn_builder)
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
    fn update_cache_immediate(&mut self, id: usize, value: Value) {
        self.cache.registers[id] = Some(CachedValue {
            dirty: false,
            value,
        });
    }
}

#[derive(Builder, Debug, Default)]
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
    #[builder(with = |value: Value| CachedValue {
        dirty: true,
        value
    })]
    pub hilo: Option<CachedValue>,
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

#[derive(Debug, Error)]
pub enum TryFromOpcodeErr {
    #[error("invalid header")]
    InvalidHeader,
    #[error("unknown instruction")]
    UnknownInstruction,
}

#[derive(Debug, Clone, Copy)]
pub enum MipsOffset {
    RegionJump(u32),
    Relative(i32),
}

impl MipsOffset {
    pub fn calculate_address(self, base: u32) -> u32 {
        match self {
            MipsOffset::RegionJump(addr) => (base & 0xFF00_0000) + addr,
            MipsOffset::Relative(offset) => base.wrapping_add_signed(offset),
        }
    }
}

#[derive(Debug)]
pub enum BoundaryType {
    Block { offset: MipsOffset },
    BlockSplit { lhs: MipsOffset, rhs: MipsOffset },
    Function,
}

#[enum_dispatch(DecodedOp)]
pub trait Op: Sized + Display + TryFrom<OpCode> {
    fn invalidates_cache_at(&self) -> Option<u32> {
        None
    }
    fn is_block_boundary(&self) -> Option<BoundaryType>;
    fn into_opcode(self) -> crate::cpu::ops::OpCode;
    fn emit_ir(&self, state: EmitParams, fn_builder: &mut FunctionBuilder) -> Option<EmitSummary>;
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

    fn emit_ir(
        &self,
        _state: EmitParams,
        _fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
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
        Some(BoundaryType::Function)
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode(69420)
    }

    fn emit_ir(&self, _state: EmitParams, fn_builder: &mut FunctionBuilder) -> Option<EmitSummary> {
        fn_builder.ins().return_(&[]);
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
    JAL(JAL),
}

impl TryFrom<OpCode> for DecodedOp {
    type Error = impl std::error::Error;

    #[instrument(err)]
    fn try_from(opcode: OpCode) -> Result<Self, Self::Error> {
        if opcode.0 == 69420 {
            return Ok(DecodedOp::HaltBlock(HaltBlock));
        }
        if opcode == OpCode::NOP {
            return Ok(DecodedOp::NOP(NOP));
        }
        match (opcode.primary(), opcode.secondary()) {
            (PrimeOp::JAL, _) => JAL::try_from(opcode).map(Self::JAL),
            (PrimeOp::SPECIAL, SecOp::MULT) => MULT::try_from(opcode).map(Self::MULT),
            (PrimeOp::LUI, _) => LUI::try_from(opcode).map(Self::LUI),
            (PrimeOp::SPECIAL, SecOp::SRA) => SRA::try_from(opcode).map(Self::SRA),
            (PrimeOp::SPECIAL, SecOp::SRL) => SRL::try_from(opcode).map(Self::SRL),
            (PrimeOp::SPECIAL, SecOp::SLL) => SLL::try_from(opcode).map(Self::SLL),
            (PrimeOp::SPECIAL, SecOp::SRAV) => SRAV::try_from(opcode).map(Self::SRAV),
            (PrimeOp::SPECIAL, SecOp::SRLV) => SRLV::try_from(opcode).map(Self::SRLV),
            (PrimeOp::SPECIAL, SecOp::SLLV) => SLLV::try_from(opcode).map(Self::SLLV),
            (PrimeOp::XORI, _) => XORI::try_from(opcode).map(Self::XORI),
            (PrimeOp::ORI, _) => ORI::try_from(opcode).map(Self::ORI),
            (PrimeOp::ANDI, _) => ANDI::try_from(opcode).map(Self::ANDI),
            (PrimeOp::SPECIAL, SecOp::NOR) => NOR::try_from(opcode).map(Self::NOR),
            (PrimeOp::SPECIAL, SecOp::XOR) => XOR::try_from(opcode).map(Self::XOR),
            (PrimeOp::SPECIAL, SecOp::OR) => OR::try_from(opcode).map(Self::OR),
            (PrimeOp::SPECIAL, SecOp::AND) => AND::try_from(opcode).map(Self::AND),
            (PrimeOp::SLTIU, _) => SLTIU::try_from(opcode).map(Self::SLTIU),
            (PrimeOp::SLTI, _) => SLTI::try_from(opcode).map(Self::SLTI),
            (PrimeOp::SPECIAL, SecOp::SLTU) => SLTU::try_from(opcode).map(Self::SLTU),
            (PrimeOp::SPECIAL, SecOp::SLT) => SLT::try_from(opcode).map(Self::SLT),
            (PrimeOp::BEQ, _) => BEQ::try_from(opcode).map(Self::BEQ),
            (PrimeOp::J, _) => J::try_from(opcode).map(Self::J),
            (PrimeOp::ADDIU | PrimeOp::ADDI, _) => ADDIU::try_from(opcode).map(Self::ADDIU),
            (PrimeOp::SPECIAL, SecOp::SUBU | SecOp::SUB) => {
                // TODO: implement SUB separately from SUBU
                SUBU::try_from(opcode).map(Self::SUBU)
            }
            (PrimeOp::SPECIAL, SecOp::ADDU | SecOp::ADD) => {
                // TODO: implement ADD separately from ADDU
                ADDU::try_from(opcode).map(Self::ADDU)
            }
            (PrimeOp::SW, _) => SW::try_from(opcode).map(Self::SW),
            (PrimeOp::SH, _) => SH::try_from(opcode).map(Self::SH),
            (PrimeOp::SB, _) => SB::try_from(opcode).map(Self::SB),
            (PrimeOp::LW, _) => LW::try_from(opcode).map(Self::LW),
            (PrimeOp::LHU, _) => LHU::try_from(opcode).map(Self::LHU),
            (PrimeOp::LH, _) => LH::try_from(opcode).map(Self::LH),
            (PrimeOp::LB, _) => LB::try_from(opcode).map(Self::LB),
            (PrimeOp::LBU, _) => LBU::try_from(opcode).map(Self::LBU),
            _ => Err(TryFromOpcodeErr::UnknownInstruction),
        }
    }
}

impl DecodedOp {
    pub fn new(opcode: OpCode) -> Self {
        Self::try_from(opcode).unwrap()
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
    fn test_display(setup_tracing: (), #[case] op: DecodedOp, #[case] expected: &str) {
        assert_eq!(op.to_string(), expected);
    }
}

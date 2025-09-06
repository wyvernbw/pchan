use crate::{
    cpu::{
        JIT,
        ops::{lb::LB, lbu::LBU, lh::LH},
    },
    cranelift_bs::*,
};
use bon::Builder;
use enum_dispatch::enum_dispatch;
use pchan_macros::OpCode;
use std::ops::Range;
use thiserror::Error;
use tracing::instrument;

pub mod decoded_op;
pub mod lb;
pub mod lbu;
pub mod lh;
#[cfg(test)]
pub mod op_decode_tests;

pub(crate) mod prelude {
    pub(crate) use super::lb::*;
    pub(crate) use super::lbu::*;
    pub(crate) use super::lh::*;
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Opcode(pub(crate) u32);

impl core::fmt::Debug for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Op")
            .field_with(|f| write!(f, "0x{:08X}", &self.0))
            .finish()
    }
}

pub(crate) const fn nop() -> Opcode {
    Opcode::NOP
}

impl Opcode {
    pub(crate) const NOP: Opcode = Opcode(0x00000000);

    #[inline]
    pub(crate) const fn primary(&self) -> PrimeOp {
        let code = self.0 >> 26;
        PrimeOp::MAP[code as usize]
    }
    #[inline]
    pub(crate) const fn secondary(&self) -> SecondaryOp {
        let code = self.0 & 0x3F;
        SecondaryOp::MAP[code as usize]
    }
    #[inline]
    pub(crate) const fn bits(&self, range: Range<u8>) -> u32 {
        let mask = (0xFFFFFFFFu32.unbounded_shl(range.start as u32))
            ^ ((0xFFFFFFFFu32).unbounded_shl(range.end as u32));
        (self.0 & mask).unbounded_shr(range.start as u32)
    }
    #[inline]
    pub(crate) const fn set_bits(&self, range: Range<u8>, value: u32) -> Self {
        let mask = (0xFFFFFFFFu32.unbounded_shl(range.start as u32))
            ^ (0xFFFFFFFFu32.unbounded_shl(range.end as u32));

        let cleared = self.0 & !mask;

        let shifted = (value << range.start) & mask;

        Self(cleared | shifted)
    }

    pub(crate) const fn with_primary(self, primary: PrimeOp) -> Self {
        Opcode((self.0 & 0x03FF_FFFF) | ((primary as u32) << 26))
    }
    pub(crate) const fn with_secondary(self, secondary: SecondaryOp) -> Self {
        Opcode((self.0 & 0xFFFF_FFE0) | (secondary as u32))
    }
    pub(crate) fn as_primary(self, primary: PrimeOp) -> Result<Self, TryFromOpcodeErr> {
        if self.primary() == primary {
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
pub(crate) enum SecondaryOp {
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

#[derive(Builder)]
pub(crate) struct EmitParams<'a, 'b> {
    ptr_type: types::Type,
    fn_builder: &'a mut FunctionBuilder<'b>,
    registers: &'a [Option<Value>; 32],
    block: Block,
    pc: u32,
}

impl<'a, 'b> EmitParams<'a, 'b> {
    fn cpu(&self) -> Value {
        self.fn_builder.block_params(self.block)[0]
    }
    fn memory(&self) -> Value {
        self.fn_builder.block_params(self.block)[1]
    }
    fn emit_get_register(&mut self, id: usize) -> Value {
        match self.registers[id] {
            Some(value) => value,
            None => JIT::emit_load_reg()
                .builder(self.fn_builder)
                .block(self.block)
                .idx(id)
                .call(),
        }
    }
}

#[derive(Debug)]
pub struct EmitSummary {
    pub(crate) register_updates: Box<[(usize, Value)]>,
}

#[derive(Debug, Error)]
pub enum TryFromOpcodeErr {
    #[error("invalid header")]
    InvalidHeader,
}

#[derive(Debug)]
pub enum BoundaryType {
    Block { offset: u32 },
    BlockSplit { offsets: [Option<u32>; 4] },
    Function,
}

#[enum_dispatch(DecodedOp)]
pub(crate) trait Op: Sized {
    fn is_block_boundary(&self) -> Option<BoundaryType>;
    fn into_opcode(self) -> crate::cpu::ops::Opcode;
    fn emit_ir(&self, state: EmitParams<'_, '_>) -> Option<EmitSummary>;
}

impl Op for () {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> crate::cpu::ops::Opcode {
        Opcode::NOP
    }

    fn emit_ir(&self, state: EmitParams<'_, '_>) -> Option<EmitSummary> {
        None
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HaltBlock;

impl Op for HaltBlock {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::Function)
    }

    fn into_opcode(self) -> crate::cpu::ops::Opcode {
        Opcode(69420)
    }

    fn emit_ir(&self, state: EmitParams<'_, '_>) -> Option<EmitSummary> {
        None
    }
}

#[derive(Debug, Clone, Copy)]
#[enum_dispatch]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum DecodedOp {
    NOP(()),
    HaltBlock(HaltBlock),
    LB(LB),
    LBU(LBU),
    LH(LH),
}

impl DecodedOp {
    #[instrument(err)]
    pub(crate) fn try_new(opcode: Opcode) -> Result<Self, impl std::error::Error> {
        if opcode.0 == 69420 {
            return Ok(DecodedOp::HaltBlock(HaltBlock));
        }
        if opcode == Opcode::NOP {
            return Ok(DecodedOp::NOP(()));
        }
        match opcode.primary() {
            PrimeOp::LH => LH::try_from_opcode(opcode).map(Self::LH),
            PrimeOp::LB => LB::try_from_opcode(opcode).map(Self::LB),
            PrimeOp::LBU => LBU::try_from_opcode(opcode).map(Self::LBU),
            _ => Err(TryFromOpcodeErr::InvalidHeader),
        }
    }
}

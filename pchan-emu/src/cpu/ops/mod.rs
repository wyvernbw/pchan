use crate::{
    FnBuilderExt,
    cranelift_bs::*,
    dynarec::{EmitCtx, EmitSummary},
};
use bitfield::bitfield;
use cranelift::codegen::entity::EntityList;
use enum_dispatch::enum_dispatch;
use pchan_macros::OpCode;
use std::{fmt::Display, ops::Range};
use thiserror::Error;

// alu
pub mod addiu;
pub mod addu;
pub mod and;
pub mod andi;
pub mod div;
pub mod divu;
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
pub mod bgez;
pub mod bgtz;
pub mod blez;
pub mod bltz;
pub mod bne;
pub mod brk;
pub mod j;
pub mod jal;
pub mod jalr;
pub mod jr;

// loads
pub mod lb;
pub mod lbu;
pub mod lh;
pub mod lhu;
pub mod load;
pub mod lw;

// stores
pub mod sb;
pub mod sh;
pub mod store;
pub mod sw;

// cop
pub mod lwc;
pub mod mfc;
pub mod mtc;
pub mod rfe;

pub mod prelude {
    pub use super::OpCode;
    pub use super::addiu::*;
    pub use super::addu::*;
    pub use super::and::*;
    pub use super::andi::*;
    pub use super::beq::*;
    pub use super::bgez::*;
    pub use super::bgtz::*;
    pub use super::blez::*;
    pub use super::bltz::*;
    pub use super::bne::*;
    pub use super::brk::*;
    pub use super::div::*;
    pub use super::divu::*;
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
    pub use super::lwc::*;
    pub use super::mfc::*;
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
    pub use super::rfe::*;
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
        BoundaryType, CopOp, DecodedOp, MipsOffset, Op, PrimeOp, SecOp, TryFromOpcodeErr,
    };
    pub use crate::cpu::program;
    pub use crate::dynarec::{EmitCtx, EmitSummary};
}

use prelude::*;

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
    pub const fn check_bits(&self, range: Range<u8>, value: u32) -> Result<Self, TryFromOpcodeErr> {
        if self.bits(range) == value {
            Ok(*self)
        } else {
            Err(TryFromOpcodeErr::UnknownInstruction)
        }
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
        if self.cop() == cop as u8 {
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
    BcondZ = 0x01,
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
    MFCn = 0b00000,
    COP0SPEC = 0b10000,

    #[opcode(default)]
    ILLEGAL,
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
    Function,
}

#[derive(Debug, Clone)]
pub struct Hazard<'a> {
    pub op: &'a DecodedOp,
    pub emit: fn(&'a DecodedOp, EmitCtx) -> EmitSummary,
    pub trigger: u32,
}

#[enum_dispatch(DecodedOp)]
pub trait Op: Sized + Display {
    fn invalidates_cache_at(&self) -> Option<u32> {
        None
    }
    fn cycles(&self) -> u64 {
        1
    }
    fn is_block_boundary(&self) -> Option<BoundaryType>;
    fn into_opcode(self) -> crate::cpu::ops::OpCode;
    fn emit_ir(&self, ctx: EmitCtx) -> EmitSummary;
    fn is_function_boundary(&self) -> bool {
        matches!(self.is_block_boundary(), Some(BoundaryType::Function))
    }
    fn hazard(&self) -> Option<u32> {
        None
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct NOP;

impl Op for NOP {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::NOP
    }

    fn emit_ir(&self, ctx: EmitCtx) -> EmitSummary {
        EmitSummary::builder()
            .instructions([])
            .build(ctx.fn_builder)
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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
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

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        use crate::dynarec::*;

        let ret = ctx
            .fn_builder
            .pure()
            .MultiAry(Opcode::Return, types::INVALID, EntityList::new())
            .0;
        let write_pc = ctx.emit_store_pc_imm(ctx.pc + 4);

        EmitSummary::builder()
            .instructions(
                [
                    write_pc.map(bottom).as_slice(),
                    [terminator(bomb(1, ret))].as_slice(),
                ]
                .concat(),
            )
            .build(ctx.fn_builder)
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

#[derive(Debug, Clone, Copy, strum::Display, PartialEq, Eq, Hash)]
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
    DIV(DIV),
    #[strum(transparent)]
    DIVU(DIVU),
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
    #[strum(transparent)]
    BGEZ(BGEZ),
    #[strum(transparent)]
    BLEZ(BLEZ),
    #[strum(transparent)]
    BLTZ(BLTZ),
    #[strum(transparent)]
    BGTZ(BGTZ),
    #[strum(transparent)]
    BREAK(BREAK),

    // cop
    #[strum(transparent)]
    MTCn(MTCn),
    #[strum(transparent)]
    MFCn(MFCn),
    #[strum(transparent)]
    RFE(RFE),
    #[strum(transparent)]
    LWCn(LWCn),

    ILLEGAL(ILLEGAL),
}

#[derive(Debug, Clone, Copy, derive_more::Display, Hash, PartialEq, Eq)]
pub struct ILLEGAL;

impl Op for ILLEGAL {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
    }

    fn emit_ir(&self, ctx: EmitCtx) -> EmitSummary {
        EmitSummary::builder()
            .instructions([])
            .build(ctx.fn_builder)
    }
}

// #[derive(Debug, Clone, Copy)]
// pub enum OpFields {
//     Nop,
//     Rtype {
//         funct: u8,
//         shamt: u8,
//         rd: u8,
//         rt: u8,
//         rs: u8,
//         opcode: u8,
//     },
//     Itype {
//         imm: i16,
//         rt: u8,
//         rs: u8,
//         opcode: u8,
//     },
//     Jtype {
//         target: u32,
//         opcode: u8,
//     },
//     CoP(CoPOpFields),
//     Malformed,
// }

// #[derive(Debug, Clone, Copy)]
// pub enum CoPOpFields {
//     Reg {
//         cop: u8,
//         rs: u8,
//         rt: u8,
//         rd: u8,
//         funct: u8,
//     },
//     Imm16 {
//         opcode: u8,
//         cop: u8,
//         rs: u8,
//         rt: u8,
//         imm16: i16,
//     },
//     Imm25 {
//         cop: u8,
//         imm25: i32,
//         opcode: u8,
//     },
// }

bitfield! {
    #[derive(Clone, Copy, PartialEq, Eq)]
    #[derive_const(Default)]
    pub struct OpCode(u32);

    u8, funct, _: 5, 0;
    u8, shamt, _: 10, 6;
    u8, rd, _: 15, 11;
    u8, rt, _: 20, 16;
    u8, rs, _: 25, 21;
    u8, opcode, _: 31, 26;
    i16, imm16, _: 15, 0;
    u32, imm26, _: 25, 0;
    u8, cop, _: 27, 26;
}

impl OpCode {
    const NOP_FIELDS: Self = Self::default();
    const HALT_FIELDS: Self = OpCode(69420);
}

impl From<u32> for OpCode {
    fn from(value: u32) -> Self {
        OpCode(value)
    }
}

impl DecodedOp {
    pub const fn illegal() -> Self {
        Self::ILLEGAL(ILLEGAL)
    }
    pub fn new(fields: OpCode) -> Self {
        let [op] = Self::decode([fields]);
        op
    }
    pub fn decode<const N: usize>(fields: [impl Into<OpCode>; N]) -> [Self; N] {
        fields.map(|fields| {
            let fields = fields.into();
            if fields == OpCode::NOP_FIELDS {
                return Self::NOP(NOP);
            }
            if fields == OpCode::HALT_FIELDS {
                return Self::HaltBlock(HaltBlock);
            }

            let opcode = fields.opcode();
            let rs = fields.rs();
            let rt = fields.rt();
            let rd = fields.rd();
            let funct = fields.funct();

            match (opcode, rs, rt, funct) {
                // r type
                (0x0, _, _, 0x0) => Self::SLL(SLL::new(rd, rt, fields.shamt() as i8)),
                (0x0, _, _, 0x1) => Self::illegal(),
                (0x0, _, _, 0x2) => Self::SRL(SRL::new(rd, rt, fields.shamt() as i8)),
                (0x0, _, _, 0x3) => Self::SRA(SRA::new(rd, rt, fields.shamt() as i8)),
                (0x0, _, _, 0x4) => Self::SLLV(SLLV::new(rd, rt, rs)),
                (0x0, _, _, 0x5) => Self::illegal(),
                (0x0, _, _, 0x6) => Self::SRLV(SRLV::new(rd, rt, rs)),
                (0x0, _, _, 0x7) => Self::SRAV(SRAV::new(rd, rt, rs)),
                (0x0, _, _, 0x8) => Self::JR(JR::new(rs)),
                (0x0, _, _, 0x9) => Self::JALR(JALR::new(rd, rs)),
                (0x0, _, _, 0xA) => Self::illegal(),
                (0x0, _, _, 0xB) => Self::illegal(),
                (0x0, _, _, 0xC) => {
                    todo!("syscall");
                    // tracing::error!("syscall not yet implemented");
                    // Self::illegal()
                }
                (0x0, _, _, 0xD) => Self::BREAK(BREAK),
                (0x0, _, _, 0xE) => Self::illegal(),
                (0x0, _, _, 0xF) => Self::illegal(),
                (0x0, _, _, 0x10) => Self::MFHI(MFHI::new(rd)),
                (0x0, _, _, 0x11) => Self::MTHI(MTHI::new(rs)),
                (0x0, _, _, 0x12) => Self::MFLO(MFLO::new(rd)),
                (0x0, _, _, 0x13) => Self::MTLO(MTLO::new(rs)),
                (0x0, _, _, 0x14..=0x17) => Self::illegal(),
                (0x0, _, _, 0x18) => Self::MULT(MULT::new(rs, rt)),
                (0x0, _, _, 0x19) => Self::MULTU(MULTU::new(rs, rt)),
                (0x0, _, _, 0x1A) => Self::DIV(DIV::new(rs, rt)),
                (0x0, _, _, 0x1B) => Self::DIVU(DIVU::new(rs, rt)),
                (0x0, _, _, 0x1C..=0x1F) => Self::illegal(),
                (0x0, _, _, 0x20 | 0x21) => Self::ADDU(ADDU::new(rd, rs, rt)),
                (0x0, _, _, 0x22 | 0x23) => Self::SUBU(SUBU::new(rd, rs, rt)),
                (0x0, _, _, 0x24) => Self::AND(AND::new(rd, rs, rt)),
                (0x0, _, _, 0x25) => Self::OR(OR::new(rd, rs, rt)),
                (0x0, _, _, 0x26) => Self::XOR(XOR::new(rd, rs, rt)),
                (0x0, _, _, 0x27) => Self::NOR(NOR::new(rd, rs, rt)),
                (0x0, _, _, 0x28..=0x29) => Self::illegal(),
                (0x0, _, _, 0x2A) => Self::SLT(SLT::new(rd, rs, rt)),
                (0x0, _, _, 0x2B) => Self::SLTU(SLTU::new(rd, rs, rt)),
                (0x0, _, _, 0x2C..) => Self::illegal(),

                // i type
                (0x1, _, _, _) => {
                    match rt {
                        0x0 => Self::BLTZ(BLTZ::new(rs, fields.imm16())),
                        0x1 => Self::BGEZ(BGEZ::new(rs, fields.imm16())),
                        0x10 => todo!("bltzal"),
                        0x11 => todo!("bgezal"),
                        other if other & 0b110 != 0 => match other & 1 {
                            0 => todo!("bltz dupe"),
                            1 => todo!("bgez dupe"),
                            _ => unreachable!(),
                        },
                        // TODO: add the bltz and bgez dupes, just to be sure
                        _ => Self::illegal(),
                    }
                }
                (0x4, _, _, _) => Self::BEQ(BEQ::new(rs, rt, fields.imm16())),
                (0x5, _, _, _) => Self::BNE(BNE::new(rs, rt, fields.imm16())),
                (0x6, _, _, _) => Self::BLEZ(BLEZ::new(rs, fields.imm16())),
                (0x7, _, _, _) => Self::BGTZ(BGTZ::new(rs, fields.imm16())),
                (0x8 | 0x9, _, _, _) => Self::ADDIU(ADDIU::new(rs, rt, fields.imm16())),
                (0xA, _, _, _) => Self::SLTI(SLTI::new(rt, rs, fields.imm16())),
                (0xB, _, _, _) => Self::SLTIU(SLTIU::new(rt, rs, fields.imm16())),
                (0xC, _, _, _) => Self::ANDI(ANDI::new(rs, rt, fields.imm16() as u16)),
                (0xD, _, _, _) => Self::ORI(ORI::new(rs, rt, fields.imm16() as u16)),
                (0xE, _, _, _) => Self::XORI(XORI::new(rs, rt, fields.imm16() as u16)),
                (0xF, _, _, _) => Self::LUI(LUI::new(rt, fields.imm16())),
                (0x14..=0x1F, _, _, _) => Self::illegal(),
                (0x20, _, _, _) => Self::LB(LB::new(rt, rs, fields.imm16())),
                (0x21, _, _, _) => Self::LH(LH::new(rt, rs, fields.imm16())),
                (0x22, _, _, _) => todo!("lwl"),
                (0x23, _, _, _) => Self::LW(LW::new(rt, rs, fields.imm16())),
                (0x24, _, _, _) => Self::LBU(LBU::new(rt, rs, fields.imm16())),
                (0x25, _, _, _) => Self::LHU(LHU::new(rt, rs, fields.imm16())),
                (0x26, _, _, _) => todo!("lwr"),
                (0x27, _, _, _) => Self::illegal(),
                (0x28, _, _, _) => Self::SB(SB::new(rt, rs, fields.imm16())),
                (0x29, _, _, _) => Self::SH(SH::new(rt, rs, fields.imm16())),
                (0x2A, _, _, _) => todo!("swl"),
                (0x2B, _, _, _) => Self::SW(SW::new(rt, rs, fields.imm16())),
                (0x2C..=0x2D, _, _, _) => Self::illegal(),
                (0x2E, _, _, _) => todo!("swr"),
                (0x2F, _, _, _) => Self::illegal(),
                (0x34..=0x37, _, _, _) => Self::illegal(),
                (0x3C..=0x3F, _, _, _) => Self::illegal(),

                // j type
                (0x2, _, _, _) => Self::J(J::new(fields.imm26())),
                (0x3, _, _, _) => Self::JAL(JAL::new(fields.imm26())),

                // cop reg
                (0x10..=0x13, 0x0, _, 0x0) => Self::MFCn(MFCn::new(fields.cop(), rt, rd)),
                (_, 0x2, _, 0x0) => todo!("cfcn"),
                (_, 0x4, _, 0x0) => Self::MTCn(MTCn::new(fields.cop(), rt, rd)),
                (_, 0x6, _, 0x0) => todo!("ctcn"),
                (_, 0x10, _, 0x10) => Self::RFE(RFE),

                // cop imm16
                (0x10..=0x13, 0x8, 0, _) => todo!("bcnf"),
                (0x10..=0x13, 0x8, 1, _) => todo!("bcnt"),
                (0x30..=0x33, _, _, _) => {
                    Self::LWCn(LWCn::new(fields.cop(), rt, rs, fields.imm16()))
                }
                (0x38..=0x3B, _, _, _) => todo!("swcn"),

                // cop imm25
                (0x10..=0x13, 0x10..=0x1F, _, _) => {
                    // TODO: copn command
                    todo!("cop{} imm25", fields.cop());
                    // tracing::error!("cop command not yet implemented");
                    // Self::illegal()
                }

                _ => Self::illegal(),
            }
        })
    }
}

impl DecodedOp {
    pub fn is_nop(&self) -> bool {
        matches!(self, Self::NOP(_))
    }

    pub fn is_illegal(&self) -> bool {
        matches!(self, Self::ILLEGAL(_))
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

#[cfg(test)]
mod decode_tests;

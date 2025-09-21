use crate::{
    FnBuilderExt,
    cranelift_bs::*,
    dynarec::{EmitCtx, EmitSummary},
};
use cranelift::codegen::entity::EntityList;
use enum_dispatch::enum_dispatch;
use pchan_macros::OpCode;
use pchan_utils::array;
use std::{
    fmt::Display,
    ops::Range,
    simd::{LaneCount, Simd, SupportedLaneCount},
};
use thiserror::Error;
use tracing::instrument;

// alu
pub mod addiu;
pub mod addu;
pub mod and;
pub mod andi;
pub mod div;
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
pub mod blez;
pub mod bltz;
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
pub mod load;
pub mod lw;

// stores
pub mod sb;
pub mod sh;
pub mod store;
pub mod sw;

// cop
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
    pub use super::blez::*;
    pub use super::bltz::*;
    pub use super::bne::*;
    pub use super::div::*;
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

#[derive(Debug, Clone, Copy)]
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
    DIV(DIV),
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

    // cop
    #[strum(transparent)]
    MTCn(MTCn),
    #[strum(transparent)]
    MFCn(MFCn),
    #[strum(transparent)]
    RFE(RFE),

    ILLEGAL(ILLEGAL),
}

#[derive(Debug, Clone, Copy, derive_more::Display)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive_const(Default)]
pub struct OpFields {
    opcode: u8,
    rs: u8,
    rt: u8,
    rd: u8,
    shamt: u8,
    funct: u8,
    imm16: i16,
    imm26: u32,
    imm25: i32,
    cop: u8,
}

impl OpFields {
    const NOP_FIELDS: Self = Self::default();
}

impl DecodedOp {
    pub const fn illegal() -> Self {
        Self::ILLEGAL(ILLEGAL)
    }
    pub fn extract_fields(op: &OpCode) -> OpFields {
        // let ops = Simd::<u32, N>::from_slice(ops);
        let op = op.0;
        if op == 0 {
            return OpFields::NOP_FIELDS;
        }
        if op == 69420 {
            return OpFields {
                opcode: 255,
                ..OpFields::NOP_FIELDS
            };
        }
        let opcode_shifted = op >> 26;
        let opcode = opcode_shifted;
        let rs = (op >> 21) & 0x1f;
        let rt = (op >> 16) & 0x1f;
        let rd = (op >> 11) & 0x1f;
        let shamt = (op >> 6) & 0x1f;
        let funct = op & 0x3f;
        let imm16 = op & 0xFFFF;
        let imm26 = op & 0x3FFFFFF;
        let imm25 = op & 0x1FFFFFF;
        let cop = opcode_shifted & 0x3;

        OpFields {
            opcode: opcode as u8,
            rs: rs as u8,
            rt: rt as u8,
            rd: rd as u8,
            shamt: shamt as u8,
            funct: funct as u8,
            imm16: imm16 as i16,
            imm26,
            imm25: imm25 as i32,
            cop: cop as u8,
        }
    }
    pub fn extract_fields_simd<const N: usize>(ops: &Simd<u32, N>) -> [OpFields; N]
    where
        LaneCount<N>: SupportedLaneCount,
    {
        let simd_0x1f = Simd::splat(0x1F);

        // let ops = Simd::<u32, N>::from_slice(ops);
        if ops == &Simd::<u32, N>::splat(0x0) {
            return [OpFields::NOP_FIELDS; N];
        }
        let opcode_shifted = ops >> 26;
        let opcode = opcode_shifted & Simd::splat(0x3F);
        let rs = (ops >> 21) & simd_0x1f;
        let rt = (ops >> 16) & simd_0x1f;
        let rd = (ops >> 11) & simd_0x1f;
        let shamt = (ops >> 6) & simd_0x1f;
        let funct = ops & Simd::splat(0x3F);
        let imm16 = ops & Simd::splat(0xFFFF);
        let imm26 = ops & Simd::splat(0x3FFFFFF);
        let imm25 = ops & Simd::splat(0x1FFFFFF);
        let cop = opcode_shifted & Simd::splat(0x3);
        let mut i = 0;

        opcode.to_array().map(|op| {
            let fields = OpFields {
                opcode: op as u8,
                rs: rs[i] as u8,
                rt: rt[i] as u8,
                rd: rd[i] as u8,
                shamt: shamt[i] as u8,
                funct: funct[i] as u8,
                imm16: imm16[i] as i16,
                imm26: imm26[i],
                imm25: imm25[i] as i32,
                cop: cop[i] as u8,
            };
            i += 1;
            fields
        })
    }
    #[inline(never)]
    pub fn decode_one(fields: OpFields) -> Self {
        let [op] = Self::decode([fields]);
        op
    }
    pub fn decode<const N: usize>(fields: [OpFields; N]) -> [Self; N] {
        fields.map(|fields| {
            if fields == OpFields::NOP_FIELDS {
                return Self::NOP(NOP);
            }
            if fields.opcode == 255 {
                return Self::HaltBlock(HaltBlock);
            }
            let OpFields {
                opcode,
                rs,
                rt,
                rd,
                shamt,
                funct,
                imm16,
                imm26,
                imm25,
                cop,
            } = fields;

            match (opcode, rs, rt, funct) {
                // r type
                (0x0, _, _, 0x0) => Self::SLL(SLL::new(rd, rt, shamt as i8)),
                (0x0, _, _, 0x1) => Self::illegal(),
                (0x0, _, _, 0x2) => Self::SRL(SRL::new(rd, rt, shamt as i8)),
                (0x0, _, _, 0x3) => Self::SRA(SRA::new(rd, rt, shamt as i8)),
                (0x0, _, _, 0x4) => Self::SLLV(SLLV::new(rd, rt, rs)),
                (0x0, _, _, 0x5) => Self::illegal(),
                (0x0, _, _, 0x6) => Self::SRLV(SRLV::new(rd, rt, rs)),
                (0x0, _, _, 0x7) => Self::SRAV(SRAV::new(rd, rt, rs)),
                (0x0, _, _, 0x8) => Self::JR(JR::new(rs)),
                (0x0, _, _, 0x9) => Self::JALR(JALR::new(rd, rs)),
                (0x0, _, _, 0xA) => Self::illegal(),
                (0x0, _, _, 0xB) => Self::illegal(),
                (0x0, _, _, 0xC) => {
                    // todo!("syscall");
                    tracing::error!("syscall not yet implemented");
                    Self::illegal()
                }
                (0x0, _, _, 0xD) => todo!("break"),
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
                (0x0, _, _, 0x1B) => todo!("divu"),
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
                (0x1, _, _, _) => match rt {
                    0x0 => Self::BLTZ(BLTZ::new(rs, imm16)),
                    0x1 => Self::BGEZ(BGEZ::new(rs, imm16)),
                    0x10 => todo!("bltzal"),
                    0x11 => todo!("bgezal"),
                    // TODO: add the bltz and bgez dupes, just to be sure
                    _ => Self::illegal(),
                },
                (0x4, _, _, _) => Self::BEQ(BEQ::new(rs, rt, imm16)),
                (0x5, _, _, _) => Self::BNE(BNE::new(rs, rt, imm16)),
                (0x6, _, _, _) => Self::BLEZ(BLEZ::new(rs, imm16)),
                (0x7, _, _, _) => {
                    // todo!("bgtz");
                    Self::illegal()
                }
                (0x8 | 0x9, _, _, _) => Self::ADDIU(ADDIU::new(rs, rt, imm16)),
                (0xA, _, _, _) => Self::SLTI(SLTI::new(rt, rs, imm16)),
                (0xB, _, _, _) => Self::SLTIU(SLTIU::new(rt, rs, imm16)),
                (0xC, _, _, _) => Self::ANDI(ANDI::new(rs, rt, imm16 as u16)),
                (0xD, _, _, _) => Self::ORI(ORI::new(rs, rt, imm16 as u16)),
                (0xE, _, _, _) => Self::XORI(XORI::new(rs, rt, imm16 as u16)),
                (0xF, _, _, _) => Self::LUI(LUI::new(rt, imm16)),
                (0x14..=0x1F, _, _, _) => Self::illegal(),
                (0x20, _, _, _) => Self::LB(LB::new(rt, rs, imm16)),
                (0x21, _, _, _) => Self::LH(LH::new(rt, rs, imm16)),
                (0x22, _, _, _) => todo!("lwl"),
                (0x23, _, _, _) => Self::LW(LW::new(rt, rs, imm16)),
                (0x24, _, _, _) => Self::LBU(LBU::new(rt, rs, imm16)),
                (0x25, _, _, _) => Self::LHU(LHU::new(rt, rs, imm16)),
                (0x26, _, _, _) => todo!("lwr"),
                (0x27, _, _, _) => Self::illegal(),
                (0x28, _, _, _) => Self::SB(SB::new(rt, rs, imm16)),
                (0x29, _, _, _) => Self::SH(SH::new(rt, rs, imm16)),
                (0x2A, _, _, _) => todo!("swl"),
                (0x2B, _, _, _) => Self::SW(SW::new(rt, rs, imm16)),
                (0x2C..=0x2D, _, _, _) => Self::illegal(),
                (0x2E, _, _, _) => todo!("swr"),
                (0x2F, _, _, _) => Self::illegal(),
                (0x34..=0x37, _, _, _) => Self::illegal(),
                (0x3C.., _, _, _) => Self::illegal(),

                // j type
                (0x2, _, _, _) => Self::J(J::new(imm26)),
                (0x3, _, _, _) => Self::JAL(JAL::new(imm26)),

                // cop reg
                (_, 0x0, _, 0x0) => Self::MFCn(MFCn::new(cop, rt, rd)),
                (_, 0x2, _, 0x0) => todo!("cfcn"),
                (_, 0x4, _, 0x0) => Self::MTCn(MTCn::new(cop, rt, rd)),
                (_, 0x6, _, 0x0) => todo!("ctcn"),
                (_, 0x10, _, 0x10) => Self::RFE(RFE),

                // cop imm16
                (0x10..=0x13, 0x8, 0, _) => todo!("bcnf"),
                (0x10..=0x13, 0x8, 1, _) => todo!("bcnt"),
                (0x30..=0x33, _, _, _) => todo!("lwcn"),
                (0x38..=0x3B, _, _, _) => todo!("swcn"),

                // cop imm25
                (0x10..=0x13, 0x10.., _, _) => {
                    // TODO: copn command
                    // todo!("cop{} imm25", cop);
                    tracing::error!("cop command not yet implemented");
                    Self::illegal()
                }
                _ => Self::illegal(),
            }
        })
    }
    pub fn lut_decode<const N: usize>(fields: [OpFields; N]) -> [Self; N] {
        fields.map(|fields| {
            if fields == OpFields::NOP_FIELDS {
                return Self::NOP(NOP);
            }
            let OpFields {
                opcode,
                rs,
                rt,
                rd,
                shamt,
                funct,
                imm16,
                imm26,
                imm25,
                cop,
            } = fields;

            type RTypeBuilder = fn(u8, u8, u8, u8) -> DecodedOp;
            static R_TYPE_ILLEGAL: RTypeBuilder = |_, _, _, _| DecodedOp::illegal();

            static R_TYPE_TABLE: [RTypeBuilder; 49] = array![
                0x0 => |rd, rt, rs, shamt| { DecodedOp::SLL(SLL::new(rd, rt, shamt as i8))},
                0x1 => R_TYPE_ILLEGAL,
                0x2 => |rd, rt, rs, shamt| { DecodedOp::SRL(SRL::new(rd, rt, shamt as i8))},
                0x3 => |rd, rt, rs, shamt| { DecodedOp::SRA(SRA::new(rd, rt, shamt as i8))},
                0x4 => |rd, rt, rs, shamt| { DecodedOp::SLLV(SLLV::new(rd, rt, rs))},
                0x5 => R_TYPE_ILLEGAL,
                0x6 => |rd, rt, rs, shamt| { DecodedOp::SRLV(SRLV::new(rd, rt, rs))},
                0x7 => |rd, rt, rs, shamt| { DecodedOp::SRAV(SRAV::new(rd, rt, rs))},
                0x8 => |rd, rt, rs, shamt| { DecodedOp::JR(JR::new(rs))},
                0x9 => |rd, rt, rs, shamt| { DecodedOp::JALR(JALR::new(rd, rs))},
                0xA => R_TYPE_ILLEGAL,
                0xB => R_TYPE_ILLEGAL,
                0xC => |rd, rt, rs, shamt| {
                    // TODO: syscall
                    DecodedOp::illegal()
                },
                0xD => |rd, rt, rs, shamt| { todo!("break")},
                0xE => R_TYPE_ILLEGAL,
                0xF => R_TYPE_ILLEGAL,
                0x10 => |rd, rt, rs, shamt| { DecodedOp::MFHI(MFHI::new(rd))},
                0x11 => |rd, rt, rs, shamt| { DecodedOp::MTHI(MTHI::new(rs))},
                0x12 => |rd, rt, rs, shamt| { DecodedOp::MFLO(MFLO::new(rd))},
                0x13 => |rd, rt, rs, shamt| { DecodedOp::MTLO(MTLO::new(rs))},
                0x14 => R_TYPE_ILLEGAL,
                0x15 => R_TYPE_ILLEGAL,
                0x16 => R_TYPE_ILLEGAL,
                0x17 => R_TYPE_ILLEGAL,
                0x18 => |rd, rt, rs, shamt| { DecodedOp::MULT(MULT::new(rs, rt))},
                0x19 => |rd, rt, rs, shamt| { DecodedOp::MULTU(MULTU::new(rs, rt))},
                0x1A => |rd, rt, rs, shamt| { todo!("div")},
                0x1B => |rd, rt, rs, shamt| { todo!("div")},
                0x1C => R_TYPE_ILLEGAL,
                0x1D => R_TYPE_ILLEGAL,
                0x1E => R_TYPE_ILLEGAL,
                0x1F => R_TYPE_ILLEGAL,
                0x20 => |rd, rt, rs, shamt| { DecodedOp::ADDU(ADDU::new(rd, rs, rt))},
                0x21 => |rd, rt, rs, shamt| { DecodedOp::ADDU(ADDU::new(rd, rs, rt))},
                0x22 => |rd, rt, rs, shamt| { DecodedOp::SUBU(SUBU::new(rd, rs, rt))},
                0x23 => |rd, rt, rs, shamt| { DecodedOp::SUBU(SUBU::new(rd, rs, rt))},
                0x24 => |rd, rt, rs, shamt| { DecodedOp::AND(AND::new(rd, rs, rt))},
                0x25 => |rd, rt, rs, shamt| { DecodedOp::OR(OR::new(rd, rs, rt))},
                0x26 => |rd, rt, rs, shamt| { DecodedOp::XOR(XOR::new(rd, rs, rt))},
                0x27 => |rd, rt, rs, shamt| { DecodedOp::NOR(NOR::new(rd, rs, rt))},
                0x24 => |rd, rt, rs, shamt| { DecodedOp::AND(AND::new(rd, rs, rt))},
                0x25 => |rd, rt, rs, shamt| { DecodedOp::OR(OR::new(rd, rs, rt))},
                0x26 => |rd, rt, rs, shamt| { DecodedOp::XOR(XOR::new(rd, rs, rt))},
                0x27 => |rd, rt, rs, shamt| { DecodedOp::NOR(NOR::new(rd, rs, rt))},
                0x28 => R_TYPE_ILLEGAL,
                0x29 => R_TYPE_ILLEGAL,
                0x2A => |rd, rt, rs, shamt| { DecodedOp::SLT(SLT::new(rd, rs, rt))},
                0x2B => |rd, rt, rs, shamt| { DecodedOp::SLTU(SLTU::new(rd, rs, rt))},
                0x2C => R_TYPE_ILLEGAL
            ];

            match (opcode, rs, rt, funct) {
                // r type
                (0x0, _, _, _) => R_TYPE_TABLE[funct as usize](rd, rt, rs, shamt),

                // i type
                (0x1, _, _, _) => match rt {
                    0x0 => todo!("bltz"),
                    0x1 => Self::BGEZ(BGEZ::new(rs, imm16)),
                    0x10 => todo!("bltzal"),
                    0x11 => todo!("bgezal"),
                    // TODO: add the bltz and bgez dupes, just to be sure
                    _ => Self::illegal(),
                },
                (0x4, _, _, _) => Self::BEQ(BEQ::new(rs, rt, imm16)),
                (0x5, _, _, _) => Self::BNE(BNE::new(rs, rt, imm16)),
                (0x6, _, _, _) => Self::BLEZ(BLEZ::new(rs, imm16)),
                (0x7, _, _, _) => {
                    // todo!("bgtz");
                    Self::illegal()
                }
                (0x8 | 0x9, _, _, _) => Self::ADDIU(ADDIU::new(rs, rt, imm16)),
                (0xA, _, _, _) => Self::SLTI(SLTI::new(rt, rs, imm16)),
                (0xB, _, _, _) => Self::SLTIU(SLTIU::new(rt, rs, imm16)),
                (0xC, _, _, _) => Self::ANDI(ANDI::new(rs, rt, imm16 as u16)),
                (0xD, _, _, _) => Self::ORI(ORI::new(rs, rt, imm16 as u16)),
                (0xE, _, _, _) => Self::XORI(XORI::new(rs, rt, imm16 as u16)),
                (0xF, _, _, _) => Self::LUI(LUI::new(rt, imm16)),
                (0x14..=0x1F, _, _, _) => Self::illegal(),
                (0x20, _, _, _) => Self::LB(LB::new(rt, rs, imm16)),
                (0x21, _, _, _) => Self::LH(LH::new(rt, rs, imm16)),
                (0x22, _, _, _) => todo!("lwl"),
                (0x23, _, _, _) => Self::LW(LW::new(rt, rs, imm16)),
                (0x24, _, _, _) => Self::LBU(LBU::new(rt, rs, imm16)),
                (0x25, _, _, _) => Self::LHU(LHU::new(rt, rs, imm16)),
                (0x26, _, _, _) => todo!("lwr"),
                (0x27, _, _, _) => Self::illegal(),
                (0x28, _, _, _) => Self::SB(SB::new(rt, rs, imm16)),
                (0x29, _, _, _) => Self::SH(SH::new(rt, rs, imm16)),
                (0x2A, _, _, _) => todo!("swl"),
                (0x2B, _, _, _) => Self::SW(SW::new(rt, rs, imm16)),
                (0x2C..=0x2D, _, _, _) => Self::illegal(),
                (0x2E, _, _, _) => todo!("swr"),
                (0x2F, _, _, _) => Self::illegal(),
                (0x34..=0x37, _, _, _) => Self::illegal(),
                (0x3C.., _, _, _) => Self::illegal(),

                // j type
                (0x2, _, _, _) => Self::J(J::new(imm26 as u32)),
                (0x3, _, _, _) => Self::JAL(JAL::new(imm26 as u32)),

                // cop reg
                (_, 0x0, _, 0x0) => Self::MFCn(MFCn::new(cop, rt, rd)),
                (_, 0x2, _, 0x0) => todo!("cfcn"),
                (_, 0x4, _, 0x0) => Self::MTCn(MTCn::new(cop, rt, rd)),
                (_, 0x6, _, 0x0) => todo!("ctcn"),
                (_, 0x10, _, 0x10) => Self::RFE(RFE),

                // cop imm16
                (0x10..=0x13, 0x8, 0, _) => todo!("bcnf"),
                (0x10..=0x13, 0x8, 1, _) => todo!("bcnt"),
                (0x30..=0x33, _, _, _) => todo!("lwcn"),
                (0x38..=0x3B, _, _, _) => todo!("swcn"),

                // cop imm25
                (0x10..=0x13, 0x10.., _, _) => todo!("cop imm25"),
                _ => Self::illegal(),
            }
        })
    }

    #[deprecated]
    fn decode_old(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        if opcode.0 == 69420 {
            return Ok(DecodedOp::HaltBlock(HaltBlock));
        }
        if opcode == OpCode::NOP {
            return Ok(DecodedOp::NOP(NOP));
        }

        // FIXME: streamline field extraction
        // maybe make it simd compatible?

        match (opcode.primary(), opcode.secondary()) {
            (PrimeOp::BcondZ, _) => match opcode.bits(16..21) {
                0b00001 => BGEZ::try_from(opcode).map(Self::BGEZ),
                _ => Err(TryFromOpcodeErr::UnknownInstruction),
            },
            (PrimeOp::COP0 | PrimeOp::COP1 | PrimeOp::COP2 | PrimeOp::COP3, _) => {
                match (opcode.cop(), opcode.bits(0..6)) {
                    (CopOp::MFCn, 0b000000) => MFCn::try_from(opcode).map(Self::MFCn),
                    (CopOp::MTCn, 0b000000) => MTCn::try_from(opcode).map(Self::MTCn),
                    (CopOp::COP0SPEC, 0b010000) => {
                        tracing::info!("matched rfe");
                        RFE::try_from(opcode).map(Self::RFE)
                    }
                    _ => Err(TryFromOpcodeErr::UnknownInstruction),
                }
            }
            (PrimeOp::SPECIAL, SecOp::JALR) => JALR::try_from(opcode).map(Self::JALR),
            (PrimeOp::SPECIAL, SecOp::JR) => JR::try_from(opcode).map(Self::JR),
            (PrimeOp::JAL, _) => JAL::try_from(opcode).map(Self::JAL),
            (PrimeOp::SPECIAL, SecOp::MTLO) => MTLO::try_from(opcode).map(Self::MTLO),
            (PrimeOp::SPECIAL, SecOp::MTHI) => MTHI::try_from(opcode).map(Self::MTHI),
            (PrimeOp::SPECIAL, SecOp::MFHI) => MFHI::try_from(opcode).map(Self::MFHI),
            (PrimeOp::SPECIAL, SecOp::MFLO) => MFLO::try_from(opcode).map(Self::MFLO),
            (PrimeOp::SPECIAL, SecOp::MULTU) => MULTU::try_from(opcode).map(Self::MULTU),
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
            (PrimeOp::BNE, _) => BNE::try_from(opcode).map(Self::BNE),
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

        // FIXME: streamline field extraction
        // maybe make it simd compatible?

        match (opcode.primary(), opcode.secondary()) {
            (PrimeOp::BcondZ, _) => match opcode.bits(16..21) {
                0b00001 => BGEZ::try_from(opcode).map(Self::BGEZ),
                _ => Err(TryFromOpcodeErr::UnknownInstruction),
            },
            (PrimeOp::COP0 | PrimeOp::COP1 | PrimeOp::COP2 | PrimeOp::COP3, _) => {
                match (opcode.cop(), opcode.bits(0..6)) {
                    (CopOp::MFCn, 0b000000) => MFCn::try_from(opcode).map(Self::MFCn),
                    (CopOp::MTCn, 0b000000) => MTCn::try_from(opcode).map(Self::MTCn),
                    (CopOp::COP0SPEC, 0b010000) => {
                        tracing::info!("matched rfe");
                        RFE::try_from(opcode).map(Self::RFE)
                    }
                    _ => Err(TryFromOpcodeErr::UnknownInstruction),
                }
            }
            (PrimeOp::SPECIAL, SecOp::JALR) => JALR::try_from(opcode).map(Self::JALR),
            (PrimeOp::SPECIAL, SecOp::JR) => JR::try_from(opcode).map(Self::JR),
            (PrimeOp::JAL, _) => JAL::try_from(opcode).map(Self::JAL),
            (PrimeOp::SPECIAL, SecOp::MTLO) => MTLO::try_from(opcode).map(Self::MTLO),
            (PrimeOp::SPECIAL, SecOp::MTHI) => MTHI::try_from(opcode).map(Self::MTHI),
            (PrimeOp::SPECIAL, SecOp::MFHI) => MFHI::try_from(opcode).map(Self::MFHI),
            (PrimeOp::SPECIAL, SecOp::MFLO) => MFLO::try_from(opcode).map(Self::MFLO),
            (PrimeOp::SPECIAL, SecOp::MULTU) => MULTU::try_from(opcode).map(Self::MULTU),
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
            (PrimeOp::BNE, _) => BNE::try_from(opcode).map(Self::BNE),
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

    pub fn is_nop(&self) -> bool {
        matches!(self, Self::NOP(_))
    }
}

#[derive(Debug, Clone, Copy, derive_more::Display)]
pub struct OpForceBoundary(pub DecodedOp);

#[derive(Debug, Error)]
pub enum TryFromOpcodeErr {
    #[error("invalid header")]
    InvalidHeader,
    #[error("unknown instruction")]
    UnknownInstruction,
    #[error("found opcode for invalid coprocessor {0:?}")]
    InvalidCoprocessor(PrimeOp),
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
    #[case::j(DecodedOp::new(j(0x0000_2000)), "j 0x00002000")]
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
    #[case::mfc(DecodedOp::new(mfc0(8, 16)), "mfc0 $t0, $r16")]
    #[case::rfe(DecodedOp::new(rfe()), "rfe")]
    #[case::bgez(DecodedOp::new(bgez(8, 0x20)), "bgez $t0 0x0020")]
    fn test_display(setup_tracing: (), #[case] op: DecodedOp, #[case] expected: &str) {
        assert_eq!(op.to_string(), expected);
    }
}

use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub struct SRL {
    pub rd: u8,
    pub rt: u8,
    pub imm: i8,
}

impl SRL {
    pub fn new(rd: u8, rt: u8, imm: i8) -> Self {
        Self { rd, rt, imm }
    }
}

impl TryFrom<OpCode> for SRL {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::SRL)?;
        Ok(SRL {
            rt: opcode.bits(16..21) as u8,
            rd: opcode.bits(11..16) as u8,
            imm: opcode.bits(6..11) as i8,
        })
    }
}

impl Display for SRL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "srl ${} ${} {}",
            REG_STR[self.rd as usize],
            REG_STR[self.rt as usize],
            hex(self.imm)
        )
    }
}

impl Op for SRL {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::SRL)
            .set_bits(11..16, self.rd as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(6..11, (self.imm as i32 as i16) as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        shiftimm!(self, ctx, Opcode::UshrImm)
    }
}

#[inline]
pub fn srl(rd: u8, rt: u8, imm: i8) -> OpCode {
    SRL { rd, rt, imm }.into_opcode()
}

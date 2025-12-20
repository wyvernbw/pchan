use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct MULTU {
    rs: u8,
    rt: u8,
}

impl MULTU {
    pub const fn new(rs: u8, rt: u8) -> Self {
        Self { rs, rt }
    }
}

impl Op for MULTU {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::MULTU)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        mult!(self, ctx, Opcode::Umulhi)
    }
}

impl Display for MULTU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "multu ${} ${}",
            REG_STR[self.rs as usize], REG_STR[self.rt as usize]
        )
    }
}

impl TryFrom<OpCode> for MULTU {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::MULTU)?;
        Ok(MULTU {
            rs: value.bits(21..26) as u8,
            rt: value.bits(16..21) as u8,
        })
    }
}

pub fn multu(rs: u8, rt: u8) -> OpCode {
    MULTU { rs, rt }.into_opcode()
}

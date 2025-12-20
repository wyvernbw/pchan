use std::fmt::Display;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct MFLO {
    rd: u8,
}

impl MFLO {
    pub fn new(rd: u8) -> Self {
        Self { rd }
    }
}

impl Op for MFLO {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::MFLO)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        let (lo, loadlo) = state.emit_get_lo();
        EmitSummary::builder()
            .instructions([now(loadlo)])
            .register_updates([(self.rd, lo)])
            .build(state.fn_builder)
    }
}

impl TryFrom<OpCode> for MFLO {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::MFLO)?;
        Ok(MFLO {
            rd: value.bits(11..16) as u8,
        })
    }
}

impl Display for MFLO {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mflo ${}", REG_STR[self.rd as usize])
    }
}

pub fn mflo(rd: u8) -> OpCode {
    MFLO { rd }.into_opcode()
}

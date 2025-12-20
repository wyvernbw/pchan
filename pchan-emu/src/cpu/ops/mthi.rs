use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct MTHI {
    rs: u8,
}

impl MTHI {
    pub fn new(rs: u8) -> Self {
        Self { rs }
    }
}

impl Op for MTHI {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::MTHI)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        let (rs, loadreg) = state.emit_get_register(self.rs);
        EmitSummary::builder()
            .hi(rs)
            .instructions([now(loadreg)])
            .build(state.fn_builder)
    }
}

impl TryFrom<OpCode> for MTHI {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::MTHI)?;
        Ok(MTHI {
            rs: value.bits(21..26) as u8,
        })
    }
}

impl Display for MTHI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mthi ${}", REG_STR[self.rs as usize])
    }
}

pub fn mthi(rs: u8) -> OpCode {
    MTHI { rs }.into_opcode()
}

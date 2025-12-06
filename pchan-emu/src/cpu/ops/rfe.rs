use crate::dynarec::prelude::*;

use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct RFE;

impl Display for RFE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rfe")
    }
}

impl TryFrom<OpCode> for RFE {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value.as_primary(PrimeOp::COP0)?;
        let _rfe = value.check_bits(0..6, 0b010000)?.as_cop(CopOp::COP0SPEC)?;
        Ok(RFE)
    }
}

impl Op for RFE {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        // Some(BoundaryType::Function)
        None
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
            .with_primary(PrimeOp::COP0)
            .set_bits(0..6, 0b010000)
            .with_cop(CopOp::COP0SPEC)
    }

    fn emit_ir(&self, ctx: EmitCtx) -> EmitSummary {
        tracing::info!("emitting rfe!");
        let cpu = ctx.cpu();
        let handle_rfe = ctx
            .fn_builder
            .pure()
            .call(ctx.func_ref_table.handle_rfe, &[cpu]);
        EmitSummary::builder()
            .instructions([now(handle_rfe)])
            .build(ctx.fn_builder)
    }
}

pub fn rfe() -> OpCode {
    RFE.into_opcode()
}

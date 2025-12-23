use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SLTIU {
    pub rt: u8,
    pub rs: u8,
    pub imm: u16,
}

impl SLTIU {
    pub const fn new(rt: u8, rs: u8, imm: u16) -> Self {
        Self { rt, rs, imm }
    }
}

pub fn sltiu(rt: u8, rs: u8, imm: u16) -> OpCode {
    SLTIU { rs, rt, imm }.into_opcode()
}

impl Display for SLTIU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sltiu ${} ${} {}",
            REG_STR[self.rt as usize],
            REG_STR[self.rs as usize],
            hex(self.imm)
        )
    }
}

impl TryFrom<OpCode> for SLTIU {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value.as_primary(PrimeOp::SLTIU)?;
        Ok(SLTIU {
            imm: value.bits(0..16) as u16,
            rt: value.bits(16..21) as u8,
            rs: value.bits(21..26) as u8,
        })
    }
}

impl Op for SLTIU {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SLTIU)
            .set_bits(0..16, self.imm as i32 as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        // x < 0u64 (u64::MIN) = false
        if self.imm == 0 {
            let (zero, loadzero) = ctx.emit_get_zero();
            return EmitSummary::builder()
                .instructions([now(loadzero)])
                .register_updates([(self.rt, zero)])
                .build(ctx.fn_builder);
        }

        // 0u64 < x = true
        if self.rs == 0 {
            let (one, loadone) = ctx.emit_get_one();
            return EmitSummary::builder()
                .instructions([now(loadone)])
                .register_updates([(self.rt, one)])
                .build(ctx.fn_builder);
        }

        icmpimm!(self, ctx, IntCC::UnsignedLessThan)
    }
}

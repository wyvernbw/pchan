use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub struct OR {
    pub rd: u8,
    pub rs: u8,
    pub rt: u8,
}

impl OR {
    pub const fn new(rd: u8, rs: u8, rt: u8) -> Self {
        Self { rd, rs, rt }
    }
}

impl TryFrom<OpCode> for OR {
    type Error = TryFromOpcodeErr;
    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_secondary(SecOp::OR)?;
        Ok(OR {
            rs: opcode.bits(21..26) as u8,
            rt: opcode.bits(16..21) as u8,
            rd: opcode.bits(11..16) as u8,
        })
    }
}

impl Display for OR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "or ${} ${} ${}",
            REG_STR[self.rd as usize], REG_STR[self.rs as usize], REG_STR[self.rt as usize]
        )
    }
}

impl Op for OR {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::OR)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        if self.rs == 0 {
            // case 1: x | 0 = x
            let (rt, loadrt) = state.emit_get_register(self.rt);

            EmitSummary::builder()
                .instructions([now(loadrt)])
                .register_updates([(self.rd, rt)])
                .build(state.fn_builder)
        } else if self.rt == 0 {
            // case 2: 0 | x = x
            let (rs, loadrs) = state.emit_get_register(self.rs);

            EmitSummary::builder()
                .instructions([now(loadrs)])
                .register_updates([(self.rd, rs)])
                .build(state.fn_builder)
        } else {
            // case 3: x | y = z
            let (rs, loadrs) = state.emit_get_register(self.rs);
            let (rt, loadrt) = state.emit_get_register(self.rt);
            let (rd, bor) = state.inst(|f| f.pure().Binary(Opcode::Bor, types::I32, rs, rt).0);

            EmitSummary::builder()
                .instructions([now(loadrs), now(loadrt), now(bor)])
                .register_updates([(self.rd, rd)])
                .build(state.fn_builder)
        }
    }
}

#[inline]
pub fn or(rd: u8, rs: u8, rt: u8) -> OpCode {
    OR { rd, rs, rt }.into_opcode()
}

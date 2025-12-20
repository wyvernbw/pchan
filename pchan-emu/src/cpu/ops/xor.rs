use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub struct XOR {
    pub rd: u8,
    pub rs: u8,
    pub rt: u8,
}

impl XOR {
    pub const fn new(rd: u8, rs: u8, rt: u8) -> Self {
        Self { rd, rs, rt }
    }
}

impl TryFrom<OpCode> for XOR {
    type Error = TryFromOpcodeErr;
    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_secondary(SecOp::XOR)?;
        Ok(XOR {
            rs: opcode.bits(21..26) as u8,
            rt: opcode.bits(16..21) as u8,
            rd: opcode.bits(11..16) as u8,
        })
    }
}

impl Display for XOR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "xor ${} ${} ${}",
            REG_STR[self.rd as usize], REG_STR[self.rs as usize], REG_STR[self.rt as usize]
        )
    }
}

impl Op for XOR {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::XOR)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        if self.rs == 0 {
            let (rt, loadrt) = state.emit_get_register(self.rt);
            return EmitSummary::builder()
                .instructions([now(loadrt)])
                .register_updates([(self.rd, rt)])
                .build(state.fn_builder);
        } else if self.rt == 0 {
            let (rs, loadrs) = state.emit_get_register(self.rs);
            return EmitSummary::builder()
                .instructions([now(loadrs)])
                .register_updates([(self.rd, rs)])
                .build(state.fn_builder);
        }
        let (rs, loadrs) = state.emit_get_register(self.rs);
        let (rt, loadrt) = state.emit_get_register(self.rt);
        let (rd, bxor) = state.inst(|f| f.pure().Binary(Opcode::Bxor, types::I32, rs, rt).0);
        EmitSummary::builder()
            .instructions([now(loadrs), now(loadrt), now(bxor)])
            .register_updates([(self.rd, rd)])
            .build(state.fn_builder)
    }
}

#[inline]
pub fn xor(rd: u8, rs: u8, rt: u8) -> OpCode {
    XOR { rd, rs, rt }.into_opcode()
}

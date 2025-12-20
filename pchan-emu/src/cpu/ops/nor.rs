use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub struct NOR {
    pub rd: u8,
    pub rs: u8,
    pub rt: u8,
}

impl NOR {
    pub const fn new(rd: u8, rs: u8, rt: u8) -> Self {
        Self { rd, rs, rt }
    }
}

impl TryFrom<OpCode> for NOR {
    type Error = TryFromOpcodeErr;
    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_secondary(SecOp::NOR)?;
        Ok(NOR {
            rs: opcode.bits(21..26) as u8,
            rt: opcode.bits(16..21) as u8,
            rd: opcode.bits(11..16) as u8,
        })
    }
}

impl Display for NOR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "nor ${} ${} ${}",
            REG_STR[self.rd as usize], REG_STR[self.rs as usize], REG_STR[self.rt as usize]
        )
    }
}

impl Op for NOR {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::NOR)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        use crate::cranelift_bs::*;
        // case 1: x | 0 = x
        if self.rs == 0 {
            let (rt, loadrt) = state.emit_get_register(self.rt);
            let (rd, bnot) = state.inst(|f| f.pure().Unary(Opcode::Bnot, types::I32, rt).0);

            EmitSummary::builder()
                .instructions([now(loadrt), now(bnot)])
                .register_updates([(self.rd, rd)])
                .build(state.fn_builder)
        // case 2: 0 | x = x
        } else if self.rt == 0 {
            let (rs, loadrs) = state.emit_get_register(self.rs);
            let (rd, bnot) = state.inst(|f| f.pure().Unary(Opcode::Bnot, types::I32, rs).0);

            EmitSummary::builder()
                .instructions([now(loadrs), now(bnot)])
                .register_updates([(self.rd, rd)])
                .build(state.fn_builder)
        // case 3: x | y = z
        } else {
            let (rs, loadrs) = state.emit_get_register(self.rs);
            let (rt, loadrt) = state.emit_get_register(self.rt);
            let (rs_or_rt, bor) =
                state.inst(|f| f.pure().Binary(Opcode::Bor, types::I32, rs, rt).0);
            let (rd, bnot) = state.inst(|f| f.pure().Unary(Opcode::Bnot, types::I32, rs_or_rt).0);

            EmitSummary::builder()
                .instructions([now(loadrs), now(loadrt), now(bor), now(bnot)])
                .register_updates([(self.rd, rd)])
                .build(state.fn_builder)
        }
    }
}

#[inline]
pub fn nor(rd: u8, rs: u8, rt: u8) -> OpCode {
    NOR { rd, rs, rt }.into_opcode()
}

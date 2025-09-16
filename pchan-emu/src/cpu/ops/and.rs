use std::fmt::Display;

use crate::FnBuilderExt;
use crate::dynarec::prelude::*;

use super::PrimeOp;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct AND {
    rd: usize,
    rs: usize,
    rt: usize,
}

impl TryFrom<OpCode> for AND {
    type Error = TryFromOpcodeErr;
    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_secondary(SecOp::AND)?;
        Ok(AND {
            rs: opcode.bits(21..26) as usize,
            rt: opcode.bits(16..21) as usize,
            rd: opcode.bits(11..16) as usize,
        })
    }
}

impl Display for AND {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "and ${} ${} ${}",
            REG_STR[self.rd], REG_STR[self.rs], REG_STR[self.rt]
        )
    }
}

impl Op for AND {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::AND)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        use crate::cranelift_bs::*;
        // shortcuts:
        // - case 1: x & 0 = 0
        // - case 2: 0 & x = 0
        if self.rs == 0 || self.rt == 0 {
            let (zero, loadzero) = state.emit_get_zero();
            return EmitSummary::builder()
                .instructions([now(loadzero)])
                .register_updates([(self.rd, zero)])
                .build(state.fn_builder);
        }
        let (rs, load0) = state.emit_get_register(self.rs);
        let (rt, load1) = state.emit_get_register(self.rt);
        let (rd, band) = state
            .fn_builder
            .inst(|f| f.ins().Binary(Opcode::Band, types::I32, rs, rt).0);
        EmitSummary::builder()
            .instructions([now(load0), now(load1), now(band)])
            .register_updates([(self.rd, rd)])
            .build(state.fn_builder)
    }
}

#[inline]
pub fn and(rd: usize, rs: usize, rt: usize) -> OpCode {
    AND { rd, rs, rt }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::cpu::ops::prelude::*;
    use crate::dynarec::JitSummary;
    use crate::{Emu, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    #[case(1, 1, 1)]
    #[case(1, 0, 0)]
    #[case(0, 1, 0)]
    #[case(0, 0, 0)]
    #[case(0b00110101, 0b0000111, 0b00000101)]
    fn and_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: i16,
        #[case] b: i16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        emulator.mem.write_array(
            KSEG0Addr::from_phys(0),
            &[addiu(8, 0, a), addiu(9, 0, b), and(10, 8, 9), OpCode(69420)],
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
}

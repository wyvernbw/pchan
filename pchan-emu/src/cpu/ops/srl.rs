use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct SRL {
    rd: usize,
    rt: usize,
    imm: i16,
}

impl TryFrom<OpCode> for SRL {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::SRL)?;
        Ok(SRL {
            rt: opcode.bits(16..21) as usize,
            rd: opcode.bits(11..16) as usize,
            imm: opcode.bits(6..11) as i16,
        })
    }
}

impl Display for SRL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "srl ${} ${} {}",
            REG_STR[self.rd], REG_STR[self.rt], self.imm
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
pub fn srl(rd: usize, rt: usize, imm: i16) -> OpCode {
    SRL { rd, rt, imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    #[case(64, 6, 1)]
    #[case(32, 0, 32)]
    #[case(-32, 2, 1073741816)]
    #[case(0b11110000, 4, 0b00001111)]
    fn srl_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: i16,
        #[case] b: i16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator.mem.write_many(
            0x0,
            &program([addiu(8, 0, a), srl(10, 8, b), OpCode(69420)]),
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
    #[rstest]
    #[case(8)]
    #[case(0b00001111)]
    fn srl_2(setup_tracing: (), mut emulator: Emu, #[case] imm: i16) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator
            .mem
            .write_many(0x0, &program([srl(10, 0, imm), OpCode(69420)]));
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 0);
        Ok(())
    }
}

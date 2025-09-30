use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub struct SLL {
    pub rd: u8,
    pub rt: u8,
    pub imm: i8,
}

impl SLL {
    pub fn new(rd: u8, rt: u8, imm: i8) -> Self {
        Self { rd, rt, imm }
    }
}

impl TryFrom<OpCode> for SLL {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::SLL)?;
        Ok(SLL {
            rt: opcode.bits(16..21) as u8,
            rd: opcode.bits(11..16) as u8,
            imm: opcode.bits(6..11) as i8,
        })
    }
}

impl Display for SLL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sll ${} ${} {}",
            REG_STR[self.rd as usize],
            REG_STR[self.rt as usize],
            hex(self.imm)
        )
    }
}

impl Op for SLL {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::SLL)
            .set_bits(11..16, self.rd as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(6..11, (self.imm as i32 as i16) as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        shiftimm!(self, ctx, Opcode::IshlImm)
    }
}

#[inline]
pub fn sll(rd: u8, rt: u8, imm: i8) -> OpCode {
    SLL { rd, rt, imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    #[case(1, 6, 64)]
    #[case(32, 0, 32)]
    #[case(0b00001111, 4, 0b11110000)]
    fn sll_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: i16,
        #[case] b: i8,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator.write_many(
            0x0,
            &program([addiu(8, 0, a), sll(10, 8, b), OpCode(69420)]),
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
    #[rstest]
    #[case(8)]
    fn sll_2(setup_tracing: (), mut emulator: Emu, #[case] imm: i8) -> color_eyre::Result<()> {
        emulator.write_many(0, &program([sll(10, 0, imm), OpCode(69420)]));
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 0);
        Ok(())
    }
}

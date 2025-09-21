use std::fmt::Display;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy)]
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

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::Emu;
    use crate::dynarec::prelude::*;
    use crate::test_utils::emulator;

    #[rstest]
    #[case(2, 3, 6)]
    #[case(0xFFFFFFFF, 2, 0xFFFFFFFE)] // low 32 bits of 0x1FFFFFFFE
    #[case(0x80000000, 2, 0)] // low 32 bits of 0x100000000
    #[case(0xFFFFFFFF, 0xFFFFFFFF, 1)] // low 32 bits of 0xFFFFFFFE00000001
    pub fn mflo_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: u32,
        #[case] b: u32,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        emulator
            .mem
            .write_many(0x0, &program([mult(8, 9), nop(), mflo(10), OpCode(69420)]));
        emulator.cpu.gpr[8] = a;
        emulator.cpu.gpr[9] = b;

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        let output = emulator.cpu.gpr[10];
        assert_eq!(output, expected);

        Ok(())
    }
}

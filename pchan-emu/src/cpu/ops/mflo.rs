use std::fmt::Display;

use crate::cpu::REG_STR;
use crate::cpu::ops::prelude::*;
use crate::cranelift_bs::*;

#[derive(Debug, Clone, Copy)]
pub struct MFLO {
    rd: usize,
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

    fn emit_ir(
        &self,
        mut state: EmitParams,
        fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
        let lo = state.emit_get_lo(fn_builder);
        Some(
            EmitSummary::builder()
                .register_updates([(self.rd, lo)])
                .build(),
        )
    }
}

impl TryFrom<OpCode> for MFLO {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::MFLO)?;
        Ok(MFLO {
            rd: value.bits(11..16) as usize,
        })
    }
}

impl Display for MFLO {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mflo ${}", REG_STR[self.rd])
    }
}

pub fn mflo(rd: usize) -> OpCode {
    MFLO { rd }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::Emu;
    use crate::cpu::ops::prelude::*;
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
        use crate::{JitSummary, memory::KSEG0Addr};

        emulator.mem.write_array(
            KSEG0Addr::from_phys(0),
            &[mult(8, 9), nop(), mflo(10), OpCode(69420)],
        );
        emulator.cpu.gpr[8] = a;
        emulator.cpu.gpr[9] = b;

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        let output = emulator.cpu.gpr[10];
        assert_eq!(output, expected);

        Ok(())
    }
}

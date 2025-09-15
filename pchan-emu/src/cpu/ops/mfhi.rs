use std::fmt::Display;

use crate::cpu::REG_STR;
use crate::cpu::ops::prelude::*;
use crate::cranelift_bs::*;

#[derive(Debug, Clone, Copy)]
pub struct MFHI {
    rd: usize,
}

impl Op for MFHI {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::MFHI)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(
        &self,
        mut state: EmitParams,
        fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
        let hi = state.emit_get_hi(fn_builder);
        Some(
            EmitSummary::builder()
                .register_updates([(self.rd, hi)])
                .build(fn_builder),
        )
    }
}

impl TryFrom<OpCode> for MFHI {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::MFHI)?;
        Ok(MFHI {
            rd: value.bits(11..16) as usize,
        })
    }
}

impl Display for MFHI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mfhi ${}", REG_STR[self.rd])
    }
}

pub fn mfhi(rd: usize) -> OpCode {
    MFHI { rd }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::Emu;
    use crate::cpu::ops::prelude::*;
    use crate::test_utils::emulator;

    #[rstest]
    #[case(2, 3, 0)]
    #[case(0xFFFF_FFFF, 2, 1)]
    #[case(0x8000_0000, 2, 1)]
    #[case(0x1234_5678, 0x1000, 0x00000123)]
    #[case(0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFE)]
    pub fn mfhi_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: u32,
        #[case] b: u32,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        use crate::{dynarec::JitSummary, memory::KSEG0Addr};

        emulator.mem.write_array(
            KSEG0Addr::from_phys(0),
            &[multu(8, 9), nop(), mfhi(10), OpCode(69420)],
        );
        emulator.cpu.gpr[8] = a;
        emulator.cpu.gpr[9] = b;

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        let output = emulator.cpu.gpr[10];
        tracing::info!("hilo=0x{:016X}", emulator.cpu.hilo);
        assert_eq!(output, expected);

        Ok(())
    }
}

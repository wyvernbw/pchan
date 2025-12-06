use std::fmt::Display;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy, Hash)]
pub struct MFHI {
    rd: u8,
}

impl MFHI {
    pub fn new(rd: u8) -> Self {
        Self { rd }
    }
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

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        let (hi, loadhi) = state.emit_get_hi();
        EmitSummary::builder()
            .instructions([now(loadhi)])
            .register_updates([(self.rd, hi)])
            .build(state.fn_builder)
    }
}

impl TryFrom<OpCode> for MFHI {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::MFHI)?;
        Ok(MFHI {
            rd: value.bits(11..16) as u8,
        })
    }
}

impl Display for MFHI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mfhi ${}", REG_STR[self.rd as usize])
    }
}

pub fn mfhi(rd: u8) -> OpCode {
    MFHI { rd }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::Emu;
    use crate::dynarec::prelude::*;
    use crate::test_utils::{emulator, jit};

    #[rstest]
    #[case(2, 3, 0)]
    #[case(0xFFFF_FFFF, 2, 1)]
    #[case(0x8000_0000, 2, 1)]
    #[case(0x1234_5678, 0x1000, 0x00000123)]
    #[case(0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFE)]
    pub fn mfhi_1(
        setup_tracing: (),
        mut emulator: Emu,
        mut jit: crate::jit::JIT,
        #[case] a: u32,
        #[case] b: u32,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        emulator.write_many(0, &program([multu(8, 9), nop(), mfhi(10), OpCode(69420)]));
        emulator.cpu.gpr[8] = a;
        emulator.cpu.gpr[9] = b;

        let summary = emulator.step_jit_summarize::<JitSummary>(&mut jit)?;
        tracing::info!(?summary.function);
        let output = emulator.cpu.gpr[10];
        tracing::info!("hilo=0x{:016X}", emulator.cpu.hilo);
        assert_eq!(output, expected);

        Ok(())
    }
}

use crate::{cpu::reg_str, dynarec::prelude::*};
use std::fmt::Display;

#[derive(Debug, Clone, Copy)]
pub struct MTLO {
    rs: u8,
}

impl MTLO {
    pub const fn new(rs: u8) -> Self {
        Self { rs }
    }
}

impl Op for MTLO {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::MTLO)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        let (rs, loadreg) = state.emit_get_register(self.rs);
        EmitSummary::builder()
            .lo(rs)
            .instructions([now(loadreg)])
            .build(state.fn_builder)
    }
}

impl TryFrom<OpCode> for MTLO {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::MTLO)?;
        Ok(MTLO {
            rs: value.bits(21..26) as u8,
        })
    }
}

impl Display for MTLO {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mtlo ${}", reg_str(self.rs))
    }
}

pub fn mtlo(rs: u8) -> OpCode {
    MTLO { rs }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::Emu;
    use crate::dynarec::prelude::*;
    use crate::test_utils::emulator;

    #[rstest]
    pub fn mtlo_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator
            .mem
            .write_many(0, &program([multu(8, 9), nop(), mtlo(10), OpCode(69420)]));
        emulator.cpu.gpr[10] = 0x1234;
        emulator.cpu.gpr[8] = 16;
        emulator.cpu.gpr[9] = 16;

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        let output = emulator.cpu.hilo;
        tracing::info!("hilo=0x{:016X}", emulator.cpu.hilo);
        assert_eq!(output as u32, 0x1234);

        Ok(())
    }
}

use std::fmt::Display;

use crate::cpu::REG_STR;
use crate::cpu::ops::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct MTHI {
    rs: usize,
}

impl Op for MTHI {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::MTHI)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(&self, mut state: EmitParams) -> Option<EmitSummary> {
        let rs = state.emit_get_register(self.rs);
        Some(EmitSummary::builder().hi(rs).build(state.fn_builder))
    }
}

impl TryFrom<OpCode> for MTHI {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::MTHI)?;
        Ok(MTHI {
            rs: value.bits(21..26) as usize,
        })
    }
}

impl Display for MTHI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mthi ${}", REG_STR[self.rs])
    }
}

pub fn mthi(rs: usize) -> OpCode {
    MTHI { rs }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::Emu;
    use crate::cpu::ops::prelude::*;
    use crate::test_utils::emulator;

    #[rstest]
    pub fn mthi_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::{dynarec::JitSummary, memory::KSEG0Addr};

        emulator.mem.write_array(
            KSEG0Addr::from_phys(0),
            &[multu(8, 9), nop(), mthi(10), OpCode(69420)],
        );
        emulator.cpu.gpr[10] = 0x1234;
        emulator.cpu.gpr[8] = 16;
        emulator.cpu.gpr[9] = 16;

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        let output = emulator.cpu.hilo;
        tracing::info!("hilo=0x{:016X}", emulator.cpu.hilo);
        assert_eq!(output >> 32, 0x1234);

        Ok(())
    }
}

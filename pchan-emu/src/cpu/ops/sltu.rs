use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SLTU {
    rd: u8,
    rs: u8,
    rt: u8,
}

impl SLTU {
    pub const fn new(rd: u8, rs: u8, rt: u8) -> Self {
        Self { rd, rs, rt }
    }
}

pub fn sltu(rd: u8, rs: u8, rt: u8) -> OpCode {
    SLTU { rd, rs, rt }.into_opcode()
}

impl Display for SLTU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sltu ${} ${} ${}",
            REG_STR[self.rd as usize], REG_STR[self.rs as usize], REG_STR[self.rt as usize]
        )
    }
}

impl TryFrom<OpCode> for SLTU {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::SLTU)?;
        Ok(SLTU {
            rd: value.bits(11..16) as u8,
            rt: value.bits(16..21) as u8,
            rs: value.bits(21..26) as u8,
        })
    }
}

impl Op for SLTU {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::SLTU)
            .set_bits(11..16, self.rd as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        icmp!(self, ctx, IntCC::UnsignedLessThan)
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::jit::JIT;
    use crate::test_utils::jit;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    fn basic_sltu(setup_tracing: (), mut emulator: Emu, mut jit: JIT) -> color_eyre::Result<()> {
        emulator.write_many(
            0,
            &program([
                addiu(8, 0, 16),
                addiu(9, 0, -3),
                sltu(10, 9, 8),
                OpCode(69420),
            ]),
        );
        let summary = emulator.step_jit_summarize::<JitSummary>(&mut jit)?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 0);

        Ok(())
    }
}

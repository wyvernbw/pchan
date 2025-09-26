use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy)]
pub struct MULTU {
    rs: u8,
    rt: u8,
}

impl MULTU {
    pub const fn new(rs: u8, rt: u8) -> Self {
        Self { rs, rt }
    }
}

impl Op for MULTU {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::MULTU)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        mult!(self, ctx, Opcode::Umulhi)
    }
}

impl Display for MULTU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "multu ${} ${}",
            REG_STR[self.rs as usize], REG_STR[self.rt as usize]
        )
    }
}

impl TryFrom<OpCode> for MULTU {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::MULTU)?;
        Ok(MULTU {
            rs: value.bits(21..26) as u8,
            rt: value.bits(16..21) as u8,
        })
    }
}

pub fn multu(rs: u8, rt: u8) -> OpCode {
    MULTU { rs, rt }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    #[case(1, 1, 1)]
    #[case(0xFFFFFFFF, 2, 0x1FFFFFFFE)]
    #[case(2, 0, 0)]
    #[case(2_000_000_000, 2_000_000_000, 4_000_000_000_000_000_000)]
    pub fn multu_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: u32,
        #[case] b: u32,
        #[case] expected: u64,
    ) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator.write_many(0, &program([multu(8, 9), OpCode(69420)]));
        emulator.cpu.gpr[8] = a;
        emulator.cpu.gpr[9] = b;

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        let output = emulator.cpu.hilo;
        assert_eq!(output, expected);

        Ok(())
    }

    #[rstest]
    #[case(2, 0, 0)]
    #[case(0, 2, 0)]
    #[case(0, 0, 0)]
    pub fn multu_2_shortpath(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: u8,
        #[case] b: u8,
        #[case] expected: u64,
    ) -> color_eyre::Result<()> {
        assert!(a == 0 || b == 0);

        emulator.write_many(0, &program([multu(a, b), OpCode(69420)]));
        if a != 0 {
            emulator.cpu.gpr[a as usize] = 32;
        }
        if b != 0 {
            emulator.cpu.gpr[b as usize] = 1;
        }

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        let output = emulator.cpu.hilo;
        assert_eq!(output, expected);

        let op_count = summary.function.unwrap().dfg.num_insts();

        assert!(op_count <= 3 + 2 + 1 + 3);

        Ok(())
    }
}

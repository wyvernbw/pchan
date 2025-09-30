use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash)]
pub struct SLLV {
    rd: u8,
    rt: u8,
    rs: u8,
}

impl SLLV {
    pub fn new(rd: u8, rt: u8, rs: u8) -> Self {
        Self { rd, rt, rs }
    }
}

impl Display for SLLV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sllv ${} ${} ${}",
            REG_STR[self.rd as usize], REG_STR[self.rt as usize], REG_STR[self.rs as usize]
        )
    }
}

impl TryFrom<OpCode> for SLLV {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::SLLV)?;
        Ok(SLLV {
            rd: value.bits(11..16) as u8,
            rt: value.bits(16..21) as u8,
            rs: value.bits(21..26) as u8,
        })
    }
}

impl Op for SLLV {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::SLLV)
            .set_bits(11..16, self.rd as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        shift!(self, ctx, Opcode::Ishl)
    }
}

#[inline]
pub fn sllv(rd: u8, rt: u8, rs: u8) -> OpCode {
    SLLV { rd, rs, rt }.into_opcode()
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
    fn sllv_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: i16,
        #[case] b: i16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        emulator.write_many(
            0x0,
            &program([
                addiu(8, 0, a),
                addiu(9, 0, b),
                sllv(10, 8, 9),
                OpCode(69420),
            ]),
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
    #[rstest]
    #[case(8, 0)]
    #[case(0b00001111, 0)]
    fn sllv_2(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] value: i16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        emulator.write_many(
            0x0,
            &program([addiu(9, 0, value), sllv(10, 0, 9), OpCode(69420)]),
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
}

use std::fmt::Display;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct ANDI {
    rs: usize,
    rt: usize,
    imm: u16,
}

impl TryFrom<OpCode> for ANDI {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::ANDI)?;
        Ok(ANDI {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as u16,
        })
    }
}

impl Display for ANDI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "andi ${} ${} {}",
            REG_STR[self.rt], REG_STR[self.rs], self.imm
        )
    }
}

impl Op for ANDI {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::ANDI)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        use crate::cranelift_bs::*;
        // shortcuts:
        // - case 1: x & 0 = 0
        // - case 2: 0 & x = 0
        if self.rs == 0 || self.imm == 0 {
            let (rt, loadzero) = state.emit_get_zero();
            return EmitSummary::builder()
                .instructions([now(loadzero)])
                .register_updates([(self.rt, rt)])
                .build(state.fn_builder);
        }
        let (rs, loadrs) = state.emit_get_register(self.rs);
        let (rt, band) = state.inst(|f| {
            f.pure()
                .BinaryImm64(Opcode::BandImm, types::I32, Imm64::new(self.imm as i64), rs)
                .0
        });
        EmitSummary::builder()
            .instructions([now(loadrs), now(band)])
            .register_updates([(self.rt, rt)])
            .build(state.fn_builder)
    }
}

#[inline]
pub fn andi(rt: usize, rs: usize, imm: u16) -> OpCode {
    ANDI { rt, rs, imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::cpu::ops::prelude::*;
    use crate::cpu::program;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    #[case(1, 1, 1)]
    #[case(1, 0, 0)]
    #[case(0, 1, 0)]
    #[case(0, 0, 0)]
    #[case(0b00110101, 0b0000111, 0b00000101)]
    fn andi_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: i16,
        #[case] b: u16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator.mem.write_many(
            0x0,
            &program([addiu(8, 0, a), andi(10, 8, b), OpCode(69420)]),
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
    #[rstest]
    #[case(0b11110000)]
    fn andi_2(setup_tracing: (), mut emulator: Emu, #[case] imm: u16) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator
            .mem
            .write_many(0x0, &program([andi(10, 0, imm), OpCode(69420)]));
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 0);
        Ok(())
    }
}

use std::fmt::Display;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct LUI {
    rt: usize,
    imm: i16,
}

impl Display for LUI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lui ${} {}", REG_STR[self.rt], self.imm)
    }
}

impl TryFrom<OpCode> for LUI {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value.as_primary(PrimeOp::LUI)?;
        Ok(LUI {
            rt: value.bits(16..21) as usize,
            imm: value.bits(0..16) as i16,
        })
    }
}

impl Op for LUI {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::LUI)
            .set_bits(0..16, self.imm as i32 as u32)
            .set_bits(16..21, self.rt as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        if self.imm == 0 {
            let (zero, loadzero) = ctx.emit_get_zero();
            return EmitSummary::builder()
                .instructions([now(loadzero)])
                .register_updates([(self.rt, zero)])
                .build(ctx.fn_builder);
        }
        let (rt, iconst) = ctx.inst(|f| {
            f.pure()
                .UnaryImm(
                    Opcode::Iconst,
                    types::I32,
                    Imm64::new(((self.imm as i32) << 16) as i64),
                )
                .0
        });
        EmitSummary::builder()
            .instructions([now(iconst)])
            .register_updates([(self.rt, rt)])
            .build(ctx.fn_builder)
    }
}

pub fn lui(rt: usize, imm: i16) -> OpCode {
    LUI { rt, imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::cpu::ops::prelude::*;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    #[case(8, 0x1234, 0x1234_0000)]
    #[case(8, 0, 0)]
    fn lui_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] reg: usize,
        #[case] value: i16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator
            .mem
            .write_many(0x0, &program([lui(reg, value), OpCode(69420)]));

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[reg], expected);

        Ok(())
    }

    #[rstest]
    fn lui_2(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator
            .mem
            .write_many(0x0, &program([lui(8, 0x1234), OpCode(69420)]));
        emulator.cpu.gpr[8] = 0x1111_1111;

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[8], 0x1234_0000);

        Ok(())
    }
}

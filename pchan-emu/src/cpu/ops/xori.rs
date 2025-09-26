use crate::memory::ext;
use crate::{FnBuilderExt, dynarec::prelude::*};
use std::fmt::Display;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct XORI {
    rs: u8,
    rt: u8,
    imm: u16,
}

impl XORI {
    pub const fn new(rs: u8, rt: u8, imm: u16) -> Self {
        Self { rs, rt, imm }
    }
}

impl TryFrom<OpCode> for XORI {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::XORI)?;
        Ok(XORI {
            rt: opcode.bits(16..21) as u8,
            rs: opcode.bits(21..26) as u8,
            imm: opcode.bits(0..16) as u16,
        })
    }
}

impl Display for XORI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "xori ${} ${} {}",
            REG_STR[self.rt as usize],
            REG_STR[self.rs as usize],
            hex(self.imm)
        )
    }
}

impl Op for XORI {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::XORI)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(0..16, ext::zero(self.imm))
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        if self.rs == 0 {
            let (rt, iconst) = state.fn_builder.IConst(self.imm);
            return EmitSummary::builder()
                .instructions([now(iconst)])
                .register_updates([(self.rt, rt)])
                .build(state.fn_builder);
        } else if self.imm == 0 {
            let (rs, loadrs) = state.emit_get_register(self.rs);
            return EmitSummary::builder()
                .instructions([now(loadrs)])
                .register_updates([(self.rt, rs)])
                .build(state.fn_builder);
        }

        let (rs, loadrs) = state.emit_get_register(self.rs);
        let (rt, bxorimm) = state.inst(|f| {
            f.pure()
                .BinaryImm64(Opcode::BxorImm, types::I32, Imm64::new(self.imm.into()), rs)
                .0
        });
        EmitSummary::builder()
            .instructions([now(loadrs), now(bxorimm)])
            .register_updates([(self.rt, rt)])
            .build(state.fn_builder)
    }
}

#[inline]
pub fn xori(rt: u8, rs: u8, imm: u16) -> OpCode {
    XORI { rt, rs, imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    #[case(1, 1, 0)]
    #[case(1, 0, 1)]
    #[case(0, 1, 1)]
    #[case(0, 0, 0)]
    #[case(0b11110000, 0b00111100, 0b11001100)]
    // FIXME: idk what the fuck is going on
    // #[case(-1i16 as u16 as i16, -1i16 as u16, 0)] // 0xFFFF ^ 0xFFFF = 0
    // #[case(i16::MIN, 0, i32::MIN as u32)]
    fn xori_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: i16,
        #[case] b: u16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        tracing::info!(op = %DecodedOp::new(xori(10, 8, b)));
        emulator.write_many(0, &program([addiu(8, 0, a), xori(10, 8, b), OpCode(69420)]));
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
    #[rstest]
    #[case(0b11110000)]
    fn xori_2(setup_tracing: (), mut emulator: Emu, #[case] imm: u16) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator.write_many(0, &program([xori(10, 0, imm), OpCode(69420)]));
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary);
        assert_eq!(emulator.cpu.gpr[10], imm as u32);
        Ok(())
    }
}

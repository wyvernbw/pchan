use std::fmt::Display;

use cranelift::prelude::FunctionBuilder;

use crate::cpu::{
    REG_STR,
    ops::{BoundaryType, EmitParams, EmitSummary, Op, OpCode, TryFromOpcodeErr},
};

use super::PrimeOp;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct XORI {
    rs: usize,
    rt: usize,
    imm: i16,
}

impl TryFrom<OpCode> for XORI {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::XORI)?;
        Ok(XORI {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Display for XORI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "xori ${} ${} {}",
            REG_STR[self.rt], REG_STR[self.rs], self.imm
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
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }

    fn emit_ir(
        &self,
        mut state: EmitParams,
        fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
        use crate::cranelift_bs::*;
        if self.rs == 0 {
            let rt = fn_builder.ins().iconst(types::I64, self.imm as i64);
            return Some(
                EmitSummary::builder()
                    .register_updates(vec![(self.rt, rt)].into())
                    .build(),
            );
        }
        let rs = state.emit_get_register(fn_builder, self.rs);
        let rt = fn_builder.ins().bxor_imm(rs, self.imm as i64);
        Some(
            EmitSummary::builder()
                .register_updates(vec![(self.rt, rt)].into())
                .build(),
        )
    }
}

#[inline]
pub fn xori(rt: usize, rs: usize, imm: i16) -> OpCode {
    XORI { rt, rs, imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::cpu::ops::prelude::*;
    use crate::{Emu, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    #[case(1, 1, 0)]
    #[case(1, 0, 1)]
    #[case(0, 1, 1)]
    #[case(0, 0, 0)]
    #[case(0b11110000, 0b00111100, 0b11001100)]
    #[case(-1, -1, 0)] // 0xFFFF ^ 0xFFFF = 0
    #[case(i16::MIN, 0, i16::MIN as u64)] // -32768 ^ 0 = -32768
    fn xori_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: i16,
        #[case] b: i16,
        #[case] expected: u64,
    ) -> color_eyre::Result<()> {
        use crate::JitSummary;

        emulator.mem.write_array(
            KSEG0Addr::from_phys(0),
            &[addiu(8, 0, a), xori(10, 8, b), OpCode(69420)],
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
    #[rstest]
    #[case(0b11110000)]
    fn xori_2(setup_tracing: (), mut emulator: Emu, #[case] imm: i16) -> color_eyre::Result<()> {
        use crate::JitSummary;

        emulator
            .mem
            .write_array(KSEG0Addr::from_phys(0), &[xori(10, 0, imm), OpCode(69420)]);
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], imm as u64);
        Ok(())
    }
}

use std::fmt::Display;

use cranelift::prelude::FunctionBuilder;

use crate::cpu::REG_STR;
use crate::cpu::ops::prelude::*;

use super::PrimeOp;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct SRA {
    rd: usize,
    rt: usize,
    imm: i16,
}

impl TryFrom<OpCode> for SRA {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::SRA)?;
        Ok(SRA {
            rt: opcode.bits(16..21) as usize,
            rd: opcode.bits(11..16) as usize,
            imm: opcode.bits(6..11) as i16,
        })
    }
}

impl Display for SRA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sra ${} ${} {}",
            REG_STR[self.rd], REG_STR[self.rt], self.imm
        )
    }
}

impl Op for SRA {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::SRA)
            .set_bits(11..16, self.rd as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(6..11, (self.imm as i32 as i16) as u32)
    }

    fn emit_ir(
        &self,
        mut state: EmitParams,
        fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
        use crate::cranelift_bs::*;
        tracing::info!(?self);
        // case 1: $rt << 0 = $rt
        if self.imm == 0 {
            let rt = state.emit_get_register(fn_builder, self.rt);
            return Some(
                EmitSummary::builder()
                    .register_updates([(self.rd, rt)])
                    .build(&fn_builder),
            );
        }
        // case 2: 0 << imm = 0
        if self.rt == 0 {
            let rt = state.emit_get_zero(fn_builder);
            return Some(
                EmitSummary::builder()
                    .register_updates([(self.rd, rt)])
                    .build(&fn_builder),
            );
        }
        // case 3: $rt << imm = $rd
        let rt = state.emit_get_register(fn_builder, self.rt);
        let rd = fn_builder.ins().sshr_imm(rt, self.imm as i64);
        Some(
            EmitSummary::builder()
                .register_updates([(self.rd, rd)])
                .build(&fn_builder),
        )
    }
}

#[inline]
pub fn sra(rd: usize, rt: usize, imm: i16) -> OpCode {
    SRA { rd, rt, imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::cpu::ops::prelude::*;
    use crate::{Emu, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    #[case(64, 6, 1)]
    #[case(32, 0, 32)]
    #[case(-32, 2, -8i32 as u32)]
    #[case(0b11110000, 4, 0b00001111)]
    fn sra_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: i16,
        #[case] b: i16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        use crate::JitSummary;

        emulator.mem.write_array(
            KSEG0Addr::from_phys(0),
            &[addiu(8, 0, a), sra(10, 8, b), OpCode(69420)],
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
    #[rstest]
    #[case(8)]
    #[case(0b00001111)]
    fn sra_2(setup_tracing: (), mut emulator: Emu, #[case] imm: i16) -> color_eyre::Result<()> {
        use crate::JitSummary;

        emulator
            .mem
            .write_array(KSEG0Addr::from_phys(0), &[sra(10, 0, imm), OpCode(69420)]);
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 0);
        Ok(())
    }
}

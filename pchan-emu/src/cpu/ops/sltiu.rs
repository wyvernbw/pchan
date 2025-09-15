use std::fmt::Display;

use cranelift::prelude::FunctionBuilder;

use crate::cpu::REG_STR;
use crate::cpu::ops::{OpCode, prelude::*};

#[derive(Debug, Clone, Copy)]
pub struct SLTIU {
    rt: usize,
    rs: usize,
    imm: i16,
}

pub fn sltiu(rt: usize, rs: usize, imm: i16) -> OpCode {
    SLTIU { rs, rt, imm }.into_opcode()
}

impl Display for SLTIU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sltiu ${} ${} {}",
            REG_STR[self.rt], REG_STR[self.rs], self.imm
        )
    }
}

impl TryFrom<OpCode> for SLTIU {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value.as_primary(PrimeOp::SLTIU)?;
        Ok(SLTIU {
            imm: value.bits(0..16) as i16,
            rt: value.bits(16..21) as usize,
            rs: value.bits(21..26) as usize,
        })
    }
}

impl Op for SLTIU {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SLTIU)
            .set_bits(0..16, self.imm as i32 as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(
        &self,
        mut state: EmitParams,
        fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
        use crate::cranelift_bs::*;

        // x < 0u64 (u64::MIN) = false
        if self.imm == 0 {
            return Some(
                EmitSummary::builder()
                    .register_updates([(self.rt, state.emit_get_zero(fn_builder))])
                    .build(fn_builder),
            );
        }

        // 0u64 < x = true
        if self.rs == 0 {
            return Some(
                EmitSummary::builder()
                    .register_updates([(self.rt, state.emit_get_one(fn_builder))])
                    .build(fn_builder),
            );
        }

        let rs = state.emit_get_register(fn_builder, self.rs);
        let rt = fn_builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThan, rs, self.imm as i64);
        let rt = fn_builder.ins().uextend(types::I32, rt);
        Some(
            EmitSummary::builder()
                .register_updates([(self.rt, rt)])
                .build(fn_builder),
        )
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::dynarec::JitSummary;
    use crate::cpu::ops::prelude::*;
    use crate::memory::KSEG0Addr;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    fn slti_test(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator.mem.write_array(
            KSEG0Addr::from_phys(0),
            &[addiu(8, 0, -3), sltiu(9, 8, 32), OpCode(69420)],
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[9], 0);

        Ok(())
    }

    /// $t0 < 0 = false
    #[rstest]
    fn sltiu_2_shortpath(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator
            .mem
            .write_array(KSEG0Addr::from_phys(0), &[sltiu(10, 8, 0), OpCode(69420)]);
        emulator.cpu.gpr[8] = 32;
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 0);
        let op_count = summary.function.unwrap().dfg.num_insts();
        assert!(op_count <= 5);
        Ok(())
    }

    /// $zero < 8 = true
    #[rstest]
    fn sltiu_3_shortpath(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator
            .mem
            .write_array(KSEG0Addr::from_phys(0), &[sltiu(10, 0, 8), OpCode(69420)]);
        emulator.cpu.gpr[8] = 32;
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 1);
        let op_count = summary.function.unwrap().dfg.num_insts();
        assert!(op_count <= 5);
        Ok(())
    }
}

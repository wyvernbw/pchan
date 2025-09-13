use std::fmt::Display;

use crate::cpu::REG_STR;
use crate::cpu::ops::prelude::*;
use crate::cranelift_bs::*;

#[derive(Debug, Clone, Copy)]
pub struct MULTU {
    rs: usize,
    rt: usize,
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

    fn emit_ir(
        &self,
        mut state: EmitParams,
        fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
        // case 1: $rs = 0 or $rt = 0 => $hi:$lo=0
        if self.rs == 0 || self.rt == 0 {
            return Some(
                EmitSummary::builder()
                    .hi(state.emit_get_zero(fn_builder))
                    .lo(state.emit_get_zero(fn_builder))
                    .build(),
            );
        }

        let rs = state.emit_get_register(fn_builder, self.rs);
        // let rs = fn_builder.ins().uextend(types::I64, rs);
        let rt = state.emit_get_register(fn_builder, self.rt);
        // let rt = fn_builder.ins().uextend(types::I64, rt);

        let lo = fn_builder.ins().imul(rs, rt);
        let hi = fn_builder.ins().umulhi(rs, rt);

        // // Extend to 64-bit
        // let lo64 = fn_builder.ins().uextend(types::I64, lo);
        // let hi64 = fn_builder.ins().uextend(types::I64, hi);

        // // Shift high half into upper 32 bits
        // let hi64_shifted = fn_builder.ins().ishl_imm(hi64, 32);

        // // Combine high and low halves
        // let full64 = fn_builder.ins().bor(hi64_shifted, lo64);
        Some(EmitSummary::builder().hi(hi).lo(lo).build())
    }
}

impl Display for MULTU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "multu ${} ${}", REG_STR[self.rs], REG_STR[self.rt])
    }
}

impl TryFrom<OpCode> for MULTU {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::MULTU)?;
        Ok(MULTU {
            rs: value.bits(21..26) as usize,
            rt: value.bits(16..21) as usize,
        })
    }
}

pub fn multu(rs: usize, rt: usize) -> OpCode {
    MULTU { rs, rt }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::cpu::ops::prelude::*;
    use crate::memory::KSEG0Addr;
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
        use crate::JitSummary;

        emulator
            .mem
            .write_array(KSEG0Addr::from_phys(0), &[multu(8, 9), OpCode(69420)]);
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
        #[case] a: usize,
        #[case] b: usize,
        #[case] expected: u64,
    ) -> color_eyre::Result<()> {
        assert!(a == 0 || b == 0);

        use crate::JitSummary;

        emulator
            .mem
            .write_array(KSEG0Addr::from_phys(0), &[multu(a, b), OpCode(69420)]);
        if a != 0 {
            emulator.cpu.gpr[a] = 32;
        }
        if b != 0 {
            emulator.cpu.gpr[b] = 1;
        }

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        let output = emulator.cpu.hilo;
        assert_eq!(output, expected);

        let op_count = summary.function.unwrap().dfg.num_insts();

        assert!(op_count <= 3 + 2 + 1);

        Ok(())
    }
}

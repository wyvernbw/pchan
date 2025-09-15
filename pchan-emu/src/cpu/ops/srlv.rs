use std::fmt::Display;

use crate::cpu::REG_STR;
use crate::cpu::ops::prelude::*;
use crate::cranelift_bs::*;

#[derive(Debug, Clone, Copy)]
pub struct SRLV {
    rd: usize,
    rt: usize,
    rs: usize,
}

impl Display for SRLV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "srlv ${} ${} ${}",
            REG_STR[self.rd], REG_STR[self.rt], REG_STR[self.rs]
        )
    }
}

impl TryFrom<OpCode> for SRLV {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::SRLV)?;
        Ok(SRLV {
            rd: value.bits(11..16) as usize,
            rt: value.bits(16..21) as usize,
            rs: value.bits(21..26) as usize,
        })
    }
}

impl Op for SRLV {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::SRLV)
            .set_bits(11..16, self.rd as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(
        &self,
        mut state: EmitParams,
        fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
        // optimize 0 >> x = 0
        if self.rt == 0 {
            let rd = state.emit_get_zero(fn_builder);
            return Some(
                EmitSummary::builder()
                    .register_updates([(self.rd, rd)])
                    .build(&fn_builder),
            );
        }
        // optimize x >> 0 = x
        let rt = state.emit_get_register(fn_builder, self.rt);
        if self.rs == 0 {
            return Some(
                EmitSummary::builder()
                    .register_updates([(self.rd, rt)])
                    .build(&fn_builder),
            );
        }
        let rs = state.emit_get_register(fn_builder, self.rs);
        let rd = fn_builder.ins().ushr(rt, rs);
        Some(
            EmitSummary::builder()
                .register_updates([(self.rd, rd)])
                .build(&fn_builder),
        )
    }
}

#[inline]
pub fn srlv(rd: usize, rt: usize, rs: usize) -> OpCode {
    SRLV { rd, rs, rt }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::dynarec::JitSummary;
    use crate::cpu::ops::prelude::*;
    use crate::{Emu, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    #[case(64, 6, 1)]
    #[case(32, 0, 32)]
    #[case(-32, 2, 1073741816)]
    #[case(0b11110000, 4, 0b00001111)]
    fn srlv_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: i16,
        #[case] b: i16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        emulator.mem.write_array(
            KSEG0Addr::from_phys(0),
            &[
                addiu(8, 0, a),
                addiu(9, 0, b),
                srlv(10, 8, 9),
                OpCode(69420),
            ],
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }

    #[rstest]
    #[case(8, 0)]
    #[case(0b00001111, 0)]
    fn srlv_2(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] value: i16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        emulator.mem.write_array(
            KSEG0Addr::from_phys(0),
            &[addiu(9, 0, value), srlv(10, 0, 9), OpCode(69420)],
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
}

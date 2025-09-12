use std::fmt::Display;

use cranelift::prelude::FunctionBuilder;

use crate::cpu::REG_STR;
use crate::cpu::ops::{OpCode, prelude::*};

#[derive(Debug, Clone, Copy)]
pub struct SLTU {
    rd: usize,
    rs: usize,
    rt: usize,
}

pub fn sltu(rd: usize, rs: usize, rt: usize) -> OpCode {
    SLTU { rd, rs, rt }.into_opcode()
}

impl Display for SLTU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sltu ${} ${} ${}",
            REG_STR[self.rd], REG_STR[self.rs], REG_STR[self.rt]
        )
    }
}

impl TryFrom<OpCode> for SLTU {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::SLTU)?;
        Ok(SLTU {
            rd: value.bits(11..16) as usize,
            rt: value.bits(16..21) as usize,
            rs: value.bits(21..26) as usize,
        })
    }
}

impl Op for SLTU {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::SLTU)
            .set_bits(11..16, self.rd as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(
        &self,
        mut state: EmitParams,
        fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
        use crate::cranelift_bs::*;

        if self.rs == 0 {
            return Some(
                EmitSummary::builder()
                    .register_updates([(self.rd, state.emit_get_one(fn_builder))])
                    .build(),
            );
        } else if self.rt == 0 {
            return Some(
                EmitSummary::builder()
                    .register_updates([(self.rd, state.emit_get_zero(fn_builder))])
                    .build(),
            );
        }
        let rt = state.emit_get_register(fn_builder, self.rt);
        let rs = state.emit_get_register(fn_builder, self.rs);
        let rd = fn_builder.ins().icmp(IntCC::UnsignedLessThan, rs, rt);

        Some(
            EmitSummary::builder()
                .register_updates([(self.rd, rd)])
                .build(),
        )
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::JitSummary;
    use crate::cpu::ops::prelude::*;
    use crate::memory::KSEG0Addr;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    fn basic_sltu(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator.mem.write_array(
            KSEG0Addr::from_phys(0),
            &[
                addiu(8, 0, 16),
                addiu(9, 0, -3),
                sltu(10, 9, 8),
                OpCode(69420),
            ],
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 0);

        Ok(())
    }
}

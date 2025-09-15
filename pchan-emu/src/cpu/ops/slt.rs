use std::fmt::Display;

use cranelift::prelude::FunctionBuilder;

use crate::cpu::REG_STR;
use crate::cpu::ops::{OpCode, prelude::*};

#[derive(Debug, Clone, Copy)]
pub struct SLT {
    rd: usize,
    rs: usize,
    rt: usize,
}

pub fn slt(rd: usize, rs: usize, rt: usize) -> OpCode {
    SLT { rd, rs, rt }.into_opcode()
}

impl Display for SLT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "slt ${} ${} ${}",
            REG_STR[self.rd], REG_STR[self.rs], REG_STR[self.rt]
        )
    }
}

impl TryFrom<OpCode> for SLT {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::SLT)?;
        Ok(SLT {
            rd: value.bits(11..16) as usize,
            rt: value.bits(16..21) as usize,
            rs: value.bits(21..26) as usize,
        })
    }
}

impl Op for SLT {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::SLT)
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

        let rs = state.emit_get_register(fn_builder, self.rs);
        let rt = state.emit_get_register(fn_builder, self.rt);
        let rd = fn_builder.ins().icmp(IntCC::SignedLessThan, rs, rt);
        let rd = fn_builder.ins().uextend(types::I32, rd);
        Some(
            EmitSummary::builder()
                .register_updates(vec![(self.rd, rd)].into_boxed_slice())
                .build(fn_builder),
        )
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::dynarec::JitSummary;
    use crate::cpu::ops::prelude::*;
    use crate::memory::KSEG0Addr;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    fn basic_slt(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator.mem.write_array(
            KSEG0Addr::from_phys(0),
            &[
                addiu(8, 0, 16),
                addiu(9, 0, -3),
                slt(10, 9, 8),
                OpCode(69420),
            ],
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 1);

        Ok(())
    }
}

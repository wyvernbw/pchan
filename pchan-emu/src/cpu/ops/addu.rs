use std::fmt::Display;

use crate::cpu::{
    REG_STR,
    ops::{BoundaryType, EmitParams, EmitSummary, Op, OpCode, SecOp, TryFromOpcodeErr},
};

use super::PrimeOp;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct ADDU {
    rd: usize,
    rs: usize,
    rt: usize,
}

impl ADDU {
    pub fn try_from_opcode(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_secondary(SecOp::ADDU)?;
        Ok(ADDU {
            rs: opcode.bits(21..26) as usize,
            rt: opcode.bits(16..21) as usize,
            rd: opcode.bits(11..16) as usize,
        })
    }
}

impl Display for ADDU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "addu ${} ${} ${}",
            REG_STR[self.rd], REG_STR[self.rs], REG_STR[self.rt]
        )
    }
}

impl Op for ADDU {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::ADDU)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(&self, mut state: EmitParams) -> Option<EmitSummary> {
        use crate::cranelift_bs::*;
        let rs = state.emit_get_register(self.rs);
        let rt = state.emit_get_register(self.rt);
        let rd = state.fn_builder.ins().iadd(rs, rt);
        Some(
            EmitSummary::builder()
                .register_updates(vec![(self.rd, rd)].into())
                .build(),
        )
    }
}

#[inline]
pub fn addu(rd: usize, rs: usize, rt: usize) -> OpCode {
    ADDU { rd, rs, rt }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, cpu::ops::OpCode, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    fn addu_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;
        let program = [addu(10, 8, 9), OpCode(69420)];
        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc as u32), program);
        emulator.cpu.gpr[8] = 32;
        emulator.cpu.gpr[9] = 64;
        emulator.step_jit()?;
        assert_eq!(
            emulator.cpu.gpr[10],
            emulator.cpu.gpr[8] + emulator.cpu.gpr[9]
        );
        Ok(())
    }
    #[rstest]
    fn addu_2(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;
        let program = [addu(10, 8, 9), OpCode(69420)];
        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc as u32), program);
        emulator.cpu.gpr[8] = u32::MAX as u64;
        emulator.cpu.gpr[9] = 1;
        emulator.step_jit()?;
        assert_eq!(
            emulator.cpu.gpr[10],
            emulator.cpu.gpr[8] + emulator.cpu.gpr[9]
        );
        Ok(())
    }
}

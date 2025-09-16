use std::fmt::Display;

use crate::FnBuilderExt;
use crate::dynarec::prelude::*;

use super::PrimeOp;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct ADDU {
    rd: usize,
    rs: usize,
    rt: usize,
}

impl TryFrom<OpCode> for ADDU {
    type Error = TryFromOpcodeErr;
    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
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

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        use crate::cranelift_bs::*;
        // case 1: x + 0 = x
        if self.rs == 0 {
            let (rt, loadreg) = state.emit_get_register(self.rt);
            return EmitSummary::builder()
                .instructions([now(loadreg)])
                .register_updates([(self.rd, rt)])
                .build(state.fn_builder);
        }
        // case 2: 0 + x = x
        if self.rt == 0 {
            let (rs, loadreg) = state.emit_get_register(self.rs);
            return EmitSummary::builder()
                .instructions([now(loadreg)])
                .register_updates([(self.rd, rs)])
                .build(state.fn_builder);
        }
        let (rs, load0) = state.emit_get_register(self.rs);
        let (rt, load1) = state.emit_get_register(self.rt);
        let (rd, iadd) = state
            .fn_builder
            .inst(|f| f.ins().Binary(Opcode::Iadd, types::I32, rs, rt).0);
        EmitSummary::builder()
            .instructions([now(load0), now(load1), now(iadd)])
            .register_updates([(self.rd, rd)])
            .build(state.fn_builder)
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

    use crate::{Emu, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    fn addu_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;
        let program = [addu(10, 8, 9), OpCode(69420)];
        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc), program);
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
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc), program);
        emulator.cpu.gpr[8] = u32::MAX;
        emulator.cpu.gpr[9] = 1;
        emulator.step_jit()?;
        assert_eq!(emulator.cpu.gpr[10], u32::MAX.wrapping_add(1));
        Ok(())
    }
}

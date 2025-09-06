use crate::cpu::ops::{BoundaryType, EmitParams, EmitSummary, Op, OpCode, SecOp, TryFromOpcodeErr};

use super::PrimeOp;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct ADDIU {
    rs: usize,
    rt: usize,
    imm: i16,
}

impl ADDIU {
    pub(crate) fn try_from_opcode(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::ADDIU)?;
        Ok(ADDIU {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Op for ADDIU {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::ADDIU)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }

    fn emit_ir(&self, mut state: EmitParams<'_, '_>) -> Option<EmitSummary> {
        use crate::cranelift_bs::*;
        let rs = state.emit_get_register(self.rs);
        let rt = state.fn_builder.ins().iadd_imm(rs, self.imm as i64);
        Some(
            EmitSummary::builder()
                .register_updates(vec![(self.rt, rt)].into())
                .build(),
        )
    }
}

#[inline]
pub(crate) fn addiu(rt: usize, rs: usize, imm: i16) -> OpCode {
    ADDIU { rt, rs, imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, cpu::ops::OpCode, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    fn addiu_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;
        let program = [addiu(9, 8, -16), addiu(10, 0, 8), OpCode(69420)];
        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc as u32), program);
        emulator.cpu.gpr[8] = 32;
        emulator.advance_jit()?;
        assert_eq!(emulator.cpu.gpr[9], emulator.cpu.gpr[8] - 16);
        assert_eq!(emulator.cpu.gpr[10], 8);
        Ok(())
    }
}

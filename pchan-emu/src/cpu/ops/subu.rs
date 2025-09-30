use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub struct SUBU {
    rd: u8,
    rs: u8,
    rt: u8,
}

impl SUBU {
    pub const fn new(rd: u8, rs: u8, rt: u8) -> Self {
        Self { rd, rs, rt }
    }
}

impl TryFrom<OpCode> for SUBU {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_secondary(SecOp::SUBU)?;
        Ok(SUBU {
            rs: opcode.bits(21..26) as u8,
            rt: opcode.bits(16..21) as u8,
            rd: opcode.bits(11..16) as u8,
        })
    }
}

impl Display for SUBU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "subu ${} ${} ${}",
            REG_STR[self.rd as usize], REG_STR[self.rs as usize], REG_STR[self.rt as usize]
        )
    }
}

impl Op for SUBU {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::SUBU)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        use crate::cranelift_bs::*;
        let (rs, loadrs) = state.emit_get_register(self.rs);
        let (rt, loadrt) = state.emit_get_register(self.rt);
        let (rd, sub) = state.inst(|f| f.pure().Binary(Opcode::Isub, types::I32, rs, rt).0);

        EmitSummary::builder()
            .instructions([now(loadrs), now(loadrt), now(sub)])
            .register_updates([(self.rd, rd)])
            .build(state.fn_builder)
    }
}

#[inline]
pub fn subu(rd: u8, rs: u8, rt: u8) -> OpCode {
    SUBU { rd, rs, rt }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    fn addu_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        let program = program([subu(10, 8, 9), OpCode(69420)]);
        emulator.write_many(emulator.cpu.pc, &program);
        emulator.cpu.gpr[8] = 64;
        emulator.cpu.gpr[9] = 32;
        emulator.step_jit()?;
        assert_eq!(
            emulator.cpu.gpr[10],
            emulator.cpu.gpr[8] - emulator.cpu.gpr[9]
        );
        Ok(())
    }
    #[rstest]
    fn addu_2(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        let program = program([subu(10, 8, 9), OpCode(69420)]);
        emulator.write_many(emulator.cpu.pc, &program);
        emulator.cpu.gpr[8] = u32::MAX;
        emulator.cpu.gpr[9] = 1;
        emulator.step_jit()?;
        assert_eq!(
            emulator.cpu.gpr[10],
            emulator.cpu.gpr[8] - emulator.cpu.gpr[9]
        );
        Ok(())
    }
}

use crate::dynarec::prelude::*;
use std::fmt::Display;

use super::{EmitCtx, OpCode, PrimeOp};

#[derive(Debug, Clone, Copy)]
pub struct SB {
    rt: usize,
    rs: usize,
    imm: i16,
}

pub fn sb(rt: usize, rs: usize, imm: i16) -> OpCode {
    SB { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for SB {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::SB)?;
        Ok(SB {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Display for SB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sb ${} ${} {}",
            REG_STR[self.rt], REG_STR[self.rs], self.imm
        )
    }
}

impl Op for SB {
    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        store!(self, ctx, write8)
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SB)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::memory::ext;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    pub fn test_sb(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator
            .mem
            .write_many(0, &program([sb(9, 8, 0), OpCode(69420)]));

        emulator.cpu.gpr[8] = 32; // base register
        emulator.cpu.gpr[9] = 69;

        emulator.step_jit()?;

        assert_eq!(emulator.mem.read::<u8, ext::Sign>(32), 69);

        Ok(())
    }
}

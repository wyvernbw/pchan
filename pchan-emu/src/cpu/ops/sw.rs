use crate::dynarec::prelude::*;
use std::fmt::Display;

use super::{EmitCtx, OpCode, PrimeOp};

#[derive(Debug, Clone, Copy, Hash)]
pub struct SW {
    rt: u8,
    rs: u8,
    imm: i16,
}

impl SW {
    pub const fn new(rt: u8, rs: u8, imm: i16) -> Self {
        Self { rt, rs, imm }
    }
}

pub fn sw(rt: u8, rs: u8, imm: i16) -> OpCode {
    SW { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for SW {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::SW)?;
        Ok(SW {
            rt: opcode.bits(16..21) as u8,
            rs: opcode.bits(21..26) as u8,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Display for SW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sw ${} ${} {}",
            REG_STR[self.rt as usize],
            REG_STR[self.rs as usize],
            hex(self.imm)
        )
    }
}

impl Op for SW {
    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        store!(self, ctx, write32)
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SW)
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
    pub fn test_sw(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator.write_many(0x0, &program([sw(9, 8, 0), lw(10, 8, 0), OpCode(69420)]));

        emulator.cpu.gpr[8] = 32; // base register
        emulator.cpu.gpr[9] = u32::MAX;

        emulator.step_jit()?;

        assert_eq!(emulator.read::<u32, ext::NoExt>(32), u32::MAX);

        Ok(())
    }
}

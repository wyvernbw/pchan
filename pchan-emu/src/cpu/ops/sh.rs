use std::fmt::Display;

use pchan_utils::hex;

use crate::cpu::REG_STR;
use crate::cpu::ops::{self, BoundaryType, EmitSummary, Op, TryFromOpcodeErr};
use crate::{cranelift_bs::*, store};

use super::{EmitCtx, OpCode, PrimeOp};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SH {
    pub rt: u8,
    pub rs: u8,
    pub imm: i16,
}

impl SH {
    pub const fn new(rt: u8, rs: u8, imm: i16) -> Self {
        Self { rt, rs, imm }
    }
}

pub fn sh(rt: u8, rs: u8, imm: i16) -> ops::OpCode {
    SH { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for SH {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::SH)?;
        Ok(SH {
            rt: opcode.bits(16..21) as u8,
            rs: opcode.bits(21..26) as u8,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Display for SH {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sh ${} ${} {}",
            REG_STR[self.rt as usize],
            REG_STR[self.rs as usize],
            hex(self.imm)
        )
    }
}

impl Op for SH {
    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        store!(self, ctx, write16)
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> ops::OpCode {
        ops::OpCode::default()
            .with_primary(PrimeOp::SH)
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
    use crate::jit::JIT;
    use crate::memory::ext;
    use crate::test_utils::jit;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    pub fn test_sh(setup_tracing: (), mut emulator: Emu, mut jit: JIT) -> color_eyre::Result<()> {
        emulator.write_many(0, &program([sh(9, 8, 0), OpCode(69420)]));

        emulator.cpu.gpr[8] = 32; // base register
        emulator.cpu.gpr[9] = 690;

        emulator.step_jit(&mut jit)?;

        assert_eq!(emulator.read::<u16, ext::Sign>(32), 690);

        Ok(())
    }
}

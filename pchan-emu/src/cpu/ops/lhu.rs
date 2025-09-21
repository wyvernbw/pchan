use std::fmt::Display;

use crate::cpu::REG_STR;
use crate::cpu::ops::{self, BoundaryType, EmitSummary, Op, OpCode, TryFromOpcodeErr};
use crate::{cranelift_bs::*, load};

use super::{EmitCtx, PrimeOp};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct LHU {
    rt: u8,
    rs: u8,
    imm: i16,
}

impl LHU {
    pub const fn new(rt: u8, rs: u8, imm: i16) -> Self {
        Self { rt, rs, imm }
    }
}

#[inline]
pub fn lhu(rt: u8, rs: u8, imm: i16) -> ops::OpCode {
    LHU { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for LHU {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: ops::OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::LHU)?;
        Ok(LHU {
            rt: opcode.bits(16..21) as u8,
            rs: opcode.bits(21..26) as u8,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Display for LHU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lhu ${} ${} {}",
            REG_STR[self.rt as usize], REG_STR[self.rs as usize], self.imm
        )
    }
}

impl Op for LHU {
    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        load!(self, ctx, readu16)
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> ops::OpCode {
        ops::OpCode::default()
            .with_primary(PrimeOp::LHU)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::{Emu, memory::ext, test_utils::emulator};

    #[rstest]
    pub fn test_lhu(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        emulator
            .mem
            .write_many(0x0, &program([lhu(8, 9, 4), nop(), OpCode(69420)]));

        let op = emulator.mem.read::<OpCode, ext::NoExt>(0);
        tracing::debug!(decoded = ?DecodedOp::try_from(op));
        tracing::debug!("{:08X?}", &emulator.mem.as_ref()[..22]);

        emulator.cpu.gpr[9] = 16;

        emulator.mem.write::<u16>(20, 0xABCD);

        emulator.step_jit()?;

        assert_eq!(emulator.cpu.gpr[8], 0xABCDu32);

        Ok(())
    }
}

use crate::dynarec::prelude::*;
use std::fmt::Display;

use super::{OpCode, PrimeOp};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct LH {
    rt: u8,
    rs: u8,
    imm: i16,
}

impl LH {
    pub const fn new(rt: u8, rs: u8, imm: i16) -> Self {
        Self { rt, rs, imm }
    }
}

pub fn lh(rt: u8, rs: u8, imm: i16) -> OpCode {
    LH { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for LH {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::LH)?;
        Ok(LH {
            rt: opcode.bits(16..21) as u8,
            rs: opcode.bits(21..26) as u8,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Display for LH {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lh ${} ${} {}",
            REG_STR[self.rt as usize],
            REG_STR[self.rs as usize],
            hex(self.imm)
        )
    }
}

impl Op for LH {
    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        load!(self, ctx, readi16)
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::LH)
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
    use crate::{
        Emu,
        jit::JIT,
        test_utils::{emulator, jit},
    };

    #[rstest]
    pub fn test_lh_sign_extension(
        setup_tracing: (),
        mut emulator: Emu,
        mut jit: JIT,
    ) -> color_eyre::Result<()> {
        emulator.write::<u16>(32, 0x8000); // -32768
        emulator.write::<u16>(34, 0x7FFF); // +32767

        emulator.write_many(
            0x0,
            &program([lh(8, 9, 0), lh(10, 9, 2), nop(), OpCode(69420)]),
        );

        emulator.cpu.gpr[9] = 32; // base register

        // Run the block
        emulator.step_jit(&mut jit)?;

        assert_eq!(emulator.cpu.gpr[8], 0xFFFF_8000u32);

        assert_eq!(emulator.cpu.gpr[10], 0x7FFF);

        Ok(())
    }
}

use std::fmt::Display;

use crate::dynarec::prelude::*;

use super::{EmitCtx, OpCode, PrimeOp};
use crate::load;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct LB {
    rt: u8,
    rs: u8,
    imm: i16,
}

impl LB {
    pub const fn new(rt: u8, rs: u8, imm: i16) -> Self {
        Self { rt, rs, imm }
    }
}

pub fn lb(rt: u8, rs: u8, imm: i16) -> OpCode {
    LB { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for LB {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::LB)?;
        Ok(LB {
            rt: opcode.bits(16..21) as u8,
            rs: opcode.bits(21..26) as u8,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Op for LB {
    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        load!(self, ctx, readi8)
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::LB)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }
}

impl Display for LB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lb ${} ${} {}",
            REG_STR[self.rt as usize],
            REG_STR[self.rs as usize],
            hex(self.imm)
        )
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::jit::JIT;
    use crate::test_utils::jit;
    use crate::{Emu, memory::ext, test_utils::emulator};

    #[rstest]
    pub fn test_lb_sign_extension(
        setup_tracing: (),
        mut emulator: Emu,
        mut jit: JIT,
    ) -> color_eyre::Result<()> {
        emulator.write(32, 0xFFu8);
        emulator.write(33, 0x7Fu8);
        emulator.write_many(
            0x0,
            &program([lb(8, 9, 0), lb(10, 9, 1), nop(), OpCode(69420)]),
        );

        emulator.cpu.gpr[9] = 32; // base register

        // Run the block
        emulator.step_jit(&mut jit)?;

        assert_eq!(emulator.cpu.gpr[8], ext::sign::<u8>(0xFFu8) as u32);

        // 0x7F should be sign-extended to 0x7F
        assert_eq!(emulator.cpu.gpr[10], ext::sign::<u8>(0x7Fu8) as u32);

        Ok(())
    }
}

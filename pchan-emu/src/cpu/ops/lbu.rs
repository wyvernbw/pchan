use std::fmt::Display;

use cranelift::codegen::ir::immediates::Offset32;

use crate::cpu::REG_STR;
use crate::cpu::ops::{self, BoundaryType, EmitSummary, Op, OpCode, TryFromOpcodeErr};
use crate::{cranelift_bs::*, load};

use super::{EmitCtx, PrimeOp};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct LBU {
    rt: usize,
    rs: usize,
    imm: i16,
}

#[inline]
pub fn lbu(rt: usize, rs: usize, imm: i16) -> ops::OpCode {
    LBU { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for LBU {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: ops::OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::LBU)?;
        Ok(LBU {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Op for LBU {
    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        load!(self, ctx, Opcode::Uload8)
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> ops::OpCode {
        ops::OpCode::default()
            .with_primary(PrimeOp::LBU)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }
}

impl Display for LBU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lbu ${} ${} {}",
            REG_STR[self.rt], REG_STR[self.rs], self.imm
        )
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::{
        Emu,
        cpu::ops::{self, DecodedOp, lbu::lbu, nop},
        memory::{KSEG0Addr, PhysAddr},
        test_utils::emulator,
    };

    #[rstest]
    pub fn test_lbu(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator.mem.write_all(
            KSEG0Addr::from_phys(0),
            [lbu(8, 9, 4), nop(), ops::OpCode(69420)],
        );
        let op = emulator.mem.read::<ops::OpCode>(PhysAddr(0));
        tracing::debug!(decoded = ?DecodedOp::try_from(op));
        tracing::debug!("{:08X?}", &emulator.mem.as_ref()[..21]);
        emulator.cpu.gpr[9] = 16;
        emulator.mem.as_mut()[20] = 69;
        emulator.step_jit()?;
        assert_eq!(emulator.cpu.gpr[8], emulator.mem.as_ref()[20] as u32);
        Ok(())
    }
}

use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub struct LBU {
    rt: u8,
    rs: u8,
    imm: i16,
}

impl LBU {
    pub const fn new(rt: u8, rs: u8, imm: i16) -> Self {
        Self { rt, rs, imm }
    }
}

#[inline]
pub fn lbu(rt: u8, rs: u8, imm: i16) -> OpCode {
    LBU { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for LBU {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::LBU)?;
        Ok(LBU {
            rt: opcode.bits(16..21) as u8,
            rs: opcode.bits(21..26) as u8,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Op for LBU {
    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        load!(self, ctx, readu8)
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
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

    use crate::Emu;
    use crate::dynarec::prelude::*;
    use crate::jit::JIT;
    use crate::memory::ext;
    use crate::test_utils::{emulator, jit};

    #[rstest]
    pub fn test_lbu(setup_tracing: (), mut emulator: Emu, mut jit: JIT) -> color_eyre::Result<()> {
        emulator.write_many(0x0, &program([lbu(8, 9, 4), nop(), OpCode(69420)]));
        let op = emulator.read::<OpCode, ext::NoExt>(0x0);
        tracing::debug!(decoded = ?DecodedOp::new(op));
        tracing::debug!("{:08X?}", &emulator.mem.buf.as_ref()[..21]);
        emulator.cpu.gpr[9] = 16;
        emulator.mem.buf.as_mut()[20] = 69;
        emulator.step_jit(&mut jit)?;
        assert_eq!(emulator.cpu.gpr[8], emulator.mem.buf.as_ref()[20] as u32);
        Ok(())
    }
}

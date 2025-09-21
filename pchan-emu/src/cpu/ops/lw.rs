use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy)]
pub struct LW {
    rt: u8,
    rs: u8,
    imm: i16,
}

impl LW {
    pub const fn new(rt: u8, rs: u8, imm: i16) -> Self {
        Self { rt, rs, imm }
    }
}

pub fn lw(rt: u8, rs: u8, imm: i16) -> OpCode {
    LW { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for LW {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::LW)?;
        Ok(LW {
            rt: opcode.bits(16..21) as u8,
            rs: opcode.bits(21..26) as u8,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Display for LW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lw ${} ${} {}",
            REG_STR[self.rt as usize], REG_STR[self.rs as usize], self.imm
        )
    }
}

impl Op for LW {
    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        load!(self, ctx, read32)
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::LW)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }
}

#[cfg(test)]
mod tests {
    use crate::dynarec::prelude::*;
    use crate::memory::ext;
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use pretty_assertions::assert_ne;
    use rstest::rstest;

    use crate::{
        Emu,
        cpu::ops::{self},
        dynarec::JitSummary,
        test_utils::emulator,
    };

    #[rstest]
    pub fn test_lw(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator.mem.write::<u32>(32, 0xDEAD_BEEF); // -32768

        emulator
            .mem
            .write_many(0, &program([lw(8, 9, 0), nop(), ops::OpCode(69420)]));

        emulator.cpu.gpr[9] = 32; // base register

        // Run the block
        emulator.step_jit()?;

        assert_eq!(emulator.cpu.gpr[8], 0xDEAD_BEEF);

        Ok(())
    }

    #[rstest]
    pub fn load_delay_hazard_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator.mem.write::<u32>(32, 0xDEAD_BEEF);

        emulator.mem.write_many(
            0,
            &program([
                addiu(8, 0, 0),
                lw(8, 9, 0),
                addiu(8, 8, 12),
                ops::OpCode(69420),
            ]),
        );

        emulator.cpu.gpr[9] = 32; // base register

        // Run the block
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);

        // $t0 is replaced with 0xDEAD_BEEF, even tho load happened before
        assert_eq!(emulator.cpu.gpr[8], 0xDEAD_BEEF);

        Ok(())
    }

    #[rstest]
    pub fn load_delay_hazard_2(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator.mem.write::<u32>(32, 0xDEAD_BEEF);

        emulator.mem.write_many(
            0,
            &program([lw(8, 9, 0), nop(), sb(8, 10, 0), ops::OpCode(69420)]),
        );

        emulator.cpu.gpr[9] = 32; // base register
        emulator.cpu.gpr[10] = 36;

        // Run the block
        emulator.step_jit()?;

        assert_eq!(emulator.cpu.gpr[8], 0xDEAD_BEEF);
        assert_ne!(emulator.mem.read::<u32, ext::NoExt>(36), 0);

        Ok(())
    }
}

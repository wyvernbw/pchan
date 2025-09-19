use std::fmt::Display;

use crate::dynarec::prelude::*;
use tracing::instrument;

use crate::cpu::ops::{
    BoundaryType, EmitCtx, EmitSummary, MipsOffset, Op, OpCode, PrimeOp, TryFromOpcodeErr,
};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct J {
    pub imm: i32,
}

impl TryFrom<OpCode> for J {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::J)?;
        Ok(J {
            imm: (opcode.bits(0..26) as i16 as i32) << 2,
        })
    }
}

impl Display for J {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "j 0x{:08X}", self.imm)
    }
}

impl Op for J {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::Block {
            offset: MipsOffset::Relative(self.imm),
        })
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::J)
            .set_bits(0..26, self.imm as u32 >> 2)
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    #[instrument("j", skip_all, fields(node = ?ctx.node, block = ?ctx.block().clif_block()))]
    fn emit_ir(&self, ctx: EmitCtx) -> EmitSummary {
        EmitSummary::builder()
            .instructions([terminator(bomb(
                1,
                lazy(|mut ctx: EmitCtx| {
                    debug_assert_eq!(ctx.neighbour_count(), 1);
                    let (_, block_call) = ctx.block_call(ctx.next_at(0));

                    ctx.fn_builder
                        .pure()
                        .Jump(Opcode::Jump, types::INVALID, block_call)
                        .0
                }),
            ))])
            .pc_update(MipsOffset::Relative(self.imm).calculate_address(ctx.pc))
            .build(ctx.fn_builder)
    }
}

#[inline]
pub fn j(imm: i32) -> OpCode {
    J { imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    fn basic_jump(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let main = program([addiu(8, 0, 32), j(0x0000_2000 - 4), nop()]);

        let function = program([addiu(9, 0, 69), nop(), OpCode(69420)]);

        emulator.mem.write_many(emulator.cpu.pc, &main);
        emulator.mem.write_many(0x0000_2000, &function);

        emulator.step_jit()?;
        assert_eq!(emulator.cpu.gpr[9], 69);

        Ok(())
    }
    #[rstest]
    fn jump_delay_hazard_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let main = program([addiu(8, 0, 32), j(0x0000_2000 - 4), addiu(10, 0, 42)]);

        let function = program([addiu(9, 0, 69), nop(), OpCode(69420)]);

        emulator.mem.write_many(emulator.cpu.pc, &main);
        emulator.mem.write_many(0x0000_2000, &function);

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);

        assert_eq!(emulator.cpu.gpr[9], 69);
        assert_eq!(emulator.cpu.gpr[10], 42);

        Ok(())
    }
}

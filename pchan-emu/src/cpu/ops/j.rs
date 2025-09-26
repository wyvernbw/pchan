use std::fmt::Display;

use crate::dynarec::prelude::*;
use tracing::instrument;

use crate::cpu::ops::{
    BoundaryType, EmitCtx, EmitSummary, MipsOffset, Op, OpCode, PrimeOp, TryFromOpcodeErr,
};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct J {
    pub imm: u32,
}

impl J {
    pub const fn new(imm: u32) -> Self {
        Self { imm }
    }
}

impl Display for J {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "j {}", hex(self.imm << 2))
    }
}

impl Op for J {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::Block {
            offset: MipsOffset::RegionJump(self.imm << 2),
        })
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::J)
            .set_bits(0..26, self.imm)
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    #[instrument("j", skip_all, fields(node = ?ctx.node, block = ?ctx.block().clif_block()))]
    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        let jump_address = MipsOffset::RegionJump(self.imm << 2).calculate_address(ctx.pc);

        let [create_new_pc, store_new_pc] = ctx.emit_store_pc_imm(jump_address);

        let cached_call = ctx.try_fn_call(jump_address);
        if let Some([create_address, call]) = cached_call {
            let ret = ctx.fn_builder.pure().return_(&[]);
            return EmitSummary::builder()
                .instructions([
                    now(create_new_pc),
                    now(store_new_pc),
                    now(create_address),
                    delayed(1, call),
                    terminator(bomb(1, ret)),
                ])
                .build(ctx.fn_builder);
        };

        EmitSummary::builder()
            .instructions([
                now(create_new_pc),
                now(store_new_pc),
                terminator(bomb(
                    1,
                    lazy(move |mut ctx: EmitCtx| {
                        debug_assert_eq!(ctx.neighbour_count(), 1);
                        let (_, block_call) = ctx.block_call(ctx.next_at(0));

                        ctx.fn_builder
                            .pure()
                            .Jump(Opcode::Jump, types::INVALID, block_call)
                            .0
                    }),
                )),
            ])
            .build(ctx.fn_builder)
    }
}

#[inline]
pub fn j(imm: u32) -> OpCode {
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

        let main = program([addiu(8, 0, 32), j((0x0000_2000 - 4) >> 2), nop()]);

        let function = program([addiu(9, 0, 69), nop(), OpCode(69420)]);

        emulator.write_many(emulator.cpu.pc, &main);
        emulator.write_many(0x0000_2000, &function);

        emulator.step_jit()?;
        assert_eq!(emulator.cpu.gpr[9], 69);

        Ok(())
    }
    #[rstest]
    fn jump_delay_hazard_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let main = program([addiu(8, 0, 32), j((0x0000_2000 - 4) >> 2), addiu(10, 0, 42)]);

        let function = program([addiu(9, 0, 69), nop(), OpCode(69420)]);

        emulator.write_many(emulator.cpu.pc, &main);
        emulator.write_many(0x0000_2000, &function);

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);

        assert_eq!(emulator.cpu.gpr[9], 69);
        assert_eq!(emulator.cpu.gpr[10], 42);

        Ok(())
    }
}

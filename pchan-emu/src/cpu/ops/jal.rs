use std::fmt::Display;

use crate::{cpu::RA, dynarec::prelude::*};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct JAL {
    pub imm: u32,
}

impl TryFrom<OpCode> for JAL {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::JAL)?;
        Ok(JAL {
            imm: opcode.bits(0..26) << 2,
        })
    }
}

impl Display for JAL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "jal 0x{:08X}", self.imm)
    }
}

impl Op for JAL {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::Block {
            offset: MipsOffset::RegionJump(self.imm),
        })
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::JAL)
            .set_bits(0..26, self.imm >> 2)
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        tracing::info!("jal: saving pc 0x{:08X}", ctx.pc);
        debug_assert_eq!(ctx.neighbour_count(), 1);
        let pc = ctx.pc as i64;
        let (pc, iconst) = ctx.inst(|f| {
            f.pure()
                .UnaryImm(Opcode::Iconst, types::I32, Imm64::new(pc + 8))
                .0
        });
        ctx.update_cache_immediate(RA, pc);

        EmitSummary::builder()
            .instructions([
                now(iconst),
                terminator(bomb(
                    1,
                    lazy(|mut ctx| {
                        let (params, block_call) = ctx.block_call(ctx.next_at(0));

                        tracing::debug!("jumping with {} dependencies", params.len());

                        ctx.fn_builder
                            .pure()
                            .Jump(Opcode::Jump, types::INVALID, block_call)
                            .0
                    }),
                )),
            ])
            .pc_update(MipsOffset::RegionJump(self.imm).calculate_address(ctx.pc))
            .build(ctx.fn_builder)
    }
}

#[inline]
pub fn jal(imm: u32) -> OpCode {
    JAL { imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, cpu::RA, dynarec::JitSummary, test_utils::emulator};

    #[rstest]
    fn jal_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let main = program([
            addiu(8, 0, 32),
            jal(0x0000_2000), // 4
            nop(),            // 8
            nop(),            // 12
        ]);

        let function = program([
            addiu(9, 0, 69),
            nop(),
            // load return address into $t2
            addiu(10, RA, 0),
            OpCode(69420),
        ]);

        emulator.mem.write_many(emulator.cpu.pc, &main);
        emulator.mem.write_many(0x0000_2000, &function);

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 12);

        Ok(())
    }
}

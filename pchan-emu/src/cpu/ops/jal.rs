use std::fmt::Display;

use crate::{cpu::RA, dynarec::prelude::*};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct JAL {
    pub imm: u32,
}

impl JAL {
    pub const fn new(imm: u32) -> Self {
        Self { imm }
    }
}

impl TryFrom<OpCode> for JAL {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::JAL)?;
        Ok(JAL {
            imm: opcode.bits(0..26),
        })
    }
}

impl Display for JAL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "jal {}", hex(self.imm << 2))
    }
}

impl Op for JAL {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::Block {
            offset: MipsOffset::RegionJump(self.imm << 2),
        })
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::JAL)
            .set_bits(0..26, self.imm)
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        // tracing::info!("jal: saving pc 0x{:08X}", ctx.pc);
        let jump_address = MipsOffset::RegionJump(self.imm << 2).calculate_address(ctx.pc);
        debug_assert_eq!(ctx.neighbour_count(), 1);

        let pc = ctx.pc as i64;
        let (new_ra, create_ra) = ctx.inst(|f| {
            f.pure()
                .UnaryImm(Opcode::Iconst, types::I32, Imm64::new(pc + 8))
                .0
        });
        let [create_new_pc, store_new_pc] = ctx.emit_store_pc_imm(jump_address);

        let cached_call = ctx.try_fn_call(jump_address);
        if let Some([create_address, call]) = cached_call {
            let ret = ctx.fn_builder.pure().return_(&[]);
            return EmitSummary::builder()
                .register_updates([(RA, new_ra)])
                .instructions([
                    now(create_ra),
                    now(create_new_pc),
                    now(store_new_pc),
                    now(create_address),
                    delayed(1, call),
                    terminator(bomb(1, ret)),
                ])
                .build(ctx.fn_builder);
        };

        EmitSummary::builder()
            .register_updates([(RA, new_ra)])
            .instructions([
                now(create_ra),
                now(create_new_pc),
                now(store_new_pc),
                terminator(bomb(
                    1,
                    lazy(|mut ctx| {
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
pub fn jal(imm: u32) -> OpCode {
    JAL { imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{
        Emu,
        cpu::{RA, SP},
        dynarec::JitSummary,
        test_utils::emulator,
    };

    #[rstest]
    fn jal_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let main = program([
            addiu(8, 0, 32),
            jal(0x0000_2000 >> 2), // 4
            nop(),                 // 8
            nop(),                 // 12
        ]);

        let function = program([
            addiu(9, 0, 69),
            nop(),
            // load return address into $t2
            addiu(10, RA, 0),
            OpCode(69420),
        ]);

        emulator.write_many(emulator.cpu.pc, &main);
        emulator.write_many(0x0000_2000, &function);

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 12);

        Ok(())
    }

    #[rstest]
    fn jal_2(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::dynarec::prelude::*;

        let main = program([
            addiu(8, 0, 4), // counter = 4
            jal(0x00002000 >> 2),
            nop(),
            OpCode(69420),
        ]);

        let recursive_fn = program([
            addiu(SP, SP, -24), // allocate stack
            sw(RA, SP, 20),     // save return address
            sw(8, SP, 16),      // save counter
            addiu(8, 8, -1),    // counter--
            blez(8, 0x2),
            nop(),
            jal(0x00002000 >> 2), // recursive call
            nop(),
            // base_case:
            lw(8, SP, 16),     // restore counter
            lw(RA, SP, 20),    // restore return address
            addiu(SP, SP, 24), // deallocate stack
            jr(RA),            // return
        ]);

        emulator.cpu.gpr[SP as usize] = 0x801ffce0;
        emulator.write_many(emulator.cpu.pc, &main);
        emulator.write_many(0x00002000, &recursive_fn);

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);

        Ok(())
    }
}

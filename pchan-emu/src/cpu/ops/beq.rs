use std::fmt::Display;

use tracing::instrument;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub struct BEQ {
    rs: u8,
    rt: u8,
    imm: i16,
}

impl BEQ {
    pub const fn new(rs: u8, rt: u8, imm: i16) -> Self {
        Self { rs, rt, imm }
    }
}

#[inline]
pub fn beq(rs: u8, rt: u8, dest: i16) -> OpCode {
    BEQ { rs, rt, imm: dest }.into_opcode()
}

impl TryFrom<OpCode> for BEQ {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::BEQ)?;
        Ok(BEQ {
            rs: (opcode.bits(21..26)) as u8,
            rt: (opcode.bits(16..21)) as u8,
            imm: (opcode.bits(0..16) as i16),
        })
    }
}

impl Display for BEQ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "beq ${} ${} {}",
            REG_STR[self.rs as usize],
            REG_STR[self.rt as usize],
            hex(self.imm * 4 + 4)
        )
    }
}

impl Op for BEQ {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::BlockSplit {
            lhs: MipsOffset::Relative(self.imm as i32 * 4 + 4),
            rhs: MipsOffset::Relative(4),
        })
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
            .with_primary(PrimeOp::BEQ)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(0..16, (self.imm) as u32)
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    #[instrument("beq", skip_all, fields(node = ?ctx.node, block = ?ctx.block().clif_block()))]
    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        use crate::cranelift_bs::*;

        let (rs, load0) = ctx.emit_get_register(self.rs);
        let (rt, load1) = ctx.emit_get_register(self.rt);

        let (cond, icmp) = ctx.inst(|f| {
            f.pure()
                .IntCompare(Opcode::Icmp, types::I32, IntCC::Equal, rs, rt)
                .0
        });

        EmitSummary::builder()
            .instructions([
                now(load0),
                now(load1),
                now(icmp),
                terminator(bomb(
                    1,
                    lazy_boxed(move |ctx| {
                        let then_block = ctx.next_at(0);
                        let then_block_label = ctx.cfg[then_block].clif_block();
                        let then_params = ctx.out_params(then_block);
                        let then_block_call = ctx
                            .fn_builder
                            .pure()
                            .data_flow_graph_mut()
                            .block_call(then_block_label, &then_params);

                        let else_block = ctx.next_at(1);
                        let else_block_label = ctx.cfg[else_block].clif_block();
                        let else_params = ctx.out_params(else_block);
                        let else_block_call = ctx
                            .fn_builder
                            .pure()
                            .data_flow_graph_mut()
                            .block_call(else_block_label, &else_params);

                        ctx.fn_builder
                            .pure()
                            .Brif(
                                Opcode::Brif,
                                types::INVALID,
                                then_block_call,
                                else_block_call,
                                cond,
                            )
                            .0
                    }),
                )),
            ])
            .build(ctx.fn_builder)
    }
}

#[cfg(test)]
mod tests {
    use crate::{dynarec::prelude::*, jit::JIT, test_utils::jit};
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, test_utils::emulator};

    #[rstest]
    fn beq_basic_loop(
        setup_tracing: (),
        mut emulator: Emu,
        mut jit: JIT,
    ) -> color_eyre::Result<()> {
        let func = program([
            addiu(8, 0, 0),           // ;  0 $t0 = 0
            addiu(10, 0, 4),          // ;  4 $t2 = 4
            addiu(9, 8, 0x0000_2000), // ;  8 calculate address $t1 = $t0 + 0x0000_2000
            sb(8, 9, 0),              // ; 12 store $i at $t1
            beq(8, 10, 4),            // ; 16 if $t0=$t2(4) jump by 16+8 to reach 40
            nop(),                    // ; 20
            addiu(8, 8, 1),           // ; 24 $t0 = $t0 + 1
            nop(),                    // ; 28
            j(2),                     // ; 32 jump to 8 (return to beginning of loop)
            nop(),                    // ; 36
            nop(),                    // ; 40
            OpCode(69420),            // ; 44 halt
        ]);

        emulator.write_many(0x0, &func);

        let summary = emulator.step_jit_summarize::<JitSummary>(&mut jit)?;
        if let Some(func) = summary.function {
            tracing::info!(%func);
        }

        let slice = &emulator.mem.buf.as_ref()[0x0000_2000..(0x000_2000 + 4)];
        assert_eq!(slice, &[0, 1, 2, 3]);
        assert_eq!(emulator.cpu.pc, 48);

        Ok(())
    }
    #[rstest]
    fn beq_taken(setup_tracing: (), mut emulator: Emu, mut jit: JIT) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let func = program([
            addiu(8, 0, 5),   // $t0 = 5
            addiu(9, 0, 5),   // $t1 = 5
            beq(8, 9, 2),     // $t0 == $t1, branch taken
            addiu(10, 0, 42), // skipped
            sb(10, 0, 0x40),  // skipped
            addiu(11, 0, 99), // executed after branch target
            sb(11, 0, 0x41),  // store 99 at memory[0x41]
            OpCode(69420),    // halt
        ]);

        emulator.write_many(emulator.cpu.pc, &func);

        emulator.step_jit(&mut jit)?;

        let slice = &emulator.mem.buf.as_ref()[0x41..0x42];
        assert_eq!(slice, &[99]);

        Ok(())
    }

    #[rstest]
    fn beq_not_taken(setup_tracing: (), mut emulator: Emu, mut jit: JIT) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let func = program([
            addiu(8, 0, 1),   // $t0 = 1
            addiu(9, 0, 2),   // $t1 = 2
            beq(8, 9, 2),     // $t0 != $t1, branch not taken
            addiu(10, 0, 42), // executed because branch not taken
            sb(10, 0, 0x30),  // store 42 at memory[0x30]
            OpCode(69420),    // halt
        ]);

        emulator.write_many(emulator.cpu.pc, &func);

        emulator.step_jit(&mut jit)?;

        let slice = &emulator.mem.buf.as_ref()[0x30..0x31];
        assert_eq!(slice, &[42]);

        Ok(())
    }
}

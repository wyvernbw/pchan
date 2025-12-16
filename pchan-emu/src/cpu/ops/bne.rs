use std::fmt::Display;

use tracing::instrument;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub struct BNE {
    pub rs: u8,
    pub rt: u8,
    pub imm: i16,
}

impl BNE {
    pub const fn new(rs: u8, rt: u8, imm: i16) -> Self {
        Self { rs, rt, imm }
    }
}

#[inline]
pub fn bne(rs: u8, rt: u8, dest: i16) -> OpCode {
    BNE { rs, rt, imm: dest }.into_opcode()
}

impl TryFrom<OpCode> for BNE {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::BNE)?;
        Ok(BNE {
            rs: (opcode.bits(21..26)) as u8,
            rt: (opcode.bits(16..21)) as u8,
            imm: (opcode.bits(0..16) as i16),
        })
    }
}

impl Display for BNE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bne ${} ${} {}",
            REG_STR[self.rs as usize],
            REG_STR[self.rt as usize],
            hex(self.imm * 4 + 4)
        )
    }
}

impl Op for BNE {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::BlockSplit {
            lhs: MipsOffset::Relative(self.imm as i32 * 4 + 4),
            rhs: MipsOffset::Relative(4),
        })
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
            .with_primary(PrimeOp::BNE)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(0..16, (self.imm) as u32)
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    #[instrument("bne", skip_all, fields(node = ?ctx.node, block = ?ctx.block().clif_block()))]
    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        use crate::cranelift_bs::*;

        let (rs, load0) = ctx.emit_get_register(self.rs);
        let (rt, load1) = ctx.emit_get_register(self.rt);

        let (cond, icmp) = ctx.inst(|f| {
            f.pure()
                .IntCompare(Opcode::Icmp, types::I32, IntCC::NotEqual, rs, rt)
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

    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::test_utils::jit;
    use crate::{Emu, test_utils::emulator};

    struct Bne1Test {
        a: i16,
        b: i16,
        then: i16,
        otherwise: i16,
        expected: u32,
    }

    #[rstest]
    #[case(Bne1Test { a: 8, b: 8, then: 1, otherwise: 2, expected: 2})]
    #[case(Bne1Test { a: 8, b: 9, then: 1, otherwise: 2, expected: 1})]
    fn bne_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] test: Bne1Test,
        mut jit: crate::jit::JIT,
    ) -> color_eyre::Result<()> {
        emulator.write_many(
            0x0,
            &program([
                addiu(8, 0, test.a),
                addiu(9, 0, test.b),
                bne(8, 9, 0x4),
                nop(),
                addiu(10, 0, test.otherwise),
                OpCode(69420),
                nop(),
                addiu(10, 0, test.then),
                OpCode(69420),
            ]),
        );

        let summary = emulator.step_jit_summarize::<JitSummary>(&mut jit)?;
        tracing::info!(?summary.function);

        assert_eq!(emulator.cpu.gpr[10], test.expected);

        Ok(())
    }
}

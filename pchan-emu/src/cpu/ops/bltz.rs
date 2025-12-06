use std::fmt::Display;

use pchan_utils::hex;
use tracing::instrument;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub struct BLTZ {
    rs: u8,
    imm: i16,
}

impl BLTZ {
    pub const fn new(rs: u8, imm: i16) -> Self {
        Self { rs, imm }
    }
}

#[inline]
pub fn bltz(rs: u8, dest: i16) -> OpCode {
    BLTZ { rs, imm: dest }.into_opcode()
}

impl Display for BLTZ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bltz ${} {}", REG_STR[self.rs as usize], hex(self.imm))
    }
}

impl Op for BLTZ {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::BlockSplit {
            lhs: MipsOffset::Relative(self.imm as i32 * 4 + 4),
            rhs: MipsOffset::Relative(4),
        })
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
            .with_primary(PrimeOp::BcondZ)
            .set_bits(16..21, 0)
            .set_bits(21..26, self.rs as u32)
            .set_bits(0..16, (self.imm) as u32)
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    #[instrument("beq", skip_all, fields(node = ?ctx.node, block = ?ctx.block().clif_block()))]
    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        use crate::cranelift_bs::*;

        let (rs, load0) = ctx.emit_get_register(self.rs);

        let (cond, icmp) = ctx.inst(|f| {
            f.pure()
                .IntCompareImm(
                    Opcode::IcmpImm,
                    types::I32,
                    IntCC::SignedLessThan,
                    Imm64::new(0),
                    rs,
                )
                .0
        });

        EmitSummary::builder()
            .instructions([
                now(load0),
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

    use crate::Emu;
    use crate::dynarec::prelude::*;
    use crate::jit::JIT;
    use crate::test_utils::jit;

    #[rstest]
    #[case(0, 69)]
    #[case(i16::MAX, 69)]
    #[case(i16::MIN, 42)]
    pub fn bltz_test(
        setup_tracing: (),
        #[case] value: i16,
        #[case] expected: i16,
        mut jit: JIT,
    ) -> color_eyre::Result<()> {
        let mut emu = Emu::default();
        emu.write_many(
            0x0,
            &program([
                addiu(9, 0, value),
                bltz(9, 0x6),
                nop(),
                addiu(10, 0, 69),
                nop(),
                OpCode(69420),
                nop(),
                nop(),
                nop(),
                nop(),
                nop(),
                nop(),
                addiu(10, 0, 42),
                nop(),
                OpCode(69420),
            ]),
        );
        let summary = emu.step_jit_summarize::<JitSummary>(&mut jit)?;
        tracing::info!(?summary);
        assert_eq!(emu.cpu.gpr[10] as i16, expected);
        Ok(())
    }
}

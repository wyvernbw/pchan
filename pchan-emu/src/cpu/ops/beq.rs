use std::fmt::Display;

use tracing::instrument;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub struct BEQ {
    pub rs: u8,
    pub rt: u8,
    pub imm: i16,
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

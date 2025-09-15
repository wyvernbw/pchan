use std::fmt::Display;

use tracing::instrument;

use crate::cpu::{
    REG_STR,
    ops::{BoundaryType, EmitCtx, EmitSummary, MipsOffset, Op, OpCode, PrimeOp, TryFromOpcodeErr},
};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct BNE {
    rs: usize,
    rt: usize,
    imm: i32,
}

#[inline]
pub fn bne(rs: usize, rt: usize, dest: i32) -> OpCode {
    BNE { rs, rt, imm: dest }.into_opcode()
}

impl TryFrom<OpCode> for BNE {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::BNE)?;
        Ok(BNE {
            rs: (opcode.bits(21..26)) as usize,
            rt: (opcode.bits(16..21)) as usize,
            imm: (opcode.bits(0..16) as i16 as i32) << 2,
        })
    }
}

impl Display for BNE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bne ${} ${} 0x{:08X}",
            REG_STR[self.rs], REG_STR[self.rt], self.imm
        )
    }
}

impl Op for BNE {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::BlockSplit {
            lhs: MipsOffset::Relative(self.imm + 4),
            rhs: MipsOffset::Relative(4),
        })
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
            .with_primary(PrimeOp::BNE)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(0..16, (self.imm >> 2) as i16 as u32)
    }

    #[instrument("bne", skip_all)]
    fn emit_hazard(&self, mut state: EmitCtx) -> EmitSummary {
        use crate::cranelift_bs::*;

        let rs = state.emit_get_register(self.rs);
        let rt = state.emit_get_register(self.rt);

        let cond = state.ins().icmp(IntCC::NotEqual, rs, rt);

        let then_block = state.next_at(0).clif_block();
        let then_params = state.out_params(then_block);

        let else_block = state.next_at(1).clif_block();
        let else_params = state.out_params(else_block);

        tracing::debug!(
            "branch: then={:?}({} deps) else={:?}({} deps)",
            then_block,
            then_params.len(),
            else_block,
            else_params.len()
        );

        state
            .ins()
            .brif(cond, then_block, &then_params, else_block, &else_params);
        EmitSummary::builder().build(&state.fn_builder)
    }

    fn emit_ir(&self, ctx: EmitCtx) -> Option<EmitSummary> {
        Some(EmitSummary::builder().build(&ctx.fn_builder))
    }
}

#[cfg(test)]
mod tests {

    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::cpu::ops::prelude::*;
    use crate::{Emu, dynarec::JitSummary, memory::KSEG0Addr, test_utils::emulator};

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
    ) -> color_eyre::Result<()> {
        emulator.mem.write_array(
            KSEG0Addr::from_phys(0x0),
            &[
                addiu(8, 0, test.a),
                addiu(9, 0, test.b),
                bne(8, 9, 16),
                nop(),
                addiu(10, 0, test.otherwise),
                OpCode(69420),
                nop(),
                addiu(10, 0, test.then),
                OpCode(69420),
            ],
        );

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);

        assert_eq!(emulator.cpu.gpr[10], test.expected);

        Ok(())
    }
}

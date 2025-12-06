use crate::dynarec::prelude::*;

/// FIXME: might require the imm20 field
#[derive(Debug, Clone, Copy, derive_more::Display, Hash, PartialEq, Eq)]
#[display("break")]
pub struct BREAK;

impl Op for BREAK {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        // return control to emulator to jump to exception handler
        // this will use the cached exception handler
        //
        // at some point replace with direct jumps to cached functions
        // for all jump instructions
        Some(BoundaryType::Function)
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::BREAK)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        let cpu = ctx.cpu();
        let [create_pc, store_pc] = ctx.emit_store_pc_imm(ctx.pc);
        let handle_break = ctx
            .fn_builder
            .pure()
            .call(ctx.func_ref_table.handle_break, &[cpu]);
        let ret = ctx.fn_builder.pure().return_(&[]);
        EmitSummary::builder()
            .instructions([
                now(create_pc),
                now(store_pc),
                now(handle_break),
                terminator(bomb(0, ret)),
            ])
            .build(ctx.fn_builder)
    }
}

pub fn brk() -> OpCode {
    BREAK.into_opcode()
}

#[cfg(test)]
mod tests {
    use crate::{dynarec::prelude::*, jit::JIT, test_utils::jit};
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::Emu;

    #[rstest]
    fn break_1(setup_tracing: (), mut jit: JIT) {
        let mut emu = Emu::default();
        emu.write_many(0x0, &program([brk()]));

        let summary = emu.step_jit_summarize::<JitSummary>(&mut jit).unwrap();
        tracing::info!(?summary);

        assert_eq!(emu.cpu.pc, 0xbfc0_0180);
    }
}

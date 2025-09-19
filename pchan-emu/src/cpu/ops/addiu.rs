use std::fmt::Display;

use crate::FnBuilderExt;
use crate::dynarec::prelude::*;

use super::PrimeOp;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct ADDIU {
    rs: usize,
    rt: usize,
    imm: i16,
}

impl TryFrom<OpCode> for ADDIU {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode
            .as_primary(PrimeOp::ADDIU)
            .or_else(|_| opcode.as_primary(PrimeOp::ADDI))?;
        Ok(ADDIU {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Display for ADDIU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "addiu ${} ${} {}",
            REG_STR[self.rt], REG_STR[self.rs], self.imm
        )
    }
}

impl Op for ADDIU {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::ADDIU)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }

    // #[instrument("addiu", skip_all, fields(node = ?state.node, block = ?state.block().clif_block()))]
    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        use crate::cranelift_bs::*;

        // case 1: 0 + x = x
        // => 1 iconst instruction
        if self.rs == 0 {
            let (rt, iconst) = state.fn_builder.inst(|f| {
                f.pure()
                    .UnaryImm(Opcode::Iconst, types::I32, Imm64::new(self.imm as i64))
                    .0
            });
            return EmitSummary::builder()
                .instructions([now(iconst)])
                .register_updates([(self.rt, rt)])
                .build(state.fn_builder);
        }

        // x + 0 = x
        // => 1 iconst instruction or 0 instructions if rs is cached
        let (rs, loadrs) = state.emit_get_register(self.rs);
        if self.imm == 0 {
            return EmitSummary::builder()
                .instructions([now(loadrs)])
                .register_updates([(self.rt, rs)])
                .build(state.fn_builder);
        }

        let (rt, iadd_imm) = state.fn_builder.inst(|f| {
            f.pure()
                .BinaryImm64(Opcode::IaddImm, types::I32, Imm64::new(self.imm as i64), rs)
                .0
        });

        EmitSummary::builder()
            .instructions([now(loadrs), now(iadd_imm)])
            .register_updates([(self.rt, rt)])
            .build(state.fn_builder)
    }
}

#[inline]
pub fn addiu(rt: usize, rs: usize, imm: i16) -> OpCode {
    ADDIU { rt, rs, imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::cpu::ops::prelude::*;
    use crate::cpu::program;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    fn addiu_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;
        let program = program([addiu(9, 8, -16), addiu(10, 0, 8), OpCode(69420)]);
        emulator.mem.write_many(emulator.cpu.pc, &program);
        emulator.cpu.gpr[8] = 32;
        emulator.step_jit()?;
        assert_eq!(emulator.cpu.gpr[9], emulator.cpu.gpr[8] - 16);
        assert_eq!(emulator.cpu.gpr[10], 8);
        Ok(())
    }
    #[rstest]
    fn addiu_2_shortpath(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator
            .mem
            .write_many(0x0, &program([addiu(10, 0, 32), OpCode(69420)]));
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 32);
        let op_count = summary.function.unwrap().dfg.num_insts();
        assert!(op_count <= 5);
        Ok(())
    }
    #[rstest]
    fn addiu_3_shortpath(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::dynarec::JitSummary;

        emulator.mem.write_many(
            0x0,
            &program([addiu(8, 0, 21), addiu(10, 8, 0), OpCode(69420)]),
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 21);
        let op_count = summary.function.unwrap().dfg.num_insts();
        assert!(op_count <= 7);
        Ok(())
    }
}

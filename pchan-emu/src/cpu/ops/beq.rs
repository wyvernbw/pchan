use std::fmt::Display;

use tracing::instrument;

use crate::cpu::{
    REG_STR,
    ops::{BoundaryType, EmitCtx, EmitSummary, MipsOffset, Op, OpCode, PrimeOp, TryFromOpcodeErr},
};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct BEQ {
    rs: usize,
    rt: usize,
    imm: i32,
}

#[inline]
pub fn beq(rs: usize, rt: usize, dest: i32) -> OpCode {
    BEQ { rs, rt, imm: dest }.into_opcode()
}

impl TryFrom<OpCode> for BEQ {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::BEQ)?;
        Ok(BEQ {
            rs: (opcode.bits(21..26)) as usize,
            rt: (opcode.bits(16..21)) as usize,
            imm: (opcode.bits(0..16) as i16 as i32) << 2,
        })
    }
}

impl Display for BEQ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "beq ${} ${} 0x{:08X}",
            REG_STR[self.rs], REG_STR[self.rt], self.imm
        )
    }
}

impl Op for BEQ {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::BlockSplit {
            lhs: MipsOffset::Relative(self.imm + 4),
            rhs: MipsOffset::Relative(4),
        })
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
            .with_primary(PrimeOp::BEQ)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(0..16, (self.imm >> 2) as i16 as u32)
    }

    fn emit_ir(&self, state: EmitCtx) -> Option<EmitSummary> {
        Some(EmitSummary::builder().build(state.fn_builder))
    }

    fn hazard_trigger(&self, current_pc: u32) -> Option<u32> {
        Some(current_pc + 4)
    }

    #[instrument("beq", skip_all, fields(node = ?state.node, block = ?state.block().clif_block()))]
    fn emit_hazard(&self, mut state: EmitCtx) -> EmitSummary {
        use crate::cranelift_bs::*;

        let next = state
            .cfg
            .neighbors_directed(state.node, petgraph::Direction::Outgoing)
            .collect::<Vec<_>>();
        tracing::info!(?state.node, ?next);

        let rs = state.emit_get_register(self.rs);
        let rt = state.emit_get_register(self.rt);

        let cond = state.ins().icmp(IntCC::Equal, rs, rt);

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

        EmitSummary::builder().build(state.fn_builder)
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, dynarec::JitSummary, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    fn beq_basic_loop(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let func = [
            addiu(8, 0, 0),           // ;  0 $t0 = 0
            addiu(10, 0, 4),          // ;  4 $t2 = 4
            addiu(9, 8, 0x0000_2000), // ;  8 calculate address $t1 = $t0 + 0x0000_2000
            sb(8, 9, 0),              // ; 12 store $i at $t1
            beq(8, 10, 16),           // ; 16 if $t0=$t2(4) jump by 16+8 to reach 40
            nop(),                    // ; 20
            addiu(8, 8, 1),           // ; 24 $t0 = $t0 + 1
            nop(),                    // ; 28
            j(8),                     // ; 32 jump to 8 (return to beginning of loop)
            nop(),                    // ; 36
            nop(),                    // ; 40
            OpCode(69420),            // ; 44 halt
        ];

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc), func);

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        if let Some(func) = summary.function {
            tracing::info!(%func);
        }

        let slice = &emulator.mem.as_ref()[0x0000_2000..(0x000_2000 + 4)];
        assert_eq!(slice, &[0, 1, 2, 3]);
        assert_eq!(emulator.cpu.pc, 48);

        Ok(())
    }
    #[rstest]
    fn beq_taken(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let func = [
            addiu(8, 0, 5),   // $t0 = 5
            addiu(9, 0, 5),   // $t1 = 5
            beq(8, 9, 8),     // $t0 == $t1, branch taken
            addiu(10, 0, 42), // skipped
            sb(10, 0, 0x40),  // skipped
            addiu(11, 0, 99), // executed after branch target
            sb(11, 0, 0x41),  // store 99 at memory[0x41]
            OpCode(69420),    // halt
        ];

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc), func);

        emulator.step_jit()?;

        let slice = &emulator.mem.as_ref()[0x41..0x42];
        assert_eq!(slice, &[99]);

        Ok(())
    }

    #[rstest]
    fn beq_not_taken(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let func = [
            addiu(8, 0, 1),   // $t0 = 1
            addiu(9, 0, 2),   // $t1 = 2
            beq(8, 9, 8),     // $t0 != $t1, branch not taken
            addiu(10, 0, 42), // executed because branch not taken
            sb(10, 0, 0x30),  // store 42 at memory[0x30]
            OpCode(69420),    // halt
        ];

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc), func);

        emulator.step_jit()?;

        let slice = &emulator.mem.as_ref()[0x30..0x31];
        assert_eq!(slice, &[42]);

        Ok(())
    }
}

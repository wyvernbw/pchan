use std::fmt::Display;

use crate::cranelift_bs::*;

use crate::cpu::{RA, ops::prelude::*};

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

    fn emit_ir(&self, mut state: EmitParams) -> Option<EmitSummary> {
        tracing::info!("jal: saving pc 0x{:08X}", state.pc);
        debug_assert_eq!(state.neighbour_count(), 1);
        let pc = state.pc as i64;
        let pc = state.ins().iconst(types::I32, pc + 8);
        state.update_cache_immediate(RA, pc);

        let next_block = state.next_at(0).clif_block();
        let params = state.out_params(next_block);

        tracing::debug!(
            "jumping to {:?} with {} dependencies",
            next_block,
            params.len()
        );

        // JIT::emit_store_reg()
        //     .builder(fn_builder)
        //     .block(state.block().clif_block())
        //     .idx(RA)
        //     .value(pc)
        //     .call();
        state.ins().jump(next_block, &params);
        Some(
            EmitSummary::builder()
                .finished_block(true)
                .pc_update(MipsOffset::RegionJump(self.imm).calculate_address(state.pc))
                .build(state.fn_builder),
        )
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

    use crate::{Emu, cpu::RA, dynarec::JitSummary, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    fn jal_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let program = [
            addiu(8, 0, 32),
            jal(KSEG0Addr::from_phys(0x0000_2000).as_u32()), // 4
            nop(),                                           // 8
            nop(),                                           // 12
        ];

        let function = [
            addiu(9, 0, 69),
            nop(),
            // load return address into $t2
            addiu(10, RA, 0),
            OpCode(69420),
        ];

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc), program);
        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(0x0000_2000), function);

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 12);

        Ok(())
    }
}

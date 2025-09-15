use std::fmt::Display;

use cranelift::prelude::{FunctionBuilder, InstBuilder};

use crate::cpu::ops::{
    BoundaryType, EmitParams, EmitSummary, MipsOffset, Op, OpCode, PrimeOp, TryFromOpcodeErr,
};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct J {
    pub imm: u32,
}

impl TryFrom<OpCode> for J {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::J)?;
        Ok(J {
            imm: opcode.bits(0..26) << 2,
        })
    }
}

impl Display for J {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "j 0x{:08X}", self.imm)
    }
}

impl Op for J {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::Block {
            offset: MipsOffset::RegionJump(self.imm),
        })
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::J)
            .set_bits(0..26, self.imm >> 2)
    }

    fn emit_ir(&self, state: EmitParams, fn_builder: &mut FunctionBuilder) -> Option<EmitSummary> {
        tracing::info!("j: jump to 0x{:08X}", self.imm);
        debug_assert_eq!(state.neighbour_count(), 1);
        let next_block = state.next_at(0);

        let params = state.out_params(next_block.clif_block(), fn_builder);
        fn_builder.ins().jump(next_block.clif_block(), &params);
        Some(
            EmitSummary::builder()
                .pc_update(MipsOffset::RegionJump(self.imm).calculate_address(state.pc))
                .finished_block(true)
                .build(fn_builder),
        )
    }
}

#[inline]
pub fn j(imm: u32) -> OpCode {
    J { imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, JitSummary, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    fn basic_jump(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let program = [
            addiu(8, 0, 32),
            j(KSEG0Addr::from_phys(0x0000_2000).as_u32()),
            nop(),
        ];

        let function = [addiu(9, 0, 69), nop(), OpCode(69420)];

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc), program);
        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(0x0000_2000), function);

        emulator.step_jit()?;
        assert_eq!(emulator.cpu.gpr[9], 69);

        Ok(())
    }
    #[rstest]
    fn jump_delay_hazard_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let program = [
            addiu(8, 0, 32),
            j(KSEG0Addr::from_phys(0x0000_2000).as_u32()),
            addiu(10, 0, 32),
        ];

        let function = [addiu(9, 0, 69), nop(), OpCode(69420)];

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc), program);
        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(0x0000_2000), function);

        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);

        assert_eq!(emulator.cpu.gpr[9], 69);
        assert_eq!(emulator.cpu.gpr[10], 32);

        Ok(())
    }
}

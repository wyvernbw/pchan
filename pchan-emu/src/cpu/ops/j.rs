use cranelift::{codegen::ir::BlockArg, prelude::InstBuilder};

use crate::cpu::ops::{
    BoundaryType, EmitParams, EmitSummary, Op, OpCode, PrimeOp, TryFromOpcodeErr,
};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct J {
    imm: i32,
}

impl J {
    pub fn try_from_opcode(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::J)?;
        Ok(J {
            imm: (opcode.bits(0..26) as i32) << 2,
        })
    }
}

impl Op for J {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::Block { offset: self.imm })
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::J)
            .set_bits(0..26, (self.imm >> 2) as u32)
    }

    fn emit_ir(&self, state: EmitParams) -> Option<EmitSummary> {
        Some(
            EmitSummary::builder()
                .pc_update((state.pc & 0xF0000000).wrapping_add_signed(self.imm))
                .build(),
        )
    }

    fn post_emit_ir(&self, state: EmitParams) {
        let Some(next_block) = state.next_blocks[0] else {
            tracing::error!("jump has no next blocks");
            return;
        };
        let params = state.fn_builder.block_params(state.block);
        let params = params
            .iter()
            .cloned()
            .map(BlockArg::Value)
            .collect::<Box<[_]>>();
        state.fn_builder.ins().jump(next_block, &params);
    }
}

#[inline]
pub fn j(imm: i32) -> OpCode {
    J { imm }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, cpu::ops::OpCode, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    fn basic_jump(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let program = [
            addiu(8, 0, 32),
            j(KSEG0Addr::from_phys(0x0000_2000).as_u32() as i32),
            nop(),
        ];

        let function = [addiu(9, 0, 69), nop(), OpCode(69420)];

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc as u32), program);
        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(0x0000_2000), function);

        emulator.advance_jit()?;
        assert_eq!(emulator.cpu.gpr[9], 69);

        Ok(())
    }
    #[rstest]
    fn jump_delay_hazard_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let program = [
            addiu(8, 0, 32),
            j(KSEG0Addr::from_phys(0x0000_2000).as_u32() as i32),
            addiu(10, 0, 32),
        ];

        let function = [addiu(9, 0, 69), nop(), OpCode(69420)];

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc as u32), program);
        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(0x0000_2000), function);

        emulator.advance_jit()?;
        assert_eq!(emulator.cpu.gpr[9], 69);
        assert_eq!(emulator.cpu.gpr[10], 32);

        Ok(())
    }
}

use crate::cpu::ops::{self, BoundaryType, EmitSummary, Op, TryFromOpcodeErr};
use crate::cranelift_bs::*;

use super::{OpCode, PrimeOp};

#[derive(Debug, Clone, Copy)]
pub(crate) struct LB {
    rt: usize,
    rs: usize,
    imm: i16,
}

pub(crate) fn lb(rt: usize, rs: usize, imm: i16) -> ops::OpCode {
    LB { rt, rs, imm }.into_opcode()
}

impl LB {
    pub(crate) fn try_from_opcode(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::LB)?;
        Ok(LB {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Op for LB {
    fn emit_ir(&self, mut state: super::EmitParams<'_, '_>) -> Option<EmitSummary> {
        // get pointer to memory passed as argument to the function
        let mem_ptr = state.memory();

        // get cached register if possible, otherwise load it in
        let rs = state.emit_get_register(self.rs);
        let mem_ptr = state.fn_builder.ins().iadd(mem_ptr, rs);

        let rt =
            state
                .fn_builder
                .ins()
                .sload8(types::I64, MemFlags::new(), mem_ptr, self.imm as i32);
        Some(
            EmitSummary::builder()
                .delayed_register_updates(vec![(self.rt, rt)].into_boxed_slice())
                .build(),
        )
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> ops::OpCode {
        ops::OpCode::default()
            .with_primary(PrimeOp::LB)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::{
        Emu,
        cpu::ops::{self},
        memory::KSEG0Addr,
        test_utils::emulator,
    };

    #[rstest]
    pub fn test_lb_sign_extension(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        emulator.mem.write(KSEG0Addr::from_phys(32), 0xFFu8);
        emulator.mem.write(KSEG0Addr::from_phys(33), 0x7Fu8);
        emulator.mem.write_all(
            KSEG0Addr::from_phys(0),
            [lb(8, 9, 0), lb(10, 9, 1), nop(), ops::OpCode(69420)],
        );

        emulator.cpu.gpr[9] = 32; // base register

        // Run the block
        emulator.advance_jit()?;

        // 0xFF should be sign-extended to 0xFFFFFFFFFFFFFFFF
        assert_eq!(emulator.cpu.gpr[8], 0xFFFFFFFFFFFFFFFF);

        // 0x7F should be sign-extended to 0x7F
        assert_eq!(emulator.cpu.gpr[10], 0x7F);

        Ok(())
    }
}

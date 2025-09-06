use crate::cpu::ops::{self, BoundaryType, EmitSummary, Op, TryFromOpcodeErr};
use crate::cranelift_bs::*;

use super::PrimeOp;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) struct LHU {
    rt: usize,
    rs: usize,
    imm: i16,
}

#[inline]
pub(crate) fn lhu(rt: usize, rs: usize, imm: i16) -> ops::OpCode {
    LHU { rt, rs, imm }.into_opcode()
}

impl LHU {
    pub(crate) fn try_from_opcode(opcode: ops::OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::LHU)?;
        Ok(LHU {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Op for LHU {
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
                .uload16(types::I64, MemFlags::new(), mem_ptr, self.imm as i32);
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
            .with_primary(PrimeOp::LHU)
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
        cpu::ops::{self, DecodedOp},
        memory::{KSEG0Addr, PhysAddr},
        test_utils::emulator,
    };

    #[rstest]
    pub fn test_lhu(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        emulator.mem.write_all(
            KSEG0Addr::from_phys(0),
            [lhu(8, 9, 4), nop(), ops::OpCode(69420)],
        );

        let op = emulator.mem.read::<ops::OpCode>(PhysAddr(0));
        tracing::debug!(decoded = ?DecodedOp::try_new(op));
        tracing::debug!("{:08X?}", &emulator.mem.as_ref()[..22]);

        emulator.cpu.gpr[9] = 16;

        emulator.mem.write::<u16>(KSEG0Addr::from_phys(20), 0xABCD);

        emulator.advance_jit()?;

        assert_eq!(emulator.cpu.gpr[8], 0xABCDu64);

        Ok(())
    }
}

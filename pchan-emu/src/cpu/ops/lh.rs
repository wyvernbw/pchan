use std::fmt::Display;

use crate::cpu::REG_STR;
use crate::cpu::ops::{self, BoundaryType, EmitSummary, Op, TryFromOpcodeErr};
use crate::cranelift_bs::*;

use super::{OpCode, PrimeOp};

#[derive(Debug, Clone, Copy)]
pub struct LH {
    rt: usize,
    rs: usize,
    imm: i16,
}

pub fn lh(rt: usize, rs: usize, imm: i16) -> ops::OpCode {
    LH { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for LH {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::LH)?;
        Ok(LH {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Display for LH {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lh ${} ${} {}",
            REG_STR[self.rt], REG_STR[self.rs], self.imm
        )
    }
}

impl Op for LH {
    fn emit_ir(&self, mut state: super::EmitParams) -> Option<EmitSummary> {
        // get pointer to memory passed as argument to the function
        let mem_ptr = state.memory();

        // get cached register if possible, otherwise load it in
        let rs = state.emit_get_register(self.rs);
        let rs = state.emit_map_address_to_host(rs);
        let mem_ptr = state.ins().iadd(mem_ptr, rs);

        let rt = state
            .ins()
            .sload16(types::I32, MemFlags::new(), mem_ptr, self.imm as i32);
        Some(
            EmitSummary::builder()
                .delayed_register_updates(vec![(self.rt, rt)].into_boxed_slice())
                .build(state.fn_builder),
        )
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> ops::OpCode {
        ops::OpCode::default()
            .with_primary(PrimeOp::LH)
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
    pub fn test_lh_sign_extension(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        emulator.mem.write::<u16>(KSEG0Addr::from_phys(32), 0x8000); // -32768
        emulator.mem.write::<u16>(KSEG0Addr::from_phys(34), 0x7FFF); // +32767

        emulator.mem.write_all(
            KSEG0Addr::from_phys(0),
            [lh(8, 9, 0), lh(10, 9, 2), nop(), ops::OpCode(69420)],
        );

        emulator.cpu.gpr[9] = 32; // base register

        // Run the block
        emulator.step_jit()?;

        assert_eq!(emulator.cpu.gpr[8], 0xFFFF_8000u32);

        assert_eq!(emulator.cpu.gpr[10], 0x7FFF);

        Ok(())
    }
}

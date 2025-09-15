use std::fmt::Display;

use crate::cpu::REG_STR;
use crate::cpu::ops::{self, BoundaryType, EmitSummary, Op, TryFromOpcodeErr};
use crate::cranelift_bs::*;

use super::{EmitParams, OpCode, PrimeOp};

#[derive(Debug, Clone, Copy)]
pub struct SW {
    rt: usize,
    rs: usize,
    imm: i16,
}

pub fn sw(rt: usize, rs: usize, imm: i16) -> ops::OpCode {
    SW { rt, rs, imm }.into_opcode()
}

impl TryFrom<OpCode> for SW {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::SW)?;
        Ok(SW {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Display for SW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sw ${} ${} {}",
            REG_STR[self.rt], REG_STR[self.rs], self.imm
        )
    }
}

impl Op for SW {
    fn emit_ir(&self, mut state: EmitParams) -> Option<EmitSummary> {
        // get pointer to memory passed as argument to the function
        let mem_ptr = state.memory();

        // get cached register if possible, otherwise load it in
        let rs = state.emit_get_register(self.rs);
        let rs = state.emit_map_address_to_host(rs);

        let rt = state.emit_get_register(self.rt);
        let mem_ptr = state.ins().iadd(mem_ptr, rs);

        state
            .ins()
            .store(MemFlags::new(), rt, mem_ptr, self.imm as i32);

        None
    }

    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> ops::OpCode {
        ops::OpCode::default()
            .with_primary(PrimeOp::SW)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::{Emu, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    pub fn test_sw(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(0), [sw(9, 8, 0), OpCode(69420)]);

        emulator.cpu.gpr[8] = 32; // base register
        emulator.cpu.gpr[9] = u32::MAX;

        emulator.step_jit()?;

        assert_eq!(emulator.mem.read::<u32>(KSEG0Addr::from_phys(32)), u32::MAX);

        Ok(())
    }
}

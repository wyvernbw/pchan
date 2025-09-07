use std::fmt::Display;

use crate::cpu::REG_STR;
use crate::cpu::ops::{self, BoundaryType, EmitSummary, Op, TryFromOpcodeErr};
use crate::cranelift_bs::*;

use super::{OpCode, PrimeOp};

#[derive(Debug, Clone, Copy)]
pub struct LW {
    rt: usize,
    rs: usize,
    imm: i16,
}

pub fn lw(rt: usize, rs: usize, imm: i16) -> ops::OpCode {
    LW { rt, rs, imm }.into_opcode()
}

impl LW {
    pub fn try_from_opcode(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_primary(PrimeOp::LW)?;
        Ok(LW {
            rt: opcode.bits(16..21) as usize,
            rs: opcode.bits(21..26) as usize,
            imm: opcode.bits(0..16) as i16,
        })
    }
}

impl Display for LW {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lw ${} ${} {}",
            REG_STR[self.rt], REG_STR[self.rs], self.imm
        )
    }
}

impl Op for LW {
    fn emit_ir(&self, mut state: super::EmitParams) -> Option<EmitSummary> {
        // get pointer to memory passed as argument to the function
        let mem_ptr = state.memory();

        // get cached register if possible, otherwise load it in
        let rs = state.emit_get_register(self.rs);
        let mem_ptr = state.fn_builder.ins().iadd(mem_ptr, rs);

        let rt = state
            .fn_builder
            .ins()
            .uload32(MemFlags::new(), mem_ptr, self.imm as i32);
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
            .with_primary(PrimeOp::LW)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
            .set_bits(0..16, (self.imm as i32 as i16) as u32)
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_ne;
    use rstest::rstest;

    use crate::{
        Emu,
        cpu::ops::{self},
        memory::KSEG0Addr,
        test_utils::emulator,
    };

    #[rstest]
    pub fn test_lw(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        emulator
            .mem
            .write::<u32>(KSEG0Addr::from_phys(32), 0xDEAD_BEEF); // -32768

        emulator.mem.write_all(
            KSEG0Addr::from_phys(0),
            [lw(8, 9, 0), nop(), ops::OpCode(69420)],
        );

        emulator.cpu.gpr[9] = 32; // base register

        // Run the block
        emulator.step_jit()?;

        assert_eq!(emulator.cpu.gpr[8], 0xDEAD_BEEF);

        Ok(())
    }

    #[rstest]
    pub fn load_delay_hazard_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        emulator
            .mem
            .write::<u32>(KSEG0Addr::from_phys(32), 0xDEAD_BEEF);

        emulator.mem.write_all(
            KSEG0Addr::from_phys(0),
            [lw(8, 9, 0), sb(8, 10, 0), ops::OpCode(69420)],
        );

        emulator.cpu.gpr[9] = 32; // base register
        emulator.cpu.gpr[10] = 36;

        // Run the block
        emulator.step_jit()?;

        assert_eq!(emulator.cpu.gpr[8], 0xDEAD_BEEF);
        // 0xDEAD_BEEF wasnt loaded by the time the store happened
        assert_eq!(emulator.mem.read::<u8>(KSEG0Addr::from_phys(36)), 0);

        Ok(())
    }

    #[rstest]
    pub fn load_delay_hazard_2(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        emulator
            .mem
            .write::<u32>(KSEG0Addr::from_phys(32), 0xDEAD_BEEF);

        emulator.mem.write_all(
            KSEG0Addr::from_phys(0),
            [lw(8, 9, 0), nop(), sb(8, 10, 0), ops::OpCode(69420)],
        );

        emulator.cpu.gpr[9] = 32; // base register
        emulator.cpu.gpr[10] = 36;

        // Run the block
        emulator.step_jit()?;

        assert_eq!(emulator.cpu.gpr[8], 0xDEAD_BEEF);
        assert_ne!(emulator.mem.read::<u8>(KSEG0Addr::from_phys(36)), 0);

        Ok(())
    }
}

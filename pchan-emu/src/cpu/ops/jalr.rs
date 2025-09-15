use std::fmt::Display;

use crate::cpu::{REG_STR, ops::prelude::*};
use crate::cranelift_bs::*;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct JALR {
    pub rd: usize,
    pub rs: usize,
}

impl TryFrom<OpCode> for JALR {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::JALR)?;
        Ok(JALR {
            rd: opcode.bits(11..16) as usize,
            rs: opcode.bits(21..26) as usize,
        })
    }
}

impl Display for JALR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "jalr ${} ${}", REG_STR[self.rd], REG_STR[self.rs])
    }
}

impl Op for JALR {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::Function { auto_set_pc: false })
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::JALR)
            .set_bits(21..26, self.rs as u32)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(&self, mut state: EmitParams) -> Option<EmitSummary> {
        tracing::info!("jalr: saving pc 0x{:08X}", state.pc);
        let rs = state.emit_get_register(self.rs);
        let rs = state.emit_map_address_to_physical(rs);

        let pc = state.pc as i64;
        let pc = state.ins().iconst(types::I32, pc + 8);
        state.emit_store_pc(rs);
        state.emit_store_register(self.rd, pc);
        Some(
            EmitSummary::builder()
                .register_updates([(self.rd, pc), (self.rd, pc)])
                .finished_block(false)
                .build(state.fn_builder),
        )
    }
}

#[inline]
pub fn jalr(rd: usize, rs: usize) -> OpCode {
    JALR { rd, rs }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, cpu::RA, dynarec::JitSummary, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    fn jalr_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let program = [
            lui(8, 0x8000u16 as i16),
            ori(8, 8, 0x2000),
            jalr(RA, 8),
            nop(),
            addiu(9, 9, 32),
            OpCode(69420),
        ];

        let function = [addiu(9, 0, 69), nop(), jr(RA)];

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc), program);
        emulator.mem.write_all(KSEG0Addr(0x8000_2000), function);

        for i in 0..3 {
            let summary = emulator.step_jit_summarize::<JitSummary>()?;
            tracing::info!(?summary.function);
            tracing::info!(emulator.cpu.pc = %format!("0x{:08X}", emulator.cpu.pc));
            assert_eq!(emulator.cpu.gpr[8], 0x8000_2000);
        }
        assert_eq!(emulator.cpu.gpr[9], 69 + 32);

        Ok(())
    }

    #[rstest]
    fn jalr_1_delay_hazard(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let program = [
            // initialize state
            lui(8, 0x8000u16 as i16),
            ori(8, 8, 0x2000),
            addiu(9, 0, 0),
            // call function
            jalr(RA, 8),
            // fuck up the state in the hazard
            addiu(9, 0, -69),
            // continue
            nop(),
            addiu(9, 9, 32),
            OpCode(69420),
        ];

        let function = [
            // using fucked up state $t0=-69, end result is 0
            addiu(9, 9, 69),
            nop(),
            jr(RA),
        ];

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc), program);
        emulator.mem.write_all(KSEG0Addr(0x8000_2000), function);

        for i in 0..3 {
            let summary = emulator.step_jit_summarize::<JitSummary>()?;
            tracing::info!(?summary.function);
            tracing::info!(emulator.cpu.pc = %format!("0x{:08X}", emulator.cpu.pc));
            assert_eq!(emulator.cpu.gpr[8], 0x8000_2000);
        }
        assert_eq!(emulator.cpu.gpr[9], 32);

        Ok(())
    }
}

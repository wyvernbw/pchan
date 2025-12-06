use std::fmt::Display;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub struct JALR {
    pub rd: u8,
    pub rs: u8,
}

impl JALR {
    pub fn new(rd: u8, rs: u8) -> Self {
        Self { rd, rs }
    }
}

impl TryFrom<OpCode> for JALR {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::JALR)?;
        Ok(JALR {
            rd: opcode.bits(11..16) as u8,
            rs: opcode.bits(21..26) as u8,
        })
    }
}

impl Display for JALR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "jalr ${} ${}",
            REG_STR[self.rd as usize], REG_STR[self.rs as usize]
        )
    }
}

impl Op for JALR {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::Function)
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::JALR)
            .set_bits(21..26, self.rs as u32)
            .set_bits(11..16, self.rd as u32)
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        // tracing::info!("jalr: saving pc 0x{:08X}", state.pc);
        // compute jump address
        let (rs, loadreg) = state.emit_get_register(self.rs);
        // store jump address at pc
        let storepc = state.emit_store_pc(rs);

        // save old pc in rd
        let pc = state.pc as i64;
        let (pc, iconst) = state.inst(|f| {
            f.pure()
                .UnaryImm(Opcode::Iconst, types::I32, Imm64::new(pc + 8))
                .0
        });
        let storerd = state.emit_store_register(self.rd, pc);

        // return
        let ret = state
            .fn_builder
            .pure()
            .MultiAry(Opcode::Return, types::INVALID, ValueList::new())
            .0;

        EmitSummary::builder()
            .instructions([
                now(loadreg),
                now(storepc),
                now(iconst),
                now(storerd),
                terminator(bomb(1, ret)),
            ])
            .register_updates([(self.rd, pc)])
            .build(state.fn_builder)
    }
}

#[inline]
pub fn jalr(rd: u8, rs: u8) -> OpCode {
    JALR { rd, rs }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::jit::JIT;
    use crate::test_utils::jit;
    use crate::{Emu, cpu::RA, test_utils::emulator};

    #[rstest]
    fn jalr_1(setup_tracing: (), mut emulator: Emu, mut jit: JIT) -> color_eyre::Result<()> {
        let main = program([
            addiu(9, 0, 0),
            lui(8, 0x8000u16 as i16),
            ori(8, 8, 0x2000),
            jalr(RA, 8),
            nop(),
            addiu(9, 9, 32),
            OpCode(69420),
        ]);

        let function = program([addiu(9, 0, 69), nop(), jr(RA)]);

        emulator.write_many(emulator.cpu.pc, &main);
        emulator.write_many(0x8000_2000, &function);

        for i in 0..3 {
            let summary = emulator.step_jit_summarize::<JitSummary>(&mut jit)?;
            tracing::info!(?summary.function);
            tracing::info!(emulator.cpu.pc = %format!("0x{:08X}", emulator.cpu.pc));
            assert_eq!(emulator.cpu.gpr[8], 0x8000_2000);
        }
        assert_eq!(emulator.cpu.gpr[9], 69 + 32);

        Ok(())
    }

    #[rstest]
    fn jalr_1_delay_hazard(
        setup_tracing: (),
        mut emulator: Emu,
        mut jit: JIT,
    ) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let main = program([
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
        ]);

        let function = program([
            // using fucked up state $t0=-69, end result is 0
            addiu(9, 9, 69),
            nop(),
            jr(RA),
        ]);

        emulator.write_many(emulator.cpu.pc, &main);
        emulator.write_many(0x8000_2000, &function);

        for i in 0..3 {
            let summary = emulator.step_jit_summarize::<JitSummary>(&mut jit)?;
            tracing::info!(?summary.function);
            tracing::info!(emulator.cpu.pc = %format!("0x{:08X}", emulator.cpu.pc));
            assert_eq!(emulator.cpu.gpr[8], 0x8000_2000);
        }
        assert_eq!(emulator.cpu.gpr[9], 32);

        Ok(())
    }
}

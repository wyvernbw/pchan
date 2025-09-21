use std::fmt::Display;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct JR {
    pub rs: u8,
}

impl JR {
    pub fn new(rs: u8) -> Self {
        Self { rs }
    }
}

impl TryFrom<OpCode> for JR {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::JR)?;
        Ok(JR {
            rs: opcode.bits(21..26) as u8,
        })
    }
}

impl Display for JR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "jr ${}", REG_STR[self.rs as usize])
    }
}

impl Op for JR {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::Function)
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::JR)
            .set_bits(21..26, self.rs as u32)
    }

    fn hazard(&self) -> Option<u32> {
        Some(1)
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        let (rs, loadreg) = state.emit_get_register(self.rs);
        let (rs, mapaddr) = state.emit_map_address_to_physical(rs);

        debug_assert_eq!(
            state.fn_builder.func.dfg.value_type(rs),
            types::I32,
            "expected i32 value"
        );

        let storers = state.emit_store_pc(rs);
        let ret = state
            .fn_builder
            .pure()
            .MultiAry(Opcode::Return, types::INVALID, ValueList::new())
            .0;

        EmitSummary::builder()
            .instructions([
                now(loadreg),
                now(mapaddr),
                now(storers),
                terminator(bomb(1, ret)),
            ])
            .build(state.fn_builder)
    }
}

#[inline]
pub fn jr(rs: u8) -> OpCode {
    JR { rs }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, dynarec::JitSummary, test_utils::emulator};

    #[rstest]
    fn jr_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let main = program([lui(8, 0x8000u16 as i16), ori(8, 8, 0x2000), jr(8), nop()]);

        let function = program([addiu(9, 0, 69), nop(), OpCode(69420)]);

        emulator.mem.write_many(emulator.cpu.pc, &main);
        emulator.mem.write_many(0x8000_2000, &function);

        for i in 0..2 {
            let summary = emulator.step_jit_summarize::<JitSummary>()?;
            tracing::info!(?summary.function);
            tracing::info!(emulator.cpu.pc = %format!("0x{:08X}", emulator.cpu.pc));
            assert_eq!(emulator.cpu.gpr[8], 0x8000_2000);
        }
        assert_eq!(emulator.cpu.gpr[9], 69);

        Ok(())
    }
}

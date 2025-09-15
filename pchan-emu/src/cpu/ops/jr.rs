use std::fmt::Display;

use crate::cranelift_bs::*;

use crate::cpu::{REG_STR, ops::prelude::*};

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct JR {
    pub rs: usize,
}

impl TryFrom<OpCode> for JR {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::JR)?;
        Ok(JR {
            rs: opcode.bits(21..26) as usize,
        })
    }
}

impl Display for JR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "jr ${}", REG_STR[self.rs])
    }
}

impl Op for JR {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        Some(BoundaryType::Function { auto_set_pc: false })
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::JR)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(
        &self,
        mut state: EmitParams,
        fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
        let rs = state.emit_get_register(fn_builder, self.rs);
        let rs = state.emit_map_address_to_physical(fn_builder, rs);

        debug_assert_eq!(
            fn_builder.func.dfg.value_type(rs),
            types::I32,
            "expected i32 value"
        );

        state.emit_store_pc(fn_builder, rs);

        fn_builder.ins().return_(&[]);

        None
    }
}

#[inline]
pub fn jr(rs: usize) -> OpCode {
    JR { rs }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, dynarec::JitSummary, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    fn jr_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        use crate::cpu::ops::prelude::*;

        let program = [lui(8, 0x8000u16 as i16), ori(8, 8, 0x2000), jr(8), nop()];

        let function = [addiu(9, 0, 69), nop(), OpCode(69420)];

        emulator
            .mem
            .write_all(KSEG0Addr::from_phys(emulator.cpu.pc), program);
        emulator.mem.write_all(KSEG0Addr(0x8000_2000), function);

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

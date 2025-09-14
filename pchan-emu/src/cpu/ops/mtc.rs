use std::fmt::Display;

use crate::cpu::REG_STR;
use crate::cpu::ops::prelude::*;
use crate::cranelift_bs::*;

#[derive(Debug, Clone, Copy)]
pub struct MTCn {
    cop: u8,
    rt: u8,
    rd: u8,
}

impl TryFrom<OpCode> for MTCn {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let (value, cop) = value.as_primary_cop()?;
        let value = value.as_cop(CopOp::MTCn)?;
        Ok(MTCn {
            cop,
            rt: value.bits(16..21) as u8,
            rd: value.bits(11..16) as u8,
        })
    }
}

impl Op for MTCn {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary_cop(self.cop)
            .with_cop(CopOp::MTCn)
            .set_bits(16..21, self.rt as u32)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(
        &self,
        mut state: EmitParams,
        fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
        let rt = state.emit_get_register(fn_builder, self.rt.into());
        state.emit_store_cop_register(fn_builder, self.cop, self.rd.into(), rt);
        None
    }
}

impl Display for MTCn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mtc{} ${}, $r{}",
            self.cop, REG_STR[self.rt as usize], self.rd
        )
    }
}

pub fn mtc0(rt: u8, rd: u8) -> OpCode {
    MTCn { cop: 0, rt, rd }.into_opcode()
}

pub fn mtc2(rt: u8, rd: u8) -> OpCode {
    MTCn { cop: 2, rt, rd }.into_opcode()
}

#[cfg(test)]
mod tests {
    use crate::{JitSummary, cpu::ops::prelude::*};
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    fn mtc_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator.mem.write_array(
            KSEG0Addr::from_phys(0x0),
            &[addiu(8, 0, 69), mtc0(8, 16), OpCode(69420)],
        );

        let summary = emulator.step_jit_summarize::<JitSummary>()?;

        tracing::info!(?summary.function);

        assert_eq!(emulator.cpu.cop0.reg[16], 69);
        Ok(())
    }
}

use std::fmt::Display;

use cranelift::prelude::FunctionBuilder;

use crate::cpu::REG_STR;
use crate::cpu::ops::prelude::*;

use super::PrimeOp;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct XOR {
    rd: usize,
    rs: usize,
    rt: usize,
}

impl TryFrom<OpCode> for XOR {
    type Error = TryFromOpcodeErr;
    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_secondary(SecOp::XOR)?;
        Ok(XOR {
            rs: opcode.bits(21..26) as usize,
            rt: opcode.bits(16..21) as usize,
            rd: opcode.bits(11..16) as usize,
        })
    }
}

impl Display for XOR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "xor ${} ${} ${}",
            REG_STR[self.rd], REG_STR[self.rs], REG_STR[self.rt]
        )
    }
}

impl Op for XOR {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::XOR)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(
        &self,
        mut state: EmitParams,
        fn_builder: &mut FunctionBuilder,
    ) -> Option<EmitSummary> {
        use crate::cranelift_bs::*;
        if self.rs == 0 {
            let rt = state.emit_get_register(fn_builder, self.rt);
            return Some(
                EmitSummary::builder()
                    .register_updates([(self.rd, rt)])
                    .build(&fn_builder),
            );
        } else if self.rt == 0 {
            let rs = state.emit_get_register(fn_builder, self.rs);
            return Some(
                EmitSummary::builder()
                    .register_updates([(self.rd, rs)])
                    .build(&fn_builder),
            );
        }
        let rs = state.emit_get_register(fn_builder, self.rs);
        let rt = state.emit_get_register(fn_builder, self.rt);
        let rd = fn_builder.ins().bxor(rs, rt);
        Some(
            EmitSummary::builder()
                .register_updates([(self.rd, rd)])
                .build(&fn_builder),
        )
    }
}

#[inline]
pub fn xor(rd: usize, rs: usize, rt: usize) -> OpCode {
    XOR { rd, rs, rt }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::JitSummary;
    use crate::cpu::ops::prelude::*;
    use crate::{Emu, memory::KSEG0Addr, test_utils::emulator};

    #[rstest]
    #[case(1, 1, 0)]
    #[case(1, 0, 1)]
    #[case(0, 1, 1)]
    #[case(0, 0, 0)]
    #[case(0b11110000, 0b00111100, 0b11001100)]
    #[case(-1, -1, 0)] // 0xFFFF ^ 0xFFFF = 0
    #[case(i16::MIN, 0, i16::MIN as u32)] // -32768 ^ 0 = -32768
    fn xor_1(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: i16,
        #[case] b: i16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        emulator.mem.write_array(
            KSEG0Addr::from_phys(0),
            &[addiu(8, 0, a), addiu(9, 0, b), xor(10, 8, 9), OpCode(69420)],
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
}

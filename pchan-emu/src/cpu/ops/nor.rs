use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct NOR {
    rd: usize,
    rs: usize,
    rt: usize,
}

impl TryFrom<OpCode> for NOR {
    type Error = TryFromOpcodeErr;
    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode.as_secondary(SecOp::NOR)?;
        Ok(NOR {
            rs: opcode.bits(21..26) as usize,
            rt: opcode.bits(16..21) as usize,
            rd: opcode.bits(11..16) as usize,
        })
    }
}

impl Display for NOR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "nor ${} ${} ${}",
            REG_STR[self.rd], REG_STR[self.rs], REG_STR[self.rt]
        )
    }
}

impl Op for NOR {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::NOR)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(&self, mut state: EmitCtx) -> EmitSummary {
        use crate::cranelift_bs::*;
        // case 1: x | 0 = x
        if self.rs == 0 {
            let (rt, loadrt) = state.emit_get_register(self.rt);
            let (rd, bnot) = state.inst(|f| f.pure().Unary(Opcode::Bnot, types::I32, rt).0);

            EmitSummary::builder()
                .instructions([now(loadrt), now(bnot)])
                .register_updates([(self.rd, rd)])
                .build(state.fn_builder)
        // case 2: 0 | x = x
        } else if self.rt == 0 {
            let (rs, loadrs) = state.emit_get_register(self.rs);
            let (rd, bnot) = state.inst(|f| f.pure().Unary(Opcode::Bnot, types::I32, rs).0);

            EmitSummary::builder()
                .instructions([now(loadrs), now(bnot)])
                .register_updates([(self.rd, rd)])
                .build(state.fn_builder)
        // case 3: x | y = z
        } else {
            let (rs, loadrs) = state.emit_get_register(self.rs);
            let (rt, loadrt) = state.emit_get_register(self.rt);
            let (rs_or_rt, bor) =
                state.inst(|f| f.pure().Binary(Opcode::Bor, types::I32, rs, rt).0);
            let (rd, bnot) = state.inst(|f| f.pure().Unary(Opcode::Bnot, types::I32, rs_or_rt).0);

            EmitSummary::builder()
                .instructions([now(loadrs), now(loadrt), now(bor), now(bnot)])
                .register_updates([(self.rd, rd)])
                .build(state.fn_builder)
        }
    }
}

#[inline]
pub fn nor(rd: usize, rs: usize, rt: usize) -> OpCode {
    NOR { rd, rs, rt }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    #[case(0x7FFF, 0x7FFF, 0xFFFF_8000)] // i16::MAX | i16::MAX
    #[case(i16::MIN, i16::MIN, 0x0000_7FFF)] // -32768 | -32768 = -32768, ~(-32768) = 0x7FFF
    #[case(0xFFFFu16 as i16, 0x0000, 0)] // 0xFFFF | 0 = 0xFFFF, ~0xFFFF = 0xFFFF_0000
    #[case(0x0000, 0xFFFFu16 as i16, 0)] // 0 | 0xFFFF = 0xFFFF, ~0xFFFF = 0xFFFF_0000
    #[case(0xAAAAu16 as i16, 0x5555, 0)] // 0xAAAA | 0x5555 = 0xFFFF, ~0xFFFF = 0xFFFF_0000
    #[case(0xF0F0u16 as i16, 0x0F0F, 0)] // 0xF0F0 | 0x0F0F = 0xFFFF, ~0xFFFF = 0xFFFF_0000
    #[case(0, 0xFFFF_FFFFu32 as i16, 0)] // 0 | -1 = -1, ~(-1) = 0
    #[case(0xFFFF_FFFFu32 as i16, 0, 0)] // -1 | 0 = -1, ~(-1) = 0
    fn nor_test(
        setup_tracing: (),
        mut emulator: Emu,
        #[case] a: i16,
        #[case] b: i16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        emulator.mem.write_many(
            0x0,
            &program([addiu(8, 0, a), addiu(9, 0, b), nor(10, 8, 9), OpCode(69420)]),
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
}

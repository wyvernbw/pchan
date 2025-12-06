use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SRAV {
    rd: u8,
    rt: u8,
    rs: u8,
}

impl SRAV {
    pub fn new(rd: u8, rt: u8, rs: u8) -> Self {
        Self { rd, rt, rs }
    }
}

impl Display for SRAV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "srav ${} ${} ${}",
            REG_STR[self.rd as usize], REG_STR[self.rt as usize], REG_STR[self.rs as usize]
        )
    }
}

impl TryFrom<OpCode> for SRAV {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::SRAV)?;
        Ok(SRAV {
            rd: value.bits(11..16) as u8,
            rt: value.bits(16..21) as u8,
            rs: value.bits(21..26) as u8,
        })
    }
}

impl Op for SRAV {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::SRAV)
            .set_bits(11..16, self.rd as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        shift!(self, ctx, Opcode::Sshr)
    }
}

#[inline]
pub fn srav(rd: u8, rt: u8, rs: u8) -> OpCode {
    SRAV { rd, rs, rt }.into_opcode()
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::test_utils::jit;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    #[case(64, 6, 1)]
    #[case(32, 0, 32)]
    #[case(-32, 2, -8i32 as u32)]
    #[case(0b11110000, 4, 0b00001111)]
    fn srav_1(
        setup_tracing: (),
        mut emulator: Emu,
        mut jit: crate::jit::JIT,
        #[case] a: i16,
        #[case] b: i16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        emulator.write_many(
            0x0,
            &program([
                addiu(8, 0, a),
                addiu(9, 0, b),
                srav(10, 8, 9),
                OpCode(69420),
            ]),
        );
        let summary = emulator.step_jit_summarize::<JitSummary>(&mut jit)?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }

    #[rstest]
    #[case(8, 0)]
    #[case(0b00001111, 0)]
    fn srav_2(
        setup_tracing: (),
        mut emulator: Emu,
        mut jit: crate::jit::JIT,
        #[case] value: i16,
        #[case] expected: u32,
    ) -> color_eyre::Result<()> {
        emulator.write_many(
            0x0,
            &program([addiu(9, 0, value), srav(10, 0, 9), OpCode(69420)]),
        );
        let summary = emulator.step_jit_summarize::<JitSummary>(&mut jit)?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], expected);
        Ok(())
    }
}

#[macro_export]
macro_rules! shift {
    ($self:expr, $ctx:expr, $opcode:expr) => {{
        use $crate::dynarec::prelude::*;

        // optimize 0 >> x = 0
        if $self.rt == 0 {
            let (rd, loadzero) = $ctx.emit_get_zero();
            return EmitSummary::builder()
                .instructions([now(loadzero)])
                .register_updates([($self.rd, rd)])
                .build($ctx.fn_builder);
        }
        // optimize x >> 0 = x
        let (rt, loadrt) = $ctx.emit_get_register($self.rt);
        if $self.rs == 0 {
            return EmitSummary::builder()
                .instructions([now(loadrt)])
                .register_updates([($self.rd, rt)])
                .build($ctx.fn_builder);
        }
        let (rs, loadrs) = $ctx.emit_get_register($self.rs);
        let (rd, shift) = $ctx.inst(|f| f.pure().Binary($opcode, types::I32, rt, rs).0);

        EmitSummary::builder()
            .instructions([now(loadrt), now(loadrs), now(shift)])
            .register_updates([($self.rd, rd)])
            .build($ctx.fn_builder)
    }};
}

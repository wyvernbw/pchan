use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy)]
pub struct SLT {
    rd: usize,
    rs: usize,
    rt: usize,
}

pub fn slt(rd: usize, rs: usize, rt: usize) -> OpCode {
    SLT { rd, rs, rt }.into_opcode()
}

impl Display for SLT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "slt ${} ${} ${}",
            REG_STR[self.rd], REG_STR[self.rs], REG_STR[self.rt]
        )
    }
}

impl TryFrom<OpCode> for SLT {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::SLT)?;
        Ok(SLT {
            rd: value.bits(11..16) as usize,
            rt: value.bits(16..21) as usize,
            rs: value.bits(21..26) as usize,
        })
    }
}

impl Op for SLT {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::SLT)
            .set_bits(11..16, self.rd as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        icmp!(self, ctx, IntCC::SignedLessThan)
    }
}

#[cfg(test)]
mod tests {
    use pchan_utils::setup_tracing;
    use rstest::rstest;

    use crate::dynarec::prelude::*;
    use crate::{Emu, test_utils::emulator};

    #[rstest]
    fn basic_slt(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator.mem.write_many(
            0x0,
            &program([
                addiu(8, 0, 16),
                addiu(9, 0, -3),
                slt(10, 9, 8),
                OpCode(69420),
            ]),
        );
        let summary = emulator.step_jit_summarize::<JitSummary>()?;
        tracing::info!(?summary.function);
        assert_eq!(emulator.cpu.gpr[10], 1);

        Ok(())
    }
}

#[macro_export]
macro_rules! icmp {
    ($self:expr, $ctx:expr, $cond:expr) => {{
        use $crate::dynarec::prelude::*;

        let (rs, loadrs) = $ctx.emit_get_register($self.rs);
        let (rt, loadrt) = $ctx.emit_get_register($self.rt);
        let (rd, icmp) = $ctx.inst(|f| {
            f.pure()
                .IntCompare(Opcode::Icmp, types::I32, $cond, rs, rt)
                .0
        });
        let (rd, uextend) = $ctx.inst(|f| f.pure().Unary(Opcode::Uextend, types::I32, rd).0);
        EmitSummary::builder()
            .instructions([now(loadrs), now(loadrt), now(icmp), now(uextend)])
            .register_updates([($self.rd, rd)])
            .build($ctx.fn_builder)
    }};
}

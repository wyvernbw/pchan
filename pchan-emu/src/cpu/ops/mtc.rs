use std::fmt::Display;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy, Hash)]
pub struct MTCn {
    cop: u8,
    rt: u8,
    rd: u8,
}

impl MTCn {
    pub const fn new(cop: u8, rt: u8, rd: u8) -> Self {
        Self { cop, rt, rd }
    }
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

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        let (rt, loadreg) = ctx.emit_get_register(self.rt);
        let storecopreg = ctx.emit_store_cop_register(self.cop, self.rd.into(), rt);
        EmitSummary::builder()
            .instructions([now(loadreg), delayed_maybe(self.hazard(), storecopreg)])
            .build(ctx.fn_builder)
    }

    fn hazard(&self) -> Option<u32> {
        match self.cop {
            0 => Some(0),
            2 => Some(2),
            _ => None,
        }
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
    use crate::{cpu::ops::prelude::*, dynarec::JitSummary};
    use pchan_utils::setup_tracing;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    use crate::{Emu, test_utils::emulator};

    #[rstest]
    fn mtc_1(setup_tracing: (), mut emulator: Emu) -> color_eyre::Result<()> {
        emulator.write_many(0x0, &program([addiu(8, 0, 69), mtc0(8, 16), OpCode(69420)]));

        let summary = emulator.step_jit_summarize::<JitSummary>()?;

        tracing::info!(?summary.function);

        assert_eq!(emulator.cpu.cop0.reg[16], 69);
        Ok(())
    }
}

use std::fmt::Display;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct MFCn {
    cop: u8,
    rt: u8,
    rd: u8,
}

impl MFCn {
    pub const fn new(cop: u8, rt: u8, rd: u8) -> Self {
        Self { cop, rt, rd }
    }
}

impl TryFrom<OpCode> for MFCn {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let (value, cop) = value.as_primary_cop()?;
        let value = value.as_cop(CopOp::MFCn)?;
        Ok(MFCn {
            cop,
            rt: value.bits(16..21) as u8,
            rd: value.bits(11..16) as u8,
        })
    }
}

impl Op for MFCn {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary_cop(self.cop)
            .with_cop(CopOp::MFCn)
            .set_bits(16..21, self.rt as u32)
            .set_bits(11..16, self.rd as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        let (rd_data, loadreg) = ctx.emit_get_cop_register(self.cop, self.rd.into());
        EmitSummary::builder()
            .instructions([now(loadreg)])
            .register_updates([(self.rt, rd_data)])
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

impl Display for MFCn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mfc{} ${}, $r{}",
            self.cop, REG_STR[self.rt as usize], self.rd
        )
    }
}

pub fn mfc0(rt: u8, rd: u8) -> OpCode {
    MFCn { cop: 0, rt, rd }.into_opcode()
}

pub fn mfc2(rt: u8, rd: u8) -> OpCode {
    MFCn { cop: 2, rt, rd }.into_opcode()
}

#[cfg(test)]
mod tests {}

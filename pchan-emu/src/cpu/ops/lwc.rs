use std::fmt::Display;

use crate::dynarec::prelude::*;

#[derive(Debug, Clone, Copy, Hash)]
pub struct LWCn {
    pub cop: u8,
    pub rs: u8,
    pub rt: u8,
    pub imm: i16,
}

impl LWCn {
    pub const fn new(cop: u8, rs: u8, rt: u8, imm: i16) -> Self {
        Self { cop, rs, rt, imm }
    }
}

impl Display for LWCn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lwc{} ${} ${} {}",
            self.cop,
            reg_str(self.rt),
            reg_str(self.rs),
            hex(self.imm)
        )
    }
}

impl Op for LWCn {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
            .with_primary_cop(self.cop)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(&self, ctx: EmitCtx) -> EmitSummary {
        todo!()
    }

    fn hazard(&self) -> Option<u32> {
        match self.cop {
            0 => Some(0),
            2 => Some(2),
            _ => None,
        }
    }
}

pub fn lwc(n: u8) -> impl Fn(u8, u8, i16) -> OpCode {
    move |rt, rs, imm| {
        LWCn {
            cop: n,
            rs,
            rt,
            imm,
        }
        .into_opcode()
    }
}

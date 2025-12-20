use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub struct SRA {
    pub rd: u8,
    pub rt: u8,
    pub imm: i8,
}

impl SRA {
    pub fn new(rd: u8, rt: u8, imm: i8) -> Self {
        Self { rd, rt, imm }
    }
}

impl TryFrom<OpCode> for SRA {
    type Error = TryFromOpcodeErr;

    fn try_from(opcode: OpCode) -> Result<Self, TryFromOpcodeErr> {
        let opcode = opcode
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::SRA)?;
        Ok(SRA {
            rt: opcode.bits(16..21) as u8,
            rd: opcode.bits(11..16) as u8,
            imm: opcode.bits(6..11) as i8,
        })
    }
}

impl Display for SRA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sra ${} ${} {}",
            REG_STR[self.rd as usize],
            REG_STR[self.rt as usize],
            hex(self.imm)
        )
    }
}

impl Op for SRA {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::SRA)
            .set_bits(11..16, self.rd as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(6..11, (self.imm as i32 as i16) as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        shiftimm!(self, ctx, Opcode::SshrImm)
    }
}

#[inline]
pub fn sra(rd: u8, rt: u8, imm: i8) -> OpCode {
    SRA { rd, rt, imm }.into_opcode()
}

#[macro_export]
macro_rules! shiftimm {
    ($self:expr, $ctx:expr, $opcode:expr) => {{
        use $crate::dynarec::prelude::*;
        // case 2: 0 << imm = 0
        if $self.rt == 0 {
            let (rt, loadzero) = $ctx.emit_get_zero();
            return EmitSummary::builder()
                .instructions([now(loadzero)])
                .register_updates([($self.rd, rt)])
                .build($ctx.fn_builder);
        }

        // case 1: $rt << 0 = $rt
        let (rt, loadrt) = $ctx.emit_get_register($self.rt);
        if $self.imm == 0 {
            return EmitSummary::builder()
                .instructions([now(loadrt)])
                .register_updates([($self.rd, rt)])
                .build($ctx.fn_builder);
        }

        // case 3: $rt << imm = $rd
        let (rd, sshr_imm) = $ctx.inst(|f| {
            f.pure()
                .BinaryImm64($opcode, types::I32, Imm64::new($self.imm.into()), rt)
                .0
        });

        EmitSummary::builder()
            .instructions([now(loadrt), now(sshr_imm)])
            .register_updates([($self.rd, rd)])
            .build($ctx.fn_builder)
    }};
}

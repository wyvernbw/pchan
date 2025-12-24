use crate::dynarec::prelude::*;
use crate::mult;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct MULT {
    pub rs: u8,
    pub rt: u8,
}

impl MULT {
    pub fn new(rs: u8, rt: u8) -> Self {
        Self { rs, rt }
    }
}

impl Op for MULT {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> crate::cpu::ops::OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SPECIAL)
            .with_secondary(SecOp::MULT)
            .set_bits(21..26, self.rs as u32)
            .set_bits(16..21, self.rt as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        mult!(self, ctx, Opcode::Smulhi)
    }
}

impl Display for MULT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mult ${} ${}",
            REG_STR[self.rs as usize], REG_STR[self.rt as usize]
        )
    }
}

impl TryFrom<OpCode> for MULT {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value
            .as_primary(PrimeOp::SPECIAL)?
            .as_secondary(SecOp::MULT)?;
        Ok(MULT {
            rs: value.bits(21..26) as u8,
            rt: value.bits(16..21) as u8,
        })
    }
}

pub fn mult(rs: u8, rt: u8) -> OpCode {
    MULT { rs, rt }.into_opcode()
}

#[macro_export]
macro_rules! mult {
    ($self: expr, $ctx: expr, $hiopcode: expr) => {{
        use $crate::dynarec::prelude::*;
        // case 1: $rs = 0 or $rt = 0 => $hi:$lo=0
        if $self.rs == 0 || $self.rt == 0 {
            let (zero, loadzero) = $ctx.emit_get_zero();
            return EmitSummary::builder()
                .hi(zero)
                .lo(zero)
                .instructions([now(loadzero)])
                .build($ctx.fn_builder);
        }

        let (rs, loadrs) = $ctx.emit_get_register($self.rs);
        let (rt, loadrt) = $ctx.emit_get_register($self.rt);
        let (lo, imul) = $ctx.inst(|f| f.pure().Binary(Opcode::Imul, types::I32, rs, rt).0);
        let (hi, smulhi) = $ctx.inst(|f| f.pure().Binary($hiopcode, types::I32, rs, rt).0);

        // // Extend to 64-bit
        // let lo64 = fn_builder.ins().uextend(types::I64, lo);
        // let hi64 = fn_builder.ins().sextend(types::I64, hi);

        // // Shift high half into upper 32 bits
        // let hi64_shifted = fn_builder.ins().ishl_imm(hi64, 32);

        // // Combine high ad low halves
        // let full64 = fn_builder.ins().bor(hi64_shifted, lo64);
        EmitSummary::builder()
            .hi(hi)
            .lo(lo)
            .instructions([now(loadrs), now(loadrt), now(imul), now(smulhi)])
            .build($ctx.fn_builder)
    }};
}

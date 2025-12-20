use crate::dynarec::prelude::*;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SLTI {
    rt: u8,
    rs: u8,
    imm: i16,
}

impl SLTI {
    pub const fn new(rt: u8, rs: u8, imm: i16) -> Self {
        Self { rt, rs, imm }
    }
}

pub fn slti(rt: u8, rs: u8, imm: i16) -> OpCode {
    SLTI { rs, rt, imm }.into_opcode()
}

impl Display for SLTI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "slti ${} ${} {}",
            REG_STR[self.rt as usize],
            REG_STR[self.rs as usize],
            hex(self.imm)
        )
    }
}

impl TryFrom<OpCode> for SLTI {
    type Error = TryFromOpcodeErr;

    fn try_from(value: OpCode) -> Result<Self, Self::Error> {
        let value = value.as_primary(PrimeOp::SLTI)?;
        Ok(SLTI {
            imm: value.bits(0..16) as i16,
            rt: value.bits(16..21) as u8,
            rs: value.bits(21..26) as u8,
        })
    }
}

impl Op for SLTI {
    fn is_block_boundary(&self) -> Option<BoundaryType> {
        None
    }

    fn into_opcode(self) -> OpCode {
        OpCode::default()
            .with_primary(PrimeOp::SLTI)
            .set_bits(0..16, self.imm as i32 as u32)
            .set_bits(16..21, self.rt as u32)
            .set_bits(21..26, self.rs as u32)
    }

    fn emit_ir(&self, mut ctx: EmitCtx) -> EmitSummary {
        // x < 0u64 (u64::MIN) = false
        if self.imm == 0 {
            let (zero, loadzero) = ctx.emit_get_zero();
            return EmitSummary::builder()
                .instructions([now(loadzero)])
                .register_updates([(self.rt, zero)])
                .build(ctx.fn_builder);
        }

        // 0u64 < x = true
        if self.rs == 0 {
            let (one, loadone) = ctx.emit_get_one();
            return EmitSummary::builder()
                .instructions([now(loadone)])
                .register_updates([(self.rt, one)])
                .build(ctx.fn_builder);
        }

        icmpimm!(self, ctx, IntCC::SignedLessThan)
    }
}

#[macro_export]
macro_rules! icmpimm {
    ($self:expr, $ctx:expr, $compare:expr) => {{
        use $crate::dynarec::prelude::*;

        let (rs, loadrs) = $ctx.emit_get_register($self.rs);
        let (rt, icmpimm) = $ctx.inst(|f| {
            f.pure()
                .IntCompareImm(
                    Opcode::IcmpImm,
                    types::I32,
                    $compare,
                    Imm64::new($self.imm.into()),
                    rs,
                )
                .0
        });
        let (rt, uextend) = $ctx.inst(|f| f.pure().Unary(Opcode::Uextend, types::I32, rt).0);
        EmitSummary::builder()
            .instructions([now(loadrs), now(icmpimm), now(uextend)])
            .register_updates([($self.rt, rt)])
            .build($ctx.fn_builder)
    }};
}

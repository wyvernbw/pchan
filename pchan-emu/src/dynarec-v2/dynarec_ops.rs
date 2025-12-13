use enum_dispatch::enum_dispatch;

use crate::cpu::ops::{HaltBlock, OpCode, addiu::*};
use crate::dynarec_v2::Dynarec;
use crate::dynarec_v2::Reg;
use dynasm::dynasm;
use dynasmrt::DynasmApi;

pub trait ResultIntoInner {
    type IntoInner;
    fn into_inner(self) -> Self::IntoInner;
}

impl<T> ResultIntoInner for Result<T, T> {
    type IntoInner = T;

    fn into_inner(self) -> Self::IntoInner {
        match self {
            Ok(ok) => ok,
            Err(err) => err,
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Boundary {
    #[default]
    None = 0,
    Block,
    Function,
}

pub struct EmitCtx<'a> {
    pub dynarec: &'a mut Dynarec,
}

pub struct EmitSummary;

#[enum_dispatch(DecodedOpNew)]
pub trait DynarecOp {
    fn cycles(&self) -> u16 {
        1
    }

    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary;

    fn boundary(&self) -> Boundary;

    fn is_block_boundary(&self) -> bool {
        matches!(self.boundary(), Boundary::Block)
    }

    fn is_function_boundary(&self) -> bool {
        matches!(self.boundary(), Boundary::Function)
    }

    fn is_boundary(&self) -> bool {
        matches!(self.boundary(), Boundary::Block | Boundary::Function)
    }
}

#[enum_dispatch]
#[derive(Debug, Clone, Copy, Hash)]
pub enum DecodedOpNew {
    ILLEGAL(ILLEGAL),
    ADDIU(ADDIU),
    HaltBlock(HaltBlock),
}

#[derive(Debug, Clone, Copy, derive_more::Display, Hash, PartialEq, Eq)]
pub struct ILLEGAL;

impl DynarecOp for ILLEGAL {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        EmitSummary
    }

    fn boundary(&self) -> Boundary {
        Boundary::None
    }
}

impl DecodedOpNew {
    pub fn new(fields: OpCode) -> Self {
        let [op] = Self::decode([fields]);
        op
    }
    pub const fn illegal() -> Self {
        Self::ILLEGAL(ILLEGAL)
    }
    pub fn decode<const N: usize>(fields: [impl Into<OpCode>; N]) -> [Self; N] {
        fields.map(|fields| {
            let fields = fields.into();
            if fields == OpCode::HALT_FIELDS {
                return Self::HaltBlock(HaltBlock);
            }
            let opcode = fields.opcode();
            let rs = fields.rs();
            let rt = fields.rt();
            let rd = fields.rd();
            let funct = fields.funct();
            match (opcode, rs, rt, funct) {
                (0x8 | 0x9, _, _, _) => Self::ADDIU(ADDIU::new(rs, rt, fields.imm16())),
                _ => Self::illegal(),
            }
        })
    }
}
impl DynarecOp for ADDIU {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        // $rs = $zero case
        if self.rs == 0 {
            let loaded_rt = ctx.dynarec.emit_load_reg(self.rt, Reg::W(25));
            let rt = loaded_rt.into_inner();

            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; mov W(rt), self.imm as _
            );

            if loaded_rt.is_ok() {
                ctx.dynarec.mark_dirty(self.rt);
            } else {
                ctx.dynarec.emit_writeback(self.rt, rt);
            }
            return EmitSummary;
        }

        let rs = ctx.dynarec.emit_load_reg(self.rs, Reg::W(24)).into_inner();
        let loaded_rt = ctx.dynarec.emit_load_reg(self.rt, Reg::W(25));
        let rt = loaded_rt.into_inner();

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; add WSP(rt), WSP(rs), self.imm as _
        );

        if loaded_rt.is_ok() {
            ctx.dynarec.mark_dirty(self.rt);
        } else {
            ctx.dynarec.emit_writeback(self.rt, rt);
        }
        EmitSummary
    }

    fn boundary(&self) -> Boundary {
        Boundary::None
    }
}

impl DynarecOp for HaltBlock {
    fn emit<'a>(&self, _: EmitCtx<'a>) -> EmitSummary {
        EmitSummary
    }

    fn boundary(&self) -> Boundary {
        Boundary::Function
    }
}

use enum_dispatch::enum_dispatch;

#[cfg(test)]
use rstest::rstest;

use crate::cpu::ops::addiu::*;
use crate::cpu::ops::subu::*;
use crate::cpu::ops::{HaltBlock, OpCode};
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

    fn boundary(&self) -> Boundary {
        Boundary::None
    }

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
#[derive(Debug, Clone, Copy, Hash, strum::Display)]
pub enum DecodedOpNew {
    #[strum(transparent)]
    ILLEGAL(ILLEGAL),
    #[strum(transparent)]
    ADDIU(ADDIU),
    #[strum(transparent)]
    SUBU(SUBU),
    #[strum(transparent)]
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
                (0x0, _, _, 0x22 | 0x23) => Self::SUBU(SUBU::new(rd, rs, rt)),
                _ => Self::illegal(),
            }
        })
    }
}

impl DynarecOp for ADDIU {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        // $rt = $zero case
        if self.rt == 0 {
            return EmitSummary;
        }

        // map rt (previous value doesnt matter)
        let loaded_rt = ctx.dynarec.map_reg(self.rt, Reg::W(25));
        let rt = loaded_rt.into_inner();

        // $rs = $zero case
        if self.rs == 0 {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; mov W(rt), self.imm as _
            );

            ctx.dynarec.writeback(self.rt, loaded_rt);
            return EmitSummary;
        }

        // load $rs in order to compute $rs+imm
        let rs = ctx.dynarec.emit_load_reg(self.rs, Reg::W(24)).into_inner();

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; add WSP(rt), WSP(rs), self.imm as _
        );

        ctx.dynarec.writeback(self.rt, loaded_rt);

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

impl DynarecOp for SUBU {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary;
        };

        let rd = ctx.dynarec.map_reg(self.rd, Reg::W(24));
        let rs = ctx.dynarec.emit_load_reg(self.rs, Reg::W(25)).into_inner();
        let rt = ctx.dynarec.emit_load_reg(self.rt, Reg::W(26)).into_inner();

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; sub W(rd.into_inner()), W(rs), W(rt)
        );

        ctx.dynarec.writeback(self.rd, rd);

        EmitSummary
    }
}

#[cfg(test)]
#[rstest]
#[case(10, 2, 8)]
fn test_subu(#[case] a: u32, #[case] b: u32, #[case] expected: u32) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = a;
    emu.cpu.gpr[11] = b;
    emu.write_many(0x0, &program([subu(12, 10, 11), OpCode(69420)]));
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[12], expected);
    assert_eq!(emu.cpu.d_clock, 1);
    assert_eq!(emu.cpu.pc, 0x4);
    Ok(())
}

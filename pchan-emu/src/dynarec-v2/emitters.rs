use crate::dynarec_v2::Guest;
use std::num::NonZeroU8;

use enum_dispatch::enum_dispatch;

#[cfg(test)]
use rstest::rstest;

use crate::cpu::ops::addiu::*;
use crate::cpu::ops::addu::*;
use crate::cpu::ops::and::*;
use crate::cpu::ops::lb::*;
use crate::cpu::ops::lbu::*;
use crate::cpu::ops::lh::*;
use crate::cpu::ops::lhu::*;
use crate::cpu::ops::lw::*;
use crate::cpu::ops::nor::*;
use crate::cpu::ops::or::*;
use crate::cpu::ops::sb::*;
use crate::cpu::ops::sh::*;
use crate::cpu::ops::subu::*;
use crate::cpu::ops::sw::*;
use crate::cpu::ops::xor::*;

use crate::cpu::ops::{HaltBlock, OpCode};
use crate::dynarec_v2::Dynarec;
use crate::dynarec_v2::LoadedReg;
use crate::dynarec_v2::Reg;
#[cfg(test)]
use crate::memory;
use crate::memory::ext;
use dynasm::dynasm;
use dynasmrt::DynasmApi;
use dynasmrt::DynasmLabelApi;

#[derive(Default, Debug, Clone, Copy)]
pub enum Boundary {
    #[default]
    None = 0,
    Block,
    Function,
}

#[derive(Debug)]
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
#[derive(Debug, Clone, Copy, Hash, derive_more::Display)]
pub enum DecodedOpNew {
    ILLEGAL(ILLEGAL),
    ADDIU(ADDIU),
    ADDU(ADDU),
    SUBU(SUBU),
    AND(AND),
    OR(OR),
    XOR(XOR),
    NOR(NOR),
    HaltBlock(HaltBlock),
    SB(SB),
    SH(SH),
    SW(SW),
    LB(LB),
    LBU(LBU),
    LH(LH),
    LHU(LHU),
    LW(LW),
}

#[derive(Debug, Clone, Copy, derive_more::Display, Hash, PartialEq, Eq)]
pub struct ILLEGAL;

impl DynarecOp for ILLEGAL {
    fn emit<'a>(&self, _: EmitCtx<'a>) -> EmitSummary {
        EmitSummary
    }

    fn boundary(&self) -> Boundary {
        Boundary::Function
    }
}

impl DecodedOpNew {
    pub fn new(fields: OpCode) -> Self {
        let [op] = Self::decode([fields]);
        op
    }
    pub const fn is_illegal(&self) -> bool {
        matches!(self, Self::ILLEGAL(_))
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
                (0x0, _, _, 0x20 | 0x21) => Self::ADDU(ADDU::new(rd, rs, rt)),
                (0x0, _, _, 0x22 | 0x23) => Self::SUBU(SUBU::new(rd, rs, rt)),
                (0x0, _, _, 0x24) => Self::AND(AND::new(rd, rs, rt)),
                (0x0, _, _, 0x25) => Self::OR(OR::new(rd, rs, rt)),
                (0x0, _, _, 0x26) => Self::XOR(XOR::new(rd, rs, rt)),
                (0x0, _, _, 0x27) => Self::NOR(NOR::new(rd, rs, rt)),
                (0x8 | 0x9, _, _, _) => Self::ADDIU(ADDIU::new(rs, rt, fields.imm16())),
                (0x20, _, _, _) => Self::LB(LB::new(rt, rs, fields.imm16())),
                (0x21, _, _, _) => Self::LH(LH::new(rt, rs, fields.imm16())),
                (0x22, _, _, _) => todo!("lwl"),
                (0x23, _, _, _) => Self::LW(LW::new(rt, rs, fields.imm16())),
                (0x24, _, _, _) => Self::LBU(LBU::new(rt, rs, fields.imm16())),
                (0x25, _, _, _) => Self::LHU(LHU::new(rt, rs, fields.imm16())),
                (0x26, _, _, _) => todo!("lwr"),
                (0x27, _, _, _) => Self::illegal(),
                (0x28, _, _, _) => Self::SB(SB::new(rt, rs, fields.imm16())),
                (0x29, _, _, _) => Self::SH(SH::new(rt, rs, fields.imm16())),
                (0x2A, _, _, _) => todo!("swl"),
                (0x2B, _, _, _) => Self::SW(SW::new(rt, rs, fields.imm16())),
                _ => Self::illegal(),
            }
        })
    }
}

impl DynarecOp for ADDIU {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        // $rt = $zero case
        if self.rt == 0 {
            return EmitSummary;
        }

        let rt = ctx.dynarec.alloc_reg(self.rt);

        // $rs = $zero case
        if self.rs == 0 {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; mov W(*rt), self.imm as _
            );

            rt.restore(ctx.dynarec);
            return EmitSummary;
        }

        let rs = ctx.dynarec.emit_load_reg(self.rs);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; add WSP(*rt), WSP(*rs), self.imm as _
        );

        ctx.dynarec.mark_dirty(self.rt);

        rs.restore(ctx.dynarec);
        rt.restore(ctx.dynarec);

        EmitSummary
    }

    fn boundary(&self) -> Boundary {
        Boundary::None
    }
}

#[cfg(test)]
#[rstest]
#[case(10, 2, 12)]
fn test_addiu(#[case] a: u32, #[case] b: u32, #[case] expected: u32) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = a;
    emu.write_many(0x0, &program([addiu(12, 10, b as i16), OpCode(69420)]));
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[12], expected);
    assert_eq!(emu.cpu.d_clock, 1);
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

impl DynarecOp for HaltBlock {
    fn emit<'a>(&self, _: EmitCtx<'a>) -> EmitSummary {
        EmitSummary
    }

    fn cycles(&self) -> u16 {
        0
    }

    fn boundary(&self) -> Boundary {
        Boundary::Function
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
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

fn emit_store(
    ctx: EmitCtx,
    rt: u8,
    rs: u8,
    imm: i16,
    func_call: impl Fn(&mut EmitCtx) + 'static,
) -> EmitSummary {
    ctx.dynarec.emit_load_temp_reg(rs, Reg::W(1));
    ctx.dynarec.emit_load_temp_reg(rt, Reg::W(2));

    if imm > 0xfff {
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; mov w3, ext::sign(imm) as _
            ; add w1, w1, w3
            ; stp w1, w2, [sp, #-16]!
        );
    } else {
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; add w1, w1, ext::sign(imm) as _
            ; stp w1, w2, [sp, #-16]!
        );
    }

    ctx.dynarec.set_delay_slot(move |mut ctx| {
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; ldp w1, w2, [sp], #16
            // FIXME: store allocated temp registers as well
            // need to track caller saved registers in regalloc
            // we have to store x0 since the function call will clobber it
            ; stp x0, x1, [sp, #-16]!
            // call to function
            ;; func_call(&mut ctx)
            ; ldp x0, x1, [sp], #16

        );
        EmitSummary
    });

    EmitSummary
}

impl DynarecOp for SB {
    #[cfg(target_arch = "aarch64")]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, ->write8v2
                ; blr x3
            )
        })
    }

    fn cycles(&self) -> u16 {
        2
    }
}

impl DynarecOp for SH {
    #[cfg(target_arch = "aarch64")]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, ->write16v2
                ; blr x3
            )
        })
    }
    fn cycles(&self) -> u16 {
        2
    }
}

impl DynarecOp for SW {
    #[cfg(target_arch = "aarch64")]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, ->write32v2
                ; blr x3
            );
        })
    }
    fn cycles(&self) -> u16 {
        2
    }
}

#[cfg(test)]
#[rstest]
#[case::sb(69u8, sb(10, 11, 2))]
#[case::sh(69u8, sh(10, 11, 2))]
#[case::sw(69u8, sw(10, 11, 2))]
fn test_stores<T>(#[case] value: T, #[case] instr: OpCode) -> color_eyre::Result<()>
where
    T: Copy + std::fmt::Debug + PartialEq,
    T: const memory::Extend<ext::Zero, Out = u32>,
{
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2, memory::ext};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = ext::zero(value);
    emu.cpu.gpr[11] = 0xf;
    emu.write_many(0x0, &program([instr, OpCode(69420)]));

    PipelineV2::new(&emu).run_once(&mut emu)?;

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);
    assert_eq!(emu.read::<T, ext::Zero>(0xf + 2), ext::zero(value));

    tracing::info!("returning from test...");
    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::useless_conversion)]
fn emit_load(
    ctx: EmitCtx,
    rt: u8,
    rs: u8,
    imm: i16,
    func_call: impl Fn(&mut EmitCtx) + 'static,
) -> EmitSummary {
    ctx.dynarec.emit_load_temp_reg(rs, Reg::W(1));
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ; add w1, w1, ext::sign(imm) as _
        ; str w1, [sp, #-16]!
    );

    ctx.dynarec.set_delay_slot(move |mut ctx| {
        let rta = ctx.dynarec.alloc_reg(rt);
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; ldr w1, [sp], #16
            ; stp x0, x1, [sp, #-16]!
            ; stp x2, x3, [sp, #-16]!
            ; stp x4, x5, [sp, #-16]!
            ; stp x6, x7, [sp, #-16]!
            ;; func_call(&mut ctx)
            ; ldp x6, x7, [sp], #16
            ; ldp x4, x5, [sp], #16
            ; mov W(*rta), w0
            ; ldp x2, x3, [sp], #16
            ; ldp x0, x1, [sp], #16
        );
        ctx.dynarec.mark_dirty(rt);

        rta.restore(ctx.dynarec);
        EmitSummary
    });

    EmitSummary
}

impl DynarecOp for LB {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_load(ctx, self.rt, self.rs, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, ->readi8v2
                ; blr x3
            )
        })
    }
    fn cycles(&self) -> u16 {
        2
    }
}

impl DynarecOp for LBU {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_load(ctx, self.rt, self.rs, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, ->readu8v2
                ; blr x3
            )
        })
    }
    fn cycles(&self) -> u16 {
        2
    }
}

impl DynarecOp for LH {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_load(ctx, self.rt, self.rs, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, ->readi16v2
                ; blr x3
            )
        })
    }
    fn cycles(&self) -> u16 {
        2
    }
}

impl DynarecOp for LHU {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_load(ctx, self.rt, self.rs, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, ->readu16v2
                ; blr x3
            )
        })
    }
    fn cycles(&self) -> u16 {
        2
    }
}

impl DynarecOp for LW {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_load(ctx, self.rt, self.rs, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, ->read32v2
                ; blr x3
            )
        })
    }
    fn cycles(&self) -> u16 {
        2
    }
}

#[cfg(test)]
#[rstest]
#[case::lb(69, lb, 69)]
#[case::lb(-1i32 as u32, lb, -1i32 as u32)]
#[case::lbu(69, lbu, 69)]
#[case::lbu(0xFF, lbu, 0xFF)]
#[case::lh(69, lh, 69)]
#[case::lh(-1i32 as u32, lh, -1i32 as u32)]
#[case::lhu(69, lhu, 69)]
#[case::lhu(0xFFFF, lhu, 0xFFFF)]
#[case::lw(69, lw, 69)]
#[case::lw(-1i32 as u32, lw, -1i32 as u32)]
fn test_loads(
    #[case] value: u32,
    #[case] instr: impl Fn(u8, u8, i16) -> OpCode,
    #[case] expected: u32,
) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[11] = 0x100;
    emu.write(0x100 + 2, value);
    emu.write_many(0x0, &program([instr(10, 11, 2), OpCode(69420)]));

    PipelineV2::new(&emu).run_once(&mut emu)?;

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);
    assert_eq!(emu.cpu.gpr[10], expected);

    tracing::info!("returning from test...");
    Ok(())
}

#[cfg(test)]
#[rstest]
#[case::lb(lb)]
#[case::lbu(lbu)]
#[case::lh(lh)]
#[case::lhu(lhu)]
#[case::lw(lw)]
fn test_load_delay(#[case] instr: impl Fn(u8, u8, i16) -> OpCode) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[11] = 0x100;
    emu.write(0x100 + 2, 69);
    emu.write_many(
        0x0,
        &program([
            instr(10, 11, 2),
            addiu(12, 10, 420),
            addiu(13, 10, 420),
            OpCode(69420),
        ]),
    );

    PipelineV2::new(&emu).run_once(&mut emu)?;

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.d_clock, 4);
    assert_eq!(emu.cpu.pc, 0x10);
    assert_eq!(emu.cpu.gpr[10], 69);
    assert_eq!(emu.cpu.gpr[12], 420);
    assert_eq!(emu.cpu.gpr[13], 69 + 420);

    tracing::info!("returning from test...");
    Ok(())
}

fn emit_alu(
    mut ctx: EmitCtx,
    rd: u8,
    rs: u8,
    rt: u8,
    alu_op: impl Fn(&mut EmitCtx, &LoadedReg, &LoadedReg, &LoadedReg),
) -> EmitSummary {
    let rda = ctx.dynarec.alloc_reg(rd);
    let rsa = ctx.dynarec.emit_load_reg(rs);
    let rta = ctx.dynarec.emit_load_reg(rt);

    #[cfg(target_arch = "aarch64")]
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ;; alu_op(&mut ctx, &rda, &rsa, &rta)
    );

    ctx.dynarec.mark_dirty(rd);

    rta.restore(ctx.dynarec);
    rsa.restore(ctx.dynarec);
    rda.restore(ctx.dynarec);

    EmitSummary
}

enum EitherZero {
    None,
    Both,
    One(NonZeroU8),
}

const fn either_zero(rs: u8, rt: u8) -> EitherZero {
    unsafe {
        match (rs, rt) {
            (0, 0) => EitherZero::Both,
            (a, 0) => EitherZero::One(NonZeroU8::new_unchecked(a)),
            (0, b) => EitherZero::One(NonZeroU8::new_unchecked(b)),
            _ => EitherZero::None,
        }
    }
}

impl DynarecOp for SUBU {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary;
        };

        if self.rt == 0 {
            let rd = ctx.dynarec.alloc_reg(self.rd);
            ctx.dynarec.emit_load_temp_reg(self.rs, Reg::W(1));

            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; mov W(*rd), w1
            );

            ctx.dynarec.mark_dirty(self.rd);
            rd.restore(ctx.dynarec);

            return EmitSummary;
        }

        emit_alu(ctx, self.rd, self.rs, self.rt, move |ctx, rd, rs, rt| {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; sub W(**rd), W(**rs), W(**rt)
            )
        })
    }
}

impl DynarecOp for ADDU {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary;
        };

        match either_zero(self.rs, self.rt) {
            EitherZero::None => emit_alu(ctx, self.rd, self.rs, self.rt, |ctx, rd, rs, rt| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; add W(**rd), W(**rs), W(**rt)
                )
            }),
            EitherZero::Both => ctx.dynarec.emit_zero(self.rd),
            EitherZero::One(non_zero) => ctx
                .dynarec
                .emit_load_and_move_into(self.rd, non_zero.into()),
        }
    }
}

impl DynarecOp for AND {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary;
        }

        match either_zero(self.rs, self.rt) {
            EitherZero::None => emit_alu(ctx, self.rd, self.rs, self.rt, move |ctx, rd, rs, rt| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; and W(**rd), W(**rs), W(**rt)
                )
            }),
            EitherZero::Both | EitherZero::One(_) => ctx.dynarec.emit_zero(self.rd),
        }
    }
}

impl DynarecOp for OR {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary;
        }

        match either_zero(self.rs, self.rt) {
            EitherZero::None => emit_alu(ctx, self.rd, self.rs, self.rt, move |ctx, rd, rs, rt| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; orr W(**rd), W(**rs), W(**rt)
                )
            }),
            EitherZero::Both => ctx.dynarec.emit_zero(self.rd),
            EitherZero::One(non_zero) => ctx
                .dynarec
                .emit_load_and_move_into(self.rd, non_zero.into()),
        }
    }
}

impl DynarecOp for XOR {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary;
        }

        match either_zero(self.rs, self.rt) {
            EitherZero::None => emit_alu(ctx, self.rd, self.rs, self.rt, move |ctx, rd, rs, rt| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; eor W(**rd), W(**rs), W(**rt)
                )
            }),
            EitherZero::Both => ctx.dynarec.emit_zero(self.rd),
            EitherZero::One(non_zero) => ctx
                .dynarec
                .emit_load_and_move_into(self.rd, non_zero.into()),
        }
    }
}

impl DynarecOp for NOR {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary;
        }
        match either_zero(self.rs, self.rt) {
            EitherZero::None => emit_alu(ctx, self.rd, self.rs, self.rt, move |ctx, rd, rs, rt| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; orr W(**rd), W(**rs), W(**rt)
                    ; mvn W(**rd), W(**rd)
                )
            }),
            EitherZero::Both => ctx.dynarec.emit_immediate_large(self.rd, 0xffff_ffff),
            EitherZero::One(non_zero) => {
                let target = self.rd;
                let reg = non_zero.into();
                let rd = ctx.dynarec.alloc_reg(target);
                ctx.dynarec.emit_load_temp_reg(reg, Reg::W(1));
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; mov W(*rd), w1
                    ; mvn W(*rd), W(*rd)
                );
                ctx.dynarec.mark_dirty(target);
                rd.restore(ctx.dynarec);
                EmitSummary
            }
        }
    }
}

#[cfg(test)]
#[rstest]
#[case::subu_01(subu, (12, 93), (10, 100), (11, 7))]
#[case::subu_02(subu, (12, 100), (10, 100), (0, 0))]
#[case::addu_01(addu, (12, 21), (10, 19), (11, 2))]
#[case::addu_02(addu, (0, 0), (10, 69), (11, 15))]
#[case::addu_03(addu, (12, 0), (0, 0), (0, 0))]
#[case::addu_04(addu, (12, 15), (10, 15), (0, 0))]
#[case::and_01(and, (12, 0b1010), (10, 0b1111), (11, 0b1010))]
#[case::and_02(and, (12, 0), (0, 0), (11, 0b1010))]
#[case::and_03(and, (12, 0), (11, 0b1010), (0, 0))]
#[case::and_04(and, (12, 0), (0, 0), (0, 0))]
#[case::or_01(or, (12, 0b1111), (10, 0b1010), (11, 0b0101))]
#[case::or_02(or, (12, 0b1010), (10, 0b1010), (0, 0))]
#[case::or_03(or, (12, 0b0101), (0, 0), (11, 0b0101))]
#[case::or_04(or, (12, 0), (0, 0), (0, 0))]
#[case::or_05(or, (12, 0xFFFFFFFF), (10, 0xAAAAAAAA), (11, 0x55555555))]
#[case::xor_01(xor, (12, 0b0101), (10, 0b1111), (11, 0b1010))]
#[case::xor_02(xor, (12, 0b1010), (10, 0b1010), (0, 0))]
#[case::xor_03(xor, (12, 0b0101), (0, 0), (11, 0b0101))]
#[case::xor_04(xor, (12, 0), (0, 0), (0, 0))]
#[case::xor_05(xor, (12, 0), (10, 0xAAAAAAAA), (11, 0xAAAAAAAA))]
#[case::xor_06(xor, (12, 0xFFFFFFFF), (10, 0xAAAAAAAA), (11, 0x55555555))]
#[case::nor_01(nor, (12, !0b1111), (10, 0b1010), (11, 0b0101))]
#[case::nor_02(nor, (12, !0b1010), (10, 0b1010), (0, 0))]
#[case::nor_03(nor, (12, !0b0101), (0, 0), (11, 0b0101))]
#[case::nor_04(nor, (12, 0xFFFFFFFF), (0, 0), (0, 0))]
#[case::nor_05(nor, (12, 0), (10, 0xAAAAAAAA), (11, 0x55555555))]
#[case::nor_06(nor, (12, 0x55555555), (10, 0xAAAAAAAA), (11, 0xAAAAAAAA))]
fn test_alu_reg(
    #[case] instr: impl Fn(u8, u8, u8) -> OpCode,
    #[case] expected: (Guest, u32),
    #[case] a: (Guest, u32),
    #[case] b: (Guest, u32),
) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    if expected.0 != 0 {
        emu.cpu.gpr[expected.0 as usize] = 1231123;
    }
    emu.cpu.gpr[a.0 as usize] = a.1;
    emu.cpu.gpr[b.0 as usize] = b.1;
    emu.write_many(0x0, &program([instr(expected.0, a.0, b.0), OpCode(69420)]));
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[expected.0 as usize], expected.1);
    assert_eq!(emu.cpu.d_clock, 1);
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

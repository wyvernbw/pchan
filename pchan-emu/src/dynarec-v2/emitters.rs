use crate::Emu;
use crate::cpu;
use crate::cpu::Cpu;
use crate::cpu::ops::NOP;
use crate::dynarec_v2::Guest;
use std::num::NonZeroU8;

use bon::Builder;
use enum_dispatch::enum_dispatch;

#[cfg(test)]
use rstest::rstest;

use crate::cpu::ops::addiu::*;
use crate::cpu::ops::addu::*;
use crate::cpu::ops::and::*;
use crate::cpu::ops::andi::*;
use crate::cpu::ops::beq::*;
use crate::cpu::ops::bgez::*;
use crate::cpu::ops::bgtz::*;
use crate::cpu::ops::blez::*;
use crate::cpu::ops::bltz::*;
use crate::cpu::ops::bne::*;
use crate::cpu::ops::j::*;
use crate::cpu::ops::jal::*;
use crate::cpu::ops::jalr::*;
use crate::cpu::ops::jr::*;
use crate::cpu::ops::lb::*;
use crate::cpu::ops::lbu::*;
use crate::cpu::ops::lh::*;
use crate::cpu::ops::lhu::*;
use crate::cpu::ops::lui::*;
use crate::cpu::ops::lw::*;
use crate::cpu::ops::mtc::*;
use crate::cpu::ops::nor::*;
use crate::cpu::ops::or::*;
use crate::cpu::ops::ori::*;
use crate::cpu::ops::sb::*;
use crate::cpu::ops::sh::*;
use crate::cpu::ops::sll::*;
use crate::cpu::ops::sllv::*;
use crate::cpu::ops::sra::*;
use crate::cpu::ops::srav::*;
use crate::cpu::ops::srl::*;
use crate::cpu::ops::srlv::*;
use crate::cpu::ops::subu::*;
use crate::cpu::ops::sw::*;
use crate::cpu::ops::xor::*;
use crate::cpu::ops::xori::*;

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

#[derive(Debug)]
pub struct EmitCtx<'a> {
    pub dynarec: &'a mut Dynarec,
    pub pc: u32,
}

const MAX_SCRATCH_REG: u8 = 3;

impl<'a> EmitCtx<'a> {
    fn parity(&self) -> impl Fn(u8) -> u8 + 'static {
        let m = (self.pc >> 2) % 2;
        move |reg| reg + m as u8 * MAX_SCRATCH_REG
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    #[default]
    None,
    Soft,
    Hard,
}

#[derive(Debug, Default, Builder)]
pub struct EmitSummary {
    pub pc_updated: bool,
}

#[enum_dispatch(DecodedOpNew)]
pub trait DynarecOp {
    fn cycles(&self) -> u16 {
        1
    }

    fn hazard(&self) -> u16 {
        0
    }

    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary;

    fn boundary(&self) -> Boundary {
        Boundary::None
    }

    fn is_boundary(&self) -> bool {
        matches!(self.boundary(), Boundary::Soft | Boundary::Hard)
    }

    fn is_hard_boundary(&self) -> bool {
        matches!(self.boundary(), Boundary::Hard)
    }
}

#[enum_dispatch]
#[derive(Debug, Clone, Copy, Hash, derive_more::Display)]
pub enum DecodedOpNew {
    NOP(NOP),
    ILLEGAL(ILLEGAL),
    SLL(SLL),
    SRL(SRL),
    SRA(SRA),
    SLLV(SLLV),
    SRLV(SRLV),
    SRAV(SRAV),
    JR(JR),
    JALR(JALR),
    ADDIU(ADDIU),
    ADDU(ADDU),
    SUBU(SUBU),
    AND(AND),
    OR(OR),
    XOR(XOR),
    BLTZ(BLTZ),
    BGEZ(BGEZ),
    J(J),
    JAL(JAL),
    BLEZ(BLEZ),
    BGTZ(BGTZ),
    NOR(NOR),
    BEQ(BEQ),
    BNE(BNE),
    HaltBlock(HaltBlock),
    SB(SB),
    SH(SH),
    SW(SW),
    ANDI(ANDI),
    ORI(ORI),
    XORI(XORI),
    LUI(LUI),
    MTCn(MTCn),
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
        EmitSummary::default()
    }

    fn boundary(&self) -> Boundary {
        Boundary::Hard
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
            if fields == OpCode::NOP_FIELDS {
                return Self::NOP(NOP);
            }
            if fields == OpCode::HALT_FIELDS {
                return Self::HaltBlock(HaltBlock);
            }
            let opcode = fields.opcode();
            let rs = fields.rs();
            let rt = fields.rt();
            let rd = fields.rd();
            let funct = fields.funct();
            match (opcode, rs, rt, funct) {
                (0x0, _, _, 0x0) => Self::SLL(SLL::new(rd, rt, fields.shamt() as i8)),
                (0x0, _, _, 0x1) => Self::illegal(),
                (0x0, _, _, 0x2) => Self::SRL(SRL::new(rd, rt, fields.shamt() as i8)),
                (0x0, _, _, 0x3) => Self::SRA(SRA::new(rd, rt, fields.shamt() as i8)),
                (0x0, _, _, 0x4) => Self::SLLV(SLLV::new(rd, rt, rs)),
                (0x0, _, _, 0x5) => Self::illegal(),
                (0x0, _, _, 0x6) => Self::SRLV(SRLV::new(rd, rt, rs)),
                (0x0, _, _, 0x7) => Self::SRAV(SRAV::new(rd, rt, rs)),
                (0x0, _, _, 0x8) => Self::JR(JR::new(rs)),
                (0x0, _, _, 0x9) => Self::JALR(JALR::new(rd, rs)),
                (0x0, _, _, 0xA) => Self::illegal(),
                (0x0, _, _, 0xB) => Self::illegal(),
                (0x0, _, _, 0xC) => todo!("syscall"),
                (0x0, _, _, 0xD) => todo!("brk"),
                (0x0, _, _, 0xE) => Self::illegal(),
                (0x0, _, _, 0xF) => Self::illegal(),
                (0x0, _, _, 0x10) => todo!("mfhi"),
                (0x0, _, _, 0x11) => todo!("mthi"),
                (0x0, _, _, 0x12) => todo!("mflo"),
                (0x0, _, _, 0x13) => todo!("mtlo"),
                (0x0, _, _, 0x14..=0x17) => Self::illegal(),
                (0x0, _, _, 0x18) => todo!("mult"),
                (0x0, _, _, 0x19) => todo!("multu"),
                (0x0, _, _, 0x1A) => todo!("div"),
                (0x0, _, _, 0x1B) => todo!("divu"),
                (0x0, _, _, 0x1C..=0x1F) => Self::illegal(),
                (0x0, _, _, 0x20 | 0x21) => Self::ADDU(ADDU::new(rd, rs, rt)),
                (0x0, _, _, 0x22 | 0x23) => Self::SUBU(SUBU::new(rd, rs, rt)),
                (0x0, _, _, 0x24) => Self::AND(AND::new(rd, rs, rt)),
                (0x0, _, _, 0x25) => Self::OR(OR::new(rd, rs, rt)),
                (0x0, _, _, 0x26) => Self::XOR(XOR::new(rd, rs, rt)),
                (0x0, _, _, 0x27) => Self::NOR(NOR::new(rd, rs, rt)),
                (0x0, _, _, 0x28..=0x29) => Self::illegal(),
                (0x0, _, _, 0x2A) => todo!("slt"),
                (0x0, _, _, 0x2B) => todo!("sltu"),
                (0x0, _, _, 0x2C..) => Self::illegal(),
                (0x1, _, 0x0, _) => Self::BLTZ(BLTZ::new(rs, fields.imm16())),
                (0x1, _, 0x1, _) => Self::BGEZ(BGEZ::new(rs, fields.imm16())),
                (0x1, _, 0x10, _) => todo!("bltzal"),
                (0x1, _, 0x11, _) => todo!("bgezal"),
                // * TODO: bltz and bgez dupes * //
                (0x2, _, _, _) => Self::J(J::new(fields.imm26())),
                (0x3, _, _, _) => Self::JAL(JAL::new(fields.imm26())),
                (0x4, _, _, _) => Self::BEQ(BEQ::new(rs, rt, fields.imm16())),
                (0x5, _, _, _) => Self::BNE(BNE::new(rs, rt, fields.imm16())),
                (0x6, _, _, _) => Self::BLEZ(BLEZ::new(rs, fields.imm16())),
                (0x7, _, _, _) => Self::BGTZ(BGTZ::new(rs, fields.imm16())),
                (0x8 | 0x9, _, _, _) => Self::ADDIU(ADDIU::new(rs, rt, fields.imm16())),
                (0xA, _, _, _) => todo!("slti"),
                (0xB, _, _, _) => todo!("sltiu"),
                (0xC, _, _, _) => Self::ANDI(ANDI::new(rs, rt, fields.imm16())),
                (0xD, _, _, _) => Self::ORI(ORI::new(rs, rt, fields.imm16())),
                (0xE, _, _, _) => Self::XORI(XORI::new(rs, rt, fields.imm16())),
                (0xF, _, _, _) => Self::LUI(LUI::new(rt, fields.imm16())),
                (0x10, 0x10, _, 0x10) => todo!("rfe"),
                (0x10..=0x13, 0x0, _, 0x0) => todo!("mfcn"),
                (0x10..=0x13, 0x2, _, 0x0) => todo!("cfcn"),
                (0x10..=0x13, 0x4, _, 0x0) => Self::MTCn(MTCn::new(fields.cop(), rt, rd)),
                (0x10..=0x13, 0x8, 0, _) => todo!("bcnf"),
                (0x10..=0x13, 0x8, 1, _) => todo!("bcnt"),
                (0x10..=0x13, 0x6, _, 0x0) => todo!("ctcn"),
                (0x10..=0x13, 0x10..=0x1F, _, _) => todo!("cop{} imm25", fields.cop()),
                (0x14..=0x1F, _, _, _) => Self::illegal(),
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
                (0x2C..=0x2D, _, _, _) => Self::illegal(),
                (0x2E, _, _, _) => todo!("swr"),
                (0x2F, _, _, _) => Self::illegal(),
                (0x30..=0x33, _, _, _) => todo!("lwcn"),
                (0x34..=0x37, _, _, _) => Self::illegal(),
                (0x38..=0x3B, _, _, _) => todo!("swcn"),
                (0x3C..=0x3F, _, _, _) => Self::illegal(),
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
            return EmitSummary::default();
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

            ctx.dynarec.mark_dirty(self.rt);
            rt.restore(ctx.dynarec);
            return EmitSummary::default();
        }

        let rs = ctx.dynarec.emit_load_reg(self.rs);

        if (self.imm as u16) < 4096 {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; add WSP(*rt), WSP(*rs), self.imm as _
            );
        } else if self.imm < 0 {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; movz w1, ext::zero(self.imm)    // Load as unsigned
                ; sxth w1, w1                   // Sign-extend to 32-bit
                ; add WSP(*rt), WSP(*rs), w1
            );
        } else {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; mov w1, self.imm as _
                ; add WSP(*rt), WSP(*rs), w1
            );
        }

        ctx.dynarec.mark_dirty(self.rt);

        rs.restore(ctx.dynarec);
        rt.restore(ctx.dynarec);

        EmitSummary::default()
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
    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

impl DynarecOp for HaltBlock {
    fn emit<'a>(&self, _: EmitCtx<'a>) -> EmitSummary {
        EmitSummary::default()
    }

    fn cycles(&self) -> u16 {
        1
    }

    fn boundary(&self) -> Boundary {
        Boundary::Hard
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
    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

fn emit_call(ctx: &mut EmitCtx, emitter: impl Fn(&mut Dynarec)) {
    #[cfg(target_arch = "aarch64")]
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ;; let saved = ctx.dynarec.emit_save_volatile_registers()
        ;; emitter(ctx.dynarec)
        ;; ctx.dynarec.emit_restore_saved_registers(saved.into_iter())
    );
}

#[allow(clippy::useless_conversion)]
fn emit_store(
    ctx: EmitCtx,
    rt: u8,
    rs: u8,
    imm: i16,
    func_call: impl Fn(&mut EmitCtx) + 'static,
) -> EmitSummary {
    let s = ctx.parity();
    ctx.dynarec.emit_load_temp_reg(rs, Reg::W(1));
    ctx.dynarec.emit_load_temp_reg(rt, Reg::W(2));

    if imm > 0xfff {
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; mov w3, ext::sign(imm) as _
            ; add w1, w1, w3
            // ; stp w1, w2, [sp, #-16]!
            ; fmov S(s(9)), w1
            ; fmov S(s(10)), w2
        );
    } else {
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; add w1, w1, ext::sign(imm) as _
            // ; stp w1, w2, [sp, #-16]!
            ; fmov S(s(9)), w1
            ; fmov S(s(10)), w2
        );
    }

    ctx.dynarec.set_delay_slot(move |mut ctx| {
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            // FIXME: store allocated temp registers as well
            // need to track caller saved registers in regalloc
            // we have to store x0 since the function call will clobber it
            // call to function
            ;; let saved = ctx.dynarec.emit_save_volatile_registers()
            ; fmov w1, S(s(9))
            ; fmov w2, S(s(10))
            ;; func_call(&mut ctx)
            ;; ctx.dynarec.emit_restore_saved_registers(saved.into_iter())

        );
        EmitSummary::default()
    });

    EmitSummary::default()
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

    assert_eq!(emu.cpu.d_clock, 3);
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
    let s = ctx.parity();
    ctx.dynarec.emit_load_temp_reg(rs, Reg::W(1));
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ; add w1, w1, ext::sign(imm) as _
        // ; str w1, [sp, #-16]!
        ; fmov S(s(8)), w1
    );

    ctx.dynarec.set_delay_slot(move |mut ctx| {
        let rta = ctx.dynarec.alloc_reg(rt);
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            // ; ldr w1, [sp], #16
            ;; let saved = ctx.dynarec.emit_save_volatile_registers()
            ; fmov w1, S(s(8))
            ;; func_call(&mut ctx)
            ; fmov S(s(8)), w0 // place return value in s8+
            ;; ctx.dynarec.emit_restore_saved_registers(saved.into_iter())
            ; fmov W(*rta), S(s(8))
        );
        ctx.dynarec.mark_dirty(rt);

        rta.restore(ctx.dynarec);
        EmitSummary::default()
    });

    EmitSummary::default()
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
        3
    }
    fn hazard(&self) -> u16 {
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
        3
    }
    fn hazard(&self) -> u16 {
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
        3
    }
    fn hazard(&self) -> u16 {
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
        3
    }
    fn hazard(&self) -> u16 {
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
        3
    }
    fn hazard(&self) -> u16 {
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

    assert_eq!(emu.cpu.d_clock, 3);
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

    assert_eq!(emu.cpu.d_clock, 5);
    assert_eq!(emu.cpu.pc, 0x10);
    assert_eq!(emu.cpu.gpr[10], 69);
    assert_eq!(emu.cpu.gpr[12], 420);
    assert_eq!(emu.cpu.gpr[13], 69 + 420);

    tracing::info!("returning from test...");
    Ok(())
}

#[derive(Debug, Clone, derive_more::Deref)]
#[deref(forward)]
pub struct Rd<'a>(&'a LoadedReg);
#[derive(Debug, Clone, derive_more::Deref)]
#[deref(forward)]
pub struct Rs<'a>(&'a LoadedReg);
#[derive(Debug, Clone, derive_more::Deref)]
#[deref(forward)]
pub struct Rt<'a>(&'a LoadedReg);

pub struct AluRegs<'a> {
    rd: Rd<'a>,
    rs: Rs<'a>,
    rt: Rt<'a>,
}

fn emit_alu_reg(
    mut ctx: EmitCtx,
    rd: u8,
    rs: u8,
    rt: u8,
    alu_op: impl Fn(&mut EmitCtx, AluRegs),
) -> EmitSummary {
    let rda = ctx.dynarec.alloc_reg(rd);
    let rsa = ctx.dynarec.emit_load_reg(rs);
    let rta = ctx.dynarec.emit_load_reg(rt);

    #[cfg(target_arch = "aarch64")]
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ;; alu_op(&mut ctx, AluRegs { rd: Rd(&rda), rs: Rs(&rsa), rt: Rt(&rta) })
    );

    ctx.dynarec.mark_dirty(rd);

    rta.restore(ctx.dynarec);
    rsa.restore(ctx.dynarec);
    rda.restore(ctx.dynarec);

    EmitSummary::default()
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
            return EmitSummary::default();
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

            return EmitSummary::default();
        }

        emit_alu_reg(ctx, self.rd, self.rs, self.rt, move |ctx, regs| {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; sub W(**regs.rd), W(**regs.rs), W(**regs.rt)
            )
        })
    }
}

impl DynarecOp for ADDU {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        };

        match either_zero(self.rs, self.rt) {
            EitherZero::None => emit_alu_reg(ctx, self.rd, self.rs, self.rt, |ctx, regs| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; add W(**regs.rd), W(**regs.rs), W(**regs.rt)
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
            return EmitSummary::default();
        }

        match either_zero(self.rs, self.rt) {
            EitherZero::None => emit_alu_reg(ctx, self.rd, self.rs, self.rt, move |ctx, regs| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; and W(**regs.rd), W(**regs.rs), W(**regs.rt)
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
            return EmitSummary::default();
        }

        match either_zero(self.rs, self.rt) {
            EitherZero::None => emit_alu_reg(ctx, self.rd, self.rs, self.rt, move |ctx, regs| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; orr W(**regs.rd), W(**regs.rs), W(**regs.rt)
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
            return EmitSummary::default();
        }

        match either_zero(self.rs, self.rt) {
            EitherZero::None => emit_alu_reg(ctx, self.rd, self.rs, self.rt, move |ctx, regs| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; eor W(**regs.rd), W(**regs.rs), W(**regs.rt)
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
            return EmitSummary::default();
        }
        match either_zero(self.rs, self.rt) {
            EitherZero::None => emit_alu_reg(ctx, self.rd, self.rs, self.rt, move |ctx, regs| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; orr W(**regs.rd), W(**regs.rs), W(**regs.rt)
                    ; mvn W(**regs.rd), W(**regs.rd)
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
                EmitSummary::default()
            }
        }
    }
}

#[allow(clippy::useless_conversion)]
fn emit_shift_by_reg<'a>(
    ctx: EmitCtx<'a>,
    rd: u8,
    rs: u8,
    rt: u8,
    alu_op: impl Fn(&mut EmitCtx, AluRegs),
) -> EmitSummary {
    if rd == 0 {
        return EmitSummary::default();
    }

    if rt == 0 {
        return ctx.dynarec.emit_zero(rd);
    }

    if rs == 0 {
        return ctx.dynarec.emit_load_and_move_into(rd, rt);
    }

    emit_alu_reg(ctx, rd, rs, rt, alu_op)
}

impl DynarecOp for SLLV {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_shift_by_reg(ctx, self.rd, self.rs, self.rt, move |ctx, regs| {
            tracing::info!("emitting sllv via alu...");
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; lsl W(**regs.rd), W(**regs.rt), W(**regs.rs)
            )
        })
    }
}

impl DynarecOp for SRLV {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_shift_by_reg(ctx, self.rd, self.rs, self.rt, move |ctx, regs| {
            tracing::info!("emitting sllv via alu...");
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; lsr W(**regs.rd), W(**regs.rt), W(**regs.rs)
            )
        })
    }
}

impl DynarecOp for SRAV {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_shift_by_reg(ctx, self.rd, self.rs, self.rt, move |ctx, regs| {
            tracing::info!("emitting sllv via alu...");
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; asr W(**regs.rd), W(**regs.rt), W(**regs.rs)
            )
        })
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
#[case::sllv_01(sllv, (12, 0b10100), (10, 0b101), (11, 2))]
#[case::sllv_02(sllv, (12, 0b101), (10, 0b101), (11, 0))]
#[case::sllv_03(sllv, (12, 0), (0, 0), (11, 5))]
#[case::sllv_04(sllv, (12, 0), (0, 0), (0, 0))]
#[case::sllv_05(sllv, (12, 0x80000000), (10, 1), (11, 31))]
#[case::sllv_06(sllv, (12, 0x5555_5500), (10, 0xAAAA_AAAA), (11, 7))]
#[case::srlv_01(srlv, (12, 0b010), (10, 0b10100), (11, 3))]
#[case::srlv_02(srlv, (12, 0b101), (10, 0b101), (11, 0))]
#[case::srlv_03(srlv, (12, 0), (10, 0), (11, 5))]
#[case::srlv_04(srlv, (12, 0), (10, 0), (11, 0))]
#[case::srlv_05(srlv, (12, 1), (10, 0x80000000), (11, 31))]
#[case::srlv_06(srlv, (12, 0x0155_5555), (10, 0xAAAA_AAAA), (11, 7))]
#[case::srav_01(srav, (12, 0b11111111111111111111111111111110), (10, 0b11111111111111111111111111110100u32 as i32 as u32), (11, 3))]
#[case::srav_02(srav, (12, 0b101), (10, 0b101), (11, 0))]
#[case::srav_03(srav, (12, 0), (10, 0), (11, 5))]
#[case::srav_04(srav, (12, 0), (10, 0), (11, 0))]
#[case::srav_05(srav, (12, 0xFFFFFFFF), (10, 0x80000000), (11, 31))]
#[case::srav_06(srav, (12, 0xFF555555), (10, 0xAAAA_AAAA), (11, 7))]
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
    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

pub struct ShiftImm<'a> {
    rd: Rd<'a>,
    rt: Rt<'a>,
    imm: i8,
}

fn emit_shift_imm(
    ctx: &mut EmitCtx,
    rd: Guest,
    rt: Guest,
    imm: i8,
    emitter: impl Fn(&mut Dynarec, ShiftImm),
) -> EmitSummary {
    if rd == 0 {
        return EmitSummary::default();
    }

    if imm == 0 {
        return ctx.dynarec.emit_load_and_move_into(rd, rt);
    }

    let rd = ctx.dynarec.alloc_reg(rd);
    let rt = ctx.dynarec.emit_load_reg(rt);

    emitter(
        ctx.dynarec,
        ShiftImm {
            rd: Rd(&rd),
            rt: Rt(&rt),
            imm,
        },
    );

    rt.restore(ctx.dynarec);
    rd.restore(ctx.dynarec);

    EmitSummary::default()
}

impl DynarecOp for SLL {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        emit_shift_imm(
            &mut ctx,
            self.rd,
            self.rt,
            self.imm,
            move |dynarec, ShiftImm { rd, rt, imm }| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; lsl W(**rd), W(**rt), imm as _
                );

                dynarec.mark_dirty(self.rd);
            },
        )
    }
}

impl DynarecOp for SRL {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        emit_shift_imm(
            &mut ctx,
            self.rd,
            self.rt,
            self.imm,
            move |dynarec, ShiftImm { rd, rt, imm }| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; lsr W(**rd), W(**rt), imm as _
                );
                dynarec.mark_dirty(self.rd);
            },
        )
    }
}

impl DynarecOp for SRA {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        emit_shift_imm(
            &mut ctx,
            self.rd,
            self.rt,
            self.imm,
            move |dynarec, ShiftImm { rd, rt, imm }| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; asr W(**rd), W(**rt), imm as _
                );
                dynarec.mark_dirty(self.rd);
            },
        )
    }
}

pub struct AluImm<'a> {
    rt: Rt<'a>,
    rs: Rs<'a>,
    imm: i16,
}

fn emit_alu_imm(
    mut ctx: EmitCtx,
    rt: Guest,
    rs: Guest,
    imm: i16,
    alu_op: impl Fn(&mut EmitCtx, AluImm),
) -> EmitSummary {
    let rta = ctx.dynarec.alloc_reg(rt);
    let rsa = ctx.dynarec.emit_load_reg(rs);

    #[cfg(target_arch = "aarch64")]
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ;; alu_op(&mut ctx, AluImm { rt: Rt(&rta), rs: Rs(&rsa), imm})
    );

    ctx.dynarec.mark_dirty(rt);

    rsa.restore(ctx.dynarec);
    rta.restore(ctx.dynarec);

    EmitSummary::default()
}

impl DynarecOp for ANDI {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        match (self.rt, self.rs, self.imm) {
            (0, _, _) => EmitSummary::default(),
            (_, 0, _) | (_, _, 0) => ctx.dynarec.emit_zero(self.rt),
            _ => emit_alu_imm(
                ctx,
                self.rt,
                self.rs,
                self.imm as _,
                move |ctx, AluImm { rt, rs, imm }| {
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(
                        ctx.dynarec.asm
                        ; .arch aarch64
                        ; mov w3, ext::zero(imm)
                        ; and W(**rt), W(**rs), w3
                    );
                },
            ),
        }
    }
}

impl DynarecOp for ORI {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        match self {
            Self {
                rt: 0,
                rs: _,
                imm: _,
            } => EmitSummary::default(),
            Self {
                rt: _,
                rs: 0,
                imm: _,
            } => ctx.dynarec.emit_immediate(self.rt, self.imm),
            Self {
                rt: _,
                rs: _,
                imm: 0,
            } => ctx.dynarec.emit_load_and_move_into(self.rt, self.rs),
            _ => emit_alu_imm(
                ctx,
                self.rt,
                self.rs,
                self.imm as _,
                move |ctx, AluImm { rt, rs, imm }| {
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(
                        ctx.dynarec.asm
                        ; .arch aarch64
                        ; mov w3, ext::zero(imm)
                        ; orr W(**rt), W(**rs), w3
                    );
                },
            ),
        }
    }
}

impl DynarecOp for XORI {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        match self {
            Self {
                rt: 0,
                rs: _,
                imm: _,
            } => EmitSummary::default(),
            Self {
                rt: _,
                rs: 0,
                imm: _,
            } => ctx.dynarec.emit_immediate(self.rt, self.imm),
            Self {
                rt: _,
                rs: _,
                imm: 0,
            } => ctx.dynarec.emit_load_and_move_into(self.rt, self.rs),
            _ => emit_alu_imm(
                ctx,
                self.rt,
                self.rs,
                self.imm as _,
                move |ctx, AluImm { rt, rs, imm }| {
                    #[cfg(target_arch = "aarch64")]
                    dynasm!(
                        ctx.dynarec.asm
                        ; .arch aarch64
                        ; mov w3, ext::zero(imm)
                        ; eor W(**rt), W(**rs), w3
                    );
                },
            ),
        }
    }
}

#[cfg(test)]
#[rstest]
#[case::sll_01(sll, (12, 0b10100), (10, 0b101), 2)]
#[case::sll_02(sll, (12, 0b101), (10, 0b101), 0)]
#[case::sll_03(sll, (12, 0), (0, 0), 5)]
#[case::sll_04(sll, (12, 0), (0, 0), 0)]
#[case::sll_05(sll, (12, 0x80000000), (10, 1), 31)]
#[case::sll_06(sll, (12, 0x5555_5500), (10, 0xAAAA_AAAA), 7)]
#[case::srl_01(srl, (12, 0b010), (10, 0b10100), 3)]
#[case::srl_02(srl, (12, 0b101), (10, 0b101), 0)]
#[case::srl_03(srl, (12, 0), (10, 0), 5)]
#[case::srl_04(srl, (12, 0), (10, 0), 0)]
#[case::srl_05(srl, (12, 1), (10, 0x80000000), 31)]
#[case::srl_06(srl, (12, 0x0155_5555), (10, 0xAAAA_AAAA), 7)]
#[case::sra_01(sra, (12, 0b11111111111111111111111111111110), (10, 0b11111111111111111111111111110100u32 as i32 as u32), 3)]
#[case::sra_02(sra, (12, 0b101), (10, 0b101), 0)]
#[case::sra_03(sra, (12, 0), (10, 0), 5)]
#[case::sra_04(sra, (12, 0), (10, 0), 0)]
#[case::sra_05(sra, (12, 0xFFFFFFFF), (10, 0x80000000), 31)]
#[case::sra_06(sra, (12, 0xFF555555), (10, 0xAAAA_AAAA), 7)]
#[case::andi_01(andi, (12, 0b1010), (10, 0b1111), 0b1010i16)]
#[case::andi_02(andi, (12, 0), (0, 0), 0b1010i16)]
#[case::andi_03(andi, (12, 0), (11, 0b1010), 0i16)]
#[case::andi_04(andi, (12, 0), (0, 0), 0i16)]
#[case::ori_01(ori, (12, 0b1111), (10, 0b1010), 0b0101)]
#[case::ori_02(ori, (12, 0b1010), (10, 0b1010), 0)]
#[case::ori_03(ori, (12, 0b0101), (0, 0), 0b0101)]
#[case::ori_04(ori, (12, 0), (0, 0), 0)]
#[case::ori_05(ori, (12, 0xAAAAFFFF), (10, 0xAAAAAAAA), 0xFFFFu16 as i16)]
#[case::xori_01(xori, (12, 0b0101), (10, 0b1111), 0b1010)]
#[case::xori_02(xori, (12, 0b1010), (10, 0b1010), 0)]
#[case::xori_03(xori, (12, 0b0101), (0, 0), 0b0101)]
#[case::xori_04(xori, (12, 0), (0, 0), 0)]
#[case::xori_05(xori, (12, 0xAAAA5555), (10, 0xAAAAAAAA), 0xFFFFu16 as i16)]
#[case::xori_06(xori, (12, 0xAAAAFFFF), (10, 0xAAAAAAAA), 0x5555)]
fn test_alu_imm<I: Into<i16>>(
    #[case] instr: impl Fn(u8, u8, I) -> OpCode,
    #[case] expected: (Guest, u32),
    #[case] a: (Guest, u32),
    #[case] b: I,
) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    if expected.0 != 0 {
        emu.cpu.gpr[expected.0 as usize] = 1231123;
    }
    emu.cpu.gpr[a.0 as usize] = a.1;
    emu.write_many(0x0, &program([instr(expected.0, a.0, b), OpCode(69420)]));
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[expected.0 as usize], expected.1);
    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);

    Ok(())
}

impl DynarecOp for LUI {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rt == 0 {
            return EmitSummary::default();
        }
        let rt = ctx.dynarec.emit_load_reg(self.rt);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; movk W(*rt), ext::zero(self.imm) as _, LSL #16
        );

        ctx.dynarec.mark_dirty(self.rt);
        rt.restore(ctx.dynarec);

        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
#[case(0x0, 12, 0x1234, 0x12340000)]
#[case(0x5678, 9, 0x1234, 0x12345678)]
#[case(0x0, 0, 0x1234, 0x0)]
fn test_lui(
    #[case] initial: u32,
    #[case] rt: Guest,
    #[case] imm: i16,
    #[case] expected: u32,
) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    if rt != 0 {
        emu.cpu.gpr[rt as usize] = initial;
    }
    emu.write_many(0x0, &program([lui(rt, imm), OpCode(69420)]));
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[rt as usize], expected);
    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);

    Ok(())
}

impl DynarecOp for J {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        let new_pc = (self.imm << 2) + (ctx.pc & 0xf0000000);
        ctx.dynarec.set_delay_slot(move |ctx| {
            ctx.dynarec.emit_write_pc(Reg::W(3), new_pc);
            EmitSummary::builder().pc_updated(true).build()
        });
        EmitSummary::default()
    }

    fn cycles(&self) -> u16 {
        3
    }

    fn hazard(&self) -> u16 {
        2
    }

    fn boundary(&self) -> Boundary {
        Boundary::Soft
    }
}

#[cfg(test)]
#[rstest]
#[case(0x0, 0x0000_1000)]
fn test_j(#[case] initial_pc: u32, #[case] jump_imm: u32) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.write_many(
        initial_pc,
        &program([
            j(jump_imm),
            addiu(9, 0, 69),
            addiu(9, 0, 420),
            OpCode(69420),
        ]),
    );
    let new_pc = (jump_imm << 2) + (emu.cpu.pc & 0xf0000000);
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[9], 69);
    assert_eq!(emu.cpu.d_clock, 3);
    assert_eq!(emu.cpu.pc, new_pc);

    Ok(())
}

impl DynarecOp for JAL {
    fn cycles(&self) -> u16 {
        3
    }
    fn hazard(&self) -> u16 {
        2
    }
    fn boundary(&self) -> Boundary {
        Boundary::Soft
    }
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        let new_pc = (self.imm << 2) + (ctx.pc & 0xf0000000);
        let return_address = ctx.pc + 0x8;
        ctx.dynarec.set_delay_slot(move |ctx| {
            ctx.dynarec.emit_write_pc(Reg::W(3), new_pc);
            ctx.dynarec.emit_immediate_large(cpu::RA, return_address);

            EmitSummary::builder().pc_updated(true).build()
        });
        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
#[case(0x0, 0x0000_1000)]
fn test_jal(#[case] initial_pc: u32, #[case] jump_imm: u32) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.write_many(
        initial_pc,
        &program([
            jal(jump_imm),
            addiu(9, 0, 69),
            addiu(9, 0, 420),
            OpCode(69420),
        ]),
    );
    let new_pc = (jump_imm << 2) + (emu.cpu.pc & 0xf0000000);
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[9], 69);
    assert_eq!(emu.cpu.d_clock, 3);
    assert_eq!(emu.cpu.pc, new_pc);
    assert_eq!(emu.cpu.gpr[cpu::RA as usize], initial_pc + 0x8);

    Ok(())
}

impl DynarecOp for JR {
    fn cycles(&self) -> u16 {
        3
    }
    fn hazard(&self) -> u16 {
        2
    }
    fn boundary(&self) -> Boundary {
        Boundary::Soft
    }
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        let dest = ctx.dynarec.emit_load_reg(self.rs);
        let s = ctx.parity();

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            // ; str W(*dest), [sp, #-16]!
            ; fmov S(s(8)), W(*dest)
        );

        ctx.dynarec.set_delay_slot(move |ctx| {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; str S(s(8)), [x0, Emu::PC_OFFSET as _]
            );

            EmitSummary::builder().pc_updated(true).build()
        });
        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
#[case(0x0, (9, 0x0000_1000))]
fn test_jr(#[case] initial_pc: u32, #[case] rs: (Guest, u32)) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.cpu.gpr[rs.0 as usize] = rs.1;
    emu.write_many(
        initial_pc,
        &program([jr(rs.0), addiu(9, 0, 69), addiu(9, 0, 420), OpCode(69420)]),
    );
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[9], 69);
    assert_eq!(emu.cpu.d_clock, 3);
    assert_eq!(emu.cpu.pc, rs.1);

    Ok(())
}

impl DynarecOp for JALR {
    fn cycles(&self) -> u16 {
        3
    }
    fn hazard(&self) -> u16 {
        2
    }
    fn boundary(&self) -> Boundary {
        Boundary::Soft
    }
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        let s = ctx.parity();
        let dest = ctx.dynarec.emit_load_reg(self.rs);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            // ; str W(*dest), [sp, #-16]!
            ; fmov S(s(8)), W(*dest)
        );

        let rd = self.rd;
        // we know the delay slots runs at old pc + 4
        ctx.dynarec.set_delay_slot(move |ctx| {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                // ; ldr w3, [sp], #16
                ; str S(s(8)), [x0, Emu::PC_OFFSET as _]
            );

            let ret_address = ctx.pc + 0x8;
            ctx.dynarec.emit_immediate_large(rd, ret_address);

            EmitSummary::builder().pc_updated(true).build()
        });
        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
#[case(0x0, (9, 0x0000_1000), 10)]
fn test_jalr(
    #[case] initial_pc: u32,
    #[case] rs: (Guest, u32),
    #[case] rd: Guest,
) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.cpu.gpr[rs.0 as usize] = rs.1;
    emu.write_many(
        initial_pc,
        &program([
            jalr(rd, rs.0),
            addiu(9, 0, 69),
            addiu(9, 0, 420),
            OpCode(69420),
        ]),
    );
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[9], 69);
    assert_eq!(emu.cpu.d_clock, 3);
    assert_eq!(emu.cpu.pc, rs.1);
    assert_eq!(emu.cpu.gpr[rd as usize], initial_pc + 0x8);

    Ok(())
}

/// `selector` must look something like
/// ```
/// dynasm!(
///     ctx.dynarec.asm
///     ; csel w2, w3, w2, {SELECTOR}
/// );
/// ```
#[allow(clippy::useless_conversion)]
#[cfg(target_arch = "aarch64")]
fn emit_branch(
    ctx: EmitCtx,
    rs: u8,
    rt: u8,
    imm: i16,
    selector: impl Fn(&mut EmitCtx) + 'static,
) -> EmitSummary {
    let s = ctx.parity();
    let rs = ctx.dynarec.emit_load_reg(rs);
    let rt = ctx.dynarec.emit_load_reg(rt);

    // calculate branch value
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        // ; stp W(*rs), W(*rt), [sp, #-16]!
        ; fmov S(s(9)), W(*rs)
        ; fmov S(s(10)), W(*rt)
    );

    let branch_dest = (ctx.pc + 0x4).wrapping_add_signed(ext::sign(imm) << 2);
    ctx.dynarec.set_delay_slot(move |mut ctx| {
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            // ; ldp w2, w3, [sp], #16
            ; fmov w2, S(s(9))
            ; fmov w3, S(s(10))
            ; cmp w2, w3

            ; movz w2, (ctx.pc + 0x8) >> 16, lsl #16
            ; movk w2, (ctx.pc + 0x8) & 0x0000_ffff
            ; movz w3, branch_dest >> 16, lsl #16
            ; movk w3, branch_dest & 0x0000_ffff
            ;; selector(&mut ctx)
            ; str w2, [x0, Emu::PC_OFFSET as _]
        );

        EmitSummary::builder().pc_updated(true).build()
    });

    rt.restore(ctx.dynarec);
    rs.restore(ctx.dynarec);

    EmitSummary::default()
}

impl DynarecOp for BEQ {
    fn cycles(&self) -> u16 {
        3
    }
    fn hazard(&self) -> u16 {
        2
    }
    fn boundary(&self) -> Boundary {
        Boundary::Soft
    }
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        #[cfg(target_arch = "aarch64")]
        emit_branch(ctx, self.rs, self.rt, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; csel w2, w3, w2, eq
            )
        })
    }
}

impl DynarecOp for BNE {
    fn cycles(&self) -> u16 {
        3
    }
    fn hazard(&self) -> u16 {
        2
    }
    fn boundary(&self) -> Boundary {
        Boundary::Soft
    }
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        #[cfg(target_arch = "aarch64")]
        emit_branch(ctx, self.rs, self.rt, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; csel w2, w3, w2, ne
            )
        })
    }
}

#[cfg(test)]
#[rstest]
// beq
#[case::beq_01(beq, 0x1000, (5, 100), (6, 100), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::beq_02(beq, 0x1000, (5, 100), (6, 200), 0x10, 0x1000 + 0x8)]
#[case::beq_03(beq, 0x2000, (7, 0), (8, 0), 0x20, 0x2000 + 0x4 + (0x20 << 2))]
#[case::beq_04(beq, 0x3000, (3, 42), (4, 43), -0x8, 0x3000 + 0x8)]
#[case::beq_05(beq, 0x4000, (1, 0xFFFF), (2, 0xFFFF), 0x0, 0x4000 + 0x4)]
#[case::beq_06(beq, 0x5000, (10, 0xDEADBEEF), (11, 0xDEADBEEF), 0x7FF, 0x5000 + 0x4 + (0x7FF << 2))]
// bne
#[case::bne_01(bne, 0x1000, (5, 100), (6, 200), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::bne_02(bne, 0x1000, (5, 100), (6, 100), 0x10, 0x1000 + 0x8)]
#[case::bne_03(bne, 0x2000, (7, 0), (8, 1), 0x20, 0x2000 + 0x4 + (0x20 << 2))]
#[case::bne_04(bne, 0x3000, (3, 0), (4, 0), -0x8, 0x3000 + 0x8)]
#[case::bne_05(bne, 0x5000, (10, 0xDEADBEEF), (11, 0xDEADBEEE), 0x7FF, 0x5000 + 0x4 + (0x7FF << 2))]
#[case::bne_06(bne, 0x6000, (12, 100), (13, 200), 0x0, 0x6000 + 0x4)]
fn test_branch(
    #[case] instr: impl Fn(Guest, Guest, i16) -> OpCode,
    #[case] initial_pc: u32,
    #[case] rs: (Guest, u32),
    #[case] rt: (Guest, u32),
    #[case] offset: i16,
    #[case] expected_pc: u32,
) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.cpu.gpr[rs.0 as usize] = rs.1;
    emu.cpu.gpr[rt.0 as usize] = rt.1;

    emu.write_many(initial_pc, &program([instr(rs.0, rt.0, offset)]));
    emu.write_many(initial_pc + ext::zero(offset), &program([OpCode(69420)]));

    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.pc, expected_pc);

    Ok(())
}
/// `selector` must look something like
/// ```
/// dynasm!(
///     ctx.dynarec.asm
///     ; csel w2, w3, w2, {SELECTOR}
/// );
/// ```
#[allow(clippy::useless_conversion)]
#[cfg(target_arch = "aarch64")]
fn emit_branch_zero(
    ctx: EmitCtx,
    rs: u8,
    imm: i16,
    selector: impl Fn(&mut EmitCtx) + 'static,
) -> EmitSummary {
    let s = ctx.parity();
    let rs = ctx.dynarec.emit_load_reg(rs);

    // calculate branch value
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        // ; str W(*rs), [sp, #-16]!
        ; fmov S(s(8)), W(*rs)
    );

    let branch_dest = (ctx.pc + 0x4).wrapping_add_signed(ext::sign(imm) << 2);
    ctx.dynarec.set_delay_slot(move |mut ctx| {
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            // ; ldr w2, [sp], #16
            ; fmov w2, S(s(8))
            ; cmp w2, #0

            ; movz w2, (ctx.pc + 0x8) >> 16, lsl #16
            ; movk w2, (ctx.pc + 0x8) & 0x0000_ffff
            ; movz w3, branch_dest >> 16, lsl #16
            ; movk w3, branch_dest & 0x0000_ffff
            ;; selector(&mut ctx)
            ; str w2, [x0, Emu::PC_OFFSET as _]
        );

        EmitSummary::builder().pc_updated(true).build()
    });

    rs.restore(ctx.dynarec);

    EmitSummary::default()
}

impl DynarecOp for BLTZ {
    fn cycles(&self) -> u16 {
        3
    }
    fn hazard(&self) -> u16 {
        2
    }
    fn boundary(&self) -> Boundary {
        Boundary::Soft
    }
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        #[cfg(target_arch = "aarch64")]
        emit_branch_zero(ctx, self.rs, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; csel w2, w3, w2, lt
            )
        })
    }
}

impl DynarecOp for BGEZ {
    fn cycles(&self) -> u16 {
        3
    }
    fn hazard(&self) -> u16 {
        2
    }
    fn boundary(&self) -> Boundary {
        Boundary::Soft
    }
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        #[cfg(target_arch = "aarch64")]
        emit_branch_zero(ctx, self.rs, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; csel w2, w3, w2, ge
            )
        })
    }
}

impl DynarecOp for BLEZ {
    fn cycles(&self) -> u16 {
        3
    }
    fn hazard(&self) -> u16 {
        2
    }
    fn boundary(&self) -> Boundary {
        Boundary::Soft
    }
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        #[cfg(target_arch = "aarch64")]
        emit_branch_zero(ctx, self.rs, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; csel w2, w3, w2, le
            )
        })
    }
}

impl DynarecOp for BGTZ {
    fn cycles(&self) -> u16 {
        3
    }
    fn hazard(&self) -> u16 {
        2
    }
    fn boundary(&self) -> Boundary {
        Boundary::Soft
    }
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        #[cfg(target_arch = "aarch64")]
        emit_branch_zero(ctx, self.rs, self.imm, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; csel w2, w3, w2, gt
            )
        })
    }
}

#[cfg(test)]
#[rstest]
// bltz - branch if less than zero
#[case::bltz_01(bltz, 0x1000, (5, -1i32 as u32), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::bltz_02(bltz, 0x1000, (5, 0), 0x10, 0x1000 + 0x8)]
#[case::bltz_03(bltz, 0x1000, (5, 100), 0x10, 0x1000 + 0x8)]
#[case::bltz_04(bltz, 0x2000, (7, 0x80000000), 0x20, 0x2000 + 0x4 + (0x20 << 2))]
#[case::bltz_05(bltz, 0x3000, (3, 0x7FFFFFFF), -0x8, 0x3000 + 0x8)]
#[case::bltz_06(bltz, 0x4000, (1, -42i32 as u32), 0x0, 0x4000 + 0x4)]
// bgez - branch if greater than or equal to zero
#[case::bgez_01(bgez, 0x1000, (5, 0), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::bgez_02(bgez, 0x1000, (5, 100), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::bgez_03(bgez, 0x1000, (5, -1i32 as u32), 0x10, 0x1000 + 0x8)]
#[case::bgez_04(bgez, 0x2000, (7, 0x7FFFFFFF), 0x20, 0x2000 + 0x4 + (0x20 << 2))]
#[case::bgez_05(bgez, 0x3000, (3, 0x80000000), -0x8, 0x3000 + 0x8)]
#[case::bgez_06(bgez, 0x4000, (1, 0), 0x0, 0x4000 + 0x4)]
// blez - branch if less than or equal to zero
#[case::blez_01(blez, 0x1000, (5, 0), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::blez_02(blez, 0x1000, (5, -1i32 as u32), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::blez_03(blez, 0x1000, (5, 100), 0x10, 0x1000 + 0x8)]
#[case::blez_04(blez, 0x2000, (7, 0x80000000), 0x20, 0x2000 + 0x4 + (0x20 << 2))]
#[case::blez_05(blez, 0x3000, (3, 0x7FFFFFFF), -0x8, 0x3000 + 0x8)]
#[case::blez_06(blez, 0x4000, (1, -42i32 as u32), 0x0, 0x4000 + 0x4)]
// bgtz - branch if greater than zero
#[case::bgtz_01(bgtz, 0x1000, (5, 100), 0x10, 0x1000 + 0x4 + (0x10 << 2))]
#[case::bgtz_02(bgtz, 0x1000, (5, 0), 0x10, 0x1000 + 0x8)]
#[case::bgtz_03(bgtz, 0x1000, (5, -1i32 as u32), 0x10, 0x1000 + 0x8)]
#[case::bgtz_04(bgtz, 0x2000, (7, 0x7FFFFFFF), 0x20, 0x2000 + 0x4 + (0x20 << 2))]
#[case::bgtz_05(bgtz, 0x3000, (3, 0x80000000), -0x8, 0x3000 + 0x8)]
#[case::bgtz_06(bgtz, 0x4000, (1, 1), 0x0, 0x4000 + 0x4)]
fn test_branch_zero(
    #[case] instr: impl Fn(Guest, i16) -> OpCode,
    #[case] initial_pc: u32,
    #[case] rs: (Guest, u32),
    #[case] offset: i16,
    #[case] expected_pc: u32,
) -> color_eyre::Result<()> {
    use crate::{Emu, cpu::program, dynarec_v2::PipelineV2};
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.cpu.gpr[rs.0 as usize] = rs.1;

    emu.write_many(initial_pc, &program([instr(rs.0, offset)]));
    emu.write_many(initial_pc + ext::zero(offset), &program([OpCode(69420)]));

    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.pc, expected_pc);

    Ok(())
}

impl DynarecOp for MTCn {
    fn cycles(&self) -> u16 {
        match self.cop {
            0 => 1,
            1 => 1,
            2 => 3,
            3 => 1,
            _ => unreachable!(),
        }
    }
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        let rt = ctx.dynarec.emit_load_reg(self.rt);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; str W(*rt), [x0, Cpu::cop_reg_offset(self.cop, self.rd) as _]
        );

        rt.restore(ctx.dynarec);
        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
#[case(0, 5, 10)]
#[case(2, 5, 10)]
#[case(2, 31, 10)] // really pushing it
fn test_mtcn(#[case] cop: u8, #[case] rd: u8, #[case] rt: u8) -> color_eyre::Result<()> {
    use crate::{
        Emu,
        cpu::{ops::Op, program},
        dynarec_v2::PipelineV2,
    };
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[rt as usize] = 69;

    emu.write_many(
        0x0,
        &program([MTCn::new(cop, rt, rd).into_opcode(), OpCode(69420)]),
    );

    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    match cop {
        0 => assert_eq!(emu.cpu.cop0.reg[rd as usize], 69),
        2 => assert_eq!(emu.cpu.cop2.reg[rd as usize], 69),
        _ => panic!("get out"),
    }

    Ok(())
}

#[cfg(test)]
#[rstest]
fn test_mtcn_enable_isc() -> color_eyre::Result<()> {
    use crate::{
        Emu,
        cpu::{ops::Op, program},
        dynarec_v2::PipelineV2,
    };
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.write_many(
        0x0,
        &program([
            lui(9, 0x0001),
            MTCn::new(0, 9, 12).into_opcode(),
            OpCode(69420),
        ]),
    );

    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.cop0.reg[12], 0x0001_0000);
    assert!(emu.cpu.isc());

    Ok(())
}

impl DynarecOp for NOP {
    fn cycles(&self) -> u16 {
        1
    }
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
fn test_store_loop() -> color_eyre::Result<()> {
    use crate::{
        Emu,
        cpu::{ops::Op, program},
        dynarec_v2::PipelineV2,
    };
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.write_many(
        0x0,
        &program([
            addiu(9, 0, 69),
            addiu(10, 0, 0x0000_1000),
            addiu(11, 0, 0x0000_2000),
            sw(9, 10, 0),
            bne(10, 11, -2),
            addiu(10, 10, 0x4),
            OpCode(69420),
        ]),
    );

    loop {
        PipelineV2::new(&emu).run_once(&mut emu)?;
        assert!(emu.cpu.gpr[10] <= emu.cpu.gpr[11] + 0x8);

        if emu.cpu.pc == 24 {
            break;
        }
    }

    tracing::info!(?emu.cpu);

    Ok(())
}

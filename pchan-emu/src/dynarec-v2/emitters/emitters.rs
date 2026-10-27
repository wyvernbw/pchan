#[path = "./emitters-cpu.rs"]
mod emitters_cpu;
#[path = "./emitters-gte.rs"]
mod emitters_gte;
#[cfg(test)]
#[path = "./emitters-tests.rs"]
mod emitters_tests;

use crate::bindings::ExternFnTable;
use crate::cpu::gte::GteBindings;
use crate::cpu::{Cpu, RA};
use crate::dynarec_v2::regalloc::AllocResultStackless;
#[cfg(test)]
use crate::dynarec_v2::{CreateDynarecParams, run_step};
use crate::dynarec_v2::{DynEmitter, Guest};
use crate::{Emu, cpu};
#[cfg(test)]
use core::fmt;
use core::num::NonZeroU8;

use arbitrary_int::traits::Integer;
use enum_dispatch::enum_dispatch;

use pchan_utils::default;
#[cfg(test)]
use rstest::rstest;
use smallbox::SmallBox;

use crate::cpu::ops::prelude::*;
use crate::dynarec_v2::{Dynarec, LoadedReg, Reg};
#[cfg(test)]
use crate::memory;
use crate::memory::ext;
use dynasm::dynasm;
use dynasmrt::DynasmApi;

use super::ScheduledEmitter;
use core::alloc::Allocator;

pub struct EmitCtx<'a, A: Allocator> {
    pub dynarec:        &'a mut Dynarec<A>,
    pub pc:             u32,
    pub d_clock:        u32,
    pub delay_slot:     bool,
    pub scratch_cursor: &'a mut u8,
}

fn cop_reg(cop: u8, reg: u8) -> u8 {
    match cop {
        0 => 32 + reg,
        2 => 32 + 64 + reg,
        _ => unreachable!(),
    }
}

impl<A: Allocator + Copy> EmitCtx<'_, A> {
    fn schedule_in(
        &mut self,
        ops: u32,
        write: Option<u8>,
        emitter: impl Fn(EmitCtx<A>) -> EmitSummary + 'static,
    ) {
        let emitter = SmallBox::new(emitter) as DynEmitter<A>;
        let at = self.pc + ops * 4;
        self.dynarec
            .scheduler
            .queue
            .push(ScheduledEmitter {
                emitter,
                schedule: at,
                write,
                pc: self.pc,
            })
            .expect("binary heap is at capacity. increase allocation size or decrease block size");
    }
    fn drain_schedule(&mut self) {
        while let Some(emitter) = self.dynarec.scheduler.queue.pop() {
            emitter.emitter.call((EmitCtx {
                dynarec:        self.dynarec,
                pc:             emitter.schedule,
                d_clock:        self.d_clock,
                delay_slot:     self.delay_slot,
                scratch_cursor: self.scratch_cursor,
            },));
        }
    }

    fn alloc_scratch(&mut self) -> u32 {
        let cursor = *self.scratch_cursor as usize;
        *self.scratch_cursor = ((*self.scratch_cursor + 1) as usize % Cpu::SCRATCH_SIZE) as u8;
        (Cpu::SCRATCH_OFFSET + cursor * size_of::<u32>()) as u32
    }

    fn alloc_scratch_pair(&mut self) -> (u32, u32) {
        let mut a = *self.scratch_cursor as usize;
        if a + 2 > Cpu::SCRATCH_SIZE {
            a = 0;
        }
        let b = a + 1;
        *self.scratch_cursor = ((a + 2) % (Cpu::SCRATCH_SIZE)) as u8;
        (
            (Cpu::SCRATCH_OFFSET + a * size_of::<u32>()) as u32,
            (Cpu::SCRATCH_OFFSET + b * size_of::<u32>()) as u32,
        )
    }
}

#[must_use]
fn shadow(guest: u8) -> u32 {
    (Cpu::SHADOW_OFFSET + usize::from(guest) * size_of::<u32>()) as u32
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    #[default]
    None,
    Soft,
    Hard,
}

#[derive(Debug, Default)]
pub struct EmitSummary {
    pub pc_updated: bool,
}

impl EmitSummary {
    #[must_use]
    pub fn pc_updated() -> Self {
        Self { pc_updated: true }
    }
}

#[enum_dispatch(DecodedOp)]
pub trait DynarecOp {
    fn cycles(&self) -> u16 {
        1
    }

    fn hazard(&self) -> u16 {
        0
    }

    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary;

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
pub enum DecodedOp {
    Nop(Nop),
    Illegal(Illegal),
    Sll(Sll),
    Srl(Srl),
    Sra(Sra),
    Sllv(Sllv),
    Srlv(Srlv),
    Srav(Srav),
    Jr(Jr),
    Jalr(Jalr),
    Syscall(Syscall),
    Brk(Brk),
    Mfhi(Mfhi),
    Mthi(Mthi),
    Mflo(Mflo),
    Mtlo(Mtlo),
    Mult(Mult),
    Multu(Multu),
    Div(Div),
    Divu(Divu),
    Addu(Addu),
    Subu(Subu),
    And(And),
    Or(Or),
    Xor(Xor),
    Bltz(Bltz),
    Bgez(Bgez),
    Bltzal(Bltzal),
    Bgezal(Bgezal),
    J(J),
    Jal(Jal),
    Blez(Blez),
    Bgtz(Bgtz),
    Nor(Nor),
    Slt(Slt),
    Sltu(Sltu),
    Beq(Beq),
    Bne(Bne),
    HaltBlock(HaltBlock),
    Sb(Sb),
    Sh(Sh),
    Swl(Swl),
    Sw(Sw),
    Swr(Swr),
    Lwcn(Lwcn),
    Swcn(Swcn),
    Addiu(Addiu),
    Slti(Slti),
    Sltiu(Sltiu),
    Andi(Andi),
    Ori(Ori),
    Xori(Xori),
    Lui(Lui),
    Rfe(Rfe),
    Mfcn(Mfcn),
    Cfcn(Cfcn),
    Mtcn(Mtcn),
    Ctcn(Ctcn),
    CopNCmd(CopNCmd),
    Lb(Lb),
    Lbu(Lbu),
    Lh(Lh),
    Lhu(Lhu),
    Lwr(Lwr),
    Lwl(Lwl),
    Lw(Lw),
}

#[derive(Debug, Clone, Copy, derive_more::Display, Hash, PartialEq, Eq)]
pub struct Illegal;

impl DynarecOp for Illegal {
    fn emit<A: Allocator + Copy>(&self, _: EmitCtx<A>) -> EmitSummary {
        EmitSummary::default()
    }

    fn boundary(&self) -> Boundary {
        Boundary::Hard
    }
}

impl DecodedOp {
    #[must_use]
    pub fn new(fields: OpCode) -> Self {
        let [op] = Self::decode([fields]);
        op
    }
    #[must_use]
    pub const fn is_illegal(&self) -> bool {
        matches!(self, Self::Illegal(_))
    }
    #[must_use]
    pub const fn illegal() -> Self {
        Self::Illegal(Illegal)
    }
    pub fn decode<const N: usize>(fields: [impl Into<OpCode>; N]) -> [Self; N] {
        fields.map(|fields| {
            let fields = fields.into();
            if fields == OpCode::NOP_FIELDS {
                return Self::Nop(Nop);
            }
            if fields == OpCode::HALT {
                return Self::HaltBlock(HaltBlock);
            }
            let opcode = fields.opcode().value();
            let rs = fields.rs().value();
            let rt = fields.rt().value();
            let rd = fields.rd().value();
            let funct = fields.funct().value();
            #[allow(clippy::match_same_arms)]
            match (opcode, rs, rt, funct) {
                (0x0, _, _, 0x0) => Self::Sll(Sll::new(rd, rt, fields.shamt().value())),
                (0x0, _, _, 0x1) => Self::illegal(),
                (0x0, _, _, 0x2) => Self::Srl(Srl::new(rd, rt, fields.shamt().value())),
                (0x0, _, _, 0x3) => Self::Sra(Sra::new(rd, rt, fields.shamt().value())),
                (0x0, _, _, 0x4) => Self::Sllv(Sllv::new(rd, rt, rs)),
                (0x0, _, _, 0x5) => Self::illegal(),
                (0x0, _, _, 0x6) => Self::Srlv(Srlv::new(rd, rt, rs)),
                (0x0, _, _, 0x7) => Self::Srav(Srav::new(rd, rt, rs)),
                (0x0, _, _, 0x8) => Self::Jr(Jr::new(rs)),
                (0x0, _, _, 0x9) => Self::Jalr(Jalr::new(rd, rs)),
                (0x0, _, _, 0xA) => Self::illegal(),
                (0x0, _, _, 0xB) => Self::illegal(),
                (0x0, _, _, 0xC) => Self::Syscall(Syscall),
                (0x0, _, _, 0xD) => Self::Brk(Brk), // brk
                (0x0, _, _, 0xE) => Self::illegal(),
                (0x0, _, _, 0xF) => Self::illegal(),
                (0x0, _, _, 0x10) => Self::Mfhi(Mfhi::new(rd)),
                (0x0, _, _, 0x11) => Self::Mthi(Mthi::new(rs)),
                (0x0, _, _, 0x12) => Self::Mflo(Mflo::new(rd)),
                (0x0, _, _, 0x13) => Self::Mtlo(Mtlo::new(rs)),
                (0x0, _, _, 0x14..=0x17) => Self::illegal(),
                (0x0, _, _, 0x18) => Self::Mult(Mult::new(rs, rt)),
                (0x0, _, _, 0x19) => Self::Multu(Multu::new(rs, rt)),
                (0x0, _, _, 0x1A) => Self::Div(Div::new(rs, rt)),
                (0x0, _, _, 0x1B) => Self::Divu(Divu::new(rs, rt)),
                (0x0, _, _, 0x1C..=0x1F) => Self::illegal(),
                (0x0, _, _, 0x20 | 0x21) => Self::Addu(Addu::new(rd, rs, rt)),
                (0x0, _, _, 0x22 | 0x23) => Self::Subu(Subu::new(rd, rs, rt)),
                (0x0, _, _, 0x24) => Self::And(And::new(rd, rs, rt)),
                (0x0, _, _, 0x25) => Self::Or(Or::new(rd, rs, rt)),
                (0x0, _, _, 0x26) => Self::Xor(Xor::new(rd, rs, rt)),
                (0x0, _, _, 0x27) => Self::Nor(Nor::new(rd, rs, rt)),
                (0x0, _, _, 0x28..=0x29) => Self::illegal(),
                (0x0, _, _, 0x2A) => Self::Slt(Slt::new(rd, rs, rt)),
                (0x0, _, _, 0x2B) => Self::Sltu(Sltu::new(rd, rs, rt)),
                (0x0, _, _, 0x2C..) => Self::illegal(),
                (0x1, _, 0x10, _) => Self::Bltzal(Bltzal::new(rs, fields.imm16())),
                (0x1, _, 0x11, _) => Self::Bgezal(Bgezal::new(rs, fields.imm16())),
                // * TODO: bltz and bgez dupes * //
                (0x1, _, rt, _) if rt & 1 == 0 => Self::Bltz(Bltz::new(rs, fields.imm16())),
                (0x1, _, rt, _) if rt & 1 == 1 => Self::Bgez(Bgez::new(rs, fields.imm16())),
                (0x2, _, _, _) => Self::J(J::new(fields.imm26().value())),
                (0x3, _, _, _) => Self::Jal(Jal::new(fields.imm26().value())),
                (0x4, _, _, _) => Self::Beq(Beq::new(rs, rt, fields.imm16())),
                (0x5, _, _, _) => Self::Bne(Bne::new(rs, rt, fields.imm16())),
                (0x6, _, _, _) => Self::Blez(Blez::new(rs, fields.imm16())),
                (0x7, _, _, _) => Self::Bgtz(Bgtz::new(rs, fields.imm16())),
                (0x8 | 0x9, _, _, _) => Self::Addiu(Addiu::new(rt, rs, fields.imm16())),
                (0xA, _, _, _) => Self::Slti(Slti::new(rt, rs, fields.imm16())),
                (0xB, _, _, _) => Self::Sltiu(Sltiu::new(rt, rs, fields.imm16())),
                (0xC, _, _, _) => Self::Andi(Andi::new(rt, rs, fields.imm16())),
                (0xD, _, _, _) => Self::Ori(Ori::new(rt, rs, fields.imm16())),
                (0xE, _, _, _) => Self::Xori(Xori::new(rt, rs, fields.imm16())),
                (0xF, _, _, _) => Self::Lui(Lui::new(rt, fields.imm16())),
                (0x10, 0x10, _, 0x10) => Self::Rfe(Rfe),
                (0x10..=0x13, 0x0, _, 0x0) => {
                    Self::Mfcn(Mfcn::new(fields.cop().value() as _, rt, rd))
                }
                (0x10..=0x13, 0x2, _, 0x0) => Self::Cfcn(Cfcn::new(fields.cop().value(), rt, rd)),
                (0x10..=0x13, 0x4, _, 0x0) => {
                    Self::Mtcn(Mtcn::new(fields.cop().value() as _, rt, rd))
                }
                (0x10..=0x13, 0x8, 0, _) => todo!("bcnf"),
                (0x10..=0x13, 0x8, 1, _) => todo!("bcnt"),
                (0x10..=0x13, 0x6, _, 0x0) => match fields.cop().as_u8() {
                    2 => Self::Ctcn(Ctcn::new(2, rt, rd)),
                    n => unimplemented!("ctc{n}"),
                },
                (0x10..=0x13, 0x10..=0x1F, _, _) => {
                    Self::CopNCmd(CopNCmd::new(fields.cop().as_u8(), fields.imm26().as_u32()))
                }
                (0x14..=0x1F, _, _, _) => Self::illegal(),
                (0x20, _, _, _) => Self::Lb(Lb::new(rt, rs, fields.imm16())),
                (0x21, _, _, _) => Self::Lh(Lh::new(rt, rs, fields.imm16())),
                (0x22, _, _, _) => Self::Lwl(Lwl::new(rt, rs, fields.imm16())),
                (0x23, _, _, _) => Self::Lw(Lw::new(rt, rs, fields.imm16())),
                (0x24, _, _, _) => Self::Lbu(Lbu::new(rt, rs, fields.imm16())),
                (0x25, _, _, _) => Self::Lhu(Lhu::new(rt, rs, fields.imm16())),
                (0x26, _, _, _) => Self::Lwr(Lwr::new(rt, rs, fields.imm16())),
                (0x27, _, _, _) => Self::illegal(),
                (0x28, _, _, _) => Self::Sb(Sb::new(rt, rs, fields.imm16())),
                (0x29, _, _, _) => Self::Sh(Sh::new(rt, rs, fields.imm16())),
                (0x2A, _, _, _) => Self::Swl(Swl::new(rt, rs, fields.imm16())),
                (0x2B, _, _, _) => Self::Sw(Sw::new(rt, rs, fields.imm16())),
                (0x2C..=0x2D, _, _, _) => Self::illegal(),
                (0x2E, _, _, _) => Self::Swr(Swr::new(rt, rs, fields.imm16())),
                (0x2F, _, _, _) => Self::illegal(),
                (0x30..=0x33, _, _, _) => {
                    Self::Lwcn(Lwcn::new(fields.cop().as_u8(), rt, rs, fields.imm16()))
                }
                (0x34..=0x37, _, _, _) => Self::illegal(),
                (0x38..=0x3B, _, _, _) => {
                    Self::Swcn(Swcn::new(fields.cop().as_u8(), rt, rs, fields.imm16()))
                }
                (0x3C..=0x3F, _, _, _) => Self::illegal(),
                _ => Self::illegal(),
            }
        })
    }
}

#[non_exhaustive]
struct EmitAddImm16Args {
    dest:   Reg,
    base:   Reg,
    offset: i16,
    temp:   Option<Reg>,
}

impl<A: Allocator + Copy> Dynarec<A> {
    fn emit_add_imm16(
        &mut self,
        EmitAddImm16Args {
            dest,
            base,
            offset,
            temp,
        }: EmitAddImm16Args,
    ) {
        let temp = temp.unwrap_or(Reg::W(3));
        #[cfg(target_arch = "aarch64")]
        {
            match offset {
                ..0 => {
                    dynasm!(
                        self.asm
                        ; .arch aarch64
                        ; movz W(temp), ext::zero(offset)    // Load as unsigned
                        ; sxth W(temp), W(temp)              // Sign-extend to 32-bit
                        ; add WSP(dest), WSP(base), W(temp)
                    );
                }
                0..4095 => {
                    dynasm!(
                        self.asm
                        ; .arch aarch64
                        ; add WSP(dest), WSP(base), ext::zero(offset)
                    );
                }
                4095.. => {
                    dynasm!(
                        self.asm
                        ; .arch aarch64
                        ; mov W(temp), ext::zero(offset)
                        ; add WSP(dest), WSP(base), W(temp)
                    );
                }
            }
        }
    }

    pub fn emit_imm16_sext(&mut self, dest: Reg, imm: i16) {
        #[cfg(target_arch = "aarch64")]
        {
            match imm {
                ..0 => {
                    dynasm!(
                        self.asm
                        ; .arch aarch64
                        ; movz W(dest), ext::zero(imm)    // Load as unsigned
                        ; sxth W(dest), W(dest)              // Sign-extend to 32-bit
                    );
                }
                0.. => {
                    dynasm!(
                        self.asm
                        ; .arch aarch64
                        ; mov W(dest), ext::zero(imm)
                    );
                }
            }
        }
    }

    pub fn emit_imm16_uext(&mut self, dest: Reg, imm: i16) {
        #[cfg(target_arch = "aarch64")]
        {
            match imm {
                ..0 => {
                    dynasm!(
                        self.asm
                        ; .arch aarch64
                        ; movz W(dest), ext::zero(imm)    // Load as unsigned
                    );
                }
                0.. => {
                    dynasm!(
                        self.asm
                        ; .arch aarch64
                        ; mov W(dest), ext::zero(imm)
                    );
                }
            }
        }
    }
}

#[inline(always)]
fn emit_call<A: Allocator + Copy>(ctx: &mut EmitCtx<A>, emitter: impl Fn(&mut Dynarec<A>)) {
    #[cfg(target_arch = "aarch64")]
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ;; ctx.dynarec.emit_writeback_all()
        ;; ctx.dynarec.reg_alloc.dealloc_all()
        ;; ctx.dynarec.emit_save_critical_registers()
        // ;; let saved = ctx.dynarec.emit_save_volatile_registers()
        ;; emitter(ctx.dynarec)
        // ;; ctx.dynarec.emit_restore_saved_registers(saved.into_iter())
        ;; ctx.dynarec.emit_restore_critical_registers()
    );
}

#[cfg(target_arch = "aarch64")]
fn emit_call_fn_0_arg<A: Allocator + Copy>(ctx: &mut EmitCtx<A>, offset: u32) {
    emit_call(ctx, |dynarec| {
        dynasm!(
            dynarec.asm
            ; .arch aarch64
            ; ldr x3, [x0, #offset]
            ; blr x3
        );
    });
}

#[derive(Debug, Clone, derive_more::Deref)]
#[deref(forward)]
pub struct Rd<'a, T>(&'a LoadedReg<T>);
#[derive(Debug, Clone, derive_more::Deref)]
#[deref(forward)]
pub struct Rs<'a, T>(&'a LoadedReg<T>);
#[derive(Debug, Clone, derive_more::Deref)]
#[deref(forward)]
pub struct Rt<'a, T>(&'a LoadedReg<T>);

pub struct AluRegs<'a> {
    rd: Rd<'a, AllocResultStackless>,
    rs: Rs<'a, AllocResultStackless>,
    rt: Rt<'a, AllocResultStackless>,
}

fn emit_alu_reg<A: Allocator + Copy>(
    mut ctx: EmitCtx<A>,
    rd: u8,
    rs: u8,
    rt: u8,
    alu_op: impl Fn(&mut EmitCtx<A>, AluRegs),
) -> EmitSummary {
    let rsa = ctx.dynarec.emit_load_reg_stackless(rs).lock(ctx.dynarec);
    let rta = ctx.dynarec.emit_load_reg_stackless(rt).lock(ctx.dynarec);
    let rda = ctx.dynarec.alloc_reg_stackless(rd);

    ctx.dynarec.cancel_next_write(rd);
    #[cfg(target_arch = "aarch64")]
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ;; alu_op(&mut ctx, AluRegs { rd: Rd(&rda), rs: Rs(&rsa), rt: Rt(&rta) })
    );

    ctx.dynarec.mark_dirty(rd);

    rsa.unlock(ctx.dynarec);
    rta.unlock(ctx.dynarec);

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

#[allow(clippy::useless_conversion)]
fn emit_shift_by_reg<A: Allocator + Copy>(
    ctx: EmitCtx<A>,
    rd: u8,
    rs: u8,
    rt: u8,
    alu_op: impl Fn(&mut EmitCtx<A>, AluRegs),
) -> EmitSummary {
    if rd == 0 {
        return EmitSummary::default();
    }

    ctx.dynarec.cancel_next_write(rd);

    if rt == 0 {
        return ctx.dynarec.emit_zero(rd);
    }

    if rs == 0 {
        return ctx.dynarec.emit_load_and_move_into(rd, rt);
    }

    emit_alu_reg(ctx, rd, rs, rt, alu_op)
}

pub struct ShiftImm<'a> {
    rd:  Rd<'a, AllocResultStackless>,
    rt:  Rt<'a, AllocResultStackless>,
    imm: u8,
}

fn emit_shift_imm<A: Allocator + Copy>(
    ctx: &mut EmitCtx<A>,
    rd: Guest,
    rt: Guest,
    imm: u8,
    emitter: impl Fn(&mut Dynarec<A>, ShiftImm),
) -> EmitSummary {
    if rd == 0 {
        return EmitSummary::default();
    }

    ctx.dynarec.cancel_next_write(rd);

    if imm == 0 {
        return ctx.dynarec.emit_load_and_move_into(rd, rt);
    }

    let rt = ctx.dynarec.emit_load_reg_stackless(rt).lock(ctx.dynarec);
    let rda = ctx.dynarec.alloc_reg_stackless(rd);

    emitter(
        ctx.dynarec,
        ShiftImm {
            rd: Rd(&rda),
            rt: Rt(&rt),
            imm,
        },
    );

    rt.unlock(ctx.dynarec);
    ctx.dynarec.mark_dirty(rd);

    EmitSummary::default()
}

pub struct AluImm<'a> {
    rt:  Rt<'a, AllocResultStackless>,
    rs:  Rs<'a, AllocResultStackless>,
    imm: i16,
}

fn emit_alu_imm<A: Allocator + Copy>(
    mut ctx: EmitCtx<A>,
    rt: Guest,
    rs: Guest,
    imm: i16,
    alu_op: impl Fn(&mut EmitCtx<A>, AluImm),
) -> EmitSummary {
    let rsa = ctx.dynarec.emit_load_reg_stackless(rs).lock(ctx.dynarec);
    let rta = ctx.dynarec.alloc_reg_stackless(rt);

    ctx.dynarec.cancel_next_write(rt);
    #[cfg(target_arch = "aarch64")]
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ;; alu_op(&mut ctx, AluImm { rt: Rt(&rta), rs: Rs(&rsa), imm})
    );

    rsa.unlock(ctx.dynarec);
    ctx.dynarec.mark_dirty(rt);

    EmitSummary::default()
}

impl<A: Allocator + Copy> Dynarec<A> {
    pub fn emit_set_bt<const VALUE: bool>(&mut self) {
        let cop0reg13 = Cpu::cop0_reg_offset(13) as u32;
        match VALUE {
            true => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    self.asm
                    ; ldr w1, [x0, #cop0reg13]
                    ; orr w1, w1, #0x40000000
                    ; str w1, [x0, #cop0reg13]
                );
            }
            false => {
                dynasm!(
                    self.asm
                    ; ldr w1, [x0, #cop0reg13]
                    ; and w1, w1, #0xbfffffff
                    ; str w1, [x0, #cop0reg13]
                );
            }
        };
    }

    pub fn emit_set_bd<const VALUE: bool>(&mut self) {
        let cop0reg13 = Cpu::cop0_reg_offset(13) as u32;
        match VALUE {
            true => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    self.asm
                    ; ldr w1, [x0, #cop0reg13]
                    ; orr w1, w1, #0x80000000
                    ; str w1, [x0, #cop0reg13]
                );
            }
            false => {
                dynasm!(
                    self.asm
                    ; ldr w1, [x0, #cop0reg13]
                    ; and w1, w1, #0x7fffffff
                    ; str w1, [x0, #cop0reg13]
                );
            }
        };
    }
}

/// `selector` must look something like
/// ```
/// dynasm!(
///     ctx.dynarec.asm
///     ; csel w2, w3, w2, {SELECTOR}
/// );
/// ```
#[allow(clippy::useless_conversion)]
fn emit_branch_zero<A: Allocator + Copy>(
    mut ctx: EmitCtx<A>,
    rs: u8,
    imm: i16,
    selector: impl Fn(&mut EmitCtx<A>) + 'static,
    link: bool,
) -> EmitSummary {
    #[cfg(not(target_arch = "aarch64"))]
    {
        compile_error!("emit_branch_zero: not implemented on arch");
    }

    let s1 = ctx.alloc_scratch();
    let rs = ctx.dynarec.emit_load_reg_stackless(rs);

    // calculate branch value
    #[cfg(target_arch = "aarch64")]
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ; str W(*rs), [x0, #s1]
    );

    let branch_dest = (ctx.pc + 0x4).wrapping_add_signed(ext::sign(imm) << 2);
    let write = if link { Some(RA) } else { None };

    if link {
        let return_dest = ctx.pc + 0x8;
        let ra = ctx.dynarec.alloc_reg_stackless(RA);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; movz w1, return_dest >> 16, LSL #16
            ; movk w1, return_dest & 0x0000_ffff
            ; mov W(*ra), w1
        );

        ctx.dynarec.mark_dirty(RA);
    }

    ctx.schedule_in(1, write, move |mut ctx| {
        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; ldr w2, [x0, #s1]
            ; cmp w2, #0

            ; movz w2, (ctx.pc + 0x8) >> 16, lsl #16
            ; movk w2, (ctx.pc + 0x8) & 0x0000_ffff
            ; movz w3, branch_dest >> 16, lsl #16
            ; movk w3, branch_dest & 0x0000_ffff
            ;; selector(&mut ctx)
            ; str w2, [x0, Emu::<A>::PC_OFFSET as _]
            ;; ctx.dynarec.emit_set_bd::<true>()
        );

        EmitSummary::pc_updated()
    });

    EmitSummary::default()
}

#[repr(u8)]
#[derive(Clone, Copy)]
enum CopRegisterKind {
    Data = 0,
    Ctrl = 1,
}
#[derive(Copy, Clone)]
struct CopMove {
    cop:  u8,
    rd:   u8,
    rt:   u8,
    kind: CopRegisterKind,
}

#[allow(clippy::useless_conversion)]
fn emit_move_to_cop<A: Allocator + Copy>(
    mut ctx: EmitCtx<A>,
    CopMove { cop, rd, rt, kind }: CopMove,
) -> EmitSummary {
    let reg = rd + (kind as u8) * 32;
    match (cop, reg) {
        // N/A
        (0, 0..=2) => {
            tracing::warn!("move to invalid cop0 register {}", rd);
        }
        // read only
        (0, 6 | 8 | 14 | 15) => {}
        // cause reg
        (0, 13) => {
            ctx.dynarec.emit_load_temp_reg(rt, Reg::W(1));
            emit_call(&mut ctx, |dynarec| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; ldr x3, [x0, #ExternFnTable::<A>::COP0_SET_CAUSE]
                    ; blr x3
                );
            });
        }
        // gte screen z fifo mirror register "SXYP"
        (2, 15) => {
            emit_call(&mut ctx, |dynarec| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; ldr x3, [x0, #GteBindings::<A>::WRITE_SXYP]
                    ; blr x3
                );
            });
        }
        // gte irgb register
        (2, 28) => {
            let rt = ctx.dynarec.emit_load_reg_stackless(rt);
            emit_call(&mut ctx, |dynarec| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; mov w1, W(*rt)
                    ; ldr x3, [x0, #GteBindings::<A>::WRITE_IRGB]
                    ; blr x3
                );
            });
        }
        // gte lzcs register
        (2, 30) => {
            let rt = ctx.dynarec.emit_load_reg_stackless(rt);
            emit_call(&mut ctx, |dynarec| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; mov w1, W(*rt)
                    ; ldr x3, [x0, #GteBindings::<A>::WRITE_LZCS]
                    ; blr x3
                );
            });
        }
        (0 | 2, _) => {
            let rt = ctx.dynarec.emit_load_reg_stackless(rt);

            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; nop
                ; str W(*rt), [x0, Cpu::cop_reg_offset(cop, reg) as _]
                ; nop
            );
        }
        _ => {}
    }

    default()
}

impl DynarecOp for CopNCmd {
    fn cycles(&self) -> u16 {
        10
    }

    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, mut ctx: EmitCtx<A>) -> EmitSummary {
        if self.cop == 2 {
            use crate::cpu::gte::GteCommandField;

            let cmd = GteCommandField::new_with_raw_value(self.imm26);
            let Some(func) = Emu::<A>::gte_function(cmd) else {
                return default();
            };

            let cmd_raw = cmd.raw_value();
            emit_call(&mut ctx, |dynarec| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; ldr x3, [x0, #func]
                    ; movz w1, cmd_raw >> 16, lsl #16
                    ; movk w1, cmd_raw & 0x0000_ffff
                    ; blr x3
                )
            });
        }
        EmitSummary::default()
    }
}

#[allow(clippy::useless_conversion)]
#[cfg(target_arch = "aarch64")]
fn emit_move_from_cop<A: Allocator + Copy>(
    mut ctx: EmitCtx<A>,
    CopMove { cop, rd, rt, kind }: CopMove,
) {
    if rt == 0 {
        return;
    }

    ctx.dynarec.cancel_next_write(rt);

    let reg = (kind as u8) * 32 + rd;

    match (cop, reg) {
        (0, _) => {
            let rta = ctx.dynarec.alloc_reg_stackless(rt);

            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr W(*rta), [x0, Cpu::cop_reg_offset(cop, reg) as _]
            );

            ctx.dynarec.mark_dirty(rt);
        }
        // gte screen z fifo mirror register "SXYP"
        (2, 15) => {
            emit_move_to_cop(
                ctx,
                CopMove {
                    cop,
                    rd: 14,
                    rt,
                    kind,
                },
            );
        }
        (2, _) => {
            let rta = ctx.dynarec.alloc_reg_stackless(rt);
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr W(*rta), [x0, Cpu::cop_reg_offset(cop, reg) as _]
            );

            if rt != 0 {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; str W(*rta), [x0, #shadow(rt)]
                );
            }

            // writeback to gpr
            ctx.schedule_in(1, Some(rt), move |ctx| {
                let rta = ctx.dynarec.alloc_reg_stackless(rt);
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; ldr W(*rta), [x0, #shadow(rt)]
                );
                ctx.dynarec.mark_dirty(rt);
                EmitSummary::default()
            });
        }
        _ => todo!(),
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::useless_conversion)]
fn emit_load<const ALIGNED: bool, A: Allocator + Copy>(
    mut ctx: EmitCtx<A>,
    rt: u8,
    rs: u8,
    imm: i16,
    func_call: impl Fn(&mut EmitCtx<A>) + 'static,
) -> EmitSummary {
    ctx.dynarec.cancel_next_write(rt);
    ctx.dynarec.emit_load_temp_reg(rs, Reg::W(1));

    match ALIGNED {
        true => {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ;; ctx.dynarec.emit_writeback_all()
                ;; ctx.dynarec.reg_alloc.dealloc_all()
                ;; ctx.dynarec.emit_save_critical_registers()
                ;; ctx.dynarec.emit_add_imm16(EmitAddImm16Args { dest: Reg::W(1), base: Reg::W(1), offset: imm, temp: None })
                ;; func_call(&mut ctx)
                ; mov w1, w0
                ;; ctx.dynarec.emit_restore_critical_registers()
            );
        }
        false => {
            let rta = ctx.dynarec.emit_load_shadow_reg_stackless(rt);

            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ;; ctx.dynarec.emit_writeback_all()
                ;; ctx.dynarec.reg_alloc.dealloc_all()
                ;; ctx.dynarec.emit_save_critical_registers()
                ;; ctx.dynarec.emit_add_imm16(EmitAddImm16Args { dest: Reg::W(1), base: Reg::W(1), offset: imm, temp: None })
                ; mov w2, W(*rta)
                ;; func_call(&mut ctx)
                ; mov w1, w0
                ;; ctx.dynarec.emit_restore_critical_registers()
            );
        }
    }

    // writeback to shadow register
    if rt != 0 {
        use crate::dynarec_v2::emitters::shadow;

        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; str w1, [x0, #shadow(rt)]
        );
    }

    // writeback to gpr
    ctx.schedule_in(1, Some(rt), move |ctx| {
        if rt != 0 {
            let rta = ctx.dynarec.alloc_reg_stackless(rt);
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr W(*rta), [x0, #shadow(rt)]
            );
            ctx.dynarec.mark_dirty(rt);
        }
        EmitSummary::default()
    });

    EmitSummary::default()
}

#[allow(clippy::useless_conversion)]
fn emit_store<A: Allocator + Copy>(
    mut ctx: EmitCtx<A>,
    rt: u8,
    rs: u8,
    imm: i16,
    func_call: impl Fn(&mut EmitCtx<A>) + 'static,
) -> EmitSummary {
    ctx.dynarec.emit_load_temp_reg(rs, Reg::W(1));
    ctx.dynarec.emit_load_temp_reg(rt, Reg::W(2));

    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64

        // need to track caller saved registers in regalloc
        // we have to store x0 since the function call will clobber it
        // call to function
        ;; ctx.dynarec.emit_writeback_all()
        ;; ctx.dynarec.reg_alloc.dealloc_all()
        ;; ctx.dynarec.emit_save_critical_registers()
        ;; ctx.dynarec.emit_add_imm16(EmitAddImm16Args { dest: Reg::W(1), base: Reg::W(1), offset: imm, temp: None })

        ;; func_call(&mut ctx)
        ;; ctx.dynarec.emit_restore_critical_registers()

    );

    EmitSummary::default()
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
fn emit_branch<A: Allocator + Copy>(
    mut ctx: EmitCtx<A>,
    rs: u8,
    rt: u8,
    imm: i16,
    selector: impl Fn(&mut EmitCtx<A>) + 'static,
) -> EmitSummary {
    let (s1, _) = ctx.alloc_scratch_pair();
    let s1 = s1 as i32;
    let rs = ctx.dynarec.emit_load_reg_stackless(rs).lock(ctx.dynarec);
    let rt = ctx.dynarec.emit_load_reg_stackless(rt);

    // calculate branch value
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ; stp W(*rs), W(*rt), [x0, #s1]
    );

    let branch_dest = (ctx.pc + 0x4).wrapping_add_signed(ext::sign(imm) << 2);
    ctx.schedule_in(1, None, move |mut ctx| {
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; ldp w2, w3, [x0, #s1]
            ; cmp w2, w3

            ; movz w2, (ctx.pc + 0x8) >> 16, lsl #16
            ; movk w2, (ctx.pc + 0x8) & 0x0000_ffff
            ; movz w3, branch_dest >> 16, lsl #16
            ; movk w3, branch_dest & 0x0000_ffff
            ;; selector(&mut ctx)
            ; str w2, [x0, Emu::<A>::PC_OFFSET as _]
            ;; ctx.dynarec.emit_set_bd::<true>()
        );

        EmitSummary::pc_updated()
    });

    rs.unlock(ctx.dynarec);

    EmitSummary::default()
}

#[derive(Clone, Copy)]
enum Hilo {
    Hi,
    Lo,
}

#[allow(clippy::useless_conversion)]
fn emit_mthilo<A: Allocator + Copy>(rs: u8, ctx: EmitCtx<A>, hilo: Hilo) -> EmitSummary {
    let offset = match hilo {
        Hilo::Hi => Emu::<A>::HILO_OFFSET + size_of::<u32>(),
        Hilo::Lo => Emu::<A>::HILO_OFFSET,
    };
    let rs = ctx.dynarec.emit_load_reg_stackless(rs);
    ctx.dynarec.lock_register(&rs);

    // ctx.schedule_in(2, move |ctx| {
    #[cfg(target_arch = "aarch64")]
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ; str W(*rs), [x0, offset as _]
    );
    ctx.dynarec.unlock_register(&rs);
    // EmitSummary::default()
    // });
    EmitSummary::default()
}

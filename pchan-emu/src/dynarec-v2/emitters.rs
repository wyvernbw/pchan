use crate::cpu::*;
use crate::dynarec_v2::{DynEmitter, Guest};
use crate::{Emu, cpu};
use std::num::NonZeroU8;

use crate::dynarec_v2::regalloc::AllocResult;

use arbitrary_int::traits::Integer;
use enum_dispatch::enum_dispatch;

use pchan_utils::hex;
#[cfg(test)]
use rstest::rstest;
use smallbox::SmallBox;

use crate::cpu::ops::{HaltBlock, OpCode, *};
use crate::dynarec_v2::{Dynarec, LoadedReg, Reg};
#[cfg(test)]
use crate::memory;
use crate::memory::ext;
use dynasm::dynasm;
use dynasmrt::{DynasmApi, DynasmLabelApi};

use super::ScheduledEmitter;

#[derive(Debug)]
pub struct EmitCtx<'a> {
    pub dynarec:        &'a mut Dynarec,
    pub pc:             u32,
    pub d_clock:        u32,
    pub delay_slot:     bool,
    pub scratch_cursor: &'a mut u8,
}

const MAX_SCRATCH_REG: u8 = 3;

impl<'a> EmitCtx<'a> {
    fn schedule_in(&mut self, ops: u32, emitter: impl Fn(EmitCtx) -> EmitSummary + 'static) {
        let emitter = SmallBox::new(emitter) as DynEmitter;
        let at = self.pc + ops * 4;
        self.dynarec
            .scheduler
            .queue
            .push(ScheduledEmitter {
                emitter,
                schedule: at,
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
        const MAX_CURSOR: u8 = Cpu::SCRATCH_SIZE as u8 - 2;
        if *self.scratch_cursor == MAX_CURSOR {
            *self.scratch_cursor = 0;
        }
        let a = *self.scratch_cursor as usize;
        let b = a + 1;
        *self.scratch_cursor =
            ((*self.scratch_cursor + 2) as usize % (Cpu::SCRATCH_SIZE - 1)) as u8;
        (
            (Cpu::SCRATCH_OFFSET + a * size_of::<u32>()) as u32,
            (Cpu::SCRATCH_OFFSET + b * size_of::<u32>()) as u32,
        )
    }
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
    fn emit<'a>(&self, _: EmitCtx<'a>) -> EmitSummary {
        EmitSummary::default()
    }

    fn boundary(&self) -> Boundary {
        Boundary::Hard
    }
}

impl DecodedOp {
    pub fn new(fields: OpCode) -> Self {
        let [op] = Self::decode([fields]);
        op
    }
    pub const fn is_illegal(&self) -> bool {
        matches!(self, Self::Illegal(_))
    }
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
                (0x0, _, _, 0xD) => Self::Nop(Nop), // brk
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
                (0x1, _, 0x0, _) => Self::Bltz(Bltz::new(rs, fields.imm16())),
                (0x1, _, 0x1, _) => Self::Bgez(Bgez::new(rs, fields.imm16())),
                (0x1, _, 0x10, _) => Self::Bltzal(Bltzal::new(rs, fields.imm16())),
                (0x1, _, 0x11, _) => Self::Bgezal(Bgezal::new(rs, fields.imm16())),
                // * TODO: bltz and bgez dupes * //
                (0x2, _, _, _) => Self::J(J::new(fields.imm26().value())),
                (0x3, _, _, _) => Self::Jal(Jal::new(fields.imm26().value())),
                (0x4, _, _, _) => Self::Beq(Beq::new(rs, rt, fields.imm16())),
                (0x5, _, _, _) => Self::Bne(Bne::new(rs, rt, fields.imm16())),
                (0x6, _, _, _) => Self::Blez(Blez::new(rs, fields.imm16())),
                (0x7, _, _, _) => Self::Bgtz(Bgtz::new(rs, fields.imm16())),
                (0x8 | 0x9, _, _, _) => Self::Addiu(Addiu::new(rt, rs, fields.imm16())),
                (0xA, _, _, _) => Self::Slti(Slti::new(rt, rs, fields.imm16())),
                (0xB, _, _, _) => Self::Sltiu(Sltiu::new(rt, rs, fields.imm16() as u16)),
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
                    tracing::warn!(
                        "not yet implemented: cop{} imm25 {}",
                        fields.cop(),
                        hex(fields.imm26().value())
                    );
                    DecodedOp::Nop(Nop)
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

impl Dynarec {
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

impl DynarecOp for Addiu {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        // $rt = $zero case
        if self.rt == 0 {
            return EmitSummary::default();
        }

        // $rs = $zero case
        if self.rs == 0 {
            let rt = ctx.dynarec.alloc_reg(self.rt);

            ctx.dynarec.emit_imm16_sext(rt.reg(), self.imm16);

            ctx.dynarec.mark_dirty(self.rt);
            rt.restore(ctx.dynarec);
            return EmitSummary::default();
        }

        let rs = ctx.dynarec.emit_load_reg(self.rs);
        let rt = ctx.dynarec.alloc_reg(self.rt);

        ctx.dynarec.emit_add_imm16(EmitAddImm16Args {
            dest:   rt.reg(),
            base:   rs.reg(),
            offset: self.imm16,
            temp:   None,
        });

        ctx.dynarec.mark_dirty(self.rt);

        rt.restore(ctx.dynarec);
        rs.restore(ctx.dynarec);

        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
#[case(10, 2, 12)]
fn test_addiu(#[case] a: u32, #[case] b: u32, #[case] expected: u32) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = a;
    emu.write_many(
        0x0,
        &program([addiu(12, 10, b as i16), OpCode::new_with_raw_value(69420)]),
    );
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[12], expected);
    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

/// rough idea taken from bios spu related code
#[cfg(test)]
#[rstest]
fn test_addiu_andi_loop() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = 0x0;
    emu.write_many(
        0x0,
        &program([
            addiu(7, 0, 16),
            addiu(8, 8, 1),
            andi(8, 8, 0x00ff),
            sltu(1, 8, 7),
            bne(1, 0, -4),
            nop(),
            addiu(10, 0, 1),
            OpCode::HALT,
        ]),
    );
    while emu.cpu.gpr[10] != 1 {
        tracing::info!(
            r7 = emu.cpu.gpr[7],
            r8 = emu.cpu.gpr[8],
            r10 = emu.cpu.gpr[10]
        );
        PipelineV2::new(&emu).run_once(&mut emu)?;
    }
    assert_eq_hex!(emu.cpu.gpr[8], 16);
    assert_eq_hex!(emu.cpu.gpr[7], 16);
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
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = a;
    emu.cpu.gpr[11] = b;
    emu.write_many(
        0x0,
        &program([subu(12, 10, 11), OpCode::new_with_raw_value(69420)]),
    );
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[12], expected);
    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

#[inline(always)]
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
    mut ctx: EmitCtx,
    rt: u8,
    rs: u8,
    imm: i16,
    func_call: impl Fn(&mut EmitCtx) + 'static,
) -> EmitSummary {
    let (s1, _) = ctx.alloc_scratch_pair();
    let s1 = s1 as i32;
    ctx.dynarec.emit_load_temp_reg(rs, Reg::W(1));
    ctx.dynarec.emit_load_temp_reg(rt, Reg::W(2));

    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ;; ctx.dynarec.emit_add_imm16(EmitAddImm16Args { dest: Reg::W(1), base: Reg::W(1), offset: imm, temp: None })
        // ; stp w1, w2, [sp, #-16]!
        ; stp w1, w2, [x0, #s1]
    );

    ctx.schedule_in(1, move |mut ctx| {
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            // need to track caller saved registers in regalloc
            // we have to store x0 since the function call will clobber it
            // call to function
            ;; let saved = ctx.dynarec.emit_save_volatile_registers()

            ;; if ctx.delay_slot {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; ldr w1, [x0, #196]
                    ; orr w1, w1, #0x80000000
                    ; str w1, [x0, #196]
                )
            }

            ; ldp w1, w2, [x0, #s1]
            ;; func_call(&mut ctx)
            ;; ctx.dynarec.emit_restore_saved_registers(saved.into_iter())

        );
        EmitSummary::default()
    });

    EmitSummary::default()
}

impl DynarecOp for Sb {
    #[cfg(target_arch = "aarch64")]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm16, move |ctx| {
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

impl DynarecOp for Sh {
    #[cfg(target_arch = "aarch64")]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm16, move |ctx| {
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

impl DynarecOp for Sw {
    #[cfg(target_arch = "aarch64")]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm16, move |ctx| {
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
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use crate::memory::ext;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = ext::zero(value);
    emu.cpu.gpr[11] = 0x801ffed0;
    emu.write_many(0x0, &program([instr, OpCode::HALT]));

    PipelineV2::new(&emu).run_once(&mut emu)?;

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.d_clock, 3);
    assert_eq!(emu.cpu.pc, 0x8);
    assert_eq!(
        emu.read_ext::<T, ext::Zero>(0x801ffed0 + 2),
        ext::zero(value)
    );

    tracing::info!("returning from test...");
    Ok(())
}

/// this is a write to BIOS rom which is obviously invalid
/// in the future proper emulation should trigger the exception handler
/// for now we just panic.
#[cfg(test)]
#[rstest]
#[should_panic]
fn test_weird_store() {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = 0xbfc01a60;
    emu.cpu.gpr[cpu::SP as usize] = 0x801ffee8;
    emu.cpu.gpr[cpu::RA as usize] = 0xbfc06ed4;
    emu.write_many(
        emu.cpu.pc,
        &program([
            addiu(cpu::SP, cpu::SP, -0x0018),
            sw(cpu::RA, cpu::SP, 0x0014),
            nop(),
            OpCode::HALT,
        ]),
    );

    PipelineV2::new(&emu).run_once(&mut emu).unwrap();

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    let result = emu.read::<u32>(0x801ffee4);
    assert_eq!(result, 0xbfc06ed4);
}

impl DynarecOp for Swl {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm16, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, ->ulwrite32
                ; blr x3
            );
        })
    }

    fn cycles(&self) -> u16 {
        2
    }
}

impl DynarecOp for Swr {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm16, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, ->urwrite32
                ; blr x3
            );
        })
    }

    fn cycles(&self) -> u16 {
        2
    }
}

#[cfg(test)]
mod test_unaligned_load_stores {
    //! Tests for the `lwl`, `lwr`, `swl` and `swr` instructions.
    //!
    //! The tests assume the following memory setup:
    //! - `[0x0($sp)] = 0x0`
    //! - `[0x1($sp)] = 0x1`
    //! - `[0x2($sp)] = 0x2`
    //! - etc.
    //!
    //! # Load tests
    //!
    //! Load tests will be of the form:
    //!
    //! ```asm
    //! lwl $t1,imm($sp)
    //! lwr $t2,imm($sp)
    //! ```

    use rstest::rstest;

    use crate::cpu::ops::{OpCode, lui, lwl, lwr, ori, swl, swr};
    use crate::cpu::{SP, program};

    const fn load_par_program_one_imm(imm: i16) -> [u32; 2] {
        program([lwl(9, SP, imm), lwr(10, SP, imm)])
    }

    const fn load_seq_program_one_imm(imm: i16) -> [u32; 2] {
        program([lwl(9, SP, imm), lwr(9, SP, imm)])
    }

    const fn load_seq_two_imm(a: i16, b: i16) -> [u32; 2] {
        program([lwl(9, SP, a), lwr(9, SP, b)])
    }

    #[rstest]
    #[case(load_par_program_one_imm(0x0), 0x00ff_ffff, 0x0302_0100)]
    #[case(load_par_program_one_imm(0x1), 0x0100_ffff, 0xff03_0201)]
    #[case(load_par_program_one_imm(0x2), 0x0201_00ff, 0xffff_0302)]
    #[case(load_par_program_one_imm(0x3), 0x0302_0100, 0xffff_ff03)]
    #[case(load_par_program_one_imm(0x4), 0x04ff_ffff, 0x0706_0504)]
    fn test_par_lwl_lwr<const N: usize>(
        #[case] prog: [u32; N],
        #[case] t1: u32,
        #[case] t2: u32,
    ) -> color_eyre::Result<()> {
        use crate::Emu;
        use crate::dynarec_v2::PipelineV2;
        use assert_hex::assert_eq_hex;
        use pchan_utils::setup_tracing;

        setup_tracing();
        let mut emu = Emu::default();
        emu.cpu["$sp"] = 0x8000_00f0;
        emu.cpu.gpr[9] = 0xffff_ffff;
        emu.cpu.gpr[10] = 0xffff_ffff;
        for i in 0x0..0x10 {
            emu.write(emu.cpu["$sp"] + i, i);
        }
        let prog = [prog.as_slice(), [OpCode::HALT.raw_value()].as_slice()].concat();
        emu.write_many(0x0, &prog);

        PipelineV2::new(&emu).run_once(&mut emu)?;

        tracing::info!("finished running");
        tracing::info!(?emu.cpu);

        assert_eq_hex!(emu.cpu.gpr[9], t1);
        assert_eq_hex!(emu.cpu.gpr[10], t2);

        Ok(())
    }

    #[rstest]
    #[case(load_seq_program_one_imm(0x0), 0x0302_0100)]
    #[case(load_seq_program_one_imm(0x1), 0x0103_0201)]
    #[case(load_seq_program_one_imm(0x2), 0x0201_0302)]
    #[case(load_seq_program_one_imm(0x3), 0x0302_0103)]
    #[case(load_seq_program_one_imm(0x4), 0x0706_0504)]
    // ulw
    #[case(load_seq_two_imm(0x3, 0x0), 0x0302_0100)]
    #[case(load_seq_two_imm(0x4, 0x1), 0x0403_0201)]
    #[case(load_seq_two_imm(0x5, 0x2), 0x0504_0302)]
    #[case(load_seq_two_imm(0x6, 0x3), 0x0605_0403)]
    fn test_seq_lwl_lwr<const N: usize>(
        #[case] prog: [u32; N],
        #[case] t1: u32,
    ) -> color_eyre::Result<()> {
        use crate::Emu;
        use crate::dynarec_v2::PipelineV2;
        use assert_hex::assert_eq_hex;
        use pchan_utils::setup_tracing;

        setup_tracing();
        let mut emu = Emu::default();
        emu.cpu["$sp"] = 0x8000_00f0;
        emu.cpu.gpr[9] = 0xffff_ffff;
        for i in 0x0..0x10 {
            emu.write(emu.cpu["$sp"] + i, i);
        }
        let prog = [prog.as_slice(), [OpCode::HALT.raw_value()].as_slice()].concat();
        emu.write_many(0x0, &prog);

        PipelineV2::new(&emu).run_once(&mut emu)?;

        tracing::info!("finished running");
        tracing::info!(?emu.cpu);

        assert_eq_hex!(emu.cpu.gpr[9], t1);

        Ok(())
    }

    fn single_write_program(imm: i16, op: impl const Fn(u8, u8, i16) -> OpCode) -> [u32; 3] {
        program([lui(9, 0x0302), ori(9, 9, 0x0100), op(9, SP, imm)])
    }

    fn two_writes_program(imm: i16) -> [u32; 4] {
        program([
            lui(9, 0x0302),
            ori(9, 9, 0x0100),
            swl(9, SP, imm),
            swr(9, SP, imm),
        ])
    }

    #[rstest]
    #[case(single_write_program(0x0, swl), 0xffff_ff03)]
    #[case(single_write_program(0x1, swl), 0xffff_0302)]
    #[case(single_write_program(0x2, swl), 0xff03_0201)]
    #[case(single_write_program(0x3, swl), 0x0302_0100)]
    #[case(single_write_program(0x4, swl), 0xffff_ffff)]
    #[case(single_write_program(0x0, swr), 0x0302_0100)]
    #[case(single_write_program(0x1, swr), 0x0201_00ff)]
    #[case(single_write_program(0x2, swr), 0x0100_ffff)]
    #[case(single_write_program(0x3, swr), 0x00ff_ffff)]
    #[case(single_write_program(0x4, swr), 0xffff_ffff)]
    #[case(two_writes_program(0x0), 0x0302_0100)]
    #[case(two_writes_program(0x1), 0x0201_0002)]
    #[case(two_writes_program(0x2), 0x0100_0201)]
    #[case(two_writes_program(0x3), 0x0002_0100)]
    fn test_single_write<const N: usize>(
        #[case] prog: [u32; N],
        #[case] word: u32,
    ) -> color_eyre::Result<()> {
        use crate::Emu;
        use crate::dynarec_v2::PipelineV2;
        use assert_hex::assert_eq_hex;
        use pchan_utils::setup_tracing;

        setup_tracing();
        let mut emu = Emu::default();
        emu.cpu["$sp"] = 0x8000_00f0;
        for i in 0x0..0x10 {
            emu.write::<u8>(emu.cpu["$sp"] + i, 0xff);
        }
        let prog = [prog.as_slice(), [OpCode::HALT.raw_value()].as_slice()].concat();
        emu.write_many(0x0, &prog);

        PipelineV2::new(&emu).run_once(&mut emu)?;

        tracing::info!("finished running");
        tracing::info!(?emu.cpu);

        let written = emu.read::<u32>(emu.cpu["$sp"]);
        assert_eq_hex!(written, word);

        Ok(())
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::useless_conversion)]
fn emit_load<const ALIGNED: bool>(
    mut ctx: EmitCtx,
    rt: u8,
    rs: u8,
    imm: i16,
    func_call: impl Fn(&mut EmitCtx) + 'static,
) -> EmitSummary {
    let s1 = ctx.alloc_scratch();
    ctx.dynarec.emit_load_temp_reg(rs, Reg::W(1));
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ;; ctx.dynarec.emit_add_imm16(EmitAddImm16Args { dest: Reg::W(1), base: Reg::W(1), offset: imm, temp: None })
        ; str w1, [x0, #s1]
    );

    ctx.schedule_in(1, move |mut ctx| {
        if ALIGNED {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                // ; ldr w1, [sp], #16
                ;; let saved = ctx.dynarec.emit_save_volatile_registers()
                ; ldr w1, [x0, #s1]
                ;; func_call(&mut ctx)
                ; mov w1, w0
                ;; ctx.dynarec.emit_restore_saved_registers(saved.into_iter())
                ; str w1, [x0, #s1]
            );
        } else {
            let rt = ctx.dynarec.emit_load_reg(rt);
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ;; let saved = ctx.dynarec.emit_save_volatile_registers()
                ; ldr w1, [x0, #s1]
                ; mov w2, W(*rt)
                ;; func_call(&mut ctx)
                ; mov w1, w0
                ;; ctx.dynarec.emit_restore_saved_registers(saved.into_iter())
                ; str w1, [x0, #s1]
            );
        }

        if rt != 0 {
            let rta = ctx.dynarec.alloc_reg(rt);
            dynasm!(
                ctx.dynarec.asm
                ; ldr W(*rta), [x0, #s1]
            );
            ctx.dynarec.mark_dirty(rt);
            rta.restore(ctx.dynarec);
        }

        EmitSummary::default()
    });

    EmitSummary::default()
}

impl DynarecOp for Lb {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_load::<true>(ctx, self.rt, self.rs, self.imm16, move |ctx| {
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

impl DynarecOp for Lbu {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_load::<true>(ctx, self.rt, self.rs, self.imm16, move |ctx| {
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

impl DynarecOp for Lh {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_load::<true>(ctx, self.rt, self.rs, self.imm16, move |ctx| {
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

impl DynarecOp for Lhu {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_load::<true>(ctx, self.rt, self.rs, self.imm16, move |ctx| {
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

impl DynarecOp for Lw {
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_load::<true>(ctx, self.rt, self.rs, self.imm16, move |ctx| {
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
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[11] = 0x801ffed0;
    emu.write(0x801ffed0 + 2, value);
    emu.write_many(0x0, &program([instr(10, 11, 2), OpCode::HALT]));

    PipelineV2::new(&emu).run_once(&mut emu)?;

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.d_clock, 4);
    assert_eq!(emu.cpu.pc, 0x8);
    assert_eq!(emu.cpu.gpr[10], expected);

    tracing::info!("returning from test...");
    Ok(())
}

/// this will always try to load 0xcafe_babe
#[cfg(test)]
#[rstest]
#[case(lhu, 0x0, 0xbabe)]
#[case(lhu, 0x2, 0xcafe)]
fn test_partial_loads(
    #[case] instr: impl Fn(u8, u8, i16) -> OpCode,
    #[case] immediate: i16,
    #[case] expected: u32,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = 0x0;
    emu.cpu.gpr[11] = 0x801ffed0;
    emu.write::<u32>(0x801ffed0, 0xcafe_babeu32);
    emu.write_many(0x0, &program([instr(10, 11, immediate), OpCode::HALT]));

    PipelineV2::new(&emu).run_once(&mut emu)?;

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq_hex!(emu.cpu.d_clock, 4);
    assert_eq_hex!(emu.cpu.pc, 0x8);
    assert_eq_hex!(emu.cpu.gpr[10], expected);

    tracing::info!("returning from test...");
    Ok(())
}

#[cfg(test)]
#[rstest]
fn test_weird_load_01() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[10] = 0xf;
    emu.cpu.gpr[11] = 0x801ffed0;
    emu.write::<u32>(0x801ffcd8, 0x0d);
    emu.write::<u32>(0x801ffcd9, 0x0);
    emu.write::<u32>(0x801ffcda, 0x0);
    emu.write::<u32>(0x801ffcdb, 0x0);
    // 0x801ffc78 + 0x62
    emu.write_many(
        0x0,
        &program([
            lui(11, 0x801f_u16 as i16),
            ori(11, 11, 0xfc78_u16 as i16),
            lh(10, 11, 0x62),
            nop(),
            OpCode::HALT,
        ]),
    );

    PipelineV2::new(&emu).run_once(&mut emu)?;

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq_hex!(emu.cpu.gpr[10], 0x0000);

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
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
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
            OpCode::HALT,
        ]),
    );

    PipelineV2::new(&emu).run_once(&mut emu)?;

    tracing::info!("finished running");
    tracing::info!(?emu.cpu);

    assert_eq!(emu.cpu.d_clock, 6);
    assert_eq!(emu.cpu.pc, 0x10);
    assert_eq!(emu.cpu.gpr[10], 69);
    assert_eq!(emu.cpu.gpr[12], 420);
    assert_eq!(emu.cpu.gpr[13], 69 + 420);

    tracing::info!("returning from test...");
    Ok(())
}

impl DynarecOp for Lwl {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_load::<false>(ctx, self.rt, self.rs, self.imm16, |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, ->ulread32
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

impl DynarecOp for Lwr {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_load::<false>(ctx, self.rt, self.rs, self.imm16, |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, ->urread32
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
    rd: Rd<'a, AllocResult>,
    rs: Rs<'a, AllocResult>,
    rt: Rt<'a, AllocResult>,
}

fn emit_alu_reg(
    mut ctx: EmitCtx,
    rd: u8,
    rs: u8,
    rt: u8,
    alu_op: impl Fn(&mut EmitCtx, AluRegs),
) -> EmitSummary {
    let rsa = ctx.dynarec.emit_load_reg(rs);
    let rta = ctx.dynarec.emit_load_reg(rt);
    let rda = ctx.dynarec.alloc_reg(rd);

    #[cfg(target_arch = "aarch64")]
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ;; alu_op(&mut ctx, AluRegs { rd: Rd(&rda), rs: Rs(&rsa), rt: Rt(&rta) })
    );

    ctx.dynarec.mark_dirty(rd);

    rda.restore(ctx.dynarec);
    rta.restore(ctx.dynarec);
    rsa.restore(ctx.dynarec);

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

impl DynarecOp for Subu {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        };

        if self.rt == 0 {
            ctx.dynarec.emit_load_temp_reg(self.rs, Reg::W(1));
            let rd = ctx.dynarec.alloc_reg(self.rd);

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

impl DynarecOp for Addu {
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

impl DynarecOp for And {
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

impl DynarecOp for Or {
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

impl DynarecOp for Xor {
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

impl DynarecOp for Nor {
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
                ctx.dynarec.emit_load_temp_reg(reg, Reg::W(1));
                let rd = ctx.dynarec.alloc_reg(target);
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

impl DynarecOp for Sllv {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_shift_by_reg(ctx, self.rd, self.rs, self.rt, move |ctx, regs| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; lsl W(**regs.rd), W(**regs.rt), W(**regs.rs)
            )
        })
    }
}

impl DynarecOp for Srlv {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_shift_by_reg(ctx, self.rd, self.rs, self.rt, move |ctx, regs| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; lsr W(**regs.rd), W(**regs.rt), W(**regs.rs)
            )
        })
    }
}

impl DynarecOp for Srav {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_shift_by_reg(ctx, self.rd, self.rs, self.rt, move |ctx, regs| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; asr W(**regs.rd), W(**regs.rt), W(**regs.rs)
            )
        })
    }
}

pub struct ShiftImm<'a> {
    rd:  Rd<'a, AllocResult>,
    rt:  Rt<'a, AllocResult>,
    imm: u8,
}

fn emit_shift_imm(
    ctx: &mut EmitCtx,
    rd: Guest,
    rt: Guest,
    imm: u8,
    emitter: impl Fn(&mut Dynarec, ShiftImm),
) -> EmitSummary {
    if rd == 0 {
        return EmitSummary::default();
    }

    if imm == 0 {
        return ctx.dynarec.emit_load_and_move_into(rd, rt);
    }

    let rt = ctx.dynarec.emit_load_reg(rt);
    let rda = ctx.dynarec.alloc_reg(rd);

    emitter(
        ctx.dynarec,
        ShiftImm {
            rd: Rd(&rda),
            rt: Rt(&rt),
            imm,
        },
    );

    ctx.dynarec.mark_dirty(rd);

    rda.restore(ctx.dynarec);
    rt.restore(ctx.dynarec);

    EmitSummary::default()
}

impl DynarecOp for Sll {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        emit_shift_imm(
            &mut ctx,
            self.rd,
            self.rt,
            self.shamt,
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

#[rstest]
#[cfg(test)]
fn test_sll_case0() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = 0x0;
    emu.write_many(
        0x0,
        &program([addiu(8, 0, 0x4), sll(9, 8, 0x10), OpCode::HALT]),
    );
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq_hex!(emu.cpu.gpr[8], 0x4);
    assert_eq_hex!(emu.cpu.gpr[9], 0x4 << 0x10);
    assert_eq_hex!(emu.cpu.gpr[9], emu.cpu.gpr[8] << 0x10);

    Ok(())
}

impl DynarecOp for Srl {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        emit_shift_imm(
            &mut ctx,
            self.rd,
            self.rt,
            self.shamt,
            move |dynarec, ShiftImm { rd, rt, imm }| {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    dynarec.asm
                    ; .arch aarch64
                    ; lsr W(**rd), W(**rt), ext::zero(imm)
                );
                dynarec.mark_dirty(self.rd);
            },
        )
    }
}

impl DynarecOp for Sra {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        emit_shift_imm(
            &mut ctx,
            self.rd,
            self.rt,
            self.shamt,
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
    rt:  Rt<'a, AllocResult>,
    rs:  Rs<'a, AllocResult>,
    imm: i16,
}

fn emit_alu_imm(
    mut ctx: EmitCtx,
    rt: Guest,
    rs: Guest,
    imm: i16,
    alu_op: impl Fn(&mut EmitCtx, AluImm),
) -> EmitSummary {
    let rsa = ctx.dynarec.emit_load_reg(rs);
    let rta = ctx.dynarec.alloc_reg(rt);

    #[cfg(target_arch = "aarch64")]
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ;; alu_op(&mut ctx, AluImm { rt: Rt(&rta), rs: Rs(&rsa), imm})
    );

    ctx.dynarec.mark_dirty(rt);

    rta.restore(ctx.dynarec);
    rsa.restore(ctx.dynarec);

    EmitSummary::default()
}

impl DynarecOp for Andi {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        match (self.rt, self.rs, self.imm16) {
            (0, _, _) => EmitSummary::default(),
            (_, 0, _) | (_, _, 0) => ctx.dynarec.emit_zero(self.rt),
            _ => emit_alu_imm(
                ctx,
                self.rt,
                self.rs,
                self.imm16 as _,
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

impl DynarecOp for Ori {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        match self {
            Self {
                rt: 0,
                rs: _,
                imm16: _,
            } => EmitSummary::default(),
            Self {
                rt: _,
                rs: 0,
                imm16: _,
            } => ctx.dynarec.emit_immediate_uext(self.rt, self.imm16),
            Self {
                rt: _,
                rs: _,
                imm16: 0,
            } => ctx.dynarec.emit_load_and_move_into(self.rt, self.rs),
            _ => emit_alu_imm(
                ctx,
                self.rt,
                self.rs,
                self.imm16 as _,
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

impl DynarecOp for Xori {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        match self {
            Self {
                rt: 0,
                rs: _,
                imm16: _,
            } => EmitSummary::default(),
            Self {
                rt: _,
                rs: 0,
                imm16: _,
            } => ctx.dynarec.emit_immediate_uext(self.rt, self.imm16),
            Self {
                rt: _,
                rs: _,
                imm16: 0,
            } => ctx.dynarec.emit_load_and_move_into(self.rt, self.rs),
            _ => emit_alu_imm(
                ctx,
                self.rt,
                self.rs,
                self.imm16 as _,
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

impl DynarecOp for Lui {
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
            ; movz W(*rt), ext::zero(self.imm16) as _, LSL #16
        );

        ctx.dynarec.mark_dirty(self.rt);
        rt.restore(ctx.dynarec);

        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
#[case(0x0, 12, 0x1234, 0x12340000)]
#[case(0x5678, 9, 0x1234, 0x12340000)]
#[case(0x0, 0, 0x1234, 0x0)]
fn test_lui(
    #[case] initial: u32,
    #[case] rt: Guest,
    #[case] imm: i16,
    #[case] expected: u32,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    if rt != 0 {
        emu.cpu.gpr[rt as usize] = initial;
    }
    emu.write_many(0x0, &program([lui(rt, imm), OpCode::HALT]));
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[rt as usize], expected);
    assert_eq!(emu.cpu.d_clock, 2);
    assert_eq!(emu.cpu.pc, 0x8);

    Ok(())
}

impl Dynarec {
    pub fn emit_set_bt<const VALUE: bool>(&mut self) {
        match VALUE {
            true => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    self.asm
                    ; ldr w1, [x0, #196]
                    ; orr w1, w1, #0x40000000
                    ; str w1, [x0, #192]
                );
            }
            false => {
                dynasm!(
                    self.asm
                    ; ldr w1, [x0, #196]
                    ; and w1, w1, #0xbfffffff
                    ; str w1, [x0, #192]
                );
            }
        };
    }
}

impl DynarecOp for J {
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        let new_pc = (self.imm26 << 2) | (ctx.pc & 0xf0000000);
        ctx.schedule_in(1, move |ctx| {
            ctx.dynarec.emit_write_pc(Reg::W(1), new_pc);
            ctx.dynarec.emit_set_bt::<true>();

            // #[cfg(target_arch = "aarch64")]
            // dynasm!(
            //     ctx.dynarec.asm
            //     ; ldr x3, ->jump // defined in prelude
            //     ; str x0, [sp, #-16]!
            //     ;; ctx.dynarec.emit_write_pc(Reg::W(1), new_pc)
            //     ; blr x3
            //     ; mov x3, x0
            //     ; ldr x0, [sp], #16

            //     ; cbz x3, >miss

            //     ;; ctx.drain_schedule()
            //     ;; ctx.dynarec.emit_block_epilogue(ctx.d_clock, None, false)
            //     ; br x3
            //     ; miss:
            // );
            EmitSummary::pc_updated()
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

impl DynarecOp for Jal {
    fn cycles(&self) -> u16 {
        3
    }
    fn hazard(&self) -> u16 {
        2
    }
    fn boundary(&self) -> Boundary {
        Boundary::Soft
    }
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        let new_pc = (self.imm26 << 2) + (ctx.pc & 0xf0000000);
        let return_address = ctx.pc + 0x8;
        ctx.schedule_in(1, move |ctx| {
            ctx.dynarec.emit_write_pc(Reg::W(3), new_pc);
            ctx.dynarec.emit_immediate_large(cpu::RA, return_address);

            EmitSummary::pc_updated()
        });
        EmitSummary::pc_updated()
    }
}

#[cfg(test)]
#[rstest]
#[case(0x0, 0x0000_1000)]
fn test_jal(#[case] initial_pc: u32, #[case] jump_imm: u32) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.write_many(
        initial_pc,
        &program([
            jal(jump_imm as _),
            addiu(9, 0, 69),
            addiu(9, 0, 420),
            OpCode::HALT,
        ]),
    );
    let new_pc = (jump_imm << 2) + (emu.cpu.pc & 0xf0000000);
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[9], 69);
    assert_eq!(emu.cpu.d_clock, 4);
    assert_eq!(emu.cpu.pc, new_pc);
    assert_eq!(emu.cpu.gpr[cpu::RA as usize], initial_pc + 0x8);

    Ok(())
}

impl DynarecOp for Jr {
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
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        let dest = ctx.dynarec.emit_load_reg(self.rs);
        let s1 = ctx.alloc_scratch();
        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            // ; str W(*dest), [sp, #-16]!
            ; str W(*dest), [x0, #s1]
        );

        ctx.schedule_in(1, move |ctx| {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr w1, [x0, #s1]
                ; str w1, [x0, Emu::PC_OFFSET as _]
            );

            EmitSummary::pc_updated()
        });
        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
#[case(0x0, (9, 0x0000_1000))]
fn test_jr(#[case] initial_pc: u32, #[case] rs: (Guest, u32)) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.cpu.gpr[rs.0 as usize] = rs.1;
    emu.write_many(
        initial_pc,
        &program([jr(rs.0), addiu(9, 0, 69), addiu(9, 0, 420), OpCode::HALT]),
    );
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[9], 69);
    assert_eq!(emu.cpu.d_clock, 4);
    assert_eq!(emu.cpu.pc, rs.1);

    Ok(())
}

#[cfg(test)]
#[rstest]
fn test_jr_delay_slot() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = 0x0;
    emu.write_many(
        0x0,
        &program([
            addiu(8, 0, 0x20),
            nop(),
            jr(8),
            addu(9, 0, 8),
            nop(),
            OpCode::HALT,
            nop(),
            nop(),
            OpCode::HALT,
        ]),
    );
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq_hex!(emu.cpu.gpr[8], 0x20);
    assert_eq_hex!(emu.cpu.gpr[9], 0x20);
    assert_eq_hex!(emu.cpu.gpr[9], emu.cpu.gpr[8]);
    Ok(())
}

impl DynarecOp for Jalr {
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
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        let s1 = ctx.alloc_scratch();
        let dest = ctx.dynarec.emit_load_reg(self.rs);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; str W(*dest), [x0, #s1]
        );

        let rd = self.rd;
        // we know the delay slots runs at old pc + 4
        ctx.schedule_in(1, move |ctx| {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr w1, [x0, #s1]
                ; str w1, [x0, Emu::PC_OFFSET as _]
            );

            let ret_address = ctx.pc + 0x8;
            if rd != 0 {
                ctx.dynarec.emit_immediate_large(rd, ret_address);
            }

            EmitSummary::pc_updated()
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
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
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
            OpCode::HALT,
        ]),
    );
    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.gpr[9], 69);
    assert_eq!(emu.cpu.d_clock, 4);
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
    mut ctx: EmitCtx,
    rs: u8,
    rt: u8,
    imm: i16,
    selector: impl Fn(&mut EmitCtx) + 'static,
) -> EmitSummary {
    let (s1, _) = ctx.alloc_scratch_pair();
    let s1 = s1 as i32;
    let rs = ctx.dynarec.emit_load_reg(rs);
    let rt = ctx.dynarec.emit_load_reg(rt);

    // calculate branch value
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ; stp W(*rs), W(*rt), [x0, #s1]
    );

    let branch_dest = (ctx.pc + 0x4).wrapping_add_signed(ext::sign(imm) << 2);
    ctx.schedule_in(1, move |mut ctx| {
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
            ; str w2, [x0, Emu::PC_OFFSET as _]
        );

        EmitSummary::pc_updated()
    });

    rt.restore(ctx.dynarec);
    rs.restore(ctx.dynarec);

    EmitSummary::default()
}

impl DynarecOp for Beq {
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
        {
            // let mut ctx = ctx;
            // let rs = self.rs;
            // let rt = self.rt;
            // let imm = self.imm16;
            // let s = ctx.parity();
            // let rs = ctx.dynarec.emit_load_reg(rs);
            // let rt = ctx.dynarec.emit_load_reg(rt);

            // // calculate branch value
            // dynasm!(
            //     ctx.dynarec.asm
            //     ; .arch aarch64
            //     // ; stp W(*rs), W(*rt), [sp, #-16]!
            //     ; fmov S(s(9)), W(*rs)
            //     ; fmov S(s(10)), W(*rt)
            // );

            // let branch_dest = (ctx.pc + 0x4).wrapping_add_signed(ext::sign(imm) << 2);
            // ctx.schedule_in(1)
            //     .emitter(move |mut ctx| {
            //         dynasm!(
            //             ctx.dynarec.asm
            //             ; .arch aarch64
            //             // ; ldp w2, w3, [sp], #16
            //             ; fmov w2, S(s(9))
            //             ; fmov w3, S(s(10))
            //             ; cmp w2, w3

            //             ; b.eq >taken
            //             ; movz w1, (ctx.pc + 0x8) >> 16, lsl #16
            //             ; movk w1, (ctx.pc + 0x8) & 0x0000_ffff
            //             ; b >got_pc
            //             ; taken:
            //             ; movz w1, branch_dest >> 16, lsl #16
            //             ; movk w1, branch_dest & 0x0000_ffff
            //             ; got_pc:
            //             ; str w1, [x0, Emu::PC_OFFSET as _]

            //             ; ldr x3, ->jump
            //             ; str x0, [sp, #-16]!
            //             ; blr x3
            //             ; mov x3, x0
            //             ; ldr x0, [sp], #16

            //             ; cbz x3, >miss
            //             ;; ctx.drain_schedule()
            //             ;; ctx.dynarec.emit_block_epilogue(ctx.d_clock, None, false)
            //             ; br x3
            //             ; miss:
            //         );

            //         EmitSummary::builder().pc_updated(true).build()
            //     })
            //     .call();

            // rt.restore(ctx.dynarec);
            // rs.restore(ctx.dynarec);

            // EmitSummary::default()
        }
        #[cfg(target_arch = "aarch64")]
        emit_branch(ctx, self.rs, self.rt, self.imm16, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; csel w2, w3, w2, eq
            )
        })
    }
}

impl DynarecOp for Bne {
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
        emit_branch(ctx, self.rs, self.rt, self.imm16, move |ctx| {
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
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.cpu.gpr[rs.0 as usize] = rs.1;
    emu.cpu.gpr[rt.0 as usize] = rt.1;

    emu.write_many(initial_pc, &program([instr(rs.0, rt.0, offset)]));
    emu.write_many(initial_pc + ext::zero(offset), &program([OpCode::HALT]));

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
fn emit_branch_zero(
    mut ctx: EmitCtx,
    rs: u8,
    imm: i16,
    selector: impl Fn(&mut EmitCtx) + 'static,
    link: bool,
) -> EmitSummary {
    #[cfg(not(target_arch = "aarch64"))]
    {
        compile_error!("emit_branch_zero: not implemented on arch");
    }

    let s1 = ctx.alloc_scratch();
    let rs = ctx.dynarec.emit_load_reg(rs);

    // calculate branch value
    #[cfg(target_arch = "aarch64")]
    dynasm!(
        ctx.dynarec.asm
        ; .arch aarch64
        ; str W(*rs), [x0, #s1]
    );

    let branch_dest = (ctx.pc + 0x4).wrapping_add_signed(ext::sign(imm) << 2);
    ctx.schedule_in(1, move |mut ctx| {
        if link {
            let return_dest = ctx.pc + 0x8;
            let ra = ctx.dynarec.alloc_reg(RA);

            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; movz w1, return_dest >> 16, LSL #16
                ; movk w1, return_dest & 0x0000_ffff
                ; mov W(*ra), w1
            );

            ctx.dynarec.mark_dirty(RA);
            ra.restore(ctx.dynarec);
        }

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
            ; str w2, [x0, Emu::PC_OFFSET as _]
        );

        EmitSummary::pc_updated()
    });

    rs.restore(ctx.dynarec);

    EmitSummary::default()
}

impl DynarecOp for Bltz {
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
        emit_branch_zero(
            ctx,
            self.rs,
            self.imm16,
            move |ctx| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; csel w2, w3, w2, lt
                )
            },
            false,
        )
    }
}

impl DynarecOp for Bgez {
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
        emit_branch_zero(
            ctx,
            self.rs,
            self.imm16,
            move |ctx| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; csel w2, w3, w2, ge
                )
            },
            false,
        )
    }
}

impl DynarecOp for Blez {
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
        emit_branch_zero(
            ctx,
            self.rs,
            self.imm16,
            move |ctx| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; csel w2, w3, w2, le
                )
            },
            false,
        )
    }
}

impl DynarecOp for Bgtz {
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
        emit_branch_zero(
            ctx,
            self.rs,
            self.imm16,
            move |ctx| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; csel w2, w3, w2, gt
                )
            },
            false,
        )
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
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.pc = initial_pc;
    emu.cpu.gpr[rs.0 as usize] = rs.1;

    emu.write_many(initial_pc, &program([instr(rs.0, offset)]));
    emu.write_many(initial_pc + ext::zero(offset), &program([OpCode::HALT]));

    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    assert_eq!(emu.cpu.pc, expected_pc);

    Ok(())
}

impl DynarecOp for Mtcn {
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

impl DynarecOp for Ctcn {
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
            ; str W(*rt), [x0, Cpu::cop_reg_offset(self.cop, self.rd + 32) as _]
        );

        rt.restore(ctx.dynarec);
        EmitSummary::default()
    }
}

impl DynarecOp for Nop {
    fn cycles(&self) -> u16 {
        1
    }
    fn emit<'a>(&self, _: EmitCtx<'a>) -> EmitSummary {
        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
fn test_store_loop() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
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
            OpCode::HALT,
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

impl DynarecOp for Sltu {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        }

        let rs = ctx.dynarec.emit_load_reg(self.rs);
        let rt = ctx.dynarec.emit_load_reg(self.rt);
        let rd = ctx.dynarec.alloc_reg(self.rd);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; cmp W(*rs), W(*rt)
            ; cset W(*rd), lo
        );

        ctx.dynarec.mark_dirty(self.rd);

        rd.restore(ctx.dynarec);
        rt.restore(ctx.dynarec);
        rs.restore(ctx.dynarec);

        EmitSummary::default()
    }
}

impl DynarecOp for Sltiu {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rt == 0 {
            return EmitSummary::default();
        }

        let rt = ctx.dynarec.emit_load_reg(self.rt);
        let rs = ctx.dynarec.emit_load_reg(self.rs);

        match self.imm16 {
            0..4096 => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; cmp WSP(*rs), ext::zero(self.imm16)
                    ; cset W(*rt), lo
                )
            }
            4096.. => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; mov w1, ext::zero(self.imm16)
                    ; cmp WSP(*rs), w1
                    ; cset W(*rt), lo
                )
            }
        };

        ctx.dynarec.mark_dirty(self.rt);

        rs.restore(ctx.dynarec);
        rt.restore(ctx.dynarec);

        EmitSummary::default()
    }
}

impl DynarecOp for Slt {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        }

        let rs = ctx.dynarec.emit_load_reg(self.rs);
        let rt = ctx.dynarec.emit_load_reg(self.rt);
        let rd = ctx.dynarec.alloc_reg(self.rd);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; cmp W(*rs), W(*rt)
            ; cset W(*rd), lt
        );

        ctx.dynarec.mark_dirty(self.rd);

        rd.restore(ctx.dynarec);
        rt.restore(ctx.dynarec);
        rs.restore(ctx.dynarec);

        EmitSummary::default()
    }
}

impl DynarecOp for Slti {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rt == 0 {
            return EmitSummary::default();
        }

        let rt = ctx.dynarec.emit_load_reg(self.rt);
        let rs = ctx.dynarec.emit_load_reg(self.rs);

        match self.imm16 {
            ..0 => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; mov w1, ext::zero(self.imm16)
                    ; sxth w1, w1
                    ; cmp WSP(*rs), w1
                    ; cset W(*rt), lt
                );
            }
            0..4096 => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; cmp WSP(*rs), ext::zero(self.imm16)
                    ; cset W(*rt), lt
                );
            }
            4096.. => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; mov w1, ext::zero(self.imm16)
                    ; cmp WSP(*rs), w1
                    ; cset W(*rt), lt
                );
            }
        }
        ctx.dynarec.mark_dirty(self.rt);

        rs.restore(ctx.dynarec);
        rt.restore(ctx.dynarec);

        EmitSummary::default()
    }
}

impl DynarecOp for Mfcn {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rt == 0 {
            return EmitSummary::default();
        }
        let rt = ctx.dynarec.alloc_reg(self.rt);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; ldr W(*rt), [x0, Cpu::cop_reg_offset(self.cop, self.rd) as _]
        );

        ctx.dynarec.mark_dirty(self.rt);

        rt.restore(ctx.dynarec);

        EmitSummary::default()
    }
}

impl DynarecOp for Cfcn {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rt == 0 {
            return EmitSummary::default();
        }
        let rt = ctx.dynarec.alloc_reg(self.rt);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; ldr W(*rt), [x0, Cpu::cop_reg_offset(self.cop, self.rd) as _]
        );

        ctx.dynarec.mark_dirty(self.rt);

        rt.restore(ctx.dynarec);

        EmitSummary::default()
    }
}

impl DynarecOp for Div {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        match (self.rs, self.rt) {
            (_, 0) => {
                let rt = ctx.dynarec.emit_load_reg(self.rt);
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; stp wzr, W(*rt), [x0, Emu::HILO_OFFSET as _]
                );
            }
            (0, _) => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; str xzr, [x0, Emu::HILO_OFFSET as _]
                );
            }
            (_, _) => {
                let rs = ctx.dynarec.emit_load_reg(self.rs);
                let rt = ctx.dynarec.emit_load_reg(self.rt);

                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; sdiv w1, W(*rs), W(*rt)
                    ; msub w2, w1, W(*rt), W(*rs)
                    // rt = 0 => w1 = 0 => hi = w2 = rs
                    // remainder checks out
                    // rt = 0 => lo = rs > 0 ? -1 : 1
                    //           w1 = 0 => ~w1 = -1;
                    //           rs > 0 ? ~w1 : 1
                    //           rs <= 0 ? 1 : ~w1
                    //           (mov w3 1) rs <= 0 ? w3 : ~w1
                    //           (mov w3 1) cmp rs, 0 then csinv w3, w3, w1, le
                    ; mov w3, 1
                    ; cmp WSP(*rs), 0
                    ; csinv w3, w3, w1, le
                    ; cmp WSP(*rt), 0
                    ; csel w1, w3, w1, eq
                    ; stp w1, w2, [x0, Emu::HILO_OFFSET as _]
                );
            }
        };
        EmitSummary::default()
    }
    // average values
    fn cycles(&self) -> u16 {
        18
    }
    fn hazard(&self) -> u16 {
        18
    }
}

impl DynarecOp for Divu {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        match (self.rs, self.rt) {
            (0, _) => {
                let rt = ctx.dynarec.emit_load_reg(self.rt);
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; stp wzr, W(*rt), [x0, Emu::HILO_OFFSET as _]
                );
            }
            (_, _) => {
                let rs = ctx.dynarec.emit_load_reg(self.rs);
                let rt = ctx.dynarec.emit_load_reg(self.rt);

                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; udiv w1, W(*rs), W(*rt)
                    ; msub w2, w1, W(*rt), W(*rs)
                    // rt = 0 => w1 = 0 => hi = w2 = rs
                    // remainder checks out
                    //
                    // udiv is much simpler than sdiv
                    // rt = 0 => lo = -0x1 (0xffffffff)
                    //
                    // rt != 0 ? rs%rt : ~0 (w2)
                    ; cmp WSP(*rt), 0
                    ; csinv w1, w1, wzr, ne

                    ; stp w1, w2, [x0, Emu::HILO_OFFSET as _]
                );
            }
        };
        EmitSummary::default()
    }
    // average values
    fn cycles(&self) -> u16 {
        18
    }
    fn hazard(&self) -> u16 {
        18
    }
}

impl DynarecOp for Multu {
    fn cycles(&self) -> u16 {
        9
    }
    fn hazard(&self) -> u16 {
        9
    }
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        match (self.rs, self.rt) {
            (0, _) | (_, 0) => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; str xzr, [x0, Emu::HILO_OFFSET as _]
                );
            }
            (_, _) => {
                let rs = ctx.dynarec.emit_load_reg(self.rs);
                let rt = ctx.dynarec.emit_load_reg(self.rt);

                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; umull x1, W(*rs), W(*rt)
                    ; str x1, [x0, Emu::HILO_OFFSET as _]
                );
            }
        };
        EmitSummary::default()
    }
}

impl DynarecOp for Mult {
    fn cycles(&self) -> u16 {
        9
    }
    fn hazard(&self) -> u16 {
        9
    }
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        match (self.rs, self.rt) {
            (0, _) | (_, 0) => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; str xzr, [x0, Emu::HILO_OFFSET as _]
                );
            }
            (_, _) => {
                let rs = ctx.dynarec.emit_load_reg(self.rs);
                let rt = ctx.dynarec.emit_load_reg(self.rt);

                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; smull x1, W(*rs), W(*rt)
                    ; str x1, [x0, Emu::HILO_OFFSET as _]
                );
            }
        };
        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
#[case::div(div, (8, 10), (9, 2), 0x00000000_00000005)]
#[case::div(div, (0, 0), (8, 3), 0x00000000_00000000)]
#[case::div(div, (8, 2), (9, 0), 0x00000002_ffffffff)]
#[case::div(div, (8, -2i32 as u32), (9, 0), 0xfffffffe_00000001)]
#[case::div(div, (8, -10i32 as u32), (9, 2), 0x00000000_fffffffb)]
#[case::divu(divu, (8, 10), (9, 2), 0x00000000_00000005)]
#[case::divu(divu, (8, 2), (9, 0), 0x00000002_ffffffff)]
#[case::mult(mult, (8, 2), (9, 15), 30)]
#[case::mult(mult, (8, 2), (9, -15i32 as u32), -30i32 as u64)]
#[case::multu(multu, (8, 2), (9, 15), 30)]
/// form: {inst} ${reg1}={value1}, ${reg2}={value2} ; assert hilo = {expected}
pub fn test_mul_div(
    #[case] instr: impl Fn(u8, u8) -> OpCode,
    #[case] rs: (Guest, u32),
    #[case] rt: (Guest, u32),
    #[case] expected: u64,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.cpu.hilo = 0xDEAD_BEEF_DEAD_BEEF;
    emu.cpu.gpr[rs.0 as usize] = rs.1;
    emu.cpu.gpr[rt.0 as usize] = rt.1;
    assert_eq!(emu.cpu.gpr[0], 0);

    emu.write_many(0x0, &program([instr(rs.0, rt.0), OpCode::HALT]));
    PipelineV2::new(&emu).run_once(&mut emu)?;

    tracing::info!(?emu.cpu);
    tracing::info!(hilo = %hex(emu.cpu.hilo));
    assert_eq!(emu.cpu.hilo, expected);
    assert_eq!(emu.cpu.pc, 0x8);
    Ok(())
}

impl DynarecOp for Mflo {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        }

        let rd = ctx.dynarec.alloc_reg(self.rd);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; ldr W(*rd), [x0, Emu::HILO_OFFSET as _]
        );

        ctx.dynarec.mark_dirty(self.rd);
        rd.restore(ctx.dynarec);

        EmitSummary::default()
    }
}

impl DynarecOp for Mfhi {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        }

        let rd = ctx.dynarec.alloc_reg(self.rd);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; ldr W(*rd), [x0, (Emu::HILO_OFFSET + size_of::<u32>()) as _]
        );

        ctx.dynarec.mark_dirty(self.rd);
        rd.restore(ctx.dynarec);

        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
#[case(mfhi, 0xDEAD_BEEF_1234_5678, 0xDEAD_BEEF)]
#[case(mflo, 0xDEAD_BEEF_1234_5678, 0x1234_5678)]
pub fn test_mfhilo(
    #[case] instr: impl Fn(u8) -> OpCode,
    #[case] hilo: u64,
    #[case] expected: u32,
) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.cpu.hilo = hilo;
    emu.write_many(0x0, &program([instr(9), OpCode::HALT]));
    PipelineV2::new(&emu).run_once(&mut emu)?;

    tracing::info!(?emu.cpu);
    tracing::info!(hilo = %hex(emu.cpu.hilo));
    tracing::info!(r = %hex(emu.cpu.gpr[9]));
    assert_eq!(emu.cpu.hilo, hilo);
    assert_eq!(emu.cpu.gpr[9], expected);
    Ok(())
}

impl DynarecOp for Syscall {
    fn boundary(&self) -> Boundary {
        Boundary::Soft
    }
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ;; let saved = ctx.dynarec.emit_save_volatile_registers()
            ;; ctx.dynarec.emit_writeback_all()
            ; mov w1, ctx.delay_slot as u32
            ; ldr x3, ->handle_syscall
            ; blr x3
            ;; ctx.dynarec.emit_restore_saved_registers(saved.into_iter())
        );

        EmitSummary::pc_updated()
    }
}

impl DynarecOp for Rfe {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        emit_call(&mut ctx, |dynarec| {
            dynasm!(
                dynarec.asm
                ;; let saved = dynarec.emit_save_volatile_registers()
                ;; dynarec.emit_writeback_all()
                ; ldr x3, ->handle_rfe
                ; blr x3
                ;; dynarec.emit_restore_saved_registers(saved.into_iter())
            )
        });
        EmitSummary::default()
    }
}

#[cfg(test)]
#[rstest]
pub fn test_load_0xbfc01a78() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.write_many(0x0, &program([OpCode::HALT]));
    PipelineV2::new(&emu).run_once(&mut emu)?;

    tracing::info!(?emu.cpu);
    Ok(())
}

enum Hilo {
    Hi,
    Lo,
}

#[allow(clippy::useless_conversion)]
fn emit_mthilo<'a>(rs: u8, mut ctx: EmitCtx<'a>, hilo: Hilo) -> EmitSummary {
    let offset = match hilo {
        Hilo::Hi => Emu::HILO_OFFSET + size_of::<u32>(),
        Hilo::Lo => Emu::HILO_OFFSET,
    };
    let rs = ctx.dynarec.emit_load_reg_stackless(rs);
    ctx.dynarec.lock_register(&rs);

    ctx.schedule_in(2, move |ctx| {
        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; str W(*rs), [x0, offset as _]
        );
        ctx.dynarec.unlock_register(&rs);
        EmitSummary::default()
    });
    EmitSummary::default()
}

impl DynarecOp for Mtlo {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_mthilo(self.rs, ctx, Hilo::Lo)
    }
}

impl DynarecOp for Mthi {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, ctx: EmitCtx<'a>) -> EmitSummary {
        emit_mthilo(self.rs, ctx, Hilo::Hi)
    }
}

#[cfg(test)]
#[rstest]
fn test_branch_and_store() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.write_many(
        0x0,
        &program([
            addiu(7, 7, 0x200),
            addiu(8, 8, 0x12),
            beq(9, 0, 0x8),
            sw(8, 7, 0),
            nop(),
            nop(),
            OpCode::HALT,
        ]),
    );

    PipelineV2::new(&emu).run_once(&mut emu)?;

    assert_eq_hex!(emu.read::<u32>(0x200), 0x12);

    Ok(())
}

#[cfg(test)]
#[rstest]
fn test_0x8004f454_move_in_jump_delay() -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::run_step;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();

    emu.cpu.gpr[cpu::SP as usize] = 0x801ffd50;
    emu.cpu.gpr[4] = 0x12;
    emu.write_many(
        0x0,
        &program([
            addiu(cpu::SP, cpu::SP, 0x4),
            sw(16, cpu::SP, 0x0018),
            addu(16, 0, 4),
            sw(cpu::RA, cpu::SP, 0x001c),
            addiu(3, 16, 0x001c),
            addu(4, 3, 0),
            sw(3, cpu::SP, 0x0024),
            jal(0x0),
            addu(5, 0, 16),
        ]),
    );

    run_step(&mut emu, Box::default());

    assert_eq_hex!(emu.cpu.gpr[5], 0x12);

    Ok(())
}

impl DynarecOp for Lwcn {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        let s1 = ctx.alloc_scratch();
        ctx.dynarec.emit_load_temp_reg(self.rs, Reg::W(1));
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ;; ctx.dynarec.emit_add_imm16(EmitAddImm16Args { dest: Reg::W(1), base: Reg::W(1), offset: self.imm16, temp: None })
            ; str w1, [x0, #s1]
        );
        let cop = self.cop;
        let rt = self.rt;
        ctx.schedule_in(1, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                // ; ldr w1, [sp], #16
                ;; let saved = ctx.dynarec.emit_save_volatile_registers()
                ; ldr w1, [x0, #s1]
                ; ldr x3, ->read32v2
                ; blr x3
                ; mov w1, w0
                ;; ctx.dynarec.emit_restore_saved_registers(saved.into_iter())
                ; str w1, [x0, Cpu::cop_reg_offset(cop, rt) as _]
            );
            EmitSummary::default()
        });

        EmitSummary::default()
    }

    fn cycles(&self) -> u16 {
        3
    }

    fn hazard(&self) -> u16 {
        1
    }
}

#[cfg(test)]
#[rstest]
#[case(0, 5, 10)]
#[case(2, 5, 10)]
#[case(2, 31, 10)]
fn test_lwcn(#[case] cop: u8, #[case] rt: u8, #[case] rs: u8) -> color_eyre::Result<()> {
    use crate::Emu;
    use crate::cpu::program;
    use crate::dynarec_v2::PipelineV2;
    use pchan_utils::setup_tracing;

    setup_tracing();
    let mut emu = Emu::default();
    emu.cpu.gpr[rs as usize] = 0x100;
    emu.write(0x100, 0xcafebabe_u32);

    emu.write_many(0x0, &program([lwcn(cop, rt, rs, 0x0), OpCode::HALT]));

    PipelineV2::new(&emu).run_once(&mut emu)?;
    tracing::info!(?emu.cpu);
    match cop {
        0 => assert_eq!(emu.cpu.cop0.reg[rt as usize], 0xcafebabe),
        2 => assert_eq!(emu.cpu.cop2.reg[rt as usize], 0xcafebabe),
        _ => panic!("get out"),
    }

    Ok(())
}

impl DynarecOp for Swcn {
    #[allow(clippy::useless_conversion)]
    fn emit<'a>(&self, mut ctx: EmitCtx<'a>) -> EmitSummary {
        // FIXME: with gte nopped out, this helps games boot
        // remove once fixed.
        if false {
            let s1 = ctx.alloc_scratch();
            let Self { cop, rt, rs, imm16 } = *self;
            ctx.dynarec.emit_load_temp_reg(rs, Reg::W(1));

            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ;; ctx.dynarec.emit_add_imm16(EmitAddImm16Args { dest: Reg::W(1), base: Reg::W(1), offset: imm16, temp: None })
                ; str w1, [x0, #s1]
            );

            ctx.schedule_in(1, move |ctx| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; ldr w1, [x0, #s1]
                    ; ldr w2, [x0, Cpu::cop_reg_offset(cop, rt) as _]
                    ;; let saved = ctx.dynarec.emit_save_volatile_registers()
                    ; ldr x3, ->write32v2
                    ; blr x3
                    ;; ctx.dynarec.emit_restore_saved_registers(saved.into_iter())

                );
                EmitSummary::default()
            });
        }

        EmitSummary::default()
    }

    fn cycles(&self) -> u16 {
        3
    }

    fn hazard(&self) -> u16 {
        1
    }
}

impl DynarecOp for Bltzal {
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
        emit_branch_zero(
            ctx,
            self.rs,
            self.imm16,
            move |ctx| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; csel w2, w3, w2, lt
                )
            },
            true,
        )
    }
}

impl DynarecOp for Bgezal {
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
        emit_branch_zero(
            ctx,
            self.rs,
            self.imm16,
            move |ctx| {
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; csel w2, w3, w2, ge
                )
            },
            true,
        )
    }
}

#[cfg(test)]
#[rstest]
fn test_bltzal_bgezal() {
    use crate::Emu;
    use crate::cpu::program;
    use assert_hex::*;
    use pchan_utils::setup_tracing;

    setup_tracing();
    {
        let mut emu = Emu::default();
        emu.cpu.pc = 0x0;
        emu.write_many(
            0x0,
            &program([addiu(8, 0, -10), bltzal(8, 0x100), OpCode::HALT]),
        );
        crate::dynarec_v2::run_step(&mut emu, Box::default());
        tracing::info!(?emu.cpu);
        assert_eq_hex!(emu.cpu.gpr[8] as i32, -10);
        assert_eq_hex!(emu.cpu["$ra"], 0xc);
        assert_eq_hex!(emu.cpu.pc, 0x408);
    }

    {
        let mut emu = Emu::default();
        emu.cpu.pc = 0x0;
        emu.write_many(
            0x0,
            &program([addiu(8, 0, 10), bgezal(8, 0x100), OpCode::HALT]),
        );
        crate::dynarec_v2::run_step(&mut emu, Box::default());
        tracing::info!(?emu.cpu);
        assert_eq_hex!(emu.cpu.gpr[8] as i32, 10);
        assert_eq_hex!(emu.cpu["$ra"], 0xc);
        assert_eq_hex!(emu.cpu.pc, 0x408);
    }
}

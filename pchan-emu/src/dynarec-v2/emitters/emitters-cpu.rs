use crate::bindings::ExternFnTable;

use crate::cpu::{Cpu, RA};
use crate::dynarec_v2::emitters::{
    AluImm, Boundary, CopMove, CopRegisterKind, DynarecOp, EitherZero, EmitAddImm16Args, EmitCtx,
    EmitSummary, Hilo, ShiftImm, either_zero, emit_alu_imm, emit_alu_reg, emit_branch,
    emit_branch_zero, emit_call, emit_load, emit_move_from_cop, emit_move_to_cop, emit_mthilo,
    emit_shift_by_reg, emit_shift_imm, emit_store,
};
use crate::{Emu, cpu};

use arbitrary_int::traits::Integer;

use pchan_utils::default;

use crate::cpu::ops::prelude::*;
use crate::dynarec_v2::Reg;
use crate::memory::ext;
use dynasm::dynasm;
use dynasmrt::DynasmApi;

use core::alloc::Allocator;

impl DynarecOp for Addiu {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        // $rt = $zero case
        if self.rt == 0 {
            return EmitSummary::default();
        }

        ctx.dynarec.cancel_next_write(self.rt);

        // $rs = $zero case
        if self.rs == 0 {
            let rt = ctx.dynarec.alloc_reg_stackless(self.rt);

            ctx.dynarec.emit_imm16_sext(rt.reg(), self.imm16);

            ctx.dynarec.mark_dirty(self.rt);
            return EmitSummary::default();
        }

        let rs = ctx
            .dynarec
            .emit_load_reg_stackless(self.rs)
            .lock(ctx.dynarec);
        let rt = ctx.dynarec.alloc_reg_stackless(self.rt);

        ctx.dynarec.emit_add_imm16(EmitAddImm16Args {
            dest:   rt.reg(),
            base:   rs.reg(),
            offset: self.imm16,
            temp:   None,
        });

        ctx.dynarec.mark_dirty(self.rt);
        rs.unlock(ctx.dynarec);

        EmitSummary::default()
    }
}

impl DynarecOp for HaltBlock {
    fn emit<A: Allocator + Copy>(&self, _: EmitCtx<A>) -> EmitSummary {
        EmitSummary::default()
    }

    fn cycles(&self) -> u16 {
        1
    }

    fn boundary(&self) -> Boundary {
        Boundary::Hard
    }
}

impl DynarecOp for Sb {
    #[cfg(target_arch = "aarch64")]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        use crate::dynarec_v2::emitters::emit_store;

        emit_store(ctx, self.rt, self.rs, self.imm16, move |ctx| {
            use crate::bindings::ExternFnTable;

            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, [x0, #ExternFnTable::<A>::WRITE8V2]
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm16, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, [x0, #ExternFnTable::<A>::WRITE16V2]
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm16, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, [x0, #ExternFnTable::<A>::WRITE32V2]
                ; blr x3
            );
        })
    }
    fn cycles(&self) -> u16 {
        2
    }
}

impl DynarecOp for Swl {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm16, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, [x0, #ExternFnTable::<A>::ULWRITE32]
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_store(ctx, self.rt, self.rs, self.imm16, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, [x0, #ExternFnTable::<A>::URWRITE32]
                ; blr x3
            );
        })
    }

    fn cycles(&self) -> u16 {
        2
    }
}
impl DynarecOp for Lb {
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_load::<true, A>(ctx, self.rt, self.rs, self.imm16, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, [x0, #ExternFnTable::<A>::READI8V2]
                ; blr x3
            );
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_load::<true, A>(ctx, self.rt, self.rs, self.imm16, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, [x0, #ExternFnTable::<A>::READU8V2]
                ; blr x3
            );
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_load::<true, A>(ctx, self.rt, self.rs, self.imm16, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, [x0, #ExternFnTable::<A>::READI16V2]
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_load::<true, A>(ctx, self.rt, self.rs, self.imm16, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, [x0, #ExternFnTable::<A>::READU16V2]
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_load::<true, A>(ctx, self.rt, self.rs, self.imm16, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, [x0, #ExternFnTable::<A>::READ32V2]
                ; blr x3
            );
        })
    }
    fn cycles(&self) -> u16 {
        3
    }
    fn hazard(&self) -> u16 {
        2
    }
}

impl DynarecOp for Lwl {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_load::<false, A>(ctx, self.rt, self.rs, self.imm16, |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, [x0, #ExternFnTable::<A>::ULREAD32]
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_load::<false, A>(ctx, self.rt, self.rs, self.imm16, |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr x3, [x0, #ExternFnTable::<A>::URREAD32]
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

impl DynarecOp for Subu {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        };

        ctx.dynarec.cancel_next_write(self.rd);

        if self.rt == 0 {
            ctx.dynarec.emit_load_temp_reg(self.rs, Reg::W(1));
            let rd = ctx.dynarec.alloc_reg_stackless(self.rd);

            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; mov W(*rd), w1
            );

            ctx.dynarec.mark_dirty(self.rd);

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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        };

        ctx.dynarec.cancel_next_write(self.rd);

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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        }

        ctx.dynarec.cancel_next_write(self.rd);

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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        }

        ctx.dynarec.cancel_next_write(self.rd);

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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        }

        ctx.dynarec.cancel_next_write(self.rd);

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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        }

        ctx.dynarec.cancel_next_write(self.rd);

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
                let rd = ctx.dynarec.alloc_reg_stackless(target);
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; mov W(*rd), w1
                    ; mvn W(*rd), W(*rd)
                );
                ctx.dynarec.mark_dirty(target);
                EmitSummary::default()
            }
        }
    }
}

impl DynarecOp for Sllv {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_shift_by_reg(ctx, self.rd, self.rs, self.rt, move |ctx, regs| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; asr W(**regs.rd), W(**regs.rt), W(**regs.rs)
            )
        })
    }
}

impl DynarecOp for Sll {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, mut ctx: EmitCtx<A>) -> EmitSummary {
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
                    ; lsl W(**rd), W(**rt), imm.into()
                );

                dynarec.mark_dirty(self.rd);
            },
        )
    }
}

impl DynarecOp for Srl {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, mut ctx: EmitCtx<A>) -> EmitSummary {
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
    fn emit<A: Allocator + Copy>(&self, mut ctx: EmitCtx<A>) -> EmitSummary {
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
                    ; asr W(**rd), W(**rt), imm.into()
                );
                dynarec.mark_dirty(self.rd);
            },
        )
    }
}

impl DynarecOp for Andi {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        ctx.dynarec.cancel_next_write(self.rt);

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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        ctx.dynarec.cancel_next_write(self.rt);

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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        ctx.dynarec.cancel_next_write(self.rt);

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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rt == 0 {
            return EmitSummary::default();
        }
        ctx.dynarec.cancel_next_write(self.rt);
        let rt = ctx.dynarec.emit_load_reg_stackless(self.rt);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; movz W(*rt), ext::zero(self.imm16) as _, LSL #16
        );

        ctx.dynarec.mark_dirty(self.rt);

        EmitSummary::default()
    }
}

impl DynarecOp for J {
    fn emit<A: Allocator + Copy>(&self, mut ctx: EmitCtx<A>) -> EmitSummary {
        let new_pc = (self.imm26 << 2) | (ctx.pc & 0xf0000000);
        ctx.schedule_in(1, None, move |ctx| {
            ctx.dynarec.emit_write_pc(Reg::W(1), new_pc);
            ctx.dynarec.emit_set_bt::<true>();
            ctx.dynarec.emit_set_bd::<true>();

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
    fn emit<A: Allocator + Copy>(&self, mut ctx: EmitCtx<A>) -> EmitSummary {
        let new_pc = (self.imm26 << 2) + (ctx.pc & 0xf0000000);
        let return_address = ctx.pc + 0x8;
        ctx.dynarec.cancel_next_write(RA);
        ctx.dynarec.emit_immediate_large(cpu::RA, return_address);
        ctx.schedule_in(1, Some(RA), move |ctx| {
            ctx.dynarec.emit_write_pc(Reg::W(3), new_pc);

            EmitSummary::pc_updated()
        });
        EmitSummary::pc_updated()
    }
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
    fn emit<A: Allocator + Copy>(&self, mut ctx: EmitCtx<A>) -> EmitSummary {
        let dest = ctx.dynarec.emit_load_reg_stackless(self.rs);
        let s1 = ctx.alloc_scratch();
        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            // ; str W(*dest), [sp, #-16]!
            ; str W(*dest), [x0, #s1]
        );

        ctx.schedule_in(1, None, move |ctx| {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr w1, [x0, #s1]
                ; str w1, [x0, Emu::<A>::PC_OFFSET as _]
                ;; ctx.dynarec.emit_set_bd::<true>()
            );

            EmitSummary::pc_updated()
        });
        EmitSummary::default()
    }
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
    fn emit<A: Allocator + Copy>(&self, mut ctx: EmitCtx<A>) -> EmitSummary {
        let s1 = ctx.alloc_scratch();
        let dest = ctx.dynarec.emit_load_reg_stackless(self.rs);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; str W(*dest), [x0, #s1]
        );

        let rd = self.rd;
        ctx.dynarec.cancel_next_write(rd);

        let ret_address = ctx.pc + 0x8;
        if rd != 0 {
            ctx.dynarec.emit_immediate_large(rd, ret_address);
        }

        // we know the delay slots runs at old pc + 4
        ctx.schedule_in(1, Some(rd), move |ctx| {
            #[cfg(target_arch = "aarch64")]
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                ; ldr w1, [x0, #s1]
                ; str w1, [x0, Emu::<A>::PC_OFFSET as _]
                ;; ctx.dynarec.emit_set_bd::<true>()
            );

            EmitSummary::pc_updated()
        });
        EmitSummary::default()
    }
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
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

impl DynarecOp for Mtcn {
    fn cycles(&self) -> u16 {
        match self.cop {
            2 => 3,
            0 | 1 | 3 => 1,
            _ => unreachable!(),
        }
    }
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_move_to_cop(
            ctx,
            CopMove {
                cop:  self.cop,
                rd:   self.rd,
                rt:   self.rt,
                kind: CopRegisterKind::Data,
            },
        )
    }
}

impl DynarecOp for Ctcn {
    fn cycles(&self) -> u16 {
        match self.cop {
            2 => 3,
            0 | 1 | 3 => 1,
            _ => unreachable!(),
        }
    }
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_move_to_cop(
            ctx,
            CopMove {
                cop:  self.cop,
                rd:   self.rd,
                rt:   self.rt,
                kind: CopRegisterKind::Ctrl,
            },
        )
    }
}

impl DynarecOp for Nop {
    fn cycles(&self) -> u16 {
        1
    }
    fn emit<A: Allocator + Copy>(&self, _: EmitCtx<A>) -> EmitSummary {
        EmitSummary::default()
    }
}

impl DynarecOp for Sltu {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        }

        ctx.dynarec.cancel_next_write(self.rd);

        let rs = ctx
            .dynarec
            .emit_load_reg_stackless(self.rs)
            .lock(ctx.dynarec);
        let rt = ctx
            .dynarec
            .emit_load_reg_stackless(self.rt)
            .lock(ctx.dynarec);
        let rd = ctx.dynarec.alloc_reg_stackless(self.rd);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; cmp W(*rs), W(*rt)
            ; cset W(*rd), lo
        );

        rs.unlock(ctx.dynarec);
        rt.unlock(ctx.dynarec);
        ctx.dynarec.mark_dirty(self.rd);

        EmitSummary::default()
    }
}

impl DynarecOp for Sltiu {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rt == 0 {
            return EmitSummary::default();
        }

        ctx.dynarec.cancel_next_write(self.rt);
        let rt = ctx
            .dynarec
            .emit_load_reg_stackless(self.rt)
            .lock(ctx.dynarec);
        let rs = ctx.dynarec.emit_load_reg_stackless(self.rs);

        let imm32 = ext::sign(self.imm16);
        match self.imm16 {
            -4095..=-1 => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; cmn WSP(*rs), (-imm32).as_u32()
                    ; cset W(*rt), lo
                );
            }
            0..4096 => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; cmp WSP(*rs), imm32.as_u32()
                    ; cset W(*rt), lo
                );
            }
            _ => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; movz w1, (imm32 as u32) >> 16, lsl #16
                    ; movk w1, (imm32 as u32) & 0x0000_ffff
                    ; cmp WSP(*rs), w1
                    ; cset W(*rt), lo
                );
            }
        };

        ctx.dynarec.mark_dirty(self.rt);

        rt.unlock(ctx.dynarec);

        EmitSummary::default()
    }
}

impl DynarecOp for Slt {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        }

        ctx.dynarec.cancel_next_write(self.rd);

        let rs = ctx
            .dynarec
            .emit_load_reg_stackless(self.rs)
            .lock(ctx.dynarec);
        let rt = ctx
            .dynarec
            .emit_load_reg_stackless(self.rt)
            .lock(ctx.dynarec);
        let rd = ctx.dynarec.alloc_reg_stackless(self.rd);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; cmp W(*rs), W(*rt)
            ; cset W(*rd), lt
        );

        rs.unlock(ctx.dynarec);
        rt.unlock(ctx.dynarec);

        ctx.dynarec.mark_dirty(self.rd);

        EmitSummary::default()
    }
}

impl DynarecOp for Slti {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rt == 0 {
            return EmitSummary::default();
        }

        ctx.dynarec.cancel_next_write(self.rt);
        let rt = ctx
            .dynarec
            .emit_load_reg_stackless(self.rt)
            .lock(ctx.dynarec);
        let rs = ctx.dynarec.emit_load_reg_stackless(self.rs);

        let imm32 = ext::sign(self.imm16);
        match imm32 {
            -4095..=-1 => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; cmn WSP(*rs), (-imm32).as_u32()
                    ; cset W(*rt), lt
                );
            }
            0..4096 => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; cmp WSP(*rs), imm32.as_u32()
                    ; cset W(*rt), lt
                );
            }
            _ => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; .arch aarch64
                    ; movz w1, (imm32 as u32) >> 16, lsl #16
                    ; movk w1, (imm32 as u32) & 0x0000_ffff
                    ; cmp WSP(*rs), w1
                    ; cset W(*rt), lt
                );
            }
        }
        ctx.dynarec.mark_dirty(self.rt);

        rt.unlock(ctx.dynarec);

        EmitSummary::default()
    }
}

impl DynarecOp for Mfcn {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_move_from_cop(
            ctx,
            CopMove {
                cop:  self.cop,
                rd:   self.rd,
                rt:   self.rt,
                kind: CopRegisterKind::Data,
            },
        );
        default()
    }
}

impl DynarecOp for Cfcn {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_move_from_cop(
            ctx,
            CopMove {
                cop:  self.cop,
                rd:   self.rd,
                rt:   self.rt,
                kind: CopRegisterKind::Ctrl,
            },
        );
        default()
    }
}

impl DynarecOp for Div {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        let rs = ctx
            .dynarec
            .emit_load_reg_stackless(self.rs)
            .lock(ctx.dynarec);
        let rt = ctx.dynarec.emit_load_reg_stackless(self.rt);

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
            ; csinv w3, w3, w1, lt
            ; cmp WSP(*rt), 0
            ; csel w1, w3, w1, eq
            ; stp w1, w2, [x0, Emu::<A>::HILO_OFFSET as _]
        );
        rs.unlock(ctx.dynarec);
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        let rs = ctx
            .dynarec
            .emit_load_reg_stackless(self.rs)
            .lock(ctx.dynarec);
        let rt = ctx.dynarec.emit_load_reg_stackless(self.rt);

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

            ; stp w1, w2, [x0, Emu::<A>::HILO_OFFSET as _]
        );

        rs.unlock(ctx.dynarec);

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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        match (self.rs, self.rt) {
            (0, _) | (_, 0) => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; str xzr, [x0, Emu::<A>::HILO_OFFSET as _]
                );
            }
            (_, _) => {
                let rs = ctx
                    .dynarec
                    .emit_load_reg_stackless(self.rs)
                    .lock(ctx.dynarec);
                let rt = ctx.dynarec.emit_load_reg_stackless(self.rt);

                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; umull x1, W(*rs), W(*rt)
                    ; str x1, [x0, Emu::<A>::HILO_OFFSET as _]
                );

                rs.unlock(ctx.dynarec);
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        match (self.rs, self.rt) {
            (0, _) | (_, 0) => {
                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; str xzr, [x0, Emu::<A>::HILO_OFFSET as _]
                );
            }
            (_, _) => {
                let rs = ctx
                    .dynarec
                    .emit_load_reg_stackless(self.rs)
                    .lock(ctx.dynarec);
                let rt = ctx.dynarec.emit_load_reg_stackless(self.rt);

                #[cfg(target_arch = "aarch64")]
                dynasm!(
                    ctx.dynarec.asm
                    ; smull x1, W(*rs), W(*rt)
                    ; str x1, [x0, Emu::<A>::HILO_OFFSET as _]
                );

                rs.unlock(ctx.dynarec);
            }
        };
        EmitSummary::default()
    }
}

impl DynarecOp for Mflo {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        }

        ctx.dynarec.cancel_next_write(self.rd);

        let rd = ctx.dynarec.alloc_reg_stackless(self.rd);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; ldr W(*rd), [x0, Emu::<A>::HILO_OFFSET as _]
        );

        ctx.dynarec.mark_dirty(self.rd);

        EmitSummary::default()
    }
}

impl DynarecOp for Mfhi {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.rd == 0 {
            return EmitSummary::default();
        }

        ctx.dynarec.cancel_next_write(self.rd);

        let rd = ctx.dynarec.alloc_reg_stackless(self.rd);

        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ; ldr W(*rd), [x0, (Emu::<A>::HILO_OFFSET + size_of::<u32>()) as _]
        );

        ctx.dynarec.mark_dirty(self.rd);

        EmitSummary::default()
    }
}

impl DynarecOp for Syscall {
    fn boundary(&self) -> Boundary {
        Boundary::Hard
    }
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        assert!(!ctx.delay_slot);
        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ;; ctx.dynarec.emit_writeback_all()
            ;; ctx.dynarec.reg_alloc.dealloc_all()
            ;; ctx.dynarec.emit_save_critical_registers()
            ; movz w1, u32::from(ctx.pc) >> 16, lsl #16
            ; movk w1, u32::from(ctx.pc) & 0x0000_ffff
            ; ldr x3, [x0, #ExternFnTable::<A>::HANDLE_SYSCALL]
            ; blr x3
            ; nop
            ;; ctx.dynarec.emit_restore_critical_registers()
        );

        EmitSummary::pc_updated()
    }
}

impl DynarecOp for Brk {
    fn boundary(&self) -> Boundary {
        Boundary::Hard
    }
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        #[cfg(target_arch = "aarch64")]
        dynasm!(
            ctx.dynarec.asm
            ;; ctx.dynarec.emit_writeback_all()
            ;; ctx.dynarec.reg_alloc.dealloc_all()
            ;; ctx.dynarec.emit_save_critical_registers()
            ; movz w1, u32::from(ctx.pc) >> 16, lsl #16
            ; movk w1, u32::from(ctx.pc) & 0x0000_ffff
            ; ldr x3, [x0, #ExternFnTable::<A>::HANDLE_BREAK]
            ; blr x3
            ;; ctx.dynarec.emit_restore_critical_registers()
        );

        EmitSummary::pc_updated()
    }
}

impl DynarecOp for Rfe {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, mut ctx: EmitCtx<A>) -> EmitSummary {
        emit_call(&mut ctx, |dynarec| {
            dynasm!(
                dynarec.asm
                ;; let saved = dynarec.emit_save_volatile_registers()
                ;; dynarec.emit_writeback_all()
                ; ldr x3, [x0, #ExternFnTable::<A>::HANDLE_RFE]
                ; blr x3
                ;; dynarec.emit_restore_saved_registers(saved.into_iter())
            )
        });
        EmitSummary::default()
    }
}

impl DynarecOp for Mtlo {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_mthilo(self.rs, ctx, Hilo::Lo)
    }
}

impl DynarecOp for Mthi {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        emit_mthilo(self.rs, ctx, Hilo::Hi)
    }
}

impl DynarecOp for Lwcn {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, mut ctx: EmitCtx<A>) -> EmitSummary {
        if self.cop != 2 && self.cop != 0 {
            return default();
        }

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
        ctx.schedule_in(1, None, move |ctx| {
            dynasm!(
                ctx.dynarec.asm
                ; .arch aarch64
                // ; ldr w1, [sp], #16
                ;; ctx.dynarec.emit_writeback_all()
                ;; ctx.dynarec.reg_alloc.dealloc_all()
                ;; ctx.dynarec.emit_save_critical_registers()
                ; ldr w1, [x0, #s1]
                ; ldr x3, [x0, #ExternFnTable::<A>::READ32V2]
                ; blr x3
                ; mov w1, w0
                ;; ctx.dynarec.emit_restore_critical_registers()
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

impl DynarecOp for Swcn {
    #[allow(clippy::useless_conversion)]
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        if self.cop != 2 && self.cop != 0 {
            return default();
        }
        let Self { cop, rt, rs, imm16 } = *self;
        ctx.dynarec.emit_load_temp_reg(rs, Reg::W(1));

        dynasm!(
            ctx.dynarec.asm
            ; .arch aarch64
            ;; ctx.dynarec.emit_add_imm16(EmitAddImm16Args { dest: Reg::W(1), base: Reg::W(1), offset: imm16, temp: None })
            ; ldr w2, [x0, Cpu::cop_reg_offset(cop, rt) as _]
            ;; ctx.dynarec.emit_writeback_all()
            ;; ctx.dynarec.reg_alloc.dealloc_all()
            ;; ctx.dynarec.emit_save_critical_registers()
            ; ldr x3, [x0, #ExternFnTable::<A>::WRITE32V2]
            ; blr x3
            ;; ctx.dynarec.emit_restore_critical_registers()
        );

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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        ctx.dynarec.cancel_next_write(RA);
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
    fn emit<A: Allocator + Copy>(&self, ctx: EmitCtx<A>) -> EmitSummary {
        ctx.dynarec.cancel_next_write(RA);
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
